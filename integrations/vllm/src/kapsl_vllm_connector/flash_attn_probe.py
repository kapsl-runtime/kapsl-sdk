"""Linux/CUDA certification probe for the vLLM FlashAttention shared pool.

The probe uses the production CUDA IPC importer and allocator hook, invokes
the same native cache writer and paged reader used by vLLM's FLASH_ATTN
backend, and exercises Kapsl's attach/activate/reserve/detach lifecycle.  A
runtime allowlist value is emitted only after all ranks and gates pass.
"""

from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import gc
import hashlib
import importlib
import json
import platform
import sys
import tempfile
import threading
import traceback
import uuid
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any
from unittest.mock import patch

from .certification import (
    CERTIFICATION_SCHEMA_VERSION,
    failed_report,
    remove_stale_allowlist,
    validate_certification_report,
    write_allowlist_atomic,
    write_json_atomic,
)
from .client import KapslKvControlClient, KapslKvControlError
from .connector import ADAPTER_PROFILE_ID, ADAPTER_VERSION
from .contract import (
    ABI_VERSION,
    make_reserve_request,
    make_shared_pool_attachment,
    make_shared_pool_detach_request,
    shared_pool_registration,
)
from .shared_pool import VllmSharedPoolHook, select_cuda_binding


_LAYER_NAME = "kapsl.probe.attention"
_GROUP_ID = "vllm.group.0"
_POOL_ID = "vllm.pool.0"
_SENTINEL = 0x5A


@dataclass(frozen=True)
class ProbeGeometry:
    num_blocks: int
    block_size: int
    num_kv_heads: int
    num_query_heads: int
    head_size: int
    dtype: str
    guard_bytes: int

    @property
    def element_size(self) -> int:
        return 2

    @property
    def dense_page_bytes(self) -> int:
        return (
            self.block_size
            * self.num_kv_heads
            * (2 * self.head_size)
            * self.element_size
        )

    @property
    def padded_page_bytes(self) -> int:
        return self.dense_page_bytes + self.guard_bytes

    @property
    def allocation_bytes(self) -> int:
        return self.num_blocks * self.padded_page_bytes

    def validate(self) -> None:
        values = (
            self.num_blocks,
            self.block_size,
            self.num_kv_heads,
            self.num_query_heads,
            self.head_size,
            self.guard_bytes,
        )
        if any(value <= 0 for value in values):
            raise ValueError("all probe geometry values must be positive")
        if self.num_query_heads % self.num_kv_heads:
            raise ValueError("query heads must be divisible by KV heads")
        if self.head_size % 8:
            raise ValueError("FlashAttention head size must be divisible by 8")
        if self.dtype not in {"float16", "bfloat16"}:
            raise ValueError("probe dtype must be float16 or bfloat16")

    def as_dict(self) -> dict[str, Any]:
        return {
            "num_blocks": self.num_blocks,
            "block_size": self.block_size,
            "num_kv_heads": self.num_kv_heads,
            "num_query_heads": self.num_query_heads,
            "head_size": self.head_size,
            "dense_page_bytes": self.dense_page_bytes,
            "guard_bytes_per_block": self.guard_bytes,
            "padded_page_bytes": self.padded_page_bytes,
            "allocation_bytes": self.allocation_bytes,
        }


class _LeaseHeartbeat:
    def __init__(
        self, client: KapslKvControlClient, ttl_ms: int
    ) -> None:
        self._client = client
        self._interval_seconds = max(0.25, ttl_ms / 3000.0)
        self._stop = threading.Event()
        self._failure: KapslKvControlError | None = None
        self._calls = 0
        self._thread = threading.Thread(
            target=self._run,
            name="kapsl-vllm-probe-heartbeat",
            daemon=True,
        )

    def start(self) -> None:
        self._client.heartbeat()
        self._calls += 1
        self._thread.start()

    def stop(self, *, raise_failure: bool = True) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=self._interval_seconds + 1.0)
        if self._thread.is_alive():
            raise RuntimeError("probe heartbeat thread did not stop")
        if raise_failure and self._failure is not None:
            raise RuntimeError("Kapsl lease heartbeat failed") from self._failure
        if raise_failure and self._calls == 0:
            raise RuntimeError("probe did not renew its live lease")

    def _run(self) -> None:
        while not self._stop.wait(self._interval_seconds):
            try:
                self._client.heartbeat()
                self._calls += 1
            except KapslKvControlError as error:
                self._failure = error
                return


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    remove_stale_allowlist(args.allowlist_output)

    geometry = ProbeGeometry(
        num_blocks=args.num_blocks,
        block_size=args.block_size,
        num_kv_heads=args.num_kv_heads,
        num_query_heads=args.num_query_heads,
        head_size=args.head_size,
        dtype=args.dtype,
        guard_bytes=args.guard_bytes,
    )
    devices = _parse_devices(args.devices)
    backend_version = _installed_version("vllm") or "unavailable"
    profile = {
        "adapter_id": "kapsl-vllm-connector",
        "adapter_version": ADAPTER_VERSION,
        "backend_version": backend_version,
        "profile_id": ADAPTER_PROFILE_ID,
    }
    environment: dict[str, Any] = {
        "adapter_build_id": args.adapter_build_id,
        "backend_build_id": args.backend_build_id,
        "runtime_build_id": args.runtime_build_id,
        "host": platform.platform(),
        "python_version": platform.python_version(),
        "torch_version": "unavailable",
        "cuda_runtime_version": "unavailable",
        "cuda_driver_version": "unavailable",
    }
    matrix: dict[str, Any] = {
        "attention_backend": "FLASH_ATTN",
        "kv_layout": "LBNHC",
        "dtype": geometry.dtype,
        "cache_geometry": geometry.as_dict(),
        "tensor_parallel_world_size": len(devices),
        "devices": devices,
    }
    rank_reports: list[dict[str, Any]] = []

    try:
        _validate_digest(args.adapter_build_id, "adapter-build-id")
        _validate_digest(args.backend_build_id, "backend-build-id")
        _validate_digest(args.runtime_build_id, "runtime-build-id")
        geometry.validate()
        if not sys.platform.startswith("linux"):
            raise RuntimeError("the CUDA IPC certification probe requires Linux")
        if backend_version == "unavailable":
            raise RuntimeError("vLLM is not installed")

        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("the certification probe requires CUDA")
        if max(devices) >= torch.cuda.device_count():
            raise RuntimeError(
                f"requested CUDA device {max(devices)}, but only "
                f"{torch.cuda.device_count()} device(s) are visible"
            )
        environment.update(
            {
                "torch_version": str(torch.__version__),
                "cuda_runtime_version": str(torch.version.cuda or "unknown"),
                "cuda_driver_version": str(_cuda_driver_version()),
            }
        )

        participant_id = args.participant_id or f"kapsl-cert-{uuid.uuid4().hex}"
        registration = _registration(
            participant_id, profile, devices, geometry
        )
        profile_rejection_evidence = _expect_unallowlisted_profile_rejected(
            args.endpoint,
            participant_id,
            profile,
            devices,
            geometry,
            args.timeout_seconds,
        )
        client = KapslKvControlClient(
            args.endpoint,
            participant_id,
            timeout_seconds=args.timeout_seconds,
        )
        receipt = client.register(registration)
        if len(receipt["shared_pools"]) != len(devices):
            raise RuntimeError(
                "runtime returned a different number of physical bindings than ranks"
            )
        preactivation_evidence = _expect_activation_rejected(
            client, receipt, geometry, args.lease_ttl_ms
        )

        payload = {
            "endpoint": args.endpoint,
            "participant_id": participant_id,
            "timeout_seconds": args.timeout_seconds,
            "lease_ttl_ms": args.lease_ttl_ms,
            "devices": devices,
            "geometry": {
                "num_blocks": geometry.num_blocks,
                "block_size": geometry.block_size,
                "num_kv_heads": geometry.num_kv_heads,
                "num_query_heads": geometry.num_query_heads,
                "head_size": geometry.head_size,
                "dtype": geometry.dtype,
                "guard_bytes": geometry.guard_bytes,
            },
            "profile": profile,
            "registration": registration,
            "participant_epoch": receipt["participant_epoch"],
        }
        with tempfile.TemporaryDirectory(prefix="kapsl-vllm-probe-") as work:
            init_path = str(Path(work) / "distributed-init")
            try:
                torch.multiprocessing.spawn(
                    _rank_worker,
                    args=(payload, work, init_path),
                    nprocs=len(devices),
                    join=True,
                )
            except BaseException:
                rank_reports = _read_available_rank_reports(work, len(devices))
                raise
            rank_reports = _read_rank_reports(work, len(devices))

        gates = _aggregate_gates(
            rank_reports,
            expected_bindings={
                str(binding["binding_id"])
                for binding in receipt["shared_pools"]
            },
            contract_evidence=(
                profile_rejection_evidence + "; " + preactivation_evidence
            ),
        )
        report = {
            "schema_version": CERTIFICATION_SCHEMA_VERSION,
            "status": "passed",
            "profile": profile,
            "environment": environment,
            "matrix": matrix,
            "gates": gates,
            "ranks": rank_reports,
        }
        validate_certification_report(report)
        write_json_atomic(args.report, report)
        if args.allowlist_output:
            write_allowlist_atomic(args.allowlist_output, report)
        print(json.dumps({"status": "passed", "report": args.report}))
        return 0
    except BaseException as error:
        report = failed_report(
            profile=profile,
            environment=environment,
            matrix=matrix,
            ranks=rank_reports,
            error="".join(traceback.format_exception(error)),
        )
        write_json_atomic(args.report, report)
        print(
            json.dumps(
                {"status": "failed", "report": args.report, "error": str(error)}
            ),
            file=sys.stderr,
        )
        return 1


def _rank_worker(
    rank: int,
    payload: dict[str, Any],
    work: str,
    init_path: str,
) -> None:
    import torch
    import torch.distributed as distributed

    world_size = len(payload["devices"])
    device_id = int(payload["devices"][rank])
    result_path = Path(work) / f"rank-{rank}.json"
    report: dict[str, Any] = {
        "rank": rank,
        "device_id": device_id,
        "binding_id": "unavailable",
        "passed": False,
        "gates": {},
    }
    hook: VllmSharedPoolHook | None = None
    caches: dict[str, Any] | None = None
    buffer: Any | None = None
    client: KapslKvControlClient | None = None
    live_lease: dict[str, Any] | None = None
    heartbeat: _LeaseHeartbeat | None = None
    process_group_started = False
    try:
        torch.cuda.set_device(device_id)
        distributed.init_process_group(
            backend="gloo",
            init_method=f"file://{init_path}",
            rank=rank,
            world_size=world_size,
        )
        process_group_started = True

        geometry = ProbeGeometry(**payload["geometry"])
        config, layout = _make_vllm_kv_config(torch, geometry)
        client = KapslKvControlClient(
            payload["endpoint"],
            payload["participant_id"],
            timeout_seconds=float(payload["timeout_seconds"]),
        )
        receipt = client.register(payload["registration"])
        if int(receipt["participant_epoch"]) != int(payload["participant_epoch"]):
            raise RuntimeError("rank observed a different participant epoch")
        rank_map = {
            index: int(mapped_device)
            for index, mapped_device in enumerate(payload["devices"])
        }
        binding = select_cuda_binding(
            receipt,
            config,
            rank_map if world_size > 1 else None,
            global_rank=rank,
        )
        report["binding_id"] = str(binding["binding_id"])

        hook = VllmSharedPoolHook(binding, config)
        buffer = hook._buffer
        raw = buffer.tensor
        if raw is None:
            raise RuntimeError("CUDA IPC importer did not expose a backing tensor")
        raw.fill_(_SENTINEL)
        torch.cuda.synchronize()
        before_hash = _tensor_sha256(raw)
        allocated_before = int(torch.cuda.memory_allocated(device_id))

        worker_utils = importlib.import_module("vllm.v1.worker.utils")
        with patch.object(
            torch,
            "zeros",
            side_effect=AssertionError(
                "vLLM attempted a second, backend-owned KV allocation"
            ),
        ):
            caches = worker_utils.allocate_kv_cache(
                config, torch.device("cuda", device_id), layout
            )
        torch.cuda.synchronize()
        allocated_after = int(torch.cuda.memory_allocated(device_id))
        if allocated_after != allocated_before:
            raise AssertionError(
                "vLLM allocator seam consumed PyTorch-owned CUDA memory: "
                f"before={allocated_before}, after={allocated_after}"
            )
        views = hook.attachment_views(caches)
        attachment = make_shared_pool_attachment(
            participant_epoch=int(receipt["participant_epoch"]),
            binding_id=str(binding["binding_id"]),
            shard={
                "tensor_parallel_rank": rank,
                "tensor_parallel_world_size": world_size,
                "pipeline_parallel_rank": 0,
                "pipeline_parallel_world_size": 1,
            },
            profile=payload["profile"],
            imported_bytes=hook.imported_bytes,
            views=views,
        )
        client.attach(attachment)
        report["gates"]["allocator_attachment"] = {
            "passed": True,
            "evidence": {
                "allocator_poisoned": True,
                "pytorch_cuda_allocation_delta_bytes": (
                    allocated_after - allocated_before
                ),
                "imported_bytes": hook.imported_bytes,
                "view_count": len(views),
                "raw_sha256_before_write": before_hash,
            },
        }

        distributed.barrier()
        lifecycle_box: list[Any] = [None]
        if rank == 0:
            client.activate(int(receipt["participant_epoch"]))
            live_lease, lifecycle_box[0] = _begin_lifecycle_probe(
                client,
                receipt,
                rank,
                world_size,
                str(binding["binding_id"]),
                geometry,
                int(payload["lease_ttl_ms"]),
            )
            heartbeat = _LeaseHeartbeat(
                client, int(payload["lease_ttl_ms"])
            )
            heartbeat.start()
        distributed.broadcast_object_list(lifecycle_box, src=0)
        distributed.barrier()

        native = _run_native_probe(torch, raw, caches[_LAYER_NAME], geometry)
        report["gates"]["backend_native_write"] = {
            "passed": True,
            "evidence": native["write"],
        }
        report["gates"]["backend_native_read"] = {
            "passed": True,
            "evidence": native["read"],
        }
        report["device"] = _device_evidence(torch, device_id)
        report["lifecycle"] = lifecycle_box[0]

        distributed.barrier()
        if rank == 0:
            if live_lease is None:
                raise AssertionError("rank zero lost its lifecycle lease")
            if heartbeat is None:
                raise AssertionError("rank zero lost its heartbeat worker")
            heartbeat.stop()
            heartbeat = None
            client.release(str(live_lease["lease_id"]))
            lifecycle_box[0]["heartbeat_renewal"] = True
            lifecycle_box[0].update(
                _finish_lifecycle_probe(
                    client,
                    geometry,
                    int(payload["lease_ttl_ms"]),
                )
            )
        distributed.broadcast_object_list(lifecycle_box, src=0)
        report["lifecycle"] = lifecycle_box[0]
        distributed.barrier()

        # The backend fence and destruction happen before claiming synchronized
        # detach.  The importer itself confirms that no tensor view still owns
        # the CUDA IPC mapping.
        torch.cuda.synchronize()
        hook.shutdown()
        caches = None
        del raw
        gc.collect()
        torch.cuda.synchronize()
        if buffer.mapping_open:
            raise RuntimeError("CUDA IPC mapping survived KV view destruction")

        if rank == 0:
            client.detach(
                _detach_request(receipt, rank, world_size, str(binding["binding_id"]))
            )
            lifecycle_box[0]["post_deactivation_reserve_rejected"] = (
                _expect_reserve_rejected(
                    client,
                    geometry,
                    int(payload["lease_ttl_ms"]),
                )
            )
        distributed.broadcast_object_list(lifecycle_box, src=0)
        distributed.barrier()
        if rank != 0:
            client.detach(
                _detach_request(receipt, rank, world_size, str(binding["binding_id"]))
            )
        distributed.barrier()

        report["lifecycle"] = lifecycle_box[0]
        report["passed"] = True
        write_json_atomic(result_path, report)
    except BaseException as error:
        report["error"] = "".join(traceback.format_exception(error))
        write_json_atomic(result_path, report)
        raise
    finally:
        if heartbeat is not None:
            try:
                heartbeat.stop(raise_failure=False)
            except Exception:
                pass
        if hook is not None:
            try:
                hook.shutdown()
            except Exception:
                pass
        caches = None
        gc.collect()
        if process_group_started:
            try:
                distributed.destroy_process_group()
            except Exception:
                pass


def _run_native_probe(
    torch: Any,
    raw: Any,
    kv_cache: Any,
    geometry: ProbeGeometry,
) -> dict[str, Any]:
    fa_utils = importlib.import_module("vllm.v1.attention.backends.fa_utils")
    flash_module = importlib.import_module("vllm.v1.attention.backends.flash_attn")
    reshape_and_cache = getattr(fa_utils, "reshape_and_cache_flash", None)
    flash_attention = getattr(fa_utils, "flash_attn_varlen_func", None)
    version_resolver = getattr(flash_module, "get_flash_attn_version", None)
    implementation = getattr(flash_module, "FlashAttentionImpl", None)
    if not callable(reshape_and_cache) or not callable(flash_attention):
        raise RuntimeError(
            "vLLM FLASH_ATTN native write/read functions are unavailable"
        )
    if not callable(version_resolver) or implementation is None:
        raise RuntimeError("vLLM FLASH_ATTN implementation surface is unavailable")
    update_globals = implementation.do_kv_cache_update.__globals__
    forward_globals = implementation.forward.__globals__
    if update_globals.get("reshape_and_cache_flash") is not reshape_and_cache:
        raise RuntimeError("probe writer is not bound into FlashAttentionImpl")
    if forward_globals.get("flash_attn_varlen_func") is not flash_attention:
        raise RuntimeError("probe reader is not bound into FlashAttentionImpl")
    flash_version = version_resolver(head_size=geometry.head_size)
    if flash_version is None:
        raise RuntimeError("no native FlashAttention version supports this GPU/config")

    cache_by_token = kv_cache.transpose(1, 2)
    key_cache, value_cache = cache_by_token.split(geometry.head_size, dim=-1)
    device = kv_cache.device
    dtype = kv_cache.dtype
    scale = geometry.head_size**-0.5
    one = torch.tensor(1.0, dtype=torch.float32, device=device)

    def values(count: int, offset: float) -> Any:
        elements = count * geometry.num_kv_heads * geometry.head_size
        base = torch.arange(elements, dtype=torch.float32, device=device)
        return torch.sin(base * 0.013 + offset).reshape(
            count, geometry.num_kv_heads, geometry.head_size
        ).to(dtype)

    def queries(count: int, offset: float) -> Any:
        elements = count * geometry.num_query_heads * geometry.head_size
        base = torch.arange(elements, dtype=torch.float32, device=device)
        return torch.cos(base * 0.017 + offset).reshape(
            count, geometry.num_query_heads, geometry.head_size
        ).to(dtype)

    def write(key: Any, value: Any, slots: list[int]) -> None:
        slot_mapping = torch.tensor(slots, dtype=torch.long, device=device)
        reshape_and_cache(
            key,
            value,
            key_cache,
            value_cache,
            slot_mapping,
            "auto",
            one,
            one,
        )
        torch.cuda.synchronize(device)

    def attend(query: Any, kv_len: int, blocks: list[int]) -> Any:
        query_len = int(query.shape[0])
        cu_query_lens = torch.tensor(
            [0, query_len], dtype=torch.int32, device=device
        )
        kv_lens = torch.tensor([kv_len], dtype=torch.int32, device=device)
        block_table = torch.tensor([blocks], dtype=torch.int32, device=device)
        output = flash_attention(
            q=query,
            k=key_cache,
            v=value_cache,
            cu_seqlens_q=cu_query_lens,
            seqused_k=kv_lens,
            max_seqlen_q=query_len,
            max_seqlen_k=kv_len,
            softmax_scale=scale,
            causal=True,
            window_size=(-1, -1),
            block_table=block_table,
            softcap=0,
            fa_version=flash_version,
        )
        torch.cuda.synchronize(device)
        reference = _reference_attention(
            torch,
            query,
            key_cache,
            value_cache,
            kv_len,
            blocks,
            scale,
        )
        torch.testing.assert_close(output, reference, atol=1.5e-2, rtol=1e-2)
        return output

    prefill_tokens = min(5, geometry.block_size - 2)
    if prefill_tokens < 2:
        raise ValueError("probe block size is too small for prefill and decode")
    key = values(prefill_tokens, 0.1)
    value = values(prefill_tokens, 1.1)
    write(key, value, list(range(prefill_tokens)))
    torch.testing.assert_close(key_cache[0, :prefill_tokens], key)
    torch.testing.assert_close(value_cache[0, :prefill_tokens], value)
    prefill_output = attend(queries(prefill_tokens, 2.1), prefill_tokens, [0])

    for step in range(2):
        token_index = prefill_tokens + step
        write(values(1, 3.1 + step), values(1, 4.1 + step), [token_index])
        attend(queries(1, 5.1 + step), token_index + 1, [0])

    causal_query = queries(1, 7.1)
    causal_before = attend(causal_query, prefill_tokens + 2, [0])
    value_cache[0, 0].add_(8)
    torch.cuda.synchronize(device)
    causal_after = attend(causal_query, prefill_tokens + 2, [0])
    causal_delta = float((causal_after.float() - causal_before.float()).abs().max())
    if causal_delta <= 1e-2:
        raise AssertionError("mutating Kapsl-owned V did not causally change attention")

    reuse_key = values(geometry.block_size, 8.1)
    reuse_value = values(geometry.block_size, 9.1)
    write(reuse_key, reuse_value, list(range(geometry.block_size)))
    reuse_output = attend(queries(1, 10.1), geometry.block_size, [0])

    maximum_block = geometry.num_blocks - 1
    maximum_slot = maximum_block * geometry.block_size
    write(values(1, 11.1), values(1, 12.1), [maximum_slot])
    maximum_output = attend(queries(1, 13.1), 1, [maximum_block])

    _assert_guards(torch, raw, geometry)
    after_hash = _tensor_sha256(raw)
    before_hash = hashlib.sha256(bytes([_SENTINEL]) * int(raw.numel())).hexdigest()
    if after_hash == before_hash:
        raise AssertionError("native KV writer did not change the imported allocation")

    return {
        "write": {
            "native_function": (
                "vllm.v1.attention.backends.fa_utils.reshape_and_cache_flash"
            ),
            "raw_sha256_after_write": after_hash,
            "guard_bytes_checked": geometry.guard_bytes * geometry.num_blocks,
            "prefill_tokens": prefill_tokens,
            "decode_writes": 2,
            "maximum_block_index": maximum_block,
            "production_implementation_binding": True,
        },
        "read": {
            "native_function": (
                "vllm.v1.attention.backends.fa_utils.flash_attn_varlen_func"
            ),
            "implementation": (
                f"{implementation.__module__}.{implementation.__qualname__}"
            ),
            "flash_attention_version": int(flash_version),
            "causal_mutation_max_delta": causal_delta,
            "production_implementation_binding": True,
            "prefill_output_norm": float(prefill_output.float().norm()),
            "reuse_output_norm": float(reuse_output.float().norm()),
            "maximum_block_output_norm": float(maximum_output.float().norm()),
        },
    }


def _reference_attention(
    torch: Any,
    query: Any,
    key_cache: Any,
    value_cache: Any,
    kv_len: int,
    blocks: list[int],
    scale: float,
) -> Any:
    block_indices = torch.tensor(blocks, dtype=torch.long, device=query.device)
    keys = key_cache[block_indices].reshape(
        -1, key_cache.shape[2], key_cache.shape[3]
    )[:kv_len]
    values = value_cache[block_indices].reshape(
        -1, value_cache.shape[2], value_cache.shape[3]
    )[:kv_len]
    if query.shape[1] != keys.shape[1]:
        repeat = query.shape[1] // keys.shape[1]
        keys = torch.repeat_interleave(keys, repeat, dim=1)
        values = torch.repeat_interleave(values, repeat, dim=1)
    scores = torch.einsum("qhd,khd->hqk", query.float() * scale, keys.float())
    query_len = int(query.shape[0])
    mask = torch.triu(
        torch.ones(query_len, kv_len, dtype=torch.bool, device=query.device),
        diagonal=kv_len - query_len + 1,
    )
    scores.masked_fill_(mask, float("-inf"))
    probabilities = torch.softmax(scores, dim=-1).to(values.dtype)
    return torch.einsum("hqk,khd->qhd", probabilities, values)


def _assert_guards(torch: Any, raw: Any, geometry: ProbeGeometry) -> None:
    for block in range(geometry.num_blocks):
        start = block * geometry.padded_page_bytes + geometry.dense_page_bytes
        guard = raw.narrow(0, start, geometry.guard_bytes)
        if not bool(torch.all(guard == _SENTINEL).item()):
            raise AssertionError(
                f"native attention corrupted guard bytes in block {block}"
            )


def _begin_lifecycle_probe(
    client: KapslKvControlClient,
    receipt: dict[str, Any],
    rank: int,
    world_size: int,
    binding_id: str,
    geometry: ProbeGeometry,
    ttl_ms: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    live = client.reserve(
        make_reserve_request(
            request_id="probe-live",
            sequence_id="probe-live",
            token_capacity=geometry.block_size,
            group_ids=[_GROUP_ID],
            ttl_ms=ttl_ms,
        )
    )
    try:
        client.detach(_detach_request(receipt, rank, world_size, binding_id))
    except KapslKvControlError as error:
        if error.kind != "invalid_request":
            raise RuntimeError(
                f"live-lease detach failed with unexpected error {error.kind!r}"
            ) from error
    else:
        raise AssertionError("runtime allowed detach while a KV lease was live")
    return live, {
        "activation_after_all_attachments": True,
        "live_lease_detach_rejected": True,
    }


def _finish_lifecycle_probe(
    client: KapslKvControlClient,
    geometry: ProbeGeometry,
    ttl_ms: int,
) -> dict[str, Any]:
    cancelled = client.reserve(
        make_reserve_request(
            request_id="probe-cancel",
            sequence_id="probe-cancel",
            token_capacity=geometry.block_size,
            group_ids=[_GROUP_ID],
            ttl_ms=ttl_ms,
        )
    )
    client.release(str(cancelled["lease_id"]))

    full = client.reserve(
        make_reserve_request(
            request_id="probe-full",
            sequence_id="probe-full",
            token_capacity=geometry.num_blocks * geometry.block_size,
            group_ids=[_GROUP_ID],
            ttl_ms=ttl_ms,
        )
    )
    try:
        client.reserve(
            make_reserve_request(
                request_id="probe-overflow",
                sequence_id="probe-overflow",
                token_capacity=1,
                group_ids=[_GROUP_ID],
                ttl_ms=ttl_ms,
            )
        )
    except KapslKvControlError as error:
        if error.kind != "capacity_exhausted":
            raise RuntimeError(
                f"capacity overflow failed with unexpected error {error.kind!r}"
            ) from error
    else:
        raise AssertionError("runtime admitted a request after capacity exhaustion")
    finally:
        client.release(str(full["lease_id"]))
    return {
        "cancellation_release": True,
        "capacity_exhaustion_rejected": True,
        "maximum_capacity_tokens": geometry.num_blocks * geometry.block_size,
    }


def _expect_reserve_rejected(
    client: KapslKvControlClient,
    geometry: ProbeGeometry,
    ttl_ms: int,
) -> bool:
    try:
        client.reserve(
            make_reserve_request(
                request_id="probe-after-deactivation",
                sequence_id="probe-after-deactivation",
                token_capacity=geometry.block_size,
                group_ids=[_GROUP_ID],
                ttl_ms=ttl_ms,
            )
        )
    except KapslKvControlError as error:
        if error.kind not in {"invalid_request", "not_found"}:
            raise
        return True
    raise AssertionError("runtime admitted a request after shared-pool deactivation")


def _detach_request(
    receipt: dict[str, Any], rank: int, world_size: int, binding_id: str
) -> dict[str, Any]:
    return make_shared_pool_detach_request(
        participant_epoch=int(receipt["participant_epoch"]),
        binding_ids=[binding_id],
        shard={
            "tensor_parallel_rank": rank,
            "tensor_parallel_world_size": world_size,
            "pipeline_parallel_rank": 0,
            "pipeline_parallel_world_size": 1,
        },
    )


def _make_vllm_kv_config(
    torch: Any, geometry: ProbeGeometry
) -> tuple[Any, Any]:
    interface = importlib.import_module("vllm.v1.kv_cache_interface")
    layout_module = importlib.import_module("vllm.v1.kv_cache_layout")
    dtype = getattr(torch, geometry.dtype)
    spec = interface.FullAttentionSpec(
        block_size=geometry.block_size,
        num_kv_heads=geometry.num_kv_heads,
        head_size=geometry.head_size,
        head_size_v=geometry.head_size,
        dtype=dtype,
        page_size_padded=geometry.padded_page_bytes,
    )
    tensor = interface.KVCacheTensor(
        size=geometry.allocation_bytes,
        layers=[_LAYER_NAME],
        layer_stride=geometry.allocation_bytes,
        block_stride=geometry.padded_page_bytes,
        offset=0,
    )
    group = interface.KVCacheGroupSpec(
        layer_names=[_LAYER_NAME], kv_cache_spec=spec
    )
    config = interface.KVCacheConfig(
        num_blocks=geometry.num_blocks,
        kv_cache_tensors=[tensor],
        kv_cache_groups=[group],
        kv_cache_layout="LBNHC",
    )
    return config, layout_module.KVCacheLayout.LBNHC


def _registration(
    participant_id: str,
    profile: dict[str, str],
    devices: list[int],
    geometry: ProbeGeometry,
) -> dict[str, Any]:
    domains = [{"kind": "cuda", "device_id": device} for device in devices]
    model_payload = json.dumps(
        {"profile": profile, "geometry": geometry.as_dict()},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    model_fingerprint = "sha256:" + hashlib.sha256(model_payload).hexdigest()
    capacity_groups = [
        {
            "group_id": _GROUP_ID,
            "pool_id": _POOL_ID,
            "allocation_granularity_tokens": geometry.block_size,
            "bytes_per_allocation": geometry.padded_page_bytes,
            "memory_domains": domains,
            "max_allocations": geometry.num_blocks,
        }
    ]
    topology = {
        "abi_version": dict(ABI_VERSION),
        "model_fingerprint": model_fingerprint,
        "shard": {
            "tensor_parallel_rank": 0,
            "tensor_parallel_world_size": len(devices),
            "pipeline_parallel_rank": 0,
            "pipeline_parallel_world_size": 1,
        },
        "cache_groups": [
            {
                "group_id": _GROUP_ID,
                "layers": [{"index": 0, "name": _LAYER_NAME}],
                "geometry": {
                    "kind": "paged_attention",
                    "block_size_tokens": geometry.block_size,
                    "kv_heads": geometry.num_kv_heads,
                    "key_head_dim": geometry.head_size,
                    "value_head_dim": geometry.head_size,
                    "element_type": "f16" if geometry.dtype == "float16" else "bf16",
                    "layout": {
                        "kind": "backend_native",
                        "layout_id": "vllm:LBNHC",
                    },
                },
                "policy": {"kind": "full_attention"},
            }
        ],
    }
    return shared_pool_registration(
        participant_id,
        model_fingerprint,
        capacity_groups,
        topology,
        profile,
        backend="vllm",
    )


def _expect_activation_rejected(
    client: KapslKvControlClient,
    receipt: dict[str, Any],
    geometry: ProbeGeometry,
    ttl_ms: int,
) -> str:
    try:
        client.activate(int(receipt["participant_epoch"]))
    except KapslKvControlError as error:
        if error.kind != "invalid_request":
            raise RuntimeError(
                f"pre-attachment activation failed with unexpected error {error.kind!r}"
            ) from error
        pass
    else:
        raise AssertionError("runtime activated a shared pool before worker attachment")
    try:
        client.reserve(
            make_reserve_request(
                request_id="probe-before-activation",
                sequence_id="probe-before-activation",
                token_capacity=geometry.block_size,
                group_ids=[_GROUP_ID],
                ttl_ms=ttl_ms,
            )
        )
    except KapslKvControlError as error:
        if error.kind != "invalid_request":
            raise RuntimeError(
                f"pre-activation reserve failed with unexpected error {error.kind!r}"
            ) from error
        return "runtime rejected activation and reserve before worker attachment"
    raise AssertionError("runtime admitted a reservation before shared-pool activation")


def _expect_unallowlisted_profile_rejected(
    endpoint: str,
    participant_id: str,
    profile: dict[str, str],
    devices: list[int],
    geometry: ProbeGeometry,
    timeout_seconds: float,
) -> str:
    unapproved = dict(profile)
    unapproved["profile_id"] = (
        profile["profile_id"] + f"/unapproved-{uuid.uuid4().hex}"
    )
    negative_id = participant_id + "-unapproved"
    client = KapslKvControlClient(
        endpoint, negative_id, timeout_seconds=timeout_seconds
    )
    try:
        client.register(_registration(negative_id, unapproved, devices, geometry))
    except KapslKvControlError as error:
        if error.kind != "invalid_capabilities":
            raise RuntimeError(
                f"unapproved profile failed with unexpected error {error.kind!r}"
            ) from error
        return "runtime rejected an exact unallowlisted profile before provisioning"
    raise AssertionError("runtime provisioned an unallowlisted shared-pool profile")


def _aggregate_gates(
    ranks: list[dict[str, Any]],
    *,
    expected_bindings: set[str],
    contract_evidence: str,
) -> dict[str, Any]:
    if not ranks or any(rank.get("passed") is not True for rank in ranks):
        raise RuntimeError("one or more CUDA rank probes did not pass")
    actual_bindings = {str(rank.get("binding_id")) for rank in ranks}
    if actual_bindings != expected_bindings:
        raise RuntimeError(
            "rank probes did not consume every provisioned binding exactly once"
        )
    for gate in (
        "allocator_attachment",
        "backend_native_write",
        "backend_native_read",
    ):
        if any(
            rank.get("gates", {}).get(gate, {}).get("passed") is not True
            for rank in ranks
        ):
            raise RuntimeError(f"rank coverage is incomplete for gate {gate}")
    lifecycle = ranks[0].get("lifecycle", {})
    required_lifecycle = {
        "activation_after_all_attachments",
        "live_lease_detach_rejected",
        "cancellation_release",
        "capacity_exhaustion_rejected",
        "heartbeat_renewal",
        "post_deactivation_reserve_rejected",
    }
    if any(lifecycle.get(field) is not True for field in required_lifecycle):
        raise RuntimeError("lifecycle evidence is incomplete")
    return {
        "contract": {"passed": True, "evidence": contract_evidence},
        "allocator_attachment": {
            "passed": True,
            "evidence": f"all {len(ranks)} ranks imported and aliased Kapsl bindings",
        },
        "backend_native_write": {
            "passed": True,
            "evidence": (
                f"native FlashAttention cache writes passed on {len(ranks)} ranks"
            ),
        },
        "backend_native_read": {
            "passed": True,
            "evidence": (
                f"reference and causal mutation reads passed on {len(ranks)} ranks"
            ),
        },
        "lifecycle": {
            "passed": True,
            "evidence": (
                "prefill/decode/reuse/max/cancel/exhaustion/detach gates passed"
            ),
        },
        "parallel_coverage": {
            "passed": True,
            "evidence": (
                f"{len(ranks)} distinct ranks consumed "
                f"{len(actual_bindings)} bindings"
            ),
        },
    }


def _read_rank_reports(work: str, world_size: int) -> list[dict[str, Any]]:
    ranks = []
    for rank in range(world_size):
        path = Path(work) / f"rank-{rank}.json"
        if not path.is_file():
            raise RuntimeError(f"CUDA rank {rank} produced no conformance report")
        with path.open("r", encoding="utf-8") as source:
            value = json.load(source)
        if not isinstance(value, dict):
            raise RuntimeError(f"CUDA rank {rank} report is not an object")
        ranks.append(value)
    return ranks


def _read_available_rank_reports(
    work: str, world_size: int
) -> list[dict[str, Any]]:
    ranks = []
    for rank in range(world_size):
        path = Path(work) / f"rank-{rank}.json"
        if not path.is_file():
            continue
        try:
            with path.open("r", encoding="utf-8") as source:
                value = json.load(source)
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(value, dict):
            ranks.append(value)
    return ranks


def _device_evidence(torch: Any, device_id: int) -> dict[str, Any]:
    properties = torch.cuda.get_device_properties(device_id)
    return {
        "name": str(properties.name),
        "compute_capability": f"{properties.major}.{properties.minor}",
        "total_memory_bytes": int(properties.total_memory),
    }


def _tensor_sha256(tensor: Any) -> str:
    payload = tensor.detach().cpu().contiguous().numpy().tobytes()
    return hashlib.sha256(payload).hexdigest()


def _installed_version(distribution: str) -> str | None:
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return None


def _cuda_driver_version() -> str:
    library_name = ctypes.util.find_library("cuda") or "libcuda.so.1"
    library = ctypes.CDLL(library_name)
    get_version = library.cuDriverGetVersion
    get_version.argtypes = [ctypes.POINTER(ctypes.c_int)]
    get_version.restype = ctypes.c_int
    raw = ctypes.c_int()
    result = get_version(ctypes.byref(raw))
    if result != 0:
        raise RuntimeError(f"cuDriverGetVersion failed with CUDA error {result}")
    major = raw.value // 1000
    minor = (raw.value % 1000) // 10
    return f"{major}.{minor}"


def _validate_digest(value: str, field: str) -> None:
    if not value.startswith("sha256:") or len(value) != 71:
        raise ValueError(f"--{field} must be sha256 followed by 64 hex digits")
    try:
        int(value[7:], 16)
    except ValueError as error:
        raise ValueError(f"--{field} contains a non-hex digest") from error


def _parse_devices(value: str) -> list[int]:
    try:
        devices = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as error:
        raise ValueError(
            "--devices must contain comma-separated CUDA ordinals"
        ) from error
    if not devices or any(device < 0 for device in devices):
        raise ValueError("--devices must contain non-negative CUDA ordinals")
    if len(set(devices)) != len(devices):
        raise ValueError("--devices must not contain duplicates")
    return devices


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Certify Kapsl-owned vLLM FLASH_ATTN KV memory on Linux/CUDA"
    )
    parser.add_argument("--endpoint", required=True, help="unix:/// Kapsl KV socket")
    parser.add_argument("--devices", required=True, help="CUDA ordinals, e.g. 0 or 0,1")
    parser.add_argument("--report", required=True, help="JSON result path")
    parser.add_argument(
        "--allowlist-output",
        help="write the runtime profile value here only when every gate passes",
    )
    parser.add_argument("--adapter-build-id", required=True)
    parser.add_argument("--backend-build-id", required=True)
    parser.add_argument("--runtime-build-id", required=True)
    parser.add_argument("--participant-id")
    parser.add_argument("--timeout-seconds", type=float, default=10.0)
    parser.add_argument("--lease-ttl-ms", type=int, default=30_000)
    parser.add_argument("--num-blocks", type=int, default=8)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--num-kv-heads", type=int, default=2)
    parser.add_argument("--num-query-heads", type=int, default=4)
    parser.add_argument("--head-size", type=int, default=64)
    parser.add_argument("--guard-bytes", type=int, default=256)
    parser.add_argument(
        "--dtype", choices=("float16", "bfloat16"), default="float16"
    )
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
