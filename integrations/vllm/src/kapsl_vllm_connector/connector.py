"""Thin compatibility layer for vLLM's experimental V1 connector API."""

from __future__ import annotations

import logging
import os
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib import metadata
from typing import TYPE_CHECKING, Any, Callable

from .client import KapslKvControlClient, KapslKvControlError
from .contract import (
    ABI_VERSION,
    make_resize_ack_request,
    make_resize_poll_request,
    make_shared_pool_attachment,
    make_reserve_request,
    opaque_registration,
    shared_pool_registration,
)
from .planning import (
    UINT64_MAX,
    extract_cache_group_topologies,
    shared_pool_block_stride,
    validate_certified_shared_pool_execution,
)
from .shared_pool import (
    SharedPoolImportError,
    VllmElasticBlockPool,
    VllmSharedPoolHook,
    select_cuda_binding,
    vllm_backing_geometry,
    vllm_distributed_rank,
)

if TYPE_CHECKING:
    import torch
    from vllm.forward_context import ForwardContext
    from vllm.v1.attention.backend import AttentionMetadata
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.request import Request

try:
    from vllm.distributed.kv_transfer.kv_connector.v1.base import (
        KVConnectorBase_V1,
        KVConnectorMetadata,
        KVConnectorRole,
        SupportsHMA,
    )
except ImportError as import_error:  # Keep protocol/client usable without vLLM.
    _VLLM_IMPORT_ERROR: ImportError | None = import_error

    class KVConnectorBase_V1:  # type: ignore[no-redef]
        pass

    class KVConnectorMetadata:  # type: ignore[no-redef]
        pass

    class KVConnectorRole:  # type: ignore[no-redef]
        pass

    class SupportsHMA:  # type: ignore[no-redef]
        pass
else:
    _VLLM_IMPORT_ERROR = None


logger = logging.getLogger(__name__)
ADAPTER_VERSION = "0.7.0"
ADAPTER_PROFILE_ID = "vllm-v1-packed-cuda-ipc/flash-attn"
ELASTIC_ADAPTER_PROFILE_ID = "vllm-v1-packed-cuda-vmm/flash-attn-blnhc"


@dataclass
class KapslConnectorMetadata(KVConnectorMetadata):
    """No worker transfer metadata is needed in opaque control-only mode."""


class KapslConnectorV1(KVConnectorBase_V1, SupportsHMA):
    """KV-connected vLLM participant with optional Kapsl-owned backing.

    This connector lets Kapsl admit and lease logical KV capacity before vLLM
    allocates request blocks. It intentionally reports zero external prefix
    hits and performs no worker-side copies, so vLLM never consumes KV bytes
    that Kapsl cannot actually deliver.
    """

    @property
    def requires_kv_delivery(self) -> bool:
        return False

    @classmethod
    def get_required_kvcache_layout(cls, vllm_config: Any) -> str | None:
        transfer = getattr(vllm_config, "kv_transfer_config", None)
        if transfer is None:
            return None
        raw = _extra(transfer, "kapsl_live_resize", False)
        if not isinstance(raw, bool):
            raise ValueError("kapsl_live_resize must be a boolean")
        return "BLNHC" if raw else None

    def __init__(
        self,
        vllm_config: Any,
        role: KVConnectorRole,
        kv_cache_config: "KVCacheConfig",
    ) -> None:
        if _VLLM_IMPORT_ERROR is not None:
            raise RuntimeError(
                "KapslConnectorV1 must be installed in an environment with vLLM V1"
            ) from _VLLM_IMPORT_ERROR
        if not hasattr(KVConnectorBase_V1, "on_new_request"):
            raise RuntimeError(
                "this vLLM build lacks the on_new_request KV hook required for "
                "Kapsl pre-allocation admission"
            )

        super().__init__(
            vllm_config=vllm_config,
            role=role,
            kv_cache_config=kv_cache_config,
        )
        endpoint = _required_extra(self._kv_transfer_config, "kapsl_control_endpoint")
        model_fingerprint = _required_extra(
            self._kv_transfer_config, "kapsl_model_fingerprint"
        )
        participant_base = str(
            _extra(self._kv_transfer_config, "kapsl_participant_id", "vllm")
        ).strip()
        if not participant_base:
            raise ValueError("kapsl_participant_id must not be empty")
        role_name = getattr(role, "name", str(role)).lower()
        self._is_scheduler = role_name == "scheduler"
        engine_id = str(getattr(self._kv_transfer_config, "engine_id", "engine"))
        self._mode = str(
            _extra(self._kv_transfer_config, "kapsl_kv_mode", "opaque")
        ).strip().lower()
        if self._mode not in {"opaque", "shared_pool"}:
            raise ValueError("kapsl_kv_mode must be 'opaque' or 'shared_pool'")
        raw_live_resize = _extra(
            self._kv_transfer_config, "kapsl_live_resize", False
        )
        if not isinstance(raw_live_resize, bool):
            raise ValueError("kapsl_live_resize must be a boolean")
        self._live_resize = raw_live_resize
        if self._live_resize and self._mode != "shared_pool":
            raise ValueError("kapsl_live_resize requires shared_pool mode")
        parallel_config = getattr(vllm_config, "parallel_config", None)
        tensor_parallel_size = int(
            getattr(parallel_config, "tensor_parallel_size", 1) or 1
        )
        if self._mode == "shared_pool":
            if self._live_resize:
                _validate_elastic_shared_pool_execution(vllm_config, kv_cache_config)
            else:
                _validate_shared_pool_execution(vllm_config)
        self._participant_id = (
            f"{participant_base}:{engine_id}"
            if self._mode == "shared_pool"
            else f"{participant_base}:{engine_id}:{role_name}"
        )
        memory_domains = _required_memory_domains(self._kv_transfer_config)
        if self._mode == "shared_pool" and any(
            domain.get("kind") != "cuda" for domain in memory_domains
        ):
            raise ValueError("vLLM shared_pool currently supports CUDA domains only")
        rank_device_map = (
            _validated_shared_rank_device_map(
                self._kv_transfer_config,
                memory_domains,
                tensor_parallel_size,
            )
            if self._mode == "shared_pool"
            else None
        )
        self._capacity_groups = _vllm_capacity_groups(
            kv_cache_config,
            memory_domains,
            shared_pool=self._mode == "shared_pool",
        )
        self._group_ids = [group["group_id"] for group in self._capacity_groups]
        shared_topology = (
            _vllm_topology(
                kv_cache_config,
                model_fingerprint,
                tensor_parallel_world_size=tensor_parallel_size,
            )
            if self._mode == "shared_pool"
            else None
        )
        shared_profile = (
            _vllm_adapter_profile(vllm_config, live_resize=self._live_resize)
            if self._mode == "shared_pool"
            else None
        )
        provisioning_grant = (
            _provisioning_grant(self._kv_transfer_config)
            if self._mode == "shared_pool"
            else None
        )
        registration = (
            shared_pool_registration(
                self._participant_id,
                model_fingerprint,
                self._capacity_groups,
                shared_topology,
                shared_profile,
                backend="vllm",
                provisioning_grant=provisioning_grant,
                live_resize=self._live_resize,
            )
            if self._mode == "shared_pool"
            else opaque_registration(
                self._participant_id,
                model_fingerprint,
                self._capacity_groups,
                backend="vllm",
            )
        )
        timeout_ms = int(_extra(self._kv_transfer_config, "kapsl_timeout_ms", 2000))
        self._lease_ttl_ms = int(
            _extra(self._kv_transfer_config, "kapsl_lease_ttl_ms", 30_000)
        )
        if self._lease_ttl_ms < 1000:
            raise ValueError("kapsl_lease_ttl_ms must be at least 1000")
        self._client: KapslKvControlClient | None = None
        self._leases: dict[str, dict[str, Any]] = {}
        self._lease_lock = threading.RLock()
        self._control_failure: KapslKvControlError | None = None
        self._heartbeat_stop = threading.Event()
        self._heartbeat_thread: threading.Thread | None = None
        self._shared_pool_hook: VllmSharedPoolHook | None = None
        self._participant_epoch: int | None = None
        self._shared_shard: dict[str, int] | None = None
        self._shared_profile: dict[str, str] | None = shared_profile
        self._shared_attached = False
        self._shared_active = False
        self._activation_lock = threading.Lock()
        self._resize_lock = threading.RLock()
        self._resize_pending = False
        self._resize_applied_generation = 0
        self._pending_scheduler_operations: list[dict[str, Any]] = []
        self._elastic_block_pool: VllmElasticBlockPool | None = None
        self._worker_forward_lock = threading.Lock()
        self._worker_forward_active = False
        if self._is_scheduler or self._mode == "shared_pool":
            self._client = KapslKvControlClient(
                endpoint,
                self._participant_id,
                timeout_seconds=timeout_ms / 1000.0,
            )
            received_handles: list[int] = []
            if self._live_resize and not self._is_scheduler:
                receipt, received_handles = self._client.register_with_handles(
                    registration
                )
            else:
                receipt = self._client.register(registration)
            try:
                self._participant_epoch = int(receipt["participant_epoch"])
                if self._live_resize:
                    initial_blocks, maximum_blocks = _elastic_receipt_block_counts(
                        receipt
                    )
                    if self._is_scheduler:
                        self._elastic_block_pool = VllmElasticBlockPool(
                            initial_blocks, maximum_blocks
                        )
                if self._mode == "shared_pool" and not self._is_scheduler:
                    global_rank = (
                        0 if tensor_parallel_size == 1 else vllm_distributed_rank()
                    )
                    binding = select_cuda_binding(
                        receipt,
                        kv_cache_config,
                        rank_device_map,
                        global_rank=global_rank,
                        live_resize=self._live_resize,
                    )
                    self._shared_pool_hook = VllmSharedPoolHook(
                        binding,
                        kv_cache_config,
                        handles=received_handles if self._live_resize else None,
                    )
                    if shared_topology is None:
                        raise AssertionError("shared topology was not constructed")
                    self._shared_shard = dict(shared_topology["shard"])
                    self._shared_shard["tensor_parallel_rank"] = global_rank
            finally:
                _close_os_handles(received_handles)
            logger.info(
                "registered Kapsl KV participant %s in %s mode (%s role)",
                self._participant_id,
                self._mode,
                role_name,
            )
        self._start_scheduler_control(engine_id)

    # Scheduler-side lifecycle -------------------------------------------------

    def _start_scheduler_control(self, engine_id: str) -> None:
        if not self._is_scheduler:
            return
        # The pinned vLLM build constructs its scheduler connector only after
        # every worker has initialized and registered its KV tensors. Activate
        # here so Kapsl can publish the fully attached participant as Routable;
        # deferring activation until on_new_request creates a readiness cycle
        # because Kapsl deliberately will not route that first request yet.
        # Coordinator activation still verifies the complete binding set and
        # therefore fails startup closed if vLLM's construction order drifts.
        if self._mode == "shared_pool":
            self._ensure_shared_active()
        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop,
            name=f"kapsl-kv-heartbeat-{engine_id}",
            daemon=True,
        )
        self._heartbeat_thread.start()

    def on_new_request(self, request: "Request") -> None:
        if not self._is_scheduler:
            return
        self._raise_if_control_failed()
        self._ensure_shared_active()
        request_id = _request_id(request)
        with self._lease_lock:
            if request_id in self._leases:
                return
        reserve_request = make_reserve_request(
            request_id=request_id,
            sequence_id=request_id,
            token_capacity=_request_token_capacity(request),
            group_ids=self._group_ids,
            priority=_request_priority(request),
            ttl_ms=self._lease_ttl_ms,
        )
        client = self._control_client()
        lease = client.reserve(reserve_request)
        with self._lease_lock:
            previous = self._leases.setdefault(request_id, lease)
        if previous is not lease:
            # A duplicate callback raced with this one. Return the redundant
            # lease immediately instead of double-accounting the request.
            client.release(str(lease["lease_id"]))

    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int | None, bool]:
        del request, num_computed_tokens
        self._raise_if_control_failed()
        # Opaque phase 1 does not have a worker data plane, so claiming a cache
        # hit here would make vLLM skip computation without restoring KV bytes.
        return 0, False

    def update_state_after_alloc(
        self,
        request: "Request",
        blocks: "KVCacheBlocks",
        num_external_tokens: int,
    ) -> None:
        del request, blocks, num_external_tokens
        self._raise_if_control_failed()

    def build_connector_meta(
        self, scheduler_output: "SchedulerOutput"
    ) -> KapslConnectorMetadata:
        del scheduler_output
        self._raise_if_control_failed()
        self._apply_scheduler_resizes()
        return KapslConnectorMetadata()

    def bind_gpu_block_pool(self, gpu_block_pool: Any) -> None:
        if not self._is_scheduler or not self._live_resize:
            return
        elastic = self._elastic_block_pool
        if elastic is None:
            raise SharedPoolImportError(
                "elastic scheduler has no negotiated block-pool geometry"
            )
        elastic.bind(gpu_block_pool)

    def has_pending_push_work(self) -> bool:
        with self._resize_lock:
            pending = self._resize_pending or bool(
                self._pending_scheduler_operations
            )
            elastic = self._elastic_block_pool
            # The pinned vLLM core blocks on its request queue after the last
            # request finishes. A shrink is normally requested immediately
            # after that transition, so keep the connector's supported
            # zero-token push-work loop alive while capacity is still above
            # the startup minimum. Otherwise the background control poll can
            # discover retire_scheduler work but cannot wake the sleeping
            # scheduler thread to apply and acknowledge it.
            above_initial_capacity = (
                elastic is not None
                and elastic.current_blocks > elastic.initial_blocks
            )
            return pending or above_initial_capacity

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, dict[str, Any] | None]:
        del block_ids
        request_id = _request_id(request)
        with self._lease_lock:
            lease = self._leases.pop(request_id, None)
        if lease is None:
            return False, None

        lease_id = str(lease["lease_id"])
        first_error: Exception | None = None
        try:
            self._control_client().commit(lease_id, _request_computed_tokens(request))
        except Exception as error:  # release must still be attempted
            first_error = error
        try:
            self._control_client().release(lease_id)
        except Exception as error:
            if first_error is None:
                first_error = error
            else:
                logger.error("release also failed for KV lease %s: %s", lease_id, error)
        if first_error is not None:
            raise first_error
        return False, None

    def request_finished_all_groups(
        self,
        request: "Request",
        block_ids: tuple[list[int], ...],
    ) -> tuple[bool, dict[str, Any] | None]:
        del block_ids
        return self.request_finished(request, [])

    # Worker-side interface ----------------------------------------------------

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs: Any) -> None:
        del kwargs
        if self._is_scheduler or not self._live_resize:
            return
        has_model_forward = getattr(forward_context, "attn_metadata", None) is not None
        self._worker_forward_lock.acquire()
        self._worker_forward_active = True
        try:
            self._apply_worker_resizes()
            if not has_model_forward:
                # Pinned vLLM uses a zero-token connector step to drain
                # has_pending_push_work() while idle. There is no attention
                # forward to fence in that path and wait_for_save is skipped.
                self._worker_forward_active = False
                self._worker_forward_lock.release()
        except Exception:
            self._worker_forward_active = False
            self._worker_forward_lock.release()
            raise

    def wait_for_layer_load(self, layer_name: str) -> None:
        del layer_name

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: "torch.Tensor",
        attn_metadata: "AttentionMetadata",
        **kwargs: Any,
    ) -> None:
        del layer_name, kv_layer, attn_metadata, kwargs

    def wait_for_save(self) -> None:
        if self._worker_forward_active:
            self._worker_forward_active = False
            self._worker_forward_lock.release()
        return None

    def build_connector_worker_meta(self) -> None:
        return None

    def register_kv_caches(self, kv_caches: dict[str, "torch.Tensor"]) -> None:
        if self._mode != "shared_pool" or self._is_scheduler:
            return
        if self._shared_attached:
            return
        hook = self._shared_pool_hook
        if hook is None or not hook.used:
            raise SharedPoolImportError(
                "vLLM registered KV tensors without consuming the Kapsl CUDA IPC pool"
            )
        if (
            self._participant_epoch is None
            or self._shared_shard is None
            or self._shared_profile is None
        ):
            raise SharedPoolImportError(
                "shared-pool registration state is incomplete before attachment"
            )
        attachment = make_shared_pool_attachment(
            participant_epoch=self._participant_epoch,
            binding_id=hook.binding_id,
            shard=self._shared_shard,
            profile=self._shared_profile,
            imported_bytes=hook.imported_bytes,
            mapped_bytes=hook.mapped_bytes if self._live_resize else None,
            views=hook.attachment_views(kv_caches),
        )
        self._control_client().attach(attachment)
        self._shared_attached = True
        logger.info(
            "attached Kapsl shared KV binding %s for participant %s",
            hook.binding_id,
            self._participant_id,
        )

    def shutdown(self) -> None:
        if self._worker_forward_active:
            self._worker_forward_active = False
            self._worker_forward_lock.release()
        if self._shared_pool_hook is not None:
            self._shared_pool_hook.shutdown()
            self._shared_pool_hook = None
        if not self._is_scheduler:
            return
        self._heartbeat_stop.set()
        if self._heartbeat_thread is not None:
            self._heartbeat_thread.join(timeout=max(1.0, self._lease_ttl_ms / 1000.0))
        with self._lease_lock:
            leases = list(self._leases.values())
            self._leases.clear()
        for lease in leases:
            lease_id = str(lease["lease_id"])
            try:
                self._control_client().release(lease_id)
            except KapslKvControlError as error:
                logger.warning("failed to release KV lease %s at shutdown: %s", lease_id, error)

    def _heartbeat_loop(self) -> None:
        interval_seconds = (
            0.25 if self._live_resize else max(0.25, self._lease_ttl_ms / 3000.0)
        )
        while not self._heartbeat_stop.wait(interval_seconds):
            if self._live_resize:
                try:
                    self._poll_scheduler_resizes()
                except KapslKvControlError as error:
                    with self._lease_lock:
                        self._control_failure = error
                    logger.error(
                        "lost Kapsl live-resize authority; scheduler will fail closed: %s",
                        error,
                    )
                    return
            with self._lease_lock:
                has_live_leases = bool(self._leases)
            if not has_live_leases:
                continue
            try:
                self._control_client().heartbeat()
            except KapslKvControlError as error:
                with self._lease_lock:
                    if not self._leases:
                        continue
                    self._control_failure = error
                logger.error(
                    "lost Kapsl authority for active KV leases; scheduler will fail closed: %s",
                    error,
                )
                return

    def _poll_scheduler_resizes(self) -> None:
        if self._participant_epoch is None:
            raise SharedPoolImportError("elastic scheduler has no participant epoch")
        operations, handles, pending = self._control_client().poll_resize_state_with_handles(
            make_resize_poll_request(
                participant_epoch=self._participant_epoch,
                actor={"role": "scheduler"},
                applied_generation=self._resize_applied_generation,
            )
        )
        try:
            if handles:
                raise SharedPoolImportError(
                    "scheduler resize phase received CUDA allocation handles"
                )
            with self._resize_lock:
                known = {
                    int(operation["resize_generation"])
                    for operation in self._pending_scheduler_operations
                }
                self._pending_scheduler_operations.extend(
                    operation
                    for operation in operations
                    if int(operation["resize_generation"])
                    > self._resize_applied_generation
                    and int(operation["resize_generation"]) not in known
                )
                self._resize_pending = pending
        finally:
            _close_os_handles(handles)

    def _apply_scheduler_resizes(self) -> None:
        if not self._live_resize:
            return
        elastic = self._elastic_block_pool
        if elastic is None:
            raise SharedPoolImportError("elastic scheduler block pool is unavailable")
        with self._resize_lock:
            operations = list(self._pending_scheduler_operations)
            self._pending_scheduler_operations.clear()
        for operation in operations:
            stage = str(operation["stage"])
            if stage not in {"activate_scheduler", "retire_scheduler"}:
                raise SharedPoolImportError("scheduler received a worker resize stage")
            target = int(operation["target_block_count"])
            elastic.apply(target)
            generation = int(operation["resize_generation"])
            self._control_client().ack_resize(
                make_resize_ack_request(
                    participant_epoch=int(operation["participant_epoch"]),
                    actor={"role": "scheduler"},
                    binding_id=str(operation["binding_id"]),
                    resize_generation=generation,
                    stage=stage,
                    applied_block_count=target,
                )
            )
            self._resize_applied_generation = max(
                self._resize_applied_generation, generation
            )

    def _apply_worker_resizes(self) -> None:
        if (
            not self._live_resize
            or self._participant_epoch is None
            or self._shared_shard is None
        ):
            return
        hook = self._shared_pool_hook
        if hook is None:
            raise SharedPoolImportError("elastic worker has no shared-pool hook")
        operations, handles, _ = self._control_client().poll_resize_state_with_handles(
            make_resize_poll_request(
                participant_epoch=self._participant_epoch,
                actor={"role": "worker", "shard": self._shared_shard},
                applied_generation=self._resize_applied_generation,
            )
        )
        try:
            for operation in operations:
                hook.apply_worker_resize(operation, handles)
                target = int(operation["target_block_count"])
                generation = int(operation["resize_generation"])
                self._control_client().ack_resize(
                    make_resize_ack_request(
                        participant_epoch=int(operation["participant_epoch"]),
                        actor={"role": "worker", "shard": self._shared_shard},
                        binding_id=str(operation["binding_id"]),
                        resize_generation=generation,
                        stage=str(operation["stage"]),
                        applied_block_count=target,
                    )
                )
                self._resize_applied_generation = max(
                    self._resize_applied_generation, generation
                )
        finally:
            _close_os_handles(handles)

    def _ensure_shared_active(self) -> None:
        if self._mode != "shared_pool" or self._shared_active:
            return
        with self._activation_lock:
            if self._shared_active:
                return
            if self._participant_epoch is None:
                raise SharedPoolImportError(
                    "shared-pool participant has no registration epoch"
                )
            self._control_client().activate(self._participant_epoch)
            self._shared_active = True
            logger.info(
                "activated Kapsl shared KV participant %s epoch=%s",
                self._participant_id,
                self._participant_epoch,
            )

    def _raise_if_control_failed(self) -> None:
        with self._lease_lock:
            failure = self._control_failure
        if failure is not None:
            raise KapslKvControlError(
                f"Kapsl KV authority heartbeat failed: {failure}",
                kind=failure.kind,
            ) from failure

    def _control_client(self) -> KapslKvControlClient:
        if self._client is None:
            raise RuntimeError("KV control client is not initialized for this connector role")
        return self._client


def _extra(config: Any, key: str, default: Any) -> Any:
    getter = getattr(config, "get_from_extra_config", None)
    if callable(getter):
        return getter(key, default)
    values = getattr(config, "kv_connector_extra_config", None) or {}
    return values.get(key, default)


def _required_extra(config: Any, key: str) -> str:
    value = str(_extra(config, key, "")).strip()
    if not value:
        raise ValueError(f"{key} is required for KapslConnectorV1")
    return value


def _validate_shared_pool_execution(vllm_config: Any) -> None:
    validate_certified_shared_pool_execution(vllm_config)


def _validate_elastic_shared_pool_execution(
    vllm_config: Any, kv_cache_config: Any
) -> None:
    _validate_shared_pool_execution(vllm_config)
    if str(getattr(kv_cache_config, "kv_cache_layout", "")).strip().upper() != "BLNHC":
        raise SharedPoolImportError(
            "live-resize shared_pool requires vLLM's block-outermost BLNHC layout"
        )
    concurrent_batches = getattr(vllm_config, "max_concurrent_batches", 1)
    if not isinstance(concurrent_batches, int) or isinstance(concurrent_batches, bool):
        raise SharedPoolImportError("vLLM max_concurrent_batches must be an integer")
    if concurrent_batches != 1:
        raise SharedPoolImportError(
            "live-resize shared_pool requires one in-flight model batch per worker"
        )


def _vllm_adapter_profile(
    vllm_config: Any, *, live_resize: bool = False
) -> dict[str, str]:
    # Keep the production profile coupled to the same constraints exercised by
    # the hardware probe.  A generic "vLLM" profile would accidentally allow a
    # build to switch from FlashAttention to another reader at startup.
    _validate_shared_pool_execution(vllm_config)
    try:
        backend_version = metadata.version("vllm")
    except metadata.PackageNotFoundError:
        try:
            import vllm

            backend_version = str(getattr(vllm, "__version__", "")).strip()
        except ImportError as error:
            raise SharedPoolImportError(
                "cannot identify the vLLM build for shared-pool attachment"
            ) from error
    if not backend_version:
        raise SharedPoolImportError(
            "cannot identify the vLLM build for shared-pool attachment"
        )
    return {
        "adapter_id": "kapsl-vllm-connector",
        "adapter_version": ADAPTER_VERSION,
        "backend_version": backend_version,
        "profile_id": (
            ELASTIC_ADAPTER_PROFILE_ID if live_resize else ADAPTER_PROFILE_ID
        ),
    }


def _request_id(request: Any) -> str:
    value = str(getattr(request, "request_id", "")).strip()
    if not value:
        raise ValueError("vLLM request has no request_id")
    return value


def _request_token_capacity(request: Any) -> int:
    prompt = getattr(request, "prompt_token_ids", None) or []
    all_tokens = getattr(request, "all_token_ids", None) or prompt
    sampling = getattr(request, "sampling_params", None)
    max_tokens = getattr(sampling, "max_tokens", 0) if sampling is not None else 0
    return max(1, len(all_tokens) + max(0, int(max_tokens or 0)))


def _request_computed_tokens(request: Any) -> int:
    explicit = getattr(request, "num_computed_tokens", None)
    if explicit is not None:
        return max(0, int(explicit))
    return len(getattr(request, "all_token_ids", None) or [])


def _request_priority(request: Any) -> int:
    return int(getattr(request, "priority", 0) or 0)


def _required_memory_domains(config: Any) -> list[dict[str, Any]]:
    raw_domains = _extra(config, "kapsl_memory_domains", None)
    if (
        not isinstance(raw_domains, Sequence)
        or isinstance(raw_domains, (str, bytes))
        or not raw_domains
    ):
        raise ValueError(
            "kapsl_memory_domains must be a non-empty list of runtime-visible domains"
        )
    domains: list[dict[str, Any]] = []
    for index, raw_domain in enumerate(raw_domains):
        if not isinstance(raw_domain, Mapping):
            raise ValueError(f"kapsl_memory_domains[{index}] must be an object")
        domains.append(dict(raw_domain))
    return domains


def _provisioning_grant(config: Any) -> dict[str, Any] | None:
    raw_grant = _extra(config, "kapsl_provisioning_grant", None)
    if raw_grant is None:
        return None
    if not isinstance(raw_grant, Mapping):
        raise ValueError("kapsl_provisioning_grant must be an object")
    return dict(raw_grant)


def _elastic_receipt_block_counts(receipt: Mapping[str, Any]) -> tuple[int, int]:
    pools = receipt.get("shared_pools")
    if not isinstance(pools, list) or not pools:
        raise SharedPoolImportError("elastic registration returned no shared pools")
    shapes: set[tuple[int, int, int]] = set()
    for pool in pools:
        if not isinstance(pool, Mapping):
            raise SharedPoolImportError("elastic shared pool must be an object")
        elastic = pool.get("elastic")
        if not isinstance(elastic, Mapping):
            raise SharedPoolImportError("elastic shared pool metadata is missing")
        shapes.add(
            (
                _positive_integer(
                    elastic.get("minimum_block_count"), "minimum_block_count"
                ),
                _positive_integer(
                    elastic.get("mapped_block_count"), "mapped_block_count"
                ),
                _positive_integer(
                    elastic.get("maximum_block_count"), "maximum_block_count"
                ),
            )
        )
    if len(shapes) != 1:
        raise SharedPoolImportError(
            "tensor-parallel elastic bindings must share one block-count state"
        )
    minimum, initial, maximum = shapes.pop()
    if minimum > initial or initial > maximum:
        raise SharedPoolImportError(
            "elastic minimum/mapped capacity exceeds its next capacity bound"
        )
    return initial, maximum


def _close_os_handles(handles: Sequence[int]) -> None:
    for descriptor in handles:
        try:
            os.close(descriptor)
        except OSError:
            pass


def _positive_integer(value: Any, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _vllm_capacity_groups(
    kv_cache_config: Any,
    memory_domains: Sequence[Mapping[str, Any]],
    *,
    shared_pool: bool = False,
    spec_kind_classifier: Callable[[Any], Any] | None = None,
) -> list[dict[str, Any]]:
    """Extract byte-accounting hints without exposing vLLM block handles."""

    groups = getattr(kv_cache_config, "kv_cache_groups", None)
    raw_max_allocations = getattr(kv_cache_config, "num_blocks", None)
    if not groups or raw_max_allocations is None:
        raise ValueError(
            "vLLM KVCacheConfig must expose non-empty kv_cache_groups and num_blocks"
        )
    max_allocations = _positive_integer(raw_max_allocations, "vLLM num_blocks")

    # Modern packed layouts publish the exact shared allocation size. Opaque
    # compatibility keeps the page-derived fallback for older vLLM releases.
    if shared_pool:
        topologies = extract_cache_group_topologies(
            kv_cache_config,
            spec_kind_classifier=spec_kind_classifier,
        )
        group_shapes = [
            (group.block_size_tokens, group.bytes_per_group_block)
            for group in topologies
        ]
        certified_pool_stride = shared_pool_block_stride(topologies)
        _, configured_blocks, pool_stride = vllm_backing_geometry(kv_cache_config)
        if configured_blocks != max_allocations:
            raise AssertionError("vLLM backing geometry changed during extraction")
        if pool_stride != certified_pool_stride:
            raise ValueError(
                "resolved packed allocation block stride disagrees with "
                "certified cache-group page accounting"
            )
    else:
        group_shapes = []
        for index, group in enumerate(groups):
            spec = getattr(group, "kv_cache_spec", None)
            layer_names = getattr(group, "layer_names", None)
            if spec is None or not layer_names:
                raise ValueError(
                    f"vLLM KV cache group {index} has no spec or layers"
                )
            block_size = _positive_integer(
                getattr(spec, "block_size", None),
                f"vLLM KV cache group {index} block_size",
            )
            page_size = _positive_integer(
                getattr(spec, "page_size_bytes", None),
                f"vLLM KV cache group {index} page_size_bytes",
            )
            # UniformTypeKVCacheSpecs.page_size_bytes already sums its per-layer
            # specs; an ordinary merged spec describes one layer and must be
            # scaled by the number of layers sharing the block table.
            group_bytes = (
                page_size
                if getattr(spec, "kv_cache_specs", None) is not None
                else page_size * len(layer_names)
            )
            group_shapes.append((block_size, group_bytes))
        # vLLM HMA groups alias one backing block pool; a shared pool ID tells
        # Kapsl to charge the maximum reservation rather than summing groups.
        pool_stride = max(group_bytes for _, group_bytes in group_shapes)
    capacity_groups: list[dict[str, Any]] = []
    for index, (block_size, _) in enumerate(group_shapes):
        capacity_groups.append(
            {
                "group_id": f"vllm.group.{index}",
                "pool_id": "vllm.pool.0",
                "allocation_granularity_tokens": block_size,
                "bytes_per_allocation": pool_stride,
                "memory_domains": [dict(domain) for domain in memory_domains],
                "max_allocations": max_allocations,
            }
        )
    return capacity_groups


def _vllm_topology(
    kv_cache_config: Any,
    model_fingerprint: str,
    *,
    tensor_parallel_world_size: int = 1,
    spec_kind_classifier: Callable[[Any], Any] | None = None,
) -> dict[str, Any]:
    """Translate supported vLLM attention specs into ABI topology metadata."""

    raw_layout_id = getattr(kv_cache_config, "kv_cache_layout", None)
    if not isinstance(raw_layout_id, str) or not raw_layout_id.strip():
        raise ValueError("resolved vLLM KVCacheConfig must expose kv_cache_layout")
    layout_id = raw_layout_id.strip()
    resolved_groups = extract_cache_group_topologies(
        kv_cache_config,
        spec_kind_classifier=spec_kind_classifier,
    )
    cache_groups: list[dict[str, Any]] = []
    next_layer_index = 0
    for group in resolved_groups:
        if group.policy_kind == "sliding_window":
            policy = {
                "kind": "sliding_window",
                "window_tokens": group.window_tokens,
            }
        else:
            policy = {"kind": "full_attention"}
        layers = [
            {"index": next_layer_index + offset, "name": str(layer_name)}
            for offset, layer_name in enumerate(group.layers)
        ]
        next_layer_index += len(layers)
        cache_groups.append(
            {
                "group_id": group.group_id,
                "layers": layers,
                "geometry": {
                    "kind": "paged_attention",
                    "block_size_tokens": group.block_size_tokens,
                    "kv_heads": group.kv_heads,
                    "key_head_dim": group.key_head_dim,
                    "value_head_dim": group.value_head_dim,
                    "element_type": _element_type(group.element_type.name),
                    "layout": {
                        "kind": "backend_native",
                        "layout_id": f"vllm:{layout_id}",
                    },
                },
                "policy": policy,
            }
        )
    return {
        "abi_version": dict(ABI_VERSION),
        "model_fingerprint": model_fingerprint,
        # One registration describes the collective vLLM engine. Physical TP
        # replicas are enumerated by capacity_model.memory_domains.
        "shard": {
            "tensor_parallel_rank": 0,
            "tensor_parallel_world_size": tensor_parallel_world_size,
            "pipeline_parallel_rank": 0,
            "pipeline_parallel_world_size": 1,
        },
        "cache_groups": cache_groups,
    }


def _element_type(dtype: Any) -> dict[str, Any]:
    name = str(dtype).lower().removeprefix("torch.")
    known = {
        "float16": "f16",
        "half": "f16",
        "bfloat16": "bf16",
        "float32": "f32",
        "float": "f32",
        "int8": "i8",
        "float8_e4m3fn": "fp8_e4m3",
        "float8_e4m3fnuz": "fp8_e4m3",
    }
    if name in known:
        return {"kind": known[name]}
    if not name or name == "none":
        raise ValueError("vLLM shared_pool KV dtype is unavailable")
    return {"kind": "custom", "name": name}


def _strict_nonnegative_integer(value: Any, field: str) -> int:
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or value < 0
        or value > UINT64_MAX
    ):
        raise ValueError(
            f"{field} must be a non-negative unsigned 64-bit integer"
        )
    return value


def _canonical_rank(value: Any) -> int:
    if isinstance(value, int) and not isinstance(value, bool):
        return _strict_nonnegative_integer(value, "kapsl_rank_device_map rank")
    if isinstance(value, str) and (
        value == "0"
        or (
            value
            and value[0] in "123456789"
            and all(character in "0123456789" for character in value[1:])
        )
    ):
        return _strict_nonnegative_integer(
            int(value), "kapsl_rank_device_map rank"
        )
    raise ValueError(
        "kapsl_rank_device_map ranks must be canonical non-negative integers"
    )


def _rank_device_map(config: Any) -> dict[int, int] | None:
    raw_map = _extra(config, "kapsl_rank_device_map", None)
    if raw_map is None:
        return None
    if not isinstance(raw_map, Mapping):
        raise ValueError("kapsl_rank_device_map must be an object")
    result: dict[int, int] = {}
    for raw_rank, raw_device_id in raw_map.items():
        rank = _canonical_rank(raw_rank)
        device_id = _strict_nonnegative_integer(
            raw_device_id, "kapsl_rank_device_map device ID"
        )
        if rank in result:
            raise ValueError(
                "kapsl_rank_device_map contains duplicate or colliding ranks"
            )
        result[rank] = device_id
    if not result:
        raise ValueError("kapsl_rank_device_map must not be empty")
    return result


def _validated_shared_rank_device_map(
    config: Any,
    memory_domains: Sequence[Mapping[str, Any]],
    tensor_parallel_size: int,
) -> dict[int, int] | None:
    tensor_parallel_size = _positive_integer(
        tensor_parallel_size, "vLLM tensor_parallel_size"
    )
    device_ids = [
        _strict_nonnegative_integer(
            domain.get("device_id"),
            f"kapsl_memory_domains[{index}].device_id",
        )
        for index, domain in enumerate(memory_domains)
    ]
    if (
        len(device_ids) != tensor_parallel_size
        or len(set(device_ids)) != len(device_ids)
    ):
        raise ValueError(
            "vLLM shared_pool requires exactly one distinct CUDA domain per tensor-parallel rank"
        )
    rank_device_map = _rank_device_map(config)
    if rank_device_map is None:
        if tensor_parallel_size == 1:
            return None
        raise ValueError(
            "tensor-parallel shared_pool requires kapsl_rank_device_map"
        )
    expected_ranks = set(range(tensor_parallel_size))
    if set(rank_device_map) != expected_ranks:
        raise ValueError(
            "kapsl_rank_device_map must contain every tensor-parallel global rank exactly once"
        )
    if set(rank_device_map.values()) != set(device_ids):
        raise ValueError(
            "kapsl_rank_device_map devices must exactly match kapsl_memory_domains"
        )
    return rank_device_map
