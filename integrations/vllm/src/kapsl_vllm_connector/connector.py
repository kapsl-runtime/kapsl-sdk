"""Thin compatibility layer for vLLM's experimental V1 connector API."""

from __future__ import annotations

import logging
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .client import KapslKvControlClient, KapslKvControlError
from .contract import make_reserve_request, opaque_registration

if TYPE_CHECKING:
    import torch
    from vllm.forward_context import ForwardContext
    from vllm.v1.attention.backend import AttentionMetadata
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.core.kv_cache_interface import KVCacheConfig
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


@dataclass
class KapslConnectorMetadata(KVConnectorMetadata):
    """No worker transfer metadata is needed in opaque control-only mode."""


class KapslConnectorV1(KVConnectorBase_V1, SupportsHMA):
    """KV-connected, opaque vLLM participant.

    This connector lets Kapsl admit and lease logical KV capacity before vLLM
    allocates request blocks. It intentionally reports zero external prefix
    hits and performs no worker-side copies, so vLLM never consumes KV bytes
    that Kapsl cannot actually deliver.
    """

    @property
    def requires_kv_delivery(self) -> bool:
        return False

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
        self._participant_id = f"{participant_base}:{engine_id}:{role_name}"
        memory_domains = _required_memory_domains(self._kv_transfer_config)
        self._capacity_groups = _vllm_capacity_groups(kv_cache_config, memory_domains)
        self._group_ids = [group["group_id"] for group in self._capacity_groups]
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
        if self._is_scheduler:
            self._client = KapslKvControlClient(
                endpoint,
                self._participant_id,
                timeout_seconds=timeout_ms / 1000.0,
            )
            self._client.register(
                opaque_registration(
                    self._participant_id,
                    model_fingerprint,
                    self._capacity_groups,
                    backend="vllm",
                )
            )
            logger.info(
                "registered Kapsl KV participant %s as kv_connected/opaque",
                self._participant_id,
            )
            self._heartbeat_thread = threading.Thread(
                target=self._heartbeat_loop,
                name=f"kapsl-kv-heartbeat-{engine_id}",
                daemon=True,
            )
            self._heartbeat_thread.start()

    # Scheduler-side lifecycle -------------------------------------------------

    def on_new_request(self, request: "Request") -> None:
        if not self._is_scheduler:
            return
        self._raise_if_control_failed()
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
        return KapslConnectorMetadata()

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
        del forward_context, kwargs

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
        return None

    def shutdown(self) -> None:
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
        interval_seconds = max(0.25, self._lease_ttl_ms / 3000.0)
        while not self._heartbeat_stop.wait(interval_seconds):
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
            raise RuntimeError("KV control operations are scheduler-side only")
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


def _vllm_capacity_groups(
    kv_cache_config: Any,
    memory_domains: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Extract byte-accounting hints without exposing vLLM block handles."""

    groups = getattr(kv_cache_config, "kv_cache_groups", None)
    max_allocations = int(getattr(kv_cache_config, "num_blocks", 0) or 0)
    if not groups or max_allocations <= 0:
        raise ValueError(
            "vLLM KVCacheConfig must expose non-empty kv_cache_groups and num_blocks"
        )

    group_shapes: list[tuple[int, int]] = []
    for index, group in enumerate(groups):
        spec = getattr(group, "kv_cache_spec", None)
        layer_names = getattr(group, "layer_names", None)
        if spec is None or not layer_names:
            raise ValueError(f"vLLM KV cache group {index} has no spec or layers")
        block_size = int(getattr(spec, "block_size", 0) or 0)
        page_size = int(getattr(spec, "page_size_bytes", 0) or 0)
        # UniformTypeKVCacheSpecs.page_size_bytes already sums its per-layer
        # specs; an ordinary merged spec describes one layer and must be scaled
        # by the number of layers sharing the block table.
        group_bytes = (
            page_size
            if getattr(spec, "kv_cache_specs", None) is not None
            else page_size * len(layer_names)
        )
        if block_size <= 0 or group_bytes <= 0:
            raise ValueError(f"vLLM KV cache group {index} has invalid page accounting")
        group_shapes.append((block_size, group_bytes))

    # vLLM HMA groups alias one backing block pool; the allocator uses the
    # largest group as the physical block stride. A shared pool ID tells Kapsl
    # to charge the maximum reservation rather than summing every group.
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
