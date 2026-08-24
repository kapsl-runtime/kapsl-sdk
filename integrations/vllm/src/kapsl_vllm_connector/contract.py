"""Pure-Python representation of the `kapsl-kv-abi` JSON contract.

Keep this module dependency-free: vLLM scheduler and worker processes both
import it, and contract tests should run on machines without CUDA or vLLM.
"""

from __future__ import annotations

from copy import deepcopy
from collections.abc import Sequence
from typing import Any, Mapping

ABI_VERSION: dict[str, int] = {"major": 1, "minor": 2}


class ContractValidationError(ValueError):
    """A local or remote message violates the negotiated KV contract."""


def abi_version() -> dict[str, int]:
    return dict(ABI_VERSION)


def accepts_version(peer: Mapping[str, Any]) -> bool:
    """Whether this connector can consume a message from `peer`."""

    return (
        peer.get("major") == ABI_VERSION["major"]
        and isinstance(peer.get("minor"), int)
        and peer["minor"] <= ABI_VERSION["minor"]
    )


def opaque_capabilities() -> dict[str, Any]:
    """Minimum official Kapsl tier with a backend-private block layout."""

    return {
        "abi_version": abi_version(),
        "tier": "kv_connected",
        "metadata_mode": "opaque",
        "ownership": "backend",
        "features": ["capacity_leasing"],
        "transports": [{"kind": "backend_opaque"}],
    }


def shared_pool_capabilities() -> dict[str, Any]:
    """Kapsl-owned CUDA pool with block selection left to vLLM."""

    return {
        "abi_version": abi_version(),
        "tier": "shared_pool",
        "metadata_mode": "structured",
        "ownership": "kapsl_runtime",
        "features": [
            "capacity_leasing",
            "direct_attention_access",
            "participant_block_selection",
        ],
        "transports": [{"kind": "cuda_ipc"}],
    }


def opaque_registration(
    participant_id: str,
    model_fingerprint: str,
    capacity_groups: Sequence[Mapping[str, Any]],
    *,
    backend: str = "vllm",
) -> dict[str, Any]:
    participant_id = _nonempty(participant_id, "participant_id")
    registration = {
        "participant_id": participant_id,
        "backend": _nonempty(backend, "backend"),
        "model_fingerprint": _nonempty(model_fingerprint, "model_fingerprint"),
        "capabilities": opaque_capabilities(),
        "capacity_model": {"groups": [dict(group) for group in capacity_groups]},
    }
    validate_registration(registration)
    return registration


def shared_pool_registration(
    participant_id: str,
    model_fingerprint: str,
    capacity_groups: Sequence[Mapping[str, Any]],
    topology: Mapping[str, Any],
    *,
    backend: str = "vllm",
) -> dict[str, Any]:
    capabilities = shared_pool_capabilities()
    if len(capacity_groups) > 1:
        capabilities["features"].append("multiple_cache_groups")
    registration = {
        "participant_id": _nonempty(participant_id, "participant_id"),
        "backend": _nonempty(backend, "backend"),
        "model_fingerprint": _nonempty(model_fingerprint, "model_fingerprint"),
        "capabilities": capabilities,
        "capacity_model": {"groups": [dict(group) for group in capacity_groups]},
        "topology": deepcopy(dict(topology)),
    }
    validate_registration(registration)
    return registration


def validate_registration(registration: Mapping[str, Any]) -> None:
    _nonempty(registration.get("participant_id"), "participant_id")
    _nonempty(registration.get("backend"), "backend")
    model_fingerprint = _nonempty(
        registration.get("model_fingerprint"), "model_fingerprint"
    )
    capabilities = _mapping(registration.get("capabilities"), "capabilities")
    if not accepts_version(_mapping(capabilities.get("abi_version"), "abi_version")):
        raise ContractValidationError("unsupported capability ABI version")
    features = capabilities.get("features")
    transports = capabilities.get("transports")
    if not isinstance(features, list) or not isinstance(transports, list):
        raise ContractValidationError("capability features and transports must be lists")
    if "capacity_leasing" not in features:
        raise ContractValidationError("capacity_leasing feature is required")
    tier = capabilities.get("tier")
    if tier == "kv_connected":
        if capabilities.get("metadata_mode") != "opaque":
            raise ContractValidationError("kv_connected vLLM must use opaque metadata")
        if capabilities.get("ownership") != "backend":
            raise ContractValidationError("opaque vLLM KV must remain backend owned")
        if {"kind": "backend_opaque"} not in transports:
            raise ContractValidationError("backend_opaque transport is required")
        if "topology" in registration:
            raise ContractValidationError("opaque registrations cannot include a topology")
    elif tier == "shared_pool":
        if capabilities.get("metadata_mode") != "structured":
            raise ContractValidationError("shared_pool vLLM requires structured metadata")
        if capabilities.get("ownership") != "kapsl_runtime":
            raise ContractValidationError("shared_pool KV must be Kapsl owned")
        required = {"direct_attention_access", "participant_block_selection"}
        if not required.issubset(features):
            raise ContractValidationError(
                "shared_pool requires direct attention and participant block selection"
            )
        if {"kind": "cuda_ipc"} not in transports:
            raise ContractValidationError("shared_pool vLLM requires cuda_ipc")
    else:
        raise ContractValidationError("vLLM connector has an unsupported KV tier")

    capacity_model = _mapping(registration.get("capacity_model"), "capacity_model")
    groups = capacity_model.get("groups")
    if not isinstance(groups, list) or not groups:
        raise ContractValidationError("capacity_model.groups must be a non-empty list")
    seen: set[str] = set()
    pool_domains: dict[str, set[tuple[Any, ...]]] = {}
    for raw_group in groups:
        group = _mapping(raw_group, "capacity group")
        group_id = _nonempty(group.get("group_id"), "capacity group_id")
        pool_id = _nonempty(group.get("pool_id"), "capacity pool_id")
        if group_id in seen:
            raise ContractValidationError("capacity group IDs must be unique")
        seen.add(group_id)
        _positive_int(
            group.get("allocation_granularity_tokens"),
            "allocation_granularity_tokens",
        )
        _positive_int(group.get("bytes_per_allocation"), "bytes_per_allocation")
        memory_domains = group.get("memory_domains")
        if not isinstance(memory_domains, list) or not memory_domains:
            raise ContractValidationError(
                "capacity group memory_domains must be a non-empty list"
            )
        seen_domains: set[tuple[Any, ...]] = set()
        for raw_domain in memory_domains:
            domain = _mapping(raw_domain, "memory domain")
            kind = _nonempty(domain.get("kind"), "memory domain kind")
            if kind == "host":
                identity = (kind,)
            elif kind == "cuda":
                identity = (
                    kind,
                    _nonnegative_int(domain.get("device_id"), "device_id"),
                )
            elif kind in {"host_pinned", "host_mapped", "provider"}:
                provider = _nonempty(domain.get("provider"), "memory domain provider")
                device_id = domain.get("device_id")
                if device_id is not None:
                    device_id = _nonnegative_int(device_id, "device_id")
                identity = (kind, provider, device_id)
            else:
                raise ContractValidationError(f"unknown KV memory domain kind {kind!r}")
            if identity in seen_domains:
                raise ContractValidationError("capacity memory domains must be unique")
            seen_domains.add(identity)
        previous_domains = pool_domains.setdefault(pool_id, seen_domains)
        if previous_domains != seen_domains:
            raise ContractValidationError(
                "capacity groups sharing a pool must name the same memory domains"
            )
        if group.get("max_allocations") is not None:
            _positive_int(group["max_allocations"], "max_allocations")
        elif tier == "shared_pool":
            raise ContractValidationError("shared_pool groups require max_allocations")

    if tier == "shared_pool":
        topology = _mapping(registration.get("topology"), "topology")
        _validate_topology(topology, model_fingerprint, seen, capabilities)


def _validate_topology(
    topology: Mapping[str, Any],
    model_fingerprint: str,
    capacity_group_ids: set[str],
    capabilities: Mapping[str, Any],
) -> None:
    topology_version = _mapping(topology.get("abi_version"), "topology abi_version")
    if not accepts_version(topology_version):
        raise ContractValidationError("unsupported topology ABI version")
    if dict(topology_version) != dict(
        _mapping(capabilities.get("abi_version"), "capability ABI version")
    ):
        raise ContractValidationError("topology and capability ABI versions must match")
    if topology.get("model_fingerprint") != model_fingerprint:
        raise ContractValidationError("topology model fingerprint must match registration")
    shard = _mapping(topology.get("shard", {}), "topology shard")
    tp_world = _positive_int(
        shard.get("tensor_parallel_world_size", 1), "tensor_parallel_world_size"
    )
    pp_world = _positive_int(
        shard.get("pipeline_parallel_world_size", 1), "pipeline_parallel_world_size"
    )
    tp_rank = _nonnegative_int(
        shard.get("tensor_parallel_rank", 0), "tensor_parallel_rank"
    )
    pp_rank = _nonnegative_int(
        shard.get("pipeline_parallel_rank", 0), "pipeline_parallel_rank"
    )
    if tp_rank >= tp_world or pp_rank >= pp_world:
        raise ContractValidationError("topology shard rank is outside its world size")

    cache_groups = topology.get("cache_groups")
    if not isinstance(cache_groups, list) or not cache_groups:
        raise ContractValidationError("topology cache_groups must be non-empty")
    if len(cache_groups) > 1 and "multiple_cache_groups" not in capabilities.get(
        "features", []
    ):
        raise ContractValidationError("multiple topology groups require a capability")
    topology_ids: set[str] = set()
    for raw_group in cache_groups:
        group = _mapping(raw_group, "topology cache group")
        group_id = _nonempty(group.get("group_id"), "topology group_id")
        if group_id in topology_ids:
            raise ContractValidationError("topology cache group IDs must be unique")
        topology_ids.add(group_id)
        layers = group.get("layers")
        if not isinstance(layers, list) or not layers:
            raise ContractValidationError("topology cache group layers must be non-empty")
        layer_indices: set[int] = set()
        for raw_layer in layers:
            layer = _mapping(raw_layer, "topology layer")
            index = _nonnegative_int(layer.get("index"), "topology layer index")
            if index in layer_indices:
                raise ContractValidationError("topology layer indices must be unique")
            layer_indices.add(index)
            if layer.get("name") is not None:
                _nonempty(layer["name"], "topology layer name")
        geometry = _mapping(group.get("geometry"), "topology geometry")
        if geometry.get("kind") != "paged_attention":
            raise ContractValidationError(
                "initial vLLM shared_pool support requires paged_attention geometry"
            )
        for field in (
            "block_size_tokens",
            "kv_heads",
            "key_head_dim",
            "value_head_dim",
        ):
            _positive_int(geometry.get(field), f"topology geometry {field}")
        element_type = geometry.get("element_type")
        if not isinstance(element_type, (str, Mapping)):
            raise ContractValidationError("topology element_type is invalid")
        layout = _mapping(geometry.get("layout"), "topology layout")
        _nonempty(layout.get("kind"), "topology layout kind")
        policy = _mapping(group.get("policy"), "topology policy")
        if policy.get("kind") not in {"full_attention", "sliding_window"}:
            raise ContractValidationError("unsupported vLLM cache policy")
        if policy.get("kind") == "sliding_window":
            _positive_int(policy.get("window_tokens"), "sliding window tokens")
    if topology_ids != capacity_group_ids:
        raise ContractValidationError(
            "topology and capacity model must describe the same groups"
        )


def make_reserve_request(
    *,
    request_id: str,
    sequence_id: str,
    token_capacity: int,
    group_ids: Sequence[str] = ("vllm.default",),
    priority: int = 0,
    ttl_ms: int | None = None,
) -> dict[str, Any]:
    request = {
        "sequence": {
            "request_id": _nonempty(request_id, "request_id"),
            "sequence_id": _nonempty(sequence_id, "sequence_id"),
        },
        "groups": [
            {
                "group_id": _nonempty(group_id, "group_id"),
                "token_capacity": _positive_int(token_capacity, "token_capacity"),
            }
            for group_id in group_ids
        ],
        "priority": int(priority),
    }
    if ttl_ms is not None:
        request["ttl_ms"] = _positive_int(ttl_ms, "ttl_ms")
    validate_reserve_request(request)
    return request


def validate_reserve_request(request: Mapping[str, Any]) -> None:
    sequence = _mapping(request.get("sequence"), "sequence")
    _nonempty(sequence.get("request_id"), "sequence.request_id")
    _nonempty(sequence.get("sequence_id"), "sequence.sequence_id")
    groups = request.get("groups")
    if not isinstance(groups, list) or not groups:
        raise ContractValidationError("groups must be a non-empty list")
    seen: set[str] = set()
    for raw_group in groups:
        group = _mapping(raw_group, "group")
        group_id = _nonempty(group.get("group_id"), "group_id")
        if group_id in seen:
            raise ContractValidationError("group IDs must be unique")
        seen.add(group_id)
        _positive_int(group.get("token_capacity"), "token_capacity")
    if request.get("ttl_ms") is not None:
        _positive_int(request["ttl_ms"], "ttl_ms")


def validate_lease(lease: Mapping[str, Any]) -> dict[str, Any]:
    _nonempty(lease.get("lease_id"), "lease_id")
    sequence = _mapping(lease.get("sequence"), "sequence")
    _nonempty(sequence.get("request_id"), "sequence.request_id")
    _nonempty(sequence.get("sequence_id"), "sequence.sequence_id")
    groups = lease.get("groups")
    if not isinstance(groups, list) or not groups:
        raise ContractValidationError("lease groups must be a non-empty list")
    for raw_group in groups:
        group = _mapping(raw_group, "lease group")
        _nonempty(group.get("group_id"), "group_id")
        _positive_int(group.get("token_capacity"), "token_capacity")
        blocks = group.get("blocks", [])
        if not isinstance(blocks, list):
            raise ContractValidationError("lease blocks must be a list")
        for raw_block in blocks:
            block = _mapping(raw_block, "lease block")
            if block.get("kind") == "runtime_pool":
                _nonempty(block.get("pool_id"), "lease block pool_id")
                _nonnegative_int(block.get("block_index"), "block_index")
                _positive_int(block.get("generation"), "generation")
            elif block.get("kind") in {"backend_opaque", "transport"}:
                # Opaque/transfer handles are interpreted by their negotiated
                # backend transport; their complete Rust-side validation still
                # runs at the coordinator boundary.
                continue
            else:
                raise ContractValidationError("unknown lease block handle kind")
    return deepcopy(dict(lease))


def validate_registration_receipt(
    receipt: Mapping[str, Any], participant_id: str
) -> dict[str, Any]:
    if receipt.get("participant_id") != _nonempty(participant_id, "participant_id"):
        raise ContractValidationError(
            "registration receipt participant_id does not match registration"
        )
    _positive_int(receipt.get("participant_epoch"), "participant_epoch")
    pools = receipt.get("shared_pools", [])
    if not isinstance(pools, list):
        raise ContractValidationError("shared_pools must be a list")
    for raw_pool in pools:
        pool = _mapping(raw_pool, "shared pool")
        _nonempty(pool.get("binding_id"), "binding_id")
        _nonempty(pool.get("capacity_pool_id"), "capacity_pool_id")
        _positive_int(pool.get("generation"), "generation")
        _positive_int(pool.get("block_count"), "block_count")
        _positive_int(pool.get("bytes_per_block"), "bytes_per_block")
        if pool.get("allocation_mode", "runtime_leased") not in {
            "runtime_leased",
            "participant_managed",
        }:
            raise ContractValidationError("unknown shared pool allocation_mode")
        _nonempty(pool.get("descriptor"), "descriptor")
        group_ids = pool.get("group_ids")
        if not isinstance(group_ids, list) or not group_ids:
            raise ContractValidationError("shared pool group_ids must be non-empty")
        for group_id in group_ids:
            _nonempty(group_id, "shared pool group_id")
        _mapping(pool.get("memory_domain"), "shared pool memory_domain")
        _mapping(pool.get("transport"), "shared pool transport")
    return deepcopy(dict(receipt))


def make_envelope(request_id: str, operation: str, **payload: Any) -> dict[str, Any]:
    envelope = {
        "abi_version": abi_version(),
        "request_id": _nonempty(request_id, "request_id"),
        "operation": _nonempty(operation, "operation"),
    }
    envelope.update(payload)
    return envelope


def validate_response(response: Mapping[str, Any], request_id: str) -> None:
    if not accepts_version(_mapping(response.get("abi_version"), "abi_version")):
        raise ContractValidationError("unsupported response ABI version")
    if response.get("request_id") != request_id:
        raise ContractValidationError("response request_id does not match request")
    if response.get("result") not in {"registered", "lease", "ack", "error"}:
        raise ContractValidationError("unknown control response result")


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ContractValidationError(f"{field} must be an object")
    return value


def _nonempty(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ContractValidationError(f"{field} must be a non-empty string")
    return value


def _positive_int(value: Any, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ContractValidationError(f"{field} must be a positive integer")
    return value


def _nonnegative_int(value: Any, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ContractValidationError(f"{field} must be a non-negative integer")
    return value
