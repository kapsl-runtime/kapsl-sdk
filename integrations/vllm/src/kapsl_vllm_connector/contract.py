"""Pure-Python representation of the `kapsl-kv-abi` JSON contract.

Keep this module dependency-free: vLLM scheduler and worker processes both
import it, and contract tests should run on machines without CUDA or vLLM.
"""

from __future__ import annotations

from copy import deepcopy
from collections.abc import Sequence
import re
from typing import Any, Mapping

ABI_VERSION: dict[str, int] = {"major": 1, "minor": 5}


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


def shared_pool_capabilities(*, live_resize: bool = False) -> dict[str, Any]:
    """Kapsl-owned CUDA pool with block selection left to vLLM."""

    capabilities = {
        "abi_version": abi_version(),
        "tier": "shared_pool",
        "metadata_mode": "structured",
        "ownership": "kapsl_runtime",
        "features": [
            "capacity_leasing",
            "direct_attention_access",
            "external_pool_attachment",
            "participant_block_selection",
        ],
        "transports": [{"kind": "cuda_ipc"}],
    }
    if live_resize:
        capabilities["features"].append("live_pool_resize")
        capabilities["transports"] = [{"kind": "cuda_vmm"}]
    return capabilities


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
    profile: Mapping[str, Any],
    *,
    backend: str = "vllm",
    provisioning_grant: Mapping[str, Any] | None = None,
    live_resize: bool = False,
) -> dict[str, Any]:
    capabilities = shared_pool_capabilities(live_resize=live_resize)
    if len(capacity_groups) > 1:
        capabilities["features"].append("multiple_cache_groups")
    registration = {
        "participant_id": _nonempty(participant_id, "participant_id"),
        "backend": _nonempty(backend, "backend"),
        "model_fingerprint": _nonempty(model_fingerprint, "model_fingerprint"),
        "capabilities": capabilities,
        "capacity_model": {"groups": [dict(group) for group in capacity_groups]},
        "adapter_profile": dict(profile),
        "topology": deepcopy(dict(topology)),
    }
    if provisioning_grant is not None:
        capabilities["features"].append("provisioning_grant")
        registration["provisioning_grant"] = dict(provisioning_grant)
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
        if "adapter_profile" in registration:
            raise ContractValidationError(
                "opaque registrations cannot include an adapter_profile"
            )
    elif tier == "shared_pool":
        if capabilities.get("metadata_mode") != "structured":
            raise ContractValidationError("shared_pool vLLM requires structured metadata")
        if capabilities.get("ownership") != "kapsl_runtime":
            raise ContractValidationError("shared_pool KV must be Kapsl owned")
        required = {
            "direct_attention_access",
            "external_pool_attachment",
            "participant_block_selection",
        }
        if not required.issubset(features):
            raise ContractValidationError(
                "shared_pool requires direct attention, attachment, and participant block selection"
            )
        live_resize = "live_pool_resize" in features
        expected_transport = {"kind": "cuda_vmm" if live_resize else "cuda_ipc"}
        if expected_transport not in transports:
            raise ContractValidationError(
                "shared_pool vLLM transport does not match live-resize capabilities"
            )
        if live_resize and transports != [{"kind": "cuda_vmm"}]:
            raise ContractValidationError(
                "live-resize shared_pool vLLM requires only cuda_vmm transport"
            )
        _validate_adapter_profile(
            _mapping(registration.get("adapter_profile"), "adapter profile")
        )
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

    grant = registration.get("provisioning_grant")
    advertises_grant = "provisioning_grant" in features
    if advertises_grant != (grant is not None):
        raise ContractValidationError(
            "provisioning_grant capability and proof must be present together"
        )
    if grant is not None:
        if tier != "shared_pool" or capabilities.get("ownership") != "kapsl_runtime":
            raise ContractValidationError(
                "provisioning grants require a runtime-owned shared pool"
            )
        _validate_provisioning_grant(_mapping(grant, "provisioning grant"))


def _validate_provisioning_grant(grant: Mapping[str, Any]) -> None:
    token = _nonempty(grant.get("token"), "provisioning grant token")
    if len(token) > 256:
        raise ContractValidationError("provisioning grant token is too long")
    digest = _nonempty(
        grant.get("geometry_digest"), "provisioning grant geometry_digest"
    )
    if re.fullmatch(r"sha256:[0-9a-f]{64}", digest) is None:
        raise ContractValidationError(
            "provisioning grant geometry_digest must be canonical sha256"
        )
    _positive_int(
        grant.get("authority_generation"),
        "provisioning grant authority_generation",
    )
    _positive_int(
        grant.get("expires_at_unix_ms"),
        "provisioning grant expires_at_unix_ms",
    )


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
        element_type = _mapping(
            geometry.get("element_type"), "topology element_type"
        )
        element_kind = _nonempty(
            element_type.get("kind"), "topology element_type kind"
        )
        if element_kind == "custom":
            _nonempty(
                element_type.get("name"), "topology element_type custom name"
            )
        elif element_kind not in {"f16", "bf16", "f32", "i8", "fp8_e4m3"}:
            raise ContractValidationError(
                f"unsupported topology element_type kind {element_kind!r}"
            )
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
    live_resize: bool | None = None
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
        transport = _mapping(pool.get("transport"), "shared pool transport")
        elastic = pool.get("elastic")
        pool_is_live = transport.get("kind") == "cuda_vmm"
        if live_resize is None:
            live_resize = pool_is_live
        elif live_resize != pool_is_live:
            raise ContractValidationError(
                "shared pool receipts cannot mix fixed and elastic transports"
            )
        if pool_is_live:
            _validate_elastic_pool(
                _mapping(elastic, "elastic shared pool"),
                block_count=int(pool["block_count"]),
                bytes_per_block=int(pool["bytes_per_block"]),
            )
        elif elastic is not None:
            raise ContractValidationError(
                "elastic shared pool metadata requires cuda_vmm transport"
            )
    return deepcopy(dict(receipt))


def make_shared_pool_attachment(
    *,
    participant_epoch: int,
    binding_id: str,
    shard: Mapping[str, Any],
    profile: Mapping[str, Any],
    imported_bytes: int,
    mapped_bytes: int | None = None,
    views: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    attachment = {
        "participant_epoch": participant_epoch,
        "binding_id": binding_id,
        "shard": dict(shard),
        "profile": dict(profile),
        "imported_bytes": imported_bytes,
        "views": [dict(view) for view in views],
    }
    if mapped_bytes is not None:
        attachment["mapped_bytes"] = mapped_bytes
    validate_shared_pool_attachment(attachment)
    return attachment


def validate_shared_pool_attachment(attachment: Mapping[str, Any]) -> None:
    _positive_int(attachment.get("participant_epoch"), "participant_epoch")
    _nonempty(attachment.get("binding_id"), "binding_id")
    _positive_int(attachment.get("imported_bytes"), "imported_bytes")
    mapped_bytes = attachment.get("mapped_bytes")
    if mapped_bytes is not None:
        mapped_bytes = _positive_int(mapped_bytes, "mapped_bytes")
        if mapped_bytes > int(attachment["imported_bytes"]):
            raise ContractValidationError("mapped_bytes exceeds imported_bytes")
    shard = _mapping(attachment.get("shard"), "attachment shard")
    tp_rank = _nonnegative_int(
        shard.get("tensor_parallel_rank"), "tensor_parallel_rank"
    )
    tp_world = _positive_int(
        shard.get("tensor_parallel_world_size"), "tensor_parallel_world_size"
    )
    pp_rank = _nonnegative_int(
        shard.get("pipeline_parallel_rank"), "pipeline_parallel_rank"
    )
    pp_world = _positive_int(
        shard.get("pipeline_parallel_world_size"), "pipeline_parallel_world_size"
    )
    if tp_rank >= tp_world or pp_rank >= pp_world:
        raise ContractValidationError("attachment shard rank is outside its world size")
    _validate_adapter_profile(
        _mapping(attachment.get("profile"), "adapter profile")
    )
    views = attachment.get("views")
    if not isinstance(views, list) or not views:
        raise ContractValidationError("attachment views must be a non-empty list")
    layers: set[tuple[str, int]] = set()
    imported_bytes = int(attachment["imported_bytes"])
    for raw_view in views:
        view = _mapping(raw_view, "attachment view")
        group_id = _nonempty(view.get("group_id"), "attachment view group_id")
        layer = _mapping(view.get("layer"), "attachment view layer")
        layer_index = _nonnegative_int(layer.get("index"), "attachment layer index")
        if layer.get("name") is not None:
            _nonempty(layer["name"], "attachment layer name")
        if (group_id, layer_index) in layers:
            raise ContractValidationError(
                "attachment group/layer pairs must be unique"
            )
        layers.add((group_id, layer_index))
        offset = _nonnegative_int(view.get("offset_bytes"), "attachment offset")
        length = _positive_int(view.get("length_bytes"), "attachment length")
        if offset + length > imported_bytes:
            raise ContractValidationError("attachment view exceeds imported bytes")


def _validate_adapter_profile(profile: Mapping[str, Any]) -> None:
    for field in (
        "adapter_id",
        "adapter_version",
        "backend_version",
        "profile_id",
    ):
        _nonempty(profile.get(field), f"adapter profile {field}")


def make_shared_pool_detach_request(
    *,
    participant_epoch: int,
    binding_ids: Sequence[str],
    shard: Mapping[str, Any],
) -> dict[str, Any]:
    request = {
        "participant_epoch": participant_epoch,
        "binding_ids": list(binding_ids),
        "shard": dict(shard),
        "completion": {"kind": "backend_synchronized"},
    }
    validate_shared_pool_detach_request(request)
    return request


def validate_shared_pool_detach_request(request: Mapping[str, Any]) -> None:
    _positive_int(request.get("participant_epoch"), "participant_epoch")
    binding_ids = request.get("binding_ids")
    if not isinstance(binding_ids, list) or not binding_ids:
        raise ContractValidationError("detach binding_ids must be a non-empty list")
    normalized = [_nonempty(value, "detach binding_id") for value in binding_ids]
    if len(set(normalized)) != len(normalized):
        raise ContractValidationError("detach binding_ids must be unique")
    shard = _mapping(request.get("shard"), "detach shard")
    tp_rank = _nonnegative_int(
        shard.get("tensor_parallel_rank"), "tensor_parallel_rank"
    )
    tp_world = _positive_int(
        shard.get("tensor_parallel_world_size"), "tensor_parallel_world_size"
    )
    pp_rank = _nonnegative_int(
        shard.get("pipeline_parallel_rank"), "pipeline_parallel_rank"
    )
    pp_world = _positive_int(
        shard.get("pipeline_parallel_world_size"), "pipeline_parallel_world_size"
    )
    if tp_rank >= tp_world or pp_rank >= pp_world:
        raise ContractValidationError("detach shard rank is outside its world size")
    completion = _mapping(request.get("completion"), "detach completion")
    if completion.get("kind") != "backend_synchronized":
        raise ContractValidationError(
            "initial shared-pool detach requires backend_synchronized completion"
        )


def make_resize_poll_request(
    *, participant_epoch: int, actor: Mapping[str, Any], applied_generation: int
) -> dict[str, Any]:
    request = {
        "participant_epoch": participant_epoch,
        "actor": dict(actor),
        "applied_generation": applied_generation,
    }
    validate_resize_poll_request(request)
    return request


def validate_resize_poll_request(request: Mapping[str, Any]) -> None:
    _positive_int(request.get("participant_epoch"), "participant_epoch")
    _validate_resize_actor(_mapping(request.get("actor"), "resize actor"))
    _nonnegative_int(request.get("applied_generation"), "applied_generation")


def make_resize_ack_request(
    *,
    participant_epoch: int,
    actor: Mapping[str, Any],
    binding_id: str,
    resize_generation: int,
    stage: str,
    applied_block_count: int,
) -> dict[str, Any]:
    request = {
        "participant_epoch": participant_epoch,
        "actor": dict(actor),
        "binding_id": binding_id,
        "resize_generation": resize_generation,
        "stage": stage,
        "applied_block_count": applied_block_count,
    }
    validate_resize_ack_request(request)
    return request


def validate_resize_ack_request(request: Mapping[str, Any]) -> None:
    _positive_int(request.get("participant_epoch"), "participant_epoch")
    _validate_resize_actor(_mapping(request.get("actor"), "resize actor"))
    _nonempty(request.get("binding_id"), "binding_id")
    _positive_int(request.get("resize_generation"), "resize_generation")
    _validate_resize_stage(request.get("stage"))
    _positive_int(request.get("applied_block_count"), "applied_block_count")


def validate_resize_operation(operation: Mapping[str, Any]) -> dict[str, Any]:
    _positive_int(operation.get("participant_epoch"), "participant_epoch")
    _positive_int(operation.get("resize_generation"), "resize_generation")
    _nonempty(operation.get("binding_id"), "binding_id")
    stage = _validate_resize_stage(operation.get("stage"))
    from_blocks = _positive_int(
        operation.get("from_block_count"), "from_block_count"
    )
    target_blocks = _positive_int(
        operation.get("target_block_count"), "target_block_count"
    )
    if from_blocks == target_blocks:
        raise ContractValidationError("resize block counts must differ")
    growing = target_blocks > from_blocks
    if (growing and stage not in {"map_workers", "activate_scheduler"}) or (
        not growing and stage not in {"retire_scheduler", "unmap_workers"}
    ):
        raise ContractValidationError("resize stage does not match direction")
    stride = _positive_int(operation.get("bytes_per_block"), "bytes_per_block")
    granularity = _positive_int(
        operation.get("allocation_granularity_bytes"),
        "allocation_granularity_bytes",
    )
    start = min(from_blocks, target_blocks) * stride
    end = max(from_blocks, target_blocks) * stride
    if start % granularity or end % granularity:
        raise ContractValidationError("resize endpoints are not VMM aligned")
    segments = operation.get("segments", [])
    if not isinstance(segments, list):
        raise ContractValidationError("resize segments must be a list")
    physical = stage in {"map_workers", "unmap_workers"}
    if physical != bool(segments):
        raise ContractValidationError(
            "worker resize phases require segments and scheduler phases forbid them"
        )
    expected_offset = start
    seen_ids: set[str] = set()
    seen_handles: set[int] = set()
    for raw_segment in sorted(
        segments,
        key=lambda raw: _nonnegative_int(
            _mapping(raw, "resize segment").get("offset_bytes"), "offset_bytes"
        ),
    ):
        segment = _mapping(raw_segment, "resize segment")
        segment_id = _nonempty(segment.get("segment_id"), "segment_id")
        offset = _nonnegative_int(segment.get("offset_bytes"), "offset_bytes")
        length = _positive_int(segment.get("length_bytes"), "length_bytes")
        handle_index = _nonnegative_int(
            segment.get("handle_index"), "handle_index"
        )
        if (
            segment_id in seen_ids
            or handle_index in seen_handles
            or offset != expected_offset
            or offset % granularity
            or length % granularity
        ):
            raise ContractValidationError(
                "resize segments must uniquely and densely cover the changed tail"
            )
        seen_ids.add(segment_id)
        seen_handles.add(handle_index)
        expected_offset += length
    if physical and expected_offset != end:
        raise ContractValidationError(
            "resize segments do not exactly cover the changed tail"
        )
    return deepcopy(dict(operation))


def _validate_resize_actor(actor: Mapping[str, Any]) -> None:
    role = actor.get("role")
    if role == "scheduler":
        if set(actor) != {"role"}:
            raise ContractValidationError("scheduler resize actor has unknown fields")
        return
    if role != "worker":
        raise ContractValidationError("resize actor role is unsupported")
    shard = _mapping(actor.get("shard"), "resize worker shard")
    tp_rank = _nonnegative_int(
        shard.get("tensor_parallel_rank"), "tensor_parallel_rank"
    )
    tp_world = _positive_int(
        shard.get("tensor_parallel_world_size"), "tensor_parallel_world_size"
    )
    pp_rank = _nonnegative_int(
        shard.get("pipeline_parallel_rank"), "pipeline_parallel_rank"
    )
    pp_world = _positive_int(
        shard.get("pipeline_parallel_world_size"), "pipeline_parallel_world_size"
    )
    if tp_rank >= tp_world or pp_rank >= pp_world:
        raise ContractValidationError("resize worker shard rank is outside its world size")


def _validate_resize_stage(value: Any) -> str:
    stage = _nonempty(value, "resize stage")
    if stage not in {
        "map_workers",
        "activate_scheduler",
        "retire_scheduler",
        "unmap_workers",
    }:
        raise ContractValidationError("resize stage is unsupported")
    return stage


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
    if response.get("result") not in {
        "registered",
        "lease",
        "resize",
        "ack",
        "error",
    }:
        raise ContractValidationError("unknown control response result")


def _validate_elastic_pool(
    elastic: Mapping[str, Any], *, block_count: int, bytes_per_block: int
) -> None:
    mapped_blocks = _positive_int(
        elastic.get("mapped_block_count"), "mapped_block_count"
    )
    maximum_blocks = _positive_int(
        elastic.get("maximum_block_count"), "maximum_block_count"
    )
    granularity = _positive_int(
        elastic.get("allocation_granularity_bytes"),
        "allocation_granularity_bytes",
    )
    alignment = _positive_int(
        elastic.get("resize_alignment_blocks"), "resize_alignment_blocks"
    )
    if maximum_blocks != block_count or mapped_blocks > maximum_blocks:
        raise ContractValidationError(
            "elastic block counts must fit and match the shared pool maximum"
        )
    mapped_bytes = mapped_blocks * bytes_per_block
    maximum_bytes = maximum_blocks * bytes_per_block
    if (
        mapped_blocks % alignment
        or mapped_bytes % granularity
        or maximum_bytes % granularity
    ):
        raise ContractValidationError("elastic pool geometry is not VMM aligned")
    segments = elastic.get("segments")
    if not isinstance(segments, list) or not segments:
        raise ContractValidationError("elastic pool requires mapped segments")
    expected_offset = 0
    seen_ids: set[str] = set()
    seen_handles: set[int] = set()
    ordered = sorted(
        (_mapping(segment, "VMM segment") for segment in segments),
        key=lambda segment: _nonnegative_int(
            segment.get("offset_bytes"), "VMM segment offset_bytes"
        ),
    )
    for segment in ordered:
        segment_id = _nonempty(segment.get("segment_id"), "VMM segment_id")
        offset = _nonnegative_int(segment.get("offset_bytes"), "offset_bytes")
        length = _positive_int(segment.get("length_bytes"), "length_bytes")
        handle_index = _nonnegative_int(
            segment.get("handle_index"), "handle_index"
        )
        if (
            segment_id in seen_ids
            or handle_index in seen_handles
            or offset != expected_offset
            or offset % granularity
            or length % granularity
        ):
            raise ContractValidationError(
                "VMM segments must uniquely and densely cover an aligned prefix"
            )
        seen_ids.add(segment_id)
        seen_handles.add(handle_index)
        expected_offset += length
    if expected_offset != mapped_bytes:
        raise ContractValidationError(
            "VMM segments do not exactly cover mapped physical bytes"
        )


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
