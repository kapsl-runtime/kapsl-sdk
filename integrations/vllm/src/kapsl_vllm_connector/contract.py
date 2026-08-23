"""Pure-Python representation of the `kapsl-kv-abi` JSON contract.

Keep this module dependency-free: vLLM scheduler and worker processes both
import it, and contract tests should run on machines without CUDA or vLLM.
"""

from __future__ import annotations

from copy import deepcopy
from collections.abc import Sequence
from typing import Any, Mapping

ABI_VERSION: dict[str, int] = {"major": 1, "minor": 0}


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


def validate_registration(registration: Mapping[str, Any]) -> None:
    _nonempty(registration.get("participant_id"), "participant_id")
    _nonempty(registration.get("backend"), "backend")
    _nonempty(registration.get("model_fingerprint"), "model_fingerprint")
    capabilities = _mapping(registration.get("capabilities"), "capabilities")
    if not accepts_version(_mapping(capabilities.get("abi_version"), "abi_version")):
        raise ContractValidationError("unsupported capability ABI version")
    if capabilities.get("tier") != "kv_connected":
        raise ContractValidationError("vLLM connector must advertise kv_connected")
    if capabilities.get("metadata_mode") != "opaque":
        raise ContractValidationError("vLLM connector must advertise opaque metadata")
    if capabilities.get("ownership") != "backend":
        raise ContractValidationError("opaque vLLM KV must remain backend owned")
    if "capacity_leasing" not in capabilities.get("features", []):
        raise ContractValidationError("capacity_leasing feature is required")
    if {"kind": "backend_opaque"} not in capabilities.get("transports", []):
        raise ContractValidationError("backend_opaque transport is required")
    if "topology" in registration:
        raise ContractValidationError("opaque registrations cannot include a topology")
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
    return deepcopy(dict(lease))


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
