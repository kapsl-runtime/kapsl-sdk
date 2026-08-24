"""Fail-closed certification records for Kapsl-owned vLLM KV pools.

This module intentionally has no torch or vLLM dependency.  Host CI can check
that a hardware probe cannot emit a runtime allowlist entry from incomplete or
hand-waved evidence.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


CERTIFICATION_SCHEMA_VERSION = 1
REQUIRED_GATES = (
    "contract",
    "allocator_attachment",
    "backend_native_write",
    "backend_native_read",
    "lifecycle",
    "parallel_coverage",
)


class CertificationError(ValueError):
    """A probe report is not sufficient to authorize ``shared_pool``."""


def validate_certification_report(report: Mapping[str, Any]) -> None:
    """Validate a complete, passing hardware certification report.

    This is deliberately stricter than JSON-shape validation: every required
    gate must pass, every declared tensor-parallel rank must be represented,
    and every rank must name a distinct physical binding.
    """

    if report.get("schema_version") != CERTIFICATION_SCHEMA_VERSION:
        raise CertificationError("unsupported certification schema version")
    if report.get("status") != "passed":
        raise CertificationError("certification status is not passed")

    profile = _mapping(report.get("profile"), "profile")
    for field in (
        "adapter_id",
        "adapter_version",
        "backend_version",
        "profile_id",
    ):
        _safe_field(profile.get(field), f"profile.{field}")

    environment = _mapping(report.get("environment"), "environment")
    for field in (
        "adapter_build_id",
        "backend_build_id",
        "runtime_build_id",
    ):
        _sha256_id(environment.get(field), f"environment.{field}")
    for field in (
        "torch_version",
        "cuda_runtime_version",
        "cuda_driver_version",
    ):
        _safe_field(environment.get(field), f"environment.{field}")

    matrix = _mapping(report.get("matrix"), "matrix")
    world_size = _positive_int(
        matrix.get("tensor_parallel_world_size"),
        "matrix.tensor_parallel_world_size",
    )
    for field in ("attention_backend", "kv_layout", "dtype"):
        _safe_field(matrix.get(field), f"matrix.{field}")
    geometry = _mapping(matrix.get("cache_geometry"), "matrix.cache_geometry")
    for field in (
        "num_blocks",
        "block_size",
        "num_kv_heads",
        "num_query_heads",
        "head_size",
        "dense_page_bytes",
        "guard_bytes_per_block",
        "padded_page_bytes",
        "allocation_bytes",
    ):
        _positive_int(geometry.get(field), f"matrix.cache_geometry.{field}")
    if geometry["padded_page_bytes"] != (
        geometry["dense_page_bytes"] + geometry["guard_bytes_per_block"]
    ):
        raise CertificationError("cache page and guard byte accounting is inconsistent")
    if geometry["allocation_bytes"] != (
        geometry["num_blocks"] * geometry["padded_page_bytes"]
    ):
        raise CertificationError("cache allocation byte accounting is inconsistent")

    devices = matrix.get("devices")
    if not isinstance(devices, Sequence) or isinstance(devices, (str, bytes)):
        raise CertificationError("matrix.devices must be an array")
    if len(devices) != world_size:
        raise CertificationError("matrix.devices must cover the declared world")
    normalized_devices = [
        _nonnegative_int(device, f"matrix.devices[{index}]")
        for index, device in enumerate(devices)
    ]
    if len(set(normalized_devices)) != world_size:
        raise CertificationError("matrix devices must be distinct")

    gates = _mapping(report.get("gates"), "gates")
    missing_gates = [name for name in REQUIRED_GATES if name not in gates]
    if missing_gates:
        raise CertificationError(
            "certification is missing required gates: " + ", ".join(missing_gates)
        )
    for name in REQUIRED_GATES:
        gate = _mapping(gates[name], f"gates.{name}")
        if gate.get("passed") is not True:
            raise CertificationError(f"certification gate {name!r} did not pass")
        _safe_field(gate.get("evidence"), f"gates.{name}.evidence")

    ranks = report.get("ranks")
    if not isinstance(ranks, Sequence) or isinstance(ranks, (str, bytes)):
        raise CertificationError("ranks must be an array")
    if len(ranks) != world_size:
        raise CertificationError(
            f"expected {world_size} rank reports, received {len(ranks)}"
        )

    seen_ranks: set[int] = set()
    seen_bindings: set[str] = set()
    seen_devices: set[int] = set()
    for index, raw_rank in enumerate(ranks):
        rank = _mapping(raw_rank, f"ranks[{index}]")
        rank_index = _nonnegative_int(rank.get("rank"), f"ranks[{index}].rank")
        if rank_index >= world_size or rank_index in seen_ranks:
            raise CertificationError("rank coverage is duplicated or out of range")
        seen_ranks.add(rank_index)
        binding_id = _safe_field(
            rank.get("binding_id"), f"ranks[{index}].binding_id"
        )
        if binding_id in seen_bindings:
            raise CertificationError("physical binding IDs must be unique per rank")
        seen_bindings.add(binding_id)
        device_id = _nonnegative_int(
            rank.get("device_id"), f"ranks[{index}].device_id"
        )
        if device_id in seen_devices or device_id not in normalized_devices:
            raise CertificationError("rank device coverage is duplicated or unexpected")
        seen_devices.add(device_id)
        if rank.get("passed") is not True:
            raise CertificationError(f"rank {rank_index} did not pass")
        device = _mapping(rank.get("device"), f"ranks[{index}].device")
        _safe_field(device.get("name"), f"ranks[{index}].device.name")
        _safe_field(
            device.get("compute_capability"),
            f"ranks[{index}].device.compute_capability",
        )
        _positive_int(
            device.get("total_memory_bytes"),
            f"ranks[{index}].device.total_memory_bytes",
        )
        rank_gates = _mapping(rank.get("gates"), f"ranks[{index}].gates")
        for name in (
            "allocator_attachment",
            "backend_native_write",
            "backend_native_read",
        ):
            gate = _mapping(rank_gates.get(name), f"ranks[{index}].gates.{name}")
            if gate.get("passed") is not True:
                raise CertificationError(
                    f"rank {rank_index} gate {name!r} did not pass"
                )
            evidence = _mapping(
                gate.get("evidence"), f"ranks[{index}].gates.{name}.evidence"
            )
            _validate_rank_gate_evidence(name, evidence, geometry, index)

        allocator_evidence = _mapping(
            _mapping(
                rank_gates["allocator_attachment"],
                f"ranks[{index}].gates.allocator_attachment",
            )["evidence"],
            f"ranks[{index}].gates.allocator_attachment.evidence",
        )
        write_evidence = _mapping(
            _mapping(
                rank_gates["backend_native_write"],
                f"ranks[{index}].gates.backend_native_write",
            )["evidence"],
            f"ranks[{index}].gates.backend_native_write.evidence",
        )
        sentinel_hash = hashlib.sha256(
            bytes([0x5A]) * geometry["allocation_bytes"]
        ).hexdigest()
        if allocator_evidence["raw_sha256_before_write"] != sentinel_hash:
            raise CertificationError(
                f"rank {rank_index} did not initialize the whole pool sentinel"
            )
        if (
            write_evidence["raw_sha256_after_write"]
            == allocator_evidence["raw_sha256_before_write"]
        ):
            raise CertificationError(
                f"rank {rank_index} native writer did not change Kapsl memory"
            )

        lifecycle = _mapping(rank.get("lifecycle"), f"ranks[{index}].lifecycle")
        for field in (
            "activation_after_all_attachments",
            "live_lease_detach_rejected",
            "cancellation_release",
            "capacity_exhaustion_rejected",
            "heartbeat_renewal",
            "post_deactivation_reserve_rejected",
        ):
            if lifecycle.get(field) is not True:
                raise CertificationError(
                    f"rank {rank_index} has incomplete lifecycle evidence"
                )

    if seen_ranks != set(range(world_size)):
        raise CertificationError("rank reports do not cover the declared world")
    if seen_devices != set(normalized_devices):
        raise CertificationError("rank reports do not cover the declared devices")


def allowlist_entry(report: Mapping[str, Any]) -> str:
    """Return the exact value accepted by ``--kv-shared-pool-profile``."""

    validate_certification_report(report)
    profile = _mapping(report["profile"], "profile")
    return ",".join(
        str(profile[field])
        for field in (
            "adapter_id",
            "adapter_version",
            "backend_version",
            "profile_id",
        )
    )


def write_json_atomic(path: str | os.PathLike[str], value: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    _write_atomic(
        destination,
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False).encode("utf-8")
        + b"\n",
    )


def write_allowlist_atomic(
    path: str | os.PathLike[str], report: Mapping[str, Any]
) -> None:
    """Write an allowlist value only after strict report validation passes."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    _write_atomic(destination, (allowlist_entry(report) + "\n").encode("utf-8"))


def remove_stale_allowlist(path: str | os.PathLike[str] | None) -> None:
    """Remove an earlier output before a new probe can fail.

    The path is an explicit CLI output, never a discovered or recursive target.
    Leaving an older successful tuple beside a newer failed report is unsafe.
    """

    if path is not None:
        Path(path).unlink(missing_ok=True)


def failed_report(
    *,
    profile: Mapping[str, str],
    environment: Mapping[str, Any],
    matrix: Mapping[str, Any],
    ranks: Sequence[Mapping[str, Any]],
    error: str,
) -> dict[str, Any]:
    """Construct an auditable report that can never produce an allowlist."""

    gates = {
        name: {"passed": False, "evidence": "probe did not complete"}
        for name in REQUIRED_GATES
    }
    return {
        "schema_version": CERTIFICATION_SCHEMA_VERSION,
        "status": "failed",
        "profile": dict(profile),
        "environment": dict(environment),
        "matrix": dict(matrix),
        "gates": gates,
        "ranks": [dict(rank) for rank in ranks],
        "error": error,
    }


def _write_atomic(destination: Path, data: bytes) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as temporary:
            temporary.write(data)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_name, destination)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CertificationError(f"{field} must be an object")
    return value


def _validate_rank_gate_evidence(
    name: str,
    evidence: Mapping[str, Any],
    geometry: Mapping[str, Any],
    rank_index: int,
) -> None:
    prefix = f"ranks[{rank_index}].gates.{name}.evidence"
    if name == "allocator_attachment":
        if evidence.get("allocator_poisoned") is not True:
            raise CertificationError(f"{prefix} did not poison the native allocator")
        if evidence.get("pytorch_cuda_allocation_delta_bytes") != 0:
            raise CertificationError(f"{prefix} observed a second CUDA allocation")
        imported_bytes = _positive_int(
            evidence.get("imported_bytes"), f"{prefix}.imported_bytes"
        )
        if imported_bytes != geometry["allocation_bytes"]:
            raise CertificationError(f"{prefix} imported byte count is inconsistent")
        _positive_int(evidence.get("view_count"), f"{prefix}.view_count")
        _hex_digest(
            evidence.get("raw_sha256_before_write"),
            f"{prefix}.raw_sha256_before_write",
        )
    elif name == "backend_native_write":
        _safe_field(evidence.get("native_function"), f"{prefix}.native_function")
        _hex_digest(
            evidence.get("raw_sha256_after_write"),
            f"{prefix}.raw_sha256_after_write",
        )
        guard_bytes = _positive_int(
            evidence.get("guard_bytes_checked"), f"{prefix}.guard_bytes_checked"
        )
        if guard_bytes != (
            geometry["guard_bytes_per_block"] * geometry["num_blocks"]
        ):
            raise CertificationError(f"{prefix} guard coverage is incomplete")
        _positive_int(evidence.get("prefill_tokens"), f"{prefix}.prefill_tokens")
        if evidence.get("decode_writes") != 2:
            raise CertificationError(f"{prefix} did not perform two decode writes")
        if evidence.get("maximum_block_index") != geometry["num_blocks"] - 1:
            raise CertificationError(f"{prefix} did not write the maximum block")
        if evidence.get("production_implementation_binding") is not True:
            raise CertificationError(f"{prefix} is not bound to production vLLM")
    elif name == "backend_native_read":
        _safe_field(evidence.get("native_function"), f"{prefix}.native_function")
        _safe_field(evidence.get("implementation"), f"{prefix}.implementation")
        _positive_int(
            evidence.get("flash_attention_version"),
            f"{prefix}.flash_attention_version",
        )
        delta = evidence.get("causal_mutation_max_delta")
        if (
            not isinstance(delta, (int, float))
            or isinstance(delta, bool)
            or not math.isfinite(delta)
        ):
            raise CertificationError(f"{prefix} has no causal mutation delta")
        if delta <= 0:
            raise CertificationError(f"{prefix} causal mutation did not change output")
        if evidence.get("production_implementation_binding") is not True:
            raise CertificationError(f"{prefix} is not bound to production vLLM")


def _safe_field(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CertificationError(f"{field} must be a non-empty string")
    if any(character in value for character in (",", "\n", "\r", "\x00")):
        raise CertificationError(f"{field} contains an unsafe delimiter")
    return value


def _sha256_id(value: Any, field: str) -> str:
    value = _safe_field(value, field)
    if not value.startswith("sha256:") or len(value) != 71:
        raise CertificationError(f"{field} must contain a SHA-256 digest")
    try:
        int(value[7:], 16)
    except ValueError as error:
        raise CertificationError(f"{field} contains a non-hex digest") from error
    return value


def _hex_digest(value: Any, field: str) -> str:
    value = _safe_field(value, field)
    if len(value) != 64:
        raise CertificationError(f"{field} must contain 64 hex digits")
    try:
        int(value, 16)
    except ValueError as error:
        raise CertificationError(f"{field} contains a non-hex digest") from error
    return value


def _positive_int(value: Any, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise CertificationError(f"{field} must be a positive integer")
    return value


def _nonnegative_int(value: Any, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise CertificationError(f"{field} must be a non-negative integer")
    return value
