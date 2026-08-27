"""Dependency-light contracts for exact managed-vLLM KV-cache planning.

The production planner must obtain cache specs from the certified vLLM build.
This module deliberately does not infer geometry from a Hugging Face model
family.  It validates already-resolved vLLM cache groups, turns their own
``max_memory_usage_bytes`` results into hybrid-safe per-request requirements,
and applies Kapsl's bounded sizing policy with checked unsigned arithmetic.

Keeping the contract independent of torch and vLLM lets host CI exercise the
wire shape, digest stability, and overflow behavior without a CUDA machine.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Callable


PLANNER_SCHEMA_VERSION = 1
PACKED_LAYOUT_SCHEMA_VERSION = 1
UINT64_MAX = (1 << 64) - 1


class PlanningError(ValueError):
    """A cache plan cannot be produced without guessing or overflowing."""


CacheSpecClassifier = Callable[[Any], Any]


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PlanningError(f"{field} must be a non-empty string")
    return value.strip()


def _uint(value: Any, field: str, *, allow_zero: bool = False) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise PlanningError(f"{field} must be an integer")
    minimum = 0 if allow_zero else 1
    if value < minimum or value > UINT64_MAX:
        qualifier = "non-negative" if allow_zero else "positive"
        raise PlanningError(f"{field} must be a {qualifier} unsigned 64-bit integer")
    return value


def validate_certified_shared_pool_execution(vllm_config: Any) -> None:
    """Validate the execution constraints named by the certified profile.

    Planning and connector registration must apply this same check.  A caller-
    supplied profile label is not evidence that the resolved ``VllmConfig``
    actually selected that profile.
    """

    parallel_config = getattr(vllm_config, "parallel_config", None)
    unsupported_parallelism: dict[str, int] = {}
    for name in (
        "pipeline_parallel_size",
        "data_parallel_size",
        "decode_context_parallel_size",
    ):
        raw_size = getattr(parallel_config, name, 1)
        unsupported_parallelism[name] = _uint(raw_size, name)
    # Tensor parallelism is certified, but still reject malformed resolved
    # configuration instead of letting false/float values coerce to integers.
    _uint(
        getattr(parallel_config, "tensor_parallel_size", 1),
        "tensor_parallel_size",
    )
    enabled = [
        name for name, size in unsupported_parallelism.items() if size != 1
    ]
    if enabled:
        raise PlanningError(
            "vLLM shared_pool does not yet support " + ", ".join(enabled)
        )

    model_config = getattr(vllm_config, "model_config", None)
    sleep_mode = getattr(model_config, "enable_sleep_mode", False)
    if not isinstance(sleep_mode, bool):
        raise PlanningError("vLLM enable_sleep_mode must be a boolean")
    if sleep_mode:
        raise PlanningError(
            "vLLM shared_pool does not yet support sleep mode because the "
            "Kapsl-owned CUDA allocation must remain exported"
        )

    attention_config = getattr(vllm_config, "attention_config", None)
    backend = getattr(attention_config, "backend", None)
    backend_name = str(getattr(backend, "name", backend or "")).strip().upper()
    if backend_name != "FLASH_ATTN":
        raise PlanningError(
            "vLLM shared_pool currently requires the explicitly selected "
            "FLASH_ATTN backend; automatic or different attention backends "
            "need their own conformance profile"
        )
    backend_per_kind = getattr(attention_config, "backend_per_kind", None)
    if backend_per_kind is None:
        backend_per_kind = {}
    if not isinstance(backend_per_kind, Mapping):
        raise PlanningError("vLLM backend_per_kind must be a mapping")
    if backend_per_kind:
        raise PlanningError(
            "vLLM shared_pool does not yet support per-cache-kind attention "
            "backend overrides"
        )


def _pinned_vllm_spec_classifier(spec: Any) -> Any:
    try:
        from vllm.v1.kv_cache_interface import get_kv_cache_spec_kind
    except (ImportError, AttributeError) as error:
        raise PlanningError(
            "pinned vLLM get_kv_cache_spec_kind is unavailable"
        ) from error
    try:
        return get_kv_cache_spec_kind(spec)
    except Exception as error:
        raise PlanningError(
            "pinned vLLM could not classify the resolved cache spec"
        ) from error


def resolve_vllm_cache_spec_kind(
    spec: Any,
    *,
    classifier: CacheSpecClassifier | None = None,
) -> str:
    """Resolve one certified allocation kind through the pinned vLLM registry.

    Synthetic host tests may inject a classifier explicitly.  Production data
    cannot opt in by attaching a magic attribute to an otherwise unknown spec.
    Uniform wrappers are classified from every member and must be homogeneous.
    """

    classifier = classifier or _pinned_vllm_spec_classifier
    try:
        marker = classifier(spec)
    except PlanningError:
        raise
    except Exception as error:
        raise PlanningError(
            "pinned vLLM could not classify the resolved cache spec"
        ) from error
    marker = getattr(marker, "value", marker)
    kind = str(marker).strip().lower()
    if kind not in {"full_attention", "sliding_window"}:
        raise PlanningError(
            f"cache spec kind {kind or '<missing>'!r} is not certified by the packed profile"
        )

    members = getattr(spec, "kv_cache_specs", None)
    if isinstance(members, Mapping):
        if not members:
            raise PlanningError("uniform vLLM cache spec has no member specs")
        member_kinds = {
            resolve_vllm_cache_spec_kind(member, classifier=classifier)
            for member in members.values()
        }
        if member_kinds != {kind}:
            raise PlanningError(
                "uniform vLLM cache spec kind differs from its member specs"
            )
    return kind


def checked_add(left: int, right: int, field: str) -> int:
    left = _uint(left, field, allow_zero=True)
    right = _uint(right, field, allow_zero=True)
    value = left + right
    if value > UINT64_MAX:
        raise PlanningError(f"{field} overflows unsigned 64-bit arithmetic")
    return value


def checked_mul(left: int, right: int, field: str) -> int:
    left = _uint(left, field, allow_zero=True)
    right = _uint(right, field, allow_zero=True)
    value = left * right
    if value > UINT64_MAX:
        raise PlanningError(f"{field} overflows unsigned 64-bit arithmetic")
    return value


def ceil_div(value: int, divisor: int, field: str) -> int:
    value = _uint(value, field, allow_zero=True)
    divisor = _uint(divisor, f"{field} divisor")
    return value // divisor + int(value % divisor != 0)


def round_up(value: int, alignment: int, field: str) -> int:
    value = _uint(value, field, allow_zero=True)
    alignment = _uint(alignment, f"{field} alignment")
    return checked_mul(ceil_div(value, alignment, field), alignment, field)


def round_down(value: int, alignment: int, field: str) -> int:
    value = _uint(value, field, allow_zero=True)
    alignment = _uint(alignment, f"{field} alignment")
    return value - value % alignment


@dataclass(frozen=True, slots=True)
class PlannerIdentity:
    adapter_id: str
    adapter_version: str
    backend_version: str
    profile_id: str
    layout_version: int = PACKED_LAYOUT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for field in (
            "adapter_id",
            "adapter_version",
            "backend_version",
            "profile_id",
        ):
            object.__setattr__(self, field, _text(getattr(self, field), field))
        _uint(self.layout_version, "layout_version")
        if self.layout_version != PACKED_LAYOUT_SCHEMA_VERSION:
            raise PlanningError("unsupported packed-layout contract version")

    def to_dict(self) -> dict[str, Any]:
        return {
            "adapter_id": self.adapter_id,
            "adapter_version": self.adapter_version,
            "backend_version": self.backend_version,
            "profile_id": self.profile_id,
            "layout_version": self.layout_version,
        }


@dataclass(frozen=True, slots=True)
class ElementType:
    name: str
    bits: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _text(self.name, "element_type.name"))
        _uint(self.bits, "element_type.bits")
        if self.bits % 8 != 0:
            raise PlanningError(
                "the certified packed CUDA-IPC profile requires byte-addressable elements"
            )

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "bits": self.bits, "bytes": self.bits // 8}


@dataclass(frozen=True, slots=True)
class CacheGroupGeometry:
    group_id: str
    layers: tuple[str, ...]
    block_size_tokens: int
    bytes_per_group_block: int
    required_blocks_per_sequence: int
    kv_heads: int
    key_head_dim: int
    value_head_dim: int
    element_type: ElementType
    policy_kind: str
    window_tokens: int | None = None
    extra_retained_tokens: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "group_id", _text(self.group_id, "group_id"))
        layers = tuple(_text(layer, "layer name") for layer in self.layers)
        if not layers or len(set(layers)) != len(layers):
            raise PlanningError("cache-group layer names must be non-empty and unique")
        object.__setattr__(self, "layers", layers)
        for field in (
            "block_size_tokens",
            "bytes_per_group_block",
            "required_blocks_per_sequence",
            "kv_heads",
            "key_head_dim",
            "value_head_dim",
        ):
            _uint(getattr(self, field), field)
        if self.policy_kind not in {"full_attention", "sliding_window"}:
            raise PlanningError("unsupported certified cache policy")
        if self.policy_kind == "sliding_window":
            if self.window_tokens is None:
                raise PlanningError("sliding-window groups require window_tokens")
            _uint(self.window_tokens, "window_tokens")
        elif self.window_tokens is not None:
            raise PlanningError("full-attention groups cannot set window_tokens")
        _uint(
            self.extra_retained_tokens,
            "extra_retained_tokens",
            allow_zero=True,
        )
        if self.policy_kind != "sliding_window" and self.extra_retained_tokens:
            raise PlanningError(
                "only sliding-window groups can retain extra trailing tokens"
            )

    def to_dict(self) -> dict[str, Any]:
        policy: dict[str, Any] = {"kind": self.policy_kind}
        if self.window_tokens is not None:
            policy["window_tokens"] = self.window_tokens
            policy["extra_retained_tokens"] = self.extra_retained_tokens
        return {
            "group_id": self.group_id,
            "layers": list(self.layers),
            "block_size_tokens": self.block_size_tokens,
            "bytes_per_group_block": self.bytes_per_group_block,
            "required_blocks_per_sequence": self.required_blocks_per_sequence,
            "kv_heads": self.kv_heads,
            "key_head_dim": self.key_head_dim,
            "value_head_dim": self.value_head_dim,
            "element_type": self.element_type.to_dict(),
            "policy": policy,
        }


@dataclass(frozen=True, slots=True)
class CacheGroupTopology:
    """Certified topology fields shared by planning and registration.

    vLLM's ``UniformTypeKVCacheSpecs`` wrapper exposes aggregate accounting,
    but the connector ABI describes one geometry for every layer in the group.
    Keep member validation in one place so registration cannot silently use
    only the first member while the planner rejects a mixed group.
    """

    group_id: str
    layers: tuple[str, ...]
    block_size_tokens: int
    page_size_bytes: int
    bytes_per_group_block: int
    kv_heads: int
    key_head_dim: int
    value_head_dim: int
    element_type: ElementType
    policy_kind: str
    window_tokens: int | None = None
    extra_retained_tokens: int = 0


def shared_pool_block_stride(
    groups: Sequence[CacheGroupGeometry | CacheGroupTopology],
) -> int:
    """Return the one certified physical stride for a shared block pool."""

    if not groups:
        raise PlanningError("shared pool requires at least one cache group")
    strides = [
        _uint(group.bytes_per_group_block, "bytes_per_group_block")
        for group in groups
    ]
    return max(strides)


@dataclass(frozen=True, slots=True)
class RankGeometry:
    rank: int
    device_id: int
    pool_bytes_per_block: int
    groups: tuple[CacheGroupGeometry, ...]
    fixed_overhead_blocks: int = 1

    def __post_init__(self) -> None:
        _uint(self.rank, "rank", allow_zero=True)
        _uint(self.device_id, "device_id", allow_zero=True)
        _uint(self.pool_bytes_per_block, "pool_bytes_per_block")
        _uint(self.fixed_overhead_blocks, "fixed_overhead_blocks", allow_zero=True)
        groups = tuple(sorted(self.groups, key=lambda group: group.group_id))
        if not groups:
            raise PlanningError("rank geometry requires at least one cache group")
        if len({group.group_id for group in groups}) != len(groups):
            raise PlanningError("rank cache-group IDs must be unique")
        expected_pool_stride = shared_pool_block_stride(groups)
        if self.pool_bytes_per_block != expected_pool_stride:
            raise PlanningError(
                "shared pool block stride must equal the largest cache-group stride"
            )
        layers = [layer for group in groups for layer in group.layers]
        if len(layers) != len(set(layers)):
            raise PlanningError(
                "cache-group layer names must be globally unique within a rank"
            )
        object.__setattr__(self, "groups", groups)

    @property
    def required_blocks_per_sequence(self) -> int:
        total = 0
        for group in self.groups:
            total = checked_add(
                total,
                group.required_blocks_per_sequence,
                "required_blocks_per_sequence",
            )
        return total

    def to_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "device_id": self.device_id,
            "pool_bytes_per_block": self.pool_bytes_per_block,
            "fixed_overhead_blocks": self.fixed_overhead_blocks,
            "required_blocks_per_sequence": self.required_blocks_per_sequence,
            "cache_groups": [group.to_dict() for group in self.groups],
        }


@dataclass(frozen=True, slots=True)
class GeometryDescriptor:
    identity: PlannerIdentity
    model_fingerprint: str
    max_model_len: int
    tensor_parallel_size: int
    attention_backend: str
    layout_id: str
    ranks: tuple[RankGeometry, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "model_fingerprint",
            _text(self.model_fingerprint, "model_fingerprint"),
        )
        object.__setattr__(
            self,
            "attention_backend",
            _text(self.attention_backend, "attention_backend").upper(),
        )
        if self.attention_backend != "FLASH_ATTN":
            raise PlanningError(
                "the current certified shared-pool profile requires FLASH_ATTN"
            )
        object.__setattr__(self, "layout_id", _text(self.layout_id, "layout_id"))
        _uint(self.max_model_len, "max_model_len")
        _uint(self.tensor_parallel_size, "tensor_parallel_size")
        ranks = tuple(sorted(self.ranks, key=lambda rank: rank.rank))
        if len(ranks) != self.tensor_parallel_size:
            raise PlanningError("rank geometry must cover the tensor-parallel world")
        if {rank.rank for rank in ranks} != set(range(self.tensor_parallel_size)):
            raise PlanningError("rank geometry is duplicated or out of range")
        if len({rank.device_id for rank in ranks}) != len(ranks):
            raise PlanningError("rank geometry must use distinct device IDs")

        # The current connector registers one capacity model across every TP
        # domain, and vLLM reconciles workers to one common block count. Fail
        # closed if a future backend produces per-rank physical shapes that the
        # ABI cannot yet express.
        first = ranks[0]
        first_groups = [group.to_dict() for group in first.groups]
        for rank in ranks[1:]:
            if (
                rank.pool_bytes_per_block != first.pool_bytes_per_block
                or rank.fixed_overhead_blocks != first.fixed_overhead_blocks
                or [group.to_dict() for group in rank.groups] != first_groups
            ):
                raise PlanningError(
                    "the certified shared-pool profile requires identical cache "
                    "geometry on every tensor-parallel rank"
                )
        object.__setattr__(self, "ranks", ranks)
        # Validate the cross-rank aggregate during construction rather than
        # deferring a possible overflow until serialization.
        _ = self.total_pool_bytes_per_block

    @property
    def total_pool_bytes_per_block(self) -> int:
        total = 0
        for rank in self.ranks:
            total = checked_add(
                total,
                rank.pool_bytes_per_block,
                "total_pool_bytes_per_block",
            )
        return total

    def to_dict(self) -> dict[str, Any]:
        return {
            "identity": self.identity.to_dict(),
            "model_fingerprint": self.model_fingerprint,
            "max_model_len": self.max_model_len,
            "tensor_parallel_size": self.tensor_parallel_size,
            "attention_backend": self.attention_backend,
            "layout_id": self.layout_id,
            "total_pool_bytes_per_block": self.total_pool_bytes_per_block,
            "ranks": [rank.to_dict() for rank in self.ranks],
        }

    def canonical_payload(self) -> dict[str, Any]:
        return {
            "schema_version": PLANNER_SCHEMA_VERSION,
            "geometry": self.to_dict(),
        }

    def geometry_digest(self) -> str:
        encoded = canonical_json(self.canonical_payload()).encode("utf-8")
        return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


@dataclass(frozen=True, slots=True)
class SizingPolicy:
    target_concurrency: int
    headroom_percent: int = 0
    prefix_blocks: int = 0
    alignment_blocks: int = 1
    min_bytes: int | None = None
    max_bytes: int | None = None
    strict_concurrency: bool = False

    def __post_init__(self) -> None:
        _uint(self.target_concurrency, "target_concurrency")
        _uint(self.headroom_percent, "headroom_percent", allow_zero=True)
        if self.headroom_percent > 100:
            raise PlanningError("headroom_percent cannot exceed 100")
        _uint(self.prefix_blocks, "prefix_blocks", allow_zero=True)
        _uint(self.alignment_blocks, "alignment_blocks")
        if self.min_bytes is not None:
            _uint(self.min_bytes, "min_bytes")
        if self.max_bytes is not None:
            _uint(self.max_bytes, "max_bytes")
        if (
            self.min_bytes is not None
            and self.max_bytes is not None
            and self.min_bytes > self.max_bytes
        ):
            raise PlanningError("min_bytes cannot exceed max_bytes")
        if not isinstance(self.strict_concurrency, bool):
            raise PlanningError("strict_concurrency must be a boolean")

    def to_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            "target_concurrency": self.target_concurrency,
            "headroom_percent": self.headroom_percent,
            "prefix_blocks": self.prefix_blocks,
            "alignment_blocks": self.alignment_blocks,
            "strict_concurrency": self.strict_concurrency,
        }
        if self.min_bytes is not None:
            value["min_bytes"] = self.min_bytes
        if self.max_bytes is not None:
            value["max_bytes"] = self.max_bytes
        return value


@dataclass(frozen=True, slots=True)
class RankSizing:
    rank: int
    device_id: int
    bytes_per_block: int
    sequence_blocks: int
    minimum_blocks: int
    minimum_bytes: int
    base_blocks: int
    headroom_blocks: int
    desired_blocks: int
    desired_bytes: int
    effective_target_concurrency: int
    concurrency_reduced: bool

    def to_dict(self) -> dict[str, int]:
        return {
            "rank": self.rank,
            "device_id": self.device_id,
            "bytes_per_block": self.bytes_per_block,
            "sequence_blocks": self.sequence_blocks,
            "minimum_blocks": self.minimum_blocks,
            "minimum_bytes": self.minimum_bytes,
            "base_blocks": self.base_blocks,
            "headroom_blocks": self.headroom_blocks,
            "desired_blocks": self.desired_blocks,
            "desired_bytes": self.desired_bytes,
            "effective_target_concurrency": self.effective_target_concurrency,
            "concurrency_reduced": self.concurrency_reduced,
        }


@dataclass(frozen=True, slots=True)
class PlanningResult:
    geometry: GeometryDescriptor
    policy: SizingPolicy
    ranks: tuple[RankSizing, ...]

    def __post_init__(self) -> None:
        ranks = tuple(self.ranks)
        if len(ranks) != self.geometry.tensor_parallel_size:
            raise PlanningError("sizing results must cover every rank")
        if {rank.rank for rank in ranks} != set(range(len(ranks))):
            raise PlanningError("sizing result ranks are duplicated or out of range")
        if len({rank.desired_blocks for rank in ranks}) != 1:
            raise PlanningError("vLLM tensor-parallel ranks require one block count")
        if len({rank.desired_bytes for rank in ranks}) != 1:
            raise PlanningError(
                "the pinned vLLM exact-byte CLI requires the same byte grant on every rank"
            )
        object.__setattr__(self, "ranks", ranks)
        _ = self.total_desired_bytes

    @property
    def total_desired_bytes(self) -> int:
        total = 0
        for rank in self.ranks:
            total = checked_add(total, rank.desired_bytes, "total_desired_bytes")
        return total

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": PLANNER_SCHEMA_VERSION,
            "status": "planned",
            "supported": True,
            "geometry_digest": self.geometry.geometry_digest(),
            "geometry": self.geometry.to_dict(),
            "policy": self.policy.to_dict(),
            "sizing": {
                "ranks": [rank.to_dict() for rank in self.ranks],
                "total_desired_bytes": self.total_desired_bytes,
            },
        }


def canonical_json(value: Mapping[str, Any]) -> str:
    """Serialize a contract payload in the one form used for its digest."""

    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as error:
        raise PlanningError(f"geometry is not canonical JSON: {error}") from error


def size_rank(rank: RankGeometry, policy: SizingPolicy) -> RankSizing:
    """Apply a bounded policy to one exact vLLM block-pool geometry."""

    sequence_blocks = rank.required_blocks_per_sequence
    minimum_unaligned = checked_add(
        rank.fixed_overhead_blocks,
        sequence_blocks,
        "minimum_blocks",
    )
    minimum_blocks = round_up(
        minimum_unaligned, policy.alignment_blocks, "minimum_blocks"
    )
    minimum_bytes = checked_mul(
        minimum_blocks, rank.pool_bytes_per_block, "minimum_bytes"
    )

    workload_blocks = checked_mul(
        sequence_blocks, policy.target_concurrency, "workload_blocks"
    )
    workload_and_prefix = checked_add(
        workload_blocks, policy.prefix_blocks, "base_blocks"
    )
    headroom_numerator = checked_mul(
        workload_and_prefix, policy.headroom_percent, "headroom_blocks"
    )
    headroom_blocks = ceil_div(headroom_numerator, 100, "headroom_blocks")
    base_blocks = checked_add(
        rank.fixed_overhead_blocks, workload_and_prefix, "base_blocks"
    )
    desired_unaligned = checked_add(
        base_blocks, headroom_blocks, "desired_blocks"
    )
    desired_blocks = round_up(
        desired_unaligned, policy.alignment_blocks, "desired_blocks"
    )

    floor_blocks: int | None = None
    if policy.min_bytes is not None:
        if policy.min_bytes < minimum_bytes:
            raise PlanningError(
                f"rank {rank.rank} min_bytes is below the one max_model_len sequence minimum"
            )
        floor_blocks = round_up(
            ceil_div(policy.min_bytes, rank.pool_bytes_per_block, "min_bytes"),
            policy.alignment_blocks,
            "min_bytes blocks",
        )
        desired_blocks = max(desired_blocks, floor_blocks)

    if policy.max_bytes is not None:
        cap_blocks = round_down(
            policy.max_bytes // rank.pool_bytes_per_block,
            policy.alignment_blocks,
            "max_bytes blocks",
        )
        if cap_blocks < minimum_blocks:
            raise PlanningError(
                f"rank {rank.rank} max_bytes cannot hold one max_model_len sequence"
            )
        if floor_blocks is not None and cap_blocks < floor_blocks:
            raise PlanningError(
                "min_bytes and max_bytes conflict after block alignment"
            )
        desired_blocks = min(desired_blocks, cap_blocks)

    desired_bytes = checked_mul(
        desired_blocks, rank.pool_bytes_per_block, "desired_bytes"
    )
    # Prefix retention and headroom are optional capacity.  A byte cap sheds
    # both before it reduces the number of whole max-length sequences that the
    # pool can hold.  Counting the configured prefix allowance here would let
    # an optional allowance make a valid one-sequence minimum look unusable.
    usable_for_sequences = desired_blocks - rank.fixed_overhead_blocks
    effective = min(
        policy.target_concurrency,
        usable_for_sequences // sequence_blocks,
    )
    if effective < 1:
        raise PlanningError("resolved plan cannot hold one max_model_len sequence")
    reduced = effective < policy.target_concurrency
    if reduced and policy.strict_concurrency:
        raise PlanningError(
            f"rank {rank.rank} max_bytes reduces target concurrency from "
            f"{policy.target_concurrency} to {effective} in strict mode"
        )
    return RankSizing(
        rank=rank.rank,
        device_id=rank.device_id,
        bytes_per_block=rank.pool_bytes_per_block,
        sequence_blocks=sequence_blocks,
        minimum_blocks=minimum_blocks,
        minimum_bytes=minimum_bytes,
        base_blocks=base_blocks,
        headroom_blocks=headroom_blocks,
        desired_blocks=desired_blocks,
        desired_bytes=desired_bytes,
        effective_target_concurrency=effective,
        concurrency_reduced=reduced,
    )


def build_plan(
    geometry: GeometryDescriptor, policy: SizingPolicy
) -> PlanningResult:
    return PlanningResult(
        geometry=geometry,
        policy=policy,
        ranks=tuple(size_rank(rank, policy) for rank in geometry.ranks),
    )


_DTYPE_BITS = {
    "float16": 16,
    "half": 16,
    "bfloat16": 16,
    "float32": 32,
    "float": 32,
    "int8": 8,
    "uint8": 8,
    "float8_e4m3fn": 8,
    "float8_e4m3fnuz": 8,
    "float8_e5m2": 8,
    "float8_e5m2fnuz": 8,
}


def _element_type(dtype: Any) -> ElementType:
    name = str(dtype).strip().lower().removeprefix("torch.")
    bits = _DTYPE_BITS.get(name)
    if bits is None:
        raise PlanningError(
            f"cache dtype {name or '<missing>'!r} is not certified by the packed CUDA-IPC planner"
        )
    return ElementType(name=name, bits=bits)


def _quant_mode(value: Any) -> str:
    if value is None:
        return "NONE"
    name = getattr(value, "name", None)
    if name is None:
        name = str(value).rsplit(".", 1)[-1]
    return str(name).strip().upper()


def _validate_dense_attention_layout(
    members: Sequence[Any],
    element_type: ElementType,
    block_size: int,
    kv_heads: int,
    key_head_dim: int,
    value_head_dim: int,
) -> None:
    """Reject packed/quantized shapes not represented by connector ABI v1."""

    quant_mode = _uniform(
        [_quant_mode(getattr(member, "kv_quant_mode", None)) for member in members],
        "KV quantization modes",
    )
    if quant_mode != "NONE":
        raise PlanningError("quantized KV layouts require a new certified planner profile")
    if any(getattr(member, "page_size_padded", None) is not None for member in members):
        raise PlanningError("padded KV pages require a new certified planner profile")
    head_slots = _uniform(
        [getattr(member, "num_head_slots", None) for member in members],
        "packed head-slot counts",
    )
    if head_slots is not None:
        if _uint(head_slots, "num_head_slots") != kv_heads:
            raise PlanningError(
                "packed KV head slots require a new certified planner profile"
            )
    state_bytes = _uniform(
        [getattr(member, "state_content_bytes", None) for member in members],
        "packed state sizes",
    )
    dense_state_bytes = checked_mul(
        checked_add(key_head_dim, value_head_dim, "dense state dimensions"),
        element_type.bits // 8,
        "dense state bytes",
    )
    if state_bytes is not None:
        if _uint(state_bytes, "state_content_bytes") != dense_state_bytes:
            raise PlanningError(
                "packed KV state content requires a new certified planner profile"
            )
    dense_page_bytes = checked_mul(
        checked_mul(block_size, kv_heads, "dense KV page elements"),
        dense_state_bytes,
        "dense KV page bytes",
    )
    for member in members:
        if (
            _positive_attr(member, ("page_size_bytes",), "page_size_bytes")
            != dense_page_bytes
        ):
            raise PlanningError(
                "resolved KV page bytes do not match the certified dense attention geometry"
            )
    token_ratios = []
    for member in members:
        raw_ratio = getattr(member, "tokens_per_state", 1)
        if isinstance(raw_ratio, bool) or not isinstance(raw_ratio, (int, Fraction)):
            raise PlanningError("tokens_per_state is not an exact rational value")
        try:
            ratio = Fraction(raw_ratio)
        except (TypeError, ValueError, ZeroDivisionError) as error:
            raise PlanningError("tokens_per_state is not a valid rational value") from error
        token_ratios.append(ratio)
    if _uniform(token_ratios, "tokens-per-state ratios") != Fraction(1, 1):
        raise PlanningError("compressed KV states require a new certified planner profile")


def _positive_attr(value: Any, names: Sequence[str], field: str) -> int:
    for name in names:
        raw = getattr(value, name, None)
        if raw is not None:
            return _uint(raw, field)
    raise PlanningError(f"resolved vLLM spec does not expose {field}")


def _member_specs(spec: Any, layer_names: tuple[str, ...]) -> tuple[Any, ...]:
    members = getattr(spec, "kv_cache_specs", None)
    if isinstance(members, Mapping):
        if set(map(str, members)) != set(layer_names):
            raise PlanningError(
                "uniform vLLM cache spec layer membership differs from its cache group"
            )
        return tuple(members[name] for name in layer_names)
    return (spec,)


def _uniform(values: Sequence[Any], field: str) -> Any:
    if not values or any(value != values[0] for value in values[1:]):
        raise PlanningError(
            f"the certified topology cannot represent mixed {field} inside one cache group"
        )
    return values[0]


def _extract_group_topology(
    raw_group: Any,
    group_index: int,
    spec_kind_classifier: CacheSpecClassifier | None,
) -> CacheGroupTopology:
    raw_layers = getattr(raw_group, "layer_names", None)
    if not isinstance(raw_layers, Sequence) or isinstance(raw_layers, (str, bytes)):
        raise PlanningError(f"vLLM cache group {group_index} has no layer list")
    layers = tuple(_text(str(layer), "layer name") for layer in raw_layers)
    if not layers:
        raise PlanningError(f"vLLM cache group {group_index} has no layers")
    spec = getattr(raw_group, "kv_cache_spec", None)
    if spec is None:
        raise PlanningError(f"vLLM cache group {group_index} has no cache spec")
    block_size = _positive_attr(spec, ("block_size",), "block_size_tokens")
    page_size = _positive_attr(spec, ("page_size_bytes",), "page_size_bytes")

    members = _member_specs(spec, layers)
    spec_kind = resolve_vllm_cache_spec_kind(
        spec,
        classifier=spec_kind_classifier,
    )
    member_block_size = _uniform(
        [
            _positive_attr(member, ("block_size",), "block_size_tokens")
            for member in members
        ],
        "block sizes",
    )
    if member_block_size != block_size:
        raise PlanningError(
            "uniform vLLM cache spec block size differs from its member specs"
        )
    kv_heads = _uniform(
        [
            _positive_attr(member, ("num_kv_heads", "num_heads"), "kv_heads")
            for member in members
        ],
        "KV-head counts",
    )
    key_head_dim = _uniform(
        [_positive_attr(member, ("head_size",), "key_head_dim") for member in members],
        "key head dimensions",
    )
    value_head_dim = _uniform(
        [
            _positive_attr(
                member,
                ("head_size_v", "head_size"),
                "value_head_dim",
            )
            for member in members
        ],
        "value head dimensions",
    )
    element_type = _uniform(
        [_element_type(getattr(member, "dtype", None)) for member in members],
        "cache element types",
    )
    _validate_dense_attention_layout(
        members,
        element_type,
        block_size,
        kv_heads,
        key_head_dim,
        value_head_dim,
    )
    chunks = [getattr(member, "attention_chunk_size", None) for member in members]
    if any(chunk is not None for chunk in chunks):
        raise PlanningError(
            "the certified packed CUDA-IPC profile does not support chunked-local attention"
        )
    if spec_kind == "sliding_window":
        windows = [getattr(member, "sliding_window", None) for member in members]
        window = _uniform(windows, "sliding-window policies")
        if window is None:
            raise PlanningError("resolved sliding-window spec has no window size")
        policy_kind = "sliding_window"
        window_tokens = _uint(window, "window_tokens")
        extra_retained_tokens = _uniform(
            [getattr(member, "extra_retained_tokens", 0) for member in members],
            "extra retained token counts",
        )
        _uint(extra_retained_tokens, "extra_retained_tokens", allow_zero=True)
        if extra_retained_tokens != 0:
            raise PlanningError(
                "the certified KV ABI cannot represent nonzero extra retained tokens"
            )
    else:
        policy_kind = "full_attention"
        window_tokens = None
        extra_retained_tokens = 0

    inner = getattr(spec, "kv_cache_specs", None)
    if isinstance(inner, Mapping):
        _uniform(
            [
                _positive_attr(member, ("page_size_bytes",), "page_size_bytes")
                for member in members
            ],
            "member page sizes",
        )
        inner_page_total = 0
        for member in members:
            inner_page_total = checked_add(
                inner_page_total,
                _positive_attr(member, ("page_size_bytes",), "page_size_bytes"),
                "bytes_per_group_block",
            )
        if inner_page_total != page_size:
            raise PlanningError(
                "uniform vLLM cache spec page accounting is internally inconsistent"
            )
        group_bytes = page_size
    else:
        group_bytes = checked_mul(page_size, len(layers), "bytes_per_group_block")

    return CacheGroupTopology(
        group_id=f"vllm.group.{group_index}",
        layers=layers,
        block_size_tokens=block_size,
        page_size_bytes=page_size,
        bytes_per_group_block=group_bytes,
        kv_heads=kv_heads,
        key_head_dim=key_head_dim,
        value_head_dim=value_head_dim,
        element_type=element_type,
        policy_kind=policy_kind,
        window_tokens=window_tokens,
        extra_retained_tokens=extra_retained_tokens,
    )


def extract_cache_group_topologies(
    kv_cache_config: Any,
    *,
    spec_kind_classifier: CacheSpecClassifier | None = None,
) -> tuple[CacheGroupTopology, ...]:
    """Extract the exact certified group topology used by both wire paths."""

    raw_groups = getattr(kv_cache_config, "kv_cache_groups", None)
    if not isinstance(raw_groups, Sequence) or not raw_groups:
        raise PlanningError("resolved vLLM KVCacheConfig has no cache groups")
    groups = tuple(
        _extract_group_topology(group, index, spec_kind_classifier)
        for index, group in enumerate(raw_groups)
    )
    layers = [layer for group in groups for layer in group.layers]
    if len(layers) != len(set(layers)):
        raise PlanningError(
            "cache-group layer names must be globally unique within a rank"
        )
    return groups


def _size_group(
    raw_group: Any,
    topology: CacheGroupTopology,
    group_index: int,
    vllm_config: Any,
) -> CacheGroupGeometry:
    spec = getattr(raw_group, "kv_cache_spec", None)
    maximum = getattr(spec, "max_memory_usage_bytes", None)
    if not callable(maximum):
        raise PlanningError(
            "resolved vLLM cache specs must provide max_memory_usage_bytes"
        )
    try:
        maximum_bytes = _uint(maximum(vllm_config), "max_memory_usage_bytes")
    except PlanningError:
        raise
    except Exception as error:
        raise PlanningError(
            f"vLLM cache group {group_index} could not size one sequence: {error}"
        ) from error
    sequence_blocks = ceil_div(
        maximum_bytes,
        topology.page_size_bytes,
        "required_blocks_per_sequence",
    )
    return CacheGroupGeometry(
        group_id=topology.group_id,
        layers=topology.layers,
        block_size_tokens=topology.block_size_tokens,
        bytes_per_group_block=topology.bytes_per_group_block,
        required_blocks_per_sequence=sequence_blocks,
        kv_heads=topology.kv_heads,
        key_head_dim=topology.key_head_dim,
        value_head_dim=topology.value_head_dim,
        element_type=topology.element_type,
        policy_kind=topology.policy_kind,
        window_tokens=topology.window_tokens,
        extra_retained_tokens=topology.extra_retained_tokens,
    )


def extract_rank_geometry(
    kv_cache_config: Any,
    vllm_config: Any,
    *,
    rank: int,
    device_id: int,
    spec_kind_classifier: CacheSpecClassifier | None = None,
) -> RankGeometry:
    """Extract one worker's certified shape from a resolved vLLM config.

    ``kv_cache_config`` may be the final packed config, in which case its
    tensor allocation is cross-checked, or a planner-created config carrying
    only resolved groups. No fallback formula is used.
    """

    raw_groups = getattr(kv_cache_config, "kv_cache_groups", None)
    topologies = extract_cache_group_topologies(
        kv_cache_config,
        spec_kind_classifier=spec_kind_classifier,
    )
    groups = tuple(
        _size_group(
            raw_groups[index],
            topology,
            index,
            vllm_config,
        )
        for index, topology in enumerate(topologies)
    )
    pool_stride = shared_pool_block_stride(groups)

    tensors = getattr(kv_cache_config, "kv_cache_tensors", None)
    if tensors:
        num_blocks = _positive_attr(kv_cache_config, ("num_blocks",), "num_blocks")
        sizes = {
            _positive_attr(tensor, ("size",), "kv_cache_tensor.size")
            for tensor in tensors
        }
        if len(sizes) != 1:
            raise PlanningError(
                "packed vLLM KV tensor placements do not share one backing size"
            )
        packed_bytes = sizes.pop()
        packed_stride, remainder = divmod(packed_bytes, num_blocks)
        if remainder or packed_stride != pool_stride:
            raise PlanningError(
                "resolved packed allocation disagrees with cache-group page accounting"
            )

    return RankGeometry(
        rank=rank,
        device_id=device_id,
        pool_bytes_per_block=pool_stride,
        groups=groups,
        # vLLM's BlockPool permanently retains one null block. Its own memory
        # sufficiency check subtracts this block before testing max_model_len.
        fixed_overhead_blocks=1,
    )


def geometry_from_resolved_configs(
    kv_cache_configs: Sequence[Any],
    vllm_config: Any,
    *,
    identity: PlannerIdentity,
    model_fingerprint: str,
    max_model_len: int,
    attention_backend: str,
    layout_id: str,
    device_ids: Sequence[int],
    spec_kind_classifier: CacheSpecClassifier | None = None,
) -> GeometryDescriptor:
    """Build the versioned geometry contract from pinned vLLM output."""

    validate_certified_shared_pool_execution(vllm_config)
    if not kv_cache_configs:
        raise PlanningError("pinned vLLM returned no worker cache configurations")
    if len(kv_cache_configs) != len(device_ids):
        raise PlanningError("device IDs must cover every vLLM worker configuration")
    parallel_config = getattr(vllm_config, "parallel_config", None)
    configured_tp = getattr(parallel_config, "tensor_parallel_size", None)
    if configured_tp is None:
        raise PlanningError(
            "resolved vLLM config must expose tensor_parallel_size"
        )
    configured_tp = _uint(configured_tp, "tensor_parallel_size")
    if configured_tp != len(kv_cache_configs):
        raise PlanningError(
            "resolved tensor_parallel_size differs from the worker cache configurations"
        )
    for config in kv_cache_configs:
        resolved_layout = getattr(config, "kv_cache_layout", None)
        if not isinstance(resolved_layout, str) or not resolved_layout.strip():
            raise PlanningError(
                "resolved vLLM KVCacheConfig must expose kv_cache_layout"
            )
        if resolved_layout.strip() != layout_id:
            raise PlanningError(
                "planner layout_id differs from the resolved vLLM KVCacheConfig"
            )
    configured_max_len = getattr(
        getattr(vllm_config, "model_config", None), "max_model_len", None
    )
    if configured_max_len is None:
        raise PlanningError("resolved vLLM config must expose max_model_len")
    configured_max_len = _uint(configured_max_len, "configured max_model_len")
    if configured_max_len != max_model_len:
        raise PlanningError(
            "planner max_model_len differs from the resolved vLLM config"
        )
    ranks = tuple(
        extract_rank_geometry(
            config,
            vllm_config,
            rank=rank,
            device_id=_uint(device_ids[rank], "device_id", allow_zero=True),
            spec_kind_classifier=spec_kind_classifier,
        )
        for rank, config in enumerate(kv_cache_configs)
    )
    return GeometryDescriptor(
        identity=identity,
        model_fingerprint=model_fingerprint,
        max_model_len=max_model_len,
        tensor_parallel_size=len(ranks),
        attention_backend=attention_backend,
        layout_id=layout_id,
        ranks=ranks,
    )


def planner_json_schema() -> dict[str, Any]:
    """Return the immutable JSON Schema for successful planner output."""

    positive = {"type": "integer", "minimum": 1, "maximum": UINT64_MAX}
    nonnegative = {"type": "integer", "minimum": 0, "maximum": UINT64_MAX}
    text = {"type": "string", "minLength": 1}
    rank_sizing_properties: dict[str, Any] = {
        "rank": nonnegative,
        "device_id": nonnegative,
        "bytes_per_block": positive,
        "sequence_blocks": positive,
        "minimum_blocks": positive,
        "minimum_bytes": positive,
        "base_blocks": positive,
        "headroom_blocks": nonnegative,
        "desired_blocks": positive,
        "desired_bytes": positive,
        "effective_target_concurrency": positive,
        "concurrency_reduced": {"type": "boolean"},
    }
    element = {
        "type": "object",
        "additionalProperties": False,
        "required": ["name", "bits", "bytes"],
        "properties": {
            "name": text,
            "bits": positive,
            "bytes": positive,
        },
    }
    cache_policy = {
        "oneOf": [
            {
                "type": "object",
                "additionalProperties": False,
                "required": ["kind"],
                "properties": {"kind": {"const": "full_attention"}},
            },
            {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "kind",
                    "window_tokens",
                    "extra_retained_tokens",
                ],
                "properties": {
                    "kind": {"const": "sliding_window"},
                    "window_tokens": positive,
                    "extra_retained_tokens": nonnegative,
                },
            },
        ]
    }
    cache_group = {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "group_id",
            "layers",
            "block_size_tokens",
            "bytes_per_group_block",
            "required_blocks_per_sequence",
            "kv_heads",
            "key_head_dim",
            "value_head_dim",
            "element_type",
            "policy",
        ],
        "properties": {
            "group_id": text,
            "layers": {
                "type": "array",
                "minItems": 1,
                "uniqueItems": True,
                "items": text,
            },
            "block_size_tokens": positive,
            "bytes_per_group_block": positive,
            "required_blocks_per_sequence": positive,
            "kv_heads": positive,
            "key_head_dim": positive,
            "value_head_dim": positive,
            "element_type": element,
            "policy": cache_policy,
        },
    }
    rank_geometry = {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "rank",
            "device_id",
            "pool_bytes_per_block",
            "fixed_overhead_blocks",
            "required_blocks_per_sequence",
            "cache_groups",
        ],
        "properties": {
            "rank": nonnegative,
            "device_id": nonnegative,
            "pool_bytes_per_block": positive,
            "fixed_overhead_blocks": nonnegative,
            "required_blocks_per_sequence": positive,
            "cache_groups": {
                "type": "array",
                "minItems": 1,
                "items": cache_group,
            },
        },
    }
    identity = {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "adapter_id",
            "adapter_version",
            "backend_version",
            "profile_id",
            "layout_version",
        ],
        "properties": {
            "adapter_id": text,
            "adapter_version": text,
            "backend_version": text,
            "profile_id": text,
            "layout_version": positive,
        },
    }
    geometry = {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "identity",
            "model_fingerprint",
            "max_model_len",
            "tensor_parallel_size",
            "attention_backend",
            "layout_id",
            "total_pool_bytes_per_block",
            "ranks",
        ],
        "properties": {
            "identity": identity,
            "model_fingerprint": text,
            "max_model_len": positive,
            "tensor_parallel_size": positive,
            "attention_backend": {"const": "FLASH_ATTN"},
            "layout_id": text,
            "total_pool_bytes_per_block": positive,
            "ranks": {
                "type": "array",
                "minItems": 1,
                "items": rank_geometry,
            },
        },
    }
    sizing_policy = {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "target_concurrency",
            "headroom_percent",
            "prefix_blocks",
            "alignment_blocks",
            "strict_concurrency",
        ],
        "properties": {
            "target_concurrency": positive,
            "headroom_percent": {
                "type": "integer",
                "minimum": 0,
                "maximum": 100,
            },
            "prefix_blocks": nonnegative,
            "alignment_blocks": positive,
            "strict_concurrency": {"type": "boolean"},
            "min_bytes": positive,
            "max_bytes": positive,
        },
    }
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://kapsl.ai/schemas/vllm-kv-plan-v1.json",
        "title": "Kapsl certified vLLM KV plan",
        "type": "object",
        "additionalProperties": False,
        "required": [
            "schema_version",
            "status",
            "supported",
            "geometry_digest",
            "geometry",
            "policy",
            "sizing",
        ],
        "properties": {
            "schema_version": {"const": PLANNER_SCHEMA_VERSION},
            "status": {"const": "planned"},
            "supported": {"const": True},
            "geometry_digest": {
                "type": "string",
                "pattern": "^sha256:[0-9a-f]{64}$",
            },
            "geometry": geometry,
            "policy": sizing_policy,
            "sizing": {
                "type": "object",
                "additionalProperties": False,
                "required": ["ranks", "total_desired_bytes"],
                "properties": {
                    "ranks": {
                        "type": "array",
                        "minItems": 1,
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": list(rank_sizing_properties),
                            "properties": rank_sizing_properties,
                        },
                    },
                    "total_desired_bytes": positive,
                },
            },
        },
    }


def planner_error_json_schema() -> dict[str, Any]:
    """Return the immutable schema for fail-closed planner output."""

    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://kapsl.ai/schemas/vllm-kv-plan-error-v1.json",
        "title": "Kapsl certified vLLM KV planning failure",
        "type": "object",
        "additionalProperties": False,
        "required": ["schema_version", "status", "supported", "error"],
        "properties": {
            "schema_version": {"const": PLANNER_SCHEMA_VERSION},
            "status": {"const": "error"},
            "supported": {"const": False},
            "error": {
                "type": "object",
                "additionalProperties": False,
                "required": ["kind", "message"],
                "properties": {
                    "kind": {
                        "enum": [
                            "runtime_geometry_unavailable",
                            "planning_failed",
                        ]
                    },
                    "message": {"type": "string", "minLength": 1},
                },
            },
        },
    }
