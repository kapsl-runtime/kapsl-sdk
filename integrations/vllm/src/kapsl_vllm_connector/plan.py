"""Command-line boundary for certified vLLM KV-cache planning.

Phase 0 defines and validates the planning contract.  It intentionally refuses
to derive cache geometry from model configuration alone.  A subsequent
executor-backed provider will obtain resolved cache specs from the pinned vLLM
runtime and pass them to :mod:`kapsl_vllm_connector.planning`.
"""

from __future__ import annotations

import argparse
import importlib.metadata as metadata
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

from .connector import ADAPTER_PROFILE_ID, ADAPTER_VERSION
from .planning import (
    PLANNER_SCHEMA_VERSION,
    UINT64_MAX,
    GeometryDescriptor,
    PlanningError,
    SizingPolicy,
    build_plan,
    planner_error_json_schema,
    planner_json_schema,
)


@dataclass(frozen=True, slots=True)
class RuntimePlanningRequest:
    model_path: Path
    model_fingerprint: str
    max_model_len: int
    tensor_parallel_size: int
    attention_backend: str
    device_ids: tuple[int, ...]


GeometryProvider = Callable[[RuntimePlanningRequest], GeometryDescriptor]
BackendVersionProvider = Callable[[], str]
MAX_TENSOR_PARALLEL_SIZE = 1024


class RuntimeGeometryUnavailable(PlanningError):
    """The certified runtime could not supply resolved cache geometry."""


class PlanningArgumentParser(argparse.ArgumentParser):
    """Turn malformed CLI input into the same structured failure boundary."""

    def error(self, message: str) -> None:
        raise PlanningError(message)


def installed_vllm_version() -> str:
    try:
        return metadata.version("vllm")
    except metadata.PackageNotFoundError as error:
        raise RuntimeGeometryUnavailable(
            "pinned vLLM is not installed; certified runtime geometry is unavailable"
        ) from error


def obtain_pinned_runtime_geometry(
    request: RuntimePlanningRequest,
) -> GeometryDescriptor:
    """Fail closed until an executor-backed runtime provider is certified.

    Import checks make failures actionable when the command is accidentally
    run outside the managed bundle.  Merely finding vLLM is not sufficient:
    its cache specs are created by instantiated attention modules and backend
    ``customize_spec`` hooks, so Phase 0 never substitutes an HF-config formula.
    """

    if not request.model_path.is_dir():
        raise RuntimeGeometryUnavailable(
            f"model must be an existing Hugging Face directory: {request.model_path}"
        )
    backend_version = installed_vllm_version()
    try:
        from vllm.v1.core.kv_cache_utils import (  # noqa: F401
            get_kv_cache_configs,
            get_kv_cache_groups,
        )
    except (ImportError, AttributeError) as error:
        raise RuntimeGeometryUnavailable(
            f"vLLM {backend_version} lacks the certified cache-planning APIs"
        ) from error

    raise RuntimeGeometryUnavailable(
        "pinned vLLM cache APIs are present, but no certified executor-backed "
        "geometry provider is installed; refusing to estimate from model config"
    )


def _parser() -> argparse.ArgumentParser:
    parser = PlanningArgumentParser(
        prog="kapsl-vllm-plan",
        description="Emit a versioned exact KV-cache plan from certified vLLM geometry.",
    )
    parser.add_argument("--print-schema", action="store_true")
    parser.add_argument("--print-error-schema", action="store_true")
    parser.add_argument("--model")
    parser.add_argument("--model-fingerprint")
    parser.add_argument("--max-model-len", type=int)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--attention-backend", default="FLASH_ATTN")
    parser.add_argument("--devices")
    parser.add_argument("--target-concurrency", type=int, default=16)
    parser.add_argument("--headroom-percent", type=int, default=20)
    parser.add_argument("--prefix-blocks", type=int, default=0)
    parser.add_argument("--alignment-blocks", type=int, default=1)
    parser.add_argument("--min-bytes", type=int)
    parser.add_argument("--max-bytes", type=int)
    parser.add_argument("--strict-concurrency", action="store_true")
    return parser


def _required(value: str | int | None, flag: str) -> str | int:
    if value is None or (isinstance(value, str) and not value.strip()):
        raise PlanningError(f"{flag} is required")
    return value


def _device_ids(raw: str | None, world_size: int) -> tuple[int, ...]:
    if world_size <= 0 or world_size > MAX_TENSOR_PARALLEL_SIZE:
        raise PlanningError(
            f"--tensor-parallel-size must be between 1 and {MAX_TENSOR_PARALLEL_SIZE}"
        )
    if raw is None:
        return tuple(range(world_size))
    try:
        devices = tuple(int(value.strip()) for value in raw.split(","))
    except ValueError as error:
        raise PlanningError("--devices must be a comma-separated integer list") from error
    if (
        len(devices) != world_size
        or any(device < 0 for device in devices)
        or any(device > UINT64_MAX for device in devices)
        or len(set(devices)) != len(devices)
    ):
        raise PlanningError(
            "--devices must name one distinct uint64 device per tensor-parallel rank"
        )
    return devices


def _request(arguments: argparse.Namespace) -> RuntimePlanningRequest:
    model = str(_required(arguments.model, "--model"))
    fingerprint = str(
        _required(arguments.model_fingerprint, "--model-fingerprint")
    )
    max_model_len = int(
        _required(arguments.max_model_len, "--max-model-len")
    )
    if max_model_len <= 0:
        raise PlanningError("--max-model-len must be positive")
    if max_model_len > UINT64_MAX:
        raise PlanningError("--max-model-len exceeds the planner uint64 limit")
    if arguments.attention_backend.strip().upper() != "FLASH_ATTN":
        raise PlanningError(
            "the current certified shared-pool profile requires FLASH_ATTN"
        )
    devices = _device_ids(arguments.devices, arguments.tensor_parallel_size)
    return RuntimePlanningRequest(
        model_path=Path(model).expanduser().resolve(),
        model_fingerprint=fingerprint,
        max_model_len=max_model_len,
        tensor_parallel_size=arguments.tensor_parallel_size,
        attention_backend="FLASH_ATTN",
        device_ids=devices,
    )


def _validate_provider_result(
    request: RuntimePlanningRequest,
    geometry: GeometryDescriptor,
    installed_backend_version: str,
) -> None:
    if (
        geometry.identity.adapter_id != "kapsl-vllm-connector"
        or geometry.identity.adapter_version != ADAPTER_VERSION
        or geometry.identity.profile_id != ADAPTER_PROFILE_ID
    ):
        raise PlanningError(
            "runtime geometry adapter/profile identity is not this certified connector"
        )
    if geometry.identity.backend_version != installed_backend_version:
        raise PlanningError(
            "runtime geometry backend version does not match the installed pinned vLLM"
        )
    if geometry.model_fingerprint != request.model_fingerprint:
        raise PlanningError("runtime geometry model fingerprint does not match the request")
    if geometry.max_model_len != request.max_model_len:
        raise PlanningError("runtime geometry max_model_len does not match the request")
    if geometry.tensor_parallel_size != request.tensor_parallel_size:
        raise PlanningError(
            "runtime geometry tensor-parallel world does not match the request"
        )
    if tuple(rank.device_id for rank in geometry.ranks) != request.device_ids:
        raise PlanningError("runtime geometry device mapping does not match the request")
    if geometry.attention_backend.upper() != request.attention_backend:
        raise PlanningError("runtime geometry attention backend does not match the request")


def _sizing_policy(arguments: argparse.Namespace) -> SizingPolicy:
    """Validate operator sizing inputs before starting the runtime planner."""

    return SizingPolicy(
        target_concurrency=arguments.target_concurrency,
        headroom_percent=arguments.headroom_percent,
        prefix_blocks=arguments.prefix_blocks,
        alignment_blocks=arguments.alignment_blocks,
        min_bytes=arguments.min_bytes,
        max_bytes=arguments.max_bytes,
        strict_concurrency=arguments.strict_concurrency,
    )


def _runtime_geometry(
    request: RuntimePlanningRequest,
    provider: GeometryProvider,
) -> GeometryDescriptor:
    try:
        geometry = provider(request)
    except PlanningError:
        raise
    except Exception as error:
        raise RuntimeGeometryUnavailable(
            f"certified runtime geometry provider failed: {error}"
        ) from error
    if not isinstance(geometry, GeometryDescriptor):
        raise RuntimeGeometryUnavailable(
            "certified runtime geometry provider returned a malformed result: "
            "expected GeometryDescriptor"
        )
    return geometry


def _installed_backend_version(provider: BackendVersionProvider) -> str:
    try:
        version = provider()
    except RuntimeGeometryUnavailable:
        raise
    except Exception as error:
        raise RuntimeGeometryUnavailable(
            f"could not identify the installed pinned vLLM build: {error}"
        ) from error
    if not isinstance(version, str) or not version.strip():
        raise RuntimeGeometryUnavailable(
            "could not identify the installed pinned vLLM build: "
            "version provider returned an empty or non-string value"
        )
    return version.strip()


def _validate_runtime_geometry(
    request: RuntimePlanningRequest,
    geometry: GeometryDescriptor,
    installed_backend_version: str,
) -> None:
    try:
        _validate_provider_result(request, geometry, installed_backend_version)
    except PlanningError:
        raise
    except Exception as error:
        # GeometryDescriptor construction validates the certified packed
        # profile. Treat any remaining shape/access failure as a provider
        # boundary failure instead of leaking an implementation traceback.
        raise RuntimeGeometryUnavailable(
            f"certified runtime geometry provider returned a malformed result: {error}"
        ) from error


def run(
    argv: Sequence[str] | None = None,
    *,
    geometry_provider: GeometryProvider = obtain_pinned_runtime_geometry,
    backend_version_provider: BackendVersionProvider = installed_vllm_version,
) -> int:
    try:
        arguments = _parser().parse_args(argv)
        if arguments.print_schema:
            print(json.dumps(planner_json_schema(), indent=2, sort_keys=True))
            return 0
        if arguments.print_error_schema:
            print(json.dumps(planner_error_json_schema(), indent=2, sort_keys=True))
            return 0
        request = _request(arguments)
        # Reject invalid arithmetic and policy bounds before invoking the
        # potentially expensive executor-backed geometry provider.
        policy = _sizing_policy(arguments)
        installed_backend_version = _installed_backend_version(
            backend_version_provider
        )
        geometry = _runtime_geometry(request, geometry_provider)
        _validate_runtime_geometry(
            request,
            geometry,
            installed_backend_version,
        )
        result = build_plan(geometry, policy)
    except PlanningError as error:
        kind = (
            "runtime_geometry_unavailable"
            if isinstance(error, RuntimeGeometryUnavailable)
            else "planning_failed"
        )
        failure = {
            "schema_version": PLANNER_SCHEMA_VERSION,
            "status": "error",
            "supported": False,
            "error": {
                "kind": kind,
                "message": str(error),
            },
        }
        print(json.dumps(failure, sort_keys=True), file=sys.stderr)
        return 2
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    return run(argv)


if __name__ == "__main__":
    raise SystemExit(main())
