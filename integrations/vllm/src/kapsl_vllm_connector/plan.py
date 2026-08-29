"""Command-line boundary for certified vLLM KV-cache planning.

The provider instantiates the pinned vLLM executor far enough to load the
model, resolve backend-customized cache specs, and build packed cache metadata.
It then shuts the executor down before vLLM's physical KV-cache allocation
boundary.  No Hugging Face configuration formula is used as an authority.
"""

from __future__ import annotations

import argparse
import importlib.metadata as metadata
import json
import os
import sys
from contextlib import contextmanager, redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence

from .connector import (
    ADAPTER_PROFILE_ID,
    ADAPTER_VERSION,
    ELASTIC_ADAPTER_PROFILE_ID,
)
from .planning import (
    PLANNER_SCHEMA_VERSION,
    UINT64_MAX,
    GeometryDescriptor,
    PlannerIdentity,
    PlanningError,
    SizingPolicy,
    build_plan,
    checked_add,
    geometry_from_resolved_configs,
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
    live_resize: bool = False


GeometryProvider = Callable[[RuntimePlanningRequest], GeometryDescriptor]
BackendVersionProvider = Callable[[], str]
MAX_TENSOR_PARALLEL_SIZE = 1024


class RuntimeGeometryUnavailable(PlanningError):
    """The certified runtime could not supply resolved cache geometry."""


@contextmanager
def _runtime_logs_to_stderr() -> Iterator[None]:
    """Keep the planner's machine-readable stdout free of runtime logs.

    vLLM, torch, NCCL, and progress helpers do not all resolve ``sys.stdout``
    at write time. Some retain the original stream object and native code may
    write directly to file descriptor 1. Redirect both layers while the
    executor-backed provider runs, then restore stdout before emitting the
    final planning document.

    The descriptor redirect is skipped for in-memory streams used by host
    tests, while the Python redirect still preserves the same contract.
    """

    original_stdout = sys.stdout
    original_stderr = sys.stderr
    stdout_fd: int | None = None
    stderr_fd: int | None = None
    saved_stdout_fd: int | None = None
    try:
        stdout_fd = original_stdout.fileno()
        stderr_fd = original_stderr.fileno()
    except (AttributeError, OSError, ValueError):
        pass
    else:
        original_stdout.flush()
        original_stderr.flush()
        saved_stdout_fd = os.dup(stdout_fd)
        try:
            os.dup2(stderr_fd, stdout_fd)
        except BaseException:
            os.close(saved_stdout_fd)
            raise

    try:
        with redirect_stdout(original_stderr):
            yield
    finally:
        if saved_stdout_fd is not None and stdout_fd is not None:
            try:
                original_stdout.flush()
                original_stderr.flush()
            finally:
                os.dup2(saved_stdout_fd, stdout_fd)
                os.close(saved_stdout_fd)


@dataclass(frozen=True, slots=True)
class _RuntimePlannerApis:
    """The exact pinned-vLLM seam used before physical KV allocation.

    Keeping these callables in one value makes the production imports explicit
    and lets host tests prove that planning stops before
    ``initialize_from_config`` without importing CUDA or vLLM.
    """

    engine_args_factory: Callable[..., Any]
    executor_class_resolver: Callable[[Any], type[Any]]
    register_all_kvcache_specs: Callable[[Any], None]
    resolve_kv_cache_layout: Callable[[Any, Any, Any], Any]
    get_kv_cache_groups: Callable[[Any, dict[str, Any]], list[Any]]
    get_kv_cache_configs: Callable[[Any, list[dict[str, Any]], list[int]], list[Any]]
    pool_bytes_per_block: Callable[[list[Any]], int]
    max_memory_usage_bytes_from_groups: Callable[[Any, list[Any]], int]
    spec_kind_classifier: Callable[[Any], Any] | None = None


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


def _load_pinned_runtime_apis(backend_version: str) -> _RuntimePlannerApis:
    try:
        from vllm.engine.arg_utils import EngineArgs
        from vllm.v1.attention.backends.utils import resolve_kv_cache_layout
        from vllm.v1.core.kv_cache_utils import (
            _max_memory_usage_bytes_from_groups,
            _pool_bytes_per_block,
            get_kv_cache_configs,
            get_kv_cache_groups,
        )
        from vllm.v1.core.single_type_kv_cache_manager import (
            register_all_kvcache_specs,
        )
        from vllm.v1.executor.abstract import Executor
    except (ImportError, AttributeError) as error:
        raise RuntimeGeometryUnavailable(
            f"vLLM {backend_version} lacks the certified executor planning APIs"
        ) from error

    return _RuntimePlannerApis(
        engine_args_factory=EngineArgs,
        executor_class_resolver=Executor.get_class,
        register_all_kvcache_specs=register_all_kvcache_specs,
        resolve_kv_cache_layout=resolve_kv_cache_layout,
        get_kv_cache_groups=get_kv_cache_groups,
        get_kv_cache_configs=get_kv_cache_configs,
        pool_bytes_per_block=_pool_bytes_per_block,
        max_memory_usage_bytes_from_groups=_max_memory_usage_bytes_from_groups,
    )


def _minimum_planning_memory(
    vllm_config: Any,
    kv_cache_groups: list[Any],
    apis: _RuntimePlannerApis,
) -> int:
    """Return one-sequence memory plus vLLM's permanently reserved null block."""

    try:
        stride = apis.pool_bytes_per_block(kv_cache_groups)
        sequence_bytes = apis.max_memory_usage_bytes_from_groups(
            vllm_config, kv_cache_groups
        )
    except Exception as error:
        raise RuntimeGeometryUnavailable(
            f"pinned vLLM could not size its resolved cache groups: {error}"
        ) from error
    if not isinstance(stride, int) or isinstance(stride, bool) or stride <= 0:
        raise RuntimeGeometryUnavailable(
            "pinned vLLM returned a non-positive cache-pool block stride"
        )
    if (
        not isinstance(sequence_bytes, int)
        or isinstance(sequence_bytes, bool)
        or sequence_bytes <= 0
    ):
        raise RuntimeGeometryUnavailable(
            "pinned vLLM returned a non-positive one-sequence cache requirement"
        )
    try:
        return checked_add(sequence_bytes, stride, "minimum planning memory")
    except PlanningError as error:
        raise RuntimeGeometryUnavailable(str(error)) from error


def _geometry_from_executor(
    request: RuntimePlanningRequest,
    backend_version: str,
    apis: _RuntimePlannerApis,
) -> GeometryDescriptor:
    """Load the pinned executor, resolve cache specs, and stop before KV allocation."""

    engine_options: dict[str, Any] = {
        "model": str(request.model_path),
        "max_model_len": request.max_model_len,
        "tensor_parallel_size": request.tensor_parallel_size,
        "attention_backend": request.attention_backend,
        "enforce_eager": True,
    }
    if request.live_resize:
        # Safe tail unmapping requires every worker forward to have settled
        # before a free block is retired. Pinned vLLM otherwise enables async
        # scheduling by default and overlaps two model batches.
        engine_options["async_scheduling"] = False
    engine_args = apis.engine_args_factory(**engine_options)
    vllm_config = engine_args.create_engine_config()
    if request.live_resize:
        vllm_config.cache_config.kv_cache_layout = "BLNHC"
    executor_class = apis.executor_class_resolver(vllm_config)
    executor = executor_class(vllm_config)
    try:
        apis.register_all_kvcache_specs(vllm_config)
        kv_cache_specs = executor.get_kv_cache_specs()
        if (
            not isinstance(kv_cache_specs, list)
            or len(kv_cache_specs) != request.tensor_parallel_size
            or not kv_cache_specs
            or any(not isinstance(specs, dict) or not specs for specs in kv_cache_specs)
        ):
            raise RuntimeGeometryUnavailable(
                "pinned vLLM cache specs do not cover every tensor-parallel worker"
            )
        first_specs = kv_cache_specs[0]
        if any(specs != first_specs for specs in kv_cache_specs[1:]):
            raise RuntimeGeometryUnavailable(
                "the certified tensor-parallel profile requires identical cache specs on every rank"
            )

        supported_layouts = executor.get_supported_kv_cache_layouts()
        layout = apis.resolve_kv_cache_layout(
            vllm_config,
            supported_layouts,
            [spec for specs in kv_cache_specs for spec in specs.values()],
        )
        layout_name = getattr(layout, "name", None)
        if not isinstance(layout_name, str) or not layout_name.strip():
            raise RuntimeGeometryUnavailable(
                "pinned vLLM returned an invalid resolved cache layout"
            )
        layout_name = layout_name.strip()
        executor.set_kv_cache_layout(layout_name)

        # Planning is TP-only in this profile, so every worker has the same
        # cache groups. Ask vLLM for exactly one full sequence plus its null
        # block; this produces its authoritative packed metadata without ever
        # calling the physical allocation boundary initialize_from_config.
        groups = apis.get_kv_cache_groups(vllm_config, dict(first_specs))
        if not isinstance(groups, list) or not groups:
            raise RuntimeGeometryUnavailable(
                "pinned vLLM returned no resolved cache groups"
            )
        planning_memory = _minimum_planning_memory(vllm_config, groups, apis)
        kv_cache_configs = apis.get_kv_cache_configs(
            vllm_config,
            kv_cache_specs,
            [planning_memory] * request.tensor_parallel_size,
        )
        if not isinstance(kv_cache_configs, list):
            raise RuntimeGeometryUnavailable(
                "pinned vLLM returned malformed worker cache configurations"
            )
        for config in kv_cache_configs:
            config.kv_cache_layout = layout_name

        return geometry_from_resolved_configs(
            kv_cache_configs,
            vllm_config,
            identity=PlannerIdentity(
                adapter_id="kapsl-vllm-connector",
                adapter_version=ADAPTER_VERSION,
                backend_version=backend_version,
                profile_id=(
                    ELASTIC_ADAPTER_PROFILE_ID
                    if request.live_resize
                    else ADAPTER_PROFILE_ID
                ),
            ),
            model_fingerprint=request.model_fingerprint,
            max_model_len=request.max_model_len,
            attention_backend=request.attention_backend,
            layout_id=layout_name,
            device_ids=request.device_ids,
            spec_kind_classifier=apis.spec_kind_classifier,
        )
    finally:
        # A planner process must fully release its model/CUDA/NCCL footprint
        # before MemoryAuthority grants and starts the serving generation.
        executor.shutdown()


def obtain_pinned_runtime_geometry(
    request: RuntimePlanningRequest,
) -> GeometryDescriptor:
    """Obtain cache geometry from the pinned executor before KV allocation."""

    if not request.model_path.is_dir():
        raise RuntimeGeometryUnavailable(
            f"model must be an existing Hugging Face directory: {request.model_path}"
        )
    backend_version = installed_vllm_version()
    apis = _load_pinned_runtime_apis(backend_version)
    return _geometry_from_executor(request, backend_version, apis)


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
    parser.add_argument("--maximum-concurrency", type=int)
    parser.add_argument("--headroom-percent", type=int, default=20)
    parser.add_argument("--prefix-blocks", type=int, default=0)
    parser.add_argument("--alignment-blocks", type=int, default=1)
    parser.add_argument("--min-bytes", type=int)
    parser.add_argument("--max-bytes", type=int)
    parser.add_argument("--strict-concurrency", action="store_true")
    parser.add_argument("--live-resize", action="store_true")
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
        live_resize=bool(getattr(arguments, "live_resize", False)),
    )


def _validate_provider_result(
    request: RuntimePlanningRequest,
    geometry: GeometryDescriptor,
    installed_backend_version: str,
) -> None:
    if (
        geometry.identity.adapter_id != "kapsl-vllm-connector"
        or geometry.identity.adapter_version != ADAPTER_VERSION
        or geometry.identity.profile_id
        != (
            ELASTIC_ADAPTER_PROFILE_ID
            if request.live_resize
            else ADAPTER_PROFILE_ID
        )
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
    if request.live_resize and geometry.layout_id.upper() != "BLNHC":
        raise PlanningError("live-resize runtime geometry must use BLNHC layout")


def _sizing_policy(arguments: argparse.Namespace) -> SizingPolicy:
    """Validate operator sizing inputs before starting the runtime planner."""

    return SizingPolicy(
        target_concurrency=arguments.target_concurrency,
        maximum_concurrency=getattr(arguments, "maximum_concurrency", None),
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
        # The pinned runtime may log through Python streams, retained logging
        # handlers, tqdm, or native libraries. Keep stdout reserved for the
        # single planner JSON document consumed by Kapsl.
        with _runtime_logs_to_stderr():
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
