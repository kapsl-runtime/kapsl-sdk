from __future__ import annotations

import io
import json
import os
import re
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import cast

from kapsl_vllm_connector.connector import (
    ADAPTER_PROFILE_ID,
    ADAPTER_VERSION,
    _vllm_capacity_groups,
    _vllm_topology,
)
from kapsl_vllm_connector.plan import (
    RuntimeGeometryUnavailable,
    RuntimePlanningRequest,
    _geometry_from_executor,
    _RuntimePlannerApis,
    run,
)
from kapsl_vllm_connector.planning import (
    PLANNER_SCHEMA_VERSION,
    UINT64_MAX,
    GeometryDescriptor,
    PlannerIdentity,
    PlanningError,
    RankGeometry,
    SizingPolicy,
    build_plan,
    geometry_from_resolved_configs,
    planner_error_json_schema,
    planner_json_schema,
)


class SyntheticSpec:
    def __init__(
        self,
        *,
        block_size: int,
        page_size_bytes: int,
        maximum_pages: int,
        dtype: str = "float16",
        num_kv_heads: int = 4,
        head_size: int = 4,
        head_size_v: int | None = None,
        sliding_window: int | None = None,
        extra_retained_tokens: int = 0,
        spec_kind: str | None = None,
    ) -> None:
        self.block_size = block_size
        self.page_size_bytes = page_size_bytes
        self.maximum_pages = maximum_pages
        self.dtype = dtype
        self.num_kv_heads = num_kv_heads
        self.head_size = head_size
        self.head_size_v = head_size if head_size_v is None else head_size_v
        self.sliding_window = sliding_window
        self.extra_retained_tokens = extra_retained_tokens
        self.attention_chunk_size = None
        self.synthetic_spec_kind = spec_kind or (
            "sliding_window" if sliding_window is not None else "full_attention"
        )

    def max_memory_usage_bytes(self, _vllm_config: object) -> int:
        return self.maximum_pages * self.page_size_bytes


class SyntheticUniformSpec:
    def __init__(self, specs: dict[str, SyntheticSpec]) -> None:
        self.kv_cache_specs = specs
        self.block_size = next(iter(specs.values())).block_size
        self.page_size_bytes = sum(spec.page_size_bytes for spec in specs.values())

    def max_memory_usage_bytes(self, vllm_config: object) -> int:
        maximum_pages = max(
            (
                spec.max_memory_usage_bytes(vllm_config)
                + spec.page_size_bytes
                - 1
            )
            // spec.page_size_bytes
            for spec in self.kv_cache_specs.values()
        )
        return maximum_pages * self.page_size_bytes


def _synthetic_spec_classifier(spec: object) -> str:
    members = getattr(spec, "kv_cache_specs", None)
    if isinstance(members, dict):
        return _synthetic_spec_classifier(next(iter(members.values())))
    return cast(str, getattr(spec, "synthetic_spec_kind"))


def _vllm_config(
    *,
    max_model_len: int = 1024,
    attention_backend: str = "FLASH_ATTN",
    pipeline_parallel_size: int = 1,
) -> SimpleNamespace:
    return SimpleNamespace(
        cache_config=SimpleNamespace(kv_cache_layout=None),
        model_config=SimpleNamespace(
            max_model_len=max_model_len,
            enable_sleep_mode=False,
        ),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=1,
            pipeline_parallel_size=pipeline_parallel_size,
            data_parallel_size=1,
            decode_context_parallel_size=1,
        ),
        attention_config=SimpleNamespace(
            backend=SimpleNamespace(name=attention_backend),
            backend_per_kind={},
        ),
    )


def _assert_schema_instance(
    schema: dict[str, object],
    value: object,
    path: str = "$",
) -> None:
    """Validate the JSON-Schema subset emitted by this dependency-light package."""

    if "oneOf" in schema:
        matches = 0
        for candidate in cast(list[dict[str, object]], schema["oneOf"]):
            try:
                _assert_schema_instance(candidate, value, path)
            except AssertionError:
                continue
            matches += 1
        if matches != 1:
            raise AssertionError(f"{path}: expected exactly one schema match, got {matches}")
        return
    if "const" in schema and value != schema["const"]:
        raise AssertionError(f"{path}: {value!r} != const {schema['const']!r}")
    if "enum" in schema and value not in cast(list[object], schema["enum"]):
        raise AssertionError(f"{path}: {value!r} is not an allowed enum value")

    expected_type = schema.get("type")
    if expected_type == "object":
        if not isinstance(value, dict):
            raise AssertionError(f"{path}: expected object")
        properties = cast(dict[str, dict[str, object]], schema.get("properties", {}))
        required = cast(list[str], schema.get("required", []))
        missing = set(required) - set(value)
        if missing:
            raise AssertionError(f"{path}: missing required fields {sorted(missing)}")
        if schema.get("additionalProperties") is False:
            extra = set(value) - set(properties)
            if extra:
                raise AssertionError(f"{path}: unexpected fields {sorted(extra)}")
        for key, item in value.items():
            if key in properties:
                _assert_schema_instance(properties[key], item, f"{path}.{key}")
    elif expected_type == "array":
        if not isinstance(value, list):
            raise AssertionError(f"{path}: expected array")
        if len(value) < cast(int, schema.get("minItems", 0)):
            raise AssertionError(f"{path}: array is too short")
        if schema.get("uniqueItems"):
            encoded = [json.dumps(item, sort_keys=True) for item in value]
            if len(encoded) != len(set(encoded)):
                raise AssertionError(f"{path}: array items are not unique")
        item_schema = cast(dict[str, object] | None, schema.get("items"))
        if item_schema is not None:
            for index, item in enumerate(value):
                _assert_schema_instance(item_schema, item, f"{path}[{index}]")
    elif expected_type == "string":
        if not isinstance(value, str):
            raise AssertionError(f"{path}: expected string")
        if len(value) < cast(int, schema.get("minLength", 0)):
            raise AssertionError(f"{path}: string is too short")
        pattern = schema.get("pattern")
        if pattern is not None and re.search(cast(str, pattern), value) is None:
            raise AssertionError(f"{path}: string does not match {pattern!r}")
    elif expected_type == "integer":
        if not isinstance(value, int) or isinstance(value, bool):
            raise AssertionError(f"{path}: expected integer")
        if "minimum" in schema and value < cast(int, schema["minimum"]):
            raise AssertionError(f"{path}: integer is below its minimum")
        if "maximum" in schema and value > cast(int, schema["maximum"]):
            raise AssertionError(f"{path}: integer is above its maximum")
    elif expected_type == "boolean" and not isinstance(value, bool):
        raise AssertionError(f"{path}: expected boolean")


def _identity() -> PlannerIdentity:
    return PlannerIdentity(
        adapter_id="kapsl-vllm-connector",
        adapter_version=ADAPTER_VERSION,
        backend_version="0.test",
        profile_id=ADAPTER_PROFILE_ID,
    )


def _resolved_geometry() -> GeometryDescriptor:
    full = SyntheticSpec(
        block_size=16,
        page_size_bytes=1024,
        maximum_pages=64,
    )
    sliding = SyntheticSpec(
        block_size=32,
        page_size_bytes=4096,
        maximum_pages=8,
        head_size=8,
        sliding_window=256,
    )
    config = SimpleNamespace(
        num_blocks=256,
        kv_cache_layout="BHLNC",
        kv_cache_groups=[
            SimpleNamespace(
                layer_names=["model.layers.0.attn", "model.layers.1.attn"],
                kv_cache_spec=full,
            ),
            SimpleNamespace(
                layer_names=["model.layers.2.attn"],
                kv_cache_spec=sliding,
            ),
        ],
        # The largest group occupies 4096 bytes for each shared pool block.
        kv_cache_tensors=[
            SimpleNamespace(size=256 * 4096),
            SimpleNamespace(size=256 * 4096),
        ],
    )
    vllm_config = _vllm_config()
    return geometry_from_resolved_configs(
        [config],
        vllm_config,
        identity=_identity(),
        model_fingerprint="sha256:model",
        max_model_len=1024,
        attention_backend="FLASH_ATTN",
        layout_id="BHLNC",
        device_ids=[0],
        spec_kind_classifier=_synthetic_spec_classifier,
    )


class PlanningTests(unittest.TestCase):
    def test_elastic_sizing_separates_initial_and_virtual_capacity(self) -> None:
        geometry = _resolved_geometry()
        result = build_plan(
            geometry,
            SizingPolicy(
                target_concurrency=2,
                maximum_concurrency=6,
                headroom_percent=0,
                alignment_blocks=2,
            ),
        )
        sizing = result.ranks[0]
        self.assertGreater(sizing.maximum_blocks, sizing.desired_blocks)
        self.assertEqual(
            sizing.maximum_bytes,
            sizing.maximum_blocks * sizing.bytes_per_block,
        )
        self.assertEqual(
            result.to_dict()["sizing"]["total_maximum_bytes"],
            sizing.maximum_bytes,
        )

    def _executor_planner_fixture(
        self,
        *,
        tensor_parallel_size: int = 2,
        malformed_specs: object | None = None,
        resolve_error: Exception | None = None,
    ) -> tuple[
        RuntimePlanningRequest,
        _RuntimePlannerApis,
        dict[str, object],
    ]:
        calls: dict[str, object] = {
            "engine_args": None,
            "resolved_config": None,
            "registered": 0,
            "supported_layouts": 0,
            "published_layouts": [],
            "available_memory": None,
            "shutdown": 0,
            "initialize": 0,
            "profile": 0,
        }
        resolved = _vllm_config()
        resolved.parallel_config.tensor_parallel_size = tensor_parallel_size
        spec = SyntheticSpec(
            block_size=16,
            page_size_bytes=1024,
            maximum_pages=64,
        )
        worker_specs = [
            {
                "model.layers.0.attn": spec,
                "model.layers.1.attn": spec,
            }
            for _ in range(tensor_parallel_size)
        ]
        group = SimpleNamespace(
            layer_names=list(worker_specs[0]),
            kv_cache_spec=spec,
        )

        class FakeEngineArgs:
            def __init__(self, **kwargs: object) -> None:
                calls["engine_args"] = kwargs

            def create_engine_config(self) -> SimpleNamespace:
                calls["resolved_config"] = resolved
                return resolved

        class FakeExecutor:
            def __init__(self, config: object) -> None:
                self.config = config

            def get_kv_cache_specs(self) -> object:
                return worker_specs if malformed_specs is None else malformed_specs

            def get_supported_kv_cache_layouts(self) -> list[list[str]]:
                calls["supported_layouts"] = cast(
                    int, calls["supported_layouts"]
                ) + 1
                return [["BHLNC"] for _ in range(tensor_parallel_size)]

            def set_kv_cache_layout(self, layout: str) -> None:
                cast(list[str], calls["published_layouts"]).append(layout)

            def determine_available_memory(self) -> list[int]:
                calls["profile"] = cast(int, calls["profile"]) + 1
                raise AssertionError("planner crossed the memory-profile boundary")

            def initialize_from_config(self, _configs: object) -> None:
                calls["initialize"] = cast(int, calls["initialize"]) + 1
                raise AssertionError("planner crossed the physical KV boundary")

            def shutdown(self) -> None:
                calls["shutdown"] = cast(int, calls["shutdown"]) + 1

        def resolve_layout(
            config: object,
            supported: object,
            specs: object,
        ) -> SimpleNamespace:
            self.assertIs(config, resolved)
            self.assertEqual(
                supported,
                [["BHLNC"] for _ in range(tensor_parallel_size)],
            )
            self.assertEqual(len(cast(list[object], specs)), 2 * tensor_parallel_size)
            if resolve_error is not None:
                raise resolve_error
            return SimpleNamespace(name="BHLNC")

        def get_groups(
            config: object, specs: dict[str, object]
        ) -> list[SimpleNamespace]:
            self.assertIs(config, resolved)
            self.assertEqual(set(specs), set(worker_specs[0]))
            return [group]

        def get_configs(
            config: object,
            specs: list[dict[str, object]],
            available_memory: list[int],
        ) -> list[SimpleNamespace]:
            self.assertIs(config, resolved)
            self.assertIs(specs, worker_specs)
            # One full sequence consumes 64 shared blocks and vLLM retains one
            # additional null block. The physical stride is two 1024-byte pages.
            self.assertEqual(
                available_memory,
                [65 * 2048 for _ in range(tensor_parallel_size)],
            )
            calls["available_memory"] = list(available_memory)
            return [
                SimpleNamespace(
                    num_blocks=65,
                    kv_cache_groups=[group],
                    kv_cache_tensors=[SimpleNamespace(size=65 * 2048)],
                    kv_cache_layout=None,
                )
                for _ in range(tensor_parallel_size)
            ]

        def register(config: object) -> None:
            self.assertIs(config, resolved)
            calls["registered"] = cast(int, calls["registered"]) + 1

        request = RuntimePlanningRequest(
            model_path=Path("/models/test"),
            model_fingerprint="sha256:model",
            max_model_len=1024,
            tensor_parallel_size=tensor_parallel_size,
            attention_backend="FLASH_ATTN",
            device_ids=tuple(range(4, 4 + tensor_parallel_size)),
        )
        apis = _RuntimePlannerApis(
            engine_args_factory=FakeEngineArgs,
            executor_class_resolver=lambda config: (
                self.assertIs(config, resolved) or FakeExecutor
            ),
            register_all_kvcache_specs=register,
            resolve_kv_cache_layout=resolve_layout,
            get_kv_cache_groups=get_groups,
            get_kv_cache_configs=get_configs,
            pool_bytes_per_block=lambda groups: (
                self.assertEqual(groups, [group]) or 2048
            ),
            max_memory_usage_bytes_from_groups=lambda config, groups: (
                self.assertIs(config, resolved)
                or self.assertEqual(groups, [group])
                or 64 * 2048
            ),
            spec_kind_classifier=_synthetic_spec_classifier,
        )
        return request, apis, calls

    def test_executor_planner_stops_before_physical_kv_allocation(self) -> None:
        request, apis, calls = self._executor_planner_fixture()

        geometry = _geometry_from_executor(request, "0.test", apis)

        self.assertEqual(
            calls["engine_args"],
            {
                "model": str(request.model_path),
                "max_model_len": 1024,
                "tensor_parallel_size": 2,
                "attention_backend": "FLASH_ATTN",
                "enforce_eager": True,
                # Geometry planning must not inherit vLLM's card-sized 0.92
                # default when another admitted replica is already resident.
                "gpu_memory_utilization": 1e-9,
            },
        )
        self.assertEqual(calls["registered"], 1)
        self.assertEqual(calls["supported_layouts"], 1)
        self.assertEqual(calls["published_layouts"], ["BHLNC"])
        self.assertEqual(calls["shutdown"], 1)
        self.assertEqual(calls["profile"], 0)
        self.assertEqual(calls["initialize"], 0)
        self.assertEqual(geometry.identity.backend_version, "0.test")
        self.assertEqual(geometry.model_fingerprint, "sha256:model")
        self.assertEqual(geometry.max_model_len, 1024)
        self.assertEqual(geometry.tensor_parallel_size, 2)
        self.assertEqual([rank.device_id for rank in geometry.ranks], [4, 5])
        self.assertEqual(geometry.layout_id, "BHLNC")
        self.assertEqual(geometry.ranks[0].pool_bytes_per_block, 2048)
        self.assertEqual(geometry.ranks[0].required_blocks_per_sequence, 64)

    def test_executor_planner_shuts_down_on_resolved_geometry_failure(self) -> None:
        request, apis, calls = self._executor_planner_fixture(
            resolve_error=RuntimeError("layout unavailable")
        )

        with self.assertRaisesRegex(RuntimeError, "layout unavailable"):
            _geometry_from_executor(request, "0.test", apis)

        self.assertEqual(calls["shutdown"], 1)
        self.assertEqual(calls["initialize"], 0)

    def test_elastic_executor_planner_disables_overlapping_batches(self) -> None:
        request, apis, calls = self._executor_planner_fixture(
            resolve_error=RuntimeError("stop after engine configuration")
        )
        request = replace(request, live_resize=True)

        with self.assertRaisesRegex(RuntimeError, "stop after engine configuration"):
            _geometry_from_executor(request, "0.test", apis)

        engine_args = cast(dict[str, object], calls["engine_args"])
        self.assertIs(engine_args["async_scheduling"], False)
        resolved = cast(SimpleNamespace, calls["resolved_config"])
        self.assertEqual(resolved.cache_config.kv_cache_layout, "BLNHC")
        self.assertEqual(calls["shutdown"], 1)

    def test_executor_planner_rejects_incomplete_worker_specs_and_shuts_down(
        self,
    ) -> None:
        request, apis, calls = self._executor_planner_fixture(
            malformed_specs=[{"only.rank": object()}]
        )

        with self.assertRaisesRegex(RuntimeGeometryUnavailable, "every tensor"):
            _geometry_from_executor(request, "0.test", apis)

        self.assertEqual(calls["shutdown"], 1)
        self.assertEqual(calls["supported_layouts"], 0)
        self.assertEqual(calls["initialize"], 0)

    def test_resolved_hybrid_groups_use_vllm_per_group_requirements(self) -> None:
        geometry = _resolved_geometry()
        rank = geometry.ranks[0]

        self.assertEqual(rank.pool_bytes_per_block, 4096)
        self.assertEqual(geometry.total_pool_bytes_per_block, 4096)
        self.assertEqual(rank.groups[0].bytes_per_group_block, 2048)
        self.assertEqual(rank.groups[0].required_blocks_per_sequence, 64)
        self.assertEqual(rank.groups[1].bytes_per_group_block, 4096)
        self.assertEqual(rank.groups[1].required_blocks_per_sequence, 8)
        # Hybrid groups consume independent block-table entries from one pool.
        self.assertEqual(rank.required_blocks_per_sequence, 72)
        self.assertEqual(rank.fixed_overhead_blocks, 1)

    def test_final_geometry_stays_in_parity_with_connector_helpers(self) -> None:
        geometry = _resolved_geometry()
        config = SimpleNamespace(
            num_blocks=256,
            kv_cache_layout="BHLNC",
            kv_cache_groups=[],
            kv_cache_tensors=[SimpleNamespace(size=256 * 4096)],
        )
        for group in geometry.ranks[0].groups:
            config.kv_cache_groups.append(
                SimpleNamespace(
                    layer_names=list(group.layers),
                    kv_cache_spec=SyntheticSpec(
                        block_size=group.block_size_tokens,
                        page_size_bytes=(
                            group.bytes_per_group_block // len(group.layers)
                        ),
                        maximum_pages=group.required_blocks_per_sequence,
                        num_kv_heads=group.kv_heads,
                        head_size=group.key_head_dim,
                        head_size_v=group.value_head_dim,
                        sliding_window=group.window_tokens,
                        spec_kind=group.policy_kind,
                    ),
                )
            )
        capacity = _vllm_capacity_groups(
            config,
            [{"kind": "cuda", "device_id": 0}],
            shared_pool=True,
            spec_kind_classifier=_synthetic_spec_classifier,
        )
        topology = _vllm_topology(
            config,
            "sha256:model",
            spec_kind_classifier=_synthetic_spec_classifier,
        )

        self.assertEqual(
            {entry["allocation_granularity_tokens"] for entry in capacity},
            {group.block_size_tokens for group in geometry.ranks[0].groups},
        )
        self.assertEqual(
            {entry["bytes_per_allocation"] for entry in capacity},
            {geometry.ranks[0].pool_bytes_per_block},
        )
        self.assertEqual(
            [group["group_id"] for group in topology["cache_groups"]],
            [group.group_id for group in geometry.ranks[0].groups],
        )
        for planned, registered in zip(
            geometry.ranks[0].groups,
            topology["cache_groups"],
            strict=True,
        ):
            registered_geometry = registered["geometry"]
            self.assertEqual(
                [layer["name"] for layer in registered["layers"]],
                list(planned.layers),
            )
            self.assertEqual(
                (
                    registered_geometry["block_size_tokens"],
                    registered_geometry["kv_heads"],
                    registered_geometry["key_head_dim"],
                    registered_geometry["value_head_dim"],
                ),
                (
                    planned.block_size_tokens,
                    planned.kv_heads,
                    planned.key_head_dim,
                    planned.value_head_dim,
                ),
            )
            self.assertEqual(registered["policy"]["kind"], planned.policy_kind)

    def test_connector_rejects_packed_stride_that_disagrees_with_planner(self) -> None:
        config = SimpleNamespace(
            num_blocks=256,
            kv_cache_layout="BHLNC",
            kv_cache_groups=[
                SimpleNamespace(
                    layer_names=["layer.0", "layer.1"],
                    kv_cache_spec=SyntheticSpec(
                        block_size=16,
                        page_size_bytes=1024,
                        maximum_pages=64,
                    ),
                )
            ],
            # Certified group stride is 2048 bytes.  The physical placement
            # must not be accepted merely because it is block-integral.
            kv_cache_tensors=[SimpleNamespace(size=256 * 4097)],
        )

        with self.assertRaisesRegex(ValueError, "packed allocation block stride"):
            _vllm_capacity_groups(
                config,
                [{"kind": "cuda", "device_id": 0}],
                shared_pool=True,
                spec_kind_classifier=_synthetic_spec_classifier,
            )

    def test_rank_geometry_rejects_stride_slack_and_duplicate_layers(self) -> None:
        rank = _resolved_geometry().ranks[0]
        with self.assertRaisesRegex(PlanningError, "largest cache-group stride"):
            replace(
                rank,
                pool_bytes_per_block=rank.pool_bytes_per_block + 1,
            )

        duplicate_group = replace(
            rank.groups[1],
            layers=(rank.groups[0].layers[0],),
        )
        with self.assertRaisesRegex(PlanningError, "globally unique"):
            replace(rank, groups=(rank.groups[0], duplicate_group))

    def test_planner_and_connector_reject_mixed_uniform_member_geometry(self) -> None:
        cases = (
            ("head_size", 128, "mixed key head dimensions"),
            ("page_size_bytes", 2048, "dense attention geometry"),
            ("dtype", "bfloat16", "mixed cache element types"),
            ("num_head_slots", 8, "mixed packed head-slot counts"),
        )
        for field, value, message in cases:
            members = {
                "layer.0": SyntheticSpec(
                    block_size=16,
                    page_size_bytes=1024,
                    maximum_pages=64,
                ),
                "layer.1": SyntheticSpec(
                    block_size=16,
                    page_size_bytes=1024,
                    maximum_pages=64,
                ),
            }
            if field == "num_head_slots":
                members["layer.0"].num_head_slots = 4
            setattr(members["layer.1"], field, value)
            config = SimpleNamespace(
                kv_cache_layout="BHLNC",
                kv_cache_groups=[
                    SimpleNamespace(
                        layer_names=list(members),
                        kv_cache_spec=SyntheticUniformSpec(members),
                    )
                ],
                kv_cache_tensors=[],
            )
            planner_arguments = dict(
                identity=_identity(),
                model_fingerprint="sha256:model",
                max_model_len=1024,
                attention_backend="FLASH_ATTN",
                layout_id="BHLNC",
                device_ids=[0],
                spec_kind_classifier=_synthetic_spec_classifier,
            )
            with self.subTest(field=field, path="planner"), self.assertRaisesRegex(
                PlanningError, message
            ):
                geometry_from_resolved_configs(
                    [config],
                    _vllm_config(),
                    **planner_arguments,
                )
            with self.subTest(field=field, path="connector"), self.assertRaisesRegex(
                PlanningError, message
            ):
                _vllm_topology(
                    config,
                    "sha256:model",
                    spec_kind_classifier=_synthetic_spec_classifier,
                )

    def test_planner_and_connector_require_layout_and_unique_layers(self) -> None:
        duplicated = SimpleNamespace(
            kv_cache_layout="BHLNC",
            kv_cache_groups=[
                SimpleNamespace(
                    layer_names=["layer.0"],
                    kv_cache_spec=SyntheticSpec(
                        block_size=16,
                        page_size_bytes=1024,
                        maximum_pages=64,
                    ),
                ),
                SimpleNamespace(
                    layer_names=["layer.0"],
                    kv_cache_spec=SyntheticSpec(
                        block_size=16,
                        page_size_bytes=1024,
                        maximum_pages=64,
                    ),
                ),
            ],
            kv_cache_tensors=[],
        )
        planner_arguments = dict(
            identity=_identity(),
            model_fingerprint="sha256:model",
            max_model_len=1024,
            attention_backend="FLASH_ATTN",
            layout_id="BHLNC",
            device_ids=[0],
            spec_kind_classifier=_synthetic_spec_classifier,
        )
        for path, operation in (
            (
                "planner",
                lambda config: geometry_from_resolved_configs(
                    [config], _vllm_config(), **planner_arguments
                ),
            ),
            (
                "connector",
                lambda config: _vllm_topology(
                    config,
                    "sha256:model",
                    spec_kind_classifier=_synthetic_spec_classifier,
                ),
            ),
        ):
            with self.subTest(path=path, invariant="layers"), self.assertRaisesRegex(
                PlanningError, "globally unique"
            ):
                operation(duplicated)

            missing_layout = SimpleNamespace(
                kv_cache_groups=duplicated.kv_cache_groups,
                kv_cache_tensors=[],
            )
            with self.subTest(path=path, invariant="layout"), self.assertRaisesRegex(
                ValueError, "must expose kv_cache_layout"
            ):
                operation(missing_layout)

    def test_sizing_adds_null_block_prefix_headroom_and_alignment(self) -> None:
        result = build_plan(
            _resolved_geometry(),
            SizingPolicy(
                target_concurrency=2,
                headroom_percent=20,
                prefix_blocks=4,
                alignment_blocks=8,
            ),
        )
        sizing = result.ranks[0]

        # Minimum: round_up(1 null + 72 sequence blocks, 8).
        self.assertEqual(sizing.minimum_blocks, 80)
        self.assertEqual(sizing.minimum_bytes, 80 * 4096)
        # Base: 1 + 2*72 + 4 = 149. Headroom: ceil((144+4)*.20)=30.
        self.assertEqual(sizing.base_blocks, 149)
        self.assertEqual(sizing.headroom_blocks, 30)
        self.assertEqual(sizing.desired_blocks, 184)
        self.assertEqual(sizing.desired_bytes, 184 * 4096)
        self.assertEqual(sizing.effective_target_concurrency, 2)

    def test_max_byte_cap_reduces_concurrency_but_not_below_one_sequence(self) -> None:
        geometry = _resolved_geometry()
        result = build_plan(
            geometry,
            SizingPolicy(
                target_concurrency=2,
                headroom_percent=20,
                prefix_blocks=4,
                alignment_blocks=8,
                max_bytes=100 * 4096,
            ),
        )
        self.assertEqual(result.ranks[0].desired_blocks, 96)
        self.assertEqual(result.ranks[0].effective_target_concurrency, 1)
        self.assertIs(result.ranks[0].concurrency_reduced, True)

        with self.assertRaisesRegex(PlanningError, "strict mode"):
            build_plan(
                geometry,
                SizingPolicy(
                    target_concurrency=2,
                    alignment_blocks=8,
                    max_bytes=100 * 4096,
                    strict_concurrency=True,
                ),
            )

        with self.assertRaisesRegex(PlanningError, "one max_model_len"):
            build_plan(
                geometry,
                SizingPolicy(
                    target_concurrency=1,
                    alignment_blocks=8,
                    max_bytes=79 * 4096,
                ),
            )

        with self.assertRaisesRegex(PlanningError, "conflict after block alignment"):
            build_plan(
                geometry,
                SizingPolicy(
                    target_concurrency=1,
                    alignment_blocks=8,
                    min_bytes=99 * 4096,
                    max_bytes=100 * 4096,
                ),
            )

        with self.assertRaisesRegex(PlanningError, "below the one max_model_len"):
            build_plan(
                geometry,
                SizingPolicy(
                    target_concurrency=1,
                    alignment_blocks=8,
                    min_bytes=79 * 4096,
                ),
            )

    def test_cap_sheds_optional_prefix_and_headroom_before_concurrency(self) -> None:
        result = build_plan(
            _resolved_geometry(),
            SizingPolicy(
                target_concurrency=2,
                prefix_blocks=10_000,
                headroom_percent=100,
                alignment_blocks=8,
                # Exactly the aligned one-sequence minimum.
                max_bytes=80 * 4096,
            ),
        )

        sizing = result.ranks[0]
        self.assertEqual(sizing.desired_blocks, sizing.minimum_blocks)
        self.assertEqual(sizing.effective_target_concurrency, 1)
        self.assertIs(sizing.concurrency_reduced, True)

    def test_checked_sizing_rejects_unsigned_overflow(self) -> None:
        with self.assertRaisesRegex(PlanningError, "overflows"):
            build_plan(
                _resolved_geometry(),
                SizingPolicy(target_concurrency=UINT64_MAX),
            )

    def test_geometry_digest_is_canonical_and_binds_model_identity(self) -> None:
        first = _resolved_geometry()
        second = _resolved_geometry()
        self.assertEqual(first.geometry_digest(), second.geometry_digest())
        self.assertRegex(first.geometry_digest(), r"^sha256:[0-9a-f]{64}$")

        changed = GeometryDescriptor(
            identity=first.identity,
            model_fingerprint="sha256:different",
            max_model_len=first.max_model_len,
            tensor_parallel_size=first.tensor_parallel_size,
            attention_backend=first.attention_backend,
            layout_id=first.layout_id,
            ranks=first.ranks,
        )
        self.assertNotEqual(first.geometry_digest(), changed.geometry_digest())

        rank = first.ranks[0]
        reordered_rank = RankGeometry(
            rank=rank.rank,
            device_id=rank.device_id,
            pool_bytes_per_block=rank.pool_bytes_per_block,
            groups=tuple(reversed(rank.groups)),
            fixed_overhead_blocks=rank.fixed_overhead_blocks,
        )
        reordered = GeometryDescriptor(
            identity=first.identity,
            model_fingerprint=first.model_fingerprint,
            max_model_len=first.max_model_len,
            tensor_parallel_size=first.tensor_parallel_size,
            attention_backend=first.attention_backend,
            layout_id=first.layout_id,
            ranks=(reordered_rank,),
        )
        self.assertEqual(first.geometry_digest(), reordered.geometry_digest())

        normalized = GeometryDescriptor(
            identity=first.identity,
            model_fingerprint=first.model_fingerprint,
            max_model_len=first.max_model_len,
            tensor_parallel_size=first.tensor_parallel_size,
            attention_backend="flash_attn",
            layout_id=first.layout_id,
            ranks=first.ranks,
        )
        self.assertEqual(normalized.attention_backend, "FLASH_ATTN")
        self.assertEqual(first.geometry_digest(), normalized.geometry_digest())

    def test_uniform_group_page_accounting_is_not_multiplied_twice(self) -> None:
        members = {
            "layer.0": SyntheticSpec(
                block_size=16, page_size_bytes=1024, maximum_pages=64
            ),
            "layer.1": SyntheticSpec(
                block_size=16, page_size_bytes=1024, maximum_pages=64
            ),
        }
        uniform = SyntheticUniformSpec(members)
        config = SimpleNamespace(
            kv_cache_layout="BHLNC",
            kv_cache_groups=[
                SimpleNamespace(layer_names=list(members), kv_cache_spec=uniform)
            ],
            kv_cache_tensors=[],
        )
        rank = geometry_from_resolved_configs(
            [config],
            _vllm_config(),
            identity=_identity(),
            model_fingerprint="sha256:model",
            max_model_len=1024,
            attention_backend="FLASH_ATTN",
            layout_id="BHLNC",
            device_ids=[0],
            spec_kind_classifier=_synthetic_spec_classifier,
        ).ranks[0]

        self.assertEqual(rank.groups[0].bytes_per_group_block, 2048)
        self.assertEqual(rank.groups[0].required_blocks_per_sequence, 64)

    def test_extraction_fails_on_packed_or_dtype_uncertainty(self) -> None:
        bad_packed = SimpleNamespace(
            num_blocks=4,
            kv_cache_layout="BHLNC",
            kv_cache_groups=[
                SimpleNamespace(
                    layer_names=["layer.0"],
                    kv_cache_spec=SyntheticSpec(
                        block_size=16,
                        page_size_bytes=1024,
                        maximum_pages=4,
                    ),
                )
            ],
            kv_cache_tensors=[SimpleNamespace(size=4095)],
        )
        kwargs = dict(
            identity=_identity(),
            model_fingerprint="sha256:model",
            max_model_len=1024,
            attention_backend="FLASH_ATTN",
            layout_id="BHLNC",
            device_ids=[0],
        )
        vllm_config = _vllm_config()
        with self.assertRaisesRegex(PlanningError, "packed allocation"):
            geometry_from_resolved_configs(
                [bad_packed],
                vllm_config,
                spec_kind_classifier=_synthetic_spec_classifier,
                **kwargs,
            )

        bad_dense_page = SimpleNamespace(
            kv_cache_layout="BHLNC",
            kv_cache_groups=[
                SimpleNamespace(
                    layer_names=["layer.0"],
                    kv_cache_spec=SyntheticSpec(
                        block_size=16,
                        page_size_bytes=1025,
                        maximum_pages=4,
                    ),
                )
            ],
            kv_cache_tensors=[],
        )
        with self.assertRaisesRegex(PlanningError, "dense attention geometry"):
            geometry_from_resolved_configs(
                [bad_dense_page],
                vllm_config,
                spec_kind_classifier=_synthetic_spec_classifier,
                **kwargs,
            )
        with self.assertRaisesRegex(PlanningError, "dense attention geometry"):
            _vllm_topology(
                bad_dense_page,
                "sha256:model",
                spec_kind_classifier=_synthetic_spec_classifier,
            )

        bad_packed.kv_cache_tensors = []
        bad_packed.kv_cache_groups[0].kv_cache_spec.dtype = "vendor_float6"
        with self.assertRaisesRegex(PlanningError, "not certified"):
            geometry_from_resolved_configs(
                [bad_packed],
                vllm_config,
                spec_kind_classifier=_synthetic_spec_classifier,
                **kwargs,
            )

        bad_packed.kv_cache_groups[0].kv_cache_spec.dtype = "float16"
        bad_packed.kv_cache_groups[0].kv_cache_spec.synthetic_spec_kind = (
            "cross_attention"
        )
        with self.assertRaisesRegex(PlanningError, "spec kind"):
            geometry_from_resolved_configs(
                [bad_packed],
                vllm_config,
                spec_kind_classifier=_synthetic_spec_classifier,
                **kwargs,
            )

        bad_packed.kv_cache_groups[0].kv_cache_spec.synthetic_spec_kind = (
            "full_attention"
        )
        bad_packed.kv_cache_groups[0].kv_cache_spec.page_size_padded = 1024
        with self.assertRaisesRegex(PlanningError, "padded KV pages"):
            geometry_from_resolved_configs(
                [bad_packed],
                vllm_config,
                spec_kind_classifier=_synthetic_spec_classifier,
                **kwargs,
            )

    def test_planner_and_connector_share_allocation_kind_classification(self) -> None:
        # vLLM can promote a sliding-attention layer to FullAttentionSpec when
        # hybrid allocation is disabled while retaining its execution window.
        # Allocation topology must follow the resolved spec kind, not the
        # presence of the sliding_window attribute.
        promoted = SyntheticSpec(
            block_size=16,
            page_size_bytes=1024,
            maximum_pages=64,
            sliding_window=256,
            spec_kind="full_attention",
        )
        config = SimpleNamespace(
            num_blocks=80,
            kv_cache_layout="BHLNC",
            kv_cache_groups=[
                SimpleNamespace(layer_names=["layer.0"], kv_cache_spec=promoted)
            ],
            kv_cache_tensors=[SimpleNamespace(size=80 * 1024)],
        )
        geometry = geometry_from_resolved_configs(
            [config],
            _vllm_config(),
            identity=_identity(),
            model_fingerprint="sha256:model",
            max_model_len=1024,
            attention_backend="FLASH_ATTN",
            layout_id="BHLNC",
            device_ids=[0],
            spec_kind_classifier=_synthetic_spec_classifier,
        )
        topology = _vllm_topology(
            config,
            "sha256:model",
            spec_kind_classifier=_synthetic_spec_classifier,
        )

        self.assertEqual(geometry.ranks[0].groups[0].policy_kind, "full_attention")
        self.assertEqual(
            topology["cache_groups"][0]["policy"],
            {"kind": "full_attention"},
        )

        # A magic attribute cannot override the explicit/pinned classifier.
        promoted.kapsl_spec_kind = "full_attention"
        with self.assertRaisesRegex(PlanningError, "spec kind"):
            geometry_from_resolved_configs(
                [config],
                _vllm_config(),
                identity=_identity(),
                model_fingerprint="sha256:model",
                max_model_len=1024,
                attention_backend="FLASH_ATTN",
                layout_id="BHLNC",
                device_ids=[0],
                spec_kind_classifier=lambda _spec: "cross_attention",
            )

    def test_planner_and_connector_reject_unrepresented_retained_tokens(self) -> None:
        retained = SyntheticSpec(
            block_size=16,
            page_size_bytes=1024,
            maximum_pages=64,
            sliding_window=256,
            extra_retained_tokens=8,
        )
        config = SimpleNamespace(
            num_blocks=80,
            kv_cache_layout="BHLNC",
            kv_cache_groups=[
                SimpleNamespace(layer_names=["layer.0"], kv_cache_spec=retained)
            ],
            kv_cache_tensors=[SimpleNamespace(size=80 * 1024)],
        )

        with self.assertRaisesRegex(PlanningError, "extra retained tokens"):
            geometry_from_resolved_configs(
                [config],
                _vllm_config(),
                identity=_identity(),
                model_fingerprint="sha256:model",
                max_model_len=1024,
                attention_backend="FLASH_ATTN",
                layout_id="BHLNC",
                device_ids=[0],
                spec_kind_classifier=_synthetic_spec_classifier,
            )
        with self.assertRaisesRegex(PlanningError, "extra retained tokens"):
            _vllm_topology(
                config,
                "sha256:model",
                spec_kind_classifier=_synthetic_spec_classifier,
            )

    def test_resolved_geometry_rejects_nonintegral_numeric_fields(self) -> None:
        for field, value in (("block_size", True), ("page_size_bytes", 1024.5)):
            spec = SyntheticSpec(
                block_size=16,
                page_size_bytes=1024,
                maximum_pages=64,
            )
            setattr(spec, field, value)
            config = SimpleNamespace(
                kv_cache_layout="BHLNC",
                kv_cache_groups=[
                    SimpleNamespace(layer_names=["layer.0"], kv_cache_spec=spec)
                ],
                kv_cache_tensors=[],
            )
            with self.subTest(field=field, value=value), self.assertRaisesRegex(
                PlanningError, "integer"
            ):
                geometry_from_resolved_configs(
                    [config],
                    _vllm_config(),
                    identity=_identity(),
                    model_fingerprint="sha256:model",
                    max_model_len=1024,
                    attention_backend="FLASH_ATTN",
                    layout_id="BHLNC",
                    device_ids=[0],
                    spec_kind_classifier=_synthetic_spec_classifier,
                )

        with self.assertRaisesRegex(PlanningError, "device_id must be an integer"):
            geometry_from_resolved_configs(
                [SimpleNamespace(
                    kv_cache_layout="BHLNC",
                    kv_cache_groups=[SimpleNamespace(
                        layer_names=["layer.0"],
                        kv_cache_spec=SyntheticSpec(
                            block_size=16,
                            page_size_bytes=1024,
                            maximum_pages=64,
                        ),
                    )],
                    kv_cache_tensors=[],
                )],
                _vllm_config(),
                identity=_identity(),
                model_fingerprint="sha256:model",
                max_model_len=1024,
                attention_backend="FLASH_ATTN",
                layout_id="BHLNC",
                device_ids=[True],
                spec_kind_classifier=_synthetic_spec_classifier,
            )

    def test_planning_applies_the_connector_certified_profile_constraints(self) -> None:
        config = SimpleNamespace(
            kv_cache_layout="BHLNC",
            kv_cache_groups=[
                SimpleNamespace(
                    layer_names=["layer.0"],
                    kv_cache_spec=SyntheticSpec(
                        block_size=16,
                        page_size_bytes=1024,
                        maximum_pages=64,
                    ),
                )
            ],
            kv_cache_tensors=[],
        )
        kwargs = dict(
            identity=_identity(),
            model_fingerprint="sha256:model",
            max_model_len=1024,
            attention_backend="FLASH_ATTN",
            layout_id="BHLNC",
            device_ids=[0],
            spec_kind_classifier=_synthetic_spec_classifier,
        )
        for resolved, message in (
            (_vllm_config(attention_backend="FLASHINFER"), "FLASH_ATTN"),
            (_vllm_config(pipeline_parallel_size=2), "pipeline_parallel_size"),
        ):
            with self.subTest(message=message), self.assertRaisesRegex(
                PlanningError, message
            ):
                geometry_from_resolved_configs([config], resolved, **kwargs)

    def test_resolved_geometry_binds_max_length_and_tensor_parallel_world(self) -> None:
        config = SimpleNamespace(
            kv_cache_layout="BHLNC",
            kv_cache_groups=[
                SimpleNamespace(
                    layer_names=["layer.0"],
                    kv_cache_spec=SyntheticSpec(
                        block_size=16,
                        page_size_bytes=1024,
                        maximum_pages=64,
                    ),
                )
            ],
            kv_cache_tensors=[],
        )
        kwargs = dict(
            identity=_identity(),
            model_fingerprint="sha256:model",
            max_model_len=1024,
            attention_backend="FLASH_ATTN",
            layout_id="BHLNC",
            device_ids=[0],
            spec_kind_classifier=_synthetic_spec_classifier,
        )

        missing_max_len = _vllm_config()
        del missing_max_len.model_config.max_model_len
        with self.assertRaisesRegex(PlanningError, "expose max_model_len"):
            geometry_from_resolved_configs([config], missing_max_len, **kwargs)

        missing_tp = _vllm_config()
        del missing_tp.parallel_config.tensor_parallel_size
        with self.assertRaisesRegex(PlanningError, "expose tensor_parallel_size"):
            geometry_from_resolved_configs([config], missing_tp, **kwargs)

        wrong_tp = _vllm_config()
        wrong_tp.parallel_config.tensor_parallel_size = 2
        with self.assertRaisesRegex(PlanningError, "differs from the worker"):
            geometry_from_resolved_configs([config], wrong_tp, **kwargs)

    def test_published_schemas_validate_emitted_success_and_failure_payloads(self) -> None:
        arguments = [
            "--model",
            ".",
            "--model-fingerprint",
            "sha256:model",
            "--max-model-len",
            "1024",
            "--target-concurrency",
            "2",
        ]
        success_output = io.StringIO()
        with redirect_stdout(success_output):
            self.assertEqual(
                run(
                    arguments,
                    geometry_provider=lambda _request: _resolved_geometry(),
                    backend_version_provider=lambda: "0.test",
                ),
                0,
            )
        success = json.loads(success_output.getvalue())

        failure_output = io.StringIO()
        with redirect_stderr(failure_output):
            self.assertEqual(run([*arguments, "--headroom-percent", "101"]), 2)
        failure = json.loads(failure_output.getvalue())

        _assert_schema_instance(planner_json_schema(), success)
        _assert_schema_instance(planner_error_json_schema(), failure)

        drifted = json.loads(json.dumps(success))
        drifted["unexpected"] = True
        with self.assertRaisesRegex(AssertionError, "unexpected fields"):
            _assert_schema_instance(planner_json_schema(), drifted)

    def test_cli_emits_schema_success_and_fail_closed_error(self) -> None:
        schema_output = io.StringIO()
        with redirect_stdout(schema_output):
            self.assertEqual(run(["--print-schema"]), 0)
        schema = json.loads(schema_output.getvalue())
        self.assertEqual(schema["properties"]["schema_version"]["const"], 1)
        self.assertEqual(planner_json_schema()["$id"], schema["$id"])

        error_schema_output = io.StringIO()
        with redirect_stdout(error_schema_output):
            self.assertEqual(run(["--print-error-schema"]), 0)
        self.assertEqual(
            json.loads(error_schema_output.getvalue())["properties"]["status"]["const"],
            "error",
        )

        malformed_output = io.StringIO()
        with redirect_stderr(malformed_output):
            self.assertEqual(run(["--max-model-len", "not-an-int"]), 2)
        self.assertEqual(json.loads(malformed_output.getvalue())["status"], "error")

        with tempfile.TemporaryDirectory() as directory:
            arguments = [
                "--model",
                directory,
                "--model-fingerprint",
                "sha256:model",
                "--max-model-len",
                "1024",
                "--target-concurrency",
                "2",
                "--headroom-percent",
                "0",
            ]
            output = io.StringIO()
            with redirect_stdout(output):
                status = run(
                    arguments,
                    geometry_provider=lambda _request: _resolved_geometry(),
                    backend_version_provider=lambda: "0.test",
                )
            self.assertEqual(status, 0)
            plan = json.loads(output.getvalue())
            self.assertEqual(plan["schema_version"], PLANNER_SCHEMA_VERSION)
            self.assertEqual(plan["status"], "planned")
            self.assertIs(plan["supported"], True)

            error_output = io.StringIO()

            def unavailable(_request: object) -> GeometryDescriptor:
                raise PlanningError("resolved runtime geometry unavailable")

            with redirect_stderr(error_output):
                status = run(
                    arguments,
                    geometry_provider=unavailable,
                    backend_version_provider=lambda: "0.test",
                )
            self.assertEqual(status, 2)
            failure = json.loads(error_output.getvalue())
            self.assertEqual(failure["status"], "error")
            self.assertIs(failure["supported"], False)
            self.assertIn("unavailable", failure["error"]["message"])

            provider_error = io.StringIO()
            with redirect_stderr(provider_error):
                status = run(
                    arguments,
                    geometry_provider=lambda _request: (_ for _ in ()).throw(
                        RuntimeError("worker crashed")
                    ),
                    backend_version_provider=lambda: "0.test",
                )
            self.assertEqual(status, 2)
            self.assertEqual(
                json.loads(provider_error.getvalue())["error"]["kind"],
                "runtime_geometry_unavailable",
            )

    def test_cli_keeps_runtime_provider_logs_out_of_success_stdout(self) -> None:
        arguments = [
            "--model",
            ".",
            "--model-fingerprint",
            "sha256:model",
            "--max-model-len",
            "1024",
        ]
        stdout = io.StringIO()
        stderr = io.StringIO()

        def noisy_provider(_request: object) -> GeometryDescriptor:
            print("vLLM model-loading progress on stdout")
            return _resolved_geometry()

        with redirect_stdout(stdout), redirect_stderr(stderr):
            status = run(
                arguments,
                geometry_provider=noisy_provider,
                backend_version_provider=lambda: "0.test",
            )

        self.assertEqual(status, 0)
        self.assertEqual(json.loads(stdout.getvalue())["status"], "planned")
        self.assertNotIn("vLLM", stdout.getvalue())
        self.assertIn("vLLM model-loading progress", stderr.getvalue())

    def test_cli_redirects_provider_writes_through_retained_stdout_fd(self) -> None:
        arguments = [
            "--model",
            ".",
            "--model-fingerprint",
            "sha256:model",
            "--max-model-len",
            "1024",
        ]
        with (
            tempfile.TemporaryFile(mode="w+") as stdout,
            tempfile.TemporaryFile(mode="w+") as stderr,
            redirect_stdout(stdout),
            redirect_stderr(stderr),
        ):
            retained_stdout = stdout

            def noisy_provider(_request: object) -> GeometryDescriptor:
                retained_stdout.write("retained logging handler\n")
                retained_stdout.flush()
                os.write(retained_stdout.fileno(), b"native stdout write\n")
                return _resolved_geometry()

            status = run(
                arguments,
                geometry_provider=noisy_provider,
                backend_version_provider=lambda: "0.test",
            )
            stdout.flush()
            stderr.flush()
            stdout.seek(0)
            stderr.seek(0)
            stdout_text = stdout.read()
            stderr_text = stderr.read()

        self.assertEqual(status, 0)
        self.assertEqual(json.loads(stdout_text)["status"], "planned")
        self.assertNotIn("retained logging handler", stdout_text)
        self.assertNotIn("native stdout write", stdout_text)
        self.assertIn("retained logging handler", stderr_text)
        self.assertIn("native stdout write", stderr_text)

    def test_cli_structures_malformed_provider_and_backend_failures(self) -> None:
        arguments = [
            "--model",
            ".",
            "--model-fingerprint",
            "sha256:model",
            "--max-model-len",
            "1024",
        ]

        malformed_output = io.StringIO()
        with redirect_stderr(malformed_output):
            status = run(
                arguments,
                geometry_provider=lambda _request: cast(
                    GeometryDescriptor, object()
                ),
                backend_version_provider=lambda: "0.test",
            )
        self.assertEqual(status, 2)
        malformed = json.loads(malformed_output.getvalue())
        self.assertEqual(
            malformed["error"]["kind"], "runtime_geometry_unavailable"
        )
        self.assertIn("malformed result", malformed["error"]["message"])

        backend_output = io.StringIO()

        def broken_backend_version() -> str:
            raise RuntimeError("package metadata is corrupt")

        with redirect_stderr(backend_output):
            status = run(
                arguments,
                geometry_provider=lambda _request: _resolved_geometry(),
                backend_version_provider=broken_backend_version,
            )
        self.assertEqual(status, 2)
        backend_failure = json.loads(backend_output.getvalue())
        self.assertEqual(
            backend_failure["error"]["kind"], "runtime_geometry_unavailable"
        )
        self.assertIn(
            "package metadata is corrupt", backend_failure["error"]["message"]
        )

    def test_cli_rejects_noncertified_profile_identity(self) -> None:
        geometry = _resolved_geometry()
        geometry = replace(
            geometry,
            identity=replace(geometry.identity, profile_id="uncertified/profile"),
        )
        output = io.StringIO()
        with redirect_stderr(output):
            status = run(
                [
                    "--model",
                    ".",
                    "--model-fingerprint",
                    "sha256:model",
                    "--max-model-len",
                    "1024",
                ],
                geometry_provider=lambda _request: geometry,
                backend_version_provider=lambda: "0.test",
            )
        self.assertEqual(status, 2)
        failure = json.loads(output.getvalue())
        self.assertEqual(failure["error"]["kind"], "planning_failed")
        self.assertIn("adapter/profile identity", failure["error"]["message"])

    def test_cli_validates_sizing_policy_before_runtime_provider(self) -> None:
        provider_called = False
        backend_provider_called = False

        def provider(_request: object) -> GeometryDescriptor:
            nonlocal provider_called
            provider_called = True
            return _resolved_geometry()

        def backend_version() -> str:
            nonlocal backend_provider_called
            backend_provider_called = True
            return "0.test"

        output = io.StringIO()
        with redirect_stderr(output):
            status = run(
                [
                    "--model",
                    ".",
                    "--model-fingerprint",
                    "sha256:model",
                    "--max-model-len",
                    "1024",
                    "--target-concurrency",
                    "0",
                ],
                geometry_provider=provider,
                backend_version_provider=backend_version,
            )
        self.assertEqual(status, 2)
        self.assertFalse(provider_called)
        self.assertFalse(backend_provider_called)
        failure = json.loads(output.getvalue())
        self.assertEqual(failure["error"]["kind"], "planning_failed")
        self.assertIn("target_concurrency", failure["error"]["message"])

    def test_cli_rejects_unbounded_runtime_identifiers_before_provider(self) -> None:
        for flag, value, message in (
            ("--max-model-len", str(UINT64_MAX + 1), "uint64 limit"),
            ("--devices", str(UINT64_MAX + 1), "uint64 device"),
        ):
            calls: list[str] = []
            arguments = [
                "--model",
                ".",
                "--model-fingerprint",
                "sha256:model",
                "--max-model-len",
                "1024",
                flag,
                value,
            ]
            output = io.StringIO()
            with self.subTest(flag=flag), redirect_stderr(output):
                status = run(
                    arguments,
                    geometry_provider=lambda _request: calls.append("geometry")
                    or _resolved_geometry(),
                    backend_version_provider=lambda: calls.append("backend")
                    or "0.test",
                )
            self.assertEqual(status, 2)
            self.assertEqual(calls, [])
            self.assertIn(message, json.loads(output.getvalue())["error"]["message"])


if __name__ == "__main__":
    unittest.main()
