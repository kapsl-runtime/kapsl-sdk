import unittest
import threading
from types import SimpleNamespace
from unittest.mock import patch

from kapsl_vllm_connector.client import KapslKvControlError
from kapsl_vllm_connector.connector import (
    ADAPTER_PROFILE_ID,
    KapslConnectorV1,
    _element_type,
    _vllm_adapter_profile,
    _validate_shared_pool_execution,
    _validated_shared_rank_device_map,
    _vllm_topology,
    _request_computed_tokens,
    _request_priority,
    _request_token_capacity,
    _vllm_capacity_groups,
)


class ConnectorHelperTests(unittest.TestCase):
    def test_worker_resize_lock_spans_target_and_deferred_draft_forwards(self) -> None:
        connector = KapslConnectorV1.__new__(KapslConnectorV1)
        connector._is_scheduler = False
        connector._live_resize = True
        connector._worker_forward_lock = threading.Lock()
        connector._worker_forward_active = False
        applied: list[str] = []
        connector._apply_worker_resizes = lambda: applied.append("resize")

        connector.start_load_kv(SimpleNamespace(attn_metadata={}))
        self.assertEqual(applied, ["resize"])
        self.assertTrue(connector._worker_forward_active)
        self.assertTrue(connector._worker_forward_lock.locked())

        # Pinned vLLM calls this before its deferred speculative/draft forward.
        # It must not permit a physical unmap yet.
        connector.build_connector_worker_meta()
        self.assertTrue(connector._worker_forward_lock.locked())

        connector.wait_for_save()
        self.assertFalse(connector._worker_forward_active)
        self.assertFalse(connector._worker_forward_lock.locked())

    def test_worker_resize_zero_token_step_releases_without_wait_for_save(self) -> None:
        connector = KapslConnectorV1.__new__(KapslConnectorV1)
        connector._is_scheduler = False
        connector._live_resize = True
        connector._worker_forward_lock = threading.Lock()
        connector._worker_forward_active = False
        applied: list[str] = []
        connector._apply_worker_resizes = lambda: applied.append("resize")

        connector.start_load_kv(SimpleNamespace(attn_metadata=None))

        self.assertEqual(applied, ["resize"])
        self.assertFalse(connector._worker_forward_active)
        self.assertFalse(connector._worker_forward_lock.locked())

    def test_scheduler_stays_awake_until_grown_capacity_returns_to_initial(self) -> None:
        connector = KapslConnectorV1.__new__(KapslConnectorV1)
        connector._resize_lock = threading.RLock()
        connector._resize_pending = False
        connector._pending_scheduler_operations = []
        connector._elastic_block_pool = SimpleNamespace(
            initial_blocks=4,
            current_blocks=4,
        )

        self.assertFalse(connector.has_pending_push_work())

        # A completed grow can race the supervisor's follow-up shrink after
        # the last request finishes. Keeping vLLM's supported push-work loop
        # active lets the scheduler receive and acknowledge that shrink.
        connector._elastic_block_pool.current_blocks = 8
        self.assertTrue(connector.has_pending_push_work())

        connector._elastic_block_pool.current_blocks = 4
        connector._resize_pending = True
        self.assertTrue(connector.has_pending_push_work())

        connector._resize_pending = False
        connector._pending_scheduler_operations = [{"resize_generation": 2}]
        self.assertTrue(connector.has_pending_push_work())

    def test_scheduler_activates_shared_pool_once_before_admission(self) -> None:
        class FakeClient:
            def __init__(self) -> None:
                self.epochs: list[int] = []

            def activate(self, epoch: int) -> None:
                self.epochs.append(epoch)

        connector = KapslConnectorV1.__new__(KapslConnectorV1)
        connector._mode = "shared_pool"
        connector._shared_active = False
        connector._activation_lock = threading.Lock()
        connector._participant_epoch = 7
        connector._participant_id = "vllm:test"
        connector._client = FakeClient()

        connector._ensure_shared_active()
        connector._ensure_shared_active()

        self.assertEqual(connector._client.epochs, [7])

    def test_scheduler_startup_activates_before_heartbeat(self) -> None:
        events: list[str] = []

        class FakeThread:
            def __init__(self, *, target, name: str, daemon: bool) -> None:
                self.target = target
                self.name = name
                self.daemon = daemon

            def start(self) -> None:
                events.append("heartbeat")

        connector = KapslConnectorV1.__new__(KapslConnectorV1)
        connector._is_scheduler = True
        connector._mode = "shared_pool"
        connector._heartbeat_loop = lambda: None
        connector._ensure_shared_active = lambda: events.append("activate")

        with patch(
            "kapsl_vllm_connector.connector.threading.Thread", FakeThread
        ):
            connector._start_scheduler_control("engine-7")

        self.assertEqual(events, ["activate", "heartbeat"])
        self.assertEqual(
            connector._heartbeat_thread.name, "kapsl-kv-heartbeat-engine-7"
        )
        self.assertTrue(connector._heartbeat_thread.daemon)

    def test_worker_startup_does_not_activate_or_start_heartbeat(self) -> None:
        connector = KapslConnectorV1.__new__(KapslConnectorV1)
        connector._is_scheduler = False
        connector._mode = "shared_pool"
        connector._heartbeat_thread = None
        connector._ensure_shared_active = lambda: self.fail("worker activated pool")

        with patch("kapsl_vllm_connector.connector.threading.Thread") as thread:
            connector._start_scheduler_control("engine-7")

        thread.assert_not_called()
        self.assertIsNone(connector._heartbeat_thread)

    def test_scheduler_startup_fails_before_heartbeat_when_activation_fails(
        self,
    ) -> None:
        def fail_activation() -> None:
            raise KapslKvControlError("worker binding is missing")

        connector = KapslConnectorV1.__new__(KapslConnectorV1)
        connector._is_scheduler = True
        connector._mode = "shared_pool"
        connector._heartbeat_loop = lambda: None
        connector._ensure_shared_active = fail_activation

        with patch("kapsl_vllm_connector.connector.threading.Thread") as thread:
            with self.assertRaisesRegex(KapslKvControlError, "binding is missing"):
                connector._start_scheduler_control("engine-7")

        thread.assert_not_called()

    def test_shared_pool_rejects_vllm_sleep_mode(self) -> None:
        config = SimpleNamespace(
            parallel_config=SimpleNamespace(),
            model_config=SimpleNamespace(enable_sleep_mode=True),
        )

        with self.assertRaisesRegex(ValueError, "sleep mode"):
            _validate_shared_pool_execution(config)

    def test_shared_pool_requires_explicit_flash_attention_profile(self) -> None:
        base = {
            "parallel_config": SimpleNamespace(),
            "model_config": SimpleNamespace(enable_sleep_mode=False),
        }
        with self.assertRaisesRegex(ValueError, "explicitly selected FLASH_ATTN"):
            _validate_shared_pool_execution(SimpleNamespace(**base))
        with self.assertRaisesRegex(ValueError, "explicitly selected FLASH_ATTN"):
            _validate_shared_pool_execution(
                SimpleNamespace(
                    **base,
                    attention_config=SimpleNamespace(
                        backend=SimpleNamespace(name="FLASHINFER"),
                        backend_per_kind={},
                    ),
                )
            )

        config = SimpleNamespace(
            **base,
            attention_config=SimpleNamespace(
                backend=SimpleNamespace(name="FLASH_ATTN"),
                backend_per_kind={},
            ),
        )
        _validate_shared_pool_execution(config)
        with patch(
            "kapsl_vllm_connector.connector.metadata.version",
            return_value="0.test",
        ):
            profile = _vllm_adapter_profile(config)
        self.assertEqual(profile["profile_id"], ADAPTER_PROFILE_ID)
        self.assertEqual(profile["backend_version"], "0.test")

    def test_capacity_includes_prompt_and_generation_ceiling(self) -> None:
        request = SimpleNamespace(
            prompt_token_ids=[1, 2, 3],
            all_token_ids=[1, 2, 3],
            sampling_params=SimpleNamespace(max_tokens=29),
        )
        self.assertEqual(_request_token_capacity(request), 32)

    def test_completion_and_priority_are_defensive_across_vllm_versions(self) -> None:
        request = SimpleNamespace(
            num_computed_tokens=17,
            priority=-1,
            all_token_ids=[1, 2],
        )
        self.assertEqual(_request_computed_tokens(request), 17)
        self.assertEqual(_request_priority(request), -1)

    def test_opaque_capacity_model_accounts_for_every_layer_page(self) -> None:
        spec = SimpleNamespace(block_size=16, page_size_bytes=2048)
        smaller_spec = SimpleNamespace(block_size=16, page_size_bytes=1024)
        config = SimpleNamespace(
            num_blocks=512,
            kv_cache_groups=[
                SimpleNamespace(
                    layer_names=["layer.0", "layer.1", "layer.2"],
                    kv_cache_spec=spec,
                ),
                SimpleNamespace(
                    layer_names=["layer.3"],
                    kv_cache_spec=smaller_spec,
                ),
            ],
        )
        self.assertEqual(
            _vllm_capacity_groups(
                config,
                [
                    {"kind": "cuda", "device_id": 0},
                    {"kind": "cuda", "device_id": 1},
                ],
            ),
            [
                {
                    "group_id": "vllm.group.0",
                    "pool_id": "vllm.pool.0",
                    "allocation_granularity_tokens": 16,
                    "bytes_per_allocation": 6144,
                    "memory_domains": [
                        {"kind": "cuda", "device_id": 0},
                        {"kind": "cuda", "device_id": 1},
                    ],
                    "max_allocations": 512,
                },
                {
                    "group_id": "vllm.group.1",
                    "pool_id": "vllm.pool.0",
                    "allocation_granularity_tokens": 16,
                    "bytes_per_allocation": 6144,
                    "memory_domains": [
                        {"kind": "cuda", "device_id": 0},
                        {"kind": "cuda", "device_id": 1},
                    ],
                    "max_allocations": 512,
                },
            ],
        )

    def test_shared_capacity_uses_the_exact_packed_backing_stride(self) -> None:
        spec = SimpleNamespace(
            block_size=16,
            page_size_bytes=1024,
            num_kv_heads=4,
            head_size=4,
            head_size_v=4,
            dtype="float16",
            sliding_window=None,
            attention_chunk_size=None,
        )
        config = SimpleNamespace(
            num_blocks=32,
            kv_cache_layout="BHLNC",
            kv_cache_tensors=[
                SimpleNamespace(
                    size=32 * 2048,
                    layers=["model.layers.0.attn", "model.layers.1.attn"],
                )
            ],
            kv_cache_groups=[
                SimpleNamespace(
                    layer_names=["model.layers.0.attn", "model.layers.1.attn"],
                    kv_cache_spec=spec,
                )
            ],
        )

        groups = _vllm_capacity_groups(
            config,
            [{"kind": "cuda", "device_id": 0}],
            shared_pool=True,
            spec_kind_classifier=lambda _spec: "full_attention",
        )
        self.assertEqual(groups[0]["bytes_per_allocation"], 2048)
        topology = _vllm_topology(
            config,
            "sha256:model",
            spec_kind_classifier=lambda _spec: "full_attention",
        )
        self.assertEqual(
            topology["cache_groups"][0]["geometry"]["layout"]["layout_id"],
            "vllm:BHLNC",
        )
        self.assertEqual(
            topology["cache_groups"][0]["geometry"]["element_type"],
            {"kind": "f16"},
        )
        self.assertEqual(len(topology["cache_groups"][0]["layers"]), 2)

        spec.block_size = 16.5
        with self.assertRaisesRegex(ValueError, "must be an integer"):
            _vllm_topology(
                config,
                "sha256:model",
                spec_kind_classifier=lambda _spec: "full_attention",
            )

    def test_element_types_use_the_rust_internally_tagged_shape(self) -> None:
        self.assertEqual(_element_type("torch.float16"), {"kind": "f16"})
        self.assertEqual(_element_type("bfloat16"), {"kind": "bf16"})
        self.assertEqual(
            _element_type("vendor_float6"),
            {"kind": "custom", "name": "vendor_float6"},
        )

    def test_shared_rank_map_must_cover_exact_tensor_parallel_domains(self) -> None:
        config = SimpleNamespace(
            kv_connector_extra_config={"kapsl_rank_device_map": {"0": 0, "1": 2}}
        )
        domains = [
            {"kind": "cuda", "device_id": 0},
            {"kind": "cuda", "device_id": 2},
        ]

        self.assertEqual(
            _validated_shared_rank_device_map(config, domains, 2),
            {0: 0, 1: 2},
        )
        with self.assertRaisesRegex(ValueError, "exactly match"):
            _validated_shared_rank_device_map(
                SimpleNamespace(
                    kv_connector_extra_config={
                        "kapsl_rank_device_map": {"0": 0, "1": 1}
                    }
                ),
                domains,
                2,
            )

    def test_shared_rank_map_rejects_noncanonical_and_nonintegral_values(self) -> None:
        domains = [
            {"kind": "cuda", "device_id": 0},
            {"kind": "cuda", "device_id": 2},
        ]
        invalid_maps = (
            ({"00": 0, "1": 2}, "canonical"),
            ({"+0": 0, "1": 2}, "canonical"),
            ({" 0": 0, "1": 2}, "canonical"),
            ({0.0: 0, "1": 2}, "canonical"),
            ({True: 0, "1": 2}, "canonical"),
            ({"0": False, "1": 2}, "unsigned 64-bit integer"),
            ({"0": 0.0, "1": 2}, "unsigned 64-bit integer"),
            ({"0": "0", "1": 2}, "unsigned 64-bit integer"),
        )
        for rank_map, message in invalid_maps:
            with self.subTest(rank_map=rank_map), self.assertRaisesRegex(
                ValueError, message
            ):
                _validated_shared_rank_device_map(
                    SimpleNamespace(
                        kv_connector_extra_config={
                            "kapsl_rank_device_map": rank_map
                        }
                    ),
                    domains,
                    2,
                )

        with self.assertRaisesRegex(ValueError, "duplicate or colliding ranks"):
            _validated_shared_rank_device_map(
                SimpleNamespace(
                    kv_connector_extra_config={
                        "kapsl_rank_device_map": {
                            0: 0,
                            "0": 2,
                            1: 2,
                        }
                    }
                ),
                domains,
                2,
            )

    def test_shared_rank_map_rejects_nonintegral_memory_domain_ids(self) -> None:
        config = SimpleNamespace(
            kv_connector_extra_config={"kapsl_rank_device_map": {"0": 0}}
        )
        for device_id in (True, 0.0, "0", None):
            with self.subTest(device_id=device_id), self.assertRaisesRegex(
                ValueError, "unsigned 64-bit integer"
            ):
                _validated_shared_rank_device_map(
                    config,
                    [{"kind": "cuda", "device_id": device_id}],
                    1,
                )

    def test_heartbeat_failure_is_a_fail_closed_scheduler_error(self) -> None:
        connector = KapslConnectorV1.__new__(KapslConnectorV1)
        connector._lease_lock = threading.RLock()
        connector._control_failure = KapslKvControlError(
            "lease expired", kind="not_found"
        )

        with self.assertRaises(KapslKvControlError) as caught:
            connector._raise_if_control_failed()

        self.assertEqual(caught.exception.kind, "not_found")
        self.assertIn("heartbeat failed", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
