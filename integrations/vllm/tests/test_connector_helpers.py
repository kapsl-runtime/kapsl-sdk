import unittest
import threading
from types import SimpleNamespace

from kapsl_vllm_connector.client import KapslKvControlError
from kapsl_vllm_connector.connector import (
    KapslConnectorV1,
    _request_computed_tokens,
    _request_priority,
    _request_token_capacity,
    _vllm_capacity_groups,
)


class ConnectorHelperTests(unittest.TestCase):
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
