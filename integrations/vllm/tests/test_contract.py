import unittest
import json
from pathlib import Path

from kapsl_vllm_connector.contract import (
    ContractValidationError,
    make_reserve_request,
    opaque_registration,
    shared_pool_registration,
    validate_registration,
    validate_reserve_request,
)

CAPACITY_GROUPS = [
    {
        "group_id": "vllm.group.0",
        "pool_id": "vllm.pool.0",
        "allocation_granularity_tokens": 16,
        "bytes_per_allocation": 1_048_576,
        "memory_domains": [{"kind": "cuda", "device_id": 0}],
        "max_allocations": 1024,
    }
]

TOPOLOGY = {
    "abi_version": {"major": 1, "minor": 2},
    "model_fingerprint": "sha256:model",
    "shard": {
        "tensor_parallel_rank": 0,
        "tensor_parallel_world_size": 1,
        "pipeline_parallel_rank": 0,
        "pipeline_parallel_world_size": 1,
    },
    "cache_groups": [
        {
            "group_id": "vllm.group.0",
            "layers": [{"index": 0, "name": "model.layers.0.attn"}],
            "geometry": {
                "kind": "paged_attention",
                "block_size_tokens": 16,
                "kv_heads": 8,
                "key_head_dim": 128,
                "value_head_dim": 128,
                "element_type": "f16",
                "layout": {"kind": "backend_native", "layout_id": "vllm:packed"},
            },
            "policy": {"kind": "full_attention"},
        }
    ],
}


class ContractTests(unittest.TestCase):
    def test_opaque_registration_matches_rust_wire_shape(self) -> None:
        registration = opaque_registration("vllm-0", "sha256:model", CAPACITY_GROUPS)
        fixture = json.loads(
            (
                Path(__file__).resolve().parents[3]
                / "crates"
                / "kapsl-kv-abi"
                / "tests"
                / "fixtures"
                / "opaque_registration.json"
            ).read_text(encoding="utf-8")
        )

        self.assertEqual(registration, fixture)
        self.assertEqual(registration["backend"], "vllm")
        self.assertEqual(registration["model_fingerprint"], "sha256:model")
        self.assertEqual(registration["capabilities"]["tier"], "kv_connected")
        self.assertEqual(registration["capabilities"]["metadata_mode"], "opaque")
        self.assertEqual(registration["capabilities"]["ownership"], "backend")
        self.assertNotIn("topology", registration)
        validate_registration(registration)

    def test_reserve_request_is_logical_and_backend_neutral(self) -> None:
        request = make_reserve_request(
            request_id="req-1",
            sequence_id="seq-1",
            token_capacity=8192,
            group_ids=["vllm.group.0"],
            priority=-2,
            ttl_ms=30_000,
        )

        self.assertEqual(request["groups"][0]["group_id"], "vllm.group.0")
        self.assertEqual(request["groups"][0]["token_capacity"], 8192)
        self.assertEqual(request["ttl_ms"], 30_000)
        self.assertNotIn("blocks", request["groups"][0])
        validate_reserve_request(request)

    def test_invalid_opaque_registration_cannot_claim_shared_pool(self) -> None:
        registration = opaque_registration("vllm-0", "sha256:model", CAPACITY_GROUPS)
        registration["capabilities"]["tier"] = "shared_pool"

        with self.assertRaises(ContractValidationError):
            validate_registration(registration)

    def test_shared_pool_registration_declares_runtime_ownership(self) -> None:
        registration = shared_pool_registration(
            "vllm-0", "sha256:model", CAPACITY_GROUPS, TOPOLOGY
        )

        self.assertEqual(registration["capabilities"]["tier"], "shared_pool")
        self.assertEqual(
            registration["capabilities"]["ownership"], "kapsl_runtime"
        )
        self.assertIn(
            "participant_block_selection",
            registration["capabilities"]["features"],
        )
        self.assertEqual(
            registration["capabilities"]["transports"], [{"kind": "cuda_ipc"}]
        )
        validate_registration(registration)

    def test_shared_pool_topology_and_capacity_groups_must_match(self) -> None:
        topology = dict(TOPOLOGY)
        topology["cache_groups"] = [dict(TOPOLOGY["cache_groups"][0])]
        topology["cache_groups"][0]["group_id"] = "wrong"

        with self.assertRaisesRegex(ContractValidationError, "same groups"):
            shared_pool_registration(
                "vllm-0", "sha256:model", CAPACITY_GROUPS, topology
            )


if __name__ == "__main__":
    unittest.main()
