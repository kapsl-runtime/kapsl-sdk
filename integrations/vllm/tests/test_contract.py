import json
import unittest
from copy import deepcopy
from pathlib import Path

from kapsl_vllm_connector.contract import (
    ABI_VERSION,
    ContractValidationError,
    make_reserve_request,
    make_shared_pool_attachment,
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
    "abi_version": dict(ABI_VERSION),
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
                "element_type": {"kind": "f16"},
                "layout": {"kind": "backend_native", "layout_id": "vllm:packed"},
            },
            "policy": {"kind": "full_attention"},
        }
    ],
}

PROFILE = {
    "adapter_id": "kapsl-vllm-connector",
    "adapter_version": "0.4.0",
    "backend_version": "test-vllm",
    "profile_id": "vllm-v1-packed-cuda-ipc",
}


class ContractTests(unittest.TestCase):
    def test_shared_attachment_rejects_views_outside_the_imported_pool(self) -> None:
        with self.assertRaisesRegex(ContractValidationError, "exceeds"):
            make_shared_pool_attachment(
                participant_epoch=1,
                binding_id="binding-0",
                shard=TOPOLOGY["shard"],
                profile={
                    "adapter_id": "kapsl-vllm-connector",
                    "adapter_version": "0.4.0",
                    "backend_version": "test-vllm",
                    "profile_id": "vllm-v1-packed-cuda-ipc",
                },
                imported_bytes=128,
                views=[
                    {
                        "group_id": "vllm.group.0",
                        "layer": {"index": 0, "name": "model.layers.0.attn"},
                        "offset_bytes": 64,
                        "length_bytes": 128,
                    }
                ],
            )

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
            "vllm-0", "sha256:model", CAPACITY_GROUPS, TOPOLOGY, PROFILE
        )
        fixture = json.loads(
            (
                Path(__file__).resolve().parents[3]
                / "crates"
                / "kapsl-kv-abi"
                / "tests"
                / "fixtures"
                / "shared_pool_registration.json"
            ).read_text(encoding="utf-8")
        )

        self.assertEqual(registration, fixture)
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
        self.assertEqual(registration["adapter_profile"], PROFILE)
        validate_registration(registration)

        missing_profile = dict(registration)
        missing_profile.pop("adapter_profile")
        with self.assertRaisesRegex(ContractValidationError, "adapter profile"):
            validate_registration(missing_profile)

    def test_shared_pool_topology_and_capacity_groups_must_match(self) -> None:
        topology = dict(TOPOLOGY)
        topology["cache_groups"] = [dict(TOPOLOGY["cache_groups"][0])]
        topology["cache_groups"][0]["group_id"] = "wrong"

        with self.assertRaisesRegex(ContractValidationError, "same groups"):
            shared_pool_registration(
                "vllm-0", "sha256:model", CAPACITY_GROUPS, topology, PROFILE
            )

    def test_shared_pool_rejects_non_rust_element_type_shapes(self) -> None:
        for invalid in (
            "f16",
            {"kind": "unknown"},
            {"kind": "custom", "name": " "},
        ):
            with self.subTest(element_type=invalid):
                topology = deepcopy(TOPOLOGY)
                topology["cache_groups"][0]["geometry"]["element_type"] = invalid
                with self.assertRaisesRegex(
                    ContractValidationError, "topology element_type"
                ):
                    shared_pool_registration(
                        "vllm-0",
                        "sha256:model",
                        CAPACITY_GROUPS,
                        topology,
                        PROFILE,
                    )


if __name__ == "__main__":
    unittest.main()
