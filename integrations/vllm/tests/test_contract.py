import unittest
import json
from pathlib import Path

from kapsl_vllm_connector.contract import (
    ContractValidationError,
    make_reserve_request,
    opaque_registration,
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


if __name__ == "__main__":
    unittest.main()
