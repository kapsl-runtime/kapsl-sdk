from __future__ import annotations

import unittest

from kapsl_vllm_connector.flash_attn_probe import (
    ProbeGeometry,
    _LeaseHeartbeat,
    _aggregate_gates,
    _parse_devices,
    _registration,
)


class FlashAttentionProbeHostTests(unittest.TestCase):
    def test_live_lease_heartbeat_is_exercised_even_for_a_fast_probe(self) -> None:
        class FakeClient:
            def __init__(self) -> None:
                self.calls = 0

            def heartbeat(self) -> None:
                self.calls += 1

        client = FakeClient()
        heartbeat = _LeaseHeartbeat(client, 30_000)  # type: ignore[arg-type]

        heartbeat.start()
        heartbeat.stop()

        self.assertGreaterEqual(client.calls, 1)

    def test_registration_requests_guarded_runtime_owned_cuda_replicas(self) -> None:
        geometry = ProbeGeometry(8, 16, 2, 4, 64, "float16", 256)
        profile = {
            "adapter_id": "kapsl-vllm-connector",
            "adapter_version": "0.5.0",
            "backend_version": "test",
            "profile_id": "vllm-v1-packed-cuda-ipc/flash-attn",
        }

        registration = _registration("probe", profile, [0, 2], geometry)

        self.assertEqual(registration["capabilities"]["tier"], "shared_pool")
        group = registration["capacity_model"]["groups"][0]
        self.assertEqual(group["bytes_per_allocation"], 8448)
        self.assertEqual(
            group["memory_domains"],
            [
                {"kind": "cuda", "device_id": 0},
                {"kind": "cuda", "device_id": 2},
            ],
        )
        self.assertEqual(
            registration["topology"]["shard"]["tensor_parallel_world_size"],
            2,
        )
        self.assertEqual(
            registration["topology"]["cache_groups"][0]["geometry"][
                "element_type"
            ],
            {"kind": "f16"},
        )

    def test_aggregate_requires_every_physical_binding(self) -> None:
        ranks = [
            {
                "rank": rank,
                "binding_id": f"binding-{rank}",
                "passed": True,
                "gates": {
                    name: {"passed": True}
                    for name in (
                        "allocator_attachment",
                        "backend_native_write",
                        "backend_native_read",
                    )
                },
                "lifecycle": {
                    "activation_after_all_attachments": True,
                    "live_lease_detach_rejected": True,
                    "cancellation_release": True,
                    "capacity_exhaustion_rejected": True,
                    "heartbeat_renewal": True,
                    "post_deactivation_reserve_rejected": True,
                },
            }
            for rank in range(2)
        ]

        gates = _aggregate_gates(
            ranks,
            expected_bindings={"binding-0", "binding-1"},
            contract_evidence="rejected",
        )
        self.assertTrue(gates["parallel_coverage"]["passed"])

        with self.assertRaisesRegex(RuntimeError, "every provisioned binding"):
            _aggregate_gates(
                ranks,
                expected_bindings={"binding-0", "different"},
                contract_evidence="rejected",
            )

    def test_device_parser_rejects_duplicate_or_negative_ordinals(self) -> None:
        self.assertEqual(_parse_devices("0,2"), [0, 2])
        with self.assertRaisesRegex(ValueError, "duplicates"):
            _parse_devices("0,0")
        with self.assertRaisesRegex(ValueError, "non-negative"):
            _parse_devices("-1")


if __name__ == "__main__":
    unittest.main()
