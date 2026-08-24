from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

from kapsl_vllm_connector.certification import (
    CertificationError,
    allowlist_entry,
    remove_stale_allowlist,
    validate_certification_report,
    write_allowlist_atomic,
)


def _passing_report() -> dict[str, object]:
    gate_names = (
        "contract",
        "allocator_attachment",
        "backend_native_write",
        "backend_native_read",
        "lifecycle",
        "parallel_coverage",
    )
    sentinel_hash = hashlib.sha256(bytes([0x5A]) * 67584).hexdigest()
    return {
        "schema_version": 1,
        "status": "passed",
        "profile": {
            "adapter_id": "kapsl-vllm-connector",
            "adapter_version": "0.5.0",
            "backend_version": "0.test",
            "profile_id": "vllm-v1-packed-cuda-ipc/flash-attn",
        },
        "environment": {
            "adapter_build_id": "sha256:" + "1" * 64,
            "backend_build_id": "sha256:" + "2" * 64,
            "runtime_build_id": "sha256:" + "3" * 64,
            "torch_version": "2.test",
            "cuda_runtime_version": "13.test",
            "cuda_driver_version": "600.test",
        },
        "matrix": {
            "attention_backend": "FLASH_ATTN",
            "kv_layout": "LBNHC",
            "dtype": "float16",
            "cache_geometry": {
                "num_blocks": 8,
                "block_size": 16,
                "num_kv_heads": 2,
                "num_query_heads": 4,
                "head_size": 64,
                "dense_page_bytes": 8192,
                "guard_bytes_per_block": 256,
                "padded_page_bytes": 8448,
                "allocation_bytes": 67584,
            },
            "tensor_parallel_world_size": 2,
            "devices": [0, 2],
        },
        "gates": {
            name: {"passed": True, "evidence": f"{name} evidence"}
            for name in gate_names
        },
        "ranks": [
            {
                "rank": rank,
                "device_id": rank * 2,
                "binding_id": f"binding-{rank}",
                "passed": True,
                "device": {
                    "name": f"GPU {rank}",
                    "compute_capability": "9.0",
                    "total_memory_bytes": 80 * 1024**3,
                },
                "gates": {
                    "allocator_attachment": {
                        "passed": True,
                        "evidence": {
                            "allocator_poisoned": True,
                            "pytorch_cuda_allocation_delta_bytes": 0,
                            "imported_bytes": 67584,
                            "view_count": 1,
                            "raw_sha256_before_write": sentinel_hash,
                        },
                    },
                    "backend_native_write": {
                        "passed": True,
                        "evidence": {
                            "native_function": "vllm.native.write",
                            "raw_sha256_after_write": "5" * 64,
                            "guard_bytes_checked": 2048,
                            "prefill_tokens": 5,
                            "decode_writes": 2,
                            "maximum_block_index": 7,
                            "production_implementation_binding": True,
                        },
                    },
                    "backend_native_read": {
                        "passed": True,
                        "evidence": {
                            "native_function": "vllm.native.read",
                            "implementation": "vllm.FlashAttentionImpl",
                            "flash_attention_version": 3,
                            "causal_mutation_max_delta": 1.0,
                            "production_implementation_binding": True,
                        },
                    },
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
        ],
    }


class CertificationTests(unittest.TestCase):
    def test_complete_report_emits_exact_runtime_allowlist_value(self) -> None:
        report = _passing_report()

        validate_certification_report(report)

        self.assertEqual(
            allowlist_entry(report),
            "kapsl-vllm-connector,0.5.0,0.test,"
            "vllm-v1-packed-cuda-ipc/flash-attn",
        )

    def test_failed_or_incomplete_evidence_cannot_emit_allowlist(self) -> None:
        report = _passing_report()
        report["gates"]["backend_native_read"]["passed"] = False  # type: ignore[index]

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "allowlist.txt"
            with self.assertRaisesRegex(CertificationError, "native_read"):
                write_allowlist_atomic(output, report)
            self.assertFalse(output.exists())

    def test_every_rank_must_use_a_distinct_binding(self) -> None:
        report = _passing_report()
        report["ranks"][1]["binding_id"] = "binding-0"  # type: ignore[index]

        with self.assertRaisesRegex(CertificationError, "unique"):
            validate_certification_report(report)

    def test_boolean_pass_without_native_evidence_is_rejected(self) -> None:
        report = _passing_report()
        rank_zero = report["ranks"][0]  # type: ignore[index]
        read_gate = rank_zero["gates"]["backend_native_read"]
        read_evidence = read_gate["evidence"]
        del read_evidence["causal_mutation_max_delta"]

        with self.assertRaisesRegex(CertificationError, "causal mutation"):
            validate_certification_report(report)

    def test_new_probe_removes_a_stale_allowlist_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "allowlist.txt"
            output.write_text("old certification\n", encoding="utf-8")

            remove_stale_allowlist(output)

            self.assertFalse(output.exists())

    def test_profile_delimiters_cannot_inject_a_second_allowlist_field(self) -> None:
        report = _passing_report()
        report["profile"]["backend_version"] = "bad,extra"  # type: ignore[index]

        with self.assertRaisesRegex(CertificationError, "unsafe delimiter"):
            allowlist_entry(report)


if __name__ == "__main__":
    unittest.main()
