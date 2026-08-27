from __future__ import annotations

import base64
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from kapsl_vllm_connector.shared_pool import (
    CudaIpcBuffer,
    SharedPoolImportError,
    VllmElasticBlockPool,
    VllmSharedPoolHook,
    select_cuda_binding,
    vllm_backing_geometry,
)


def _config() -> SimpleNamespace:
    spec = SimpleNamespace(block_size=16)
    return SimpleNamespace(
        num_blocks=4,
        kv_cache_tensors=[SimpleNamespace(size=128, layers=["layer.0"])],
        kv_cache_groups=[
            SimpleNamespace(layer_names=["layer.0"], kv_cache_spec=spec)
        ],
    )


def _binding(device_id: int, **overrides: object) -> dict[str, object]:
    binding: dict[str, object] = {
        "binding_id": f"binding-{device_id}",
        "capacity_pool_id": "vllm.pool.0",
        "generation": 1,
        "group_ids": ["vllm.group.0"],
        "memory_domain": {"kind": "cuda", "device_id": device_id},
        "block_count": 4,
        "bytes_per_block": 32,
        "allocation_mode": "participant_managed",
        "transport": {"kind": "cuda_ipc"},
        "descriptor": base64.b64encode(bytes(64)).decode("ascii"),
    }
    binding.update(overrides)
    return binding


class SharedPoolTests(unittest.TestCase):
    def test_elastic_binding_uses_virtual_maximum_and_mapped_prefix(self) -> None:
        binding = _binding(
            0,
            transport={"kind": "cuda_vmm"},
            descriptor="scm_rights:cuda-vmm-v1",
            elastic={
                "mapped_block_count": 2,
                "maximum_block_count": 4,
                "allocation_granularity_bytes": 64,
                "resize_alignment_blocks": 2,
                "segments": [
                    {
                        "segment_id": "initial-0",
                        "offset_bytes": 0,
                        "length_bytes": 64,
                        "handle_index": 0,
                    }
                ],
            },
        )
        selected = select_cuda_binding(
            {"shared_pools": [binding]},
            _config(),
            None,
            live_resize=True,
        )
        self.assertEqual(selected["elastic"]["mapped_block_count"], 2)
        with self.assertRaisesRegex(SharedPoolImportError, "no cuda_ipc"):
            select_cuda_binding({"shared_pools": [binding]}, _config(), None)

    def test_elastic_block_pool_only_retires_free_tail_blocks(self) -> None:
        class Queue:
            def __init__(self, blocks: list[SimpleNamespace]) -> None:
                self.values = list(blocks)

            def remove(self, block: SimpleNamespace) -> None:
                self.values.remove(block)

            def append_n(self, blocks: list[SimpleNamespace]) -> None:
                self.values.extend(blocks)

        blocks = [
            SimpleNamespace(block_id=index, ref_cnt=0, is_null=index == 0)
            for index in range(8)
        ]
        pool = SimpleNamespace(
            blocks=blocks,
            num_gpu_blocks=8,
            free_block_queue=Queue(blocks[1:]),
            _maybe_evict_cached_block=lambda block: None,
        )
        elastic = VllmElasticBlockPool(4, 8)
        elastic.bind(pool)
        self.assertEqual([block.block_id for block in pool.free_block_queue.values], [1, 2, 3])
        elastic.apply(6)
        self.assertEqual(pool.num_gpu_blocks, 6)
        self.assertEqual(
            [block.block_id for block in pool.free_block_queue.values],
            [1, 2, 3, 4, 5],
        )
        blocks[5].ref_cnt = 1
        with self.assertRaisesRegex(SharedPoolImportError, "live"):
            elastic.apply(4)
        blocks[5].ref_cnt = 0
        elastic.apply(4)
        self.assertEqual(pool.num_gpu_blocks, 4)
        self.assertEqual([block.block_id for block in pool.free_block_queue.values], [1, 2, 3])

    def test_attachment_views_prove_tensor_storage_aliases_the_import(self) -> None:
        class FakeStorage:
            def __init__(self, pointer: int, size: int) -> None:
                self.pointer = pointer
                self.size = size

            def data_ptr(self) -> int:
                return self.pointer

            def nbytes(self) -> int:
                return self.size

        class FakeTensor:
            def __init__(self, storage: FakeStorage, pointer: int) -> None:
                self._storage = storage
                self._pointer = pointer
                self.shape = (2, 2)

            def untyped_storage(self) -> FakeStorage:
                return self._storage

            def data_ptr(self) -> int:
                return self._pointer

            @staticmethod
            def stride() -> tuple[int, int]:
                return (2, 1)

            @staticmethod
            def element_size() -> int:
                return 4

        imported = FakeStorage(1000, 128)
        hook = VllmSharedPoolHook.__new__(VllmSharedPoolHook)
        hook._expected_bytes = 128
        hook._used = True
        hook._layer_identity = {"layer.0": ("vllm.group.0", 0)}
        hook._buffer = SimpleNamespace(
            tensor=SimpleNamespace(untyped_storage=lambda: imported)
        )

        self.assertEqual(
            hook.attachment_views({"layer.0": FakeTensor(imported, 1016)}),
            [
                {
                    "group_id": "vllm.group.0",
                    "layer": {"index": 0, "name": "layer.0"},
                    "offset_bytes": 16,
                    "length_bytes": 16,
                }
            ],
        )
        with self.assertRaisesRegex(SharedPoolImportError, "does not alias"):
            hook.attachment_views(
                {"layer.0": FakeTensor(FakeStorage(2000, 128), 2016)}
            )

    def test_backing_geometry_matches_vllm_packed_allocation(self) -> None:
        self.assertEqual(vllm_backing_geometry(_config()), (128, 4, 32))

    def test_multi_gpu_binding_requires_an_explicit_rank_map(self) -> None:
        receipt = {"shared_pools": [_binding(0), _binding(2)]}

        with self.assertRaisesRegex(
            SharedPoolImportError, "kapsl_rank_device_map"
        ):
            select_cuda_binding(receipt, _config(), None, global_rank=1)

        selected = select_cuda_binding(
            receipt, _config(), {0: 0, 1: 2}, global_rank=1
        )
        self.assertEqual(selected["memory_domain"], {"kind": "cuda", "device_id": 2})

    def test_binding_must_match_participant_managed_geometry(self) -> None:
        receipt = {
            "shared_pools": [
                _binding(0, allocation_mode="runtime_leased", bytes_per_block=31)
            ]
        }

        with self.assertRaisesRegex(
            SharedPoolImportError, "participant_managed"
        ):
            select_cuda_binding(receipt, _config(), None)

    def test_invalid_ipc_descriptor_fails_before_cuda_import(self) -> None:
        with self.assertRaisesRegex(SharedPoolImportError, "base64"):
            CudaIpcBuffer("not-base64!", 128)

    def test_allocator_hook_reuses_one_imported_backing(self) -> None:
        class FakeTorch:
            @staticmethod
            def device(_value: object) -> SimpleNamespace:
                return SimpleNamespace(type="cuda", index=0)

        class FakeWorkerUtils:
            @staticmethod
            def create_kv_cache_views(
                raw: object,
                spec: object,
                num_blocks: int,
                layout: object,
                tensor: object,
                *,
                kernel_block_size: int | None,
            ) -> list[tuple[object, ...]]:
                return [
                    (raw, spec, num_blocks, layout, tensor, kernel_block_size)
                ]

        hook = VllmSharedPoolHook.__new__(VllmSharedPoolHook)
        hook._expected_bytes = 128
        hook._used = False
        hook._buffer = SimpleNamespace(
            tensor=SimpleNamespace(device=SimpleNamespace(type="cuda", index=0)),
            _torch=FakeTorch(),
        )
        hook._worker_utils = FakeWorkerUtils()

        caches = hook._allocate_kv_cache(
            _config(), "cuda:0", "BHLNC", kernel_block_sizes=[8]
        )

        self.assertEqual(set(caches), {"layer.0"})
        self.assertTrue(hook.used)
        with self.assertRaisesRegex(SharedPoolImportError, "more than once"):
            hook._allocate_kv_cache(_config(), "cuda:0", "BHLNC")

    def test_failed_install_restores_partially_patched_modules(self) -> None:
        supported = SimpleNamespace()
        unsupported = SimpleNamespace()

        def original(kv_cache_config: object, device: object, layout: object) -> None:
            del kv_cache_config, device, layout

        def incompatible(config: object) -> None:
            del config

        supported.allocate_kv_cache = original
        supported.create_kv_cache_views = object()
        unsupported.allocate_kv_cache = incompatible
        fake_buffer = SimpleNamespace(release=lambda: None)

        def import_module(name: str) -> object:
            if name == "vllm.v1.worker.utils":
                return supported
            if name == "vllm.v1.worker.gpu_model_runner":
                return unsupported
            raise ImportError(name)

        with (
            patch(
                "kapsl_vllm_connector.shared_pool.CudaIpcBuffer",
                return_value=fake_buffer,
            ),
            patch(
                "kapsl_vllm_connector.shared_pool.importlib.import_module",
                side_effect=import_module,
            ),
            self.assertRaisesRegex(SharedPoolImportError, "unsupported vLLM"),
        ):
            VllmSharedPoolHook(_binding(0), _config())

        self.assertIs(supported.allocate_kv_cache, original)


if __name__ == "__main__":
    unittest.main()
