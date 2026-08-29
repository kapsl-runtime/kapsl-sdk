from __future__ import annotations

import base64
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from kapsl_vllm_connector.shared_pool import (
    CudaIpcBuffer,
    CudaVmmBuffer,
    SharedPoolImportError,
    VllmElasticBlockPool,
    VllmSharedPoolHook,
    _ManagedVmmImport,
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
    def test_vmm_import_releases_generic_handle_when_mapping_fails(self) -> None:
        class FailingDriver:
            def __init__(self) -> None:
                self.released: list[int] = []
                self.freed: list[tuple[int, int]] = []

            def reserve(self, size: int, alignment: int) -> int:
                self.assertions = (size, alignment)
                return 0x10000

            def import_fd(self, descriptor: int) -> int:
                return descriptor + 100

            def map(
                self,
                address: int,
                length: int,
                handle: int,
                device_id: int,
            ) -> None:
                del address, length, handle, device_id
                raise SharedPoolImportError("injected map failure")

            def release_handle(self, handle: int) -> None:
                self.released.append(handle)

            def free_address(self, address: int, length: int) -> None:
                self.freed.append((address, length))

        driver = FailingDriver()
        imported = _ManagedVmmImport(driver, 4096, 64, 0)
        with self.assertRaisesRegex(SharedPoolImportError, "injected"):
            imported.map_segments(
                [
                    {
                        "segment_id": "grow",
                        "offset_bytes": 0,
                        "length_bytes": 64,
                        "handle_index": 0,
                    }
                ],
                [9],
            )
        self.assertEqual(driver.released, [109])
        self.assertEqual(imported.segments, {})
        imported.close()
        self.assertEqual(driver.freed, [(0x10000, 4096)])

    def test_elastic_binding_uses_virtual_maximum_and_mapped_prefix(self) -> None:
        binding = _binding(
            0,
            transport={"kind": "cuda_vmm"},
            descriptor="scm_rights:cuda-vmm-v1",
            elastic={
                "minimum_block_count": 2,
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

    def test_elastic_block_pool_rolls_back_partial_tail_removal(self) -> None:
        class Queue:
            def __init__(self, blocks: list[SimpleNamespace]) -> None:
                self.values = list(blocks)
                self.fail_block_id: int | None = None

            def remove(self, block: SimpleNamespace) -> None:
                if block.block_id == self.fail_block_id:
                    raise RuntimeError("injected queue failure")
                self.values.remove(block)

            def append_n(self, blocks: list[SimpleNamespace]) -> None:
                self.values.extend(blocks)

        blocks = [
            SimpleNamespace(block_id=index, ref_cnt=0, is_null=index == 0)
            for index in range(8)
        ]
        queue = Queue(blocks[1:])
        pool = SimpleNamespace(
            blocks=blocks,
            num_gpu_blocks=8,
            free_block_queue=queue,
            _maybe_evict_cached_block=lambda block: None,
        )
        elastic = VllmElasticBlockPool(4, 8)
        elastic.bind(pool)
        elastic.apply(8)
        queue.fail_block_id = 6

        with self.assertRaisesRegex(SharedPoolImportError, "rolled back"):
            elastic.apply(4)

        self.assertEqual(pool.num_gpu_blocks, 8)
        self.assertEqual(elastic.current_blocks, 8)
        self.assertEqual(
            {block.block_id for block in queue.values},
            set(range(1, 8)),
        )

    def test_elastic_block_pool_rolls_back_after_cache_eviction_failure(self) -> None:
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
        queue = Queue(blocks[1:])

        def evict(block: SimpleNamespace) -> None:
            if block.block_id == 6:
                raise RuntimeError("injected eviction failure")

        pool = SimpleNamespace(
            blocks=blocks,
            num_gpu_blocks=8,
            free_block_queue=queue,
            _maybe_evict_cached_block=evict,
        )
        elastic = VllmElasticBlockPool(4, 8)
        elastic.bind(pool)
        elastic.apply(8)

        with self.assertRaisesRegex(SharedPoolImportError, "rolled back"):
            elastic.apply(4)

        self.assertEqual(pool.num_gpu_blocks, 8)
        self.assertEqual(elastic.current_blocks, 8)
        self.assertEqual(
            {block.block_id for block in queue.values},
            set(range(1, 8)),
        )

    def test_worker_resize_exact_replay_is_idempotent(self) -> None:
        calls: list[tuple[str, int]] = []
        buffer = CudaVmmBuffer.__new__(CudaVmmBuffer)
        buffer.map_segments = lambda segments, handles, target: calls.append(
            ("map", target)
        )
        buffer.unmap_segments = lambda segments, target: calls.append(
            ("unmap", target)
        )
        hook = VllmSharedPoolHook.__new__(VllmSharedPoolHook)
        hook._binding_id = "binding-0"
        hook._buffer = buffer
        hook._last_worker_resize = None
        operation = {
            "binding_id": "binding-0",
            "resize_generation": 7,
            "stage": "map_workers",
            "target_block_count": 8,
            "bytes_per_block": 64,
            "segments": [
                {
                    "segment_id": "grow-7",
                    "offset_bytes": 256,
                    "length_bytes": 256,
                    "handle_index": 0,
                }
            ],
        }

        hook.apply_worker_resize(operation, [11])
        hook.apply_worker_resize(operation, [12])

        self.assertEqual(calls, [("map", 512)])
        altered = dict(operation)
        altered["target_block_count"] = 9
        with self.assertRaisesRegex(SharedPoolImportError, "non-monotonic"):
            hook.apply_worker_resize(altered, [13])

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
        hook._conformance = False
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

    def test_elastic_startup_warmup_is_capped_and_virtual_capacity_restored(
        self,
    ) -> None:
        hook = VllmSharedPoolHook.__new__(VllmSharedPoolHook)
        hook._conformance = True
        hook._startup_mapped_blocks = 2
        hook._maximum_blocks = 4
        hook._startup_warmup_lock = threading.Lock()
        hook._startup_warmup_attempted = False
        config = SimpleNamespace(num_blocks=4)
        runner = SimpleNamespace(kv_cache_config=config)
        observed: list[int] = []

        def original(
            model_runner: object,
            worker_execute_model: object,
            worker_sample_tokens: object,
        ) -> str:
            del worker_execute_model, worker_sample_tokens
            observed.append(model_runner.kv_cache_config.num_blocks)
            return "warmed"

        with self.assertLogs(
            "kapsl_vllm_connector.shared_pool", level="WARNING"
        ) as captured:
            result = hook._run_startup_warmup(
                original,
                runner,
                object(),
                object(),
            )

        self.assertEqual(result, "warmed")
        self.assertEqual(observed, [2])
        self.assertEqual(config.num_blocks, 4)
        self.assertIn(
            "capped vLLM startup warmup to 2 mapped blocks out of 4 virtual blocks",
            "\n".join(captured.output),
        )
        with self.assertRaisesRegex(SharedPoolImportError, "more than once"):
            hook._run_startup_warmup(original, runner, object(), object())

    def test_elastic_startup_warmup_restores_capacity_after_failure(self) -> None:
        hook = VllmSharedPoolHook.__new__(VllmSharedPoolHook)
        hook._conformance = False
        hook._startup_mapped_blocks = 2
        hook._maximum_blocks = 4
        hook._startup_warmup_lock = threading.Lock()
        hook._startup_warmup_attempted = False
        config = SimpleNamespace(num_blocks=4)
        runner = SimpleNamespace(kv_cache_config=config)

        def failing_warmup(
            model_runner: object,
            worker_execute_model: object,
            worker_sample_tokens: object,
        ) -> None:
            del worker_execute_model, worker_sample_tokens
            self.assertEqual(model_runner.kv_cache_config.num_blocks, 2)
            raise RuntimeError("injected warmup failure")

        with self.assertRaisesRegex(RuntimeError, "injected warmup failure"):
            hook._run_startup_warmup(
                failing_warmup,
                runner,
                object(),
                object(),
            )
        self.assertEqual(config.num_blocks, 4)

    def test_elastic_startup_warmup_patches_and_restores_both_references(
        self,
    ) -> None:
        def original(
            model_runner: object,
            worker_execute_model: object,
            worker_sample_tokens: object,
        ) -> str:
            del model_runner, worker_execute_model, worker_sample_tokens
            return "warmed"

        warmup_module = SimpleNamespace(warmup_kernels=original)
        worker_module = SimpleNamespace(warmup_kernels=original)
        modules = {
            "vllm.v1.worker.gpu.warmup": warmup_module,
            "vllm.v1.worker.gpu_worker": worker_module,
        }
        hook = VllmSharedPoolHook.__new__(VllmSharedPoolHook)
        hook._conformance = False
        hook._startup_mapped_blocks = 2
        hook._maximum_blocks = 4
        hook._startup_warmup_lock = threading.Lock()
        hook._startup_warmup_attempted = False
        hook._warmup_patches = []
        hook._warmup_replacement = None

        with patch(
            "kapsl_vllm_connector.shared_pool.importlib.import_module",
            side_effect=lambda name: modules[name],
        ):
            hook._install_startup_warmup_cap()

        replacement = warmup_module.warmup_kernels
        self.assertIs(replacement, worker_module.warmup_kernels)
        self.assertIsNot(replacement, original)
        runner = SimpleNamespace(kv_cache_config=SimpleNamespace(num_blocks=4))
        self.assertEqual(replacement(runner, object(), object()), "warmed")
        self.assertEqual(runner.kv_cache_config.num_blocks, 4)

        hook.shutdown()
        self.assertIs(warmup_module.warmup_kernels, original)
        self.assertIs(worker_module.warmup_kernels, original)

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

    def test_explicit_conformance_mode_reaches_the_cuda_vmm_buffer(self) -> None:
        binding = _binding(
            0,
            transport={"kind": "cuda_vmm"},
            descriptor="scm_rights:cuda-vmm-v1",
            elastic={
                "minimum_block_count": 2,
                "mapped_block_count": 2,
                "maximum_block_count": 4,
                "allocation_granularity_bytes": 64,
                "resize_alignment_blocks": 2,
                "segments": [],
            },
        )
        with (
            patch.object(VllmSharedPoolHook, "_install"),
            patch(
                "kapsl_vllm_connector.shared_pool.CudaVmmBuffer"
            ) as buffer_type,
        ):
            hook = VllmSharedPoolHook(
                binding,
                _config(),
                handles=[11],
                conformance=True,
            )

        self.assertTrue(hook._conformance)
        buffer_type.assert_called_once_with(binding, [11], conformance=True)
        with self.assertRaisesRegex(SharedPoolImportError, "must be a boolean"):
            VllmSharedPoolHook(binding, _config(), handles=[11], conformance=1)


if __name__ == "__main__":
    unittest.main()
