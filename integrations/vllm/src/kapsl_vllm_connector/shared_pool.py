"""CUDA IPC import and narrowly versioned vLLM allocation hook.

The Kapsl runtime owns the physical allocation. vLLM keeps its native block
allocator and block tables, but its attention tensors become views into that
allocation instead of memory obtained from PyTorch's allocator.
"""

from __future__ import annotations

import base64
import binascii
import ctypes
import ctypes.util
import importlib
import inspect
import os
import sys
import threading
from collections.abc import Mapping
from typing import Any


class SharedPoolImportError(RuntimeError):
    """A shared-pool descriptor cannot be imported without weakening safety."""


def vllm_backing_geometry(kv_cache_config: Any) -> tuple[int, int, int]:
    """Return ``(allocation bytes, blocks, bytes per block)``.

    Modern vLLM packed layouts may publish several tensor placements, but its
    allocator requires them all to address one same-sized backing allocation.
    """

    num_blocks = int(getattr(kv_cache_config, "num_blocks", 0) or 0)
    tensors = getattr(kv_cache_config, "kv_cache_tensors", None)
    if num_blocks <= 0 or not tensors:
        raise SharedPoolImportError(
            "shared_pool requires vLLM KVCacheConfig.num_blocks and kv_cache_tensors"
        )
    sizes = {int(getattr(tensor, "size", 0) or 0) for tensor in tensors}
    if len(sizes) != 1 or next(iter(sizes)) <= 0:
        raise SharedPoolImportError(
            "shared_pool requires every vLLM KV tensor placement to share one backing size"
        )
    allocation_bytes = sizes.pop()
    bytes_per_block, remainder = divmod(allocation_bytes, num_blocks)
    if remainder or bytes_per_block <= 0:
        raise SharedPoolImportError(
            "vLLM packed KV allocation is not an integral number of blocks"
        )
    return allocation_bytes, num_blocks, bytes_per_block


def select_cuda_binding(
    receipt: Mapping[str, Any],
    kv_cache_config: Any,
    rank_device_map: Mapping[str | int, int] | None,
    *,
    global_rank: int | None = None,
) -> dict[str, Any]:
    """Select and validate the physical replica imported by one worker."""

    allocation_bytes, num_blocks, bytes_per_block = vllm_backing_geometry(
        kv_cache_config
    )
    pools = receipt.get("shared_pools")
    if not isinstance(pools, list) or not pools:
        raise SharedPoolImportError("Kapsl returned no shared-pool bindings")
    candidates = [
        dict(pool)
        for pool in pools
        if isinstance(pool, Mapping)
        and pool.get("capacity_pool_id") == "vllm.pool.0"
        and isinstance(pool.get("memory_domain"), Mapping)
        and pool["memory_domain"].get("kind") == "cuda"
        and isinstance(pool.get("transport"), Mapping)
        and pool["transport"].get("kind") == "cuda_ipc"
    ]
    if not candidates:
        raise SharedPoolImportError(
            "Kapsl receipt has no CUDA IPC binding for vllm.pool.0"
        )

    if len(candidates) == 1:
        binding = candidates[0]
    else:
        if global_rank is None:
            global_rank = vllm_distributed_rank()
        if rank_device_map is None:
            raise SharedPoolImportError(
                "multiple CUDA bindings require kapsl_rank_device_map"
            )
        raw_device_id = rank_device_map.get(global_rank)
        if raw_device_id is None:
            raw_device_id = rank_device_map.get(str(global_rank))
        if raw_device_id is None:
            raise SharedPoolImportError(
                f"kapsl_rank_device_map has no entry for global rank {global_rank}"
            )
        device_id = int(raw_device_id)
        matches = [
            pool
            for pool in candidates
            if pool["memory_domain"].get("device_id") == device_id
        ]
        if len(matches) != 1:
            raise SharedPoolImportError(
                f"rank {global_rank} maps to CUDA device {device_id}, but the receipt has {len(matches)} matching bindings"
            )
        binding = matches[0]

    if binding.get("allocation_mode") != "participant_managed":
        raise SharedPoolImportError(
            "vLLM shared_pool requires participant_managed block selection"
        )
    if int(binding.get("block_count", 0)) != num_blocks:
        raise SharedPoolImportError(
            "Kapsl block count does not match vLLM's configured allocator"
        )
    if int(binding.get("bytes_per_block", 0)) != bytes_per_block:
        raise SharedPoolImportError(
            "Kapsl block stride does not match vLLM's packed allocation"
        )
    if allocation_bytes != num_blocks * bytes_per_block:
        raise AssertionError("validated vLLM backing geometry changed")
    return binding


def vllm_distributed_rank() -> int:
    try:
        import torch

        distributed = getattr(torch, "distributed", None)
        if distributed is not None and distributed.is_initialized():
            return int(distributed.get_rank())
    except (ImportError, RuntimeError):
        pass
    raw_rank = os.environ.get("RANK")
    if raw_rank is not None:
        try:
            return int(raw_rank)
        except ValueError as error:
            raise SharedPoolImportError("RANK must be an integer") from error
    raise SharedPoolImportError(
        "cannot determine the vLLM global rank for CUDA binding selection"
    )


class _CudaIpcMemHandle(ctypes.Structure):
    _fields_ = [("reserved", ctypes.c_byte * 64)]


class _DLDevice(ctypes.Structure):
    _fields_ = [("device_type", ctypes.c_int32), ("device_id", ctypes.c_int32)]


class _DLDataType(ctypes.Structure):
    _fields_ = [
        ("code", ctypes.c_uint8),
        ("bits", ctypes.c_uint8),
        ("lanes", ctypes.c_uint16),
    ]


class _DLTensor(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.c_void_p),
        ("device", _DLDevice),
        ("ndim", ctypes.c_int32),
        ("dtype", _DLDataType),
        ("shape", ctypes.POINTER(ctypes.c_int64)),
        ("strides", ctypes.POINTER(ctypes.c_int64)),
        ("byte_offset", ctypes.c_uint64),
    ]


class _DLManagedTensor(ctypes.Structure):
    pass


_DLManagedTensorDeleter = ctypes.CFUNCTYPE(
    None, ctypes.POINTER(_DLManagedTensor)
)
_DLManagedTensor._fields_ = [
    ("dl_tensor", _DLTensor),
    ("manager_ctx", ctypes.c_void_p),
    ("deleter", _DLManagedTensorDeleter),
]


class _CudaDriver:
    def __init__(self) -> None:
        if not sys.platform.startswith("linux"):
            raise SharedPoolImportError("CUDA IPC shared_pool is supported on Linux only")
        library_name = ctypes.util.find_library("cuda") or "libcuda.so.1"
        try:
            self.library = ctypes.CDLL(library_name)
        except OSError as error:
            raise SharedPoolImportError(f"cannot load CUDA driver: {error}") from error
        self.library.cuInit.argtypes = [ctypes.c_uint]
        self.library.cuInit.restype = ctypes.c_int
        self._open = getattr(self.library, "cuIpcOpenMemHandle_v2", None)
        if self._open is None:
            self._open = getattr(self.library, "cuIpcOpenMemHandle", None)
        if self._open is None:
            raise SharedPoolImportError("CUDA driver has no IPC memory import symbol")
        self._open.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            _CudaIpcMemHandle,
            ctypes.c_uint,
        ]
        self._open.restype = ctypes.c_int
        self.library.cuIpcCloseMemHandle.argtypes = [ctypes.c_void_p]
        self.library.cuIpcCloseMemHandle.restype = ctypes.c_int
        self._check(self.library.cuInit(0), "initialize CUDA driver")

    def open(self, handle_bytes: bytes) -> ctypes.c_void_p:
        handle = _CudaIpcMemHandle.from_buffer_copy(handle_bytes)
        pointer = ctypes.c_void_p()
        # CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS = 1.
        self._check(self._open(ctypes.byref(pointer), handle, 1), "open CUDA IPC handle")
        if not pointer.value:
            raise SharedPoolImportError("CUDA IPC import returned a null pointer")
        return pointer

    def close(self, pointer: ctypes.c_void_p) -> None:
        self._check(
            self.library.cuIpcCloseMemHandle(pointer), "close CUDA IPC mapping"
        )

    @staticmethod
    def _check(result: int, operation: str) -> None:
        if result != 0:
            raise SharedPoolImportError(f"failed to {operation}: CUDA error {result}")


class _ManagedImport:
    def __init__(
        self,
        driver: _CudaDriver,
        pointer: ctypes.c_void_p,
        allocation_bytes: int,
        torch_device_id: int,
    ) -> None:
        self.driver = driver
        self.pointer = pointer
        self.shape = (ctypes.c_int64 * 1)(allocation_bytes)
        self.managed = _DLManagedTensor(
            dl_tensor=_DLTensor(
                data=pointer,
                device=_DLDevice(device_type=2, device_id=torch_device_id),
                ndim=1,
                # kDLInt, 8 bits, one lane: identical to torch.int8.
                dtype=_DLDataType(code=0, bits=8, lanes=1),
                shape=self.shape,
                strides=None,
                byte_offset=0,
            ),
            manager_ctx=None,
            deleter=_managed_tensor_deleter,
        )

    @property
    def address(self) -> int:
        return ctypes.addressof(self.managed)

    def close(self) -> None:
        pointer, self.pointer = self.pointer, ctypes.c_void_p()
        if pointer.value:
            self.driver.close(pointer)


_LIVE_IMPORTS: dict[int, _ManagedImport] = {}
_LIVE_IMPORTS_LOCK = threading.Lock()
_DLPACK_CAPSULE_NAME = b"dltensor"


@_DLManagedTensorDeleter
def _managed_tensor_deleter(managed: ctypes.POINTER(_DLManagedTensor)) -> None:
    try:
        address = ctypes.addressof(managed.contents)
        with _LIVE_IMPORTS_LOCK:
            imported = _LIVE_IMPORTS.pop(address, None)
        if imported is not None:
            imported.close()
    except Exception:
        # ctypes callbacks cannot propagate into PyTorch's C++ storage deleter.
        pass


class CudaIpcBuffer:
    """A torch tensor whose storage lifetime owns one CUDA IPC mapping."""

    def __init__(self, descriptor: str, allocation_bytes: int) -> None:
        try:
            handle_bytes = base64.b64decode(descriptor, validate=True)
        except (ValueError, binascii.Error) as error:
            raise SharedPoolImportError("invalid base64 CUDA IPC descriptor") from error
        if len(handle_bytes) != 64:
            raise SharedPoolImportError("CUDA IPC descriptor must decode to 64 bytes")
        try:
            import torch
        except ImportError as error:
            raise SharedPoolImportError("shared_pool requires PyTorch") from error
        if not torch.cuda.is_available():
            raise SharedPoolImportError("shared_pool requires an active CUDA worker")
        torch_device_id = int(torch.cuda.current_device())
        driver = _CudaDriver()
        pointer = driver.open(handle_bytes)
        imported = _ManagedImport(
            driver, pointer, allocation_bytes, torch_device_id
        )
        self._managed_address = imported.address
        with _LIVE_IMPORTS_LOCK:
            _LIVE_IMPORTS[imported.address] = imported

        capsule_new = ctypes.pythonapi.PyCapsule_New
        capsule_new.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]
        capsule_new.restype = ctypes.py_object
        try:
            # CPython requires the capsule name storage to outlive the capsule;
            # use the module-level bytes object instead of a temporary value.
            capsule = capsule_new(imported.address, _DLPACK_CAPSULE_NAME, None)
            self.tensor = torch.utils.dlpack.from_dlpack(capsule)
        except Exception:
            with _LIVE_IMPORTS_LOCK:
                _LIVE_IMPORTS.pop(imported.address, None)
            imported.close()
            raise
        self._torch = torch

    def release(self) -> None:
        if self.tensor is None:
            return
        self._torch.cuda.synchronize(self.tensor.device)
        # Per-layer views may retain the storage after this base reference is
        # gone. PyTorch invokes the DLPack deleter when the last view dies.
        self.tensor = None

    @property
    def mapping_open(self) -> bool:
        """Whether a tensor view still keeps this CUDA IPC mapping alive."""

        with _LIVE_IMPORTS_LOCK:
            return self._managed_address in _LIVE_IMPORTS


class VllmSharedPoolHook:
    """Replace only vLLM's raw packed-buffer allocation function."""

    _MODULES = (
        "vllm.v1.worker.utils",
        "vllm.v1.worker.gpu_model_runner",
        "vllm.v1.worker.gpu.attn_utils",
    )

    def __init__(self, binding: Mapping[str, Any], kv_cache_config: Any) -> None:
        allocation_bytes, _, _ = vllm_backing_geometry(kv_cache_config)
        self._expected_bytes = allocation_bytes
        self._binding_id = str(binding["binding_id"])
        self._layer_identity: dict[str, tuple[str, int]] = {}
        next_layer_index = 0
        for group_index, group in enumerate(kv_cache_config.kv_cache_groups):
            for layer_name in group.layer_names:
                name = str(layer_name)
                if name in self._layer_identity:
                    raise SharedPoolImportError(
                        f"vLLM KV layer {name!r} appears in more than one cache group"
                    )
                self._layer_identity[name] = (
                    f"vllm.group.{group_index}",
                    next_layer_index,
                )
                next_layer_index += 1
        self._buffer = CudaIpcBuffer(str(binding["descriptor"]), allocation_bytes)
        self._used = False
        self._patches: list[tuple[Any, Any]] = []
        self._replacement = self._allocate_kv_cache
        self._worker_utils: Any | None = None
        try:
            self._install()
        except Exception:
            # A partially applied monkeypatch is more dangerous than refusing
            # startup, so restore every imported module before propagating.
            self.shutdown()
            raise

    def _install(self) -> None:
        canonical_allocator: Any | None = None
        for module_name in self._MODULES:
            try:
                module = importlib.import_module(module_name)
            except ImportError:
                continue
            allocator = getattr(module, "allocate_kv_cache", None)
            if not callable(allocator):
                continue
            parameters = list(inspect.signature(allocator).parameters)
            if parameters[:3] != ["kv_cache_config", "device", "layout"] or any(
                parameter
                not in {"kv_cache_config", "device", "layout", "kernel_block_sizes"}
                for parameter in parameters
            ):
                raise SharedPoolImportError(
                    f"unsupported vLLM allocate_kv_cache signature in {module_name}: {parameters}"
                )
            if module_name == "vllm.v1.worker.utils":
                self._worker_utils = module
                canonical_allocator = allocator
            original = (
                canonical_allocator
                if allocator is self._replacement and canonical_allocator is not None
                else allocator
            )
            self._patches.append((module, original))
            setattr(module, "allocate_kv_cache", self._replacement)
        if self._worker_utils is None or not self._patches:
            self.shutdown()
            raise SharedPoolImportError(
                "this vLLM build has no supported packed KV allocation hook"
            )

    def _allocate_kv_cache(
        self,
        kv_cache_config: Any,
        device: Any,
        layout: Any,
        kernel_block_sizes: list[int] | None = None,
    ) -> dict[str, Any]:
        if self._used:
            raise SharedPoolImportError(
                "vLLM attempted to allocate the imported shared KV pool more than once"
            )
        allocation_bytes, _, _ = vllm_backing_geometry(kv_cache_config)
        if allocation_bytes != self._expected_bytes:
            raise SharedPoolImportError(
                "vLLM changed packed KV geometry after Kapsl registration"
            )
        raw = self._buffer.tensor
        if raw is None:
            raise SharedPoolImportError("CUDA IPC mapping was released before allocation")
        requested_device = self._buffer._torch.device(device)
        raw_device = raw.device
        if requested_device.type != "cuda" or (
            requested_device.index is not None
            and requested_device.index != raw_device.index
        ):
            raise SharedPoolImportError(
                f"vLLM requested KV allocation on {requested_device}, but the imported pool is on {raw_device}"
            )
        worker_utils = self._worker_utils
        if worker_utils is None:
            raise SharedPoolImportError("vLLM worker utilities are unavailable")

        caches: dict[str, Any] = {}
        for tensor in kv_cache_config.kv_cache_tensors:
            layers = getattr(tensor, "layers", None)
            if not layers:
                raise SharedPoolImportError(
                    "unsupported vLLM KV tensor placement without layers"
                )
            layer_name = layers[0]
            group_id, group = next(
                (
                    (index, candidate)
                    for index, candidate in enumerate(kv_cache_config.kv_cache_groups)
                    if layer_name in candidate.layer_names
                ),
                (None, None),
            )
            if group is None or group_id is None:
                raise SharedPoolImportError(
                    f"vLLM KV tensor layer {layer_name!r} has no cache group"
                )
            spec = group.kv_cache_spec
            per_layer_specs = getattr(spec, "kv_cache_specs", None)
            if isinstance(per_layer_specs, Mapping):
                spec = per_layer_specs[layer_name]
            kernel_block_size = None
            if kernel_block_sizes is not None and group_id < len(kernel_block_sizes):
                kernel_block_size = kernel_block_sizes[group_id]
            views = worker_utils.create_kv_cache_views(
                raw,
                spec,
                kv_cache_config.num_blocks,
                layout,
                tensor,
                kernel_block_size=kernel_block_size,
            )
            if len(views) != len(tensor.layers):
                raise SharedPoolImportError(
                    "vLLM returned an unexpected number of views for a KV tensor placement"
                )
            caches.update(zip(tensor.layers, views))
        self._used = True
        return caches

    @property
    def used(self) -> bool:
        return self._used

    @property
    def binding_id(self) -> str:
        return self._binding_id

    @property
    def imported_bytes(self) -> int:
        return self._expected_bytes

    def attachment_views(self, kv_caches: Mapping[str, Any]) -> list[dict[str, Any]]:
        """Prove every registered vLLM KV tensor aliases the imported storage."""

        if not self._used:
            raise SharedPoolImportError(
                "vLLM cannot attach before consuming the imported KV allocation"
            )
        raw = self._buffer.tensor
        if raw is None:
            raise SharedPoolImportError("CUDA IPC mapping was released before attachment")
        expected_layers = set(self._layer_identity)
        actual_layers = set(kv_caches)
        if actual_layers != expected_layers:
            missing = sorted(expected_layers - actual_layers)
            unexpected = sorted(actual_layers - expected_layers)
            raise SharedPoolImportError(
                "vLLM registered KV layers that differ from the negotiated topology: "
                f"missing={missing}, unexpected={unexpected}"
            )

        raw_storage = raw.untyped_storage()
        raw_base = int(raw_storage.data_ptr())
        if int(raw_storage.nbytes()) != self._expected_bytes:
            raise SharedPoolImportError(
                "imported CUDA storage size changed before attachment"
            )
        views: list[dict[str, Any]] = []
        for layer_name in sorted(expected_layers, key=lambda name: self._layer_identity[name][1]):
            tensor = kv_caches[layer_name]
            storage = tensor.untyped_storage()
            if int(storage.data_ptr()) != raw_base:
                raise SharedPoolImportError(
                    f"vLLM KV tensor {layer_name!r} does not alias the imported CUDA storage"
                )
            offset_bytes = int(tensor.data_ptr()) - raw_base
            length_bytes = _tensor_span_bytes(tensor)
            if offset_bytes < 0 or offset_bytes + length_bytes > self._expected_bytes:
                raise SharedPoolImportError(
                    f"vLLM KV tensor {layer_name!r} extends outside the imported allocation"
                )
            group_id, layer_index = self._layer_identity[layer_name]
            views.append(
                {
                    "group_id": group_id,
                    "layer": {"index": layer_index, "name": layer_name},
                    "offset_bytes": offset_bytes,
                    "length_bytes": length_bytes,
                }
            )
        return views

    def shutdown(self) -> None:
        replacement = getattr(self, "_replacement", None)
        for module, original in reversed(getattr(self, "_patches", [])):
            if getattr(module, "allocate_kv_cache", None) is replacement:
                setattr(module, "allocate_kv_cache", original)
        self._patches = []
        buffer = getattr(self, "_buffer", None)
        if buffer is not None:
            buffer.release()


def _tensor_span_bytes(tensor: Any) -> int:
    shape = tuple(int(value) for value in tensor.shape)
    strides = tuple(int(value) for value in tensor.stride())
    if not shape or len(shape) != len(strides) or any(size <= 0 for size in shape):
        raise SharedPoolImportError("vLLM KV tensors must have a non-empty shape")
    if any(stride < 0 for stride in strides):
        raise SharedPoolImportError("negative-stride vLLM KV tensors are unsupported")
    element_size = int(tensor.element_size())
    if element_size <= 0:
        raise SharedPoolImportError("vLLM KV tensor element size must be positive")
    final_element = sum((size - 1) * stride for size, stride in zip(shape, strides))
    return (final_element + 1) * element_size
