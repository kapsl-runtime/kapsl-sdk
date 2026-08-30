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
import functools
import importlib
import inspect
import logging
import os
import sys
import threading
from collections.abc import Mapping
from typing import Any


# vLLM's default logging configuration attaches its only output handler to the
# ``vllm`` logger. A top-level plugin logger otherwise propagates into an empty
# root logger, which discards CUDA VMM evidence even though the checks ran.
logger = logging.getLogger(f"vllm.{__name__}")


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
    live_resize: bool = False,
) -> dict[str, Any]:
    """Select and validate the physical replica imported by one worker."""

    allocation_bytes, num_blocks, bytes_per_block = vllm_backing_geometry(
        kv_cache_config
    )
    pools = receipt.get("shared_pools")
    if not isinstance(pools, list) or not pools:
        raise SharedPoolImportError("Kapsl returned no shared-pool bindings")
    expected_transport = "cuda_vmm" if live_resize else "cuda_ipc"
    candidates = [
        dict(pool)
        for pool in pools
        if isinstance(pool, Mapping)
        and pool.get("capacity_pool_id") == "vllm.pool.0"
        and isinstance(pool.get("memory_domain"), Mapping)
        and pool["memory_domain"].get("kind") == "cuda"
        and isinstance(pool.get("transport"), Mapping)
        and pool["transport"].get("kind") == expected_transport
    ]
    if not candidates:
        raise SharedPoolImportError(
            f"Kapsl receipt has no {expected_transport} binding for vllm.pool.0"
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
    elastic = binding.get("elastic")
    if live_resize:
        if not isinstance(elastic, Mapping):
            raise SharedPoolImportError(
                "CUDA VMM binding is missing elastic pool geometry"
            )
        if int(elastic.get("maximum_block_count", 0)) != num_blocks:
            raise SharedPoolImportError(
                "Kapsl elastic maximum does not match vLLM's virtual allocator"
            )
        minimum_blocks = int(elastic.get("minimum_block_count", 0))
        mapped_blocks = int(elastic.get("mapped_block_count", 0))
        if (
            minimum_blocks <= 0
            or minimum_blocks > mapped_blocks
            or mapped_blocks > num_blocks
        ):
            raise SharedPoolImportError(
                "Kapsl elastic minimum/mapped block counts are outside virtual capacity"
            )
    elif elastic is not None:
        raise SharedPoolImportError(
            "fixed CUDA IPC binding cannot carry elastic pool geometry"
        )
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
        self._expected_bytes = allocation_bytes
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

    @property
    def mapped_bytes(self) -> int:
        return int(self._expected_bytes)


class _CudaVmmDriver:
    """Minimal CUDA driver VMM binding used by the certified Linux profile."""

    class _Location(ctypes.Structure):
        _fields_ = [("type", ctypes.c_int), ("id", ctypes.c_int)]

    class _AccessDesc(ctypes.Structure):
        pass

    _AccessDesc._fields_ = [
        ("location", _Location),
        ("flags", ctypes.c_uint64),
    ]

    def __init__(self) -> None:
        if not sys.platform.startswith("linux"):
            raise SharedPoolImportError("CUDA VMM shared_pool is supported on Linux only")
        library_name = ctypes.util.find_library("cuda") or "libcuda.so.1"
        try:
            self.library = ctypes.CDLL(library_name)
        except OSError as error:
            raise SharedPoolImportError(f"cannot load CUDA driver: {error}") from error
        required = (
            "cuInit",
            "cuMemAddressReserve",
            "cuMemAddressFree",
            "cuMemImportFromShareableHandle",
            "cuMemMap",
            "cuMemUnmap",
            "cuMemSetAccess",
            "cuMemRelease",
        )
        missing = [name for name in required if not hasattr(self.library, name)]
        if missing:
            raise SharedPoolImportError(
                f"CUDA driver lacks required VMM symbols: {', '.join(missing)}"
            )
        self.library.cuInit.argtypes = [ctypes.c_uint]
        self.library.cuInit.restype = ctypes.c_int
        self.library.cuMemAddressReserve.argtypes = [
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_uint64,
            ctypes.c_uint64,
        ]
        self.library.cuMemAddressReserve.restype = ctypes.c_int
        self.library.cuMemAddressFree.argtypes = [ctypes.c_uint64, ctypes.c_size_t]
        self.library.cuMemAddressFree.restype = ctypes.c_int
        self.library.cuMemImportFromShareableHandle.argtypes = [
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.c_void_p,
            ctypes.c_uint,
        ]
        self.library.cuMemImportFromShareableHandle.restype = ctypes.c_int
        self.library.cuMemMap.argtypes = [
            ctypes.c_uint64,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_uint64,
            ctypes.c_uint64,
        ]
        self.library.cuMemMap.restype = ctypes.c_int
        self.library.cuMemUnmap.argtypes = [ctypes.c_uint64, ctypes.c_size_t]
        self.library.cuMemUnmap.restype = ctypes.c_int
        self.library.cuMemSetAccess.argtypes = [
            ctypes.c_uint64,
            ctypes.c_size_t,
            ctypes.POINTER(self._AccessDesc),
            ctypes.c_size_t,
        ]
        self.library.cuMemSetAccess.restype = ctypes.c_int
        self.library.cuMemRelease.argtypes = [ctypes.c_uint64]
        self.library.cuMemRelease.restype = ctypes.c_int
        self._check(self.library.cuInit(0), "initialize CUDA driver")

    def reserve(self, size: int, alignment: int) -> int:
        pointer = ctypes.c_uint64()
        self._check(
            self.library.cuMemAddressReserve(
                ctypes.byref(pointer), size, alignment, 0, 0
            ),
            "reserve CUDA virtual address",
        )
        if pointer.value == 0:
            raise SharedPoolImportError("CUDA VMM reservation returned a null address")
        return int(pointer.value)

    def import_fd(self, descriptor: int) -> int:
        handle = ctypes.c_uint64()
        # CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR = 1.
        self._check(
            self.library.cuMemImportFromShareableHandle(
                ctypes.byref(handle), ctypes.c_void_p(descriptor), 1
            ),
            "import CUDA VMM allocation handle",
        )
        return int(handle.value)

    def map(self, address: int, length: int, handle: int, device_id: int) -> None:
        self._check(
            self.library.cuMemMap(address, length, 0, handle, 0),
            "map CUDA VMM allocation",
        )
        access = self._AccessDesc(
            location=self._Location(type=1, id=device_id),
            # CU_MEM_ACCESS_FLAGS_PROT_READWRITE = 3.
            flags=3,
        )
        try:
            self._check(
                self.library.cuMemSetAccess(
                    address, length, ctypes.byref(access), 1
                ),
                "set CUDA VMM access",
            )
        except Exception:
            self.library.cuMemUnmap(address, length)
            raise

    def unmap(self, address: int, length: int) -> None:
        self._check(self.library.cuMemUnmap(address, length), "unmap CUDA VMM range")

    def release_handle(self, handle: int) -> None:
        self._check(self.library.cuMemRelease(handle), "release CUDA VMM handle")

    def free_address(self, address: int, length: int) -> None:
        self._check(
            self.library.cuMemAddressFree(address, length),
            "free CUDA virtual address",
        )

    @staticmethod
    def _check(result: int, operation: str) -> None:
        if result != 0:
            raise SharedPoolImportError(f"failed to {operation}: CUDA error {result}")


class _ManagedVmmImport:
    def __init__(
        self,
        driver: _CudaVmmDriver,
        virtual_bytes: int,
        granularity: int,
        torch_device_id: int,
    ) -> None:
        self.driver = driver
        self.virtual_bytes = virtual_bytes
        self.granularity = granularity
        self.device_id = torch_device_id
        self.pointer = driver.reserve(virtual_bytes, granularity)
        self.segments: dict[str, tuple[int, int, int]] = {}
        self.shape = (ctypes.c_int64 * 1)(virtual_bytes)
        self.managed = _DLManagedTensor(
            dl_tensor=_DLTensor(
                data=ctypes.c_void_p(self.pointer),
                device=_DLDevice(device_type=2, device_id=torch_device_id),
                ndim=1,
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

    def map_segments(
        self, segments: list[Mapping[str, Any]], handles: list[int]
    ) -> None:
        imported: list[tuple[str, int, int, int]] = []
        try:
            for segment in sorted(segments, key=lambda item: int(item["offset_bytes"])):
                segment_id = str(segment["segment_id"])
                offset = int(segment["offset_bytes"])
                length = int(segment["length_bytes"])
                handle_index = int(segment["handle_index"])
                if (
                    not segment_id
                    or segment_id in self.segments
                    or offset < 0
                    or length <= 0
                    or offset % self.granularity
                    or length % self.granularity
                    or offset + length > self.virtual_bytes
                    or handle_index < 0
                    or handle_index >= len(handles)
                ):
                    raise SharedPoolImportError("invalid CUDA VMM segment descriptor")
                handle = self.driver.import_fd(handles[handle_index])
                try:
                    self.driver.map(
                        self.pointer + offset, length, handle, self.device_id
                    )
                except Exception:
                    self.driver.release_handle(handle)
                    raise
                imported.append((segment_id, offset, length, handle))
            for segment_id, offset, length, handle in imported:
                self.segments[segment_id] = (offset, length, handle)
        except Exception:
            for _, offset, length, handle in reversed(imported):
                try:
                    self.driver.unmap(self.pointer + offset, length)
                finally:
                    self.driver.release_handle(handle)
            raise

    def unmap_segments(self, segments: list[Mapping[str, Any]]) -> None:
        resolved: list[tuple[str, int, int, int]] = []
        for segment in segments:
            segment_id = str(segment["segment_id"])
            current = self.segments.get(segment_id)
            if current is None or current[:2] != (
                int(segment["offset_bytes"]),
                int(segment["length_bytes"]),
            ):
                raise SharedPoolImportError(
                    "CUDA VMM unmap does not match an imported segment"
                )
            resolved.append((segment_id, *current))
        for segment_id, offset, length, handle in sorted(
            resolved, key=lambda item: item[1], reverse=True
        ):
            self.driver.unmap(self.pointer + offset, length)
            self.driver.release_handle(handle)
            del self.segments[segment_id]

    def close(self) -> None:
        pointer, self.pointer = self.pointer, 0
        if pointer == 0:
            return
        first_error: Exception | None = None
        for segment_id, (offset, length, handle) in sorted(
            self.segments.items(), key=lambda item: item[1][0], reverse=True
        ):
            try:
                self.driver.unmap(pointer + offset, length)
            except Exception as error:
                first_error = first_error or error
            try:
                self.driver.release_handle(handle)
            except Exception as error:
                first_error = first_error or error
            del self.segments[segment_id]
        try:
            self.driver.free_address(pointer, self.virtual_bytes)
        except Exception as error:
            first_error = first_error or error
        if first_error is not None:
            raise first_error


class CudaVmmBuffer:
    """Stable virtual CUDA storage backed by a resizable physical prefix."""

    def __init__(
        self,
        binding: Mapping[str, Any],
        handles: list[int],
        *,
        conformance: bool = False,
        driver_factory: Any = _CudaVmmDriver,
    ) -> None:
        if not isinstance(conformance, bool):
            raise SharedPoolImportError("CUDA VMM conformance mode must be a boolean")
        elastic = binding.get("elastic")
        if not isinstance(elastic, Mapping):
            raise SharedPoolImportError("CUDA VMM binding has no elastic geometry")
        virtual_bytes = int(binding["block_count"]) * int(binding["bytes_per_block"])
        mapped_bytes = int(elastic["mapped_block_count"]) * int(
            binding["bytes_per_block"]
        )
        granularity = int(elastic["allocation_granularity_bytes"])
        try:
            import torch
        except ImportError as error:
            raise SharedPoolImportError("shared_pool requires PyTorch") from error
        if not torch.cuda.is_available():
            raise SharedPoolImportError("shared_pool requires an active CUDA worker")
        torch_device_id = int(torch.cuda.current_device())
        imported = _ManagedVmmImport(
            driver_factory(), virtual_bytes, granularity, torch_device_id
        )
        try:
            imported.map_segments(list(elastic["segments"]), handles)
            capsule_new = ctypes.pythonapi.PyCapsule_New
            capsule_new.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]
            capsule_new.restype = ctypes.py_object
            capsule = capsule_new(imported.address, _DLPACK_CAPSULE_NAME, None)
            self.tensor = torch.utils.dlpack.from_dlpack(capsule)
        except Exception:
            imported.close()
            raise
        self._managed_address = imported.address
        with _LIVE_IMPORTS_LOCK:
            _LIVE_IMPORTS[imported.address] = imported
        self._imported = imported
        self._torch = torch
        self._virtual_bytes = virtual_bytes
        self._mapped_bytes = mapped_bytes
        self._granularity = granularity
        self._conformance = conformance
        if self._conformance:
            self._verify_zero_segments(list(elastic["segments"]), "initial")
            logger.warning(
                "KAPSL_VMM_CONFORMANCE stable_address=%#x mapped_bytes=%d virtual_bytes=%d phase=initial zeroed=true",
                imported.pointer,
                mapped_bytes,
                virtual_bytes,
            )

    @property
    def mapped_bytes(self) -> int:
        return self._mapped_bytes

    def map_segments(
        self, segments: list[Mapping[str, Any]], handles: list[int], target_bytes: int
    ) -> None:
        if target_bytes <= self._mapped_bytes or target_bytes > self._virtual_bytes:
            raise SharedPoolImportError("invalid CUDA VMM growth target")
        self._torch.cuda.synchronize(self.tensor.device)
        self._imported.map_segments(segments, handles)
        if self._conformance:
            self._verify_zero_segments(segments, "grow")
            logger.warning(
                "KAPSL_VMM_CONFORMANCE stable_address=%#x mapped_bytes=%d virtual_bytes=%d phase=grow zeroed=true",
                self._imported.pointer,
                target_bytes,
                self._virtual_bytes,
            )
        self._mapped_bytes = target_bytes

    def unmap_segments(
        self, segments: list[Mapping[str, Any]], target_bytes: int
    ) -> None:
        if target_bytes <= 0 or target_bytes >= self._mapped_bytes:
            raise SharedPoolImportError("invalid CUDA VMM shrink target")
        self._torch.cuda.synchronize(self.tensor.device)
        self._imported.unmap_segments(segments)
        self._mapped_bytes = target_bytes
        if self._conformance:
            logger.warning(
                "KAPSL_VMM_CONFORMANCE stable_address=%#x mapped_bytes=%d virtual_bytes=%d phase=shrink",
                self._imported.pointer,
                target_bytes,
                self._virtual_bytes,
            )

    def release(self) -> None:
        if self.tensor is None:
            return
        self._torch.cuda.synchronize(self.tensor.device)
        self.tensor = None

    @property
    def mapping_open(self) -> bool:
        with _LIVE_IMPORTS_LOCK:
            return self._managed_address in _LIVE_IMPORTS

    def _verify_zero_segments(
        self, segments: list[Mapping[str, Any]], phase: str
    ) -> None:
        for segment in segments:
            start = int(segment["offset_bytes"])
            length = int(segment["length_bytes"])
            nonzero = int(self._torch.count_nonzero(self.tensor[start : start + length]))
            if nonzero != 0:
                raise SharedPoolImportError(
                    f"CUDA VMM {phase} segment {segment['segment_id']!r} was not zeroed"
                )
        self._torch.cuda.synchronize(self.tensor.device)


class VllmSharedPoolHook:
    """Replace only vLLM's raw packed-buffer allocation function."""

    _MODULES = (
        "vllm.v1.worker.utils",
        "vllm.v1.worker.gpu_model_runner",
        "vllm.v1.worker.gpu.attn_utils",
    )
    _WARMUP_MODULES = (
        "vllm.v1.worker.gpu.warmup",
        "vllm.v1.worker.gpu_worker",
    )

    def __init__(
        self,
        binding: Mapping[str, Any],
        kv_cache_config: Any,
        *,
        handles: list[int] | None = None,
        conformance: bool = False,
    ) -> None:
        if not isinstance(conformance, bool):
            raise SharedPoolImportError("shared-pool conformance mode must be a boolean")
        allocation_bytes, maximum_blocks, _ = vllm_backing_geometry(kv_cache_config)
        self._expected_bytes = allocation_bytes
        self._maximum_blocks = maximum_blocks
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
        transport = binding.get("transport")
        transport_kind = transport.get("kind") if isinstance(transport, Mapping) else None
        if conformance and transport_kind != "cuda_vmm":
            raise SharedPoolImportError(
                "shared-pool conformance mode requires CUDA VMM transport"
            )
        self._conformance = conformance
        self._startup_mapped_blocks: int | None = None
        if transport_kind == "cuda_vmm":
            if handles is None:
                raise SharedPoolImportError(
                    "CUDA VMM binding requires out-of-band allocation handles"
                )
            elastic = binding.get("elastic")
            if not isinstance(elastic, Mapping):
                raise SharedPoolImportError("CUDA VMM binding has no elastic geometry")
            self._startup_mapped_blocks = int(elastic["mapped_block_count"])
            self._buffer = CudaVmmBuffer(
                binding, handles, conformance=self._conformance
            )
        elif transport_kind == "cuda_ipc":
            if handles:
                raise SharedPoolImportError(
                    "fixed CUDA IPC binding cannot consume OS handles"
                )
            self._buffer = CudaIpcBuffer(str(binding["descriptor"]), allocation_bytes)
        else:
            raise SharedPoolImportError("unsupported shared-pool CUDA transport")
        self._last_worker_resize: tuple[Any, ...] | None = None
        self._used = False
        self._patches: list[tuple[Any, Any]] = []
        self._warmup_patches: list[tuple[Any, Any]] = []
        self._warmup_replacement: Any | None = None
        self._startup_warmup_lock = threading.Lock()
        self._startup_warmup_attempted = False
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
        if (
            self._startup_mapped_blocks is not None
            and self._startup_mapped_blocks < self._maximum_blocks
        ):
            self._install_startup_warmup_cap()

    def _install_startup_warmup_cap(self) -> None:
        """Keep pinned vLLM's synthetic warmup inside mapped VMM pages.

        vLLM sizes its hand-built warmup batch from ``KVCacheConfig.num_blocks``.
        For an elastic pool that value is the virtual maximum, while only the
        initial physical prefix is mapped before the engine becomes active.
        Patch both the defining module and the symbol imported by ``gpu_worker``
        so this one startup call sees the mapped prefix. The complete virtual
        block count is restored before normal scheduling starts.
        """

        modules: list[Any] = []
        original: Any | None = None
        expected_parameters = [
            "model_runner",
            "worker_execute_model",
            "worker_sample_tokens",
        ]
        for module_name in self._WARMUP_MODULES:
            try:
                module = importlib.import_module(module_name)
            except ImportError as error:
                raise SharedPoolImportError(
                    f"this vLLM build has no certified startup warmup module {module_name}"
                ) from error
            candidate = getattr(module, "warmup_kernels", None)
            if not callable(candidate):
                raise SharedPoolImportError(
                    f"this vLLM build has no callable {module_name}.warmup_kernels"
                )
            parameters = list(inspect.signature(candidate).parameters)
            if parameters != expected_parameters:
                raise SharedPoolImportError(
                    "unsupported vLLM warmup_kernels signature in "
                    f"{module_name}: {parameters}"
                )
            if original is None:
                original = candidate
            elif candidate is not original:
                raise SharedPoolImportError(
                    "vLLM startup warmup references do not share one pinned function"
                )
            modules.append(module)
        if original is None:
            raise SharedPoolImportError("vLLM startup warmup function is unavailable")

        @functools.wraps(original)
        def capped_warmup(
            model_runner: Any,
            worker_execute_model: Any,
            worker_sample_tokens: Any,
        ) -> Any:
            return self._run_startup_warmup(
                original,
                model_runner,
                worker_execute_model,
                worker_sample_tokens,
            )

        self._warmup_replacement = capped_warmup
        for module in modules:
            self._warmup_patches.append((module, original))
            setattr(module, "warmup_kernels", capped_warmup)

    def _run_startup_warmup(
        self,
        original: Any,
        model_runner: Any,
        worker_execute_model: Any,
        worker_sample_tokens: Any,
    ) -> Any:
        mapped_blocks = self._startup_mapped_blocks
        if mapped_blocks is None:
            raise SharedPoolImportError("fixed shared pool entered elastic warmup")
        with self._startup_warmup_lock:
            if self._startup_warmup_attempted:
                raise SharedPoolImportError(
                    "elastic vLLM startup warmup was invoked more than once"
                )
            self._startup_warmup_attempted = True
            kv_cache_config = getattr(model_runner, "kv_cache_config", None)
            advertised_blocks = getattr(kv_cache_config, "num_blocks", None)
            if (
                isinstance(advertised_blocks, bool)
                or not isinstance(advertised_blocks, int)
                or advertised_blocks != self._maximum_blocks
            ):
                raise SharedPoolImportError(
                    "vLLM changed the virtual KV block count before startup warmup"
                )
            if mapped_blocks <= 1 or mapped_blocks > advertised_blocks:
                raise SharedPoolImportError(
                    "elastic startup mapped block count is outside virtual capacity"
                )
            try:
                kv_cache_config.num_blocks = mapped_blocks
            except Exception as error:
                raise SharedPoolImportError(
                    "vLLM KVCacheConfig cannot be capped for elastic startup warmup"
                ) from error
            warmup_logger = (
                logger.warning
                if self._conformance
                else logger.info
            )
            warmup_logger(
                "capped vLLM startup warmup to %d mapped blocks out of %d virtual blocks",
                mapped_blocks,
                advertised_blocks,
            )
            try:
                return original(
                    model_runner,
                    worker_execute_model,
                    worker_sample_tokens,
                )
            finally:
                try:
                    kv_cache_config.num_blocks = advertised_blocks
                except Exception as error:
                    raise SharedPoolImportError(
                        "vLLM KVCacheConfig could not restore virtual capacity after warmup"
                    ) from error

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

        conformance = self._conformance
        allocated_before = (
            int(self._buffer._torch.cuda.memory_allocated(raw.device))
            if conformance
            else 0
        )
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
        if conformance:
            allocated_after = int(
                self._buffer._torch.cuda.memory_allocated(raw.device)
            )
            if allocated_after != allocated_before:
                raise SharedPoolImportError(
                    "vLLM KV view construction created a second PyTorch CUDA allocation: "
                    f"before={allocated_before}, after={allocated_after}"
                )
            logger.warning(
                "KAPSL_VMM_CONFORMANCE allocator_delta_bytes=0 virtual_bytes=%d",
                self._expected_bytes,
            )
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

    @property
    def mapped_bytes(self) -> int:
        return int(self._buffer.mapped_bytes)

    def apply_worker_resize(
        self, operation: Mapping[str, Any], handles: list[int]
    ) -> None:
        if not isinstance(self._buffer, CudaVmmBuffer):
            raise SharedPoolImportError("fixed CUDA IPC pool cannot be resized")
        if str(operation.get("binding_id")) != self._binding_id:
            raise SharedPoolImportError("resize operation targets a different binding")
        target_bytes = int(operation["target_block_count"]) * int(
            operation["bytes_per_block"]
        )
        stage = str(operation.get("stage"))
        segments = [dict(segment) for segment in operation.get("segments", [])]
        generation = int(operation["resize_generation"])
        operation_key = (
            generation,
            stage,
            target_bytes,
            tuple(
                (
                    str(segment["segment_id"]),
                    int(segment["offset_bytes"]),
                    int(segment["length_bytes"]),
                    int(segment["handle_index"]),
                )
                for segment in segments
            ),
        )
        previous = self._last_worker_resize
        if previous == operation_key:
            # The physical mutation completed but its acknowledgement may have
            # been lost. Exact coordinator replays must be harmless.
            return
        if previous is not None and generation <= int(previous[0]):
            raise SharedPoolImportError(
                "worker received a non-monotonic or altered resize replay"
            )
        if stage == "map_workers":
            self._buffer.map_segments(segments, handles, target_bytes)
        elif stage == "unmap_workers":
            if handles:
                raise SharedPoolImportError("CUDA VMM unmap must not carry handles")
            self._buffer.unmap_segments(segments, target_bytes)
        else:
            raise SharedPoolImportError("worker received a scheduler resize stage")
        self._last_worker_resize = operation_key

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
        warmup_replacement = getattr(self, "_warmup_replacement", None)
        for module, original in reversed(getattr(self, "_warmup_patches", [])):
            if getattr(module, "warmup_kernels", None) is warmup_replacement:
                setattr(module, "warmup_kernels", original)
        self._warmup_patches = []
        replacement = getattr(self, "_replacement", None)
        for module, original in reversed(getattr(self, "_patches", [])):
            if getattr(module, "allocate_kv_cache", None) is replacement:
                setattr(module, "allocate_kv_cache", original)
        self._patches = []
        buffer = getattr(self, "_buffer", None)
        if buffer is not None:
            buffer.release()


class VllmElasticBlockPool:
    """Narrow adapter around pinned vLLM's native block allocator."""

    def __init__(self, initial_blocks: int, maximum_blocks: int) -> None:
        if initial_blocks <= 1 or initial_blocks > maximum_blocks:
            raise SharedPoolImportError(
                "elastic block pool requires room for vLLM's null block"
            )
        self.initial_blocks = initial_blocks
        self.maximum_blocks = maximum_blocks
        self.current_blocks = initial_blocks
        self._pool: Any | None = None

    def bind(self, pool: Any) -> None:
        if self._pool is not None:
            raise SharedPoolImportError("vLLM block pool was bound more than once")
        blocks = getattr(pool, "blocks", None)
        queue = getattr(pool, "free_block_queue", None)
        if (
            not isinstance(blocks, list)
            or len(blocks) != self.maximum_blocks
            or int(getattr(pool, "num_gpu_blocks", 0)) != self.maximum_blocks
            or queue is None
            or not callable(getattr(queue, "remove", None))
            or not callable(getattr(queue, "append_n", None))
        ):
            raise SharedPoolImportError(
                "this vLLM build has no certified resizable BlockPool surface"
            )
        inactive = blocks[self.initial_blocks :]
        for block in inactive:
            if int(getattr(block, "ref_cnt", -1)) != 0:
                raise SharedPoolImportError(
                    "inactive vLLM block is unexpectedly referenced at startup"
                )
        removed: list[Any] = []
        try:
            for block in inactive:
                queue.remove(block)
                removed.append(block)
        except Exception as error:
            try:
                queue.append_n(removed)
            except Exception as rollback_error:
                raise SharedPoolImportError(
                    "vLLM inactive-block setup failed and its free-queue "
                    "rollback was ambiguous; the engine must restart"
                ) from rollback_error
            raise SharedPoolImportError(
                "vLLM inactive-block setup could not update the free queue"
            ) from error
        pool.num_gpu_blocks = self.initial_blocks
        self._pool = pool

    def apply(self, target_blocks: int) -> None:
        pool = self._pool
        if pool is None:
            raise SharedPoolImportError("vLLM block pool is not bound")
        if target_blocks <= 1 or target_blocks > self.maximum_blocks:
            raise SharedPoolImportError("resize target is outside virtual block capacity")
        if target_blocks == self.current_blocks:
            return
        blocks = pool.blocks
        queue = pool.free_block_queue
        if target_blocks > self.current_blocks:
            activated = blocks[self.current_blocks : target_blocks]
            if any(int(getattr(block, "ref_cnt", -1)) != 0 for block in activated):
                raise SharedPoolImportError(
                    "inactive vLLM tail block became referenced before activation"
                )
            queue.append_n(activated)
        else:
            retired = blocks[target_blocks : self.current_blocks]
            if any(
                int(getattr(block, "ref_cnt", -1)) != 0
                or bool(getattr(block, "is_null", False))
                for block in retired
            ):
                raise SharedPoolImportError(
                    "vLLM cannot retire a live or null cache block"
                )
            evict = getattr(pool, "_maybe_evict_cached_block", None)
            if not callable(evict):
                raise SharedPoolImportError(
                    "this vLLM build lacks the certified cache-eviction seam"
                )
            removed: list[Any] = []
            try:
                # Remove the complete tail before evicting prefix metadata. If
                # any queue mutation fails, every successful removal can be
                # restored without changing the advertised block capacity.
                for block in retired:
                    queue.remove(block)
                    removed.append(block)
                for block in retired:
                    evict(block)
            except Exception as error:
                try:
                    queue.append_n(removed)
                except Exception as rollback_error:
                    raise SharedPoolImportError(
                        "vLLM tail retirement failed and its free-queue "
                        "rollback was ambiguous; the engine must restart"
                    ) from rollback_error
                raise SharedPoolImportError(
                    "vLLM tail retirement was rolled back before resize "
                    "acknowledgement"
                ) from error
        pool.num_gpu_blocks = target_blocks
        self.current_blocks = target_blocks


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
