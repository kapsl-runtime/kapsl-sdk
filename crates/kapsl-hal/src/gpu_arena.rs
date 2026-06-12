//! GPU memory arena — a single large CUDA allocation sliced into regions.
//!
//! The arena owns one `cudaMalloc` and hands out typed sub-slices via a
//! bump-pointer allocator. This avoids the per-tensor fragmentation that
//! comes from thousands of individual allocations and gives the native
//! backend a contiguous, predictable memory layout.
//!
//! # Regions
//!
//! ```text
//! ┌─────────────────────────────────────────────────────┐
//! │                    GPU Arena                        │
//! ├──────────────────┬──────────────────┬───────────────┤
//! │  Weight region   │  KV block pool   │  Activation   │
//! │  (static, RO)    │  (paged, RW)     │  workspace    │
//! └──────────────────┴──────────────────┴───────────────┘
//! ```

#[cfg(feature = "cuda")]
use cudarc::driver::{CudaDevice, CudaSlice, DevicePtr};
#[cfg(feature = "cuda")]
use std::cell::UnsafeCell;
#[cfg(feature = "cuda")]
use std::sync::atomic::{AtomicUsize, Ordering};
#[cfg(feature = "cuda")]
use std::sync::{Arc, Mutex};

#[cfg(feature = "cuda")]
use thiserror::Error;

#[cfg(feature = "cuda")]
use crate::block_range::BlockRangeAllocator;

#[cfg(feature = "cuda")]
#[derive(Debug, Error)]
pub enum ArenaError {
    #[error("CUDA driver error: {0}")]
    Cuda(#[from] cudarc::driver::DriverError),
    #[error("Arena out of memory: requested {requested} bytes, {available} available")]
    Oom { requested: usize, available: usize },
    #[error("Block pool exhausted: no free blocks remaining")]
    NoFreeBlocks,
    #[error("Block pool has no contiguous run of {requested} blocks ({free} free total)")]
    NoContiguousRun { requested: usize, free: usize },
}

// ─── GpuArena ────────────────────────────────────────────────────────────────

#[cfg(feature = "cuda")]
pub struct GpuArena {
    device: Arc<CudaDevice>,
    buffer: CudaSlice<u8>,
    capacity: usize,
    cursor: usize,
}

#[cfg(feature = "cuda")]
impl GpuArena {
    /// Allocate a new arena of `bytes` on the given CUDA device.
    pub fn new(device: Arc<CudaDevice>, bytes: usize) -> Result<Self, ArenaError> {
        let buffer = device.alloc_zeros::<u8>(bytes)?;
        log::info!("GPU arena allocated: {} MiB", bytes / (1024 * 1024));
        Ok(Self {
            device,
            buffer,
            capacity: bytes,
            cursor: 0,
        })
    }

    /// Bump-allocate `count` elements of type T, aligned to `align` bytes.
    /// Returns the byte offset into the arena.
    pub fn alloc<T: cudarc::driver::DeviceRepr>(
        &mut self,
        count: usize,
        align: usize,
    ) -> Result<usize, ArenaError> {
        let bytes = count * std::mem::size_of::<T>();
        let aligned_cursor = (self.cursor + align - 1) & !(align - 1);
        if aligned_cursor + bytes > self.capacity {
            return Err(ArenaError::Oom {
                requested: bytes,
                available: self.capacity.saturating_sub(aligned_cursor),
            });
        }
        let offset = aligned_cursor;
        self.cursor = aligned_cursor + bytes;
        Ok(offset)
    }

    /// Upload host data to a previously allocated region at `offset`.
    pub fn upload(&mut self, offset: usize, data: &[u8]) -> Result<(), ArenaError> {
        self.device.htod_sync_copy_into(
            data,
            &mut self.buffer.slice_mut(offset..offset + data.len()),
        )?;
        Ok(())
    }

    pub fn remaining(&self) -> usize {
        self.capacity.saturating_sub(self.cursor)
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn used(&self) -> usize {
        self.cursor
    }

    pub fn device(&self) -> &Arc<CudaDevice> {
        &self.device
    }

    pub fn reset(&mut self) {
        self.cursor = 0;
    }
}

// ─── GpuBlockPool ────────────────────────────────────────────────────────────
//
// Paged KV cache on GPU, designed for sharing across multiple backend instances.
//
// Layout: [num_blocks, 2, num_kv_heads, block_size, head_dim] in f16.
//
// Dim 0: physical block index
// Dim 1: 0 = key, 1 = value
// Dim 2: KV head index
// Dim 3: token position within block
// Dim 4: head dimension element
//
// Sharing safety invariant: each physical block is owned by at most one
// session at any time, enforced by the Mutex-protected free set.  Concurrent
// backends writing to *different* physical blocks never alias, so the
// UnsafeCell on storage is sound.

#[cfg(feature = "cuda")]
pub struct GpuBlockPool {
    device: Arc<CudaDevice>,
    // Interior mutability: concurrent writes to disjoint physical blocks are
    // safe at the GPU level.  The free_stack Mutex guarantees each block is
    // allocated to at most one owner at a time.
    storage: UnsafeCell<CudaSlice<half::f16>>,
    num_blocks: usize,
    block_size: usize,
    num_kv_heads: usize,
    head_dim: usize,
    /// Sorted free set; supports single-block and contiguous-run allocation.
    free_blocks: Mutex<BlockRangeAllocator>,
}

#[cfg(feature = "cuda")]
impl std::fmt::Debug for GpuBlockPool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuBlockPool")
            .field("num_blocks", &self.num_blocks)
            .field("block_size", &self.block_size)
            .field("num_kv_heads", &self.num_kv_heads)
            .field("head_dim", &self.head_dim)
            .field("free", &self.free_blocks.lock().unwrap().free_count())
            .finish()
    }
}

// SAFETY: The free_blocks Mutex serialises alloc/free, and the storage
// UnsafeCell is only written to non-overlapping regions (one per allocated
// block, owned exclusively by the session that called alloc_block).
#[cfg(feature = "cuda")]
unsafe impl Sync for GpuBlockPool {}
#[cfg(feature = "cuda")]
unsafe impl Send for GpuBlockPool {}

#[cfg(feature = "cuda")]
impl GpuBlockPool {
    /// Create a new block pool of `num_blocks` blocks on `device`.
    pub fn new(
        device: Arc<CudaDevice>,
        num_blocks: usize,
        block_size: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) -> Result<Self, ArenaError> {
        let elems_per_block = 2 * num_kv_heads * block_size * head_dim;
        let total_elems = num_blocks * elems_per_block;
        let storage = device.alloc_zeros::<half::f16>(total_elems)?;
        let free_blocks = BlockRangeAllocator::new(num_blocks);

        let bytes = total_elems * 2;
        log::info!(
            "GPU block pool: {} blocks × {} tokens = {} MiB ({}h × {}d)",
            num_blocks,
            block_size,
            bytes / (1024 * 1024),
            num_kv_heads,
            head_dim,
        );

        Ok(Self {
            device,
            storage: UnsafeCell::new(storage),
            num_blocks,
            block_size,
            num_kv_heads,
            head_dim,
            free_blocks: Mutex::new(free_blocks),
        })
    }

    /// Allocate a free physical block. Returns the block index.
    pub fn alloc_block(&self) -> Result<u32, ArenaError> {
        self.free_blocks
            .lock()
            .unwrap()
            .alloc()
            .ok_or(ArenaError::NoFreeBlocks)
    }

    /// Release a physical block back to the free pool.
    pub fn free_block(&self, block_id: u32) {
        debug_assert!((block_id as usize) < self.num_blocks);
        self.free_blocks.lock().unwrap().free(block_id);
    }

    /// Allocate `n` *contiguous* physical blocks, returning the first index.
    ///
    /// Used by consumers that need flat device buffers carved out of the pool
    /// (e.g. the ONNX Runtime allocator) rather than paged single blocks.
    pub fn alloc_blocks_contiguous(&self, n: usize) -> Result<u32, ArenaError> {
        let mut free = self.free_blocks.lock().unwrap();
        free.alloc_run(n).ok_or(ArenaError::NoContiguousRun {
            requested: n,
            free: free.free_count(),
        })
    }

    /// Release a contiguous run previously returned by
    /// [`alloc_blocks_contiguous`](Self::alloc_blocks_contiguous).
    pub fn free_blocks_contiguous(&self, first: u32, n: usize) {
        debug_assert!(first as usize + n <= self.num_blocks);
        self.free_blocks.lock().unwrap().free_run(first, n);
    }

    /// Number of free blocks remaining.
    pub fn free_count(&self) -> usize {
        self.free_blocks.lock().unwrap().free_count()
    }

    /// Total number of blocks in the pool.
    pub fn total_blocks(&self) -> usize {
        self.num_blocks
    }

    pub fn block_size(&self) -> usize {
        self.block_size
    }

    pub fn num_kv_heads(&self) -> usize {
        self.num_kv_heads
    }

    pub fn head_dim(&self) -> usize {
        self.head_dim
    }

    /// Elements per physical block (key + value).
    pub fn elems_per_block(&self) -> usize {
        2 * self.num_kv_heads * self.block_size * self.head_dim
    }

    /// Bytes per physical block.
    pub fn bytes_per_block(&self) -> usize {
        self.elems_per_block() * std::mem::size_of::<half::f16>()
    }

    /// Total byte capacity owned by this pool.
    pub fn capacity_bytes(&self) -> usize {
        self.num_blocks.saturating_mul(self.bytes_per_block())
    }

    /// Number of physical blocks currently allocated from the pool.
    pub fn used_count(&self) -> usize {
        self.num_blocks.saturating_sub(self.free_count())
    }

    /// Bytes currently allocated from the pool.
    pub fn used_bytes(&self) -> usize {
        self.used_count().saturating_mul(self.bytes_per_block())
    }

    /// Returns true if this pool has compatible geometry for the given model dimensions.
    pub fn is_compatible(&self, num_kv_heads: usize, head_dim: usize) -> bool {
        self.num_kv_heads == num_kv_heads && self.head_dim == head_dim
    }

    /// Download one block from GPU to a host Vec.
    /// Returned layout: `[K_data || V_data]` where each half is
    /// `[num_kv_heads * block_size * head_dim]` f16 elements.
    pub fn download_block(&self, block_id: u32) -> Result<Vec<half::f16>, ArenaError> {
        let elems = self.elems_per_block();
        let base  = block_id as usize * elems;
        let storage = self.storage();
        Ok(self.device.dtoh_sync_copy(&storage.slice(base..base + elems))?)
    }

    /// Copy one block from this pool to a block in another pool (same or different device).
    /// Routes through host memory — works over PCIe without peer access.
    pub fn copy_block_to_pool(
        &self,
        src_block: u32,
        dst_pool: &GpuBlockPool,
        dst_block: u32,
    ) -> Result<(), ArenaError> {
        let data     = self.download_block(src_block)?;
        let half_len = data.len() / 2;
        dst_pool.upload_block(dst_block, &data[..half_len], &data[half_len..])
    }

    /// Read-only view of the storage slice (for attention kernel reads).
    pub fn storage(&self) -> &CudaSlice<half::f16> {
        unsafe { &*self.storage.get() }
    }

    /// Raw CUDA device pointer for FFI integrations that need to wrap the
    /// externally-owned KV storage without taking ownership.
    pub fn device_base_ptr(&self) -> *mut std::ffi::c_void {
        *self.storage().device_ptr() as *mut std::ffi::c_void
    }

    /// Raw CUDA device pointer to the start of a specific physical block.
    pub fn block_device_ptr(&self, block_id: u32) -> *mut std::ffi::c_void {
        debug_assert!((block_id as usize) < self.num_blocks);
        (self.device_base_ptr() as usize + block_id as usize * self.bytes_per_block())
            as *mut std::ffi::c_void
    }

    /// Mutable view of the storage slice (for KV-write kernels).
    ///
    /// # Safety
    /// Caller must ensure that the physical blocks it writes to are not
    /// concurrently written by another caller.  This invariant is upheld by
    /// the alloc_block/free_block protocol — only the block owner writes.
    pub fn storage_mut(&self) -> &mut CudaSlice<half::f16> {
        unsafe { &mut *self.storage.get() }
    }

    pub fn device(&self) -> &Arc<CudaDevice> {
        &self.device
    }

    /// Upload a single KV block from host f16 data.
    ///
    /// `host_key` layout: `[num_kv_heads, block_size, head_dim]`
    /// `host_val` layout: `[num_kv_heads, block_size, head_dim]`
    pub fn upload_block(
        &self,
        block_id: u32,
        host_key: &[half::f16],
        host_val: &[half::f16],
    ) -> Result<(), ArenaError> {
        let half_block = self.num_kv_heads * self.block_size * self.head_dim;
        assert_eq!(host_key.len(), half_block);
        assert_eq!(host_val.len(), half_block);

        let base = block_id as usize * self.elems_per_block();
        let key_offset = base;
        let val_offset = base + half_block;

        let storage = self.storage_mut();
        self.device.htod_sync_copy_into(
            host_key,
            &mut storage.slice_mut(key_offset..key_offset + half_block),
        )?;
        self.device.htod_sync_copy_into(
            host_val,
            &mut storage.slice_mut(val_offset..val_offset + half_block),
        )?;
        Ok(())
    }
}

// ─── GpuPoolHandle ───────────────────────────────────────────────────────────
//
// A handle to a (possibly shared) GpuBlockPool plus a dynamic per-engine quota.
//
// `blocks_per_engine` is intentionally stored per handle. Cloning a handle
// shares the same quota, while `for_engine()` creates a new handle for the same
// physical pool with an independent quota. Runtime policy can then rebalance
// per-model caps by updating each engine's own atomic.

#[cfg(feature = "cuda")]
#[derive(Clone)]
pub struct GpuPoolHandle {
    pub pool: Arc<GpuBlockPool>,
    /// Maximum blocks this engine may hold simultaneously.
    /// Shared across all engines on the same device; updated on attach/detach.
    pub blocks_per_engine: Arc<AtomicUsize>,
}

#[cfg(feature = "cuda")]
impl GpuPoolHandle {
    /// Wrap a freshly-created private pool (cap = all blocks).
    pub fn private(pool: Arc<GpuBlockPool>) -> Self {
        let cap = pool.total_blocks();
        Self::with_cap(pool, cap)
    }

    /// Wrap an existing pool with an explicit per-engine cap.
    pub fn with_cap(pool: Arc<GpuBlockPool>, cap: usize) -> Self {
        Self {
            pool,
            blocks_per_engine: Arc::new(AtomicUsize::new(cap.max(1))),
        }
    }

    /// Create a new per-engine handle for the same physical pool.
    pub fn for_engine(&self, cap: usize) -> Self {
        Self::with_cap(self.pool.clone(), cap)
    }

    pub fn cap(&self) -> usize {
        self.blocks_per_engine.load(Ordering::Relaxed)
    }

    pub fn set_cap(&self, cap: usize) {
        self.blocks_per_engine.store(cap.max(1), Ordering::Relaxed);
    }
}

// ─── BlockTable ──────────────────────────────────────────────────────────────
//
// Per-sequence mapping from logical block → physical block.

#[cfg(feature = "cuda")]
pub struct BlockTable {
    /// Logical-to-physical block mapping (host side for management).
    table: Vec<i32>,
    /// Length of the table (number of logical blocks allocated).
    len: usize,
}

#[cfg(feature = "cuda")]
impl BlockTable {
    pub fn new() -> Self {
        Self {
            table: Vec::new(),
            len: 0,
        }
    }

    pub fn push(&mut self, physical_block: u32) {
        if self.len < self.table.len() {
            self.table[self.len] = physical_block as i32;
        } else {
            self.table.push(physical_block as i32);
        }
        self.len += 1;
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn as_slice(&self) -> &[i32] {
        &self.table[..self.len]
    }

    pub fn clear(&mut self) {
        self.len = 0;
    }

    /// Upload to GPU as a CudaSlice for kernel consumption.
    pub fn to_device(&self, device: &Arc<CudaDevice>) -> Result<CudaSlice<i32>, ArenaError> {
        Ok(device.htod_sync_copy(&self.table[..self.len])?)
    }
}

#[cfg(feature = "cuda")]
impl Default for BlockTable {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(all(test, feature = "cuda"))]
mod tests {
    use super::*;
    use half::f16;

    // A tiny pool: 8 blocks, 4 tokens/block, 2 KV heads, 8-dim.
    // Requires CUDA device 0.
    fn small_pool() -> Arc<GpuBlockPool> {
        let device = CudaDevice::new(0).expect("CUDA device 0 required for these tests");
        Arc::new(GpuBlockPool::new(device, 8, 4, 2, 8).unwrap())
    }

    #[test]
    fn new_pool_all_blocks_free() {
        let pool = small_pool();
        assert_eq!(pool.total_blocks(), 8);
        assert_eq!(pool.free_count(), 8);
        assert_eq!(pool.used_count(), 0);
    }

    #[test]
    fn alloc_reduces_free_count() {
        let pool = small_pool();
        let b = pool.alloc_block().unwrap();
        assert_eq!(pool.free_count(), 7);
        assert_eq!(pool.used_count(), 1);
        pool.free_block(b);
    }

    #[test]
    fn free_increments_free_count() {
        let pool = small_pool();
        let b = pool.alloc_block().unwrap();
        assert_eq!(pool.free_count(), 7);
        pool.free_block(b);
        assert_eq!(pool.free_count(), 8);
    }

    #[test]
    fn alloc_exhausted_returns_error() {
        let pool = small_pool();
        let blocks: Vec<u32> = (0..8).map(|_| pool.alloc_block().unwrap()).collect();
        assert_eq!(pool.free_count(), 0);
        let err = pool.alloc_block().unwrap_err();
        assert!(matches!(err, ArenaError::NoFreeBlocks));
        for b in blocks { pool.free_block(b); }
    }

    #[test]
    fn contiguous_alloc_and_free() {
        let pool = small_pool();
        let first = pool.alloc_blocks_contiguous(3).unwrap();
        assert_eq!(pool.free_count(), 5);
        // The run is contiguous, so block pointers advance by bytes_per_block.
        let bpb = pool.bytes_per_block();
        let base = pool.block_device_ptr(first) as usize;
        assert_eq!(pool.block_device_ptr(first + 1) as usize, base + bpb);
        assert_eq!(pool.block_device_ptr(first + 2) as usize, base + 2 * bpb);
        pool.free_blocks_contiguous(first, 3);
        assert_eq!(pool.free_count(), 8);
    }

    #[test]
    fn contiguous_alloc_fragmented_fails() {
        let pool = small_pool();
        // Hold every other block so no 2-run exists.
        let all: Vec<u32> = (0..8).map(|_| pool.alloc_block().unwrap()).collect();
        for b in all.iter().filter(|b| *b % 2 == 0) {
            pool.free_block(*b);
        }
        let err = pool.alloc_blocks_contiguous(2).unwrap_err();
        assert!(matches!(err, ArenaError::NoContiguousRun { requested: 2, .. }));
        for b in all.iter().filter(|b| *b % 2 == 1) {
            pool.free_block(*b);
        }
    }

    #[test]
    fn upload_download_roundtrip() {
        let pool = small_pool();
        let half_block = pool.num_kv_heads() * pool.block_size() * pool.head_dim(); // 2*4*8 = 64
        let key: Vec<f16> = (0..half_block).map(|i| f16::from_f32(i as f32)).collect();
        let val: Vec<f16> = (0..half_block).map(|i| f16::from_f32(i as f32 + 100.0)).collect();

        let b = pool.alloc_block().unwrap();
        pool.upload_block(b, &key, &val).unwrap();

        let downloaded = pool.download_block(b).unwrap();
        assert_eq!(&downloaded[..half_block], key.as_slice());
        assert_eq!(&downloaded[half_block..], val.as_slice());
        pool.free_block(b);
    }

    #[test]
    fn uploaded_zeros_overwritten() {
        let pool = small_pool();
        let half_block = pool.num_kv_heads() * pool.block_size() * pool.head_dim();
        let zeros = vec![f16::ZERO; half_block];
        let ones  = vec![f16::ONE; half_block];

        let b = pool.alloc_block().unwrap();
        // First write: all ones
        pool.upload_block(b, &ones, &ones).unwrap();
        // Overwrite with zeros
        pool.upload_block(b, &zeros, &zeros).unwrap();
        let downloaded = pool.download_block(b).unwrap();
        assert!(downloaded.iter().all(|&x| x == f16::ZERO));
        pool.free_block(b);
    }

    #[test]
    fn is_compatible_correct_geometry() {
        let pool = small_pool(); // 2h × 8d
        assert!(pool.is_compatible(2, 8));
        assert!(!pool.is_compatible(2, 16));
        assert!(!pool.is_compatible(4, 8));
    }

    #[test]
    fn copy_block_to_pool_preserves_data() {
        let pool_a = small_pool();
        let pool_b = small_pool();
        let half_block = pool_a.num_kv_heads() * pool_a.block_size() * pool_a.head_dim();
        let key: Vec<f16> = (0..half_block).map(|i| f16::from_f32(i as f32)).collect();
        let val: Vec<f16> = (0..half_block).map(|i| f16::from_f32(-(i as f32))).collect();

        let src = pool_a.alloc_block().unwrap();
        let dst = pool_b.alloc_block().unwrap();
        pool_a.upload_block(src, &key, &val).unwrap();
        pool_a.copy_block_to_pool(src, &pool_b, dst).unwrap();

        let result = pool_b.download_block(dst).unwrap();
        assert_eq!(&result[..half_block], key.as_slice());
        assert_eq!(&result[half_block..], val.as_slice());
        pool_a.free_block(src);
        pool_b.free_block(dst);
    }

    #[test]
    fn capacity_bytes_matches_geometry() {
        let pool = small_pool(); // 8 blocks, 4 tokens, 2 heads, 8 dim
        // elems_per_block = 2 * 2 * 4 * 8 = 128; bytes = 128 * 2 = 256; total = 8 * 256
        assert_eq!(pool.elems_per_block(), 128);
        assert_eq!(pool.bytes_per_block(), 256);
        assert_eq!(pool.capacity_bytes(), 8 * 256);
    }
}
