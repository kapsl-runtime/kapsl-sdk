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
use cudarc::driver::{CudaDevice, CudaSlice, DeviceSlice};
#[cfg(feature = "cuda")]
use std::sync::Arc;

#[cfg(feature = "cuda")]
use thiserror::Error;

#[cfg(feature = "cuda")]
#[derive(Debug, Error)]
pub enum ArenaError {
    #[error("CUDA driver error: {0}")]
    Cuda(#[from] cudarc::driver::DriverError),
    #[error("Arena out of memory: requested {requested} bytes, {available} available")]
    Oom { requested: usize, available: usize },
    #[error("Block pool exhausted: no free blocks remaining")]
    NoFreeBlocks,
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
    pub fn upload(&self, offset: usize, data: &[u8]) -> Result<(), ArenaError> {
        self.device
            .htod_copy_into(data.to_vec(), &mut self.buffer.slice(offset..offset + data.len()))?;
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
// Paged KV cache on GPU.
//
// Layout: [num_blocks, 2, num_kv_heads, block_size, head_dim] in f16.
//
// Dim 0: physical block index
// Dim 1: 0 = key, 1 = value
// Dim 2: KV head index
// Dim 3: token position within block
// Dim 4: head dimension element

#[cfg(feature = "cuda")]
pub struct GpuBlockPool {
    device: Arc<CudaDevice>,
    storage: CudaSlice<half::f16>,
    num_blocks: usize,
    block_size: usize,
    num_kv_heads: usize,
    head_dim: usize,
    /// Stack of free physical block indices (LIFO for cache locality).
    free_stack: Vec<u32>,
}

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
        let free_stack = (0..num_blocks as u32).rev().collect();

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
            storage,
            num_blocks,
            block_size,
            num_kv_heads,
            head_dim,
            free_stack,
        })
    }

    /// Allocate a free physical block. Returns the block index.
    pub fn alloc_block(&mut self) -> Result<u32, ArenaError> {
        self.free_stack.pop().ok_or(ArenaError::NoFreeBlocks)
    }

    /// Release a physical block back to the free pool.
    pub fn free_block(&mut self, block_id: u32) {
        debug_assert!((block_id as usize) < self.num_blocks);
        self.free_stack.push(block_id);
    }

    /// Number of free blocks remaining.
    pub fn free_count(&self) -> usize {
        self.free_stack.len()
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

    /// Raw device pointer to the start of block storage.
    /// Used by CUDA kernels that need the base address.
    pub fn storage(&self) -> &CudaSlice<half::f16> {
        &self.storage
    }

    pub fn storage_mut(&mut self) -> &mut CudaSlice<half::f16> {
        &mut self.storage
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

        self.device.htod_sync_copy_into(host_key, &mut self.storage.slice(key_offset..key_offset + half_block))?;
        self.device.htod_sync_copy_into(host_val, &mut self.storage.slice(val_offset..val_offset + half_block))?;
        Ok(())
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
