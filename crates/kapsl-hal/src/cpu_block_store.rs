//! CPU-side KV block storage — the eviction tier below GPU pools.
//!
//! Blocks evicted from a `GpuBlockPool` under pressure land here. When the
//! owning session is scheduled again, blocks are uploaded back to GPU.
//!
//! The backing allocation is plain heap memory today. In production this should
//! become CUDA pinned memory (`cudaMallocHost`) so async GPU↔CPU DMA can
//! overlap with compute. That change is transparent to callers — only the
//! constructor needs to change.

use half::f16;

#[derive(Debug, thiserror::Error)]
pub enum CpuStoreError {
    #[error("CPU block store is full ({capacity} blocks allocated)")]
    Full { capacity: usize },
    #[error("Slot {0} is out of range")]
    OutOfRange(u32),
    #[error("Data length {got} does not match expected {expected} elements per block")]
    SizeMismatch { got: usize, expected: usize },
}

/// CPU-side flat block storage.
///
/// Each slot holds one KV block: `2 * num_kv_heads * block_size * head_dim` f16
/// elements (K half then V half, matching `GpuBlockPool`'s layout).
pub struct CpuBlockStore {
    data:           Vec<f16>,
    elems_per_block: usize,
    capacity:       usize,
    free_slots:     Vec<u32>,
}

impl CpuBlockStore {
    /// Allocate storage for `capacity` blocks with the given KV geometry.
    pub fn new(
        capacity: usize,
        num_kv_heads: usize,
        block_size: usize,
        head_dim: usize,
    ) -> Self {
        let elems_per_block = 2 * num_kv_heads * block_size * head_dim;
        Self {
            data: vec![f16::ZERO; capacity * elems_per_block],
            elems_per_block,
            capacity,
            free_slots: (0..capacity as u32).rev().collect(),
        }
    }

    /// Number of f16 elements per block.
    pub fn elems_per_block(&self) -> usize { self.elems_per_block }

    /// Total block capacity.
    pub fn capacity(&self) -> usize { self.capacity }

    /// Number of free (unoccupied) slots.
    pub fn free_count(&self) -> usize { self.free_slots.len() }

    /// Number of occupied slots.
    pub fn used_count(&self) -> usize { self.capacity - self.free_slots.len() }

    /// Write `data` into a free slot and return its slot index.
    ///
    /// `data` must have exactly `elems_per_block()` elements.
    pub fn store_block(&mut self, data: &[f16]) -> Result<u32, CpuStoreError> {
        if data.len() != self.elems_per_block {
            return Err(CpuStoreError::SizeMismatch {
                got: data.len(),
                expected: self.elems_per_block,
            });
        }
        let slot = self.free_slots.pop().ok_or(CpuStoreError::Full { capacity: self.capacity })?;
        let off  = slot as usize * self.elems_per_block;
        self.data[off..off + self.elems_per_block].copy_from_slice(data);
        Ok(slot)
    }

    /// Read the block at `slot`.
    pub fn load_block(&self, slot: u32) -> Result<&[f16], CpuStoreError> {
        if slot as usize >= self.capacity {
            return Err(CpuStoreError::OutOfRange(slot));
        }
        let off = slot as usize * self.elems_per_block;
        Ok(&self.data[off..off + self.elems_per_block])
    }

    /// Return `slot` to the free list.
    pub fn free_slot(&mut self, slot: u32) {
        debug_assert!((slot as usize) < self.capacity, "slot out of range");
        self.free_slots.push(slot);
    }

    /// Release all slots belonging to a session (bulk free).
    pub fn free_slots_bulk(&mut self, slots: &[u32]) {
        self.free_slots.extend_from_slice(slots);
    }
}
