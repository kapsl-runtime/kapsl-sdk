use super::ShmAllocatorSnapshot;
use crate::memory::ShmManager;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

const LEASE_WORD_BYTES: usize = std::mem::size_of::<AtomicU64>();
const SHARED_SLOT_SIZES: [usize; 6] = [
    256 * 1024,
    1024 * 1024,
    4 * 1024 * 1024,
    16 * 1024 * 1024,
    64 * 1024 * 1024,
    128 * 1024 * 1024,
];

#[derive(Debug, Clone, Copy)]
struct SharedSlot {
    control_offset: usize,
    payload_offset: usize,
    payload_capacity: usize,
}

/// A process-shared tensor slot reservation.
///
/// The token contains both a generation and expiry. Releasing an expired lease
/// cannot free a slot that another process has subsequently reclaimed because
/// release compares the complete token atomically.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SharedShmLease {
    slot_index: u32,
    offset: usize,
    capacity: usize,
    token: u64,
}

impl SharedShmLease {
    pub fn offset(self) -> usize {
        self.offset
    }

    pub fn capacity(self) -> usize {
        self.capacity
    }

    pub fn token(self) -> u64 {
        self.token
    }
}

/// Tensor allocator whose ownership words live inside the shared-memory map.
///
/// Every process derives the same tiered slot layout from the immutable region
/// header. The first aligned word of each physical slot is an atomic lease; the
/// remainder is payload capacity. This avoids process-local allocation state
/// and prevents clients and the server from selecting overlapping live slots.
pub struct SharedShmAllocator {
    shm: Arc<ShmManager>,
    slots: Vec<SharedSlot>,
    lease_ttl: Duration,
}

impl SharedShmAllocator {
    /// Initialize lease words while a newly created region is still private.
    pub(crate) fn initialize(shm: &ShmManager) {
        for slot in shared_slot_layout(shm.tensor_pool_offset(), shm.max_tensor_size()) {
            // SAFETY: creation has exclusive access, every control offset is
            // aligned, and the slot layout is contained in the tensor pool.
            unsafe {
                std::ptr::write(
                    shm.as_ptr().add(slot.control_offset).cast::<AtomicU64>(),
                    AtomicU64::new(0),
                );
            }
        }
    }

    /// Connect an allocator to an initialized shared-memory region.
    pub fn connect(shm: Arc<ShmManager>, lease_ttl: Duration) -> Self {
        let slots = shared_slot_layout(shm.tensor_pool_offset(), shm.max_tensor_size());
        Self {
            shm,
            slots,
            lease_ttl: lease_ttl.max(Duration::from_secs(1)),
        }
    }

    /// Atomically reserve the smallest available slot that fits `required_size`.
    pub fn try_allocate(&self, required_size: usize) -> Option<SharedShmLease> {
        if required_size == 0 {
            return None;
        }
        let now = unix_seconds();
        let deadline = now
            .saturating_add(self.lease_ttl.as_secs())
            .min(u32::MAX as u64) as u32;

        for (slot_index, slot) in self.slots.iter().enumerate() {
            if required_size > slot.payload_capacity {
                continue;
            }
            let control = self.control(slot);
            let mut observed = control.load(Ordering::Acquire);
            loop {
                if observed != 0 && lease_deadline(observed) > now {
                    break;
                }
                let sequence = self.shm.next_lease_sequence();
                let token = (u64::from(deadline) << 32) | u64::from(sequence);
                match control.compare_exchange_weak(
                    observed,
                    token,
                    Ordering::AcqRel,
                    Ordering::Acquire,
                ) {
                    Ok(_) => {
                        return Some(SharedShmLease {
                            slot_index: slot_index as u32,
                            offset: slot.payload_offset,
                            capacity: slot.payload_capacity,
                            token,
                        });
                    }
                    Err(current) => observed = current,
                }
            }
        }
        None
    }

    /// Validate a lease received through a request or response envelope.
    pub fn validate(
        &self,
        offset: usize,
        required_size: usize,
        token: u64,
    ) -> Option<SharedShmLease> {
        if token == 0 || required_size == 0 {
            return None;
        }
        let slot_index = self
            .slots
            .binary_search_by_key(&offset, |slot| slot.payload_offset)
            .ok()?;
        let slot = self.slots[slot_index];
        if required_size > slot.payload_capacity
            || self.control(&slot).load(Ordering::Acquire) != token
            || lease_deadline(token) <= unix_seconds()
        {
            return None;
        }
        Some(SharedShmLease {
            slot_index: slot_index as u32,
            offset,
            capacity: slot.payload_capacity,
            token,
        })
    }

    /// Acquire a wire lease for active access and renew its deadline.
    ///
    /// Renewal is one atomic compare/exchange, so a reader either pins the
    /// exact advertised generation or loses to an expiry reclaimer before
    /// touching payload bytes.
    pub fn acquire(
        &self,
        offset: usize,
        required_size: usize,
        token: u64,
    ) -> Option<SharedShmLease> {
        let lease = self.validate(offset, required_size, token)?;
        let deadline = unix_seconds()
            .saturating_add(self.lease_ttl.as_secs())
            .min(u32::MAX as u64);
        let renewed_token = (deadline << 32) | (token & u64::from(u32::MAX));
        let slot = self.slots[lease.slot_index as usize];
        self.control(&slot)
            .compare_exchange(token, renewed_token, Ordering::AcqRel, Ordering::Acquire)
            .ok()?;
        Some(SharedShmLease {
            token: renewed_token,
            ..lease
        })
    }

    /// Release a lease if it still owns the advertised slot generation.
    pub fn release(&self, lease: SharedShmLease) -> bool {
        let Some(slot) = self.slots.get(lease.slot_index as usize) else {
            return false;
        };
        slot.payload_offset == lease.offset
            && self
                .control(slot)
                .compare_exchange(lease.token, 0, Ordering::AcqRel, Ordering::Acquire)
                .is_ok()
    }

    /// Release a lease reconstructed from fixed-layout wire fields.
    pub fn release_wire(&self, offset: usize, token: u64) -> bool {
        let Some(slot_index) = self
            .slots
            .binary_search_by_key(&offset, |slot| slot.payload_offset)
            .ok()
        else {
            return false;
        };
        self.release(SharedShmLease {
            slot_index: slot_index as u32,
            offset,
            capacity: self.slots[slot_index].payload_capacity,
            token,
        })
    }

    pub fn snapshot(&self) -> ShmAllocatorSnapshot {
        let now = unix_seconds();
        let ttl = self.lease_ttl.as_secs();
        let mut in_use_slots = 0;
        let mut oldest_lease_ms = 0;
        for slot in &self.slots {
            let lease = self.control(slot).load(Ordering::Acquire);
            let deadline = lease_deadline(lease);
            if lease == 0 || deadline <= now {
                continue;
            }
            in_use_slots += 1;
            let started = deadline.saturating_sub(ttl);
            oldest_lease_ms = oldest_lease_ms.max(now.saturating_sub(started) * 1_000);
        }
        ShmAllocatorSnapshot {
            in_use_slots,
            oldest_lease_ms,
        }
    }

    pub fn largest_slot_size(&self) -> usize {
        self.slots
            .iter()
            .map(|slot| slot.payload_capacity)
            .max()
            .unwrap_or(0)
    }

    pub fn slot_count(&self) -> usize {
        self.slots.len()
    }

    pub fn layout_summary(&self) -> String {
        format!(
            "shared-atomic(base={} bytes={} slots={} largest={})",
            self.shm.tensor_pool_offset(),
            self.shm.max_tensor_size(),
            self.slot_count(),
            self.largest_slot_size()
        )
    }

    fn control(&self, slot: &SharedSlot) -> &AtomicU64 {
        // SAFETY: the region owns the mapping for this allocator's lifetime and
        // slot construction guarantees an aligned, in-bounds control word.
        unsafe {
            &*self
                .shm
                .as_ptr()
                .add(slot.control_offset)
                .cast::<AtomicU64>()
        }
    }
}

fn shared_slot_layout(base_offset: usize, pool_bytes: usize) -> Vec<SharedSlot> {
    // Give each active size class roughly the same byte budget. Smaller
    // tensors therefore receive many concurrent slots without eliminating the
    // larger classes needed by model inputs and outputs.
    // Preserve one slot for every geometric size class that fits while keeping
    // at least 15% of the pool for additional concurrent small allocations.
    let class_reserve = pool_bytes.saturating_mul(85) / 100;
    let mut reserved = 0_usize;
    let mut slot_sizes = Vec::new();
    for slot_size in SHARED_SLOT_SIZES {
        if reserved.saturating_add(slot_size) > class_reserve {
            break;
        }
        slot_sizes.push(slot_size);
        reserved = reserved.saturating_add(slot_size);
    }
    if slot_sizes.is_empty() {
        slot_sizes.push(pool_bytes.max(LEASE_WORD_BYTES + 1));
    }
    let mut counts = vec![0_usize; slot_sizes.len()];
    let mut remaining = pool_bytes;
    for (index, slot_size) in slot_sizes.iter().copied().enumerate() {
        if remaining >= slot_size {
            counts[index] = 1;
            remaining -= slot_size;
        }
    }
    loop {
        let next = slot_sizes
            .iter()
            .copied()
            .enumerate()
            .filter(|(_, slot_size)| *slot_size <= remaining)
            .min_by_key(|(index, slot_size)| counts[*index].saturating_mul(*slot_size));
        let Some((index, slot_size)) = next else {
            break;
        };
        counts[index] = counts[index].saturating_add(1);
        remaining -= slot_size;
    }

    let mut slots = Vec::new();
    let mut cursor = base_offset;
    for (slot_size, count) in slot_sizes.into_iter().zip(counts) {
        for _ in 0..count {
            let control_offset = cursor;
            let payload_offset = control_offset + LEASE_WORD_BYTES;
            let payload_capacity = slot_size.saturating_sub(LEASE_WORD_BYTES);
            if payload_capacity > 0 {
                slots.push(SharedSlot {
                    control_offset,
                    payload_offset,
                    payload_capacity,
                });
            }
            cursor = cursor.saturating_add(slot_size);
        }
    }
    slots.sort_unstable_by_key(|slot| slot.payload_offset);
    slots
}

fn unix_seconds() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn lease_deadline(token: u64) -> u64 {
    token >> 32
}

#[cfg(test)]
mod tests;
