//! GPU memory arenas and the runtime-owned, geometry-neutral device pool.
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
use cudarc::driver::{CudaDevice, CudaSlice, CudaView, CudaViewMut, DevicePtr, DeviceSlice};
#[cfg(feature = "cuda")]
use std::cell::UnsafeCell;
#[cfg(feature = "cuda")]
use std::collections::{BTreeMap, HashMap, HashSet};
#[cfg(feature = "cuda")]
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
#[cfg(feature = "cuda")]
use std::sync::{Arc, Mutex};

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
    #[error("Block pool has no contiguous run of {requested} blocks ({free} free total)")]
    NoContiguousRun { requested: usize, free: usize },
    #[error("invalid allocation request: bytes and alignment must both be non-zero")]
    InvalidAllocationRequest,
    #[error(
        "device-pool quota exceeded for {owner:?}: requested {requested} bytes, {available} available"
    )]
    QuotaExceeded {
        owner: PoolOwner,
        requested: usize,
        available: usize,
    },
    #[error("allocation is not live or does not belong to {owner:?}")]
    InvalidFree { owner: PoolOwner },
    #[error("cannot release reservation for {owner:?} while it still owns {usage} bytes")]
    OwnerInUse { owner: PoolOwner, usage: usize },
}

// ─── Geometry-neutral device pool ──────────────────────────────────────────

/// Logical consumer of a range in [`GpuDevicePool`].
#[cfg(feature = "cuda")]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PoolOwner {
    Onnx,
    GgufKv { model_id: u32 },
    NativeKv { model_id: u32 },
}

/// A byte extent allocated from [`GpuDevicePool`].
///
/// Fields are deliberately private: callers can inspect an allocation but
/// cannot forge one with a different owner before returning it to the pool.
#[cfg(feature = "cuda")]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GpuAllocation {
    offset: usize,
    bytes: usize,
    owner: PoolOwner,
    pool_id: u64,
}

#[cfg(feature = "cuda")]
impl GpuAllocation {
    pub fn offset(&self) -> usize {
        self.offset
    }

    pub fn bytes(&self) -> usize {
        self.bytes
    }

    pub fn owner(&self) -> PoolOwner {
        self.owner
    }
}

/// Per-owner protected reservation and hard maximum.
#[cfg(feature = "cuda")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OwnerQuota {
    pub guaranteed_bytes: usize,
    pub max_bytes: usize,
}

/// Reservation policy for a device pool.
///
/// Guarantees are protected only for admitted owners. This permits elastic
/// borrowing before a workload is admitted while ensuring that, after
/// admission, other owners cannot consume the bytes needed to reach its
/// guarantee.
#[cfg(feature = "cuda")]
#[derive(Debug)]
pub struct PoolPolicy {
    capacity_bytes: usize,
    quotas: HashMap<PoolOwner, OwnerQuota>,
    usage: HashMap<PoolOwner, usize>,
    admitted: HashSet<PoolOwner>,
}

#[cfg(feature = "cuda")]
impl PoolPolicy {
    pub fn new(capacity_bytes: usize) -> Self {
        Self {
            capacity_bytes,
            quotas: HashMap::new(),
            usage: HashMap::new(),
            admitted: HashSet::new(),
        }
    }

    pub fn set_quota(
        &mut self,
        owner: PoolOwner,
        guaranteed_bytes: usize,
        max_bytes: usize,
    ) -> Result<(), ArenaError> {
        if guaranteed_bytes > max_bytes || max_bytes > self.capacity_bytes {
            return Err(ArenaError::InvalidAllocationRequest);
        }
        let usage = self.usage_bytes(owner);
        if usage > max_bytes {
            return Err(ArenaError::QuotaExceeded {
                owner,
                requested: 0,
                available: max_bytes,
            });
        }
        self.quotas.insert(
            owner,
            OwnerQuota {
                guaranteed_bytes,
                max_bytes,
            },
        );
        Ok(())
    }

    fn set_admitted(&mut self, owner: PoolOwner, admitted: bool) {
        if admitted {
            self.admitted.insert(owner);
        } else if self.usage_bytes(owner) == 0 {
            self.admitted.remove(&owner);
        }
    }

    pub fn quota(&self, owner: PoolOwner) -> OwnerQuota {
        self.quotas.get(&owner).copied().unwrap_or(OwnerQuota {
            guaranteed_bytes: 0,
            max_bytes: self.capacity_bytes,
        })
    }

    pub fn usage_bytes(&self, owner: PoolOwner) -> usize {
        self.usage.get(&owner).copied().unwrap_or(0)
    }

    fn unmet_other_reservations(&self, owner: PoolOwner) -> usize {
        self.admitted
            .iter()
            .filter(|&&other| other != owner)
            .map(|&other| {
                self.quota(other)
                    .guaranteed_bytes
                    .saturating_sub(self.usage_bytes(other))
            })
            .fold(0usize, usize::saturating_add)
    }

    fn unmet_reservations(&self) -> usize {
        self.admitted
            .iter()
            .map(|&owner| {
                self.quota(owner)
                    .guaranteed_bytes
                    .saturating_sub(self.usage_bytes(owner))
            })
            .fold(0usize, usize::saturating_add)
    }

    fn available_for(&self, owner: PoolOwner, free_bytes: usize) -> usize {
        let owner_remaining = self
            .quota(owner)
            .max_bytes
            .saturating_sub(self.usage_bytes(owner));
        let reservation_safe = free_bytes.saturating_sub(self.unmet_other_reservations(owner));
        owner_remaining.min(reservation_safe)
    }

    fn account_alloc(&mut self, owner: PoolOwner, bytes: usize) {
        *self.usage.entry(owner).or_default() += bytes;
    }

    fn account_free(&mut self, owner: PoolOwner, bytes: usize) {
        let usage = self.usage.entry(owner).or_default();
        *usage = usage.saturating_sub(bytes);
    }
}

/// First-fit allocator over aligned byte extents. Free extents are coalesced,
/// so metadata grows with fragmentation rather than pool capacity.
#[cfg(feature = "cuda")]
#[derive(Debug)]
struct AlignedRangeAllocator {
    capacity: usize,
    free: BTreeMap<usize, usize>,
    live: BTreeMap<usize, GpuAllocation>,
}

#[cfg(feature = "cuda")]
impl AlignedRangeAllocator {
    fn new(capacity: usize) -> Self {
        let mut free = BTreeMap::new();
        if capacity > 0 {
            free.insert(0, capacity);
        }
        Self {
            capacity,
            free,
            live: BTreeMap::new(),
        }
    }

    fn align_up(value: usize, alignment: usize) -> Option<usize> {
        if alignment == 0 {
            return None;
        }
        let remainder = value % alignment;
        value.checked_add(if remainder == 0 {
            0
        } else {
            alignment - remainder
        })
    }

    fn alloc(
        &mut self,
        pool_id: u64,
        owner: PoolOwner,
        bytes: usize,
        alignment: usize,
    ) -> Option<GpuAllocation> {
        if bytes == 0 || alignment == 0 {
            return None;
        }
        let selected = self.free.iter().find_map(|(&start, &len)| {
            let aligned = Self::align_up(start, alignment)?;
            let end = aligned.checked_add(bytes)?;
            (end <= start.checked_add(len)?).then_some((start, len, aligned, end))
        })?;
        let (start, len, aligned, end) = selected;
        self.free.remove(&start);
        if aligned > start {
            self.free.insert(start, aligned - start);
        }
        let extent_end = start + len;
        if end < extent_end {
            self.free.insert(end, extent_end - end);
        }
        let allocation = GpuAllocation {
            offset: aligned,
            bytes,
            owner,
            pool_id,
        };
        self.live.insert(aligned, allocation.clone());
        Some(allocation)
    }

    fn free(&mut self, allocation: &GpuAllocation) -> Result<(), ArenaError> {
        match self.live.get(&allocation.offset) {
            Some(live) if live == allocation => {}
            _ => {
                return Err(ArenaError::InvalidFree {
                    owner: allocation.owner,
                })
            }
        }
        self.live.remove(&allocation.offset);

        let mut start = allocation.offset;
        let mut len = allocation.bytes;
        if let Some((&prev_start, &prev_len)) = self.free.range(..start).next_back() {
            if prev_start + prev_len == start {
                self.free.remove(&prev_start);
                start = prev_start;
                len += prev_len;
            }
        }
        if let Some((&next_start, &next_len)) = self.free.range(start..).next() {
            if start + len == next_start {
                self.free.remove(&next_start);
                len += next_len;
            }
        }
        self.free.insert(start, len);
        debug_assert!(start + len <= self.capacity);
        Ok(())
    }

    fn free_bytes(&self) -> usize {
        self.free.values().copied().sum()
    }

    fn max_allocatable_units(&self, unit_bytes: usize, alignment: usize) -> usize {
        if unit_bytes == 0 || alignment == 0 {
            return 0;
        }
        let Some(stride) = Self::align_up(unit_bytes, alignment) else {
            return 0;
        };
        self.free
            .iter()
            .map(|(&start, &len)| {
                let Some(aligned) = Self::align_up(start, alignment) else {
                    return 0;
                };
                let prefix = aligned.saturating_sub(start);
                let usable = len.saturating_sub(prefix);
                if usable < unit_bytes {
                    0
                } else {
                    1 + (usable - unit_bytes) / stride
                }
            })
            .fold(0usize, usize::saturating_add)
    }
}

/// One stable backing allocation and byte-range allocator for a CUDA device.
#[cfg(feature = "cuda")]
static NEXT_DEVICE_POOL_ID: AtomicU64 = AtomicU64::new(1);

#[cfg(feature = "cuda")]
pub struct GpuDevicePool {
    pool_id: u64,
    device: Arc<CudaDevice>,
    storage: UnsafeCell<CudaSlice<u8>>,
    allocator: Mutex<AlignedRangeAllocator>,
    policy: Mutex<PoolPolicy>,
}

#[cfg(feature = "cuda")]
impl std::fmt::Debug for GpuDevicePool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuDevicePool")
            .field("capacity_bytes", &self.capacity_bytes())
            .field("free_bytes", &self.free_bytes())
            .finish()
    }
}

// SAFETY: the allocator assigns non-overlapping extents. Consumers may mutate
// only extents they own, and the backing CudaSlice remains pinned for the pool
// lifetime.
#[cfg(feature = "cuda")]
unsafe impl Send for GpuDevicePool {}
#[cfg(feature = "cuda")]
unsafe impl Sync for GpuDevicePool {}

#[cfg(feature = "cuda")]
impl GpuDevicePool {
    pub fn new(device: Arc<CudaDevice>, capacity_bytes: usize) -> Result<Self, ArenaError> {
        if capacity_bytes == 0 {
            return Err(ArenaError::InvalidAllocationRequest);
        }
        let storage = device.alloc_zeros::<u8>(capacity_bytes)?;
        log::info!(
            "GPU device pool allocated: {} MiB",
            capacity_bytes / (1024 * 1024)
        );
        Ok(Self {
            pool_id: NEXT_DEVICE_POOL_ID.fetch_add(1, Ordering::Relaxed),
            device,
            storage: UnsafeCell::new(storage),
            allocator: Mutex::new(AlignedRangeAllocator::new(capacity_bytes)),
            policy: Mutex::new(PoolPolicy::new(capacity_bytes)),
        })
    }

    pub fn alloc(
        &self,
        owner: PoolOwner,
        bytes: usize,
        alignment: usize,
    ) -> Result<GpuAllocation, ArenaError> {
        if bytes == 0 || alignment == 0 {
            return Err(ArenaError::InvalidAllocationRequest);
        }
        // Policy is always locked before allocator throughout this type.
        let mut policy = self.policy.lock().unwrap();
        let mut allocator = self.allocator.lock().unwrap();
        let available = policy.available_for(owner, allocator.free_bytes());
        if bytes > available {
            return Err(ArenaError::QuotaExceeded {
                owner,
                requested: bytes,
                available,
            });
        }
        let allocation = allocator
            .alloc(self.pool_id, owner, bytes, alignment)
            .ok_or(ArenaError::Oom {
                requested: bytes,
                available: allocator.free_bytes(),
            })?;
        policy.account_alloc(owner, bytes);
        Ok(allocation)
    }

    pub fn free(&self, allocation: GpuAllocation) -> Result<(), ArenaError> {
        let mut policy = self.policy.lock().unwrap();
        let mut allocator = self.allocator.lock().unwrap();
        allocator.free(&allocation)?;
        policy.account_free(allocation.owner, allocation.bytes);
        Ok(())
    }

    pub fn set_owner_quota(
        &self,
        owner: PoolOwner,
        guaranteed_bytes: usize,
        max_bytes: usize,
    ) -> Result<(), ArenaError> {
        let mut policy = self.policy.lock().unwrap();
        let allocator = self.allocator.lock().unwrap();
        let previous = policy.quotas.get(&owner).copied();
        policy.set_quota(owner, guaranteed_bytes, max_bytes)?;
        if policy.unmet_reservations() > allocator.free_bytes() {
            if let Some(previous) = previous {
                policy.quotas.insert(owner, previous);
            } else {
                policy.quotas.remove(&owner);
            }
            return Err(ArenaError::QuotaExceeded {
                owner,
                requested: guaranteed_bytes,
                available: allocator.free_bytes(),
            });
        }
        Ok(())
    }

    pub fn set_owner_admitted(&self, owner: PoolOwner, admitted: bool) -> Result<(), ArenaError> {
        let mut policy = self.policy.lock().unwrap();
        let allocator = self.allocator.lock().unwrap();
        if !admitted && policy.usage_bytes(owner) != 0 {
            return Err(ArenaError::OwnerInUse {
                owner,
                usage: policy.usage_bytes(owner),
            });
        }
        let was_admitted = policy.admitted.contains(&owner);
        policy.set_admitted(owner, admitted);
        if admitted && policy.unmet_reservations() > allocator.free_bytes() {
            if was_admitted {
                policy.admitted.insert(owner);
            } else {
                policy.admitted.remove(&owner);
            }
            return Err(ArenaError::QuotaExceeded {
                owner,
                requested: policy.quota(owner).guaranteed_bytes,
                available: allocator.free_bytes(),
            });
        }
        Ok(())
    }

    pub fn owner_usage_bytes(&self, owner: PoolOwner) -> usize {
        self.policy.lock().unwrap().usage_bytes(owner)
    }

    pub fn owner_quota(&self, owner: PoolOwner) -> OwnerQuota {
        self.policy.lock().unwrap().quota(owner)
    }

    pub fn free_bytes(&self) -> usize {
        self.allocator.lock().unwrap().free_bytes()
    }

    /// Maximum number of `unit_bytes` allocations currently possible for an
    /// owner, including quota/reservation checks and alignment fragmentation.
    pub fn max_allocatable(&self, owner: PoolOwner, unit_bytes: usize, alignment: usize) -> usize {
        if unit_bytes == 0 || alignment == 0 {
            return 0;
        }
        let policy = self.policy.lock().unwrap();
        let allocator = self.allocator.lock().unwrap();
        let quota_count = policy.available_for(owner, allocator.free_bytes()) / unit_bytes;
        quota_count.min(allocator.max_allocatable_units(unit_bytes, alignment))
    }

    pub fn base_ptr(&self) -> *mut std::ffi::c_void {
        let storage = unsafe { &*self.storage.get() };
        *storage.device_ptr() as *mut std::ffi::c_void
    }

    pub fn allocation_ptr(&self, allocation: &GpuAllocation) -> *mut std::ffi::c_void {
        debug_assert_eq!(allocation.pool_id, self.pool_id);
        (self.base_ptr() as usize + allocation.offset) as *mut std::ffi::c_void
    }

    pub fn capacity_bytes(&self) -> usize {
        let storage = unsafe { &*self.storage.get() };
        storage.len()
    }

    pub fn device(&self) -> &Arc<CudaDevice> {
        &self.device
    }

    fn f16_storage(&self) -> CudaView<'_, half::f16> {
        let storage = unsafe { &*self.storage.get() };
        // CUDA allocations are sufficiently aligned for f16; a trailing odd
        // byte, if any, is intentionally not exposed through the typed view.
        unsafe { storage.transmute(storage.len() / std::mem::size_of::<half::f16>()) }
            .expect("f16 view fits device pool")
    }

    /// # Safety
    /// The caller must write only extents allocated to it. Multiple mutable
    /// views may coexist because ownership is enforced by the range allocator.
    unsafe fn f16_storage_mut(&self) -> CudaViewMut<'_, half::f16> {
        let storage = unsafe { &mut *self.storage.get() };
        let len = storage.len() / std::mem::size_of::<half::f16>();
        unsafe { storage.transmute_mut(len) }.expect("f16 view fits device pool")
    }
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

// ─── GPU KV pool view ────────────────────────────────────────────────────────
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
// The view owns no CUDA allocation. Each block is a byte extent obtained from
// the runtime-owned `GpuDevicePool`; different model geometries therefore
// coexist safely in one backing allocation.

#[cfg(feature = "cuda")]
pub struct GpuKvPoolView {
    device_pool: Arc<GpuDevicePool>,
    owner: PoolOwner,
    block_size: usize,
    num_kv_heads: usize,
    head_dim: usize,
    cap_blocks: AtomicUsize,
    /// Physical block id → allocation. Retaining the allocation is required so
    /// the central allocator can validate owner, offset, and length on free.
    live_blocks: Mutex<HashMap<u32, GpuAllocation>>,
    /// Compatibility path for callers that request a contiguous run.
    live_runs: Mutex<HashMap<u32, (usize, GpuAllocation)>>,
}

#[cfg(feature = "cuda")]
impl std::fmt::Debug for GpuKvPoolView {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuKvPoolView")
            .field("owner", &self.owner)
            .field("cap_blocks", &self.total_blocks())
            .field("block_size", &self.block_size)
            .field("num_kv_heads", &self.num_kv_heads)
            .field("head_dim", &self.head_dim)
            .field("free", &self.free_count())
            .finish()
    }
}

#[cfg(feature = "cuda")]
impl GpuKvPoolView {
    /// Compatibility constructor for a private KV pool. Runtime integrations
    /// should create one [`GpuDevicePool`] per device and call
    /// [`from_device_pool`](Self::from_device_pool) instead.
    pub fn new(
        device: Arc<CudaDevice>,
        num_blocks: usize,
        block_size: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) -> Result<Self, ArenaError> {
        let bytes_per_block = 2usize
            .checked_mul(num_kv_heads)
            .and_then(|n| n.checked_mul(block_size))
            .and_then(|n| n.checked_mul(head_dim))
            .and_then(|n| n.checked_mul(std::mem::size_of::<half::f16>()))
            .ok_or(ArenaError::InvalidAllocationRequest)?;
        let bytes = num_blocks
            .checked_mul(bytes_per_block)
            .ok_or(ArenaError::InvalidAllocationRequest)?;
        let device_pool = Arc::new(GpuDevicePool::new(device, bytes)?);
        Self::from_device_pool(
            device_pool,
            PoolOwner::NativeKv { model_id: 0 },
            num_blocks,
            block_size,
            num_kv_heads,
            head_dim,
        )
    }

    /// Create a model-specific KV geometry view over a runtime-owned pool.
    pub fn from_device_pool(
        device_pool: Arc<GpuDevicePool>,
        owner: PoolOwner,
        cap_blocks: usize,
        block_size: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) -> Result<Self, ArenaError> {
        if cap_blocks == 0
            || block_size == 0
            || num_kv_heads == 0
            || head_dim == 0
            || !matches!(owner, PoolOwner::GgufKv { .. } | PoolOwner::NativeKv { .. })
        {
            return Err(ArenaError::InvalidAllocationRequest);
        }
        let bytes_per_block = 2usize
            .checked_mul(num_kv_heads)
            .and_then(|n| n.checked_mul(block_size))
            .and_then(|n| n.checked_mul(head_dim))
            .and_then(|n| n.checked_mul(std::mem::size_of::<half::f16>()))
            .ok_or(ArenaError::InvalidAllocationRequest)?;
        let quota_cap = device_pool.owner_quota(owner).max_bytes / bytes_per_block;
        let backing_cap = device_pool.capacity_bytes() / bytes_per_block;
        let cap_blocks = cap_blocks.min(quota_cap).min(backing_cap);
        if cap_blocks == 0 {
            return Err(ArenaError::Oom {
                requested: bytes_per_block,
                available: device_pool.free_bytes(),
            });
        }
        log::info!(
            "GPU KV view: owner={:?}, cap={} blocks × {} tokens ({}h × {}d), device pool={} MiB",
            owner,
            cap_blocks,
            block_size,
            num_kv_heads,
            head_dim,
            device_pool.capacity_bytes() / (1024 * 1024),
        );

        Ok(Self {
            device_pool,
            owner,
            block_size,
            num_kv_heads,
            head_dim,
            cap_blocks: AtomicUsize::new(cap_blocks),
            live_blocks: Mutex::new(HashMap::new()),
            live_runs: Mutex::new(HashMap::new()),
        })
    }

    /// Allocate a free physical block. Returns the block index.
    pub fn alloc_block(&self) -> Result<u32, ArenaError> {
        let mut live = self.live_blocks.lock().unwrap();
        let run_count: usize = self
            .live_runs
            .lock()
            .unwrap()
            .values()
            .map(|(count, _)| *count)
            .sum();
        if live.len().saturating_add(run_count) >= self.total_blocks() {
            return Err(ArenaError::NoFreeBlocks);
        }
        let bytes = self.bytes_per_block();
        let allocation = self.device_pool.alloc(self.owner, bytes, bytes)?;
        let physical = allocation.offset() / bytes;
        let physical = u32::try_from(physical).map_err(|_| ArenaError::InvalidAllocationRequest)?;
        if live.contains_key(&physical) {
            let _ = self.device_pool.free(allocation);
            return Err(ArenaError::InvalidAllocationRequest);
        }
        live.insert(physical, allocation);
        Ok(physical)
    }

    /// Release a physical block back to the free pool.
    pub fn free_block(&self, block_id: u32) {
        if let Err(error) = self.try_free_block(block_id) {
            debug_assert!(false, "failed to free KV block {block_id}: {error}");
            log::error!("failed to free KV block {}: {}", block_id, error);
        }
    }

    pub fn try_free_block(&self, block_id: u32) -> Result<(), ArenaError> {
        let allocation = {
            self.live_blocks
                .lock()
                .unwrap()
                .remove(&block_id)
                .ok_or(ArenaError::InvalidFree { owner: self.owner })?
        };
        if let Err(error) = self.device_pool.free(allocation.clone()) {
            self.live_blocks
                .lock()
                .unwrap()
                .insert(block_id, allocation);
            return Err(error);
        }
        Ok(())
    }

    /// Allocate `n` *contiguous* physical blocks, returning the first index.
    ///
    /// Used by consumers that need flat device buffers carved out of the pool
    /// rather than paged single blocks. ORT now allocates directly from the
    /// geometry-neutral device pool and does not use this compatibility API.
    pub fn alloc_blocks_contiguous(&self, n: usize) -> Result<u32, ArenaError> {
        if n == 0 || self.used_count().saturating_add(n) > self.total_blocks() {
            return Err(ArenaError::NoContiguousRun {
                requested: n,
                free: self.free_count(),
            });
        }
        let block_bytes = self.bytes_per_block();
        let bytes = n
            .checked_mul(block_bytes)
            .ok_or(ArenaError::InvalidAllocationRequest)?;
        let allocation = self
            .device_pool
            .alloc(self.owner, bytes, block_bytes)
            .map_err(|_| ArenaError::NoContiguousRun {
                requested: n,
                free: self.free_count(),
            })?;
        let first = u32::try_from(allocation.offset() / block_bytes)
            .map_err(|_| ArenaError::InvalidAllocationRequest)?;
        self.live_runs
            .lock()
            .unwrap()
            .insert(first, (n, allocation));
        Ok(first)
    }

    /// Release a contiguous run previously returned by
    /// [`alloc_blocks_contiguous`](Self::alloc_blocks_contiguous).
    pub fn free_blocks_contiguous(&self, first: u32, n: usize) {
        let run = self.live_runs.lock().unwrap().remove(&first);
        match run {
            Some((actual, allocation)) if actual == n => {
                if let Err(error) = self.device_pool.free(allocation.clone()) {
                    self.live_runs
                        .lock()
                        .unwrap()
                        .insert(first, (actual, allocation));
                    log::error!("failed to free contiguous KV run: {error}");
                }
            }
            Some((actual, allocation)) => {
                self.live_runs
                    .lock()
                    .unwrap()
                    .insert(first, (actual, allocation));
                debug_assert_eq!(actual, n, "contiguous KV run length mismatch");
            }
            None => debug_assert!(false, "free of unknown contiguous KV run"),
        }
    }

    /// Number of free blocks remaining.
    pub fn free_count(&self) -> usize {
        self.max_allocatable_blocks()
    }

    /// Blocks this view can allocate now after applying its logical cap, the
    /// central quota policy, reservations, and current range fragmentation.
    pub fn max_allocatable_blocks(&self) -> usize {
        let cap_remaining = self.total_blocks().saturating_sub(self.used_count());
        cap_remaining.min(self.device_pool.max_allocatable(
            self.owner,
            self.bytes_per_block(),
            self.bytes_per_block(),
        ))
    }

    /// Total number of blocks in the pool.
    pub fn total_blocks(&self) -> usize {
        self.cap_blocks.load(Ordering::Relaxed)
    }

    pub fn set_cap_blocks(&self, cap: usize) {
        self.cap_blocks.store(cap, Ordering::Relaxed);
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

    /// Logical byte cap visible through this geometry view.
    pub fn capacity_bytes(&self) -> usize {
        self.total_blocks().saturating_mul(self.bytes_per_block())
    }

    /// Number of physical blocks currently allocated from the pool.
    pub fn used_count(&self) -> usize {
        let singles = self.live_blocks.lock().unwrap().len();
        let runs = self
            .live_runs
            .lock()
            .unwrap()
            .values()
            .map(|(count, _)| *count)
            .sum::<usize>();
        singles.saturating_add(runs)
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
        let base = block_id as usize * elems;
        let storage = self.storage();
        Ok(self
            .device()
            .dtoh_sync_copy(&storage.slice(base..base + elems))?)
    }

    /// Copy one block from this pool to a block in another pool (same or different device).
    /// Routes through host memory — works over PCIe without peer access.
    pub fn copy_block_to_pool(
        &self,
        src_block: u32,
        dst_pool: &GpuKvPoolView,
        dst_block: u32,
    ) -> Result<(), ArenaError> {
        let data = self.download_block(src_block)?;
        let half_len = data.len() / 2;
        dst_pool.upload_block(dst_block, &data[..half_len], &data[half_len..])
    }

    /// Read-only view of the storage slice (for attention kernel reads).
    pub fn storage(&self) -> CudaView<'_, half::f16> {
        self.device_pool.f16_storage()
    }

    /// Raw CUDA device pointer for FFI integrations that need to wrap the
    /// externally-owned KV storage without taking ownership.
    pub fn device_base_ptr(&self) -> *mut std::ffi::c_void {
        self.device_pool.base_ptr()
    }

    /// Raw CUDA device pointer to the start of a specific physical block.
    pub fn block_device_ptr(&self, block_id: u32) -> *mut std::ffi::c_void {
        (self.device_base_ptr() as usize + block_id as usize * self.bytes_per_block())
            as *mut std::ffi::c_void
    }

    /// Mutable view of the storage slice (for KV-write kernels).
    ///
    /// # Safety
    /// Caller must ensure that the physical blocks it writes to are not
    /// concurrently written by another caller.  This invariant is upheld by
    /// the alloc_block/free_block protocol — only the block owner writes.
    pub fn storage_mut(&self) -> CudaViewMut<'_, half::f16> {
        unsafe { self.device_pool.f16_storage_mut() }
    }

    pub fn device(&self) -> &Arc<CudaDevice> {
        self.device_pool.device()
    }

    pub fn device_pool(&self) -> &Arc<GpuDevicePool> {
        &self.device_pool
    }

    pub fn owner(&self) -> PoolOwner {
        self.owner
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

        {
            let mut storage = self.storage_mut();
            self.device().htod_sync_copy_into(
                host_key,
                &mut storage.slice_mut(key_offset..key_offset + half_block),
            )?;
        }
        {
            let mut storage = self.storage_mut();
            self.device().htod_sync_copy_into(
                host_val,
                &mut storage.slice_mut(val_offset..val_offset + half_block),
            )?;
        }
        Ok(())
    }
}

#[cfg(feature = "cuda")]
impl Drop for GpuKvPoolView {
    fn drop(&mut self) {
        let singles = self.live_blocks.get_mut().unwrap().drain().map(|(_, a)| a);
        let runs = self
            .live_runs
            .get_mut()
            .unwrap()
            .drain()
            .map(|(_, (_, a))| a);
        for allocation in singles.chain(runs) {
            if let Err(error) = self.device_pool.free(allocation) {
                log::error!("failed to release KV allocation while dropping view: {error}");
            }
        }
    }
}

/// Backward-compatible name for the KV geometry view.
#[cfg(feature = "cuda")]
pub type GpuBlockPool = GpuKvPoolView;

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

    #[test]
    fn aligned_ranges_support_arbitrary_alignment_and_coalesce() {
        let owner = PoolOwner::Onnx;
        let mut allocator = AlignedRangeAllocator::new(1024);
        let first = allocator.alloc(1, owner, 100, 256).unwrap();
        let second = allocator.alloc(1, owner, 100, 300).unwrap();
        assert_eq!(first.offset(), 0);
        assert_eq!(second.offset(), 300);
        assert_eq!(allocator.free_bytes(), 824);
        assert_eq!(allocator.max_allocatable_units(100, 256), 2);

        allocator.free(&first).unwrap();
        allocator.free(&second).unwrap();
        assert_eq!(allocator.free_bytes(), 1024);
        assert_eq!(allocator.max_allocatable_units(100, 256), 4);
        assert_eq!(allocator.free, BTreeMap::from([(0, 1024)]));
    }

    #[test]
    fn range_free_verifies_owner_and_live_extent() {
        let owner = PoolOwner::GgufKv { model_id: 7 };
        let mut allocator = AlignedRangeAllocator::new(1024);
        let allocation = allocator.alloc(1, owner, 128, 64).unwrap();
        let wrong_owner = GpuAllocation {
            owner: PoolOwner::Onnx,
            ..allocation.clone()
        };
        assert!(matches!(
            allocator.free(&wrong_owner),
            Err(ArenaError::InvalidFree { .. })
        ));
        allocator.free(&allocation).unwrap();
        assert!(matches!(
            allocator.free(&allocation),
            Err(ArenaError::InvalidFree { .. })
        ));
    }

    #[test]
    fn quota_policy_protects_other_admitted_owner() {
        let gguf = PoolOwner::GgufKv { model_id: 1 };
        let onnx = PoolOwner::Onnx;
        let mut policy = PoolPolicy::new(120);
        policy.set_quota(gguf, 60, 90).unwrap();
        policy.set_quota(onnx, 30, 60).unwrap();
        policy.set_admitted(gguf, true);
        policy.set_admitted(onnx, true);

        assert_eq!(policy.available_for(gguf, 120), 90);
        policy.account_alloc(gguf, 80);
        // Ten elastic bytes remain available; ONNX's unmet 30-byte guarantee
        // remains protected.
        assert_eq!(policy.available_for(gguf, 40), 10);
        assert_eq!(policy.available_for(onnx, 40), 40);
    }

    #[test]
    fn quota_policy_allows_borrowing_from_unadmitted_owner() {
        let gguf = PoolOwner::GgufKv { model_id: 1 };
        let onnx = PoolOwner::Onnx;
        let mut policy = PoolPolicy::new(120);
        policy.set_quota(gguf, 60, 120).unwrap();
        policy.set_quota(onnx, 30, 60).unwrap();
        policy.set_admitted(gguf, true);

        assert_eq!(policy.available_for(gguf, 120), 120);
        policy.set_admitted(onnx, true);
        assert_eq!(policy.available_for(gguf, 120), 90);
    }

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
        for b in blocks {
            pool.free_block(b);
        }
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
        assert!(matches!(
            err,
            ArenaError::NoContiguousRun { requested: 2, .. }
        ));
        for b in all.iter().filter(|b| *b % 2 == 1) {
            pool.free_block(*b);
        }
    }

    #[test]
    fn upload_download_roundtrip() {
        let pool = small_pool();
        let half_block = pool.num_kv_heads() * pool.block_size() * pool.head_dim(); // 2*4*8 = 64
        let key: Vec<f16> = (0..half_block).map(|i| f16::from_f32(i as f32)).collect();
        let val: Vec<f16> = (0..half_block)
            .map(|i| f16::from_f32(i as f32 + 100.0))
            .collect();

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
        let ones = vec![f16::ONE; half_block];

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
        let val: Vec<f16> = (0..half_block)
            .map(|i| f16::from_f32(-(i as f32)))
            .collect();

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
