use parking_lot::Mutex;
use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

/// Identifies which engine and sequence a physical block belongs to.
///
/// Stamped onto a block when it is allocated for a sequence and recorded in the
/// pool's ownership registry, so any block in a shared pool can be traced back
/// to its owner — the basis for safe shared-pool reuse and foreign-free
/// detection. `model_id` is represented by `engine_id`, which the runtime maps
/// 1:1 to a loaded model/replica.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BlockOwner {
    pub engine_id: u32,
    pub sequence_id: u64,
}

/// Represents a physical block of memory in the KV cache.
///
/// Identity is `(block_number, device_id)` — `owner` is metadata that travels
/// with the handle and is intentionally excluded from equality/hashing so a
/// block compares equal regardless of which sequence currently holds it.
#[derive(Debug, Clone, Copy)]
pub struct PhysicalTokenBlock {
    pub block_number: usize,
    pub block_size: usize,
    pub device_id: usize,
    /// Owner stamped at allocation time (`None` for ownerless/private allocs).
    pub owner: Option<BlockOwner>,
}

impl PartialEq for PhysicalTokenBlock {
    fn eq(&self, other: &Self) -> bool {
        self.block_number == other.block_number && self.device_id == other.device_id
    }
}
impl Eq for PhysicalTokenBlock {}
impl std::hash::Hash for PhysicalTokenBlock {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.block_number.hash(state);
        self.device_id.hash(state);
    }
}

/// Manages the allocation of physical blocks
#[derive(Debug)]
pub struct BlockAllocator {
    free_blocks: VecDeque<usize>,
    total_blocks: usize,
    block_size: usize,
    device_id: usize,
    /// Ownership registry: physical block_number → current owner. The single
    /// source of truth for who holds each slot in a (possibly shared) pool.
    owners: HashMap<usize, BlockOwner>,
}

impl BlockAllocator {
    pub fn new(total_blocks: usize, block_size: usize, device_id: usize) -> Self {
        let free_blocks: VecDeque<usize> = (0..total_blocks).collect();
        Self {
            free_blocks,
            total_blocks,
            block_size,
            device_id,
            owners: HashMap::new(),
        }
    }

    /// Allocate an ownerless block (private/owned pools, no registry tracking).
    pub fn allocate(&mut self) -> Option<PhysicalTokenBlock> {
        self.free_blocks
            .pop_front()
            .map(|block_number| PhysicalTokenBlock {
                block_number,
                block_size: self.block_size,
                device_id: self.device_id,
                owner: None,
            })
    }

    /// Allocate a block for `owner`, stamping the handle and recording ownership
    /// in the registry so the slot can be traced back to its owner.
    pub fn allocate_for(&mut self, owner: BlockOwner) -> Option<PhysicalTokenBlock> {
        let block_number = self.free_blocks.pop_front()?;
        self.owners.insert(block_number, owner);
        Some(PhysicalTokenBlock {
            block_number,
            block_size: self.block_size,
            device_id: self.device_id,
            owner: Some(owner),
        })
    }

    pub fn free(&mut self, block: PhysicalTokenBlock) {
        if block.device_id != self.device_id {
            return;
        }
        // Reconcile against the registry to catch cross-model corruption: a
        // block being returned whose recorded owner differs from the handle's,
        // or a block with no registry entry (double/untracked free).
        let recorded = self.owners.remove(&block.block_number);
        match (block.owner, recorded) {
            (Some(handle), Some(reg)) if handle != reg => {
                log::warn!(
                    "[kv-pool] foreign free of block {}: handle owner {:?} != registry owner {:?}",
                    block.block_number,
                    handle,
                    reg
                );
            }
            (Some(handle), None) => {
                log::warn!(
                    "[kv-pool] untracked free of block {} (handle owner {:?}); possible double free",
                    block.block_number,
                    handle
                );
            }
            _ => {}
        }
        self.free_blocks.push_front(block.block_number);
    }

    pub fn get_num_free_blocks(&self) -> usize {
        self.free_blocks.len()
    }

    pub fn get_num_total_blocks(&self) -> usize {
        self.total_blocks
    }

    /// Current owner of a physical block, or `None` if free/untracked.
    pub fn owner_of(&self, block_number: usize) -> Option<BlockOwner> {
        self.owners.get(&block_number).copied()
    }

    /// Number of blocks in this pool currently owned by `engine_id` (diagnostics).
    pub fn count_for_engine(&self, engine_id: u32) -> usize {
        self.owners
            .values()
            .filter(|o| o.engine_id == engine_id)
            .count()
    }
}

/// A reference-counted, mutex-wrapped [`BlockAllocator`] that can be shared
/// across multiple [`BlockManager`] instances.
///
/// All loaded models that draw from the same `SharedBlockAllocator` form a
/// unified KV block pool: blocks freed by one engine are immediately available
/// to any other engine sharing the same handle. Create one with
/// [`new_shared_allocator`] and hand clones of the `Arc` to each engine's
/// [`BlockManager::new_shared`].
pub type SharedBlockAllocator = Arc<Mutex<BlockAllocator>>;

/// Construct a new shared allocator owning `total_blocks` blocks.
pub fn new_shared_allocator(
    total_blocks: usize,
    block_size: usize,
    device_id: usize,
) -> SharedBlockAllocator {
    Arc::new(Mutex::new(BlockAllocator::new(
        total_blocks,
        block_size,
        device_id,
    )))
}

/// Internal dispatch: each `BlockManager` holds either a private allocator or
/// a reference to a pool shared with other managers.
enum BlockManagerAllocator {
    Owned(BlockAllocator),
    Shared(SharedBlockAllocator),
}

impl BlockManagerAllocator {
    fn allocate(&mut self) -> Option<PhysicalTokenBlock> {
        match self {
            Self::Owned(a) => a.allocate(),
            Self::Shared(a) => a.lock().allocate(),
        }
    }

    fn allocate_for(&mut self, owner: BlockOwner) -> Option<PhysicalTokenBlock> {
        match self {
            Self::Owned(a) => a.allocate_for(owner),
            Self::Shared(a) => a.lock().allocate_for(owner),
        }
    }

    fn owner_of(&self, block_number: usize) -> Option<BlockOwner> {
        match self {
            Self::Owned(a) => a.owner_of(block_number),
            Self::Shared(a) => a.lock().owner_of(block_number),
        }
    }

    fn free(&mut self, block: PhysicalTokenBlock) {
        match self {
            Self::Owned(a) => a.free(block),
            Self::Shared(a) => a.lock().free(block),
        }
    }

    fn get_num_free_blocks(&self) -> usize {
        match self {
            Self::Owned(a) => a.get_num_free_blocks(),
            Self::Shared(a) => a.lock().get_num_free_blocks(),
        }
    }

    fn get_num_total_blocks(&self) -> usize {
        match self {
            Self::Owned(a) => a.get_num_total_blocks(),
            Self::Shared(a) => a.lock().get_num_total_blocks(),
        }
    }
}

/// Maps logical blocks to physical blocks for a sequence
#[derive(Debug, Clone)]
pub struct BlockTable {
    logical_to_physical: Vec<PhysicalTokenBlock>,
    #[allow(dead_code)]
    block_size: usize,
}

impl BlockTable {
    pub fn new(block_size: usize) -> Self {
        Self {
            logical_to_physical: Vec::new(),
            block_size,
        }
    }

    pub fn append(&mut self, block: PhysicalTokenBlock) {
        self.logical_to_physical.push(block);
    }

    pub fn get_physical_blocks(&self) -> &[PhysicalTokenBlock] {
        &self.logical_to_physical
    }

    pub fn len(&self) -> usize {
        self.logical_to_physical.len()
    }

    pub fn is_empty(&self) -> bool {
        self.logical_to_physical.is_empty()
    }
}

/// Manages blocks for all sequences.
///
/// Can operate in two modes:
/// - **Owned**: private `BlockAllocator` (original behaviour, created via [`BlockManager::new`]).
/// - **Shared**: draws from a [`SharedBlockAllocator`] that is also given to other managers,
///   enabling a unified KV block pool across multiple loaded models (created via
///   [`BlockManager::new_shared`]).
pub struct BlockManager {
    allocator: BlockManagerAllocator,
    block_tables: HashMap<u64, BlockTable>, // sequence_id -> BlockTable
    block_size: usize,
    /// Number of blocks this manager currently holds across all sequences.
    /// Maintained incrementally so the per-engine quota can be checked in O(1).
    held_blocks: usize,
    /// Optional hard per-engine block quota. When set (shared-pool mode), this
    /// manager will not allocate beyond `live_cap` blocks even if the shared
    /// pool has more free — that headroom belongs to other models' fair shares.
    /// The runtime updates the atomic on engine join/leave so the cap rebalances
    /// without a restart.
    live_cap: Option<Arc<AtomicUsize>>,
    /// This manager's engine id. When set, allocations stamp ownership so blocks
    /// in a shared pool are traceable to their owning engine/sequence.
    engine_id: Option<u32>,
}

impl BlockManager {
    /// Create a `BlockManager` with a private block allocator.
    pub fn new(total_blocks: usize, block_size: usize, device_id: usize) -> Self {
        Self {
            allocator: BlockManagerAllocator::Owned(BlockAllocator::new(
                total_blocks,
                block_size,
                device_id,
            )),
            block_tables: HashMap::new(),
            block_size,
            held_blocks: 0,
            live_cap: None,
            engine_id: None,
        }
    }

    /// Create a `BlockManager` that draws from a shared pool.
    ///
    /// Multiple engines sharing the same `SharedBlockAllocator` form a unified
    /// KV block pool: blocks freed by one engine become immediately available
    /// to any other engine holding a clone of the same allocator.
    pub fn new_shared(allocator: SharedBlockAllocator, block_size: usize) -> Self {
        Self {
            allocator: BlockManagerAllocator::Shared(allocator),
            block_tables: HashMap::new(),
            block_size,
            held_blocks: 0,
            live_cap: None,
            engine_id: None,
        }
    }

    /// Set this manager's engine id so allocations stamp ownership metadata.
    pub fn set_engine_id(&mut self, engine_id: u32) {
        self.engine_id = Some(engine_id);
    }

    /// Current owner of a physical block in the pool (diagnostics / validation).
    pub fn block_owner(&self, block_number: usize) -> Option<BlockOwner> {
        self.allocator.owner_of(block_number)
    }

    /// Attach a hard per-engine block quota (shared with the runtime, updated on
    /// engine join/leave). Once set, `allocate` / `can_allocate` will not let
    /// this engine exceed the cap regardless of free blocks in the shared pool.
    pub fn set_live_cap(&mut self, cap: Arc<AtomicUsize>) {
        self.live_cap = Some(cap);
    }

    /// Number of blocks this manager currently holds across all sequences.
    pub fn held_blocks(&self) -> usize {
        self.held_blocks
    }

    /// Current quota ceiling, or `None` if uncapped.
    fn cap(&self) -> Option<usize> {
        self.live_cap.as_ref().map(|c| c.load(Ordering::Relaxed))
    }

    pub fn allocate(&mut self, sequence_id: u64) -> Option<PhysicalTokenBlock> {
        // Hard per-engine quota: refuse once this engine is at its cap, even if
        // the shared pool still has free blocks reserved for other models.
        if let Some(cap) = self.cap() {
            if self.held_blocks >= cap {
                return None;
            }
        }
        // Stamp ownership when this manager has an engine id, so blocks drawn
        // from a shared pool are traceable back to their owning engine/sequence.
        let allocated = match self.engine_id {
            Some(engine_id) => self.allocator.allocate_for(BlockOwner {
                engine_id,
                sequence_id,
            }),
            None => self.allocator.allocate(),
        };
        if let Some(block) = allocated {
            self.block_tables
                .entry(sequence_id)
                .or_insert_with(|| BlockTable::new(self.block_size))
                .append(block);
            self.held_blocks += 1;
            Some(block)
        } else {
            None
        }
    }

    pub fn free(&mut self, sequence_id: u64) {
        if let Some(table) = self.block_tables.remove(&sequence_id) {
            let count = table.len();
            for block in table.get_physical_blocks() {
                self.allocator.free(*block);
            }
            self.held_blocks = self.held_blocks.saturating_sub(count);
        }
    }

    pub fn get_block_table(&self, sequence_id: u64) -> Option<&BlockTable> {
        self.block_tables.get(&sequence_id)
    }

    pub fn can_allocate(&self, num_blocks: usize) -> bool {
        self.allocatable_blocks() >= num_blocks
    }

    /// Number of blocks this manager could allocate immediately, accounting
    /// for both the shared physical pool and its live per-engine quota.
    ///
    /// Schedulers use this to request only the actual shortage during
    /// priority preemption. Asking a donor to release the full request size
    /// would otherwise evict more work than necessary when some headroom is
    /// already available locally.
    pub fn allocatable_blocks(&self) -> usize {
        let physical = self.allocator.get_num_free_blocks();
        self.cap()
            .map(|cap| physical.min(cap.saturating_sub(self.held_blocks)))
            .unwrap_or(physical)
    }

    pub fn block_size(&self) -> usize {
        self.block_size
    }

    /// Number of blocks currently held by `sequence_id`.
    pub fn blocks_for_sequence(&self, sequence_id: u64) -> usize {
        self.block_tables
            .get(&sequence_id)
            .map(|t| t.len())
            .unwrap_or(0)
    }

    /// Current free-block count in the pool (owned or shared).
    pub fn free_blocks(&self) -> usize {
        self.allocator.get_num_free_blocks()
    }

    /// Total block count in the pool (owned or shared).
    pub fn total_blocks(&self) -> usize {
        self.allocator.get_num_total_blocks()
    }
}

impl Drop for BlockManager {
    fn drop(&mut self) {
        let seq_ids: Vec<u64> = self.block_tables.keys().copied().collect();
        for seq_id in seq_ids {
            self.free(seq_id);
        }
    }
}

#[path = "block_manager_tests.rs"]
mod block_manager_tests;
