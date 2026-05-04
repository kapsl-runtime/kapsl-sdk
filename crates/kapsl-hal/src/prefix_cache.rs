//! Prefix KV-block cache for prompt-prefix reuse.
//!
//! When multiple requests share the same prompt prefix (system prompt, few-shot
//! examples, document context), their KV blocks for that prefix are identical.
//! Instead of recomputing and re-allocating them per request, this cache keeps
//! those GPU blocks alive and loans them out by reference.
//!
//! # Block hashing
//!
//! Each block's identity is a chained hash:
//!
//! ```text
//! hash[0] = H(model_fingerprint, 0,       tokens[0..block_size])
//! hash[i] = H(model_fingerprint, hash[i-1], tokens[i*B..(i+1)*B])
//! ```
//!
//! Chaining means `hash[i]` implicitly encodes the entire prefix up to and
//! including block `i`, so two blocks at position `i` only match if the whole
//! preceding context is also identical.  Trailing partial blocks are excluded
//! because their KV content is not yet final.
//!
//! # Ownership and refcounts
//!
//! A cached entry holds a single GPU block.  Ownership rules:
//!
//! - `refcount == 0` — no active session is using this block; eligible for LRU
//!   eviction.
//! - `refcount > 0` — one or more sessions are using this block; it cannot be
//!   evicted until all borrows are released.
//!
//! `insert` takes a `refcount` argument:
//! - Pass `1` when the session that just computed the block still needs it.
//! - Pass `0` for a block that is immediately available for sharing.
//!
//! `lookup` increments refcounts for every hit.
//! `release` decrements one borrow (called when a session ends).

use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::collections::hash_map::DefaultHasher;
use std::sync::Arc;
use std::time::Instant;

use crate::gpu_arena::GpuBlockPool;

// ── Cache key ─────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct BlockCacheKey {
    pub model_fingerprint: u64,
    pub block_hash:        u64,
}

// ── Entry (internal) ──────────────────────────────────────────────────────────

struct PrefixCacheEntry {
    device_id: usize,
    pool:      Arc<GpuBlockPool>,
    block_id:  u32,
    refcount:  usize,
    last_used: Instant,
}

// ── Public borrow handle ──────────────────────────────────────────────────────

/// A borrowed reference to a cached KV block, returned by [`PrefixBlockCache::lookup`].
pub struct CachedBlockRef {
    pub device_id:  usize,
    pub pool:       Arc<GpuBlockPool>,
    pub block_id:   u32,
    /// The chained hash that identifies this block in the cache.
    pub block_hash: u64,
}

// ── Cache ─────────────────────────────────────────────────────────────────────

/// In-process GPU KV-block prefix cache.
pub struct PrefixBlockCache {
    entries:  HashMap<BlockCacheKey, PrefixCacheEntry>,
    capacity: usize,
}

impl PrefixBlockCache {
    pub fn new(capacity: usize) -> Self {
        Self { entries: HashMap::new(), capacity }
    }

    // ── Hashing helpers ───────────────────────────────────────────────────────

    /// Compute the chained hash for a single KV block.
    ///
    /// `prev_hash` must be `0` for the first block and the hash of the
    /// immediately preceding block otherwise.
    pub fn compute_block_hash(model_fingerprint: u64, prev_hash: u64, tokens: &[u32]) -> u64 {
        let mut h = DefaultHasher::new();
        model_fingerprint.hash(&mut h);
        prev_hash.hash(&mut h);
        tokens.hash(&mut h);
        h.finish()
    }

    /// Compute per-block chained hashes for `tokens`.
    ///
    /// Only complete blocks of `block_size` tokens are hashed — the trailing
    /// partial block is excluded because its KV state is not yet final.
    pub fn compute_prefix_hashes(
        model_fingerprint: u64,
        tokens: &[u32],
        block_size: usize,
    ) -> Vec<u64> {
        let mut hashes  = Vec::new();
        let mut prev    = 0u64;
        for chunk in tokens.chunks(block_size) {
            if chunk.len() < block_size { break; }
            let h = Self::compute_block_hash(model_fingerprint, prev, chunk);
            hashes.push(h);
            prev = h;
        }
        hashes
    }

    // ── Core operations ───────────────────────────────────────────────────────

    /// Look up a contiguous prefix of cached blocks, incrementing refcounts.
    ///
    /// Stops at the first miss — downstream blocks are invalid without their
    /// full preceding context.
    pub fn lookup(
        &mut self,
        model_fingerprint: u64,
        hashes: &[u64],
    ) -> Vec<CachedBlockRef> {
        let mut result = Vec::new();
        for &hash in hashes {
            let key = BlockCacheKey { model_fingerprint, block_hash: hash };
            match self.entries.get_mut(&key) {
                Some(entry) => {
                    entry.refcount  += 1;
                    entry.last_used  = Instant::now();
                    result.push(CachedBlockRef {
                        device_id:  entry.device_id,
                        pool:       entry.pool.clone(),
                        block_id:   entry.block_id,
                        block_hash: hash,
                    });
                }
                None => break,
            }
        }
        result
    }

    /// Insert a block into the cache.
    ///
    /// If the entry already exists, returns `true` immediately (idempotent).
    /// If the cache is at capacity, evicts the oldest zero-refcount entry.
    /// Returns `false` and **frees `block_id`** when no evictable entry exists.
    pub fn insert(
        &mut self,
        model_fingerprint: u64,
        block_hash: u64,
        device_id: usize,
        pool: Arc<GpuBlockPool>,
        block_id: u32,
        refcount: usize,
    ) -> bool {
        let key = BlockCacheKey { model_fingerprint, block_hash };
        if self.entries.contains_key(&key) {
            return true;
        }
        if self.entries.len() >= self.capacity && !self.evict_one_lru() {
            pool.free_block(block_id);
            return false;
        }
        self.entries.insert(key, PrefixCacheEntry {
            device_id,
            pool,
            block_id,
            refcount,
            last_used: Instant::now(),
        });
        true
    }

    /// Release one borrow on a cached block (decrement refcount).
    ///
    /// When refcount reaches 0 the block becomes eligible for LRU eviction.
    pub fn release(&mut self, model_fingerprint: u64, block_hash: u64) {
        let key = BlockCacheKey { model_fingerprint, block_hash };
        if let Some(entry) = self.entries.get_mut(&key) {
            entry.refcount = entry.refcount.saturating_sub(1);
        }
    }

    // ── Eviction ──────────────────────────────────────────────────────────────

    /// Evict up to `n` zero-refcount blocks from `device_id`, oldest first.
    ///
    /// Returns `(pool, block_id)` pairs.  The caller is responsible for calling
    /// `pool.free_block(block_id)` to return memory to the GPU allocator.
    pub fn evict_lru_for_device(
        &mut self,
        device_id: usize,
        n: usize,
    ) -> Vec<(Arc<GpuBlockPool>, u32)> {
        let mut candidates: Vec<(BlockCacheKey, Instant)> = self.entries.iter()
            .filter(|(_, e)| e.device_id == device_id && e.refcount == 0)
            .map(|(k, e)| (k.clone(), e.last_used))
            .collect();
        candidates.sort_by_key(|(_, t)| *t);

        let mut freed = Vec::new();
        for (key, _) in candidates.into_iter().take(n) {
            if let Some(entry) = self.entries.remove(&key) {
                freed.push((entry.pool, entry.block_id));
            }
        }
        freed
    }

    /// Evict up to `n` zero-refcount blocks globally, oldest first.
    pub fn evict_lru(&mut self, n: usize) -> Vec<(Arc<GpuBlockPool>, u32)> {
        let mut candidates: Vec<(BlockCacheKey, Instant)> = self.entries.iter()
            .filter(|(_, e)| e.refcount == 0)
            .map(|(k, e)| (k.clone(), e.last_used))
            .collect();
        candidates.sort_by_key(|(_, t)| *t);

        let mut freed = Vec::new();
        for (key, _) in candidates.into_iter().take(n) {
            if let Some(entry) = self.entries.remove(&key) {
                freed.push((entry.pool, entry.block_id));
            }
        }
        freed
    }

    // ── Metrics ───────────────────────────────────────────────────────────────

    pub fn entry_count(&self) -> usize { self.entries.len() }
    pub fn capacity(&self)    -> usize { self.capacity }
    pub fn is_full(&self)     -> bool  { self.entries.len() >= self.capacity }

    // ── Internal ──────────────────────────────────────────────────────────────

    /// Evict the single least-recently-used zero-refcount entry, freeing its
    /// GPU block.  Returns `false` if every entry has a live borrow.
    fn evict_one_lru(&mut self) -> bool {
        let key = self.entries.iter()
            .filter(|(_, e)| e.refcount == 0)
            .min_by_key(|(_, e)| e.last_used)
            .map(|(k, _)| k.clone());
        match key {
            Some(k) => {
                if let Some(entry) = self.entries.remove(&k) {
                    entry.pool.free_block(entry.block_id);
                }
                true
            }
            None => false,
        }
    }
}
