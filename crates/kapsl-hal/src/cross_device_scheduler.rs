//! Cross-device KV block pool scheduler.
//!
//! Manages a fleet of `GpuBlockPool`s across multiple CUDA devices plus an
//! optional CPU eviction tier, presenting a single admission interface to
//! request handlers.
//!
//! # What this enables vs vLLM
//!
//! vLLM's PagedAttention allocates KV blocks dynamically, but its memory
//! budget is still per-process (`gpu_memory_utilization`). When one process
//! is idle, its GPU memory is wasted.
//!
//! `CrossDevicePoolScheduler` is GPU-wide: it sees every device and can move
//! idle sessions to a less-loaded device or to CPU, then restore them when
//! needed. This lets the GPU fleet serve many more concurrent models than a
//! static per-process allocation would.
//!
//! # Admission flow
//!
//! ```text
//! Request arrives
//!   └─► reserve_blocks()
//!         ├─ Try preferred device (most cache-warm)
//!         ├─ Try other devices by free-block count descending
//!         └─ If all at capacity:
//!               evict_until_capacity() — LRU sessions → CPU
//!               retry preferred device
//! ```
//!
//! # Migration flow
//!
//! ```text
//! Device A pressure > threshold
//!   └─► find sessions on A, sort by LRU
//!         ├─ For each session: evict_session_to_cpu()
//!         │     download GPU blocks → CpuBlockStore, free GPU blocks
//!         └─ On next request for that session:
//!               restore_session_to_gpu() — uploads CPU blocks → GPU
//! ```

#[cfg(feature = "cuda")]
mod inner {
    use std::collections::HashMap;
    use std::sync::Arc;
    use std::time::Instant;

    use crate::cpu_block_store::{CpuBlockStore, CpuStoreError};
    use crate::gpu_arena::{ArenaError, GpuBlockPool};

    // ── Error ──────────────────────────────────────────────────────���──────────

    #[derive(Debug, thiserror::Error)]
    pub enum SchedulerError {
        #[error("No device has {blocks_needed} free blocks for geometry ({kv_heads}h × {head_dim}d)")]
        InsufficientCapacity { blocks_needed: usize, kv_heads: usize, head_dim: usize },
        #[error("Session {0} not found")]
        SessionNotFound(u64),
        #[error("Device {0} not registered")]
        DeviceNotFound(usize),
        #[error("GPU error: {0}")]
        Gpu(#[from] ArenaError),
        #[error("CPU store error: {0}")]
        CpuStore(#[from] CpuStoreError),
    }

    // ── Session location ──────────────────────────────────────────────────────

    /// Where a session's KV blocks currently reside.
    #[derive(Debug, Clone, PartialEq, Eq)]
    pub enum KvTier {
        Gpu { device_id: usize },
        Cpu,
    }

    #[derive(Debug)]
    pub struct SessionState {
        pub tier:        KvTier,
        /// The GPU pool that holds the blocks when `tier == Gpu`. None on CPU.
        pub pool:        Option<Arc<GpuBlockPool>>,
        /// Physical block indices (GPU block IDs when on GPU, CPU slot IDs when on CPU).
        pub blocks:      Vec<u32>,
        /// Last time this session executed a forward pass — used for LRU eviction.
        pub last_active: Instant,
    }

    // ── Reserve result ────────────────────────────────────────────────────────

    pub struct ReserveResult {
        pub device_id: usize,
        pub pool:      Arc<GpuBlockPool>,
        pub blocks:    Vec<u32>,
    }

    // ── Scheduler ─────────────────────────────────────────────────────────────

    /// GPU-wide KV block admission and migration controller.
    pub struct CrossDevicePoolScheduler {
        /// device_id → list of pools registered for that device.
        device_pools: HashMap<usize, Vec<Arc<GpuBlockPool>>>,
        /// session_id → current location + block ownership.
        sessions: HashMap<u64, SessionState>,
        /// Pool utilisation fraction above which LRU eviction to CPU is triggered.
        evict_threshold: f32,
    }

    impl CrossDevicePoolScheduler {
        pub fn new(evict_threshold: f32) -> Self {
            Self {
                device_pools: HashMap::new(),
                sessions: HashMap::new(),
                evict_threshold: evict_threshold.clamp(0.5, 0.99),
            }
        }

        // ── Pool registration ─────────────────────────────────────────────────

        /// Register a `GpuBlockPool` for a device.  Multiple pools per device
        /// are allowed (e.g. one per KV geometry).
        pub fn register_pool(&mut self, device_id: usize, pool: Arc<GpuBlockPool>) {
            self.device_pools.entry(device_id).or_default().push(pool);
        }

        /// Remove all pools for a device (e.g. on hot-unplug or shutdown).
        pub fn unregister_device(&mut self, device_id: usize) {
            self.device_pools.remove(&device_id);
        }

        // ── Session lifecycle ─────────────────────────────────────────────────

        /// Reserve `blocks_needed` blocks for a session.
        ///
        /// 1. Tries `preferred_device` first.
        /// 2. Falls back to other devices ordered by free-block count.
        /// 3. If all devices are at capacity, evicts LRU sessions to `cpu_store`
        ///    and retries on `preferred_device`.
        pub fn reserve_blocks(
            &mut self,
            session_id: u64,
            blocks_needed: usize,
            preferred_device: Option<usize>,
            kv_heads: usize,
            head_dim: usize,
            cpu_store: &mut CpuBlockStore,
        ) -> Result<ReserveResult, SchedulerError> {
            // 1. Preferred device
            if let Some(dev) = preferred_device {
                if let Some(r) = self.try_reserve_on(dev, blocks_needed, kv_heads, head_dim)? {
                    self.record_session(session_id, r.device_id, r.pool.clone(), r.blocks.clone());
                    return Ok(r);
                }
            }

            // 2. All other devices, most-free first
            let mut others: Vec<usize> = self.device_pools.keys()
                .filter(|&&d| Some(d) != preferred_device)
                .copied()
                .collect();
            others.sort_by_key(|&d| {
                usize::MAX - self.device_free_blocks(d, kv_heads, head_dim)
            });
            for dev in others {
                if let Some(r) = self.try_reserve_on(dev, blocks_needed, kv_heads, head_dim)? {
                    self.record_session(session_id, r.device_id, r.pool.clone(), r.blocks.clone());
                    return Ok(r);
                }
            }

            // 3. Evict LRU to CPU, then retry preferred (or first) device
            let target = preferred_device
                .or_else(|| self.device_pools.keys().copied().next())
                .ok_or(SchedulerError::InsufficientCapacity { blocks_needed, kv_heads, head_dim })?;

            self.evict_until_capacity(target, blocks_needed, kv_heads, head_dim, cpu_store)?;

            if let Some(r) = self.try_reserve_on(target, blocks_needed, kv_heads, head_dim)? {
                self.record_session(session_id, r.device_id, r.pool.clone(), r.blocks.clone());
                return Ok(r);
            }

            Err(SchedulerError::InsufficientCapacity { blocks_needed, kv_heads, head_dim })
        }

        /// Release all blocks held by a session (GPU or CPU).
        ///
        /// `cpu_store` is only mutated if the session is on CPU.
        pub fn release_session(&mut self, session_id: u64, cpu_store: &mut CpuBlockStore) {
            let Some(state) = self.sessions.remove(&session_id) else { return };
            match state.tier {
                KvTier::Gpu { .. } => {
                    if let Some(pool) = state.pool {
                        for b in state.blocks { pool.free_block(b); }
                    }
                }
                KvTier::Cpu => {
                    cpu_store.free_slots_bulk(&state.blocks);
                }
            }
        }

        /// Update the LRU timestamp for a session that just completed a forward pass.
        pub fn mark_active(&mut self, session_id: u64) {
            if let Some(s) = self.sessions.get_mut(&session_id) {
                s.last_active = Instant::now();
            }
        }

        // ── Cross-tier migration ──────────────────────────────────────────────

        /// Evict a GPU session to CPU storage, freeing its GPU blocks.
        pub fn evict_session_to_cpu(
            &mut self,
            session_id: u64,
            cpu_store: &mut CpuBlockStore,
        ) -> Result<(), SchedulerError> {
            let state = self.sessions.get(&session_id)
                .ok_or(SchedulerError::SessionNotFound(session_id))?;

            // Already on CPU — nothing to do.
            if state.tier == KvTier::Cpu { return Ok(()); }

            let pool       = state.pool.clone().unwrap();
            let gpu_blocks = state.blocks.clone();

            let mut cpu_slots = Vec::with_capacity(gpu_blocks.len());
            for &blk in &gpu_blocks {
                let data = pool.download_block(blk)?;
                let slot = cpu_store.store_block(&data)?;
                pool.free_block(blk);
                cpu_slots.push(slot);
            }

            let state = self.sessions.get_mut(&session_id).unwrap();
            state.tier   = KvTier::Cpu;
            state.pool   = None;
            state.blocks = cpu_slots;

            log::debug!(
                "[xdev-sched] evicted session {} → CPU ({} blocks)",
                session_id, gpu_blocks.len(),
            );
            Ok(())
        }

        /// Restore a CPU-offloaded session to a GPU device.
        pub fn restore_session_to_gpu(
            &mut self,
            session_id: u64,
            target_device: usize,
            kv_heads: usize,
            head_dim: usize,
            cpu_store: &CpuBlockStore,
        ) -> Result<(), SchedulerError> {
            let state = self.sessions.get(&session_id)
                .ok_or(SchedulerError::SessionNotFound(session_id))?;

            if state.tier != KvTier::Cpu { return Ok(()); }

            let n_blocks  = state.blocks.len();
            let cpu_slots = state.blocks.clone();

            let result = self.try_reserve_on(target_device, n_blocks, kv_heads, head_dim)?
                .ok_or(SchedulerError::InsufficientCapacity {
                    blocks_needed: n_blocks, kv_heads, head_dim
                })?;

            for (&gpu_blk, &cpu_slot) in result.blocks.iter().zip(&cpu_slots) {
                let data     = cpu_store.load_block(cpu_slot)?;
                let half_len = data.len() / 2;
                result.pool.upload_block(gpu_blk, &data[..half_len], &data[half_len..])?;
            }

            let state = self.sessions.get_mut(&session_id).unwrap();
            state.tier   = KvTier::Gpu { device_id: target_device };
            state.pool   = Some(result.pool);
            state.blocks = result.blocks;

            log::debug!(
                "[xdev-sched] restored session {} → GPU {} ({} blocks)",
                session_id, target_device, n_blocks,
            );
            Ok(())
        }

        /// Move a session from one GPU to another.
        ///
        /// Blocks are copied through host memory (PCIe / NVLink, no peer access
        /// required). On NVLink systems this completes in tens of microseconds
        /// per block; on PCIe in hundreds.
        pub fn migrate_session(
            &mut self,
            session_id: u64,
            target_device: usize,
            kv_heads: usize,
            head_dim: usize,
        ) -> Result<(), SchedulerError> {
            let state = self.sessions.get(&session_id)
                .ok_or(SchedulerError::SessionNotFound(session_id))?;

            let KvTier::Gpu { device_id: src_device } = state.tier else {
                return Err(SchedulerError::SessionNotFound(session_id));
            };
            if src_device == target_device { return Ok(()); }

            let n_blocks  = state.blocks.len();
            let src_pool  = state.pool.clone().unwrap();
            let src_blocks = state.blocks.clone();

            let result = self.try_reserve_on(target_device, n_blocks, kv_heads, head_dim)?
                .ok_or(SchedulerError::InsufficientCapacity {
                    blocks_needed: n_blocks, kv_heads, head_dim
                })?;

            for (&src_blk, &dst_blk) in src_blocks.iter().zip(&result.blocks) {
                src_pool.copy_block_to_pool(src_blk, &result.pool, dst_blk)?;
                src_pool.free_block(src_blk);
            }

            let state = self.sessions.get_mut(&session_id).unwrap();
            state.tier   = KvTier::Gpu { device_id: target_device };
            state.pool   = Some(result.pool);
            state.blocks = result.blocks;

            log::info!(
                "[xdev-sched] migrated session {} GPU {}→{} ({} blocks)",
                session_id, src_device, target_device, n_blocks,
            );
            Ok(())
        }

        // ── Pressure / metrics ────────────────────────────────────────────────

        /// Pool utilisation on a device (0.0 = empty, 1.0 = full).
        pub fn device_pressure(&self, device_id: usize) -> f32 {
            let Some(pools) = self.device_pools.get(&device_id) else { return 0.0 };
            let (total, used) = pools.iter()
                .fold((0usize, 0usize), |(t, u), p| (t + p.total_blocks(), u + p.used_count()));
            if total == 0 { 0.0 } else { used as f32 / total as f32 }
        }

        /// Total free blocks on a device for the given KV geometry.
        pub fn device_free_blocks(&self, device_id: usize, kv_heads: usize, head_dim: usize) -> usize {
            self.device_pools.get(&device_id)
                .map(|pools| pools.iter()
                    .filter(|p| p.is_compatible(kv_heads, head_dim))
                    .map(|p| p.free_count())
                    .sum())
                .unwrap_or(0)
        }

        pub fn registered_devices(&self) -> Vec<usize> {
            let mut v: Vec<usize> = self.device_pools.keys().copied().collect();
            v.sort_unstable();
            v
        }

        pub fn session_count(&self)                       -> usize { self.sessions.len() }
        pub fn sessions_on_cpu(&self)                     -> usize {
            self.sessions.values().filter(|s| s.tier == KvTier::Cpu).count()
        }
        pub fn sessions_on_device(&self, device_id: usize) -> usize {
            self.sessions.values()
                .filter(|s| s.tier == KvTier::Gpu { device_id })
                .count()
        }

        // ── Internal helpers ──────────────────────────────────────────────────

        fn try_reserve_on(
            &self,
            device_id: usize,
            n: usize,
            kv_heads: usize,
            head_dim: usize,
        ) -> Result<Option<ReserveResult>, SchedulerError> {
            let Some(pools) = self.device_pools.get(&device_id) else {
                return Ok(None);
            };
            for pool in pools {
                if !pool.is_compatible(kv_heads, head_dim) { continue; }
                if pool.free_count() < n { continue; }

                let mut blocks = Vec::with_capacity(n);
                let mut ok = true;
                for _ in 0..n {
                    match pool.alloc_block() {
                        Ok(b)  => blocks.push(b),
                        Err(_) => { ok = false; break; }
                    }
                }
                if ok {
                    return Ok(Some(ReserveResult { device_id, pool: pool.clone(), blocks }));
                }
                // Partial allocation: roll back and try next pool.
                for b in blocks { pool.free_block(b); }
            }
            Ok(None)
        }

        fn record_session(
            &mut self,
            session_id: u64,
            device_id: usize,
            pool: Arc<GpuBlockPool>,
            blocks: Vec<u32>,
        ) {
            self.sessions.insert(session_id, SessionState {
                tier:        KvTier::Gpu { device_id },
                pool:        Some(pool),
                blocks,
                last_active: Instant::now(),
            });
        }

        /// Evict least-recently-used sessions from `device_id` until at least
        /// `blocks_needed` blocks are free.
        fn evict_until_capacity(
            &mut self,
            device_id: usize,
            blocks_needed: usize,
            kv_heads: usize,
            head_dim: usize,
            cpu_store: &mut CpuBlockStore,
        ) -> Result<(), SchedulerError> {
            // Snapshot candidate session IDs (oldest first) before mutating.
            let mut candidates: Vec<(u64, Instant)> = self.sessions.iter()
                .filter(|(_, s)| {
                    s.tier == KvTier::Gpu { device_id } &&
                    s.pool.as_ref().map_or(false, |p| p.is_compatible(kv_heads, head_dim))
                })
                .map(|(&id, s)| (id, s.last_active))
                .collect();
            candidates.sort_by_key(|(_, t)| *t);

            for (evict_id, _) in candidates {
                if self.device_free_blocks(device_id, kv_heads, head_dim) >= blocks_needed {
                    break;
                }
                self.evict_session_to_cpu(evict_id, cpu_store)?;
            }
            Ok(())
        }
    }
}

#[cfg(feature = "cuda")]
pub use inner::{CrossDevicePoolScheduler, KvTier, ReserveResult, SchedulerError, SessionState};
