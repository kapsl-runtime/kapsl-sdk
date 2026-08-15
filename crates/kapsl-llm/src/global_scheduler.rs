//! Cross-model KV token-budget coordinator.
//!
//! # Purpose
//!
//! Each [`LLMEngine`] owns an [`LLMScheduler`] that manages one model's
//! sequences independently.  Without a shared coordinator, a single busy model
//! can consume the entire GPU KV block pool and starve all others.
//!
//! [`GlobalKvScheduler`] sits above the per-engine schedulers and enforces
//! two things:
//!
//! 1. **Proportional token budget** — each registered engine is allocated a
//!    slice of the global `max_batched_tokens` budget proportional to its
//!    declared share weight.  An engine that goes idle donates its unused budget
//!    to active engines up to the configured maximum.
//!
//! 2. **Cross-model preemption** — when an engine cannot schedule a high-
//!    priority request because its own block budget is exhausted, it can ask the
//!    global scheduler to reclaim blocks from lower-priority engines.
//!
//! # Integration
//!
//! The global scheduler computes per-engine budgets and routes executable
//! preemption commands.  Each [`LLMEngine`] services commands in its scheduling
//! loop, releases the corresponding physical KV entries, and reports the
//! result so block-cap loans can be tracked and restored.
//!
//! ```no_run
//! use kapsl_llm::block_manager::new_shared_allocator;
//! use kapsl_llm::global_scheduler::{GlobalKvScheduler, EngineHandle};
//!
//! // Build one shared block allocator for the device.
//! let shared_pool = new_shared_allocator(4096, 16, 0);
//!
//! // Create the global scheduler.
//! let mut global = GlobalKvScheduler::new(8192); // 8192 tokens / round
//!
//! // Register engines as they load.
//! global.register(EngineHandle {
//!     engine_id: 0,
//!     share_weight: 1,
//!     guaranteed_min_tokens: 0,
//!     max_tokens: None,
//! });
//! global.register(EngineHandle {
//!     engine_id: 1,
//!     share_weight: 2, // 2× the budget
//!     guaranteed_min_tokens: 0,
//!     max_tokens: None,
//! });
//!
//! // Each scheduling round, ask for per-engine budgets.
//! let budgets = global.allocate_budgets();
//! // budgets[0].max_tokens ≈ 2730  (1/3 of 8192)
//! // budgets[1].max_tokens ≈ 5461  (2/3 of 8192)
//! ```

use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

/// Maximum times a single engine may be selected as a preemption donor per
/// scheduling round. Prevents a greedy engine from repeatedly draining one
/// peer across multiple back-to-back preemption requests in the same round.
const MAX_PREEMPTIONS_DONATED_PER_ROUND: usize = 2;

/// Operational health of a registered engine, as seen by the global scheduler.
///
/// The scheduler uses this to stop sending work to engines that are failing or
/// hung, so that a single bad model cannot consume budget or be selected as a
/// preemption donor while it is unable to make progress.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum EngineHealth {
    /// Engine is processing normally; eligible for budget and donation.
    #[default]
    Healthy,
    /// Engine is failing (e.g. its circuit breaker tripped) and is being
    /// drained. Receives no new budget and is never picked as a preemption
    /// donor, but may finish in-flight work and can recover to [`EngineHealth::Healthy`].
    Degraded,
    /// Engine is unresponsive or gone (e.g. a stalled `run_loop` detected by the
    /// watchdog). Receives no budget, is never a donor, and its share is
    /// redistributed to healthy engines until it recovers or is deregistered.
    Dead,
}

/// Panic-safe, shared handle to the cross-model [`GlobalKvScheduler`].
///
/// Uses `parking_lot::Mutex` so that a panic while the lock is held does not
/// poison the mutex and cascade a failure to every other engine waiting on the
/// scheduler.
pub type SharedGlobalScheduler = std::sync::Arc<parking_lot::Mutex<GlobalKvScheduler>>;

/// Lightweight descriptor for one registered engine.
#[derive(Debug, Clone)]
pub struct EngineHandle {
    /// Stable identifier that maps to a loaded model/engine instance.
    pub engine_id: u32,
    /// Relative share weight.  An engine with weight 2 gets twice the token
    /// budget of one with weight 1 when both are active.
    pub share_weight: u32,
    /// Minimum tokens always guaranteed to this engine regardless of load.
    /// Defaults to 0 (no guarantee).
    pub guaranteed_min_tokens: usize,
    /// Hard cap on this engine's total token budget (elastic + guaranteed).
    /// `None` means uncapped — the engine may absorb idle peers' shares.
    pub max_tokens: Option<usize>,
}

/// Per-engine token budget issued by [`GlobalKvScheduler::allocate_budgets`].
#[derive(Debug, Clone)]
pub struct EngineTokenBudget {
    pub engine_id: u32,
    /// Maximum number of tokens (prefill + decode) the engine may schedule
    /// in the current round.
    pub max_tokens: usize,
}

/// Cross-engine preemption request raised when an engine has insufficient
/// free blocks for an incoming request.
#[derive(Debug, Clone)]
pub struct PreemptionRequest {
    /// Engine that is requesting more blocks.
    pub requesting_engine_id: u32,
    /// Minimum number of blocks needed.
    pub blocks_needed: usize,
    /// Priority of the incoming request — used to target engines that are
    /// currently running lower-priority work.
    pub request_priority: u8,
}

/// Result of a preemption request.
#[derive(Debug, Clone)]
pub struct PreemptionResult {
    /// Engine that was asked to free blocks.
    pub donor_engine_id: u32,
    /// Number of blocks that were freed on that engine's scheduler.
    pub blocks_freed: usize,
}

#[derive(Debug, Clone, Copy, Default)]
struct EnginePreemptionState {
    /// Least-important running tier. Larger values are more important in the
    /// ONNX sequence scheduler, so this is the first tier a donor may evict.
    lowest_running_priority: Option<u8>,
    /// Blocks held by that tier and therefore safe to advertise atomically.
    freeable_blocks: usize,
}

#[derive(Debug, Clone, Copy)]
struct PreemptionLoan {
    donor_engine_id: u32,
    borrower_engine_id: u32,
    blocks: usize,
}

/// Internal per-engine state tracked by the global scheduler.
#[derive(Debug)]
struct EngineState {
    share_weight: u32,
    /// Tokens consumed during the most recent scheduling round (filled in by
    /// the caller via [`GlobalKvScheduler::report_usage`]).
    last_used_tokens: usize,
    /// Whether the engine had any pending requests in the last round.
    was_active: bool,
    /// Elastic quota lower bound — always reserved for this engine.
    guaranteed_min_tokens: usize,
    /// Elastic quota upper bound — engine never exceeds this, even when
    /// absorbing idle peers' shares.  `None` = uncapped.
    max_tokens: Option<usize>,
    /// Operational health. Non-[`EngineHealth::Healthy`] engines receive zero
    /// budget and are excluded from preemption donor selection.
    health: EngineHealth,
}

/// Cross-model token-budget coordinator.
///
/// Maintains a registry of active [`EngineHandle`]s and distributes the
/// global token budget proportionally each scheduling round. Also accepts and
/// routes cross-engine preemption requests.
///
/// ## Hard admission (Phase 2)
///
/// Callers use [`try_reserve_tokens`] to claim a token budget before queuing a
/// request.  The reservation is deducted from the engine's current computed
/// budget so that concurrent requests can't overcommit.  [`release_tokens`]
/// returns the budget when the request completes or is cancelled.
#[derive(Debug)]
pub struct GlobalKvScheduler {
    /// Total tokens available per scheduling round across all engines.
    global_max_tokens: usize,
    engines: HashMap<u32, EngineState>,
    /// Ordered list of engine IDs in registration order (for stable iteration).
    engine_order: Vec<u32>,
    /// Maximum fraction of the global budget any single engine may consume,
    /// expressed as a per-mille value (1000 = 100 %).  Prevents monopolisation
    /// even when all other engines are idle.  Default: 900 ‰ (90 %).
    max_single_engine_permille: u32,
    /// In-flight token reservations per engine (engine_id → tokens reserved).
    /// Updated atomically by try_reserve_tokens / complete_tokens.
    reserved_tokens: HashMap<u32, usize>,
    /// Number of requests currently in-flight per engine.
    /// Used to drive set_active: the engine is marked active when inflight 0→1
    /// and idle when inflight 1→0, so allocate_budgets reflects real activity.
    inflight: HashMap<u32, usize>,
    /// How many times each engine has been a preemption donor in the current
    /// round. Reset by [`reset_preemption_round`] at the start of each round.
    preemption_donations_this_round: HashMap<u32, usize>,
    /// Latest donor state published by each live engine loop.
    preemption_states: HashMap<u32, EnginePreemptionState>,
    /// Routed commands waiting for the selected donor engine to execute them.
    pending_preemptions: HashMap<u32, VecDeque<PreemptionRequest>>,
    /// Requester -> donor while a command is queued or executing. This keeps a
    /// blocked engine's fast loop from enqueuing the same reclaim repeatedly.
    pending_requesters: HashMap<u32, u32>,
    /// Live per-engine block caps supplied by the runtime. Successful
    /// preemption transfers cap from donor to borrower for the lifetime of the
    /// borrower's in-flight high-priority work, preventing the donor from
    /// immediately reacquiring the blocks it just released.
    block_caps: HashMap<u32, Arc<AtomicUsize>>,
    preemption_loans: Vec<PreemptionLoan>,
    /// Monotonic counter bumped on every *change* to any engine's health.
    /// The runtime polls this to know when to recompute KV block caps so a
    /// degraded/dead engine's quota is reclaimed for healthy engines.
    health_epoch: u64,
}

impl GlobalKvScheduler {
    /// Create a new coordinator with the given total token budget per round.
    pub fn new(global_max_tokens: usize) -> Self {
        Self {
            global_max_tokens,
            engines: HashMap::new(),
            engine_order: Vec::new(),
            max_single_engine_permille: 900,
            reserved_tokens: HashMap::new(),
            inflight: HashMap::new(),
            preemption_donations_this_round: HashMap::new(),
            preemption_states: HashMap::new(),
            pending_preemptions: HashMap::new(),
            pending_requesters: HashMap::new(),
            block_caps: HashMap::new(),
            preemption_loans: Vec::new(),
            health_epoch: 0,
        }
    }

    /// Set the maximum fraction of the global budget any single engine may
    /// receive, in per-mille units (1000 = 100 %).
    pub fn with_max_single_engine_permille(mut self, permille: u32) -> Self {
        self.max_single_engine_permille = permille.clamp(100, 1000);
        self
    }

    /// Register an engine.  If the engine was already registered its weight
    /// and quota bounds are updated.
    pub fn register(&mut self, handle: EngineHandle) {
        if !self.engines.contains_key(&handle.engine_id) {
            self.engine_order.push(handle.engine_id);
            // Clear any stale reservation from a previous incarnation.
            self.reserved_tokens.remove(&handle.engine_id);
        }
        self.engines.insert(
            handle.engine_id,
            EngineState {
                share_weight: handle.share_weight.max(1),
                last_used_tokens: 0,
                was_active: false,
                guaranteed_min_tokens: handle.guaranteed_min_tokens,
                max_tokens: handle.max_tokens,
                // (Re-)registration resets health: a recovered/reloaded engine
                // starts Healthy again.
                health: EngineHealth::Healthy,
            },
        );
    }

    /// Deregister an engine (e.g. after it is unloaded).
    pub fn deregister(&mut self, engine_id: u32) {
        self.clear_preemption_state_for(engine_id);
        self.engines.remove(&engine_id);
        self.engine_order.retain(|&id| id != engine_id);
        self.reserved_tokens.remove(&engine_id);
        self.inflight.remove(&engine_id);
        self.block_caps.remove(&engine_id);
    }

    /// Mark an engine as active or idle for the coming round.
    ///
    /// Idle engines donate their share to active ones (up to the per-engine
    /// cap).  Call this before [`allocate_budgets`].
    pub fn set_active(&mut self, engine_id: u32, active: bool) {
        if let Some(state) = self.engines.get_mut(&engine_id) {
            state.was_active = active;
        }
    }

    /// Update an engine's operational health.
    ///
    /// A non-[`EngineHealth::Healthy`] engine is excluded from budget allocation
    /// and preemption donor selection, so a failing or hung model is isolated
    /// from healthy ones without being fully deregistered. Reported by the
    /// engine's circuit breaker and stall watchdog.
    pub fn set_health(&mut self, engine_id: u32, health: EngineHealth) {
        if let Some(state) = self.engines.get_mut(&engine_id) {
            if state.health != health {
                state.health = health;
                // Only bump on an actual transition so repeated same-value
                // reports (e.g. a circuit breaker re-reporting Degraded each
                // failed step) don't trigger needless cap rebalancing.
                self.health_epoch = self.health_epoch.wrapping_add(1);
                if health != EngineHealth::Healthy {
                    self.clear_preemption_state_for(engine_id);
                }
            }
        }
    }

    /// Monotonic counter incremented on every health transition. The runtime
    /// polls this to know when to recompute KV block caps.
    pub fn health_epoch(&self) -> u64 {
        self.health_epoch
    }

    /// Current health of an engine, or `None` if it is not registered.
    pub fn health_of(&self, engine_id: u32) -> Option<EngineHealth> {
        self.engines.get(&engine_id).map(|s| s.health)
    }

    /// Report how many tokens engine `engine_id` actually consumed last round.
    ///
    /// Used to compute utilisation metrics and to adjust budget donations in
    /// future rounds.
    pub fn report_usage(&mut self, engine_id: u32, used_tokens: usize) {
        if let Some(state) = self.engines.get_mut(&engine_id) {
            state.last_used_tokens = used_tokens;
        }
    }

    /// Compute per-engine token budgets for the current scheduling round.
    ///
    /// ## Algorithm
    ///
    /// 1. **Guarantee phase** — each engine receives at least its
    ///    `guaranteed_min_tokens`, even if idle.
    /// 2. **Elastic phase** — the remaining budget is distributed among *active*
    ///    engines proportional to `share_weight`, capped by each engine's
    ///    `max_tokens` and by `max_single_engine_permille ‰` of the elastic pool.
    /// 3. **Idle donation** — elastic shares from idle engines flow to the first
    ///    active engine (up to its cap).
    /// 4. **Rounding correction** — integer division leftover is added to the
    ///    first active engine.
    ///
    /// When all `guaranteed_min_tokens` are zero and all `max_tokens` are `None`
    /// the output is identical to the previous proportional algorithm.
    pub fn allocate_budgets(&self) -> Vec<EngineTokenBudget> {
        if self.engines.is_empty() || self.global_max_tokens == 0 {
            return Vec::new();
        }

        // ── Guarantee phase ───────────────────────────────────────────────────
        // Only healthy engines reserve guaranteed tokens: an unhealthy engine
        // cannot use its guarantee, so it must not shrink the elastic pool.
        let total_guaranteed: usize = self
            .engine_order
            .iter()
            .filter_map(|id| self.engines.get(id))
            .filter(|s| s.health == EngineHealth::Healthy)
            .map(|s| s.guaranteed_min_tokens)
            .sum();

        // Elastic pool: budget left after honouring all guarantees.
        let elastic_pool = self.global_max_tokens.saturating_sub(total_guaranteed);

        // ── Weight totals ─────────────────────────────────────────────────────
        let active_total_weight: u64 = self
            .engine_order
            .iter()
            .filter_map(|id| self.engines.get(id))
            .filter(|s| s.was_active && s.health == EngineHealth::Healthy)
            .map(|s| s.share_weight as u64)
            .sum();
        let treat_all_active = active_total_weight == 0;

        let all_total_weight: u64 = self
            .engines
            .values()
            .map(|s| s.share_weight as u64)
            .sum::<u64>()
            .max(1);

        // Per-engine elastic cap from the permille setting.
        let elastic_permille_cap =
            (elastic_pool as u64 * self.max_single_engine_permille as u64 / 1000) as usize;

        // ── Per-engine budget ─────────────────────────────────────────────────
        let mut budgets: Vec<EngineTokenBudget> = Vec::with_capacity(self.engines.len());
        let mut idle_elastic_pool: usize = 0;
        let mut natural_elastic_sum: usize = 0;

        for &engine_id in &self.engine_order {
            let Some(state) = self.engines.get(&engine_id) else {
                continue;
            };

            // Natural elastic share based on weight (before caps).
            let natural_elastic =
                (elastic_pool as u64 * state.share_weight as u64 / all_total_weight) as usize;
            natural_elastic_sum += natural_elastic;

            // Unhealthy engines receive no budget at all; their elastic share is
            // donated to healthy active engines (via the idle pool) and they get
            // no guarantee they cannot use.
            if state.health != EngineHealth::Healthy {
                idle_elastic_pool += natural_elastic;
                budgets.push(EngineTokenBudget {
                    engine_id,
                    max_tokens: 0,
                });
                continue;
            }

            let is_active = state.was_active || treat_all_active;

            let elastic_share = if is_active {
                // Cap by per-mille limit and by the engine's own max_tokens bound.
                let per_engine_cap = state
                    .max_tokens
                    .map(|m| m.saturating_sub(state.guaranteed_min_tokens))
                    .unwrap_or(elastic_pool);
                natural_elastic
                    .min(elastic_permille_cap)
                    .min(per_engine_cap)
            } else {
                idle_elastic_pool += natural_elastic;
                0
            };

            budgets.push(EngineTokenBudget {
                engine_id,
                max_tokens: state.guaranteed_min_tokens + elastic_share,
            });
        }

        // ── Idle donation ─────────────────────────────────────────────────────
        if idle_elastic_pool > 0 {
            if let Some(b) = budgets.iter_mut().find(|b| b.max_tokens > 0) {
                let state = self.engines.get(&b.engine_id).unwrap();
                let elastic_used = b.max_tokens.saturating_sub(state.guaranteed_min_tokens);
                let per_engine_cap = state
                    .max_tokens
                    .map(|m| m.saturating_sub(state.guaranteed_min_tokens))
                    .unwrap_or(elastic_pool);
                let headroom = per_engine_cap.saturating_sub(elastic_used);
                b.max_tokens += idle_elastic_pool.min(headroom);
            }
        }

        // ── Rounding correction ───────────────────────────────────────────────
        let rounding = elastic_pool.saturating_sub(natural_elastic_sum);
        if rounding > 0 {
            if let Some(b) = budgets.iter_mut().find(|b| b.max_tokens > 0) {
                let state = self.engines.get(&b.engine_id).unwrap();
                let elastic_used = b.max_tokens.saturating_sub(state.guaranteed_min_tokens);
                let per_engine_cap = state
                    .max_tokens
                    .map(|m| m.saturating_sub(state.guaranteed_min_tokens))
                    .unwrap_or(elastic_pool);
                let headroom = per_engine_cap.saturating_sub(elastic_used);
                b.max_tokens += rounding.min(headroom);
            }
        }

        budgets
    }

    /// Return the budget ceiling for a single engine, or `None` if it is not
    /// registered.
    pub fn budget_for(&self, engine_id: u32) -> Option<usize> {
        self.allocate_budgets()
            .into_iter()
            .find(|b| b.engine_id == engine_id)
            .map(|b| b.max_tokens)
    }

    // ── Hard admission ────────────────────────────────────────────────────────

    /// Attempt to reserve `tokens` from `engine_id`'s current budget.
    ///
    /// Returns `true` and:
    /// - deducts `tokens` from the engine's reservation counter
    /// - increments the in-flight request count
    /// - marks the engine **active** if this is its first in-flight request
    ///   (so `allocate_budgets` treats it as active for the next round)
    ///
    /// Returns `false` without mutating state if the request would exceed the
    /// computed budget.  Callers **must** call [`complete_tokens`] when the
    /// request finishes so budget and activity tracking stay consistent.
    pub fn try_reserve_tokens(&mut self, engine_id: u32, tokens: usize) -> bool {
        let budget = match self.budget_for(engine_id) {
            Some(b) => b,
            None => return false,
        };
        let already = self.reserved_tokens.get(&engine_id).copied().unwrap_or(0);
        if already + tokens > budget {
            return false;
        }
        *self.reserved_tokens.entry(engine_id).or_insert(0) += tokens;

        // Track in-flight count and mark active on first request.
        let was_idle = {
            let inflight = self.inflight.entry(engine_id).or_insert(0);
            let was_idle = *inflight == 0;
            *inflight += 1;
            was_idle
        };
        if was_idle {
            self.set_active(engine_id, true);
        }

        true
    }

    /// Complete a request: return `tokens` to the reservation counter, record
    /// actual usage for future budget rounds, and mark the engine **idle** when
    /// no more requests are in-flight.
    ///
    /// `tokens_actual` should be the actual tokens consumed (or the reserved
    /// estimate if the actual count is unavailable).  Safe to call even if the
    /// values exceed current counters (saturating arithmetic).
    pub fn complete_tokens(&mut self, engine_id: u32, tokens_actual: usize) {
        // Return reservation.
        if let Some(r) = self.reserved_tokens.get_mut(&engine_id) {
            *r = r.saturating_sub(tokens_actual);
        }
        // Record usage for allocate_budgets rounds.
        self.report_usage(engine_id, tokens_actual);
        // Decrement in-flight; mark idle when queue drains.
        if let Some(c) = self.inflight.get_mut(&engine_id) {
            *c = c.saturating_sub(1);
            if *c == 0 {
                self.set_active(engine_id, false);
                self.restore_preemption_loans_for(engine_id);
            }
        }
    }

    /// Attach the runtime-owned live block cap for an engine. Registration is
    /// idempotent and intentionally separate from [`EngineHandle`] so existing
    /// embedders that do not use shared physical blocks remain source-compatible.
    pub fn register_block_cap(&mut self, engine_id: u32, cap: Arc<AtomicUsize>) {
        if self.engines.contains_key(&engine_id) {
            self.block_caps.insert(engine_id, cap);
        }
    }

    /// Total number of registered engines.
    pub fn engine_count(&self) -> usize {
        self.engines.len()
    }

    /// Determine which engine should donate blocks for a preemption request.
    ///
    /// Chooses the registered engine (other than the requesting one) whose
    /// running work has the lowest priority, as a heuristic for which engine
    /// is least harmed by a swap-out.  Returns the donor engine ID and how
    /// many blocks it could theoretically free if its scheduler evicts its
    /// lowest-priority groups.
    ///
    /// This lower-level selector does not enqueue work. Production callers use
    /// [`request_preemption`](Self::request_preemption), which retains an
    /// executable command for the selected donor engine.
    pub fn find_preemption_donor(
        &mut self,
        request: &PreemptionRequest,
        // Per-engine (engine_id → lowest running priority) reported by engines
        engine_priorities: &HashMap<u32, u8>,
        // Per-engine (engine_id → estimated free-able blocks)
        engine_freeable_blocks: &HashMap<u32, usize>,
    ) -> Option<u32> {
        // Among other engines, find the one with the lowest minimum priority
        // (most evictable) that has enough freeable blocks and hasn't hit
        // its per-round donation cap.
        let donor = self
            .engine_order
            .iter()
            .filter(|&&id| id != request.requesting_engine_id)
            .filter(|&&id| self.engines.contains_key(&id))
            .filter(|&&id| {
                // Never reclaim blocks from an unhealthy engine: a Dead one
                // cannot act on the request, and a Degraded one is already
                // being drained.
                self.engines
                    .get(&id)
                    .map(|s| s.health == EngineHealth::Healthy)
                    .unwrap_or(false)
            })
            .filter(|&&id| {
                self.preemption_donations_this_round
                    .get(&id)
                    .copied()
                    .unwrap_or(0)
                    < MAX_PREEMPTIONS_DONATED_PER_ROUND
            })
            .filter(|&&id| {
                engine_priorities
                    .get(&id)
                    .map(|&p| p < request.request_priority)
                    .unwrap_or(false)
            })
            .filter(|&&id| {
                engine_freeable_blocks
                    .get(&id)
                    .map(|&b| b >= request.blocks_needed)
                    .unwrap_or(false)
            })
            .min_by_key(|&&id| engine_priorities.get(&id).copied().unwrap_or(u8::MAX))
            .copied();

        if let Some(id) = donor {
            *self.preemption_donations_this_round.entry(id).or_insert(0) += 1;
        }
        donor
    }

    /// Publish the concrete reclaimable state observed by one engine's live
    /// scheduling loop. Stale or unregistered engines are ignored.
    pub fn report_preemption_state(
        &mut self,
        engine_id: u32,
        lowest_running_priority: Option<u8>,
        freeable_blocks: usize,
    ) {
        if self
            .engines
            .get(&engine_id)
            .is_none_or(|state| state.health != EngineHealth::Healthy)
        {
            self.preemption_states.remove(&engine_id);
            return;
        }
        self.preemption_states.insert(
            engine_id,
            EnginePreemptionState {
                lowest_running_priority,
                freeable_blocks,
            },
        );
    }

    /// Select a donor from live engine reports and enqueue an executable
    /// preemption command for its scheduling loop.
    ///
    /// Returning a donor now means more than an advisory choice: the command is
    /// retained until that donor calls [`take_preemption_requests`]. Duplicate
    /// pressure reports from the same requester are coalesced while the first
    /// command is in flight.
    pub fn request_preemption(&mut self, request: PreemptionRequest) -> Option<u32> {
        if let Some(donor) = self
            .pending_requesters
            .get(&request.requesting_engine_id)
            .copied()
        {
            return Some(donor);
        }

        let priorities = self
            .preemption_states
            .iter()
            .filter_map(|(&engine_id, state)| {
                state
                    .lowest_running_priority
                    .map(|priority| (engine_id, priority))
            })
            .collect::<HashMap<_, _>>();
        let freeable = self
            .preemption_states
            .iter()
            .map(|(&engine_id, state)| (engine_id, state.freeable_blocks))
            .collect::<HashMap<_, _>>();
        let donor = self.find_preemption_donor(&request, &priorities, &freeable)?;
        self.pending_preemptions
            .entry(donor)
            .or_default()
            .push_back(request.clone());
        self.pending_requesters
            .insert(request.requesting_engine_id, donor);
        Some(donor)
    }

    /// Drain the commands routed to `donor_engine_id`. The caller must report
    /// each result through [`finish_preemption`] after touching its scheduler.
    pub fn take_preemption_requests(&mut self, donor_engine_id: u32) -> Vec<PreemptionRequest> {
        self.pending_preemptions
            .remove(&donor_engine_id)
            .map(VecDeque::into_iter)
            .into_iter()
            .flatten()
            .collect()
    }

    /// Complete one routed command and release its de-duplication slot.
    pub fn finish_preemption(&mut self, result: PreemptionResult, requesting_engine_id: u32) {
        if self.pending_requesters.get(&requesting_engine_id).copied()
            == Some(result.donor_engine_id)
        {
            self.pending_requesters.remove(&requesting_engine_id);
        }
        // The borrower may have been cancelled after routing but before the
        // donor serviced its command. Only establish a loan while admitted work
        // is still alive; otherwise the donor's released blocks simply remain
        // available in the shared pool under its existing cap.
        let borrower_is_live = self
            .inflight
            .get(&requesting_engine_id)
            .copied()
            .unwrap_or(0)
            > 0
            && self
                .engines
                .get(&requesting_engine_id)
                .is_some_and(|state| state.health == EngineHealth::Healthy);
        if result.blocks_freed > 0 && borrower_is_live {
            if let (Some(donor), Some(borrower)) = (
                self.block_caps.get(&result.donor_engine_id),
                self.block_caps.get(&requesting_engine_id),
            ) {
                // Never create more borrower cap than was actually removed
                // from the donor. This matters when a health rebalance changes
                // the donor cap between selection and completion.
                let donor_cap = donor
                    .fetch_update(Ordering::AcqRel, Ordering::Acquire, |current| {
                        Some(current.saturating_sub(result.blocks_freed))
                    })
                    .unwrap_or_else(|current| current);
                let transferred = donor_cap.min(result.blocks_freed);
                if transferred > 0 {
                    borrower.fetch_add(transferred, Ordering::AcqRel);
                    self.preemption_loans.push(PreemptionLoan {
                        donor_engine_id: result.donor_engine_id,
                        borrower_engine_id: requesting_engine_id,
                        blocks: transferred,
                    });
                }
            }
        }
        if self.pending_requesters.is_empty() {
            self.reset_preemption_round();
        }
    }

    fn restore_preemption_loans_for(&mut self, borrower_engine_id: u32) {
        let loans: Vec<_> = self
            .preemption_loans
            .iter()
            .copied()
            .filter(|loan| loan.borrower_engine_id == borrower_engine_id)
            .collect();
        if loans.is_empty() {
            return;
        }
        for loan in &loans {
            if let Some(borrower) = self.block_caps.get(&borrower_engine_id) {
                borrower
                    .fetch_update(Ordering::AcqRel, Ordering::Acquire, |current| {
                        Some(current.saturating_sub(loan.blocks))
                    })
                    .ok();
            }
            if let Some(donor) = self.block_caps.get(&loan.donor_engine_id) {
                donor.fetch_add(loan.blocks, Ordering::AcqRel);
            }
        }
        self.preemption_loans
            .retain(|loan| loan.borrower_engine_id != borrower_engine_id);
    }

    /// Cancel queued work and settle every temporary cap transfer touching one
    /// engine. Used by unload and health isolation so no command or loan can
    /// outlive either endpoint.
    fn clear_preemption_state_for(&mut self, engine_id: u32) {
        self.restore_preemption_loans_for(engine_id);

        // If this engine was a donor, remove its outstanding loans from the
        // borrowers. The donor side is intentionally not restored: unload drops
        // that cap, while a health rebalance will replace it from policy.
        let orphaned = self
            .preemption_loans
            .iter()
            .copied()
            .filter(|loan| loan.donor_engine_id == engine_id)
            .collect::<Vec<_>>();
        for loan in &orphaned {
            if let Some(borrower) = self.block_caps.get(&loan.borrower_engine_id) {
                borrower
                    .fetch_update(Ordering::AcqRel, Ordering::Acquire, |current| {
                        Some(current.saturating_sub(loan.blocks))
                    })
                    .ok();
            }
        }
        self.preemption_loans
            .retain(|loan| loan.donor_engine_id != engine_id);

        self.preemption_states.remove(&engine_id);
        self.pending_preemptions.remove(&engine_id);
        for queue in self.pending_preemptions.values_mut() {
            queue.retain(|request| request.requesting_engine_id != engine_id);
        }
        self.pending_preemptions
            .retain(|_, queue| !queue.is_empty());
        self.pending_requesters
            .retain(|requester, donor| *requester != engine_id && *donor != engine_id);
    }

    /// Reset per-engine preemption donation counts for the new scheduling round.
    ///
    /// Call this once at the start of each round (typically alongside
    /// `set_active` / `report_usage`) so donation caps are enforced per-round
    /// rather than accumulating across the lifetime of the scheduler.
    pub fn reset_preemption_round(&mut self) {
        self.preemption_donations_this_round.clear();
    }

    /// Current global token budget.
    pub fn global_max_tokens(&self) -> usize {
        self.global_max_tokens
    }

    /// Update the global token budget.
    pub fn set_global_max_tokens(&mut self, tokens: usize) {
        self.global_max_tokens = tokens;
    }
}

#[cfg(test)]
mod global_scheduler_tests {
    use super::*;

    fn make_scheduler(total: usize, engines: &[(u32, u32, bool)]) -> GlobalKvScheduler {
        let mut sched = GlobalKvScheduler::new(total);
        for &(id, weight, active) in engines {
            sched.register(EngineHandle {
                engine_id: id,
                share_weight: weight,
                guaranteed_min_tokens: 0,
                max_tokens: None,
            });
            sched.set_active(id, active);
        }
        sched
    }

    #[test]
    fn equal_weights_split_evenly() {
        let sched = make_scheduler(1000, &[(0, 1, true), (1, 1, true)]);
        let budgets = sched.allocate_budgets();
        assert_eq!(budgets.len(), 2);
        let total: usize = budgets.iter().map(|b| b.max_tokens).sum();
        assert_eq!(total, 1000);
        // Each engine should get approximately half.
        for b in &budgets {
            assert!(b.max_tokens >= 490 && b.max_tokens <= 510, "{b:?}");
        }
    }

    #[test]
    fn weighted_distribution() {
        // Engine 0 weight 1, engine 1 weight 3 → 25% / 75%.
        let sched = make_scheduler(1000, &[(0, 1, true), (1, 3, true)]);
        let budgets = sched.allocate_budgets();
        let b0 = budgets
            .iter()
            .find(|b| b.engine_id == 0)
            .unwrap()
            .max_tokens;
        let b1 = budgets
            .iter()
            .find(|b| b.engine_id == 1)
            .unwrap()
            .max_tokens;
        // Allow ±1 for integer rounding.
        assert!((249..=251).contains(&b0), "b0={b0}");
        assert!((749..=751).contains(&b1), "b1={b1}");
        assert_eq!(b0 + b1, 1000);
    }

    #[test]
    fn idle_engine_gets_zero() {
        let sched = make_scheduler(1000, &[(0, 1, true), (1, 1, false)]);
        let budgets = sched.allocate_budgets();
        let b1 = budgets
            .iter()
            .find(|b| b.engine_id == 1)
            .unwrap()
            .max_tokens;
        assert_eq!(b1, 0, "idle engine should get zero budget");
    }

    #[test]
    fn active_engine_absorbs_idle_share() {
        let sched = make_scheduler(1000, &[(0, 1, true), (1, 1, false)]);
        let budgets = sched.allocate_budgets();
        let b0 = budgets
            .iter()
            .find(|b| b.engine_id == 0)
            .unwrap()
            .max_tokens;
        // Active engine should absorb the idle engine's share (all 1000).
        assert_eq!(b0, 1000);
    }

    #[test]
    fn single_engine_cap_respected() {
        let mut sched = GlobalKvScheduler::new(1000).with_max_single_engine_permille(500);
        sched.register(EngineHandle {
            engine_id: 0,
            share_weight: 1,
            guaranteed_min_tokens: 0,
            max_tokens: None,
        });
        sched.set_active(0, true);
        let budgets = sched.allocate_budgets();
        let b0 = budgets[0].max_tokens;
        // Cap is 50% of 1000 = 500.
        assert!(b0 <= 500, "b0={b0} exceeds cap");
    }

    #[test]
    fn deregister_removes_engine() {
        let mut sched = make_scheduler(1000, &[(0, 1, true), (1, 1, true)]);
        sched.deregister(1);
        assert_eq!(sched.engine_count(), 1);
        let budgets = sched.allocate_budgets();
        assert_eq!(budgets.len(), 1);
        assert_eq!(budgets[0].engine_id, 0);
    }

    #[test]
    fn find_preemption_donor_picks_lowest_priority() {
        let mut sched = make_scheduler(1000, &[(0, 1, true), (1, 1, true), (2, 1, true)]);
        let req = PreemptionRequest {
            requesting_engine_id: 0,
            blocks_needed: 10,
            request_priority: 5,
        };
        // Engine 1 runs priority 1, engine 2 runs priority 3; both have enough blocks.
        let priorities: HashMap<u32, u8> = [(1, 1), (2, 3)].into();
        let freeable: HashMap<u32, usize> = [(1, 20), (2, 20)].into();
        let donor = sched.find_preemption_donor(&req, &priorities, &freeable);
        // Engine 1 has lower priority (1 < 3) and both < request_priority (5), so engine 1 wins.
        assert_eq!(donor, Some(1));
    }

    #[test]
    fn no_donor_when_all_higher_priority() {
        let mut sched = make_scheduler(1000, &[(0, 1, true), (1, 1, true)]);
        let req = PreemptionRequest {
            requesting_engine_id: 0,
            blocks_needed: 10,
            request_priority: 1, // lower than what engines are running
        };
        let priorities: HashMap<u32, u8> = [(1, 10)].into(); // engine 1 runs priority 10 > 1
        let freeable: HashMap<u32, usize> = [(1, 20)].into();
        let donor = sched.find_preemption_donor(&req, &priorities, &freeable);
        // Engine 1's priority (10) is NOT < request_priority (1), so no donor.
        assert_eq!(donor, None);
    }

    #[test]
    fn unhealthy_engine_gets_zero_budget_and_share_redistributed() {
        let mut sched = make_scheduler(1000, &[(0, 1, true), (1, 1, true)]);
        sched.set_health(1, EngineHealth::Degraded);
        let budgets = sched.allocate_budgets();
        let b0 = budgets
            .iter()
            .find(|b| b.engine_id == 0)
            .unwrap()
            .max_tokens;
        let b1 = budgets
            .iter()
            .find(|b| b.engine_id == 1)
            .unwrap()
            .max_tokens;
        // Degraded engine gets nothing; healthy engine absorbs its share.
        assert_eq!(b1, 0, "degraded engine should get zero budget");
        assert_eq!(b0, 1000, "healthy engine should absorb the degraded share");
    }

    #[test]
    fn dead_engine_excluded_from_preemption_donor() {
        let mut sched = make_scheduler(1000, &[(0, 1, true), (1, 1, true), (2, 1, true)]);
        sched.set_health(1, EngineHealth::Dead);
        let req = PreemptionRequest {
            requesting_engine_id: 0,
            blocks_needed: 10,
            request_priority: 5,
        };
        // Engine 1 (Dead) has the lowest priority but must be skipped; engine 2 wins.
        let priorities: HashMap<u32, u8> = [(1, 1), (2, 3)].into();
        let freeable: HashMap<u32, usize> = [(1, 20), (2, 20)].into();
        let donor = sched.find_preemption_donor(&req, &priorities, &freeable);
        assert_eq!(donor, Some(2), "dead engine must not be chosen as donor");
    }

    #[test]
    fn reregister_resets_health_to_healthy() {
        let mut sched = make_scheduler(1000, &[(0, 1, true)]);
        sched.set_health(0, EngineHealth::Dead);
        assert_eq!(sched.health_of(0), Some(EngineHealth::Dead));
        sched.register(EngineHandle {
            engine_id: 0,
            share_weight: 1,
            guaranteed_min_tokens: 0,
            max_tokens: None,
        });
        assert_eq!(sched.health_of(0), Some(EngineHealth::Healthy));
    }

    #[test]
    fn health_epoch_bumps_only_on_transition() {
        let mut sched = make_scheduler(1000, &[(0, 1, true)]);
        let e0 = sched.health_epoch();

        // First transition bumps.
        sched.set_health(0, EngineHealth::Degraded);
        let e1 = sched.health_epoch();
        assert_eq!(e1, e0 + 1);

        // Same value again does not bump.
        sched.set_health(0, EngineHealth::Degraded);
        assert_eq!(sched.health_epoch(), e1);

        // A different value bumps.
        sched.set_health(0, EngineHealth::Healthy);
        assert_eq!(sched.health_epoch(), e1 + 1);

        // Unknown engine never bumps.
        sched.set_health(999, EngineHealth::Dead);
        assert_eq!(sched.health_epoch(), e1 + 1);
    }

    #[test]
    fn preemption_cap_limits_donations_per_round() {
        let mut sched = make_scheduler(1000, &[(0, 1, true), (1, 1, true), (2, 1, true)]);
        let req = PreemptionRequest {
            requesting_engine_id: 0,
            blocks_needed: 5,
            request_priority: 10,
        };
        let priorities: HashMap<u32, u8> = [(1, 1), (2, 1)].into();
        let freeable: HashMap<u32, usize> = [(1, 20), (2, 20)].into();

        // Engine 1 should be selected up to MAX_PREEMPTIONS_DONATED_PER_ROUND times.
        for _ in 0..MAX_PREEMPTIONS_DONATED_PER_ROUND {
            let donor = sched.find_preemption_donor(&req, &priorities, &freeable);
            assert!(donor.is_some());
        }
        // After the cap is reached for engine 1, it must not be selected again this round.
        // Engine 2 may be selected instead (also at priority 1), but engine 1 is blocked.
        let donations_1 = sched
            .preemption_donations_this_round
            .get(&1)
            .copied()
            .unwrap_or(0);
        assert_eq!(donations_1, MAX_PREEMPTIONS_DONATED_PER_ROUND);

        // Reset clears the cap for the next round.
        sched.reset_preemption_round();
        assert!(sched.preemption_donations_this_round.is_empty());
    }

    #[test]
    fn routed_preemption_is_executable_deduplicated_and_cap_scoped() {
        let mut sched = make_scheduler(1000, &[(0, 1, true), (1, 1, true)]);
        let requester_cap = Arc::new(AtomicUsize::new(10));
        let donor_cap = Arc::new(AtomicUsize::new(10));
        sched.register_block_cap(0, requester_cap.clone());
        sched.register_block_cap(1, donor_cap.clone());
        assert!(sched.try_reserve_tokens(0, 1));

        sched.report_preemption_state(1, Some(1), 8);
        let request = PreemptionRequest {
            requesting_engine_id: 0,
            blocks_needed: 6,
            request_priority: 9,
        };
        assert_eq!(sched.request_preemption(request.clone()), Some(1));
        // A fast requester loop cannot enqueue the same command repeatedly.
        assert_eq!(sched.request_preemption(request.clone()), Some(1));

        let commands = sched.take_preemption_requests(1);
        assert_eq!(commands.len(), 1);
        assert_eq!(commands[0].blocks_needed, 6);
        assert!(sched.take_preemption_requests(1).is_empty());

        sched.finish_preemption(
            PreemptionResult {
                donor_engine_id: 1,
                blocks_freed: 6,
            },
            0,
        );
        assert_eq!(requester_cap.load(Ordering::Acquire), 16);
        assert_eq!(donor_cap.load(Ordering::Acquire), 4);

        // The loan is returned when the borrower's last admitted request ends.
        sched.complete_tokens(0, 1);
        assert_eq!(requester_cap.load(Ordering::Acquire), 10);
        assert_eq!(donor_cap.load(Ordering::Acquire), 10);
    }

    #[test]
    fn routed_preemption_never_targets_equal_priority_work() {
        let mut sched = make_scheduler(1000, &[(0, 1, true), (1, 1, true)]);
        sched.report_preemption_state(1, Some(5), 20);
        assert_eq!(
            sched.request_preemption(PreemptionRequest {
                requesting_engine_id: 0,
                blocks_needed: 4,
                request_priority: 5,
            }),
            None,
        );
        assert!(sched.take_preemption_requests(1).is_empty());
    }

    #[test]
    fn cancelled_borrower_cannot_leave_a_cap_loan() {
        let mut sched = make_scheduler(1000, &[(0, 1, true), (1, 1, true)]);
        let requester_cap = Arc::new(AtomicUsize::new(10));
        let donor_cap = Arc::new(AtomicUsize::new(10));
        sched.register_block_cap(0, requester_cap.clone());
        sched.register_block_cap(1, donor_cap.clone());
        assert!(sched.try_reserve_tokens(0, 1));
        sched.report_preemption_state(1, Some(1), 8);
        assert_eq!(
            sched.request_preemption(PreemptionRequest {
                requesting_engine_id: 0,
                blocks_needed: 6,
                request_priority: 9,
            }),
            Some(1),
        );
        assert_eq!(sched.take_preemption_requests(1).len(), 1);

        // Cancellation releases the admission guard before the donor reports.
        sched.complete_tokens(0, 1);
        sched.finish_preemption(
            PreemptionResult {
                donor_engine_id: 1,
                blocks_freed: 6,
            },
            0,
        );

        assert_eq!(requester_cap.load(Ordering::Acquire), 10);
        assert_eq!(donor_cap.load(Ordering::Acquire), 10);
        assert!(sched.preemption_loans.is_empty());
    }

    #[test]
    fn cap_loan_is_clamped_to_the_donors_live_cap() {
        let mut sched = make_scheduler(1000, &[(0, 1, true), (1, 1, true)]);
        let requester_cap = Arc::new(AtomicUsize::new(10));
        let donor_cap = Arc::new(AtomicUsize::new(3));
        sched.register_block_cap(0, requester_cap.clone());
        sched.register_block_cap(1, donor_cap.clone());
        assert!(sched.try_reserve_tokens(0, 1));

        sched.finish_preemption(
            PreemptionResult {
                donor_engine_id: 1,
                blocks_freed: 6,
            },
            0,
        );
        assert_eq!(requester_cap.load(Ordering::Acquire), 13);
        assert_eq!(donor_cap.load(Ordering::Acquire), 0);

        sched.complete_tokens(0, 1);
        assert_eq!(requester_cap.load(Ordering::Acquire), 10);
        assert_eq!(donor_cap.load(Ordering::Acquire), 3);
    }

    #[test]
    fn deregistered_requester_is_removed_from_donor_queue() {
        let mut sched = make_scheduler(1000, &[(0, 1, true), (1, 1, true)]);
        sched.report_preemption_state(1, Some(1), 8);
        assert_eq!(
            sched.request_preemption(PreemptionRequest {
                requesting_engine_id: 0,
                blocks_needed: 6,
                request_priority: 9,
            }),
            Some(1),
        );
        sched.deregister(0);
        assert!(sched.take_preemption_requests(1).is_empty());
    }
}
