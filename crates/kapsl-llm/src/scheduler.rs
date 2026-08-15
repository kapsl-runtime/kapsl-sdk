use crate::block_manager::BlockManager;
use crate::sequence::{FinishReason, Sequence, SequenceGroup, SequenceGroupOutput, SequenceStatus};
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

#[derive(Clone)]
pub struct SchedulerConfig {
    pub max_num_batched_tokens: usize,
    pub max_num_seqs: usize,
    pub max_paddings: usize,
}

pub struct SchedulerOutputs {
    pub scheduled_seq_groups: Vec<Arc<Mutex<SequenceGroup>>>,
    /// Sequence KV stores invalidated by local priority preemption this round.
    /// The engine owns the concrete host/device KV cache and must release these
    /// entries before executing the newly admitted work.
    pub(crate) preempted_sequence_ids: Vec<u64>,
    /// Cross-engine pressure that could not be satisfied locally. The engine
    /// routes this through `GlobalKvScheduler` after publishing its own donor
    /// state, and retries the waiting request on a later loop iteration.
    pub(crate) preemption_request: Option<SchedulerPreemptionRequest>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct SchedulerPreemptionRequest {
    pub(crate) blocks_needed: usize,
    /// Internal sequence priority: larger values are more important.
    pub(crate) request_priority: u8,
}

#[derive(Debug, Default, PartialEq, Eq)]
pub(crate) struct SchedulerPreemptionOutcome {
    pub(crate) blocks_freed: usize,
    pub(crate) sequence_ids: Vec<u64>,
}

pub struct LLMScheduler {
    config: SchedulerConfig,
    waiting_queue: VecDeque<Arc<Mutex<SequenceGroup>>>,
    running_queue: VecDeque<Arc<Mutex<SequenceGroup>>>,
    swapped_queue: VecDeque<Arc<Mutex<SequenceGroup>>>,
    block_manager: BlockManager,
}

impl LLMScheduler {
    pub fn new(config: SchedulerConfig, block_manager: BlockManager) -> Self {
        Self {
            config,
            waiting_queue: VecDeque::new(),
            running_queue: VecDeque::new(),
            swapped_queue: VecDeque::new(),
            block_manager,
        }
    }

    /// Expose scheduler config fields for engine rebuild in `with_shared_pool`.
    pub fn config_max_num_seqs(&self) -> usize {
        self.config.max_num_seqs
    }

    pub fn config_max_paddings(&self) -> usize {
        self.config.max_paddings
    }

    pub fn config_max_num_batched_tokens(&self) -> usize {
        self.config.max_num_batched_tokens
    }

    /// Attach a hard per-engine KV block quota to the underlying block manager.
    pub fn set_block_live_cap(&mut self, cap: std::sync::Arc<std::sync::atomic::AtomicUsize>) {
        self.block_manager.set_live_cap(cap);
    }

    /// Set the engine id used to stamp KV block ownership in the block manager.
    pub fn set_block_engine_id(&mut self, engine_id: u32) {
        self.block_manager.set_engine_id(engine_id);
    }

    pub fn add_sequence_group(&mut self, seq_group: SequenceGroup) {
        let priority = seq_group.priority;
        let group = Arc::new(Mutex::new(seq_group));
        // Keep the waiting queue ordered by descending priority with FIFO order
        // within a priority tier, so a high-priority request is not stuck behind
        // already-queued lower-priority ones (head-of-line blocking).
        let pos = self
            .waiting_queue
            .iter()
            .position(|g| g.lock().map(|g| g.priority < priority).unwrap_or(false))
            .unwrap_or(self.waiting_queue.len());
        self.waiting_queue.insert(pos, group);
    }

    pub fn active_sequence_ids(&self) -> Vec<u64> {
        let mut ids = Vec::new();
        for group_arc in self
            .waiting_queue
            .iter()
            .chain(self.running_queue.iter())
            .chain(self.swapped_queue.iter())
        {
            if let Ok(group) = group_arc.lock() {
                ids.extend(group.sequences.keys().copied());
            }
        }
        ids
    }

    pub fn schedule(&mut self) -> SchedulerOutputs {
        // Calculate current running tokens (decode): 1 token per running sequence.
        let mut num_batched_tokens = 0usize;
        let mut preempted_sequence_ids = Vec::new();
        let mut preemption_request = None;

        for group in &self.running_queue {
            let group = group.lock().unwrap();
            num_batched_tokens += group.cached_running_count();
        }

        let block_size = self.block_manager.block_size();

        // 1. Swap in swapped-out groups (best-effort FIFO).
        while let Some(group_arc) = self.swapped_queue.front().cloned() {
            let cancelled = {
                let group = group_arc.lock().unwrap();
                group
                    .cancellation
                    .as_ref()
                    .is_some_and(|t| t.is_cancelled())
            };
            if cancelled {
                Self::cancel_group(&group_arc);
                let _ = self.swapped_queue.pop_front();
                continue;
            }

            let seqs: Vec<(u64, Arc<Mutex<Sequence>>)> = {
                let group = group_arc.lock().unwrap();
                if group.is_finished() {
                    // Nothing to do; drop it.
                    drop(group);
                    let _ = self.swapped_queue.pop_front();
                    continue;
                }

                group
                    .sequences
                    .iter()
                    .map(|(seq_id, seq_arc)| (*seq_id, seq_arc.clone()))
                    .collect()
            };

            let mut requirements: Vec<(u64, Arc<Mutex<Sequence>>, usize)> =
                Vec::with_capacity(seqs.len());
            let mut total_additional_blocks = 0usize;

            for (seq_id, seq_arc) in &seqs {
                let (is_finished, seq_len) = {
                    let seq = seq_arc.lock().unwrap();
                    (seq.is_finished(), seq.get_len())
                };
                if is_finished {
                    continue;
                }

                let required_blocks = seq_len.div_ceil(block_size);
                let existing_blocks = self
                    .block_manager
                    .get_block_table(*seq_id)
                    .map(|table| table.len())
                    .unwrap_or(0);
                let additional_blocks = required_blocks.saturating_sub(existing_blocks);
                total_additional_blocks += additional_blocks;
                requirements.push((*seq_id, seq_arc.clone(), additional_blocks));
            }

            let group_decode_tokens = requirements.len();
            if group_decode_tokens == 0 {
                let _ = self.swapped_queue.pop_front();
                continue;
            }

            if num_batched_tokens + group_decode_tokens > self.config.max_num_batched_tokens {
                break;
            }

            if !self.block_manager.can_allocate(total_additional_blocks) {
                break;
            }

            let mut status_updates: Vec<(SequenceStatus, SequenceStatus)> =
                Vec::with_capacity(requirements.len());
            for (seq_id, seq_arc, additional_blocks) in &requirements {
                for _ in 0..*additional_blocks {
                    let _ = self.block_manager.allocate(*seq_id);
                }

                let (old_status, new_status) = {
                    let mut seq = seq_arc.lock().unwrap();
                    let old_status = seq.status;
                    if !seq.is_finished() && seq.status != SequenceStatus::Running {
                        seq.status = SequenceStatus::Running;
                    }
                    (old_status, seq.status)
                };
                if old_status != new_status {
                    status_updates.push((old_status, new_status));
                }
            }
            if !status_updates.is_empty() {
                let mut group = group_arc.lock().unwrap();
                for (old_status, new_status) in status_updates {
                    group.update_seq_status(old_status, new_status);
                }
            }

            let group_arc = self.swapped_queue.pop_front().unwrap();
            self.running_queue.push_back(group_arc);
            num_batched_tokens += group_decode_tokens;
        }

        // 2. Schedule waiting groups (Prefill).
        // Try to add new groups from waiting queue
        while let Some(group_arc) = self.waiting_queue.front().cloned() {
            let cancelled = {
                let group = group_arc.lock().unwrap();
                group
                    .cancellation
                    .as_ref()
                    .is_some_and(|t| t.is_cancelled())
            };
            if cancelled {
                Self::cancel_group(&group_arc);
                let _ = self.waiting_queue.pop_front();
                continue;
            }

            let (seqs, request_priority) = {
                let group = group_arc.lock().unwrap();
                if group.is_finished() {
                    // Unexpected in the waiting queue, but safe to drop.
                    drop(group);
                    let _ = self.waiting_queue.pop_front();
                    continue;
                }

                (
                    group
                        .sequences
                        .iter()
                        .map(|(seq_id, seq_arc)| (*seq_id, seq_arc.clone()))
                        .collect::<Vec<_>>(),
                    group.priority,
                )
            };

            let mut num_pending_tokens = 0usize;
            let mut requirements: Vec<(u64, Arc<Mutex<Sequence>>, usize)> =
                Vec::with_capacity(seqs.len());
            let mut total_additional_blocks = 0usize;

            for (seq_id, seq_arc) in &seqs {
                let (is_finished, seq_len, kv_cached_len) = {
                    let seq = seq_arc.lock().unwrap();
                    (seq.is_finished(), seq.get_len(), seq.kv_cached_len)
                };
                if is_finished {
                    continue;
                }

                num_pending_tokens += seq_len.saturating_sub(kv_cached_len);

                let required_blocks = seq_len.div_ceil(block_size);
                let existing_blocks = self
                    .block_manager
                    .get_block_table(*seq_id)
                    .map(|table| table.len())
                    .unwrap_or(0);
                let additional_blocks = required_blocks.saturating_sub(existing_blocks);
                total_additional_blocks += additional_blocks;
                requirements.push((*seq_id, seq_arc.clone(), additional_blocks));
            }

            if num_pending_tokens == 0 {
                // Nothing to prefill; move to running to let decode proceed.
                let group_arc = self.waiting_queue.pop_front().unwrap();
                self.running_queue.push_back(group_arc);
                continue;
            }

            if num_batched_tokens + num_pending_tokens > self.config.max_num_batched_tokens {
                break; // Token limit reached
            }

            if !self.block_manager.can_allocate(total_additional_blocks) {
                // Before stalling, attempt to reclaim blocks from lower-priority
                // running groups. Only proceed if preemption succeeds and we can
                // now satisfy the request.
                let shortage =
                    total_additional_blocks.saturating_sub(self.block_manager.allocatable_blocks());
                let outcome = self.preempt_lower_priority(shortage, request_priority);
                preempted_sequence_ids.extend(outcome.sequence_ids);
                if !self.block_manager.can_allocate(total_additional_blocks) {
                    preemption_request = Some(SchedulerPreemptionRequest {
                        blocks_needed: total_additional_blocks
                            .saturating_sub(self.block_manager.allocatable_blocks()),
                        request_priority,
                    });
                    break; // Not enough memory even after preemption
                }
            }

            let mut status_updates: Vec<(SequenceStatus, SequenceStatus)> =
                Vec::with_capacity(requirements.len());
            for (seq_id, seq_arc, additional_blocks) in &requirements {
                for _ in 0..*additional_blocks {
                    let _ = self.block_manager.allocate(*seq_id);
                }

                let (old_status, new_status) = {
                    let mut seq = seq_arc.lock().unwrap();
                    let old_status = seq.status;
                    if !seq.is_finished() && seq.status != SequenceStatus::Running {
                        seq.status = SequenceStatus::Running;
                    }
                    (old_status, seq.status)
                };
                if old_status != new_status {
                    status_updates.push((old_status, new_status));
                }
            }
            if !status_updates.is_empty() {
                let mut group = group_arc.lock().unwrap();
                for (old_status, new_status) in status_updates {
                    group.update_seq_status(old_status, new_status);
                }
            }

            let group_arc = self.waiting_queue.pop_front().unwrap();
            self.running_queue.push_back(group_arc);
            num_batched_tokens += num_pending_tokens;
        }

        let scheduled_seq_groups = self.running_queue.iter().cloned().collect();

        SchedulerOutputs {
            scheduled_seq_groups,
            preempted_sequence_ids,
            preemption_request,
        }
    }

    /// Attempt to free at least `needed_blocks` by swapping out running groups
    /// in ascending priority order (lowest-priority first, then FIFO within a tier).
    ///
    /// Only work strictly less important than `request_priority` is eligible.
    /// Blocks are returned to the pool immediately and each sequence's KV
    /// cursor is reset so the engine can release its concrete KV store and
    /// recompute it when the group is swapped back in.
    pub(crate) fn preempt_lower_priority(
        &mut self,
        needed_blocks: usize,
        request_priority: u8,
    ) -> SchedulerPreemptionOutcome {
        if needed_blocks == 0 {
            return SchedulerPreemptionOutcome::default();
        }

        // Build (priority, queue_index) pairs for non-finished running groups.
        let mut candidates: Vec<(u8, usize)> = self
            .running_queue
            .iter()
            .enumerate()
            .filter_map(|(idx, group_arc)| {
                let group = group_arc.lock().unwrap();
                if group.is_finished() || group.priority >= request_priority {
                    None
                } else {
                    Some((group.priority, idx))
                }
            })
            .collect();

        // Lowest priority first; break ties by queue position (FIFO within a tier).
        candidates.sort_unstable_by_key(|&(priority, idx)| (priority, idx));

        let mut freed = 0usize;
        let mut to_swap: Vec<usize> = Vec::new();

        for (_, idx) in &candidates {
            if freed >= needed_blocks {
                break;
            }
            let group_arc = &self.running_queue[*idx];
            let seq_ids: Vec<u64> = group_arc
                .lock()
                .unwrap()
                .sequences
                .keys()
                .copied()
                .collect();
            for seq_id in &seq_ids {
                freed += self.block_manager.blocks_for_sequence(*seq_id);
            }
            to_swap.push(*idx);
        }

        if freed < needed_blocks {
            return SchedulerPreemptionOutcome::default();
        }

        // Remove selected entries from running_queue in reverse-index order so
        // earlier indices stay valid during removal.
        to_swap.sort_unstable_by(|a, b| b.cmp(a));
        let mut sequence_ids = Vec::new();
        for idx in to_swap {
            let group_arc = self.running_queue.remove(idx).unwrap();

            let seq_entries: Vec<(u64, Arc<Mutex<Sequence>>)> = {
                let group = group_arc.lock().unwrap();
                group
                    .sequences
                    .iter()
                    .map(|(id, arc)| (*id, arc.clone()))
                    .collect()
            };

            let mut status_updates: Vec<(SequenceStatus, SequenceStatus)> = Vec::new();
            for (seq_id, seq_arc) in &seq_entries {
                let (old_status, new_status) = {
                    let mut seq = seq_arc.lock().unwrap();
                    let old = seq.status;
                    if !seq.is_finished() {
                        seq.status = SequenceStatus::Swapped;
                        // The concrete KV allocation is released by the engine
                        // using the returned sequence id. Re-admission must
                        // therefore rebuild the sequence from token zero.
                        seq.kv_cached_len = 0;
                    }
                    (old, seq.status)
                };
                if old_status != new_status {
                    status_updates.push((old_status, new_status));
                }
                self.block_manager.free(*seq_id);
                sequence_ids.push(*seq_id);
            }

            if !status_updates.is_empty() {
                let mut group = group_arc.lock().unwrap();
                for (old, new) in status_updates {
                    group.update_seq_status(old, new);
                }
            }

            self.swapped_queue.push_back(group_arc);
        }

        SchedulerPreemptionOutcome {
            blocks_freed: freed,
            sequence_ids,
        }
    }

    /// Conservative donor advertisement for cross-engine preemption.
    ///
    /// We report only blocks owned by the least-important running tier. This
    /// can understate what a very high-priority requester could reclaim across
    /// several tiers, but never promises blocks that the donor would then have
    /// to take from equal- or higher-priority work.
    pub(crate) fn preemption_state(&self) -> (Option<u8>, usize) {
        let mut lowest_priority = None;
        let mut freeable_blocks = 0usize;
        for group_arc in &self.running_queue {
            let group = group_arc.lock().unwrap();
            if group.is_finished() {
                continue;
            }
            let group_blocks = group
                .sequences
                .keys()
                .map(|sequence_id| self.block_manager.blocks_for_sequence(*sequence_id))
                .fold(0usize, usize::saturating_add);
            match lowest_priority {
                None => {
                    lowest_priority = Some(group.priority);
                    freeable_blocks = group_blocks;
                }
                Some(priority) if group.priority < priority => {
                    lowest_priority = Some(group.priority);
                    freeable_blocks = group_blocks;
                }
                Some(priority) if group.priority == priority => {
                    freeable_blocks = freeable_blocks.saturating_add(group_blocks);
                }
                Some(_) => {}
            }
        }
        (lowest_priority, freeable_blocks)
    }

    fn cancel_group(group_arc: &Arc<Mutex<SequenceGroup>>) {
        let (request_id, response_tx, seqs, already_finished) = {
            let group = group_arc.lock().unwrap();
            (
                group.request_id.clone(),
                group.response_tx.clone(),
                group.sequences.values().cloned().collect::<Vec<_>>(),
                group.is_finished(),
            )
        };
        if already_finished {
            return;
        }

        let mut status_updates: Vec<(SequenceStatus, SequenceStatus)> = Vec::new();
        for seq_arc in &seqs {
            let (old_status, new_status) = {
                let mut seq = seq_arc.lock().unwrap();
                let old_status = seq.status;
                if !seq.is_finished() {
                    seq.status = SequenceStatus::Finished(FinishReason::Cancelled);
                }
                (old_status, seq.status)
            };
            if old_status != new_status {
                status_updates.push((old_status, new_status));
            }
        }

        if !status_updates.is_empty() {
            let mut group = group_arc.lock().unwrap();
            for (old_status, new_status) in status_updates {
                group.update_seq_status(old_status, new_status);
            }
        }

        let _ = response_tx.try_send(SequenceGroupOutput {
            request_id,
            text: String::new(),
            finish_reason: Some(FinishReason::Cancelled),
        });
    }

    pub fn free_finished_sequences(&mut self) -> Vec<u64> {
        let mut finished_ids = Vec::new();
        self.running_queue.retain(|group_arc| {
            let group = group_arc.lock().unwrap();
            if group.is_finished() {
                if group.session_id.is_none() {
                    // Free blocks for non-session sequences.
                    for seq_id in group.sequences.keys() {
                        self.block_manager.free(*seq_id);
                        finished_ids.push(*seq_id);
                    }
                }
                false
            } else {
                true
            }
        });
        finished_ids
    }
}

#[path = "scheduler_tests.rs"]
mod scheduler_tests;
