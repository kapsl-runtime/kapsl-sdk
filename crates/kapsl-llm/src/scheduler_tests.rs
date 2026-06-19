#[cfg(test)]
mod tests {
    use super::super::{LLMScheduler, SchedulerConfig};
    use crate::block_manager::BlockManager;
    use crate::sequence::{FinishReason, SamplingParams, SequenceGroup, SequenceStatus};
    use tokio::sync::mpsc;

    fn make_group(prompt_len: usize) -> SequenceGroup {
        let (tx, _rx) = mpsc::channel(1);
        SequenceGroup::new(
            "req".to_string(),
            None,
            "prompt".to_string(),
            vec![0u32; prompt_len],
            SamplingParams {
                max_tokens: 8,
                min_tokens: 0,
                temperature: 0.7,
                top_p: 0.9,
                top_k: 40,
                stop_token_ids: Vec::new(),
                repetition_penalty: 1.0,
                seed: None,
            },
            None,
            tx,
        )
    }

    #[test]
    fn schedule_moves_waiting_to_running_and_allocates_blocks() {
        let config = SchedulerConfig {
            max_num_batched_tokens: 64,
            max_num_seqs: 4,
            max_paddings: 0,
        };
        let block_manager = BlockManager::new(4, 16, 0);
        let mut scheduler = LLMScheduler::new(config, block_manager);

        scheduler.add_sequence_group(make_group(4));
        let outputs = scheduler.schedule();
        assert_eq!(outputs.scheduled_seq_groups.len(), 1);

        let group = outputs.scheduled_seq_groups[0].lock().unwrap();
        let seq_arc = group.sequences.values().next().unwrap();
        let seq = seq_arc.lock().unwrap();
        assert_eq!(seq.status, SequenceStatus::Running);
        assert!(scheduler
            .block_manager
            .get_block_table(seq.sequence_id)
            .is_some());
    }

    #[test]
    fn free_finished_sequences_releases_blocks() {
        let config = SchedulerConfig {
            max_num_batched_tokens: 64,
            max_num_seqs: 4,
            max_paddings: 0,
        };
        let block_manager = BlockManager::new(1, 16, 0);
        let mut scheduler = LLMScheduler::new(config, block_manager);

        scheduler.add_sequence_group(make_group(1));
        let _ = scheduler.schedule();

        let group_arc = scheduler
            .running_queue
            .front()
            .expect("running group")
            .clone();
        let seq_arc = group_arc
            .lock()
            .unwrap()
            .sequences
            .values()
            .next()
            .unwrap()
            .clone();
        let (old_status, new_status) = {
            let mut seq = seq_arc.lock().unwrap();
            let old_status = seq.status;
            seq.status = SequenceStatus::Finished(FinishReason::Stop);
            (old_status, seq.status)
        };
        {
            let mut group = group_arc.lock().unwrap();
            group.update_seq_status(old_status, new_status);
        }

        let finished = scheduler.free_finished_sequences();
        assert_eq!(finished.len(), 1);
        assert!(scheduler.block_manager.can_allocate(1));
    }

    #[test]
    fn waiting_queue_orders_by_priority_then_fifo() {
        let config = SchedulerConfig {
            max_num_batched_tokens: 64,
            max_num_seqs: 4,
            max_paddings: 0,
        };
        let block_manager = BlockManager::new(4, 16, 0);
        let mut scheduler = LLMScheduler::new(config, block_manager);

        let mut g_low = make_group(1);
        g_low.priority = 0;
        g_low.request_id = "low".to_string();
        let mut g_mid_a = make_group(1);
        g_mid_a.priority = 5;
        g_mid_a.request_id = "mid_a".to_string();
        let mut g_high = make_group(1);
        g_high.priority = 9;
        g_high.request_id = "high".to_string();
        let mut g_mid_b = make_group(1);
        g_mid_b.priority = 5;
        g_mid_b.request_id = "mid_b".to_string();

        // Arrival order: low, mid_a, high, mid_b.
        scheduler.add_sequence_group(g_low);
        scheduler.add_sequence_group(g_mid_a);
        scheduler.add_sequence_group(g_high);
        scheduler.add_sequence_group(g_mid_b);

        let order: Vec<String> = scheduler
            .waiting_queue
            .iter()
            .map(|g| g.lock().unwrap().request_id.clone())
            .collect();
        // Descending priority, FIFO within a tier: mid_a before mid_b.
        assert_eq!(order, vec!["high", "mid_a", "mid_b", "low"]);
    }
}
