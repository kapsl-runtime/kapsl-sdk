use kapsl_engine_api::CancellationToken;
use std::collections::HashMap;
use std::sync::Arc;
use tokenizers::Tokenizer;
use tokio::sync::mpsc;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SequenceStatus {
    Waiting,
    Running,
    Swapped,
    Finished(FinishReason),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FinishReason {
    Stop,
    Length,
    Cancelled,
    Error,
}

#[derive(Debug, Clone)]
pub struct Sequence {
    pub sequence_id: u64,
    pub prompt: String,
    pub prompt_token_ids: Vec<u32>,
    pub output_token_ids: Vec<u32>,
    pub status: SequenceStatus,
    pub cumulative_logprob: f32,
    pub generated_this_turn: usize,
    pub kv_cached_len: usize,
    pub rng_state: u64,
    /// Text emitted during the current generation turn.
    pub decode_stream_prefix: String,
    pub(crate) decode_stream: IncrementalDecodeState,
}

/// Owned state for Hugging Face tokenizers' incremental decoder.
///
/// `tokenizers::DecodeStream` borrows its tokenizer, which prevents storing it
/// alongside a long-lived sequence. This owns the corrected bounded state used
/// by newer tokenizers releases, so each decode only revisits the small unstable
/// suffix instead of every token generated so far.
#[derive(Debug, Clone, Default)]
pub(crate) struct IncrementalDecodeState {
    ids: Vec<u32>,
    prefix: String,
    prefix_index: usize,
    cumulative_fallback: bool,
}

impl IncrementalDecodeState {
    pub(crate) fn step(
        &mut self,
        tokenizer: &Tokenizer,
        token_id: u32,
    ) -> tokenizers::Result<Option<String>> {
        self.ids.push(token_id);
        let decoded = tokenizer.decode(&self.ids, true)?;
        let delta = if decoded.len() > self.prefix.len() && !decoded.ends_with('\u{fffd}') {
            if !decoded.starts_with(&self.prefix) {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "incremental tokenizer decode encountered an invalid prefix",
                )
                .into());
            }

            let delta = decoded[self.prefix.len()..].to_string();
            let new_prefix_index =
                self.ids
                    .len()
                    .checked_sub(self.prefix_index)
                    .ok_or_else(|| {
                        std::io::Error::new(
                            std::io::ErrorKind::InvalidData,
                            "incremental tokenizer decode index became invalid",
                        )
                    })?;
            self.ids = self.ids.drain(self.prefix_index..).collect();
            self.prefix = tokenizer.decode(&self.ids, true)?;
            self.prefix_index = new_prefix_index;
            Some(delta)
        } else {
            None
        };

        Ok(delta)
    }

    pub(crate) fn enter_cumulative_fallback(&mut self) {
        self.cumulative_fallback = true;
    }

    pub(crate) fn uses_cumulative_fallback(&self) -> bool {
        self.cumulative_fallback
    }

    pub(crate) fn reset(&mut self) {
        *self = Self::default();
    }

    #[cfg(test)]
    pub(crate) fn buffered_token_count(&self) -> usize {
        self.ids.len()
    }
}

impl Sequence {
    pub fn new(sequence_id: u64, prompt: String, prompt_token_ids: Vec<u32>) -> Self {
        Self {
            sequence_id,
            prompt,
            prompt_token_ids,
            output_token_ids: Vec::new(),
            status: SequenceStatus::Waiting,
            cumulative_logprob: 0.0,
            generated_this_turn: 0,
            kv_cached_len: 0,
            rng_state: 0x4d595df4d0f33173,
            decode_stream_prefix: String::new(),
            decode_stream: IncrementalDecodeState::default(),
        }
    }

    pub fn append_token_id(&mut self, token_id: u32, logprob: f32) {
        self.output_token_ids.push(token_id);
        self.cumulative_logprob += logprob;
        self.generated_this_turn += 1;
    }

    pub fn get_len(&self) -> usize {
        self.prompt_token_ids.len() + self.output_token_ids.len()
    }

    pub fn is_finished(&self) -> bool {
        matches!(self.status, SequenceStatus::Finished(_))
    }

    pub fn reset_decode_stream(&mut self) {
        self.decode_stream_prefix.clear();
        self.decode_stream.reset();
    }
}

pub struct SequenceGroup {
    pub request_id: String,
    pub session_id: Option<String>,
    pub sequences: HashMap<u64, Arc<std::sync::Mutex<Sequence>>>,
    pub arrival_time: std::time::Instant,
    pub sampling_params: SamplingParams,
    pub cancellation: Option<CancellationToken>,
    /// Scheduling priority for preemption decisions.
    ///
    /// Higher values are more important: when the block pool is under pressure,
    /// running groups with a *lower* `priority` are swapped out first.
    /// `0` = normal (default).  `255` = highest (latency-critical).
    pub priority: u8,
    // Channel for streaming incremental outputs
    pub response_tx: mpsc::Sender<SequenceGroupOutput>,
    cached_total_len: usize,
    cached_running_count: usize,
    cached_finished_count: usize,
    cached_seq_lens: HashMap<u64, usize>,
}

impl SequenceGroup {
    pub fn new(
        request_id: String,
        session_id: Option<String>,
        prompt: String,
        prompt_token_ids: Vec<u32>,
        sampling_params: SamplingParams,
        cancellation: Option<CancellationToken>,
        response_tx: mpsc::Sender<SequenceGroupOutput>,
    ) -> Self {
        let mut sequences = HashMap::new();
        // Use a simple ID for now, managed by the engine
        let seq_id = 0;
        let initial_len = prompt_token_ids.len();
        let sequence = Sequence::new(seq_id, prompt, prompt_token_ids);
        sequences.insert(seq_id, Arc::new(std::sync::Mutex::new(sequence)));
        let mut cached_seq_lens = HashMap::new();
        cached_seq_lens.insert(seq_id, initial_len);

        Self {
            request_id,
            session_id,
            sequences,
            arrival_time: std::time::Instant::now(),
            sampling_params,
            cancellation,
            priority: 0,
            response_tx,
            cached_total_len: initial_len,
            cached_running_count: 0,
            cached_finished_count: 0,
            cached_seq_lens,
        }
    }

    pub fn get_seqs(&self) -> Vec<Arc<std::sync::Mutex<Sequence>>> {
        self.sequences.values().cloned().collect()
    }

    pub fn is_finished(&self) -> bool {
        self.cached_finished_count >= self.sequences.len()
    }

    pub fn cached_total_len(&self) -> usize {
        self.cached_total_len
    }

    pub fn cached_running_count(&self) -> usize {
        self.cached_running_count
    }

    pub fn cached_seq_lens(&self) -> Vec<(u64, usize)> {
        self.cached_seq_lens
            .iter()
            .map(|(seq_id, len)| (*seq_id, *len))
            .collect()
    }

    pub fn reset_cache_for_single_seq(&mut self, seq_id: u64, len: usize, status: SequenceStatus) {
        self.cached_seq_lens.clear();
        self.cached_seq_lens.insert(seq_id, len);
        self.cached_total_len = len;
        self.cached_running_count = if matches!(status, SequenceStatus::Running) {
            1
        } else {
            0
        };
        self.cached_finished_count = if matches!(status, SequenceStatus::Finished(_)) {
            1
        } else {
            0
        };
    }

    pub fn update_seq_len(&mut self, seq_id: u64, new_len: usize) {
        match self.cached_seq_lens.insert(seq_id, new_len) {
            Some(old_len) => {
                if new_len >= old_len {
                    self.cached_total_len += new_len - old_len;
                } else {
                    self.cached_total_len -= old_len - new_len;
                }
            }
            None => {
                self.cached_total_len += new_len;
            }
        }
    }

    pub fn update_seq_status(&mut self, old_status: SequenceStatus, new_status: SequenceStatus) {
        let was_running = matches!(old_status, SequenceStatus::Running);
        let is_running = matches!(new_status, SequenceStatus::Running);
        if was_running != is_running {
            if is_running {
                self.cached_running_count += 1;
            } else {
                self.cached_running_count = self.cached_running_count.saturating_sub(1);
            }
        }

        let was_finished = matches!(old_status, SequenceStatus::Finished(_));
        let is_finished = matches!(new_status, SequenceStatus::Finished(_));
        if was_finished != is_finished {
            if is_finished {
                self.cached_finished_count += 1;
            } else {
                self.cached_finished_count = self.cached_finished_count.saturating_sub(1);
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct SamplingParams {
    pub max_tokens: usize,
    pub min_tokens: usize,
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: usize,
    pub stop_token_ids: Vec<u32>,
    pub repetition_penalty: f32,
    pub seed: Option<u64>,
}

#[derive(Debug, Clone)]
pub struct SequenceGroupOutput {
    pub request_id: String,
    // Incremental text produced by the tokenizer's stateful decode stream.
    pub text: String,
    pub finish_reason: Option<FinishReason>,
}

#[path = "sequence_tests.rs"]
mod sequence_tests;
