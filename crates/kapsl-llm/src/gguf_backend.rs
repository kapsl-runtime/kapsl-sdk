use async_stream::stream;
use async_trait::async_trait;
use kapsl_engine_api::{
    BinaryTensorPacket, Engine, EngineError, EngineMetrics, EngineModelInfo, EngineStream,
    InferenceRequest, TensorDtype,
};
use std::collections::VecDeque;
use std::num::NonZeroU32;
use std::path::Path;
use std::sync::mpsc as std_mpsc;
use std::sync::Arc;
use std::time::{Duration, Instant};

#[cfg(feature = "gguf")]
use llama_cpp_2::{
    context::params::LlamaContextParams,
    llama_backend::LlamaBackend,
    llama_batch::LlamaBatch,
    model::{params::LlamaModelParams, AddBos, LlamaModel},
    token::LlamaToken,
    TokenToStringError,
};
#[cfg(feature = "gguf")]
use llama_cpp_sys_2::LLAMA_FLASH_ATTN_TYPE_AUTO;

// ─── Configuration ────────────────────────────────────────────────────────────

const MAX_CONCURRENT_DEFAULT: usize = 32;
const N_CTX_PER_SEQ_DEFAULT: u32 = 2048;
const GGUF_QUEUE_DELAY_US_DEFAULT: u64 = 1_000;
const GGUF_PREFILL_CHUNK_SIZE_DEFAULT: usize = 512;
const GGUF_EXACT_PROMPT_KV_REUSE_DEFAULT: bool = true;

#[cfg(feature = "gguf")]
fn max_concurrent() -> usize {
    std::env::var("KAPSL_GGUF_MAX_CONCURRENT")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&v| v > 0)
        .unwrap_or(MAX_CONCURRENT_DEFAULT)
}

#[cfg(feature = "gguf")]
fn n_ctx_per_seq() -> u32 {
    std::env::var("KAPSL_GGUF_CTX_PER_SEQ")
        .ok()
        .and_then(|v| v.parse::<u32>().ok())
        .filter(|&v| v > 0)
        .unwrap_or(N_CTX_PER_SEQ_DEFAULT)
}

#[cfg(feature = "gguf")]
fn gguf_queue_delay() -> Duration {
    let micros = std::env::var("KAPSL_GGUF_QUEUE_DELAY_US")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(GGUF_QUEUE_DELAY_US_DEFAULT);
    Duration::from_micros(micros)
}

#[cfg(feature = "gguf")]
fn gguf_prefill_chunk_size() -> usize {
    std::env::var("KAPSL_GGUF_PREFILL_CHUNK_SIZE")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&v| v > 0)
        .unwrap_or(GGUF_PREFILL_CHUNK_SIZE_DEFAULT)
}

#[cfg(feature = "gguf")]
fn gguf_exact_prompt_kv_reuse() -> bool {
    std::env::var("KAPSL_GGUF_EXACT_PROMPT_KV_REUSE")
        .ok()
        .map(|v| {
            let v = v.trim().to_ascii_lowercase();
            !matches!(v.as_str(), "0" | "false" | "no" | "off")
        })
        .unwrap_or(GGUF_EXACT_PROMPT_KV_REUSE_DEFAULT)
}

// ─── Shared model weights cache ───────────────────────────────────────────────

#[cfg(feature = "gguf")]
struct GgufWeights {
    backend: Arc<LlamaBackend>,
    model: Arc<LlamaModel>,
    n_ctx_train: u32,
}

#[cfg(feature = "gguf")]
static GGUF_WEIGHTS_CACHE: std::sync::OnceLock<
    std::sync::Mutex<std::collections::HashMap<std::path::PathBuf, std::sync::Weak<GgufWeights>>>,
> = std::sync::OnceLock::new();

#[cfg(feature = "gguf")]
fn gguf_weights_cache() -> &'static std::sync::Mutex<
    std::collections::HashMap<std::path::PathBuf, std::sync::Weak<GgufWeights>>,
> {
    GGUF_WEIGHTS_CACHE.get_or_init(|| std::sync::Mutex::new(std::collections::HashMap::new()))
}

// Global LlamaBackend singleton — llama.cpp allows only one backend per process.
#[cfg(feature = "gguf")]
static GGUF_BACKEND: std::sync::OnceLock<Arc<LlamaBackend>> = std::sync::OnceLock::new();

#[cfg(feature = "gguf")]
static GGUF_BACKEND_INIT_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[cfg(feature = "gguf")]
fn global_gguf_backend() -> Result<Arc<LlamaBackend>, EngineError> {
    if let Some(b) = GGUF_BACKEND.get() {
        return Ok(Arc::clone(b));
    }
    let _lock = GGUF_BACKEND_INIT_LOCK.lock().unwrap();
    if let Some(b) = GGUF_BACKEND.get() {
        return Ok(Arc::clone(b));
    }
    let backend = LlamaBackend::init()
        .map_err(|e| EngineError::backend(format!("llama backend init failed: {e}")))?;
    let arc = Arc::new(backend);
    let _ = GGUF_BACKEND.set(Arc::clone(&arc));
    Ok(arc)
}

// ─── Scheduler types ──────────────────────────────────────────────────────────

/// A request submitted to the scheduler thread.
#[cfg(feature = "gguf")]
struct GgufRequest {
    tokens: Vec<LlamaToken>,
    max_tokens: i32,
    min_tokens: i32,
    response: GgufResponse,
}

#[cfg(feature = "gguf")]
enum GgufResponse {
    Final(std_mpsc::Sender<Result<Vec<u8>, EngineError>>),
    Stream(std_mpsc::Sender<Result<Vec<u8>, EngineError>>),
}

#[cfg(feature = "gguf")]
impl GgufResponse {
    fn emit_piece(&self, output: &mut Vec<u8>, piece: Vec<u8>) -> bool {
        match self {
            Self::Final(_) => {
                output.extend_from_slice(&piece);
                true
            }
            Self::Stream(tx) => tx.send(Ok(piece)).is_ok(),
        }
    }

    fn finish(self, output: Vec<u8>) {
        if let Self::Final(tx) = self {
            let _ = tx.send(Ok(output));
        }
    }

    fn send_error(self, error: EngineError) {
        match self {
            Self::Final(tx) | Self::Stream(tx) => {
                let _ = tx.send(Err(error));
            }
        }
    }
}

/// A request that has been tokenized and assigned a sequence slot, awaiting prefill.
#[cfg(feature = "gguf")]
struct PendingPrefill {
    seq_id: i32,
    tokens: Vec<LlamaToken>,
    next_token: usize,
    max_tokens: i32,
    min_tokens: i32,
    response: GgufResponse,
    copies: Vec<PendingPrefillCopy>,
}

#[cfg(feature = "gguf")]
struct PendingPrefillCopy {
    seq_id: i32,
    max_tokens: i32,
    min_tokens: i32,
    response: GgufResponse,
}

/// A sequence in the decode phase.
#[cfg(feature = "gguf")]
struct ActiveSeq {
    seq_id: i32,
    /// Next KV-cache position to write.
    pos: i32,
    n_generated: i32,
    max_tokens: i32,
    min_tokens: i32,
    /// Token to feed into the next decode step.
    last_token: LlamaToken,
    output: Vec<u8>,
    response: GgufResponse,
    error: Option<EngineError>,
}

// ─── Backend ──────────────────────────────────────────────────────────────────

pub struct GgufBackend {
    #[cfg(feature = "gguf")]
    inner: Option<GgufInner>,
    metrics: EngineMetrics,
}

#[cfg(feature = "gguf")]
struct GgufInner {
    weights: Arc<GgufWeights>,
    request_tx: std_mpsc::Sender<GgufRequest>,
    max_concurrent: usize,
}

impl GgufBackend {
    pub fn new() -> Self {
        Self {
            #[cfg(feature = "gguf")]
            inner: None,
            metrics: EngineMetrics::new(),
        }
    }

    fn extract_prompt(request: &InferenceRequest) -> Result<String, EngineError> {
        String::from_utf8(request.input.data.clone())
            .map_err(|e| EngineError::invalid_input(format!("Input is not valid UTF-8: {e}")))
    }

    fn max_new_tokens(request: &InferenceRequest) -> i32 {
        request
            .metadata
            .as_ref()
            .and_then(|m| m.max_new_tokens)
            .unwrap_or(512) as i32
    }

    fn min_new_tokens(request: &InferenceRequest) -> i32 {
        request
            .metadata
            .as_ref()
            .and_then(|m| m.min_new_tokens)
            .unwrap_or(0) as i32
    }
}

impl Default for GgufBackend {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Scheduler loop ───────────────────────────────────────────────────────────

/// Sample the next token greedily from the logits row produced for `batch_pos`.
/// Skipping EOS here is equivalent to the old logit-bias sampler path but avoids
/// allocating a llama sampler chain for every generated token.
#[cfg(feature = "gguf")]
fn sample_token(
    ctx: &llama_cpp_2::context::LlamaContext,
    batch_pos: i32,
    eos_token: LlamaToken,
    ban_eos: bool,
) -> LlamaToken {
    let logits = ctx.get_logits_ith(batch_pos);
    let mut best_token = 0_i32;
    let mut best_logit = f32::NEG_INFINITY;

    for (idx, &logit) in logits.iter().enumerate() {
        let token = idx as i32;
        if ban_eos && token == eos_token.0 {
            continue;
        }
        if logit > best_logit {
            best_logit = logit;
            best_token = token;
        }
    }

    LlamaToken::new(best_token)
}

#[cfg(feature = "gguf")]
fn token_piece_bytes(model: &LlamaModel, token: LlamaToken) -> Result<Vec<u8>, EngineError> {
    match model.token_to_piece_bytes(token, 256, true, None) {
        Ok(piece) => Ok(piece),
        Err(TokenToStringError::InsufficientBufferSpace(size)) if size < 0 => model
            .token_to_piece_bytes(token, (-size) as usize, true, None)
            .map_err(|e| EngineError::backend(format!("token decode failed: {e}"))),
        Err(e) => Err(EngineError::backend(format!("token decode failed: {e}"))),
    }
}

#[cfg(feature = "gguf")]
fn sequence_needs_kv_after_first(
    max_tokens: i32,
    min_tokens: i32,
    first_tok: LlamaToken,
    eos_token: LlamaToken,
) -> bool {
    max_tokens > 1 && !(first_tok == eos_token && min_tokens <= 0)
}

#[cfg(feature = "gguf")]
fn finish_or_activate_prefilled_sequence(
    ctx: &mut llama_cpp_2::context::LlamaContext,
    available_ids: &mut Vec<i32>,
    active: &mut Vec<ActiveSeq>,
    seq_id: i32,
    prompt_len: i32,
    max_tokens: i32,
    min_tokens: i32,
    response: GgufResponse,
    eos_token: LlamaToken,
    first_tok: LlamaToken,
    first_piece: Option<&[u8]>,
) {
    let mut output = Vec::with_capacity((max_tokens.max(0) as usize).saturating_mul(4));

    if max_tokens <= 0 || (first_tok == eos_token && min_tokens <= 0) {
        let _ = ctx.clear_kv_cache_seq(Some(seq_id as u32), None, None);
        available_ids.push(seq_id);
        response.finish(output);
        return;
    }

    let Some(piece) = first_piece else {
        let _ = ctx.clear_kv_cache_seq(Some(seq_id as u32), None, None);
        available_ids.push(seq_id);
        response.send_error(EngineError::backend("missing first token piece"));
        return;
    };

    if !response.emit_piece(&mut output, piece.to_vec()) {
        let _ = ctx.clear_kv_cache_seq(Some(seq_id as u32), None, None);
        available_ids.push(seq_id);
        return;
    }

    if max_tokens <= 1 {
        let _ = ctx.clear_kv_cache_seq(Some(seq_id as u32), None, None);
        available_ids.push(seq_id);
        response.finish(output);
    } else {
        active.push(ActiveSeq {
            seq_id,
            pos: prompt_len,
            n_generated: 1,
            max_tokens,
            min_tokens,
            last_token: first_tok,
            output,
            response,
            error: None,
        });
    }
}

#[cfg(feature = "gguf")]
fn coalesce_exact_prompt_copies(pref: &mut PendingPrefill, pending: &mut VecDeque<PendingPrefill>) {
    if pref.next_token != 0 || !pref.copies.is_empty() || pending.is_empty() {
        return;
    }

    let mut rest = VecDeque::with_capacity(pending.len());
    while let Some(candidate) = pending.pop_front() {
        if candidate.next_token == 0
            && candidate.copies.is_empty()
            && candidate.min_tokens == pref.min_tokens
            && candidate.tokens == pref.tokens
        {
            pref.copies.push(PendingPrefillCopy {
                seq_id: candidate.seq_id,
                max_tokens: candidate.max_tokens,
                min_tokens: candidate.min_tokens,
                response: candidate.response,
            });
        } else {
            rest.push_back(candidate);
        }
    }

    *pending = rest;
}

#[cfg(feature = "gguf")]
fn fail_pending_prefill(
    ctx: &mut llama_cpp_2::context::LlamaContext,
    available_ids: &mut Vec<i32>,
    mut pref: PendingPrefill,
    message: &'static str,
) {
    let _ = ctx.clear_kv_cache_seq(Some(pref.seq_id as u32), None, None);
    available_ids.push(pref.seq_id);
    pref.response.send_error(EngineError::backend(message));

    for copy in pref.copies.drain(..) {
        let _ = ctx.clear_kv_cache_seq(Some(copy.seq_id as u32), None, None);
        available_ids.push(copy.seq_id);
        copy.response.send_error(EngineError::backend(message));
    }
}

/// Runs on a dedicated OS thread. Holds the single `LlamaContext` and multiplexes
/// all concurrent requests through one batched decode loop, matching the vLLM
/// continuous-batching pattern.
#[cfg(feature = "gguf")]
fn run_scheduler(
    model: Arc<LlamaModel>,
    backend: Arc<LlamaBackend>,
    request_rx: std_mpsc::Receiver<GgufRequest>,
    max_concurrent: usize,
    ctx_per_seq: u32,
) {
    let total_ctx = max_concurrent as u32 * ctx_per_seq;
    let n_ctx = match NonZeroU32::new(total_ctx) {
        Some(v) => v,
        None => {
            log::error!("[gguf] invalid total_ctx=0");
            return;
        }
    };
    // n_batch: upper bound for prefill work plus all active decode tokens.
    let n_batch = ctx_per_seq + max_concurrent as u32;
    let queue_delay = gguf_queue_delay();
    let prefill_chunk_size = gguf_prefill_chunk_size().min(n_batch as usize).max(1);
    let exact_prompt_kv_reuse = gguf_exact_prompt_kv_reuse();

    let ctx_params = LlamaContextParams::default()
        .with_n_ctx(Some(n_ctx))
        .with_n_batch(n_batch)
        .with_n_ubatch(n_batch)
        .with_n_seq_max(max_concurrent as u32)
        .with_offload_kqv(true)
        .with_flash_attention_policy(LLAMA_FLASH_ATTN_TYPE_AUTO);

    let mut ctx = match model.new_context(&backend, ctx_params) {
        Ok(c) => c,
        Err(e) => {
            log::error!("[gguf] Failed to create shared context: {e}");
            return;
        }
    };
    log::info!("[gguf] Shared context ready (n_ctx={total_ctx}, max_concurrent={max_concurrent})");
    log::info!("[gguf] Prefill chunk size={prefill_chunk_size}");
    log::info!("[gguf] Exact prompt KV reuse={exact_prompt_kv_reuse}");

    let eos_token = model.token_eos();
    let batch_cap = n_batch as usize;
    let mut batch = LlamaBatch::new(batch_cap, 1);

    // seq_id pool: 0..max_concurrent are valid sequence identifiers for the KV cache.
    let mut available_ids: Vec<i32> = (0..max_concurrent as i32).rev().collect();
    let mut waiting: VecDeque<GgufRequest> = VecDeque::new();
    let mut pending: VecDeque<PendingPrefill> = VecDeque::new();
    let mut active: Vec<ActiveSeq> = Vec::with_capacity(max_concurrent);

    'main: loop {
        // ── 1. Drain the request channel ──────────────────────────────────────
        loop {
            match request_rx.try_recv() {
                Ok(req) => waiting.push_back(req),
                Err(std_mpsc::TryRecvError::Empty) => break,
                Err(std_mpsc::TryRecvError::Disconnected) => {
                    // Backend unloaded. Finish in-flight work then exit.
                    if waiting.is_empty() && pending.is_empty() && active.is_empty() {
                        break 'main;
                    }
                    break; // keep processing existing work
                }
            }
        }

        // ── 2. Promote waiting → pending (assign seq_id) ─────────────────────
        while !waiting.is_empty() && !available_ids.is_empty() {
            let req = waiting.pop_front().unwrap();
            let seq_id = available_ids.pop().unwrap();

            if req.tokens.len() as u32 > ctx_per_seq {
                req.response.send_error(EngineError::invalid_input(format!(
                    "prompt has {} tokens, exceeding ctx_per_seq={ctx_per_seq}",
                    req.tokens.len()
                )));
                available_ids.push(seq_id);
                continue;
            }
            if req.tokens.is_empty() {
                req.response
                    .send_error(EngineError::invalid_input("prompt has no tokens"));
                available_ids.push(seq_id);
                continue;
            }

            pending.push_back(PendingPrefill {
                seq_id,
                tokens: req.tokens,
                next_token: 0,
                max_tokens: req.max_tokens,
                min_tokens: req.min_tokens,
                response: req.response,
                copies: Vec::new(),
            });
        }

        // ── 3. If completely idle, block for the next request ─────────────────
        if waiting.is_empty() && pending.is_empty() && active.is_empty() {
            match request_rx.recv() {
                Ok(req) => waiting.push_back(req),
                Err(_) => break 'main,
            }
            if !queue_delay.is_zero() {
                let deadline = Instant::now() + queue_delay;
                while waiting.len() < max_concurrent {
                    let now = Instant::now();
                    if now >= deadline {
                        break;
                    }
                    match request_rx.recv_timeout(deadline.saturating_duration_since(now)) {
                        Ok(req) => waiting.push_back(req),
                        Err(std_mpsc::RecvTimeoutError::Timeout) => break,
                        Err(std_mpsc::RecvTimeoutError::Disconnected) => break,
                    }
                }
            }
            continue;
        }

        // ── 4. Build batch: multiple prefills (if any) + all active decode tokens ───
        // Reserve `active.len()` slots for decode tokens so prefills don't crowd them out.
        batch.clear();
        let prefill_budget = batch_cap.saturating_sub(active.len());
        let mut completed_prefills: Vec<(PendingPrefill, i32)> = Vec::new();
        let mut partial_prefills: Vec<PendingPrefill> = Vec::new();

        while !pending.is_empty() {
            let prefill_slots_used = batch.n_tokens().max(0) as usize;
            if prefill_slots_used >= prefill_budget {
                break;
            }

            let slots_remaining = prefill_budget - prefill_slots_used;
            let mut pref = pending.pop_front().unwrap();
            if exact_prompt_kv_reuse {
                coalesce_exact_prompt_copies(&mut pref, &mut pending);
            }

            let remaining_prompt = pref.tokens.len().saturating_sub(pref.next_token);
            let chunk_len = remaining_prompt
                .min(prefill_chunk_size)
                .min(slots_remaining);

            if chunk_len == 0 {
                pending.push_front(pref);
                break;
            }

            let start = pref.next_token;
            let end = start + chunk_len;
            let mut last_prompt_batch_pos = None;
            let mut add_ok = true;
            for idx in start..end {
                let is_last_prompt_token = idx + 1 == pref.tokens.len();
                let batch_pos = batch.n_tokens();
                if batch
                    .add(
                        pref.tokens[idx],
                        idx as i32,
                        &[pref.seq_id],
                        is_last_prompt_token,
                    )
                    .is_err()
                {
                    add_ok = false;
                    break;
                }
                if is_last_prompt_token {
                    last_prompt_batch_pos = Some(batch_pos);
                }
            }

            if add_ok {
                pref.next_token = end;
                if pref.next_token == pref.tokens.len() {
                    if let Some(last_pos) = last_prompt_batch_pos {
                        completed_prefills.push((pref, last_pos));
                    } else {
                        fail_pending_prefill(
                            &mut ctx,
                            &mut available_ids,
                            pref,
                            "empty prefill batch",
                        );
                    }
                } else {
                    partial_prefills.push(pref);
                }
            } else {
                fail_pending_prefill(
                    &mut ctx,
                    &mut available_ids,
                    pref,
                    "batch capacity exceeded",
                );
            }
        }

        let mut decode_batch_positions: Vec<i32> = Vec::with_capacity(active.len());
        for seq in &active {
            let pos = batch.n_tokens();
            if batch
                .add(seq.last_token, seq.pos, &[seq.seq_id], true)
                .is_ok()
            {
                decode_batch_positions.push(pos);
            } else {
                decode_batch_positions.push(-1); // skipped this step
            }
        }

        if batch.n_tokens() == 0 {
            continue;
        }

        // ── 5. Execute one forward pass for all sequences in the batch ────────
        if let Err(e) = ctx.decode(&mut batch) {
            log::error!("[gguf] decode error: {e}");
            for seq in active.drain(..) {
                seq.response
                    .send_error(EngineError::backend("decode failed"));
                let _ = ctx.clear_kv_cache_seq(Some(seq.seq_id as u32), None, None);
                available_ids.push(seq.seq_id);
            }
            for pref in partial_prefills.drain(..) {
                fail_pending_prefill(&mut ctx, &mut available_ids, pref, "decode failed");
            }
            for (pref, _) in completed_prefills.drain(..) {
                fail_pending_prefill(&mut ctx, &mut available_ids, pref, "decode failed");
            }
            continue;
        }

        for pref in partial_prefills.drain(..) {
            pending.push_back(pref);
        }

        // ── 6. Sample each newly prefilled sequence and move it to active ──────
        for (mut pref, last_pos) in completed_prefills.drain(..) {
            // EOS is skipped during greedy sampling when min_tokens > 0, so first_tok is
            // guaranteed to be a real content token whenever min_tokens is nonzero.
            let first_tok = sample_token(&ctx, last_pos, eos_token, pref.min_tokens > 0);
            let prompt_len = pref.tokens.len() as i32;

            let emits_first_piece =
                pref.max_tokens > 0 && !(first_tok == eos_token && pref.min_tokens <= 0);
            let copy_emits_first_piece = pref.copies.iter().any(|copy| {
                copy.max_tokens > 0 && !(first_tok == eos_token && copy.min_tokens <= 0)
            });

            let first_piece = if emits_first_piece || copy_emits_first_piece {
                match token_piece_bytes(&model, first_tok) {
                    Ok(piece) => Some(piece),
                    Err(e) => {
                        let _ = ctx.clear_kv_cache_seq(Some(pref.seq_id as u32), None, None);
                        available_ids.push(pref.seq_id);
                        pref.response.send_error(e);
                        for copy in pref.copies.drain(..) {
                            let _ = ctx.clear_kv_cache_seq(Some(copy.seq_id as u32), None, None);
                            available_ids.push(copy.seq_id);
                            copy.response
                                .send_error(EngineError::backend("token decode failed"));
                        }
                        continue;
                    }
                }
            } else {
                None
            };

            let mut ready_copies = Vec::with_capacity(pref.copies.len());
            for copy in pref.copies.drain(..) {
                if sequence_needs_kv_after_first(
                    copy.max_tokens,
                    copy.min_tokens,
                    first_tok,
                    eos_token,
                ) {
                    if let Err(e) = ctx.copy_kv_cache_seq(
                        pref.seq_id,
                        copy.seq_id,
                        Some(0),
                        Some(prompt_len as u32),
                    ) {
                        log::warn!(
                            "[gguf] exact prompt KV copy failed from seq={} to seq={}: {e}",
                            pref.seq_id,
                            copy.seq_id
                        );
                        let _ = ctx.clear_kv_cache_seq(Some(copy.seq_id as u32), None, None);
                        available_ids.push(copy.seq_id);
                        copy.response
                            .send_error(EngineError::backend("KV cache copy failed"));
                        continue;
                    }
                }
                ready_copies.push(copy);
            }

            finish_or_activate_prefilled_sequence(
                &mut ctx,
                &mut available_ids,
                &mut active,
                pref.seq_id,
                prompt_len,
                pref.max_tokens,
                pref.min_tokens,
                pref.response,
                eos_token,
                first_tok,
                first_piece.as_deref(),
            );

            for copy in ready_copies {
                finish_or_activate_prefilled_sequence(
                    &mut ctx,
                    &mut available_ids,
                    &mut active,
                    copy.seq_id,
                    prompt_len,
                    copy.max_tokens,
                    copy.min_tokens,
                    copy.response,
                    eos_token,
                    first_tok,
                    first_piece.as_deref(),
                );
            }
        }

        // ── 7. Sample each active sequence and advance or retire it ──────────
        let mut to_retire: Vec<usize> = Vec::new();
        for (i, (seq, &batch_pos)) in active
            .iter_mut()
            .zip(decode_batch_positions.iter())
            .enumerate()
        {
            if batch_pos < 0 {
                continue; // this sequence was not in the batch this step
            }

            // Ban EOS when min_tokens not yet reached so the model is forced to emit
            // real tokens instead of cycling on suppressed EOS indefinitely.
            let next_tok =
                sample_token(&ctx, batch_pos, eos_token, seq.n_generated < seq.min_tokens);
            seq.pos += 1;

            let eos_and_ready = next_tok == eos_token && seq.n_generated >= seq.min_tokens;
            let next_generated = seq.n_generated + 1;
            let max_reached = next_generated >= seq.max_tokens;

            if eos_and_ready || max_reached {
                if eos_and_ready {
                    to_retire.push(i);
                } else {
                    let piece = match token_piece_bytes(&model, next_tok) {
                        Ok(piece) => piece,
                        Err(e) => {
                            seq.error = Some(e);
                            to_retire.push(i);
                            continue;
                        }
                    };
                    if seq.response.emit_piece(&mut seq.output, piece) {
                        seq.last_token = next_tok;
                        seq.n_generated = next_generated;
                    }
                    to_retire.push(i);
                }
            } else {
                let piece = match token_piece_bytes(&model, next_tok) {
                    Ok(piece) => piece,
                    Err(e) => {
                        seq.error = Some(e);
                        to_retire.push(i);
                        continue;
                    }
                };
                if !seq.response.emit_piece(&mut seq.output, piece) {
                    to_retire.push(i);
                } else {
                    seq.last_token = next_tok;
                    seq.n_generated = next_generated;
                }
            }
        }

        for &i in to_retire.iter().rev() {
            let done = active.remove(i);
            let _ = ctx.clear_kv_cache_seq(Some(done.seq_id as u32), None, None);
            available_ids.push(done.seq_id);
            if let Some(error) = done.error {
                done.response.send_error(error);
            } else {
                done.response.finish(done.output);
            }
        }
    }

    log::info!("[gguf] Scheduler thread exiting");
}

// ─── Engine impl ──────────────────────────────────────────────────────────────

#[cfg(feature = "gguf")]
#[async_trait]
impl Engine for GgufBackend {
    async fn load(&mut self, model_path: &Path) -> Result<(), EngineError> {
        let model_path_key = model_path.to_path_buf();

        let cached = gguf_weights_cache()
            .lock()
            .unwrap()
            .get(&model_path_key)
            .and_then(|w| w.upgrade());

        let weights = if let Some(shared) = cached {
            log::info!(
                "[gguf] Reusing shared weights for {}",
                model_path_key.display()
            );
            shared
        } else {
            let model_path_load = model_path_key.clone();
            let backend_for_load = global_gguf_backend()?;
            let backend_for_closure = Arc::clone(&backend_for_load);
            let (model, n_ctx_train) = tokio::task::spawn_blocking(move || {
                let params = LlamaModelParams::default().with_n_gpu_layers(99);
                let model =
                    LlamaModel::load_from_file(&backend_for_closure, &model_path_load, &params)
                        .map_err(|e| EngineError::backend(format!("GGUF load failed: {e}")))?;
                let n_ctx_train = model.n_ctx_train();
                Ok::<_, EngineError>((model, n_ctx_train))
            })
            .await
            .map_err(|e| EngineError::backend(format!("spawn_blocking join error: {e}")))??;

            let arc = Arc::new(GgufWeights {
                backend: backend_for_load,
                model: Arc::new(model),
                n_ctx_train: n_ctx_train as u32,
            });
            gguf_weights_cache()
                .lock()
                .unwrap()
                .insert(model_path_key, Arc::downgrade(&arc));
            arc
        };

        let max_conc = max_concurrent();
        let ctx_per_seq = n_ctx_per_seq().min(weights.n_ctx_train);

        let (tx, rx) = std_mpsc::channel::<GgufRequest>();
        let model_clone = Arc::clone(&weights.model);
        let backend_clone = Arc::clone(&weights.backend);
        std::thread::spawn(move || {
            run_scheduler(model_clone, backend_clone, rx, max_conc, ctx_per_seq);
        });

        log::info!(
            "[gguf] Scheduler started: max_concurrent={max_conc}, ctx_per_seq={ctx_per_seq}"
        );

        self.inner = Some(GgufInner {
            weights,
            request_tx: tx,
            max_concurrent: max_conc,
        });
        Ok(())
    }

    fn infer(&self, request: &InferenceRequest) -> Result<BinaryTensorPacket, EngineError> {
        let inner = self.inner.as_ref().ok_or(EngineError::ModelNotLoaded)?;
        let prompt = Self::extract_prompt(request)?;
        let tokens = inner
            .weights
            .model
            .str_to_token(&prompt, AddBos::Always)
            .map_err(|e| EngineError::backend(format!("tokenization failed: {e}")))?;
        let (resp_tx, resp_rx) = std_mpsc::channel::<Result<Vec<u8>, EngineError>>();

        inner
            .request_tx
            .send(GgufRequest {
                tokens,
                max_tokens: Self::max_new_tokens(request),
                min_tokens: Self::min_new_tokens(request),
                response: GgufResponse::Final(resp_tx),
            })
            .map_err(|_| EngineError::backend("gguf scheduler disconnected"))?;

        let data = match resp_rx.recv() {
            Ok(result) => result?,
            Err(_) => Vec::new(),
        };
        let len = data.len() as i64;
        BinaryTensorPacket::new(vec![1, len], TensorDtype::Uint8, data)
            .map_err(|e| EngineError::backend(format!("Failed to build output packet: {e}")))
    }

    fn infer_stream(&self, request: &InferenceRequest) -> EngineStream {
        let inner = match self.inner.as_ref() {
            Some(i) => i,
            None => {
                return Box::pin(stream! {
                    yield Err(EngineError::ModelNotLoaded);
                });
            }
        };

        let prompt = match Self::extract_prompt(request) {
            Ok(p) => p,
            Err(e) => {
                return Box::pin(stream! { yield Err(e); });
            }
        };
        let tokens = match inner.weights.model.str_to_token(&prompt, AddBos::Always) {
            Ok(tokens) => tokens,
            Err(e) => {
                return Box::pin(stream! {
                    yield Err(EngineError::backend(format!("tokenization failed: {e}")));
                });
            }
        };

        let (resp_tx, resp_rx) = std_mpsc::channel::<Result<Vec<u8>, EngineError>>();

        if inner
            .request_tx
            .send(GgufRequest {
                tokens,
                max_tokens: Self::max_new_tokens(request),
                min_tokens: Self::min_new_tokens(request),
                response: GgufResponse::Stream(resp_tx),
            })
            .is_err()
        {
            return Box::pin(stream! {
                yield Err(EngineError::backend("gguf scheduler disconnected"));
            });
        }

        // Bridge blocking std::mpsc → async tokio channel.
        let (tok_tx, mut tok_rx) = tokio::sync::mpsc::channel::<Result<Vec<u8>, EngineError>>(64);
        std::thread::spawn(move || {
            for piece in resp_rx {
                if tok_tx.blocking_send(piece).is_err() {
                    break;
                }
            }
        });

        Box::pin(stream! {
            while let Some(result) = tok_rx.recv().await {
                let data = result?;
                let len = data.len() as i64;
                yield BinaryTensorPacket::new(vec![1, len], TensorDtype::Uint8, data)
                    .map_err(|e| EngineError::backend(format!("Output packet error: {e}")));
            }
        })
    }

    fn unload(&mut self) {
        self.inner = None; // drops request_tx → scheduler thread exits
        log::info!("[gguf] Backend unloaded");
    }

    fn metrics(&self) -> EngineMetrics {
        self.metrics.clone()
    }

    fn health_check(&self) -> Result<(), EngineError> {
        if self.inner.is_some() {
            Ok(())
        } else {
            Err(EngineError::ModelNotLoaded)
        }
    }

    fn model_info(&self) -> Option<EngineModelInfo> {
        let inner = self.inner.as_ref()?;
        Some(EngineModelInfo {
            input_names: vec!["text".to_string()],
            output_names: vec!["text".to_string()],
            input_shapes: vec![vec![-1]],
            output_shapes: vec![vec![-1]],
            input_dtypes: vec!["uint8".to_string()],
            output_dtypes: vec!["uint8".to_string()],
            framework: Some("gguf".to_string()),
            model_version: None,
            peak_concurrency: Some(inner.max_concurrent as u32),
        })
    }
}

// ─── Stub impl when gguf feature is disabled ──────────────────────────────────

#[cfg(not(feature = "gguf"))]
#[async_trait]
impl Engine for GgufBackend {
    async fn load(&mut self, _model_path: &Path) -> Result<(), EngineError> {
        Err(EngineError::backend(
            "GGUF support not compiled in (enable the 'gguf' feature)",
        ))
    }

    fn infer(&self, _request: &InferenceRequest) -> Result<BinaryTensorPacket, EngineError> {
        Err(EngineError::backend(
            "GGUF support not compiled in (enable the 'gguf' feature)",
        ))
    }

    fn infer_stream(&self, _request: &InferenceRequest) -> EngineStream {
        Box::pin(stream! {
            yield Err(EngineError::backend(
                "GGUF support not compiled in (enable the 'gguf' feature)",
            ));
        })
    }

    fn unload(&mut self) {}

    fn metrics(&self) -> EngineMetrics {
        self.metrics.clone()
    }

    fn health_check(&self) -> Result<(), EngineError> {
        Err(EngineError::backend(
            "GGUF support not compiled in (enable the 'gguf' feature)",
        ))
    }
}
