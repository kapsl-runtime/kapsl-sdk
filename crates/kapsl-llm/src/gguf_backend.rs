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

#[cfg(feature = "gguf")]
use llama_cpp_2::{
    context::params::LlamaContextParams,
    llama_backend::LlamaBackend,
    llama_batch::LlamaBatch,
    model::{params::LlamaModelParams, AddBos, LlamaModel, Special},
    sampling::LlamaSampler,
    token::LlamaToken,
};

// ─── Configuration ────────────────────────────────────────────────────────────

const MAX_CONCURRENT_DEFAULT: usize = 8;
const N_CTX_PER_SEQ_DEFAULT: u32 = 2048;

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

// ─── Shared model weights cache ───────────────────────────────────────────────

#[cfg(feature = "gguf")]
struct GgufWeights {
    backend: Arc<LlamaBackend>,
    model: Arc<LlamaModel>,
    n_ctx_train: u32,
}

#[cfg(feature = "gguf")]
static GGUF_WEIGHTS_CACHE: std::sync::OnceLock<
    std::sync::Mutex<
        std::collections::HashMap<std::path::PathBuf, std::sync::Weak<GgufWeights>>,
    >,
> = std::sync::OnceLock::new();

#[cfg(feature = "gguf")]
fn gguf_weights_cache() -> &'static std::sync::Mutex<
    std::collections::HashMap<std::path::PathBuf, std::sync::Weak<GgufWeights>>,
> {
    GGUF_WEIGHTS_CACHE
        .get_or_init(|| std::sync::Mutex::new(std::collections::HashMap::new()))
}

// ─── Scheduler types ──────────────────────────────────────────────────────────

/// A request submitted to the scheduler thread.
#[cfg(feature = "gguf")]
struct GgufRequest {
    prompt: String,
    max_tokens: i32,
    min_tokens: i32,
    response_tx: std_mpsc::Sender<Result<String, EngineError>>,
}

/// A request that has been tokenized and assigned a sequence slot, awaiting prefill.
#[cfg(feature = "gguf")]
struct PendingPrefill {
    seq_id: i32,
    tokens: Vec<LlamaToken>,
    max_tokens: i32,
    min_tokens: i32,
    response_tx: std_mpsc::Sender<Result<String, EngineError>>,
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
    response_tx: std_mpsc::Sender<Result<String, EngineError>>,
}

// ─── Backend ──────────────────────────────────────────────────────────────────

pub struct GgufBackend {
    #[cfg(feature = "gguf")]
    inner: Option<GgufInner>,
    metrics: EngineMetrics,
}

#[cfg(feature = "gguf")]
struct GgufInner {
    _weights: Arc<GgufWeights>,
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
    // n_batch: enough for one full prefill + all active decode tokens.
    let n_batch = ctx_per_seq + max_concurrent as u32;

    let ctx_params = LlamaContextParams::default()
        .with_n_ctx(Some(n_ctx))
        .with_n_batch(n_batch)
        .with_n_seq_max(max_concurrent as u32)
        .with_offload_kqv(true);

    let mut ctx = match model.new_context(&backend, ctx_params) {
        Ok(c) => c,
        Err(e) => {
            log::error!("[gguf] Failed to create shared context: {e}");
            return;
        }
    };
    log::info!("[gguf] Shared context ready (n_ctx={total_ctx}, max_concurrent={max_concurrent})");

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

        // ── 2. Promote waiting → pending (tokenize + assign seq_id) ──────────
        while !waiting.is_empty() && !available_ids.is_empty() {
            let req = waiting.pop_front().unwrap();
            let seq_id = available_ids.pop().unwrap();
            match model.str_to_token(&req.prompt, AddBos::Always) {
                Ok(tokens) => pending.push_back(PendingPrefill {
                    seq_id,
                    tokens,
                    max_tokens: req.max_tokens,
                    min_tokens: req.min_tokens,
                    response_tx: req.response_tx,
                }),
                Err(e) => {
                    let _ = req.response_tx.send(Err(EngineError::backend(format!(
                        "tokenization failed: {e}"
                    ))));
                    available_ids.push(seq_id);
                }
            }
        }

        // ── 3. If completely idle, block for the next request ─────────────────
        if waiting.is_empty() && pending.is_empty() && active.is_empty() {
            match request_rx.recv() {
                Ok(req) => waiting.push_back(req),
                Err(_) => break 'main,
            }
            continue;
        }

        // ── 4. Build batch: one prefill (if any) + all active decode tokens ───
        batch.clear();
        let mut prefill_item: Option<PendingPrefill> = None;
        let mut prefill_last_batch_pos: i32 = -1;

        if let Some(pref) = pending.front() {
            // Check it fits alongside the decode tokens that will follow.
            if pref.tokens.len() + active.len() <= batch_cap {
                let pref = pending.pop_front().unwrap();
                let n = pref.tokens.len();
                let mut add_ok = true;
                for (i, &tok) in pref.tokens.iter().enumerate() {
                    if batch.add(tok, i as i32, &[pref.seq_id], i == n - 1).is_err() {
                        add_ok = false;
                        break;
                    }
                }
                if add_ok {
                    prefill_last_batch_pos = batch.n_tokens() - 1;
                    prefill_item = Some(pref);
                } else {
                    let _ = pref
                        .response_tx
                        .send(Err(EngineError::backend("batch capacity exceeded")));
                    let _ = ctx.clear_kv_cache_seq(Some(pref.seq_id as u32), None, None);
                    available_ids.push(pref.seq_id);
                }
            }
        }

        let mut decode_batch_positions: Vec<i32> = Vec::with_capacity(active.len());
        for seq in &active {
            let pos = batch.n_tokens();
            if batch.add(seq.last_token, seq.pos, &[seq.seq_id], true).is_ok() {
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
                let _ = seq.response_tx.send(Err(EngineError::backend("decode failed")));
                let _ = ctx.clear_kv_cache_seq(Some(seq.seq_id as u32), None, None);
                available_ids.push(seq.seq_id);
            }
            if let Some(pref) = prefill_item {
                let _ = pref
                    .response_tx
                    .send(Err(EngineError::backend("decode failed")));
                let _ = ctx.clear_kv_cache_seq(Some(pref.seq_id as u32), None, None);
                available_ids.push(pref.seq_id);
            }
            continue;
        }

        // ── 6. Sample the newly prefilled sequence and move it to active ──────
        if let Some(pref) = prefill_item {
            let first_tok = LlamaSampler::greedy().sample(&ctx, prefill_last_batch_pos);
            let prompt_len = pref.tokens.len() as i32;

            let is_eos = first_tok == eos_token && 0 >= pref.min_tokens;
            if is_eos || pref.max_tokens <= 0 {
                let _ = ctx.clear_kv_cache_seq(Some(pref.seq_id as u32), None, None);
                available_ids.push(pref.seq_id);
            } else {
                let piece = model
                    .token_to_str(first_tok, Special::Tokenize)
                    .unwrap_or_default();
                if pref.response_tx.send(Ok(piece)).is_err() {
                    // Client dropped
                    let _ = ctx.clear_kv_cache_seq(Some(pref.seq_id as u32), None, None);
                    available_ids.push(pref.seq_id);
                } else if pref.max_tokens <= 1 {
                    // Requested only 1 token — done
                    let _ = ctx.clear_kv_cache_seq(Some(pref.seq_id as u32), None, None);
                    available_ids.push(pref.seq_id);
                } else {
                    active.push(ActiveSeq {
                        seq_id: pref.seq_id,
                        pos: prompt_len,
                        n_generated: 1,
                        max_tokens: pref.max_tokens,
                        min_tokens: pref.min_tokens,
                        last_token: first_tok,
                        response_tx: pref.response_tx,
                    });
                }
            }
        }

        // ── 7. Sample each active sequence and advance or retire it ──────────
        let mut to_retire: Vec<usize> = Vec::new();
        for (i, (seq, &batch_pos)) in active.iter_mut().zip(decode_batch_positions.iter()).enumerate() {
            if batch_pos < 0 {
                continue; // this sequence was not in the batch this step
            }

            let next_tok = LlamaSampler::greedy().sample(&ctx, batch_pos);
            seq.pos += 1;

            let done = (next_tok == eos_token && seq.n_generated >= seq.min_tokens)
                || seq.n_generated >= seq.max_tokens;

            if done {
                to_retire.push(i);
            } else {
                let piece = model
                    .token_to_str(next_tok, Special::Tokenize)
                    .unwrap_or_default();
                if seq.response_tx.send(Ok(piece)).is_err() {
                    to_retire.push(i);
                } else {
                    seq.last_token = next_tok;
                    seq.n_generated += 1;
                }
            }
        }

        for &i in to_retire.iter().rev() {
            let done = active.remove(i);
            let _ = ctx.clear_kv_cache_seq(Some(done.seq_id as u32), None, None);
            available_ids.push(done.seq_id);
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
            let (backend, model, n_ctx_train) = tokio::task::spawn_blocking(move || {
                let backend = LlamaBackend::init().map_err(|e| {
                    EngineError::backend(format!("llama backend init failed: {e}"))
                })?;
                let params = LlamaModelParams::default().with_n_gpu_layers(99);
                let model =
                    LlamaModel::load_from_file(&backend, &model_path_load, &params).map_err(
                        |e| EngineError::backend(format!("GGUF load failed: {e}")),
                    )?;
                let n_ctx_train = model.n_ctx_train();
                Ok::<_, EngineError>((backend, model, n_ctx_train))
            })
            .await
            .map_err(|e| EngineError::backend(format!("spawn_blocking join error: {e}")))??;

            let arc = Arc::new(GgufWeights {
                backend: Arc::new(backend),
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
            _weights: weights,
            request_tx: tx,
            max_concurrent: max_conc,
        });
        Ok(())
    }

    fn infer(&self, request: &InferenceRequest) -> Result<BinaryTensorPacket, EngineError> {
        let inner = self.inner.as_ref().ok_or(EngineError::ModelNotLoaded)?;
        let prompt = Self::extract_prompt(request)?;
        let (resp_tx, resp_rx) = std_mpsc::channel::<Result<String, EngineError>>();

        inner
            .request_tx
            .send(GgufRequest {
                prompt,
                max_tokens: Self::max_new_tokens(request),
                min_tokens: Self::min_new_tokens(request),
                response_tx: resp_tx,
            })
            .map_err(|_| EngineError::backend("gguf scheduler disconnected"))?;

        let mut text = String::new();
        for piece in resp_rx {
            text.push_str(&piece?);
        }

        let data = text.into_bytes();
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

        let (resp_tx, resp_rx) = std_mpsc::channel::<Result<String, EngineError>>();

        if inner
            .request_tx
            .send(GgufRequest {
                prompt,
                max_tokens: Self::max_new_tokens(request),
                min_tokens: Self::min_new_tokens(request),
                response_tx: resp_tx,
            })
            .is_err()
        {
            return Box::pin(stream! {
                yield Err(EngineError::backend("gguf scheduler disconnected"));
            });
        }

        // Bridge blocking std::mpsc → async tokio channel.
        let (tok_tx, mut tok_rx) = tokio::sync::mpsc::channel::<Result<String, EngineError>>(64);
        std::thread::spawn(move || {
            for piece in resp_rx {
                if tok_tx.blocking_send(piece).is_err() {
                    break;
                }
            }
        });

        Box::pin(stream! {
            while let Some(result) = tok_rx.recv().await {
                let piece = result?;
                let data = piece.into_bytes();
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
