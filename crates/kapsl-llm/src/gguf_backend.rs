use crate::prompt_adapter::{chat_template_from_model_identifiers, prompt_is_explicitly_formatted};
use async_stream::stream;
use async_trait::async_trait;
use kapsl_engine_api::{
    BatchingPolicy, BinaryTensorPacket, Engine, EngineError, EngineMetrics, EngineModelInfo,
    EngineStream, ExternalDeviceMemory, ExternalDeviceMemoryReport, InferenceRequest,
    MemoryAllocation, MemoryAllocationClass, MemoryAllocationSource, MemoryDomain, MemoryReport,
    TensorDtype,
};
use std::collections::VecDeque;
use std::num::NonZeroU32;
use std::path::Path;
use std::sync::mpsc as std_mpsc;
#[cfg(feature = "gguf")]
use std::sync::{
    atomic::{AtomicU64, Ordering},
    OnceLock,
};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

#[cfg(feature = "gguf-cuda-shared-kv")]
use cudarc::driver::{CudaDevice, CudaSlice, DevicePtr};
#[cfg(feature = "gguf-cuda-shared-kv")]
use kapsl_hal::cpu_block_store::CpuBlockStore;
#[cfg(feature = "gguf-cuda-shared-kv")]
use kapsl_hal::cross_device_scheduler::CrossDevicePoolScheduler;
#[cfg(feature = "gguf-cuda-shared-kv")]
use kapsl_hal::gpu_arena::{
    GpuBlockPool, GpuDevicePool, GpuPoolHandle, PoolAllocationClass, PoolOwner,
};
#[cfg(feature = "gguf-cuda-shared-kv")]
use kapsl_hal::prefix_cache::PrefixBlockCache;
#[cfg(feature = "gguf")]
use llama_cpp_2::{
    context::params::LlamaContextParams,
    llama_backend::LlamaBackend,
    llama_batch::LlamaBatch,
    model::{params::LlamaModelParams, AddBos, LlamaChatMessage, LlamaModel},
    sampling::LlamaSampler,
    token::{logit_bias::LlamaLogitBias, LlamaToken},
    TokenToStringError,
};
#[cfg(feature = "gguf")]
use llama_cpp_sys_2::LLAMA_FLASH_ATTN_TYPE_AUTO;
#[cfg(feature = "gguf-cuda-shared-kv")]
use llama_cpp_sys_2::{llama_kapsl_kv_pool_desc, LLAMA_KAPSL_KV_DTYPE_F16};

// ─── Configuration ────────────────────────────────────────────────────────────

const MAX_CONCURRENT_DEFAULT: usize = 32;
const N_CTX_PER_SEQ_DEFAULT: u32 = 2048;
const GGUF_N_GPU_LAYERS_ENV: &str = "KAPSL_GGUF_N_GPU_LAYERS";
const GGUF_TARGET_CONCURRENCY_ENV: &str = "KAPSL_GGUF_TARGET_CONCURRENCY";
const GGUF_QUEUE_DELAY_US_DEFAULT: u64 = 1_000;
const GGUF_PREFILL_CHUNK_SIZE_DEFAULT: usize = 512;
const GGUF_TIMING_ENV: &str = "KAPSL_GGUF_TIMING";
const GGUF_TIMING_LOG_EVERY_ENV: &str = "KAPSL_GGUF_TIMING_LOG_EVERY";
const GGUF_TIMING_LOG_EVERY_DEFAULT: u64 = 512;
// llama.cpp sequence-copy asserts unless the context uses a full/unified KV buffer.
// Keep this opt-in until we can detect that mode safely at runtime.
const GGUF_EXACT_PROMPT_KV_REUSE_DEFAULT: bool = false;

#[cfg(feature = "gguf")]
static GGUF_TIMING_BATCH_BUILD_CALLS: AtomicU64 = AtomicU64::new(0);
#[cfg(feature = "gguf")]
static GGUF_TIMING_BATCH_BUILD_US: AtomicU64 = AtomicU64::new(0);
#[cfg(feature = "gguf")]
static GGUF_TIMING_DECODE_CALLS: AtomicU64 = AtomicU64::new(0);
#[cfg(feature = "gguf")]
static GGUF_TIMING_DECODE_US: AtomicU64 = AtomicU64::new(0);
#[cfg(feature = "gguf")]
static GGUF_TIMING_SAMPLE_CALLS: AtomicU64 = AtomicU64::new(0);
#[cfg(feature = "gguf")]
static GGUF_TIMING_SAMPLE_US: AtomicU64 = AtomicU64::new(0);
#[cfg(feature = "gguf")]
static GGUF_TIMING_PIECE_CALLS: AtomicU64 = AtomicU64::new(0);
#[cfg(feature = "gguf")]
static GGUF_TIMING_PIECE_US: AtomicU64 = AtomicU64::new(0);
#[cfg(feature = "gguf")]
static GGUF_TIMING_EMIT_CALLS: AtomicU64 = AtomicU64::new(0);
#[cfg(feature = "gguf")]
static GGUF_TIMING_EMIT_US: AtomicU64 = AtomicU64::new(0);
#[cfg(feature = "gguf")]
static GGUF_TIMING_KV_RESERVE_CALLS: AtomicU64 = AtomicU64::new(0);
#[cfg(feature = "gguf")]
static GGUF_TIMING_KV_RESERVE_US: AtomicU64 = AtomicU64::new(0);
#[cfg(feature = "gguf")]
static GGUF_TIMING_KV_FAST_PATH_CALLS: AtomicU64 = AtomicU64::new(0);
#[cfg(feature = "gguf")]
static GGUF_TIMING_KV_EXTEND_CALLS: AtomicU64 = AtomicU64::new(0);
#[cfg(feature = "gguf")]
static GGUF_TIMING_KV_LOOKUP_CALLS: AtomicU64 = AtomicU64::new(0);
#[cfg(feature = "gguf")]
static GGUF_TIMING_LAST_LOGGED_SAMPLE_CALLS: AtomicU64 = AtomicU64::new(0);

#[cfg(feature = "gguf")]
fn max_concurrent() -> usize {
    if let Some(value) = std::env::var("KAPSL_GGUF_MAX_CONCURRENT")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&v| v > 0)
    {
        return value;
    }

    std::env::var(GGUF_TARGET_CONCURRENCY_ENV)
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
fn gguf_model_params(device_id: usize) -> Result<LlamaModelParams, EngineError> {
    let mut params = LlamaModelParams::default();
    params = match std::env::var(GGUF_N_GPU_LAYERS_ENV)
        .ok()
        .and_then(|v| v.parse::<u32>().ok())
    {
        Some(n_gpu_layers) => params.with_n_gpu_layers(n_gpu_layers),
        None => params,
    };
    #[cfg(feature = "gguf-cuda")]
    {
        let selected = llama_cpp_2::list_llama_ggml_backend_devices()
            .into_iter()
            .filter(|device| {
                device.device_type == llama_cpp_2::LlamaBackendDeviceType::Gpu
                    && device.backend.eq_ignore_ascii_case("cuda")
            })
            .nth(device_id)
            .ok_or_else(|| {
                EngineError::backend(format!(
                    "llama.cpp CUDA device {device_id} is not available"
                ))
            })?;
        params = params
            .with_devices(&[selected.index])
            .map_err(|error| EngineError::backend(format!("select llama.cpp device: {error}")))?;
    }
    #[cfg(not(feature = "gguf-cuda"))]
    let _ = device_id;
    Ok(params)
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

#[cfg(feature = "gguf")]
fn gguf_timing_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var(GGUF_TIMING_ENV)
            .ok()
            .map(|v| {
                let v = v.trim().to_ascii_lowercase();
                !matches!(v.as_str(), "" | "0" | "false" | "no" | "off")
            })
            .unwrap_or(false)
    })
}

#[cfg(feature = "gguf")]
fn gguf_timing_log_every() -> u64 {
    static LOG_EVERY: OnceLock<u64> = OnceLock::new();
    *LOG_EVERY.get_or_init(|| {
        std::env::var(GGUF_TIMING_LOG_EVERY_ENV)
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .filter(|&v| v > 0)
            .unwrap_or(GGUF_TIMING_LOG_EVERY_DEFAULT)
    })
}

#[cfg(feature = "gguf")]
fn gguf_timing_elapsed_us(start: Instant) -> u64 {
    start.elapsed().as_micros().min(u128::from(u64::MAX)) as u64
}

#[cfg(feature = "gguf")]
fn gguf_timing_record(calls: &AtomicU64, micros: &AtomicU64, start: Instant) {
    calls.fetch_add(1, Ordering::Relaxed);
    micros.fetch_add(gguf_timing_elapsed_us(start), Ordering::Relaxed);
}

#[cfg(feature = "gguf")]
fn gguf_prepare_prompt(model: &LlamaModel, prompt: String) -> Result<String, EngineError> {
    if prompt.trim().is_empty() || prompt_is_explicitly_formatted(&prompt) {
        return Ok(prompt);
    }

    let template = match model.chat_template(None) {
        Ok(template) => template,
        Err(_) => {
            let identifiers: Vec<String> = ["general.name", "general.architecture"]
                .iter()
                .filter_map(|key| model.meta_val_str(key).ok())
                .collect();
            if let Some(template) =
                chat_template_from_model_identifiers(identifiers.iter().map(String::as_str))
            {
                return Ok(template.render(&prompt));
            }
            return Ok(prompt);
        }
    };
    let message = LlamaChatMessage::new("user".to_string(), prompt.clone())
        .map_err(|e| EngineError::backend(format!("chat message build failed: {e}")))?;

    model
        .apply_chat_template(&template, &[message], true)
        .map_err(|e| EngineError::backend(format!("chat template apply failed: {e}")))
}

#[cfg(feature = "gguf")]
struct GgufTimingGuard {
    calls: &'static AtomicU64,
    micros: &'static AtomicU64,
    start: Option<Instant>,
}

#[cfg(feature = "gguf")]
impl GgufTimingGuard {
    fn new(calls: &'static AtomicU64, micros: &'static AtomicU64) -> Self {
        Self {
            calls,
            micros,
            start: gguf_timing_enabled().then(Instant::now),
        }
    }
}

#[cfg(feature = "gguf")]
impl Drop for GgufTimingGuard {
    fn drop(&mut self) {
        if let Some(start) = self.start.take() {
            gguf_timing_record(self.calls, self.micros, start);
        }
    }
}

#[cfg(feature = "gguf")]
fn gguf_timing_avg_us(calls: u64, micros: u64) -> f64 {
    if calls == 0 {
        0.0
    } else {
        micros as f64 / calls as f64
    }
}

#[cfg(feature = "gguf")]
fn gguf_timing_maybe_log(force: bool) {
    if !gguf_timing_enabled() {
        return;
    }

    let sample_calls = GGUF_TIMING_SAMPLE_CALLS.load(Ordering::Relaxed);
    if sample_calls == 0 {
        return;
    }

    let log_every = gguf_timing_log_every();
    let last = GGUF_TIMING_LAST_LOGGED_SAMPLE_CALLS.load(Ordering::Relaxed);
    if !force && sample_calls / log_every == last / log_every {
        return;
    }
    GGUF_TIMING_LAST_LOGGED_SAMPLE_CALLS.store(sample_calls, Ordering::Relaxed);

    let batch_calls = GGUF_TIMING_BATCH_BUILD_CALLS.load(Ordering::Relaxed);
    let batch_us = GGUF_TIMING_BATCH_BUILD_US.load(Ordering::Relaxed);
    let decode_calls = GGUF_TIMING_DECODE_CALLS.load(Ordering::Relaxed);
    let decode_us = GGUF_TIMING_DECODE_US.load(Ordering::Relaxed);
    let sample_us = GGUF_TIMING_SAMPLE_US.load(Ordering::Relaxed);
    let piece_calls = GGUF_TIMING_PIECE_CALLS.load(Ordering::Relaxed);
    let piece_us = GGUF_TIMING_PIECE_US.load(Ordering::Relaxed);
    let emit_calls = GGUF_TIMING_EMIT_CALLS.load(Ordering::Relaxed);
    let emit_us = GGUF_TIMING_EMIT_US.load(Ordering::Relaxed);
    let kv_calls = GGUF_TIMING_KV_RESERVE_CALLS.load(Ordering::Relaxed);
    let kv_us = GGUF_TIMING_KV_RESERVE_US.load(Ordering::Relaxed);
    let kv_fast = GGUF_TIMING_KV_FAST_PATH_CALLS.load(Ordering::Relaxed);
    let kv_extend = GGUF_TIMING_KV_EXTEND_CALLS.load(Ordering::Relaxed);
    let kv_lookup = GGUF_TIMING_KV_LOOKUP_CALLS.load(Ordering::Relaxed);

    eprintln!(
        "[gguf-timing] calls batch={} decode={} sample={} piece={} emit={} kv={} \
         avg_us batch={:.1} decode={:.1} sample={:.1} piece={:.1} emit={:.1} kv={:.1} \
         kv_paths fast={} extend={} lookup={}",
        batch_calls,
        decode_calls,
        sample_calls,
        piece_calls,
        emit_calls,
        kv_calls,
        gguf_timing_avg_us(batch_calls, batch_us),
        gguf_timing_avg_us(decode_calls, decode_us),
        gguf_timing_avg_us(sample_calls, sample_us),
        gguf_timing_avg_us(piece_calls, piece_us),
        gguf_timing_avg_us(emit_calls, emit_us),
        gguf_timing_avg_us(kv_calls, kv_us),
        kv_fast,
        kv_extend,
        kv_lookup,
    );
}

#[cfg(feature = "gguf")]
#[derive(Clone, Copy, Debug)]
struct GgufServingConfig {
    max_concurrent: usize,
    ctx_per_seq: u32,
    queue_delay: Duration,
    prefill_chunk_size: usize,
    exact_prompt_kv_reuse: bool,
    kv_bytes_per_cell: usize,
    /// True when the model's memory includes recurrent/SSM state (Mamba,
    /// RWKV, or a hybrid mix like Jamba/Granite) rather than being a pure
    /// per-token attention KV cache. `update_gguf_metrics` uses this to avoid
    /// reporting KV pressure that scales with generated token count: real
    /// recurrent-state footprint is ~constant per active sequence, not
    /// proportional to how far each sequence has decoded. These models always
    /// run on llama.cpp's native memory (the shared-KV pool guard rejects
    /// them unconditionally), regardless of which KV backend feature is
    /// compiled in.
    uses_state_space_memory: bool,
}

#[cfg(feature = "gguf")]
impl GgufServingConfig {
    fn from_model(model: &LlamaModel, n_ctx_train: u32) -> Self {
        let max_concurrent = max_concurrent();
        let ctx_per_seq = n_ctx_per_seq().min(n_ctx_train);
        let prefill_chunk_size = gguf_prefill_chunk_size().min(ctx_per_seq as usize).max(1);
        Self {
            max_concurrent,
            ctx_per_seq,
            queue_delay: gguf_queue_delay(),
            prefill_chunk_size,
            exact_prompt_kv_reuse: gguf_exact_prompt_kv_reuse(),
            kv_bytes_per_cell: estimate_kv_bytes_per_cell(model),
            uses_state_space_memory: gguf_model_uses_state_space_memory(model),
        }
    }

    fn total_ctx(self) -> usize {
        self.max_concurrent * self.ctx_per_seq as usize
    }

    fn n_batch(self) -> u32 {
        self.prefill_chunk_size
            .saturating_add(self.max_concurrent.max(1)) as u32
    }
}

#[cfg(feature = "gguf")]
fn estimate_kv_bytes_per_cell(model: &LlamaModel) -> usize {
    let n_head_kv = model.n_head_kv().max(1) as usize;
    let head_dim_k = model.n_embd_head_k().max(1) as usize;
    let head_dim_v = model.n_embd_head_v().max(1) as usize;

    // llama.cpp reports this path as K/V f16 in the load log. Treat one KV cell
    // as K + V for every layer so Prometheus/admission logic has a comparable
    // capacity signal to the native/ONNX paths.
    model.n_layer().max(1) as usize
        * n_head_kv
        * (head_dim_k + head_dim_v)
        * std::mem::size_of::<u16>()
}

#[cfg(feature = "gguf-cuda-shared-kv")]
fn gguf_shared_kv_block_count(
    n_layers: usize,
    config: GgufServingConfig,
    block_size: usize,
    windowed: Option<&GgufWindowedKvConfig>,
) -> usize {
    if let Some(blocks) = std::env::var("KAPSL_GGUF_CUDA_SHARED_KV_POOL_BLOCKS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&v| v > 0)
    {
        return blocks.max(n_layers);
    }
    let blocks_per_seq = (config.ctx_per_seq as usize)
        .div_ceil(block_size.max(1))
        .max(1);
    // Windowed SWA layers never hold more than their ring, so the pool can be
    // sized for it — this is where the Phase 2 memory saving is realized.
    let per_seq_blocks: usize = match windowed {
        Some(w) => (0..n_layers)
            .map(|il| windowed_layer_capacity(blocks_per_seq, w.layer_window(il)))
            .sum(),
        None => n_layers.saturating_mul(blocks_per_seq),
    };
    per_seq_blocks
        .saturating_mul(config.max_concurrent.max(1))
        .max(n_layers)
}

// ─── SWA windowed KV (Phase 2) ────────────────────────────────────────────────

/// Windowed KV allocation for sliding-window models (Phase 2 of SWA on
/// shared-KV). Instead of allocating `ctx_per_seq` worth of blocks on every
/// layer, each SWA layer gets a fixed ring of `window_blocks` physical blocks;
/// its block-table row maps logical block `P` to ring slot `P % window_blocks`,
/// so a block is recycled exactly when every position in it has fallen out of
/// every live query's attention window. Full-attention layers are unchanged.
///
/// The paged-attention kernel guarantees it never dereferences a block-table
/// entry below the window start (see `kapsl_swa_window_start` in kapsl-kv.cu),
/// which is what makes recycling safe.
#[cfg(feature = "gguf-cuda-shared-kv")]
struct GgufWindowedKvConfig {
    /// Ring capacity, in blocks, for every sliding-window layer.
    window_blocks: usize,
    /// Per-layer flag: `true` = SWA layer (ring-mapped), `false` = full attention.
    swa_layers: Vec<bool>,
}

#[cfg(feature = "gguf-cuda-shared-kv")]
impl GgufWindowedKvConfig {
    /// `Some(window_blocks)` when layer `il` is ring-mapped, `None` when full.
    fn layer_window(&self, il: usize) -> Option<usize> {
        self.swa_layers
            .get(il)
            .copied()
            .unwrap_or(false)
            .then_some(self.window_blocks)
    }
}

/// Number of physical blocks a sliding-window layer's KV ring must hold.
///
/// A query at position `p` reads keys in `[win_start(p), p]` with
/// `win_start(p) >= p - n_swa + 1` for every supported window type, and one
/// ubatch writes up to `n_ubatch` positions before any of its queries run
/// attention, so `n_swa + n_ubatch` positions must stay live simultaneously
/// (the same bound llama.cpp's native iSWA cache uses for its SWA buffer).
/// The `+ 1` covers block-boundary straddle at both ends.
///
/// Only the shared-KV pool sizes rings, so outside that feature this is
/// compiled solely for its unit tests.
#[cfg(any(feature = "gguf-cuda-shared-kv", test))]
fn swa_window_blocks(n_swa: usize, n_ubatch: usize, block_size: usize) -> usize {
    (n_swa + n_ubatch).div_ceil(block_size.max(1)) + 1
}

/// Physical blocks a layer needs to cover `logical` logical blocks: capped at
/// the ring size on windowed layers, uncapped on full-attention layers.
#[cfg(any(feature = "gguf-cuda-shared-kv", test))]
fn windowed_layer_capacity(logical: usize, layer_window: Option<usize>) -> usize {
    match layer_window {
        Some(window_blocks) => logical.min(window_blocks),
        None => logical,
    }
}

/// Build the windowed-KV config for a loaded model, or `None` when windowing
/// is disabled, inapplicable, or would not save memory.
///
/// Opt-in via `KAPSL_GGUF_SWA_WINDOWED_KV` (on top of the Phase 1
/// `KAPSL_GGUF_ENABLE_SWA_SHARED_KV` gate — a SWA model only reaches shared-KV
/// at all when that admitted it).
#[cfg(feature = "gguf-cuda-shared-kv")]
fn gguf_windowed_kv_config(
    model: &LlamaModel,
    config: GgufServingConfig,
    block_size: usize,
) -> Option<GgufWindowedKvConfig> {
    if !gguf_env_is_truthy("KAPSL_GGUF_SWA_WINDOWED_KV") {
        return None;
    }
    let n_swa = model.n_swa() as usize;
    if n_swa == 0 {
        return None;
    }
    let n_layers = model.n_layer().max(1) as usize;
    let swa_layers: Vec<bool> = (0..n_layers)
        .map(|il| model.is_swa_layer(il as u32))
        .collect();
    let n_swa_layers = swa_layers.iter().filter(|&&s| s).count();
    if n_swa_layers == 0 {
        return None;
    }
    let window_blocks = swa_window_blocks(n_swa, config.n_batch() as usize, block_size);
    let blocks_per_seq = (config.ctx_per_seq as usize)
        .div_ceil(block_size.max(1))
        .max(1);
    if window_blocks >= blocks_per_seq {
        log::info!(
            "[gguf] SWA windowed KV requested but the ring ({window_blocks} blocks incl. \
             ubatch slack) is not smaller than ctx_per_seq ({blocks_per_seq} blocks); \
             keeping full allocation"
        );
        return None;
    }
    log::info!(
        "[gguf] SWA windowed KV enabled: n_swa={n_swa}, swa_layers={n_swa_layers}/{n_layers}, \
         ring={window_blocks} blocks/layer vs {blocks_per_seq} full (per-seq KV ~{}% of uniform)",
        ((n_layers - n_swa_layers) * blocks_per_seq + n_swa_layers * window_blocks) * 100
            / (n_layers * blocks_per_seq).max(1)
    );
    Some(GgufWindowedKvConfig {
        window_blocks,
        swa_layers,
    })
}

// ─── Shared model weights cache ───────────────────────────────────────────────

#[cfg(feature = "gguf")]
struct GgufWeights {
    backend: Arc<LlamaBackend>,
    model: Arc<LlamaModel>,
    n_ctx_train: u32,
    allocation_id: String,
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

#[cfg(feature = "gguf")]
fn gguf_model_key(model_path: &Path) -> std::path::PathBuf {
    std::fs::canonicalize(model_path).unwrap_or_else(|_| model_path.to_path_buf())
}

#[cfg(feature = "gguf")]
fn gguf_allocation_id(model_path: &Path) -> String {
    format!("llama-gguf:{}", gguf_model_key(model_path).display())
}

// Global LlamaBackend singleton — llama.cpp allows only one backend per process.
#[cfg(feature = "gguf")]
static GGUF_BACKEND: std::sync::OnceLock<Arc<LlamaBackend>> = std::sync::OnceLock::new();

#[cfg(feature = "gguf")]
static GGUF_BACKEND_INIT_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

// ─── Cross-device KV scheduler singleton ──────────────────────────────────────

#[cfg(feature = "gguf-cuda-shared-kv")]
static GGUF_KV_SCHEDULER: std::sync::OnceLock<std::sync::Mutex<CrossDevicePoolScheduler>> =
    std::sync::OnceLock::new();

#[cfg(feature = "gguf-cuda-shared-kv")]
fn gguf_global_kv_scheduler() -> &'static std::sync::Mutex<CrossDevicePoolScheduler> {
    GGUF_KV_SCHEDULER.get_or_init(|| {
        let evict_threshold = std::env::var("KAPSL_GGUF_EVICT_THRESHOLD")
            .ok()
            .and_then(|v| v.parse::<f32>().ok())
            .filter(|&v| v > 0.0 && v < 1.0)
            .unwrap_or(0.85);
        let cpu_slots = std::env::var("KAPSL_GGUF_CROSS_DEVICE_CPU_SLOTS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|&v| v > 0)
            .unwrap_or(512);
        std::sync::Mutex::new(CrossDevicePoolScheduler::new(evict_threshold, cpu_slots))
    })
}

/// Pick the registered device with the most free KV blocks for the given geometry.
/// Falls back to `fallback_device_id` when `KAPSL_GGUF_AUTO_DEVICE` is off or when
/// no device is registered yet (e.g. the very first pool construction).
#[cfg(feature = "gguf-cuda-shared-kv")]
fn gguf_select_device(fallback_device_id: usize, kv_heads: usize, head_dim: usize) -> usize {
    if !std::env::var("KAPSL_GGUF_AUTO_DEVICE")
        .map(|v| {
            !matches!(
                v.trim().to_ascii_lowercase().as_str(),
                "0" | "false" | "no" | "off"
            )
        })
        .unwrap_or(false)
    {
        return fallback_device_id;
    }
    let sched = gguf_global_kv_scheduler().lock().unwrap();
    let devices = sched.registered_devices();
    if devices.is_empty() {
        return fallback_device_id;
    }
    devices
        .into_iter()
        .max_by_key(|&d| sched.device_free_blocks(d, kv_heads, head_dim))
        .unwrap_or(fallback_device_id)
}

/// Parse an environment variable using the same truthiness rules as the other
/// `KAPSL_GGUF_*` boolean toggles: unset, empty, `0`, `false`, `no`, `off`
/// (case-insensitive) are false; anything else is true.
#[cfg(feature = "gguf")]
fn gguf_env_is_truthy(name: &str) -> bool {
    std::env::var(name)
        .map(|v| {
            !matches!(
                v.trim().to_ascii_lowercase().as_str(),
                "" | "0" | "false" | "no" | "off"
            )
        })
        .unwrap_or(false)
}

/// Decide whether the Kapsl shared-KV (paged external KV) path can serve a
/// loaded GGUF model. Returns `Some(reason)` when it cannot and the engine must
/// fall back to llama.cpp's native KV cache instead.
///
/// The shared-KV ABI (`llama_kapsl_kv_pool_desc`) describes a *single uniform*
/// KV geometry: one `num_kv_heads`/`head_dim`, a separate-K/V `[2, …]` block
/// layout, and full causal attention only. Architectures that violate those
/// assumptions either hard-abort inside llama.cpp — `create_memory()` runs
/// `GGML_ASSERT(hparams.swa_type == LLAMA_SWA_TYPE_NONE)` when a kapsl pool is
/// attached — or silently bypass the external pool (recurrent / hybrid memory),
/// wasting the allocation. We detect those here, before the pool is built, via
/// architecture-scoped GGUF metadata so the model runs correctly on the native
/// path rather than crashing or producing a mislabeled benchmark.
#[cfg(feature = "gguf-cuda-shared-kv")]
fn gguf_shared_kv_disable_reason(model: &LlamaModel) -> Option<String> {
    // Operator override — this is the real wiring for the flag the benchmark
    // scripts already pass (previously a no-op read by nothing).
    if gguf_env_is_truthy("KAPSL_GGUF_DISABLE_SHARED_KV") {
        return Some("disabled by KAPSL_GGUF_DISABLE_SHARED_KV".to_string());
    }

    // Escape hatch for deliberately exercising the shared-KV path on an
    // architecture this guard would otherwise reject (e.g. measuring the
    // sliding-window abort, or a model whose metadata trips the conservative
    // detection below). Opt-in only; may abort or produce wrong output.
    if gguf_env_is_truthy("KAPSL_GGUF_FORCE_SHARED_KV") {
        log::warn!(
            "[gguf] KAPSL_GGUF_FORCE_SHARED_KV set: bypassing shared-KV architecture guard; \
             this may abort or yield incorrect output on unsupported models"
        );
        return None;
    }

    let arch = model
        .meta_val_str("general.architecture")
        .unwrap_or_default();
    let has_key = |suffix: &str| model.meta_val_str(&format!("{arch}.{suffix}")).is_ok();
    let pos_int = |suffix: &str| {
        model
            .meta_val_str(&format!("{arch}.{suffix}"))
            .ok()
            .and_then(|v| v.trim().parse::<i64>().ok())
            .is_some_and(|n| n > 0)
    };

    // Phase 1: the kapsl paged-attention kernel now applies a per-layer
    // sliding-window mask, so SWA models can run on the shared-KV path. This is
    // opt-in until GPU-verified — by default SWA still falls back to the native
    // cache. Recurrent/hybrid and MLA remain unconditionally unsupported.
    let allow_swa = gguf_env_is_truthy("KAPSL_GGUF_ENABLE_SWA_SHARED_KV");

    // `n_swa` is the authoritative post-load window size: it is populated by
    // llama.cpp's per-architecture hparam loader regardless of which GGUF key
    // (or hard-coded default) the window came from, so it catches models the
    // raw-metadata heuristics below would miss (see `classify_shared_kv_support`).
    classify_shared_kv_support(&arch, model.n_swa(), allow_swa, has_key, pos_int)
}

/// Architectures whose sliding-window attention is quality-verified on the
/// shared-KV paged path (sampled eval vs native shows no regression). The Gemma
/// family is all-RoPE ISWA: gemma2/gemma3 are eval-verified; gemma3n/gemma4 share
/// the same structure. Cohere2 (and other SWA families) are excluded — cohere2's
/// NoPE global-attention layers degenerate on the paged path, and the rest are
/// not yet eval-checked. Extend this as more families are verified.
#[cfg(any(feature = "gguf-cuda-shared-kv", test))]
fn swa_shared_kv_arch_verified(arch: &str) -> bool {
    matches!(arch, "gemma2" | "gemma3" | "gemma3n" | "gemma4")
}

/// Pure classification core of [`gguf_shared_kv_disable_reason`], separated so it
/// can be unit-tested without a loaded model. Returns `Some(reason)` when the
/// uniform shared-KV pool cannot represent the model and the engine must fall
/// back to llama.cpp's native KV cache.
///
/// `n_swa` is the loaded model's sliding-window size (`0` for full attention);
/// `allow_swa` opts sliding-window models onto the shared-KV path (Phase 1,
/// gated by `KAPSL_GGUF_ENABLE_SWA_SHARED_KV`); `has_key`/`pos_int` look up
/// architecture-scoped GGUF metadata (already prefixed with `<arch>.` by the
/// caller) — presence, and positive-integer value, respectively.
#[cfg(any(feature = "gguf-cuda-shared-kv", test))]
fn classify_shared_kv_support(
    arch: &str,
    n_swa: u32,
    allow_swa: bool,
    has_key: impl Fn(&str) -> bool,
    pos_int: impl Fn(&str) -> bool,
) -> Option<String> {
    if arch.is_empty() {
        // Without a known architecture we cannot vet the KV geometry; stay on
        // the safe native path rather than risk a mismatch or abort.
        return Some("unknown architecture (no general.architecture metadata)".to_string());
    }

    // 1. Sliding-window / ISWA / chunked attention (Gemma 2/3, Phi-3-SWA,
    //    Cohere2, Llama 4, gpt-oss, …). The kapsl paged-attention kernel applies
    //    the per-layer window mask (Phase 1), but this is opt-in until verified;
    //    when disabled, keep these off the pool (llama.cpp would otherwise abort
    //    on `swa_type == NONE`).
    //
    //    Prefer the authoritative `n_swa` over the metadata key: architectures
    //    such as Llama 4 hard-code a non-zero window (swa_type = CHUNKED,
    //    n_swa = 8192) when `attention.sliding_window` is absent, so the key
    //    check alone lets them slip through.
    let is_swa = n_swa > 0 || pos_int("attention.sliding_window");
    if is_swa {
        if !allow_swa {
            // Opt-in disabled: keep every SWA model on the native cache.
            return Some(format!(
                "sliding-window/chunked attention (arch={arch}, n_swa={n_swa})"
            ));
        }
        if !swa_shared_kv_arch_verified(arch) {
            // Opt-in enabled but this architecture is not on the quality-verified
            // allowlist. Cohere2 (NoPE global layers) regresses on the paged path
            // and other SWA families are not yet eval-checked, so they stay native
            // even with the flag set. Use KAPSL_GGUF_FORCE_SHARED_KV to override.
            return Some(format!(
                "sliding-window attention on unverified arch for shared-KV (arch={arch})"
            ));
        }
    }

    // 2. Multi-head latent attention (DeepSeek-V2/V3-style): the compressed
    //    latent KV does not fit the [2, num_kv_heads, head_dim] pool layout.
    if has_key("attention.kv_lora_rank") {
        return Some(format!("multi-head latent attention (arch={arch})"));
    }

    // 3. Recurrent / hybrid state-space memory (Mamba, RWKV, Jamba, Falcon-H1,
    //    Nemotron-H, …): these route through llama.cpp's recurrent / hybrid
    //    memory and ignore the external pool entirely. Detected by the presence
    //    of SSM / WKV metadata rather than an exact architecture allowlist.
    if gguf_uses_state_space_memory(&has_key) {
        return Some(format!("recurrent/hybrid state-space memory (arch={arch})"));
    }

    None
}

/// True when the model's memory is (partly or fully) recurrent state-space
/// (Mamba/RWKV) rather than a pure growing per-token attention KV cache —
/// detected via the same SSM/WKV metadata keys llama.cpp's own hparam loader
/// requires to build any of these architectures (Mamba, Mamba2, RWKV6/7,
/// Jamba, Falcon-H1, Nemotron-H, Granite-hybrid, …), so this generalizes
/// without an exact architecture allowlist. `has_key` takes an
/// architecture-scoped GGUF metadata suffix, matching the shared-KV guard's
/// convention.
///
/// Available whenever GGUF serving is compiled in (not gated behind the CUDA
/// shared-KV pool feature): these models always run on llama.cpp's native
/// memory regardless of which KV backend feature is enabled, so callers that
/// need to reason about their memory shape (e.g. KV metrics) need this
/// outside the shared-KV-only code paths too.
#[cfg(feature = "gguf")]
fn gguf_uses_state_space_memory(has_key: impl Fn(&str) -> bool) -> bool {
    has_key("ssm.state_size") || has_key("ssm.conv_kernel") || has_key("wkv.head_size")
}

/// [`gguf_uses_state_space_memory`] applied to a loaded model's own
/// architecture-scoped metadata.
#[cfg(feature = "gguf")]
fn gguf_model_uses_state_space_memory(model: &LlamaModel) -> bool {
    let arch = model
        .meta_val_str("general.architecture")
        .unwrap_or_default();
    let has_key = |suffix: &str| model.meta_val_str(&format!("{arch}.{suffix}")).is_ok();
    gguf_uses_state_space_memory(has_key)
}

// ─── SSM recurrent-state checkpoint cache (Phase 4) ──────────────────────────

/// Host-RAM cache of per-sequence recurrent-state snapshots, taken at prefill
/// chunk boundaries and keyed by the chained hash of the token prefix they
/// cover. The SSM analog of the attention prefix cache: recurrent models
/// collapse their whole context into a fixed-size state, so a snapshot at
/// position N lets a later request that shares the first N prompt tokens skip
/// prefilling them — `llama_state_seq_set_data` restores the state and prefill
/// resumes at N.
///
/// Two properties make this safe:
/// - the state at N depends only on tokens 0..N (never on sampling params), so
///   the chained token hash fully identifies it;
/// - a checkpoint is only reused when N <= prompt_len - 1, so at least one
///   prompt token is always decoded afterwards — that decode produces the
///   logits for sampling the first output token (restoring a state gives no
///   logits, and re-decoding an already-absorbed token would corrupt the
///   state).
///
/// Entries hold whatever `llama_state_seq_get_data` serializes: a small
/// constant-size state for pure-recurrent models (Mamba/RWKV ~tens of MB), but
/// state + per-token attention KV for hybrids — `max_entry_bytes` keeps
/// oversized hybrid snapshots out of the cache.
#[cfg(feature = "gguf")]
struct SsmStateCache {
    /// chain-hash of the covered token prefix -> snapshot.
    entries: std::collections::HashMap<u64, SsmStateEntry>,
    /// session_id -> resume snapshots. Unlike chunk checkpoints these sit at
    /// arbitrary positions, so they are keyed by session and matched by exact
    /// token-prefix comparison instead of chunk-boundary hashes. Each session
    /// keeps up to [`SSM_SESSION_SLOTS`] entries — the post-prompt state and
    /// the post-retirement state (prompt + generated reply). The retirement
    /// entry skips the most, but only matches when the client's resent history
    /// retokenizes to exactly the tokens the model produced; the reply seam
    /// often drifts, and the post-prompt entry survives that (the next turn
    /// begins with the previous prompt text verbatim).
    sessions: std::collections::HashMap<String, Vec<SsmSessionEntry>>,
    total_bytes: usize,
    max_bytes: usize,
    max_entry_bytes: usize,
    /// Checkpoint stride in tokens (= the prefill chunk size, so snapshots
    /// align with the positions prefill naturally pauses at).
    chunk: usize,
    /// Monotonic tick for LRU accounting.
    tick: u64,
}

#[cfg(feature = "gguf")]
struct SsmStateEntry {
    /// Number of prompt tokens the snapshot has absorbed.
    n_tokens: usize,
    data: Vec<u8>,
    last_used: u64,
}

/// Resume snapshots retained per session: the post-prompt state and the
/// post-retirement state.
#[cfg(feature = "gguf")]
const SSM_SESSION_SLOTS: usize = 2;

#[cfg(feature = "gguf")]
struct SsmSessionEntry {
    /// The exact tokens the state has absorbed (prompt, or prompt + decoded
    /// generation). Token-compare beats hashing here: retokenized history can
    /// diverge from generation-time tokens at the seam, and an exact compare
    /// degrades that to a clean miss instead of a wrong-state restore.
    tokens: Vec<LlamaToken>,
    data: Vec<u8>,
    last_used: u64,
}

#[cfg(feature = "gguf")]
impl SsmStateCache {
    fn new(chunk: usize, max_bytes: usize, max_entry_bytes: usize) -> Self {
        Self {
            entries: std::collections::HashMap::new(),
            sessions: std::collections::HashMap::new(),
            total_bytes: 0,
            max_bytes,
            max_entry_bytes,
            chunk: chunk.max(1),
            tick: 0,
        }
    }

    /// Chained hashes at every chunk boundary of `tokens`; `hashes[i]` covers
    /// tokens `0..(i+1)*chunk`, and each hash folds in its predecessor so a
    /// hash identifies the entire prefix, not just its own chunk. Local
    /// implementation (rather than `PrefixBlockCache`'s) because kapsl-hal's
    /// prefix cache is CUDA-gated while this cache also serves CPU/Metal
    /// builds; no fingerprint is needed since the cache is scheduler-local and
    /// only ever holds one model's states.
    fn prefix_hashes(&self, tokens: &[LlamaToken]) -> Vec<u64> {
        use std::hash::{Hash, Hasher};
        let n_chunks = tokens.len() / self.chunk;
        let mut hashes = Vec::with_capacity(n_chunks);
        let mut prev: u64 = 0;
        for c in 0..n_chunks {
            let mut h = std::collections::hash_map::DefaultHasher::new();
            prev.hash(&mut h);
            for t in &tokens[c * self.chunk..(c + 1) * self.chunk] {
                t.0.hash(&mut h);
            }
            prev = h.finish();
            hashes.push(prev);
        }
        hashes
    }

    /// Longest cached checkpoint covering a strict prefix of a `prompt_len`
    /// prompt (`n_tokens <= prompt_len - 1`), as `(hash, n_tokens)`. Bumps the
    /// entry's LRU tick.
    fn lookup_longest(&mut self, hashes: &[u64], prompt_len: usize) -> Option<(u64, usize)> {
        for (i, &hash) in hashes.iter().enumerate().rev() {
            let n_tokens = (i + 1) * self.chunk;
            if n_tokens >= prompt_len {
                continue;
            }
            if let Some(entry) = self.entries.get_mut(&hash) {
                debug_assert_eq!(entry.n_tokens, n_tokens);
                self.tick += 1;
                entry.last_used = self.tick;
                return Some((hash, entry.n_tokens));
            }
        }
        None
    }

    fn data(&self, hash: u64) -> Option<&[u8]> {
        self.entries.get(&hash).map(|e| e.data.as_slice())
    }

    fn contains(&self, hash: u64) -> bool {
        self.entries.contains_key(&hash)
    }

    /// Insert a snapshot, evicting least-recently-used entries until it fits.
    /// Oversized or unfittable snapshots are dropped.
    fn insert(&mut self, hash: u64, n_tokens: usize, data: Vec<u8>) {
        if data.is_empty() || data.len() > self.max_entry_bytes || data.len() > self.max_bytes {
            return;
        }
        self.tick += 1;
        if let Some(existing) = self.entries.get_mut(&hash) {
            existing.last_used = self.tick;
            return;
        }
        if !self.make_room(data.len()) {
            return;
        }
        self.total_bytes += data.len();
        self.entries.insert(
            hash,
            SsmStateEntry {
                n_tokens,
                data,
                last_used: self.tick,
            },
        );
    }

    fn remove(&mut self, hash: u64) {
        if let Some(entry) = self.entries.remove(&hash) {
            self.total_bytes -= entry.data.len();
        }
    }

    /// The session whose stored token stream is a strict prefix of `prompt`,
    /// as its absorbed-token count. Exact comparison — a divergent history is
    /// a clean miss. Bumps the entry's LRU tick.
    /// The longest of the session's snapshots whose stored token stream is a
    /// strict prefix of `prompt`, as `(entry_index, absorbed_len)`. Exact
    /// comparison — a divergent history is a clean miss. Bumps the winner's
    /// LRU tick.
    fn session_match(&mut self, session_id: &str, prompt: &[LlamaToken]) -> Option<(usize, usize)> {
        let entries = self.sessions.get_mut(session_id)?;
        let mut best: Option<(usize, usize)> = None;
        for (idx, entry) in entries.iter().enumerate() {
            let n = entry.tokens.len();
            if n == 0 || n >= prompt.len() {
                continue;
            }
            if prompt[..n] != entry.tokens[..] {
                // Usually retokenization drift: the client-side history text
                // tokenizes differently from the tokens the model actually
                // produced, most often right at the reply seam. The divergence
                // index tells that apart from a real mismatch when debugging.
                let divergence = prompt
                    .iter()
                    .zip(&entry.tokens)
                    .position(|(a, b)| a != b)
                    .unwrap_or(n);
                log::debug!(
                    "[gguf] SSM session miss: session={session_id} stored={n} \
                     prompt={} first_divergence={divergence}",
                    prompt.len(),
                );
                continue;
            }
            if best.is_none_or(|(_, best_n)| n > best_n) {
                best = Some((idx, n));
            }
        }
        let (idx, n) = best?;
        self.tick += 1;
        entries[idx].last_used = self.tick;
        Some((idx, n))
    }

    fn session_data(&self, session_id: &str, idx: usize) -> Option<&[u8]> {
        self.sessions
            .get(session_id)
            .and_then(|entries| entries.get(idx))
            .map(|e| e.data.as_slice())
    }

    /// True when the session already stores a snapshot for exactly `tokens` —
    /// lets the caller skip the device→host copy for an identical re-insert.
    fn session_is_current(&self, session_id: &str, tokens: &[LlamaToken]) -> bool {
        self.sessions
            .get(session_id)
            .is_some_and(|entries| entries.iter().any(|e| e.tokens == tokens))
    }

    /// Insert a session resume snapshot, keeping at most [`SSM_SESSION_SLOTS`]
    /// per session (the oldest is dropped once full).
    fn insert_session(&mut self, session_id: &str, tokens: Vec<LlamaToken>, data: Vec<u8>) {
        if data.is_empty()
            || tokens.is_empty()
            || data.len() > self.max_entry_bytes
            || data.len() > self.max_bytes
        {
            return;
        }
        self.tick += 1;
        let tick = self.tick;
        if let Some(entries) = self.sessions.get_mut(session_id) {
            if let Some(existing) = entries.iter_mut().find(|e| e.tokens == tokens) {
                existing.last_used = tick;
                return;
            }
            while entries.len() >= SSM_SESSION_SLOTS {
                let oldest = entries
                    .iter()
                    .enumerate()
                    .min_by_key(|(_, e)| e.last_used)
                    .map(|(i, _)| i)
                    .unwrap_or(0);
                self.total_bytes -= entries.remove(oldest).data.len();
            }
        }
        if !self.make_room(data.len()) {
            return;
        }
        self.total_bytes += data.len();
        self.sessions
            .entry(session_id.to_string())
            .or_default()
            .push(SsmSessionEntry {
                tokens,
                data,
                last_used: tick,
            });
    }

    /// Drop one snapshot of a session (after a failed restore), or the whole
    /// session when it was the last one.
    fn remove_session_entry(&mut self, session_id: &str, idx: usize) {
        if let Some(entries) = self.sessions.get_mut(session_id) {
            if idx < entries.len() {
                self.total_bytes -= entries.remove(idx).data.len();
            }
            if entries.is_empty() {
                self.sessions.remove(session_id);
            }
        }
    }

    /// Evict least-recently-used snapshots (chunk checkpoints and session
    /// states share one byte budget) until `incoming` fits. False when it
    /// cannot fit even with everything evicted.
    fn make_room(&mut self, incoming: usize) -> bool {
        while self.total_bytes + incoming > self.max_bytes {
            let chunk_lru = self
                .entries
                .iter()
                .min_by_key(|(_, e)| e.last_used)
                .map(|(&h, e)| (e.last_used, h));
            let session_lru = self
                .sessions
                .iter()
                .flat_map(|(k, entries)| {
                    entries
                        .iter()
                        .enumerate()
                        .map(move |(i, e)| (e.last_used, k.clone(), i))
                })
                .min_by_key(|(t, _, _)| *t);
            match (chunk_lru, session_lru) {
                (Some((ct, hash)), Some((st, _, _))) if ct <= st => self.remove(hash),
                (_, Some((_, sid, idx))) => self.remove_session_entry(&sid, idx),
                (Some((_, hash)), None) => self.remove(hash),
                (None, None) => return false,
            }
        }
        true
    }
}

#[cfg(feature = "gguf")]
fn gguf_ssm_state_cache_config(config: GgufServingConfig) -> Option<SsmStateCache> {
    if !config.uses_state_space_memory || !gguf_env_is_truthy("KAPSL_GGUF_SSM_STATE_CACHE") {
        return None;
    }
    let mb = |name: &str, default: usize| {
        std::env::var(name)
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|&v| v > 0)
            .unwrap_or(default)
            .saturating_mul(1024 * 1024)
    };
    let max_bytes = mb("KAPSL_GGUF_SSM_STATE_CACHE_MB", 512);
    let max_entry_bytes = mb("KAPSL_GGUF_SSM_STATE_MAX_ENTRY_MB", 128);
    log::info!(
        "[gguf] SSM state cache enabled: stride={} tokens, cap={} MiB, max-entry={} MiB",
        config.prefill_chunk_size,
        max_bytes / (1024 * 1024),
        max_entry_bytes / (1024 * 1024),
    );
    Some(SsmStateCache::new(
        config.prefill_chunk_size,
        max_bytes,
        max_entry_bytes,
    ))
}

/// Serialize `seq_id`'s per-sequence state to host RAM, or `None` when it is
/// empty or exceeds `max_entry_bytes`.
#[cfg(feature = "gguf")]
fn read_seq_state(
    ctx: &llama_cpp_2::context::LlamaContext,
    seq_id: i32,
    max_entry_bytes: usize,
) -> Option<Vec<u8>> {
    let size = ctx.state_seq_get_size_ext(seq_id, llama_cpp_2::LlamaStateSeqFlags::empty());
    if size == 0 || size > max_entry_bytes {
        return None;
    }
    let mut data: Vec<u8> = Vec::with_capacity(size);
    let copied = unsafe {
        let n = ctx.state_seq_get_data_ext(
            data.as_mut_ptr(),
            seq_id,
            llama_cpp_2::LlamaStateSeqFlags::empty(),
        );
        data.set_len(n.min(size));
        n
    };
    if copied == 0 || copied > size {
        return None;
    }
    Some(data)
}

/// Snapshot `seq_id`'s recurrent state if the prefill just paused exactly on a
/// chunk boundary that is not cached yet. Called after a successful decode, so
/// the live state has absorbed exactly `tokens[..n_decoded]`.
#[cfg(feature = "gguf")]
fn maybe_snapshot_ssm_state(
    ctx: &llama_cpp_2::context::LlamaContext,
    cache: &mut SsmStateCache,
    tokens: &[LlamaToken],
    n_decoded: usize,
    seq_id: i32,
) {
    if n_decoded == 0 || !n_decoded.is_multiple_of(cache.chunk) || n_decoded > tokens.len() {
        return;
    }
    let hashes = cache.prefix_hashes(&tokens[..n_decoded]);
    let Some(&hash) = hashes.last() else {
        return;
    };
    if cache.contains(hash) {
        return;
    }
    let Some(data) = read_seq_state(ctx, seq_id, cache.max_entry_bytes) else {
        return;
    };
    log::debug!(
        "[gguf] SSM state checkpoint: seq={seq_id} pos={n_decoded} bytes={} (cache {} MiB)",
        data.len(),
        (cache.total_bytes + data.len()) / (1024 * 1024)
    );
    cache.insert(hash, n_decoded, data);
}

/// Snapshot a retiring sequence's state as its session's resume point. The
/// state covers exactly `absorbed` (prompt + decoded generation), so the next
/// turn of the same session — whose prompt begins with that history — resumes
/// past it instead of re-prefilling the whole conversation.
#[cfg(feature = "gguf")]
fn snapshot_ssm_session_state(
    ctx: &llama_cpp_2::context::LlamaContext,
    cache: &mut SsmStateCache,
    session_id: &str,
    absorbed: &[LlamaToken],
    seq_id: i32,
) {
    if absorbed.is_empty() || cache.session_is_current(session_id, absorbed) {
        return;
    }
    let Some(data) = read_seq_state(ctx, seq_id, cache.max_entry_bytes) else {
        return;
    };
    log::debug!(
        "[gguf] SSM session state saved: session={session_id} seq={seq_id} pos={} bytes={}",
        absorbed.len(),
        data.len(),
    );
    cache.insert_session(session_id, absorbed.to_vec(), data);
}

/// Restore the best cached state for `tokens` into `seq_id`: the session's
/// resume snapshot when its history is a strict prefix of the prompt (skips
/// prompt + previous generation), else the longest chunk-boundary checkpoint.
/// Returns the number of prompt tokens the restored state already covers
/// (0 = no hit, start prefill from scratch).
#[cfg(feature = "gguf")]
fn restore_ssm_state(
    ctx: &mut llama_cpp_2::context::LlamaContext,
    cache: &mut SsmStateCache,
    session_id: Option<&str>,
    tokens: &[LlamaToken],
    seq_id: i32,
) -> usize {
    if let Some(sid) = session_id {
        if let Some((entry_idx, n_tokens)) = cache.session_match(sid, tokens) {
            let ok = cache
                .session_data(sid, entry_idx)
                .is_some_and(|data| unsafe {
                    ctx.state_seq_set_data_ext(
                        data,
                        seq_id,
                        llama_cpp_2::LlamaStateSeqFlags::empty(),
                    )
                });
            if ok {
                log::debug!(
                    "[gguf] SSM session state restored: session={sid} seq={seq_id} pos={n_tokens}"
                );
                return n_tokens;
            }
            log::warn!(
                "[gguf] SSM session restore failed: session={sid} seq={seq_id}; dropping entry"
            );
            cache.remove_session_entry(sid, entry_idx);
            let _ = ctx.clear_kv_cache_seq(u32::try_from(seq_id).ok(), None, None);
        }
    }

    if tokens.len() <= cache.chunk {
        return 0;
    }
    let hashes = cache.prefix_hashes(tokens);
    let Some((hash, n_tokens)) = cache.lookup_longest(&hashes, tokens.len()) else {
        return 0;
    };
    let ok = cache.data(hash).is_some_and(|data| unsafe {
        ctx.state_seq_set_data_ext(data, seq_id, llama_cpp_2::LlamaStateSeqFlags::empty())
    });
    if !ok {
        // A failed restore may leave the sequence's memory partially written;
        // clear it and fall back to a full prefill. Drop the entry — it is
        // no longer trusted.
        log::warn!(
            "[gguf] SSM state restore failed for seq={seq_id} pos={n_tokens}; dropping entry"
        );
        cache.remove(hash);
        let _ = ctx.clear_kv_cache_seq(u32::try_from(seq_id).ok(), None, None);
        return 0;
    }
    log::debug!(
        "[gguf] SSM state restored: seq={seq_id} pos={n_tokens} (prefill skips {n_tokens} tokens)"
    );
    n_tokens
}

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
    /// Resolved scheduling priority (0 = latency-critical, higher = lower).
    /// Drives priority-aware promotion out of the `waiting` queue.
    priority: u8,
    /// Client session key, used for SSM session resume-state snapshots.
    session_id: Option<String>,
    response: GgufResponse,
}

#[cfg(feature = "gguf")]
enum GgufResponse {
    Final(std_mpsc::Sender<Result<Vec<u8>, EngineError>>),
    Stream(std_mpsc::Sender<Result<Vec<u8>, EngineError>>),
}

#[cfg(feature = "gguf")]
impl GgufResponse {
    fn emit_bytes(&self, output: &mut Vec<u8>, bytes: Vec<u8>) -> bool {
        if bytes.is_empty() {
            return true;
        }
        let _timing = GgufTimingGuard::new(&GGUF_TIMING_EMIT_CALLS, &GGUF_TIMING_EMIT_US);
        match self {
            Self::Final(_) => {
                output.extend_from_slice(&bytes);
                true
            }
            Self::Stream(tx) => tx.send(Ok(bytes)).is_ok(),
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

#[cfg(feature = "gguf")]
const GGUF_CHAT_STOP_SEQUENCES: &[&[u8]] = &[
    b"<end_of_turn>",
    b"<start_of_turn>user",
    b"<|im_end|>",
    b"<|im_start|>user",
    b"<|eot_id|>",
];

#[cfg(feature = "gguf")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GgufEmitResult {
    Continue,
    Disconnected,
    Stopped,
}

#[cfg(feature = "gguf")]
struct GgufStopFilter {
    pending: Vec<u8>,
    stopped: bool,
}

#[cfg(feature = "gguf")]
impl GgufStopFilter {
    fn new() -> Self {
        Self {
            pending: Vec::new(),
            stopped: false,
        }
    }

    fn max_stop_len() -> usize {
        GGUF_CHAT_STOP_SEQUENCES
            .iter()
            .map(|seq| seq.len())
            .max()
            .unwrap_or(0)
    }

    /// Locate the earliest chat stop-marker in `bytes`, returning its start
    /// offset and length.
    fn find_stop(bytes: &[u8]) -> Option<(usize, usize)> {
        GGUF_CHAT_STOP_SEQUENCES
            .iter()
            .filter_map(|stop| {
                bytes
                    .windows(stop.len())
                    .position(|window| window == *stop)
                    .map(|start| (start, stop.len()))
            })
            .min_by_key(|&(start, _)| start)
    }

    /// Feed a freshly decoded token `piece` into the filter.
    ///
    /// When `may_stop` is true a chat stop-marker latches the filter and
    /// retires the sequence (emitting only the text before the marker). When
    /// `may_stop` is false — i.e. the sequence is still below its `min_tokens`
    /// floor — stop-markers are *not* honored: the text before each marker is
    /// emitted, the marker bytes themselves are dropped (so they never leak
    /// into the output), and generation continues. This mirrors the EOS gate
    /// so `min_tokens` is enforced even when the model emits a turn-marker
    /// (e.g. Qwen's `<|im_end|>`) before the floor is reached.
    fn push_piece(
        &mut self,
        response: &GgufResponse,
        output: &mut Vec<u8>,
        piece: Vec<u8>,
        may_stop: bool,
    ) -> GgufEmitResult {
        if self.stopped {
            return GgufEmitResult::Stopped;
        }

        self.pending.extend_from_slice(&piece);

        if may_stop {
            if let Some((stop_at, _stop_len)) = Self::find_stop(&self.pending) {
                let emit = self.pending[..stop_at].to_vec();
                self.pending.clear();
                self.stopped = true;
                return if response.emit_bytes(output, emit) {
                    GgufEmitResult::Stopped
                } else {
                    GgufEmitResult::Disconnected
                };
            }
        } else {
            // Below the min_tokens floor: emit the text preceding each marker,
            // discard the marker bytes, and keep decoding.
            while let Some((stop_at, stop_len)) = Self::find_stop(&self.pending) {
                let emit = self.pending[..stop_at].to_vec();
                self.pending.drain(..stop_at + stop_len);
                if !response.emit_bytes(output, emit) {
                    return GgufEmitResult::Disconnected;
                }
            }
        }

        let keep = Self::max_stop_len().saturating_sub(1);
        if self.pending.len() <= keep {
            return GgufEmitResult::Continue;
        }

        let emit_len = self.pending.len() - keep;
        let emit = self.pending.drain(..emit_len).collect::<Vec<_>>();
        if response.emit_bytes(output, emit) {
            GgufEmitResult::Continue
        } else {
            GgufEmitResult::Disconnected
        }
    }

    fn flush(&mut self, response: &GgufResponse, output: &mut Vec<u8>) -> bool {
        if self.stopped || self.pending.is_empty() {
            self.pending.clear();
            return true;
        }
        let emit = std::mem::take(&mut self.pending);
        response.emit_bytes(output, emit)
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
    session_id: Option<String>,
    response: GgufResponse,
    copies: Vec<PendingPrefillCopy>,
}

#[cfg(feature = "gguf")]
struct PendingPrefillCopy {
    seq_id: i32,
    max_tokens: i32,
    min_tokens: i32,
    session_id: Option<String>,
    response: GgufResponse,
}

/// A sequence in the decode phase.
#[cfg(feature = "gguf")]
struct ActiveSeq {
    seq_id: i32,
    /// Next KV-cache position to write.
    pos: i32,
    prompt_tokens: usize,
    n_generated: i32,
    max_tokens: i32,
    min_tokens: i32,
    suppress_eos_sampler: bool,
    /// Token to feed into the next decode step.
    last_token: LlamaToken,
    output: Vec<u8>,
    stop_filter: GgufStopFilter,
    response: GgufResponse,
    error: Option<EngineError>,
    /// Session key for the SSM resume-state snapshot taken at retirement.
    session_id: Option<String>,
    /// Every token the model has decoded for this sequence (prompt + fed-back
    /// generation) — the recurrent state at any instant covers exactly these.
    /// Only tracked while the SSM state cache is on; empty otherwise.
    absorbed: Vec<LlamaToken>,
}

#[cfg(feature = "gguf")]
struct GgufBackendSamplers {
    // One [logit_bias(eos), greedy] chain per slot, installed once and kept for the context
    // lifetime. EOS suppression toggles the bias between -inf and 0 in place, so steady-state
    // mode changes never call set_sampler and never re-trigger a backend scheduler reserve.
    chains: Vec<LlamaSampler>,
    installed_by_seq_id: Vec<bool>,
    mode_by_seq_id: Vec<bool>,
    eos_token: LlamaToken,
    sample: LlamaSampler,
}

#[cfg(feature = "gguf")]
impl GgufBackendSamplers {
    fn new(model: &LlamaModel, max_concurrent: usize, eos_token: LlamaToken) -> Self {
        let n_vocab = model.n_vocab();
        let eos_bias = [LlamaLogitBias::new(eos_token, 0.0)];
        let chains = (0..max_concurrent)
            .map(|_| {
                LlamaSampler::chain_simple([
                    LlamaSampler::logit_bias(n_vocab, &eos_bias),
                    LlamaSampler::greedy(),
                ])
            })
            .collect();

        Self {
            chains,
            installed_by_seq_id: vec![false; max_concurrent],
            mode_by_seq_id: vec![false; max_concurrent],
            eos_token,
            sample: LlamaSampler::greedy(),
        }
    }

    fn set_for_sequence(
        &mut self,
        ctx: &mut llama_cpp_2::context::LlamaContext,
        seq_id: i32,
        suppress_eos: bool,
    ) -> bool {
        let Some(slot) = usize::try_from(seq_id).ok() else {
            return false;
        };
        let Some(chain) = self.chains.get_mut(slot) else {
            return false;
        };
        if self.mode_by_seq_id[slot] != suppress_eos {
            let bias = if suppress_eos { f32::NEG_INFINITY } else { 0.0 };
            if !chain.chain_logit_bias_set(0, &[LlamaLogitBias::new(self.eos_token, bias)]) {
                return false;
            }
            self.mode_by_seq_id[slot] = suppress_eos;
        }
        if !self.installed_by_seq_id[slot] {
            if !unsafe { ctx.set_sampler(seq_id, Some(chain)) } {
                return false;
            }
            self.installed_by_seq_id[slot] = true;
        }
        true
    }

    fn sample_token(
        &mut self,
        ctx: &llama_cpp_2::context::LlamaContext,
        batch_pos: i32,
    ) -> LlamaToken {
        let _timing = GgufTimingGuard::new(&GGUF_TIMING_SAMPLE_CALLS, &GGUF_TIMING_SAMPLE_US);
        ctx.sampled_token_ith(batch_pos)
            .unwrap_or_else(|| self.sample.sample(ctx, batch_pos))
    }
}

#[cfg(feature = "gguf")]
fn record_gguf_token_metrics(
    metrics: &Arc<Mutex<EngineMetrics>>,
    prompt_tokens: usize,
    generated_tokens: usize,
) {
    if let Ok(mut snapshot) = metrics.lock() {
        snapshot.prompt_tokens_total = snapshot
            .prompt_tokens_total
            .saturating_add(prompt_tokens as u64);
        snapshot.generated_tokens_total = snapshot
            .generated_tokens_total
            .saturating_add(generated_tokens as u64);
        snapshot.refresh_timestamp();
    }
}

#[cfg(feature = "gguf-cuda-shared-kv")]
fn record_gguf_decode_work_metrics(
    metrics: &Arc<Mutex<EngineMetrics>>,
    steps: u64,
    tokens_evaluated: u64,
) {
    if steps == 0 && tokens_evaluated == 0 {
        return;
    }
    if let Ok(mut snapshot) = metrics.lock() {
        snapshot.decode_steps_total = snapshot.decode_steps_total.saturating_add(steps);
        snapshot.decode_tokens_evaluated_total = snapshot
            .decode_tokens_evaluated_total
            .saturating_add(tokens_evaluated);
        snapshot.refresh_timestamp();
    }
}

#[cfg(feature = "gguf")]
fn record_gguf_partial_reuse_metrics(
    metrics: &Arc<Mutex<EngineMetrics>>,
    hits: u64,
    tokens_saved: u64,
) {
    if hits == 0 && tokens_saved == 0 {
        return;
    }
    if let Ok(mut snapshot) = metrics.lock() {
        snapshot.kv_partial_reuse_hits_total =
            snapshot.kv_partial_reuse_hits_total.saturating_add(hits);
        snapshot.kv_partial_reuse_tokens_saved_total = snapshot
            .kv_partial_reuse_tokens_saved_total
            .saturating_add(tokens_saved);
        snapshot.refresh_timestamp();
    }
}

/// KV block data downloaded to CPU when the scheduler went idle.
///
/// Stored in `GgufSharedKvPoolState::evicted` until the next `reserve` call
/// restores the data back to freshly allocated GPU blocks.
#[cfg(feature = "gguf-cuda-shared-kv")]
struct CpuEvictedState {
    store: CpuBlockStore,
    /// Slot indices: `slots[layer * n_logical + pos]`
    /// where `pos` is 0-indexed within the evicted (non-promoted) positions.
    slots: Vec<u32>,
    /// Number of owned logical positions that were saved.
    n_logical: usize,
    /// Original `n_logical_blocks` at eviction time, used by `needs_restore`
    /// to tell C++ which token count to force-re-reserve.
    n_logical_at_eviction: usize,
}

#[cfg(feature = "gguf-cuda-shared-kv")]
struct GgufSharedKvPool {
    state: Box<GgufSharedKvPoolState>,
    desc: Box<llama_kapsl_kv_pool_desc>,
}

#[cfg(feature = "gguf-cuda-shared-kv")]
struct GgufSharedKvPoolState {
    handle: GpuPoolHandle,
    metrics: Arc<Mutex<EngineMetrics>>,
    device_id: usize,
    n_layers: usize,
    max_blocks_per_seq: usize,
    /// Optional prefix block cache. When present, `reserve_prefix` and
    /// `promote_prefix` callbacks are active.
    prefix_cache: Option<Arc<Mutex<PrefixBlockCache>>>,
    /// Stable model identity hash (set at pool construction time).
    model_fingerprint: u64,
    inner: Mutex<GgufSharedKvReservation>,
    /// Number of concurrent sequence slots the combined block table holds
    /// (= scheduler max_concurrent). 0 disables multi-sequence batching.
    n_seq_slots: usize,
    /// Multi-sequence reservation state: per-seq persistent block ownership plus
    /// the combined block table shared by all slots. Only used by the
    /// `reserve_seq` path; the single-sequence `inner` path is untouched.
    multi: Mutex<GgufMultiSeqState>,
    /// CPU-side KV block backup, populated by `evict_to_cpu()`.
    /// Cleared by the next `reserve` call which uploads data back to GPU.
    evicted: Mutex<Option<CpuEvictedState>>,
    /// When true, `run_scheduler` calls `evict_to_cpu()` before blocking on
    /// the idle receive.  Controlled by `KAPSL_GGUF_EVICT_ON_IDLE`.
    evict_when_idle: bool,
    /// Windowed KV allocation for SWA layers (Phase 2). `None` = uniform full
    /// allocation on every layer (the only mode before Phase 2, and still the
    /// default). When set, `reserve`/`reserve_prefix` route to the windowed
    /// reservation below and the prefix cache and CPU eviction are bypassed
    /// (ring blocks are overwritten in place, so their KV cannot be reused
    /// across sessions or snapshotted).
    windowed: Option<GgufWindowedKvConfig>,
    /// Single-sequence windowed reservation, used instead of `inner` when
    /// `windowed` is set. Per-layer block ownership: full layers hold one
    /// block per logical position, SWA layers hold at most the ring.
    windowed_inner: Mutex<GgufWindowedReservation>,
}

#[cfg(feature = "gguf-cuda-shared-kv")]
impl Drop for GgufSharedKvPool {
    fn drop(&mut self) {
        gguf_global_kv_scheduler()
            .lock()
            .unwrap()
            .unregister_pool(self.state.device_id, &self.state.handle.pool);
    }
}

/// Single-sequence reservation state for windowed (Phase 2) allocation.
///
/// Unlike [`GgufSharedKvReservation`]'s flat `[layer * n_new + pos]` layout,
/// ownership is per-layer because SWA layers cap out at the ring size while
/// full layers keep growing with the context.
#[cfg(feature = "gguf-cuda-shared-kv")]
#[derive(Default)]
struct GgufWindowedReservation {
    /// Per-layer physical blocks, in logical order; SWA layers wrap around
    /// (`len == min(n_logical_blocks, window_blocks)`).
    layers: Vec<Vec<u32>>,
    block_table: Option<CudaSlice<u32>>,
    n_logical_blocks: usize,
    n_tokens_reserved: usize,
}

/// Per-session reservation state.
///
/// Block layout (per-layer pools, as in llama.cpp's paged KV):
///
/// ```text
/// owned_blocks[layer * n_new_logical + pos]
/// ```
///
/// where `pos` is 0-indexed within the freshly allocated positions (i.e. the
/// positions that did NOT come from the prefix cache).
#[cfg(feature = "gguf-cuda-shared-kv")]
#[derive(Default)]
struct GgufSharedKvReservation {
    /// Freshly allocated blocks (prefix-cache hits NOT included).
    /// Layout: `owned_blocks[layer * n_new_logical + pos]`.
    owned_blocks: Vec<u32>,
    /// Logical positions covered by `owned_blocks` per layer.
    n_new_logical: usize,
    /// Chained hashes for ALL logical positions:
    ///   `all_hashes[0..n_prefix_hits]`       — prefix-cache borrows
    ///   `all_hashes[n_prefix_hits..]`         — newly allocated positions
    all_hashes: Vec<u64>,
    /// Number of leading logical positions that came from the prefix cache.
    n_prefix_hits: usize,
    /// How many of the new logical positions (from `n_prefix_hits` onward)
    /// have been promoted to the prefix cache and are now cache-owned.
    n_promoted_logical: usize,
    model_fingerprint: u64,
    block_table: Option<CudaSlice<u32>>,
    block_table_host: Vec<u32>,
    n_logical_blocks: usize,
    n_tokens_reserved: usize,
}

/// Multi-sequence reservation state for concurrent paged decode.
///
/// Each active sequence (keyed by its scheduler seq_id, which doubles as its
/// block-table slot) persistently owns physical blocks — grow-only, never freed
/// mid-decode — so its KV data survives across steps. The combined block table
/// is `[n_seq_slots, n_layers, max_blocks_per_seq]` flattened; the kernels select
/// a token's slice via `seq_slot * (n_layers * max_blocks_per_seq)`.
#[cfg(feature = "gguf-cuda-shared-kv")]
#[derive(Default)]
struct GgufMultiSeqState {
    /// seq_id -> owned[layer] = logical-ordered physical block ids.
    sessions: std::collections::HashMap<u64, Vec<Vec<u32>>>,
    /// Host mirror of the combined block table, len `n_seq_slots * seq_stride`.
    combined_host: Vec<u32>,
    /// Device copy of `combined_host`; its pointer is handed to llama.cpp.
    combined_device: Option<CudaSlice<u32>>,
    /// True when `combined_host` has changed since `combined_device` was uploaded.
    combined_dirty: bool,
    /// seq_id -> logical blocks committed to its block-table region. Needed in
    /// windowed mode, where per-layer ownership lengths cap at the ring size
    /// and can no longer be used to infer the logical coverage.
    seq_logical: std::collections::HashMap<u64, usize>,
}

#[cfg(feature = "gguf-cuda-shared-kv")]
unsafe impl Send for GgufSharedKvPool {}
#[cfg(feature = "gguf-cuda-shared-kv")]
unsafe impl Sync for GgufSharedKvPool {}

#[cfg(feature = "gguf-cuda-shared-kv")]
impl GgufSharedKvPool {
    fn new(
        handle: GpuPoolHandle,
        metrics: Arc<Mutex<EngineMetrics>>,
        device_id: usize,
        n_layers: usize,
        ctx_per_seq: usize,
        max_concurrent: usize,
        prefix_cache: Option<Arc<Mutex<PrefixBlockCache>>>,
        model_fingerprint: u64,
        windowed: Option<GgufWindowedKvConfig>,
    ) -> Self {
        let pool_arc = handle.pool.clone();
        let block_size = handle.pool.block_size().max(1);
        let max_blocks_per_seq = ctx_per_seq.div_ceil(block_size).max(1);
        let has_prefix = prefix_cache.is_some();
        // Combined block table geometry for multi-sequence batching.
        let n_seq_slots = max_concurrent.max(1);
        let block_table_seq_stride = n_layers * max_blocks_per_seq;
        let combined_host = vec![0u32; n_seq_slots * block_table_seq_stride];
        let evict_when_idle = std::env::var("KAPSL_GGUF_EVICT_ON_IDLE")
            .map(|v| {
                !matches!(
                    v.trim().to_ascii_lowercase().as_str(),
                    "0" | "false" | "no" | "off"
                )
            })
            .unwrap_or(false);

        let mut state = Box::new(GgufSharedKvPoolState {
            handle,
            metrics,
            device_id,
            n_layers,
            max_blocks_per_seq,
            prefix_cache,
            model_fingerprint,
            inner: Mutex::new(GgufSharedKvReservation::default()),
            n_seq_slots,
            multi: Mutex::new(GgufMultiSeqState {
                combined_host,
                ..Default::default()
            }),
            evicted: Mutex::new(None),
            evict_when_idle,
            windowed,
            windowed_inner: Mutex::new(GgufWindowedReservation::default()),
        });
        let state_ptr = (&mut *state) as *mut GgufSharedKvPoolState;
        let pool = &state.handle.pool;
        let desc = Box::new(llama_kapsl_kv_pool_desc {
            user_data: state_ptr.cast(),
            device_id: device_id as u32,
            block_size: pool.block_size() as u32,
            // Physical block ids are relative to the whole device backing, so
            // GGML's buffer must cover the whole addressable span. Allocation
            // remains limited independently by the view/engine quota.
            num_blocks: pool.addressable_blocks() as u32,
            num_kv_heads: pool.num_kv_heads() as u32,
            head_dim: pool.head_dim() as u32,
            dtype: LLAMA_KAPSL_KV_DTYPE_F16,
            // SAFETY: llama.cpp receives the whole backing plus a block table;
            // the shared-KV allocator supplies only live blocks owned by this view.
            device_base: unsafe { pool.device_base_ptr() },
            block_table_device: std::ptr::null_mut(),
            block_table_layer_stride: max_blocks_per_seq as u32,
            n_layers: n_layers as u32,
            max_blocks_per_seq: max_blocks_per_seq as u32,
            // Multi-sequence combined block table geometry.
            block_table_seq_stride: block_table_seq_stride as u32,
            n_seq_slots: n_seq_slots as u32,
            model_fingerprint,
            reserve: Some(gguf_kapsl_kv_reserve),
            reserve_seq: Some(gguf_kapsl_kv_reserve_seq),
            commit_seq: Some(gguf_kapsl_kv_commit_seq),
            release: Some(gguf_kapsl_kv_release),
            touch: Some(gguf_kapsl_kv_touch),
            reserve_prefix: if has_prefix {
                Some(gguf_kapsl_kv_reserve_prefix)
            } else {
                None
            },
            promote_prefix: None, // handled via promote_if_pending() from Rust
            needs_restore: Some(gguf_kapsl_kv_needs_restore),
        });
        let kv_pool = Self { state, desc };
        gguf_global_kv_scheduler()
            .lock()
            .unwrap()
            .register_pool(device_id, pool_arc);
        log::debug!("[gguf] registered device {device_id} pool with cross-device KV scheduler");
        kv_pool
    }

    /// Promote newly computed KV blocks to the prefix cache.
    ///
    /// Must be called after a successful `ctx.decode()` when a prefill batch was
    /// processed.  No-op when the prefix cache is disabled or there is nothing
    /// to promote.
    fn promote_if_pending(&mut self) {
        let state = &mut *self.state;
        let Some(cache) = &state.prefix_cache else {
            return;
        };
        let mut inner = state.inner.lock().unwrap();
        let n_new = inner.n_new_logical;
        if n_new == 0 || inner.n_promoted_logical >= n_new {
            return;
        }

        let start = inner.n_promoted_logical;
        let fp = inner.model_fingerprint;
        let pool = state.handle.pool.clone();
        let n_layers = state.n_layers;
        let device_id = self.desc.device_id as usize;
        let mut c = match cache.lock() {
            Ok(c) => c,
            Err(_) => return,
        };

        let mut promoted = start;
        for pos in start..n_new {
            let hash_idx = inner.n_prefix_hits + pos;
            let Some(&hash) = inner.all_hashes.get(hash_idx) else {
                break;
            };

            // Build the block_ids Vec for this logical position (one block per layer).
            let block_ids: Vec<u32> = (0..n_layers)
                .map(|layer| inner.owned_blocks[layer * n_new + pos])
                .collect();

            // Count a position as promoted ONLY when the cache actually took
            // ownership of its blocks. On AlreadyPresent/Rejected the blocks stay
            // session-owned (freed at release via the un-promoted range) and no
            // cache refcount was taken for them, so counting them would leak the
            // blocks and unbalance the release() accounting.
            match c.insert(fp, hash, device_id, pool.clone(), block_ids, 1) {
                kapsl_hal::prefix_cache::PrefixInsert::Inserted => promoted += 1,
                _ => break, // cache full or hash already cached — stop
            }
        }
        inner.n_promoted_logical = promoted;
    }

    /// Download un-promoted owned KV blocks to CPU and free their GPU slots.
    ///
    /// Intended to be called from `run_scheduler` when the model goes idle
    /// (no active or pending sequences).  The freed GPU blocks become available
    /// for other sessions/models in the shared pool.
    ///
    /// On the next `ctx.decode()` call, the C++ `init_batch` will detect the
    /// eviction via `needs_restore`, force a re-reserve, and our reserve
    /// callback will re-upload the saved data to freshly allocated GPU blocks.
    ///
    /// Returns `true` if blocks were actually evicted.
    fn evict_to_cpu(&mut self) -> bool {
        let state = &*self.state;

        // Skip if already evicted.
        if state.evicted.lock().unwrap().is_some() {
            return false;
        }

        let mut inner = state.inner.lock().unwrap();
        let n_new = inner.n_new_logical;
        let n_promoted = inner.n_promoted_logical;
        let n_owned = n_new.saturating_sub(n_promoted);
        let n_logical_orig = inner.n_logical_blocks;

        if n_owned == 0 {
            return false;
        }

        let n_layers = state.n_layers;
        let pool = &state.handle.pool;

        // Download each un-promoted owned block to CPU.
        let n_slots = n_owned * n_layers;
        let mut store = CpuBlockStore::new(
            n_slots,
            pool.num_kv_heads(),
            pool.block_size(),
            pool.head_dim(),
        );
        let mut slots: Vec<u32> = Vec::with_capacity(n_slots);

        for layer in 0..n_layers {
            for pos in n_promoted..n_new {
                let block_id = match inner.owned_blocks.get(layer * n_new + pos) {
                    Some(&id) => id,
                    None => return false,
                };
                match pool.download_block(block_id) {
                    Ok(data) => match store.store_block(&data) {
                        Ok(slot) => slots.push(slot),
                        Err(_) => return false,
                    },
                    Err(_) => return false,
                }
            }
        }

        // Release prefix borrows + free all un-promoted GPU blocks.
        gguf_release_reservation_inner(&mut inner, &state.prefix_cache, pool, n_layers);
        drop(inner);

        *state.evicted.lock().unwrap() = Some(CpuEvictedState {
            store,
            slots,
            n_logical: n_owned,
            n_logical_at_eviction: n_logical_orig,
        });

        log::info!(
            "[gguf] evicted {} owned positions ({} GPU blocks freed) to CPU",
            n_owned,
            n_owned * n_layers
        );
        true
    }

    fn desc_ptr(&mut self) -> *mut llama_kapsl_kv_pool_desc {
        (&mut *self.desc) as *mut llama_kapsl_kv_pool_desc
    }
}

#[cfg(feature = "gguf-cuda-shared-kv")]
impl Drop for GgufSharedKvPoolState {
    fn drop(&mut self) {
        let inner = self.inner.lock().unwrap();
        // Release prefix cache borrows (prefix hits + promoted blocks).
        if let Some(cache) = &self.prefix_cache {
            if let Ok(mut c) = cache.lock() {
                let n_borrow = inner.n_prefix_hits + inner.n_promoted_logical;
                let fp = inner.model_fingerprint;
                for &hash in inner.all_hashes.get(..n_borrow).unwrap_or(&[]) {
                    c.release(fp, hash);
                }
            }
        }
        // Free un-promoted owned blocks.  Promoted blocks are now cache-owned.
        let n_new = inner.n_new_logical;
        let n_layers = self.n_layers;
        for layer in 0..n_layers {
            for pos in inner.n_promoted_logical..n_new {
                let block = inner.owned_blocks[layer * n_new + pos];
                self.handle.pool.free_block(block);
            }
        }
        drop(inner);
        // Free the windowed (Phase 2) reservation's per-layer blocks.
        let mut w = self.windowed_inner.lock().unwrap();
        for layer in std::mem::take(&mut w.layers) {
            for b in layer {
                self.handle.pool.free_block(b);
            }
        }
    }
}

// ── CPU-eviction needs-restore callback ──────────────────────────────────────

#[cfg(feature = "gguf-cuda-shared-kv")]
unsafe extern "C" fn gguf_kapsl_kv_needs_restore(
    user_data: *mut std::ffi::c_void,
    _session_id: u64,
) -> u32 {
    if user_data.is_null() {
        return 0;
    }
    let state = &*(user_data as *mut GgufSharedKvPoolState);
    match &*state.evicted.lock().unwrap() {
        Some(e) => {
            let block_size = state.handle.pool.block_size().max(1);
            (e.n_logical_at_eviction * block_size) as u32
        }
        None => 0,
    }
}

// ── Block allocation helper ───────────────────────────────────────────────────

/// Allocate `n_needed` blocks from the pool.
///
/// On first-pass failure, evicts the oldest zero-refcount prefix-cache entries
/// to free GPU blocks, then retries.  Drains and returns `false` when blocks
/// cannot be obtained even after eviction.
#[cfg(feature = "gguf-cuda-shared-kv")]
fn alloc_blocks_with_cache_eviction(
    state: &GgufSharedKvPoolState,
    n_needed: usize,
    out: &mut Vec<u32>,
) -> bool {
    // First pass: allocate as many blocks as are immediately available.
    while out.len() < n_needed {
        match state.handle.pool.alloc_block() {
            Ok(b) => out.push(b),
            Err(_) => break,
        }
    }
    if out.len() == n_needed {
        return true;
    }

    // Evict LRU zero-refcount prefix-cache entries to reclaim GPU blocks.
    let still_needed = n_needed - out.len();
    if let Some(cache) = &state.prefix_cache {
        if let Ok(mut c) = cache.lock() {
            let freed = c.evict_lru_for_device(state.device_id, still_needed);
            for (p, ids) in freed {
                for id in ids {
                    p.free_block(id);
                }
            }
        }
    }

    // Second pass after eviction.
    while out.len() < n_needed {
        match state.handle.pool.alloc_block() {
            Ok(b) => out.push(b),
            Err(_) => {
                for b in out.drain(..) {
                    state.handle.pool.free_block(b);
                }
                return false;
            }
        }
    }
    true
}

// ── Windowed (Phase 2) reserve handler ───────────────────────────────────────

/// Reserve or grow the single-sequence windowed reservation.
///
/// Serves both the `reserve` and `reserve_prefix` callbacks when
/// `state.windowed` is set (prefix hits are always 0 in windowed mode: ring
/// blocks are overwritten in place, so cached prefixes would go stale).
/// Returns `(block_table_device_ptr, n_logical_blocks)` on success.
#[cfg(feature = "gguf-cuda-shared-kv")]
fn gguf_windowed_reserve_impl(
    state: &GgufSharedKvPoolState,
    windowed: &GgufWindowedKvConfig,
    tokens_needed: usize,
) -> Option<(*mut u32, usize)> {
    let pool = &state.handle.pool;
    let block_size = pool.block_size().max(1);
    let logical = tokens_needed.div_ceil(block_size).max(1);
    if logical > state.max_blocks_per_seq {
        return None;
    }

    let mut inner = state.windowed_inner.lock().unwrap();

    // Fast path: the current reservation already covers the request.
    if inner.block_table.is_some() && inner.n_logical_blocks >= logical {
        inner.n_tokens_reserved = inner.n_tokens_reserved.max(tokens_needed);
        let ptr = *inner.block_table.as_ref()?.device_ptr() as *mut u32;
        GGUF_TIMING_KV_FAST_PATH_CALLS.fetch_add(1, Ordering::Relaxed);
        return Some((ptr, inner.n_logical_blocks));
    }

    // Grow every layer to its (ring-capped) target, all-or-nothing. Existing
    // blocks keep their KV data; only the extension is freshly allocated.
    if inner.layers.is_empty() {
        inner.layers = vec![Vec::new(); state.n_layers];
    }
    let targets: Vec<usize> = (0..state.n_layers)
        .map(|il| windowed_layer_capacity(logical, windowed.layer_window(il)))
        .collect();
    let prev_lens: Vec<usize> = inner.layers.iter().map(|v| v.len()).collect();
    let total_needed: usize = targets
        .iter()
        .zip(&prev_lens)
        .map(|(t, c)| t.saturating_sub(*c))
        .sum();
    let mut fresh: Vec<u32> = Vec::with_capacity(total_needed);
    if total_needed > 0 && !alloc_blocks_with_cache_eviction(state, total_needed, &mut fresh) {
        return None;
    }
    let mut next = 0usize;
    for (layer, (&target, &prev)) in targets.iter().zip(&prev_lens).enumerate() {
        let take = target.saturating_sub(prev);
        inner.layers[layer].extend_from_slice(&fresh[next..next + take]);
        next += take;
    }

    // Rebuild the host table. Full layers index directly; SWA layers wrap
    // around their ring so logical position P reuses ring slot P % ring_len —
    // recycling a physical block exactly when it falls out of the window.
    let mut host_table = vec![0u32; state.n_layers * state.max_blocks_per_seq];
    for (layer, blocks) in inner.layers.iter().enumerate() {
        for pos in 0..logical {
            host_table[layer * state.max_blocks_per_seq + pos] = blocks[pos % blocks.len()];
        }
    }

    let table = match pool.device().htod_sync_copy(&host_table) {
        Ok(t) => t,
        Err(_) => {
            // Roll back the extension; established blocks keep their KV data.
            for (layer, &prev) in prev_lens.iter().enumerate() {
                for b in inner.layers[layer].drain(prev..) {
                    pool.free_block(b);
                }
            }
            return None;
        }
    };
    let table_ptr = *table.device_ptr() as *mut u32;

    inner.block_table = Some(table);
    inner.n_logical_blocks = logical;
    inner.n_tokens_reserved = tokens_needed;
    GGUF_TIMING_KV_EXTEND_CALLS.fetch_add(1, Ordering::Relaxed);
    Some((table_ptr, logical))
}

// ── CPU-restore helper ────────────────────────────────────────────────────────

/// Re-allocate GPU blocks and upload KV data from a `CpuEvictedState`.
///
/// `logical_blocks` is the total number of logical positions the caller needs.
/// `n_prefix_hits` cached positions (from the prefix cache) are placed first in
/// the block table; restored + fresh positions follow.
///
/// On success, updates `inner` and writes the output pointers.
/// On allocation failure, restores the evicted state so a future call can retry.
#[cfg(feature = "gguf-cuda-shared-kv")]
fn gguf_restore_from_cpu(
    state: &GgufSharedKvPoolState,
    evicted: CpuEvictedState,
    logical_blocks: usize,
    n_prefix_hits: usize,
    prefix_hit_block_ids: &[Vec<u32>], // per-hit, per-layer block ids; len == n_prefix_hits
    block_table_device_out: *mut *mut u32,
    n_logical_out: *mut u32,
    n_prefix_hits_out: *mut u32, // may be null for non-prefix path
) -> bool {
    let n_layers = state.n_layers;
    let pool = &state.handle.pool;
    let n_new = logical_blocks.saturating_sub(n_prefix_hits);
    let n_restore = n_new.min(evicted.n_logical);
    let n_fresh = n_new.saturating_sub(n_restore);
    let total_phys = n_new * n_layers;

    let mut new_blocks: Vec<u32> = Vec::with_capacity(total_phys);
    if !alloc_blocks_with_cache_eviction(state, total_phys, &mut new_blocks) {
        *state.evicted.lock().unwrap() = Some(evicted);
        return false;
    }

    // Upload saved KV data for the restored positions.
    for layer in 0..n_layers {
        for pos in 0..n_restore {
            let block_id = new_blocks[layer * n_new + pos];
            let cpu_slot = evicted.slots[layer * evicted.n_logical + pos];
            if let Ok(data) = evicted.store.load_block(cpu_slot) {
                let half = data.len() / 2;
                if pool
                    .upload_block(block_id, &data[..half], &data[half..])
                    .is_err()
                {
                    for b in new_blocks {
                        pool.free_block(b);
                    }
                    *state.evicted.lock().unwrap() = Some(evicted);
                    return false;
                }
            }
        }
    }
    // `n_fresh` positions (after the restored ones) start with uninitialized
    // KV data — llama.cpp will overwrite them during the forward pass.

    // Build block table: prefix-hit blocks first, then restored+fresh blocks.
    let mut host_table = vec![0u32; n_layers * state.max_blocks_per_seq];
    for layer in 0..n_layers {
        for (hit_pos, block_ids) in prefix_hit_block_ids.iter().enumerate() {
            let blk = block_ids
                .get(layer)
                .copied()
                .unwrap_or_else(|| block_ids.first().copied().unwrap_or(0));
            host_table[layer * state.max_blocks_per_seq + hit_pos] = blk;
        }
        for pos in 0..n_new {
            host_table[layer * state.max_blocks_per_seq + n_prefix_hits + pos] =
                new_blocks[layer * n_new + pos];
        }
    }

    let table = match pool.device().htod_sync_copy(&host_table) {
        Ok(t) => t,
        Err(_) => {
            for b in new_blocks {
                pool.free_block(b);
            }
            *state.evicted.lock().unwrap() = Some(evicted);
            return false;
        }
    };
    let table_ptr = *table.device_ptr() as *mut u32;

    let mut inner = state.inner.lock().unwrap();
    gguf_release_reservation_inner(&mut inner, &state.prefix_cache, pool, n_layers);
    inner.owned_blocks = new_blocks;
    inner.n_new_logical = n_new;
    inner.n_promoted_logical = 0;
    inner.all_hashes = Vec::new();
    inner.n_prefix_hits = n_prefix_hits;
    inner.model_fingerprint = state.model_fingerprint;
    inner.block_table = Some(table);
    inner.block_table_host = host_table;
    inner.n_logical_blocks = logical_blocks;
    inner.n_tokens_reserved = logical_blocks.saturating_mul(state.handle.pool.block_size().max(1));

    unsafe {
        *block_table_device_out = table_ptr;
        *n_logical_out = logical_blocks as u32;
        if !n_prefix_hits_out.is_null() {
            *n_prefix_hits_out = n_prefix_hits as u32;
        }
    }
    log::info!(
        "[gguf] restored {} positions from CPU ({} fresh) to GPU after eviction",
        n_restore,
        n_fresh
    );
    true
}

// ── Reserve callback (no prefix cache) ───────────────────────────────────────

#[cfg(feature = "gguf-cuda-shared-kv")]
unsafe extern "C" fn gguf_kapsl_kv_reserve(
    user_data: *mut std::ffi::c_void,
    _session_id: u64,
    tokens_needed: u32,
    block_table_device_out: *mut *mut u32,
    n_blocks_out: *mut u32,
) -> bool {
    if user_data.is_null() || block_table_device_out.is_null() || n_blocks_out.is_null() {
        return false;
    }
    let state = &*(user_data as *mut GgufSharedKvPoolState);

    // ── Windowed (Phase 2) path ───────────────────────────────────────────
    if let Some(windowed) = &state.windowed {
        return match gguf_windowed_reserve_impl(state, windowed, tokens_needed as usize) {
            Some((table_ptr, logical)) => {
                *block_table_device_out = table_ptr;
                *n_blocks_out = logical as u32;
                true
            }
            None => false,
        };
    }

    let block_size = state.handle.pool.block_size().max(1);
    let logical_blocks = (tokens_needed as usize).div_ceil(block_size).max(1);
    if logical_blocks > state.max_blocks_per_seq {
        return false;
    }

    // ── Restore path: re-upload CPU-evicted blocks ────────────────────────
    let evicted_opt = state.evicted.lock().unwrap().take();
    if let Some(evicted) = evicted_opt {
        return gguf_restore_from_cpu(
            state,
            evicted,
            logical_blocks,
            0,
            &[],
            block_table_device_out,
            n_blocks_out,
            std::ptr::null_mut(),
        );
    }

    // ── Fresh allocation ──────────────────────────────────────────────────
    let needed_physical = logical_blocks.saturating_mul(state.n_layers);
    if needed_physical > state.handle.cap() {
        return false;
    }

    let mut new_blocks = Vec::with_capacity(needed_physical);
    if !alloc_blocks_with_cache_eviction(state, needed_physical, &mut new_blocks) {
        return false;
    }

    let mut host_table = vec![0u32; state.n_layers * state.max_blocks_per_seq];
    for layer in 0..state.n_layers {
        for pos in 0..logical_blocks {
            host_table[layer * state.max_blocks_per_seq + pos] =
                new_blocks[layer * logical_blocks + pos];
        }
    }
    let table = match state.handle.pool.device().htod_sync_copy(&host_table) {
        Ok(t) => t,
        Err(_) => {
            for b in new_blocks {
                state.handle.pool.free_block(b);
            }
            return false;
        }
    };
    let table_ptr = *table.device_ptr() as *mut u32;

    let mut inner = state.inner.lock().unwrap();
    gguf_release_reservation_inner(
        &mut inner,
        &state.prefix_cache,
        &state.handle.pool,
        state.n_layers,
    );
    inner.owned_blocks = new_blocks;
    inner.n_new_logical = logical_blocks;
    inner.n_promoted_logical = 0;
    inner.all_hashes = Vec::new();
    inner.n_prefix_hits = 0;
    inner.model_fingerprint = state.model_fingerprint;
    inner.block_table = Some(table);
    inner.block_table_host = host_table;
    inner.n_logical_blocks = logical_blocks;
    inner.n_tokens_reserved = tokens_needed as usize;

    *block_table_device_out = table_ptr;
    *n_blocks_out = logical_blocks as u32;
    true
}

// ── Multi-sequence reserve callback ──────────────────────────────────────────
//
// Reserves blocks for ONE sequence slot (keyed by seq_id) in the combined block
// table, growing the slot's per-layer block ownership as the sequence's context
// grows. Block ownership is persistent across decode steps (never freed until
// the sequence is released) so KV data survives. The combined table upload is
// deferred to `gguf_kapsl_kv_commit_seq` so one decode batch performs at most
// one H2D copy.
#[cfg(feature = "gguf-cuda-shared-kv")]
unsafe extern "C" fn gguf_kapsl_kv_reserve_seq(
    user_data: *mut std::ffi::c_void,
    seq_id: u64,
    tokens_needed: u32,
    block_table_device_out: *mut *mut u32,
    n_blocks_out: *mut u32,
) -> bool {
    if user_data.is_null() || block_table_device_out.is_null() || n_blocks_out.is_null() {
        return false;
    }
    let state = &*(user_data as *mut GgufSharedKvPoolState);
    let block_size = state.handle.pool.block_size().max(1);
    let logical = (tokens_needed as usize).div_ceil(block_size).max(1);
    if logical > state.max_blocks_per_seq {
        return false;
    }
    let slot = seq_id as usize;
    if slot >= state.n_seq_slots {
        return false;
    }
    let n_layers = state.n_layers;
    let seq_stride = n_layers * state.max_blocks_per_seq;

    let mut m = state.multi.lock().unwrap();

    // Grow this sequence's per-layer block ownership to cover `logical`
    // blocks, all-or-nothing to keep the slot consistent on allocation
    // failure. Without windowing every layer's target is `logical` and the
    // ring mapping below degenerates to the identity; with windowing (Phase 2)
    // SWA layers cap at the ring size and their table entries wrap around.
    let have = m.seq_logical.get(&seq_id).copied().unwrap_or_else(|| {
        m.sessions
            .get(&seq_id)
            .map(|owned| owned.iter().map(|v| v.len()).min().unwrap_or(0))
            .unwrap_or(0)
    });
    if logical > have {
        let targets: Vec<usize> = (0..n_layers)
            .map(|il| {
                windowed_layer_capacity(
                    logical,
                    state.windowed.as_ref().and_then(|w| w.layer_window(il)),
                )
            })
            .collect();
        let cur_lens: Vec<usize> = m
            .sessions
            .get(&seq_id)
            .map(|owned| owned.iter().map(|v| v.len()).collect())
            .unwrap_or_else(|| vec![0; n_layers]);
        let total_needed: usize = targets
            .iter()
            .zip(&cur_lens)
            .map(|(t, c)| t.saturating_sub(*c))
            .sum();
        let mut fresh: Vec<u32> = Vec::with_capacity(total_needed);
        if total_needed > 0 && !alloc_blocks_with_cache_eviction(state, total_needed, &mut fresh) {
            return false;
        }
        let owned = m
            .sessions
            .entry(seq_id)
            .or_insert_with(|| vec![Vec::new(); n_layers]);
        let mut next = 0usize;
        for layer in 0..n_layers {
            let take = targets[layer].saturating_sub(owned[layer].len());
            owned[layer].extend_from_slice(&fresh[next..next + take]);
            next += take;
        }
        {
            let GgufMultiSeqState {
                sessions,
                combined_host,
                combined_dirty,
                seq_logical,
                ..
            } = &mut *m;
            let owned = &sessions[&seq_id];
            let base = slot * seq_stride;
            for layer in 0..n_layers {
                let layer_base = base + layer * state.max_blocks_per_seq;
                let blocks = &owned[layer];
                for l in 0..logical {
                    combined_host[layer_base + l] = blocks[l % blocks.len()];
                }
            }
            *combined_dirty = true;
            seq_logical.insert(seq_id, logical);
        }
    }

    *block_table_device_out = m
        .combined_device
        .as_ref()
        .map(|table| *table.device_ptr() as *mut u32)
        .unwrap_or(std::ptr::null_mut());
    *n_blocks_out = logical as u32;
    true
}

#[cfg(feature = "gguf-cuda-shared-kv")]
unsafe extern "C" fn gguf_kapsl_kv_commit_seq(
    user_data: *mut std::ffi::c_void,
    block_table_device_out: *mut *mut u32,
) -> bool {
    if user_data.is_null() || block_table_device_out.is_null() {
        return false;
    }
    let state = &*(user_data as *mut GgufSharedKvPoolState);
    let mut m = state.multi.lock().unwrap();

    if m.combined_dirty || m.combined_device.is_none() {
        let table = match state.handle.pool.device().htod_sync_copy(&m.combined_host) {
            Ok(t) => t,
            Err(_) => return false,
        };
        m.combined_device = Some(table);
        m.combined_dirty = false;
    }

    let Some(table) = m.combined_device.as_ref() else {
        return false;
    };
    *block_table_device_out = *table.device_ptr() as *mut u32;
    true
}

// ── Reserve callback (with prefix cache) ─────────────────────────────────────

#[cfg(feature = "gguf-cuda-shared-kv")]
unsafe extern "C" fn gguf_kapsl_kv_reserve_prefix(
    user_data: *mut std::ffi::c_void,
    _session_id: u64,
    tokens: *const i32,
    n_tokens: u32,
    block_table_device_out: *mut *mut u32,
    n_logical_blocks_out: *mut u32,
    n_prefix_hits_out: *mut u32,
) -> bool {
    let _timing = GgufTimingGuard::new(&GGUF_TIMING_KV_RESERVE_CALLS, &GGUF_TIMING_KV_RESERVE_US);
    if user_data.is_null()
        || tokens.is_null()
        || block_table_device_out.is_null()
        || n_logical_blocks_out.is_null()
        || n_prefix_hits_out.is_null()
    {
        return false;
    }
    let state = &*(user_data as *mut GgufSharedKvPoolState);

    // ── Windowed (Phase 2) path ───────────────────────────────────────────
    // No prefix lookup or promotion: ring blocks are overwritten in place as
    // the context slides, so cached prefixes would silently go stale.
    if let Some(windowed) = &state.windowed {
        let previous_tokens = state.windowed_inner.lock().unwrap().n_tokens_reserved;
        return match gguf_windowed_reserve_impl(state, windowed, n_tokens as usize) {
            Some((table_ptr, logical)) => {
                unsafe {
                    *block_table_device_out = table_ptr;
                    *n_logical_blocks_out = logical as u32;
                    *n_prefix_hits_out = 0;
                }
                if previous_tokens > 0 {
                    let delta = (n_tokens as usize).saturating_sub(previous_tokens).max(1);
                    record_gguf_decode_work_metrics(&state.metrics, 1, delta as u64);
                }
                true
            }
            None => false,
        };
    }

    let Some(cache) = &state.prefix_cache else {
        // Prefix cache disabled — fall back to plain reserve.
        return gguf_kapsl_kv_reserve(
            user_data,
            _session_id,
            n_tokens,
            block_table_device_out,
            n_logical_blocks_out,
        );
    };

    let token_slice = std::slice::from_raw_parts(tokens, n_tokens as usize);
    let block_size = state.handle.pool.block_size().max(1);
    let fp = state.model_fingerprint;

    let n_logical = (n_tokens as usize).div_ceil(block_size).max(1);
    if n_logical > state.max_blocks_per_seq {
        return false;
    }
    let n_tokens_usize = n_tokens as usize;
    let n_complete_blocks = n_tokens_usize / block_size;
    let mut computed_hashes: Option<Vec<u64>> = None;

    // 1. Same-session decode usually grows one token at a time. If the current
    // reservation already covers the requested logical block count, keep its
    // block table. Only refresh hashes when another full block has completed;
    // otherwise avoid re-hashing the entire growing token prefix on every token.
    if state.evicted.lock().unwrap().is_none() {
        let mut inner = state.inner.lock().unwrap();
        let had_reservation = inner.block_table.is_some() && inner.n_logical_blocks > 0;
        if inner.n_logical_blocks >= n_logical {
            let previous_tokens = inner.n_tokens_reserved;
            let delta_tokens = n_tokens_usize.saturating_sub(previous_tokens).max(1);
            let complete_prefix_tokens = inner.n_prefix_hits.saturating_mul(block_size);
            let would_evaluate = n_tokens_usize.saturating_sub(complete_prefix_tokens);
            let saved_tokens = would_evaluate.saturating_sub(delta_tokens);
            if inner.model_fingerprint != fp || inner.all_hashes.len() < n_complete_blocks {
                let all_hashes = computed_hashes.get_or_insert_with(|| {
                    PrefixBlockCache::compute_prefix_hashes_i32(fp, token_slice, block_size)
                });
                inner.all_hashes = all_hashes.clone();
            }
            inner.model_fingerprint = fp;
            inner.n_tokens_reserved = inner.n_tokens_reserved.max(n_tokens_usize);
            let table_ptr = match inner.block_table.as_ref() {
                Some(table) => *table.device_ptr() as *mut u32,
                None => {
                    return false;
                }
            };
            unsafe {
                *block_table_device_out = table_ptr;
                *n_logical_blocks_out = inner.n_logical_blocks as u32;
                *n_prefix_hits_out = inner.n_prefix_hits as u32;
            }
            record_gguf_decode_work_metrics(&state.metrics, 1, delta_tokens as u64);
            record_gguf_partial_reuse_metrics(&state.metrics, 1, saved_tokens as u64);
            GGUF_TIMING_KV_FAST_PATH_CALLS.fetch_add(1, Ordering::Relaxed);
            return true;
        }
        if had_reservation
            && n_logical > inner.n_logical_blocks
            && inner.block_table_host.len() == state.n_layers * state.max_blocks_per_seq
        {
            let previous_tokens = inner.n_tokens_reserved;
            let delta_tokens = n_tokens_usize.saturating_sub(previous_tokens).max(1);
            let complete_prefix_tokens = inner.n_prefix_hits.saturating_mul(block_size);
            let would_evaluate = n_tokens_usize.saturating_sub(complete_prefix_tokens);
            let saved_tokens = would_evaluate.saturating_sub(delta_tokens);
            let old_logical = inner.n_logical_blocks;
            let old_n_new = inner.n_new_logical;
            let additional_logical = n_logical.saturating_sub(old_logical);
            let needed_phys = additional_logical.saturating_mul(state.n_layers);

            let mut added_blocks = Vec::with_capacity(needed_phys);
            if !alloc_blocks_with_cache_eviction(state, needed_phys, &mut added_blocks) {
                return false;
            }

            let new_n_new = old_n_new.saturating_add(additional_logical);
            let mut owned_blocks = vec![0u32; new_n_new.saturating_mul(state.n_layers)];
            for layer in 0..state.n_layers {
                for pos in 0..old_n_new {
                    owned_blocks[layer * new_n_new + pos] =
                        inner.owned_blocks[layer * old_n_new + pos];
                }
                for pos in 0..additional_logical {
                    owned_blocks[layer * new_n_new + old_n_new + pos] =
                        added_blocks[layer * additional_logical + pos];
                }
            }

            let mut host_table = inner.block_table_host.clone();
            for layer in 0..state.n_layers {
                for pos in 0..additional_logical {
                    host_table[layer * state.max_blocks_per_seq + old_logical + pos] =
                        added_blocks[layer * additional_logical + pos];
                }
            }

            let table = match state.handle.pool.device().htod_sync_copy(&host_table) {
                Ok(table) => table,
                Err(_) => {
                    for block in added_blocks {
                        state.handle.pool.free_block(block);
                    }
                    return false;
                }
            };
            let table_ptr = *table.device_ptr() as *mut u32;

            inner.owned_blocks = owned_blocks;
            inner.n_new_logical = new_n_new;
            if inner.model_fingerprint != fp || inner.all_hashes.len() < n_complete_blocks {
                let all_hashes = computed_hashes.get_or_insert_with(|| {
                    PrefixBlockCache::compute_prefix_hashes_i32(fp, token_slice, block_size)
                });
                inner.all_hashes = all_hashes.clone();
            }
            inner.model_fingerprint = fp;
            inner.block_table = Some(table);
            inner.block_table_host = host_table;
            inner.n_logical_blocks = n_logical;
            inner.n_tokens_reserved = n_tokens_usize;

            unsafe {
                *block_table_device_out = table_ptr;
                *n_logical_blocks_out = inner.n_logical_blocks as u32;
                *n_prefix_hits_out = inner.n_prefix_hits as u32;
            }
            record_gguf_decode_work_metrics(&state.metrics, 1, delta_tokens as u64);
            record_gguf_partial_reuse_metrics(&state.metrics, 1, saved_tokens as u64);
            GGUF_TIMING_KV_EXTEND_CALLS.fetch_add(1, Ordering::Relaxed);
            return true;
        }
    }

    // 2. Compute chained hashes for all complete prefix blocks only after the
    // same-session fast path misses.
    let all_hashes = computed_hashes.unwrap_or_else(|| {
        PrefixBlockCache::compute_prefix_hashes_i32(fp, token_slice, block_size)
    });

    // 3. Lookup prefix cache — collect a contiguous run of hits.
    let hits = {
        let mut c = match cache.lock() {
            Ok(c) => c,
            Err(_) => return false,
        };
        c.lookup(fp, &all_hashes)
    };
    GGUF_TIMING_KV_LOOKUP_CALLS.fetch_add(1, Ordering::Relaxed);
    let n_hits = hits.len();
    let previous_tokens = {
        let inner = state.inner.lock().unwrap();
        inner.n_tokens_reserved
    };
    if previous_tokens > 0 && n_tokens_usize > previous_tokens {
        let complete_prefix_tokens = n_hits.saturating_mul(block_size);
        let evaluated_tokens = n_tokens_usize.saturating_sub(complete_prefix_tokens).max(1);
        record_gguf_decode_work_metrics(&state.metrics, 1, evaluated_tokens as u64);
    }

    // 4. Check for CPU-evicted state: use restore path if available.
    let evicted_opt = state.evicted.lock().unwrap().take();
    if let Some(evicted) = evicted_opt {
        // Build the per-hit block-id lists for the restore helper.
        let hit_block_ids: Vec<Vec<u32>> = hits.iter().map(|h| h.block_ids.clone()).collect();
        return gguf_restore_from_cpu(
            state,
            evicted,
            n_logical,
            n_hits,
            &hit_block_ids,
            block_table_device_out,
            n_logical_blocks_out,
            n_prefix_hits_out,
        );
    }

    // 5. Allocate fresh blocks for positions that missed the cache.
    let n_new = n_logical.saturating_sub(n_hits);
    let needed_phys = n_new.saturating_mul(state.n_layers);
    if needed_phys > state.handle.cap() {
        return false;
    }

    let mut new_blocks: Vec<u32> = Vec::with_capacity(needed_phys);
    if !alloc_blocks_with_cache_eviction(state, needed_phys, &mut new_blocks) {
        // Release prefix borrows before failing.
        if let Ok(mut c) = cache.lock() {
            for h in &hits {
                c.release(fp, h.block_hash);
            }
        }
        return false;
    }

    // 6. Build the block table: cached blocks first, then new blocks.
    let mut host_table = vec![0u32; state.n_layers * state.max_blocks_per_seq];
    for layer in 0..state.n_layers {
        for (hit_pos, hit) in hits.iter().enumerate() {
            let blk = hit
                .block_ids
                .get(layer)
                .copied()
                .unwrap_or_else(|| hit.block_ids.first().copied().unwrap_or(0));
            host_table[layer * state.max_blocks_per_seq + hit_pos] = blk;
        }
        for new_pos in 0..n_new {
            host_table[layer * state.max_blocks_per_seq + n_hits + new_pos] =
                new_blocks[layer * n_new + new_pos];
        }
    }

    let table = match state.handle.pool.device().htod_sync_copy(&host_table) {
        Ok(t) => t,
        Err(_) => {
            for b in new_blocks {
                state.handle.pool.free_block(b);
            }
            if let Ok(mut c) = cache.lock() {
                for h in &hits {
                    c.release(fp, h.block_hash);
                }
            }
            return false;
        }
    };
    let table_ptr = *table.device_ptr() as *mut u32;

    let mut inner = state.inner.lock().unwrap();
    gguf_release_reservation_inner(
        &mut inner,
        &state.prefix_cache,
        &state.handle.pool,
        state.n_layers,
    );
    inner.owned_blocks = new_blocks;
    inner.n_new_logical = n_new;
    inner.n_promoted_logical = 0;
    inner.all_hashes = all_hashes;
    inner.n_prefix_hits = n_hits;
    inner.model_fingerprint = fp;
    inner.block_table = Some(table);
    inner.block_table_host = host_table;
    inner.n_logical_blocks = n_logical;
    inner.n_tokens_reserved = n_tokens_usize;

    *block_table_device_out = table_ptr;
    *n_logical_blocks_out = n_logical as u32;
    *n_prefix_hits_out = n_hits as u32;
    true
}

// ── Release callback ──────────────────────────────────────────────────────────

/// Inline release logic, callable with explicit pool access.
#[cfg(feature = "gguf-cuda-shared-kv")]
fn gguf_release_reservation_inner(
    inner: &mut GgufSharedKvReservation,
    prefix_cache: &Option<Arc<Mutex<PrefixBlockCache>>>,
    pool: &Arc<GpuBlockPool>,
    n_layers: usize,
) {
    // Release prefix cache borrows.
    if let Some(cache) = prefix_cache {
        if let Ok(mut c) = cache.lock() {
            let n_borrow = inner.n_prefix_hits + inner.n_promoted_logical;
            let fp = inner.model_fingerprint;
            for &hash in inner.all_hashes.get(..n_borrow).unwrap_or(&[]) {
                c.release(fp, hash);
            }
        }
    }
    // Free un-promoted owned blocks.
    let n_new = inner.n_new_logical;
    for layer in 0..n_layers {
        for pos in inner.n_promoted_logical..n_new {
            let idx = layer * n_new + pos;
            if let Some(&block) = inner.owned_blocks.get(idx) {
                pool.free_block(block);
            }
        }
    }
    *inner = GgufSharedKvReservation::default();
}

#[cfg(feature = "gguf-cuda-shared-kv")]
unsafe extern "C" fn gguf_kapsl_kv_release(user_data: *mut std::ffi::c_void, session_id: u64) {
    if user_data.is_null() {
        return;
    }
    let state = &*(user_data as *mut GgufSharedKvPoolState);

    // Multi-sequence path: free this sequence slot's persistent blocks and clear
    // its block-table region so a future request reusing the seq_id starts fresh.
    // No-op in single-sequence mode (sessions is empty).
    {
        let mut m = state.multi.lock().unwrap();
        m.seq_logical.remove(&session_id);
        if let Some(owned) = m.sessions.remove(&session_id) {
            for layer in owned {
                for b in layer {
                    state.handle.pool.free_block(b);
                }
            }
            let slot = session_id as usize;
            if slot < state.n_seq_slots {
                let seq_stride = state.n_layers * state.max_blocks_per_seq;
                let base = slot * seq_stride;
                for x in &mut m.combined_host[base..base + seq_stride] {
                    *x = 0;
                }
                m.combined_dirty = true;
            }
        }
    }

    // Windowed (Phase 2) reservation: free every per-layer block.
    {
        let mut w = state.windowed_inner.lock().unwrap();
        for layer in std::mem::take(&mut w.layers) {
            for b in layer {
                state.handle.pool.free_block(b);
            }
        }
        *w = GgufWindowedReservation::default();
    }

    let mut inner = state.inner.lock().unwrap();
    gguf_release_reservation_inner(
        &mut inner,
        &state.prefix_cache,
        &state.handle.pool,
        state.n_layers,
    );
}

#[cfg(feature = "gguf-cuda-shared-kv")]
unsafe extern "C" fn gguf_kapsl_kv_touch(
    user_data: *mut std::ffi::c_void,
    _session_id: u64,
) -> bool {
    !user_data.is_null()
}

// ─── Backend ──────────────────────────────────────────────────────────────────

/// Which KV-cache implementation a loaded GGUF engine actually ended up using.
/// Recorded at load time so diagnostics and benchmarks can report the real path
/// instead of inferring it from build features or env vars (which can be wrong:
/// shared-KV silently falls back to native for unsupported architectures).
#[cfg(feature = "gguf-cuda-shared-kv")]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum GgufKvPath {
    /// Not yet loaded.
    Unloaded,
    /// Kapsl paged external KV pool (shared-KV).
    SharedKv,
    /// llama.cpp's native in-process KV cache (unified / ISWA / recurrent).
    Native,
}

#[cfg(feature = "gguf-cuda-shared-kv")]
impl GgufKvPath {
    /// Stable lowercase label for logs, info endpoints, and benchmark output.
    pub fn as_str(self) -> &'static str {
        match self {
            GgufKvPath::Unloaded => "unloaded",
            GgufKvPath::SharedKv => "shared-kv",
            GgufKvPath::Native => "native",
        }
    }

    fn as_u8(self) -> u8 {
        match self {
            GgufKvPath::Unloaded => 0,
            GgufKvPath::SharedKv => 1,
            GgufKvPath::Native => 2,
        }
    }
}

pub struct GgufBackend {
    #[cfg(feature = "gguf")]
    inner: Option<GgufInner>,
    metrics: Arc<Mutex<EngineMetrics>>,
    device_id: usize,
    #[cfg(feature = "gguf-cuda-shared-kv")]
    pool_slot: Arc<Mutex<Option<GpuPoolHandle>>>,
    /// Runtime-owned backing pool. Geometry is not known until `load()`.
    #[cfg(feature = "gguf-cuda-shared-kv")]
    device_pool: Option<(Arc<GpuDevicePool>, PoolOwner)>,
    /// Active KV path, set during `load()`. Read lock-free via `active_kv_path`.
    #[cfg(feature = "gguf-cuda-shared-kv")]
    kv_path: Arc<std::sync::atomic::AtomicU8>,
}

#[cfg(feature = "gguf")]
struct GgufInner {
    weights: Arc<GgufWeights>,
    request_tx: std_mpsc::Sender<GgufRequest>,
    scheduler_thread: Option<std::thread::JoinHandle<()>>,
    max_concurrent: usize,
}

impl GgufBackend {
    pub fn new() -> Self {
        Self {
            #[cfg(feature = "gguf")]
            inner: None,
            metrics: Arc::new(Mutex::new(EngineMetrics::new())),
            device_id: 0,
            #[cfg(feature = "gguf-cuda-shared-kv")]
            pool_slot: Arc::new(Mutex::new(None)),
            #[cfg(feature = "gguf-cuda-shared-kv")]
            device_pool: None,
            #[cfg(feature = "gguf-cuda-shared-kv")]
            kv_path: Arc::new(std::sync::atomic::AtomicU8::new(
                GgufKvPath::Unloaded.as_u8(),
            )),
        }
    }

    pub fn new_on_device(device_id: usize) -> Self {
        Self {
            device_id,
            ..Self::new()
        }
    }

    #[cfg(feature = "gguf-cuda-shared-kv")]
    pub fn new_cuda_shared_kv(device_id: usize, handle: Option<GpuPoolHandle>) -> Self {
        Self {
            #[cfg(feature = "gguf")]
            inner: None,
            metrics: Arc::new(Mutex::new(EngineMetrics::new())),
            device_id,
            pool_slot: Arc::new(Mutex::new(handle)),
            device_pool: None,
            kv_path: Arc::new(std::sync::atomic::AtomicU8::new(
                GgufKvPath::Unloaded.as_u8(),
            )),
        }
    }

    /// Construct the shared-KV backend as a client of a runtime-owned device
    /// pool. The model-specific KV view is created after geometry is read.
    #[cfg(feature = "gguf-cuda-shared-kv")]
    pub fn new_cuda_device_pool(
        device_id: usize,
        device_pool: Arc<GpuDevicePool>,
        model_id: u32,
    ) -> Self {
        Self::new_cuda_device_pool_for_replica(device_id, device_pool, model_id, 0)
    }

    /// Construct the shared-KV backend with stable model/replica attribution.
    #[cfg(feature = "gguf-cuda-shared-kv")]
    pub fn new_cuda_device_pool_for_replica(
        device_id: usize,
        device_pool: Arc<GpuDevicePool>,
        model_id: u32,
        replica_id: u32,
    ) -> Self {
        let mut backend = Self::new_cuda_shared_kv(device_id, None);
        backend.device_pool = Some((
            device_pool,
            PoolOwner::gguf(model_id, replica_id, PoolAllocationClass::KvCache),
        ));
        backend
    }

    #[cfg(feature = "gguf-cuda-shared-kv")]
    pub fn with_pool_handle(self, handle: GpuPoolHandle) -> Self {
        *self.pool_slot.lock().unwrap() = Some(handle);
        self
    }

    #[cfg(feature = "gguf-cuda-shared-kv")]
    pub fn pool_handle(&self) -> Option<GpuPoolHandle> {
        self.pool_slot.lock().unwrap().clone()
    }

    fn metrics_snapshot(&self) -> EngineMetrics {
        self.metrics
            .lock()
            .map(|m| m.clone())
            .unwrap_or_else(|_| EngineMetrics::new())
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

    /// Resolved scheduling priority (0 = latency-critical, higher = lower). The
    /// scheduler stamps this on the request's metadata before dispatch; the
    /// internal `run_scheduler` promotes lower values ahead of the FIFO order.
    /// Defaults to 1 (throughput) when unset.
    fn priority(request: &InferenceRequest) -> u8 {
        request
            .metadata
            .as_ref()
            .and_then(|m| m.priority)
            .unwrap_or(1)
    }
}

impl Default for GgufBackend {
    fn default() -> Self {
        Self::new()
    }
}

/// Index of the highest-priority (lowest value) item, breaking ties toward the
/// earliest position so requests at the same priority keep FIFO order. Returns
/// `None` for an empty iterator.
#[cfg(feature = "gguf")]
fn highest_priority_index<I: IntoIterator<Item = u8>>(priorities: I) -> Option<usize> {
    priorities
        .into_iter()
        .enumerate()
        .min_by_key(|&(idx, priority)| (priority, idx))
        .map(|(idx, _)| idx)
}

// ─── Scheduler loop ───────────────────────────────────────────────────────────

#[cfg(feature = "gguf")]
fn token_piece_bytes(model: &LlamaModel, token: LlamaToken) -> Result<Vec<u8>, EngineError> {
    let _timing = GgufTimingGuard::new(&GGUF_TIMING_PIECE_CALLS, &GGUF_TIMING_PIECE_US);
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
fn release_sequence_slot(
    ctx: &mut llama_cpp_2::context::LlamaContext,
    available_ids: &mut Vec<i32>,
    seq_id: i32,
) {
    if let Ok(seq_id_u32) = u32::try_from(seq_id) {
        let _ = ctx.clear_kv_cache_seq(Some(seq_id_u32), None, None);
    }
    available_ids.push(seq_id);
}

#[cfg(feature = "gguf")]
// Sequence activation keeps the llama context and all per-request state
// together; splitting these arguments would only move them into a transient
// parameter bag at this internal boundary.
#[allow(clippy::too_many_arguments)]
fn finish_or_activate_prefilled_sequence(
    metrics: &Arc<Mutex<EngineMetrics>>,
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
    suppress_eos_sampler: bool,
    session_id: Option<String>,
    absorbed: Vec<LlamaToken>,
) {
    let mut output = Vec::with_capacity((max_tokens.max(0) as usize).saturating_mul(4));
    let mut stop_filter = GgufStopFilter::new();

    if max_tokens <= 0 || (first_tok == eos_token && min_tokens <= 0) {
        release_sequence_slot(ctx, available_ids, seq_id);
        record_gguf_token_metrics(metrics, prompt_len.max(0) as usize, 0);
        response.finish(output);
        return;
    }

    let Some(piece) = first_piece else {
        release_sequence_slot(ctx, available_ids, seq_id);
        response.send_error(EngineError::backend("missing first token piece"));
        return;
    };

    // The first generated token sits at generated-count 0, so it may only stop
    // on a marker once the min_tokens floor is already satisfied.
    let first_may_stop = min_tokens <= 0;
    match stop_filter.push_piece(&response, &mut output, piece.to_vec(), first_may_stop) {
        GgufEmitResult::Continue => {}
        GgufEmitResult::Stopped => {
            release_sequence_slot(ctx, available_ids, seq_id);
            record_gguf_token_metrics(metrics, prompt_len.max(0) as usize, 1);
            response.finish(output);
            return;
        }
        GgufEmitResult::Disconnected => {
            release_sequence_slot(ctx, available_ids, seq_id);
            return;
        }
    }

    if max_tokens <= 1 {
        if !stop_filter.flush(&response, &mut output) {
            release_sequence_slot(ctx, available_ids, seq_id);
            return;
        }
        release_sequence_slot(ctx, available_ids, seq_id);
        record_gguf_token_metrics(metrics, prompt_len.max(0) as usize, 1);
        response.finish(output);
    } else {
        active.push(ActiveSeq {
            seq_id,
            pos: prompt_len,
            prompt_tokens: prompt_len.max(0) as usize,
            n_generated: 1,
            max_tokens,
            min_tokens,
            suppress_eos_sampler,
            last_token: first_tok,
            output,
            stop_filter,
            response,
            error: None,
            session_id,
            absorbed,
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
                session_id: candidate.session_id,
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
    release_sequence_slot(ctx, available_ids, pref.seq_id);
    pref.response.send_error(EngineError::backend(message));

    for copy in pref.copies.drain(..) {
        release_sequence_slot(ctx, available_ids, copy.seq_id);
        copy.response.send_error(EngineError::backend(message));
    }
}

#[cfg(feature = "gguf")]
fn update_gguf_metrics(
    metrics: &Arc<Mutex<EngineMetrics>>,
    config: GgufServingConfig,
    waiting: &VecDeque<GgufRequest>,
    pending: &VecDeque<PendingPrefill>,
    active: &[ActiveSeq],
    batch_tokens: usize,
) {
    let total_cells = config.total_ctx();
    let used_cells = if config.uses_state_space_memory {
        // Recurrent/hybrid memory holds one constant-size state per active
        // sequence, not a per-token cell — the position-scaled formula below
        // would report KV usage growing as generation progresses even though
        // real memory stays flat, which would feed a false pressure signal to
        // autoscaling/admission. Approximate as one used cell per in-flight
        // sequence instead.
        pending.len().saturating_add(active.len()).min(total_cells)
    } else {
        let pending_cells = pending
            .iter()
            .map(|pref| pref.next_token.min(pref.tokens.len()))
            .sum::<usize>();
        let active_cells = active
            .iter()
            .map(|seq| seq.pos.max(0) as usize)
            .sum::<usize>();
        pending_cells.saturating_add(active_cells).min(total_cells)
    };
    let capacity_bytes = total_cells.saturating_mul(config.kv_bytes_per_cell);

    if let Ok(mut snapshot) = metrics.lock() {
        snapshot.batch_size = batch_tokens;
        snapshot.queue_depth = waiting.len() + pending.len();
        snapshot.kv_cache_blocks_total = total_cells;
        snapshot.kv_cache_blocks_free = total_cells.saturating_sub(used_cells);
        snapshot.kv_cache_sequences = pending.len() + active.len();
        snapshot.kv_cache_bytes_capacity = capacity_bytes;
        snapshot.kv_cache_bytes_used = used_cells.saturating_mul(config.kv_bytes_per_cell);
        snapshot.refresh_timestamp();
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
    config: GgufServingConfig,
    metrics: Arc<Mutex<EngineMetrics>>,
    ready_tx: std_mpsc::SyncSender<Result<(), String>>,
    #[cfg(feature = "gguf-cuda-shared-kv")] mut shared_kv_pool: Option<GgufSharedKvPool>,
) {
    #[allow(unused_mut)]
    let mut config = config;
    #[cfg(feature = "gguf-cuda-shared-kv")]
    if shared_kv_pool.is_some() && config.exact_prompt_kv_reuse {
        log::warn!("[gguf] exact prompt KV reuse is unsupported by shared-KV; disabling it");
        config.exact_prompt_kv_reuse = false;
    }
    let total_ctx = config.total_ctx();
    let n_ctx = match NonZeroU32::new(total_ctx as u32) {
        Some(v) => v,
        None => {
            log::error!("[gguf] invalid total_ctx=0");
            let _ = ready_tx.send(Err("invalid GGUF total context size".to_string()));
            return;
        }
    };
    let n_batch = config.n_batch();

    #[allow(unused_mut)]
    let mut ctx_params = LlamaContextParams::default()
        .with_n_ctx(Some(n_ctx))
        .with_n_batch(n_batch)
        .with_n_ubatch(n_batch)
        .with_n_seq_max(config.max_concurrent as u32)
        .with_offload_kqv(true)
        .with_flash_attention_policy(LLAMA_FLASH_ATTN_TYPE_AUTO);
    #[cfg(feature = "gguf-cuda-shared-kv")]
    if let Some(pool) = shared_kv_pool.as_mut() {
        ctx_params = unsafe { ctx_params.with_kapsl_kv_pool_raw(pool.desc_ptr(), 1) };
        log::info!(
            "[gguf] Kapsl shared KV pool attached (blocks={}, block_size={}, max_blocks_per_seq={})",
            pool.state.handle.pool.total_blocks(),
            pool.state.handle.pool.block_size(),
            pool.state.max_blocks_per_seq
        );
    }

    let mut ctx = match model.new_context(&backend, ctx_params) {
        Ok(c) => c,
        Err(e) => {
            log::error!("[gguf] Failed to create shared context: {e}");
            let _ = ready_tx.send(Err(format!("GGUF context creation failed: {e}")));
            return;
        }
    };
    log::info!(
        "[gguf] Shared context ready (n_ctx={total_ctx}, max_concurrent={})",
        config.max_concurrent
    );
    log::info!("[gguf] Prefill chunk size={}", config.prefill_chunk_size);
    log::info!(
        "[gguf] Exact prompt KV reuse={}",
        config.exact_prompt_kv_reuse
    );
    log::info!(
        "[gguf] Estimated KV capacity={} MiB ({} bytes/cell)",
        total_ctx.saturating_mul(config.kv_bytes_per_cell) / (1024 * 1024),
        config.kv_bytes_per_cell
    );
    let eos_token = model.token_eos();
    let mut samplers = GgufBackendSamplers::new(&model, config.max_concurrent, eos_token);
    // Pre-install a sampler for every seq slot up front so the installed-sampler
    // set (and thus the compute-graph topology) is stable from the first decode.
    // `set_sampler` flips `sched_need_reserve`, forcing a worst-case
    // `sched_reserve()` (sized for n_seq_max); installing lazily and clearing on
    // retire churned that reservation on every request. Keeping all slots
    // installed for the context lifetime makes it a one-time warmup cost, and
    // `build_sampling` already routes inactive installed samplers to row 0.
    for seq_id in 0..config.max_concurrent as i32 {
        if !samplers.set_for_sequence(&mut ctx, seq_id, false) {
            log::warn!("[gguf] failed to pre-install sampler for seq slot {seq_id}");
        }
    }
    let batch_cap = n_batch as usize;
    let mut batch = LlamaBatch::new(batch_cap, 1);

    // Recurrent-state checkpoint cache for SSM/hybrid models (Phase 4),
    // opt-in via KAPSL_GGUF_SSM_STATE_CACHE. None for attention models (they
    // have the block-level prefix cache) and when the flag is off.
    let mut ssm_cache: Option<SsmStateCache> = gguf_ssm_state_cache_config(config);

    // seq_id pool: 0..max_concurrent are valid sequence identifiers for the KV cache.
    let mut available_ids: Vec<i32> = (0..config.max_concurrent as i32).rev().collect();
    let mut waiting: VecDeque<GgufRequest> = VecDeque::new();
    let mut pending: VecDeque<PendingPrefill> = VecDeque::new();
    let mut active: Vec<ActiveSeq> = Vec::with_capacity(config.max_concurrent);
    update_gguf_metrics(&metrics, config, &waiting, &pending, &active, 0);
    if ready_tx.send(Ok(())).is_err() {
        log::warn!("[gguf] loader dropped before scheduler readiness was reported");
        return;
    }

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
        // Priority-aware, not strict FIFO: pull the lowest-priority-value
        // (latency-critical first) waiting request, breaking ties toward the
        // earliest arrival so requests at the same level keep FIFO order.
        while !available_ids.is_empty() {
            let Some(idx) = highest_priority_index(waiting.iter().map(|r| r.priority)) else {
                break;
            };
            let req = waiting.remove(idx).expect("index came from waiting");
            let seq_id = available_ids.pop().unwrap();

            if req.tokens.len() as u32 > config.ctx_per_seq {
                req.response.send_error(EngineError::invalid_input(format!(
                    "prompt has {} tokens, exceeding ctx_per_seq={}",
                    req.tokens.len(),
                    config.ctx_per_seq
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
            if !samplers.set_for_sequence(&mut ctx, seq_id, req.min_tokens > 0) {
                req.response
                    .send_error(EngineError::backend("failed to install sampler"));
                available_ids.push(seq_id);
                continue;
            }

            // Skip prompt tokens already absorbed by a cached recurrent-state
            // snapshot (the session's resume state, else the longest chunk
            // checkpoint). 0 when the cache is off, missed, or failed (a
            // failed restore clears the seq slot so a full prefill stays
            // correct).
            let next_token = match ssm_cache.as_mut() {
                Some(cache) => {
                    let restored = restore_ssm_state(
                        &mut ctx,
                        cache,
                        req.session_id.as_deref(),
                        &req.tokens,
                        seq_id,
                    );
                    if restored > 0 {
                        record_gguf_partial_reuse_metrics(&metrics, 1, restored as u64);
                    }
                    restored
                }
                None => 0,
            };

            pending.push_back(PendingPrefill {
                seq_id,
                tokens: req.tokens,
                next_token,
                max_tokens: req.max_tokens,
                min_tokens: req.min_tokens,
                session_id: req.session_id,
                response: req.response,
                copies: Vec::new(),
            });
        }
        update_gguf_metrics(&metrics, config, &waiting, &pending, &active, 0);

        // ── 3. If completely idle, block for the next request ─────────────────
        if waiting.is_empty() && pending.is_empty() && active.is_empty() {
            #[cfg(feature = "gguf-cuda-shared-kv")]
            if let Some(pool) = shared_kv_pool.as_mut() {
                if pool.state.evict_when_idle {
                    pool.evict_to_cpu();
                }
                // Log cross-device pressure snapshot at debug level.
                let kv_heads = pool.state.handle.pool.num_kv_heads();
                let head_dim = pool.state.handle.pool.head_dim();
                let device_id = pool.desc.device_id as usize;
                let sched = gguf_global_kv_scheduler().lock().unwrap();
                log::debug!(
                    "[gguf] idle: device {} pressure={:.2} free_blocks={} ({}h×{}d), \
                     cross-device registered=[{}]",
                    device_id,
                    sched.device_pressure(device_id),
                    sched.device_free_blocks(device_id, kv_heads, head_dim),
                    kv_heads,
                    head_dim,
                    sched
                        .registered_devices()
                        .iter()
                        .map(|d| d.to_string())
                        .collect::<Vec<_>>()
                        .join(", "),
                );
            }
            match request_rx.recv() {
                Ok(req) => waiting.push_back(req),
                Err(_) => break 'main,
            }
            if !config.queue_delay.is_zero() {
                let deadline = Instant::now() + config.queue_delay;
                while waiting.len() < config.max_concurrent {
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
            update_gguf_metrics(&metrics, config, &waiting, &pending, &active, 0);
            continue;
        }

        // ── 4. Build batch: multiple prefills (if any) + all active decode tokens ───
        // Reserve `active.len()` slots for decode tokens so prefills don't crowd them out.
        let batch_build_timing =
            GgufTimingGuard::new(&GGUF_TIMING_BATCH_BUILD_CALLS, &GGUF_TIMING_BATCH_BUILD_US);
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
            if config.exact_prompt_kv_reuse {
                coalesce_exact_prompt_copies(&mut pref, &mut pending);
            }

            let remaining_prompt = pref.tokens.len().saturating_sub(pref.next_token);
            let chunk_len = remaining_prompt
                .min(config.prefill_chunk_size)
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
        drop(batch_build_timing);

        if batch.n_tokens() == 0 {
            update_gguf_metrics(&metrics, config, &waiting, &pending, &active, 0);
            continue;
        }
        update_gguf_metrics(
            &metrics,
            config,
            &waiting,
            &pending,
            &active,
            batch.n_tokens().max(0) as usize,
        );

        // ── 5. Execute one forward pass for all sequences in the batch ────────
        let decode_timing = GgufTimingGuard::new(&GGUF_TIMING_DECODE_CALLS, &GGUF_TIMING_DECODE_US);
        let decode_result = ctx.decode(&mut batch);
        drop(decode_timing);
        if let Err(e) = decode_result {
            log::error!("[gguf] decode error: {e}");
            for seq in active.drain(..) {
                seq.response
                    .send_error(EngineError::backend("decode failed"));
                release_sequence_slot(&mut ctx, &mut available_ids, seq.seq_id);
            }
            for pref in partial_prefills.drain(..) {
                fail_pending_prefill(&mut ctx, &mut available_ids, pref, "decode failed");
            }
            for (pref, _) in completed_prefills.drain(..) {
                fail_pending_prefill(&mut ctx, &mut available_ids, pref, "decode failed");
            }
            update_gguf_metrics(&metrics, config, &waiting, &pending, &active, 0);
            continue;
        }

        for pref in partial_prefills.drain(..) {
            // Prefill pauses exactly on chunk boundaries, which is where
            // recurrent-state checkpoints live; the state now covers
            // tokens[..next_token].
            if let Some(cache) = ssm_cache.as_mut() {
                maybe_snapshot_ssm_state(&ctx, cache, &pref.tokens, pref.next_token, pref.seq_id);
            }
            pending.push_back(pref);
        }
        // Completed prefills whose prompt length lands exactly on a chunk
        // boundary produce that boundary's checkpoint here (the partial loop
        // never sees them). Session requests additionally save the post-prompt
        // state as a session resume point: unlike the post-retirement snapshot
        // it survives reply-seam retokenization drift, because the next turn's
        // prompt begins with this prompt's text verbatim.
        if let Some(cache) = ssm_cache.as_mut() {
            for (pref, _) in completed_prefills.iter() {
                maybe_snapshot_ssm_state(&ctx, cache, &pref.tokens, pref.next_token, pref.seq_id);
                if let Some(sid) = pref.session_id.as_deref() {
                    snapshot_ssm_session_state(&ctx, cache, sid, &pref.tokens, pref.seq_id);
                }
            }
        }

        // ── 5b. Promote newly computed prefix KV blocks to cache ──────────────
        #[cfg(feature = "gguf-cuda-shared-kv")]
        if let Some(pool) = shared_kv_pool.as_mut() {
            pool.promote_if_pending();
        }

        // ── 6. Sample each newly prefilled sequence and move it to active ──────
        for (mut pref, last_pos) in completed_prefills.drain(..) {
            // EOS is skipped during greedy sampling when min_tokens > 0, so first_tok is
            // guaranteed to be a real content token whenever min_tokens is nonzero.
            let first_tok = samplers.sample_token(&ctx, last_pos);
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
                        release_sequence_slot(&mut ctx, &mut available_ids, pref.seq_id);
                        pref.response.send_error(e);
                        for copy in pref.copies.drain(..) {
                            release_sequence_slot(&mut ctx, &mut available_ids, copy.seq_id);
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
                        release_sequence_slot(&mut ctx, &mut available_ids, copy.seq_id);
                        copy.response
                            .send_error(EngineError::backend("KV cache copy failed"));
                        continue;
                    }
                }
                ready_copies.push(copy);
            }

            let suppress_eos_sampler = pref.min_tokens > 1;
            if !suppress_eos_sampler && !samplers.set_for_sequence(&mut ctx, pref.seq_id, false) {
                release_sequence_slot(&mut ctx, &mut available_ids, pref.seq_id);
                pref.response
                    .send_error(EngineError::backend("failed to install sampler"));
                continue;
            }

            // Absorbed-token tracking for SSM session resume snapshots: the
            // leader takes the prompt Vec, copies (identical prompt) clone it.
            let absorbed_for_copies: Vec<LlamaToken> =
                if ssm_cache.is_some() && !ready_copies.is_empty() {
                    pref.tokens.clone()
                } else {
                    Vec::new()
                };
            let leader_absorbed = if ssm_cache.is_some() {
                std::mem::take(&mut pref.tokens)
            } else {
                Vec::new()
            };

            finish_or_activate_prefilled_sequence(
                &metrics,
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
                suppress_eos_sampler,
                pref.session_id.take(),
                leader_absorbed,
            );

            for copy in ready_copies {
                let copy_suppress_eos_sampler = copy.min_tokens > 1;
                if !copy_suppress_eos_sampler
                    && !samplers.set_for_sequence(&mut ctx, copy.seq_id, false)
                {
                    release_sequence_slot(&mut ctx, &mut available_ids, copy.seq_id);
                    copy.response
                        .send_error(EngineError::backend("failed to install sampler"));
                    continue;
                }

                finish_or_activate_prefilled_sequence(
                    &metrics,
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
                    copy_suppress_eos_sampler,
                    copy.session_id,
                    absorbed_for_copies.clone(),
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

            // This step's decode absorbed the token fed back at batch-build
            // time into the recurrent state. Non-empty only while the SSM
            // state cache tracks this sequence (it starts as the prompt).
            if !seq.absorbed.is_empty() {
                seq.absorbed.push(seq.last_token);
            }

            // The active sampler chain suppresses EOS until min_tokens is reached.
            let next_tok = samplers.sample_token(&ctx, batch_pos);
            seq.pos += 1;

            let eos_and_ready = next_tok == eos_token && seq.n_generated >= seq.min_tokens;
            let next_generated = seq.n_generated + 1;
            let max_reached = next_generated >= seq.max_tokens;
            // A chat stop-marker may only retire the sequence once the
            // min_tokens floor is met — mirrors the EOS gate above.
            let may_stop = seq.n_generated >= seq.min_tokens;

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
                    match seq.stop_filter.push_piece(
                        &seq.response,
                        &mut seq.output,
                        piece,
                        may_stop,
                    ) {
                        GgufEmitResult::Continue | GgufEmitResult::Stopped => {
                            seq.last_token = next_tok;
                            seq.n_generated = next_generated;
                        }
                        GgufEmitResult::Disconnected => {}
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
                match seq
                    .stop_filter
                    .push_piece(&seq.response, &mut seq.output, piece, may_stop)
                {
                    GgufEmitResult::Disconnected => {
                        to_retire.push(i);
                    }
                    GgufEmitResult::Stopped => {
                        seq.last_token = next_tok;
                        seq.n_generated = next_generated;
                        to_retire.push(i);
                    }
                    GgufEmitResult::Continue => {
                        seq.last_token = next_tok;
                        seq.n_generated = next_generated;
                        if seq.suppress_eos_sampler && next_generated >= seq.min_tokens {
                            if samplers.set_for_sequence(&mut ctx, seq.seq_id, false) {
                                seq.suppress_eos_sampler = false;
                            } else {
                                seq.error = Some(EngineError::backend("failed to install sampler"));
                                to_retire.push(i);
                            }
                        }
                    }
                }
            }
        }

        for &i in to_retire.iter().rev() {
            let done = active.remove(i);
            // Save the retiring state as the session's resume point before the
            // slot release clears it. Error retirements are excluded — their
            // state may not match the absorbed-token record.
            if let (Some(cache), Some(sid), true) = (
                ssm_cache.as_mut(),
                done.session_id.as_deref(),
                done.error.is_none(),
            ) {
                snapshot_ssm_session_state(&ctx, cache, sid, &done.absorbed, done.seq_id);
            }
            release_sequence_slot(&mut ctx, &mut available_ids, done.seq_id);
            if let Some(error) = done.error {
                done.response.send_error(error);
            } else {
                let mut done = done;
                if !done.stop_filter.flush(&done.response, &mut done.output) {
                    continue;
                }
                record_gguf_token_metrics(
                    &metrics,
                    done.prompt_tokens,
                    done.n_generated.max(0) as usize,
                );
                done.response.finish(done.output);
            }
        }
        update_gguf_metrics(&metrics, config, &waiting, &pending, &active, 0);
        gguf_timing_maybe_log(false);
    }

    gguf_timing_maybe_log(true);
    log::info!("[gguf] Scheduler thread exiting");
}

// ─── Engine impl ──────────────────────────────────────────────────────────────

#[cfg(feature = "gguf")]
#[async_trait]
impl Engine for GgufBackend {
    fn planned_memory(&self, model_path: &Path) -> Result<MemoryReport, EngineError> {
        let bytes = std::fs::metadata(model_path)
            .map_err(|error| EngineError::backend(format!("stat GGUF model: {error}")))?
            .len() as usize;
        let mut report = MemoryReport {
            allocations: vec![
                MemoryAllocation {
                    allocation_id: gguf_allocation_id(model_path),
                    domain: MemoryDomain::Cuda {
                        device_id: self.device_id,
                    },
                    class: MemoryAllocationClass::PersistentWeights,
                    source: MemoryAllocationSource::BackendManaged,
                    bytes,
                },
                MemoryAllocation {
                    allocation_id: format!("gguf-scratch:{}", gguf_model_key(model_path).display()),
                    domain: MemoryDomain::Cuda {
                        device_id: self.device_id,
                    },
                    class: MemoryAllocationClass::TransientWorkspace,
                    source: MemoryAllocationSource::BackendManaged,
                    bytes: (bytes / 8).max(256 * 1024 * 1024),
                },
            ],
        };
        #[cfg(feature = "gguf-cuda-shared-kv")]
        let runtime_kv = self.device_pool.is_some();
        #[cfg(not(feature = "gguf-cuda-shared-kv"))]
        let runtime_kv = false;
        report.push(MemoryAllocation {
            allocation_id: format!("gguf-kv:{}", gguf_model_key(model_path).display()),
            domain: MemoryDomain::Cuda {
                device_id: self.device_id,
            },
            class: MemoryAllocationClass::KvCache,
            source: if runtime_kv {
                MemoryAllocationSource::RuntimeManaged
            } else {
                MemoryAllocationSource::BackendManaged
            },
            bytes: 0,
        });
        Ok(report)
    }

    fn planned_external_device_memory(
        &self,
        model_path: &Path,
    ) -> Result<ExternalDeviceMemoryReport, EngineError> {
        let bytes = std::fs::metadata(model_path)
            .map_err(|e| EngineError::backend(format!("stat GGUF model: {e}")))?
            .len() as usize;
        Ok(ExternalDeviceMemoryReport {
            allocations: vec![
                ExternalDeviceMemory {
                    allocation_id: gguf_allocation_id(model_path),
                    device_id: self.device_id,
                    bytes,
                },
                ExternalDeviceMemory {
                    allocation_id: format!("gguf-scratch:{}", gguf_model_key(model_path).display()),
                    device_id: self.device_id,
                    bytes: (bytes / 8).max(256 * 1024 * 1024),
                },
            ],
        })
    }

    async fn load(&mut self, model_path: &Path) -> Result<(), EngineError> {
        let model_path_key = gguf_model_key(model_path);

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
            let device_id = self.device_id;
            let backend_for_load = global_gguf_backend()?;
            let backend_for_closure = Arc::clone(&backend_for_load);
            let (model, n_ctx_train) = tokio::task::spawn_blocking(move || {
                let params = gguf_model_params(device_id)?;
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
                n_ctx_train,
                allocation_id: gguf_allocation_id(&model_path_key),
            });
            gguf_weights_cache()
                .lock()
                .unwrap()
                .insert(model_path_key, Arc::downgrade(&arc));
            arc
        };

        let config = GgufServingConfig::from_model(&weights.model, weights.n_ctx_train);
        #[cfg(feature = "gguf-cuda-shared-kv")]
        let shared_kv_pool = if let Some(reason) = gguf_shared_kv_disable_reason(&weights.model) {
            // Architecture/geometry the uniform shared-KV pool cannot represent
            // (or an explicit operator override): skip the pool and let
            // llama.cpp build its native KV cache for this model.
            log::info!(
                "[gguf] kv_path=native shared-KV disabled for {} ({reason}); using llama.cpp native KV path",
                model_path.display()
            );
            self.kv_path.store(
                GgufKvPath::Native.as_u8(),
                std::sync::atomic::Ordering::Relaxed,
            );
            None
        } else {
            let n_layers = weights.model.n_layer().max(1) as usize;
            let n_head_kv = weights.model.n_head_kv().max(1) as usize;
            let head_dim_k = weights.model.n_embd_head_k().max(1) as usize;
            let head_dim_v = weights.model.n_embd_head_v().max(1) as usize;
            if head_dim_k != head_dim_v {
                return Err(EngineError::backend(format!(
                    "shared KV pool requires equal K/V head dims, got K={head_dim_k} V={head_dim_v}"
                )));
            }
            let head_dim = head_dim_k;
            let block_size = 16usize;
            // Windowed KV for SWA layers (Phase 2): ring-capped per-layer
            // allocation, opt-in via KAPSL_GGUF_SWA_WINDOWED_KV. None keeps the
            // uniform full allocation.
            let windowed = gguf_windowed_kv_config(&weights.model, config, block_size);
            // Auto-select the least-loaded registered device when KAPSL_GGUF_AUTO_DEVICE=1.
            let effective_device_id = if self.device_pool.is_some() {
                self.device_id
            } else {
                gguf_select_device(self.device_id, n_head_kv, head_dim)
            };
            if effective_device_id != self.device_id {
                log::info!(
                    "[gguf] auto-device: selected device {} over {} (more free {n_head_kv}h×{head_dim}d blocks)",
                    effective_device_id, self.device_id,
                );
            }
            let requested_blocks =
                gguf_shared_kv_block_count(n_layers, config, block_size, windowed.as_ref());
            let handle = {
                let mut slot = self.pool_slot.lock().unwrap();
                if let Some((device_pool, owner)) = self.device_pool.as_ref() {
                    let bytes_per_block = 2usize
                        .saturating_mul(n_head_kv)
                        .saturating_mul(block_size)
                        .saturating_mul(head_dim)
                        .saturating_mul(std::mem::size_of::<half::f16>());
                    let required_bytes = requested_blocks.saturating_mul(bytes_per_block);
                    let current_quota = device_pool.owner_quota(*owner);
                    device_pool
                        .set_owner_quota(
                            *owner,
                            required_bytes,
                            current_quota.max_bytes.max(required_bytes),
                        )
                        .map_err(|e| {
                            EngineError::backend(format!(
                                "shared KV capacity guarantee admission failed: {e}"
                            ))
                        })?;
                    let pool = Arc::new(
                        GpuBlockPool::from_device_pool(
                            Arc::clone(device_pool),
                            *owner,
                            requested_blocks,
                            block_size,
                            n_head_kv,
                            head_dim,
                        )
                        .map_err(|e| EngineError::backend(format!("shared KV view: {e}")))?,
                    );
                    if pool.total_blocks() < requested_blocks {
                        return Err(EngineError::backend(format!(
                            "shared KV capacity admission failed: requested {requested_blocks} blocks for context={} concurrency={}, but only {} blocks are guaranteed",
                            config.ctx_per_seq,
                            config.max_concurrent,
                            pool.total_blocks()
                        )));
                    }
                    let handle = GpuPoolHandle::private(pool);
                    *slot = Some(handle.clone());
                    handle
                } else if let Some(handle) = slot.as_ref() {
                    if handle.pool.is_compatible(n_head_kv, head_dim) {
                        handle.clone()
                    } else {
                        log::warn!(
                            "[gguf] Shared KV pool geometry mismatch ({}h x {}d vs {}h x {}d); creating private pool",
                            handle.pool.num_kv_heads(),
                            handle.pool.head_dim(),
                            n_head_kv,
                            head_dim
                        );
                        let device = CudaDevice::new(effective_device_id)
                            .map_err(|e| EngineError::backend(format!("CUDA: {e}")))?;
                        let pool = Arc::new(
                            GpuBlockPool::new(
                                device,
                                requested_blocks,
                                block_size,
                                n_head_kv,
                                head_dim,
                            )
                            .map_err(|e| EngineError::backend(format!("shared KV pool: {e}")))?,
                        );
                        let handle = GpuPoolHandle::private(pool);
                        *slot = Some(handle.clone());
                        handle
                    }
                } else {
                    let device = CudaDevice::new(effective_device_id)
                        .map_err(|e| EngineError::backend(format!("CUDA: {e}")))?;
                    let pool = Arc::new(
                        GpuBlockPool::new(
                            device,
                            requested_blocks,
                            block_size,
                            n_head_kv,
                            head_dim,
                        )
                        .map_err(|e| EngineError::backend(format!("shared KV pool: {e}")))?,
                    );
                    let handle = GpuPoolHandle::private(pool);
                    *slot = Some(handle.clone());
                    handle
                }
            };
            {
                // Compute a stable model fingerprint from architecture parameters.
                let model_fingerprint = {
                    use std::collections::hash_map::DefaultHasher;
                    use std::hash::{Hash, Hasher};
                    let mut h = DefaultHasher::new();
                    n_layers.hash(&mut h);
                    n_head_kv.hash(&mut h);
                    head_dim.hash(&mut h);
                    head_dim_v.hash(&mut h);
                    h.finish()
                };
                // Build a prefix block cache sized at 1/4 of the pool's logical capacity.
                let prefix_cache_cap = {
                    let pool_blocks = handle.pool.total_blocks();
                    let env_cap = std::env::var("KAPSL_GGUF_PREFIX_CACHE_BLOCKS")
                        .ok()
                        .and_then(|v| v.parse::<usize>().ok())
                        .filter(|&v| v > 0);
                    env_cap.unwrap_or_else(|| (pool_blocks / n_layers / 4).max(1))
                };
                let prefix_cache = Some(Arc::new(Mutex::new(PrefixBlockCache::new(
                    prefix_cache_cap,
                ))));
                log::info!(
                    "[gguf] Prefix KV cache enabled: capacity={} logical positions",
                    prefix_cache_cap
                );
                self.kv_path.store(
                    GgufKvPath::SharedKv.as_u8(),
                    std::sync::atomic::Ordering::Relaxed,
                );
                log::info!(
                    "[gguf] kv_path=shared-kv Kapsl paged external KV pool active on device {effective_device_id}"
                );
                Some(GgufSharedKvPool::new(
                    handle,
                    self.metrics.clone(),
                    effective_device_id,
                    n_layers,
                    config.ctx_per_seq as usize,
                    config.max_concurrent,
                    prefix_cache,
                    model_fingerprint,
                    windowed,
                ))
            }
        };
        if let Ok(mut snapshot) = self.metrics.lock() {
            snapshot.kv_cache_blocks_total = config.total_ctx();
            snapshot.kv_cache_blocks_free = config.total_ctx();
            snapshot.kv_cache_bytes_capacity =
                config.total_ctx().saturating_mul(config.kv_bytes_per_cell);
            snapshot.kv_cache_bytes_used = 0;
            snapshot.refresh_timestamp();
        }

        let (tx, rx) = std_mpsc::channel::<GgufRequest>();
        let model_clone = Arc::clone(&weights.model);
        let backend_clone = Arc::clone(&weights.backend);
        let metrics = Arc::clone(&self.metrics);
        let (ready_tx, ready_rx) = std_mpsc::sync_channel(1);
        let scheduler_thread = std::thread::spawn(move || {
            run_scheduler(
                model_clone,
                backend_clone,
                rx,
                config,
                metrics,
                ready_tx,
                #[cfg(feature = "gguf-cuda-shared-kv")]
                shared_kv_pool,
            );
        });

        match ready_rx.recv_timeout(Duration::from_secs(300)) {
            Ok(Ok(())) => {}
            Ok(Err(error)) => {
                let _ = scheduler_thread.join();
                return Err(EngineError::backend(error));
            }
            Err(error) => {
                return Err(EngineError::backend(format!(
                    "GGUF scheduler did not become ready: {error}"
                )));
            }
        }

        log::info!(
            "[gguf] Scheduler started: max_concurrent={}, ctx_per_seq={}",
            config.max_concurrent,
            config.ctx_per_seq
        );

        self.inner = Some(GgufInner {
            weights,
            request_tx: tx,
            scheduler_thread: Some(scheduler_thread),
            max_concurrent: config.max_concurrent,
        });
        Ok(())
    }

    fn actual_external_device_memory(&self) -> ExternalDeviceMemoryReport {
        let Some(inner) = self.inner.as_ref() else {
            return ExternalDeviceMemoryReport::default();
        };
        let fallback_device = self.device_id;
        let allocations = inner
            .weights
            .model
            .device_memory()
            .into_iter()
            .enumerate()
            .filter_map(|(index, (name, bytes))| {
                if bytes == 0 {
                    return None;
                }
                let device_id = name
                    .strip_prefix("CUDA")
                    .and_then(|suffix| suffix.parse::<usize>().ok())
                    .unwrap_or(fallback_device + index);
                Some(ExternalDeviceMemory {
                    allocation_id: inner.weights.allocation_id.clone(),
                    device_id,
                    bytes: bytes as usize,
                })
            })
            .collect();
        ExternalDeviceMemoryReport { allocations }
    }

    fn actual_memory(&self) -> MemoryReport {
        let external = self.actual_external_device_memory();
        let mut report = MemoryReport {
            allocations: external
                .allocations
                .iter()
                .map(|allocation| MemoryAllocation {
                    allocation_id: allocation.allocation_id.clone(),
                    domain: MemoryDomain::Cuda {
                        device_id: allocation.device_id,
                    },
                    class: MemoryAllocationClass::PersistentWeights,
                    source: MemoryAllocationSource::BackendManaged,
                    bytes: allocation.bytes,
                })
                .collect(),
        };
        let metrics = self.metrics_snapshot();
        let fallback_domain = external
            .allocations
            .first()
            .map(|allocation| MemoryDomain::Cuda {
                device_id: allocation.device_id,
            })
            .unwrap_or(MemoryDomain::Host);
        #[cfg(feature = "gguf-cuda-shared-kv")]
        let runtime_kv = self.device_pool.is_some()
            && self.kv_path.load(std::sync::atomic::Ordering::Acquire)
                == GgufKvPath::SharedKv.as_u8();
        #[cfg(not(feature = "gguf-cuda-shared-kv"))]
        let runtime_kv = false;
        report.push(MemoryAllocation {
            allocation_id: "gguf:active-kv".to_string(),
            domain: fallback_domain,
            class: MemoryAllocationClass::KvCache,
            source: if runtime_kv {
                MemoryAllocationSource::RuntimeManaged
            } else {
                MemoryAllocationSource::BackendManaged
            },
            bytes: metrics.kv_cache_bytes_used,
        });
        report
    }

    fn infer(&self, request: &InferenceRequest) -> Result<BinaryTensorPacket, EngineError> {
        let inner = self.inner.as_ref().ok_or(EngineError::ModelNotLoaded)?;
        let prompt = gguf_prepare_prompt(&inner.weights.model, Self::extract_prompt(request)?)?;
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
                priority: Self::priority(request),
                session_id: request.session_id.clone(),
                response: GgufResponse::Final(resp_tx),
            })
            .map_err(|_| EngineError::backend("gguf scheduler disconnected"))?;

        let data = match resp_rx.recv() {
            Ok(result) => result?,
            Err(_) => {
                return Err(EngineError::backend(
                    "gguf scheduler disconnected before producing a response",
                ));
            }
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

        let prompt = match Self::extract_prompt(request)
            .and_then(|prompt| gguf_prepare_prompt(&inner.weights.model, prompt))
        {
            Ok(prompt) => prompt,
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
                priority: Self::priority(request),
                session_id: request.session_id.clone(),
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
        if let Some(inner) = self.inner.take() {
            let GgufInner {
                weights,
                request_tx,
                mut scheduler_thread,
                ..
            } = inner;
            drop(request_tx);
            if let Some(thread) = scheduler_thread.take() {
                if thread.join().is_err() {
                    log::warn!("[gguf] Scheduler thread panicked during unload");
                }
            }
            drop(weights);
        }
        if let Ok(mut metrics) = self.metrics.lock() {
            *metrics = EngineMetrics::new();
        }
        log::info!("[gguf] Backend unloaded");
    }

    fn metrics(&self) -> EngineMetrics {
        self.metrics_snapshot()
    }

    /// The GGUF backend runs its own continuous batcher (`run_scheduler`) that
    /// multiplexes concurrent requests across `max_concurrent` sequence slots.
    /// The scheduler must dispatch requests individually rather than coalesce
    /// them via `infer_batch`, so it advertises self-batching (and keeps
    /// `max_batch()` at the default 1).
    fn self_batches(&self) -> bool {
        true
    }

    fn batching_policy(&self) -> BatchingPolicy {
        let max_requests = self
            .inner
            .as_ref()
            .map(|inner| inner.max_concurrent)
            .unwrap_or_else(max_concurrent);
        BatchingPolicy::continuous(max_requests).with_priority_support()
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

#[cfg(all(test, feature = "gguf"))]
mod gguf_stop_filter_tests {
    use super::{GgufEmitResult, GgufResponse, GgufStopFilter};
    use std::sync::mpsc;

    fn drain_chunks(rx: &mpsc::Receiver<Result<Vec<u8>, super::EngineError>>) -> String {
        rx.try_iter()
            .map(|chunk| String::from_utf8(chunk.expect("chunk")).expect("utf8"))
            .collect::<Vec<_>>()
            .join("")
    }

    #[test]
    fn stop_filter_hides_split_gemma_user_turn_marker() {
        let (tx, rx) = mpsc::channel();
        let response = GgufResponse::Stream(tx);
        let mut output = Vec::new();
        let mut filter = GgufStopFilter::new();

        assert_eq!(
            filter.push_piece(&response, &mut output, b"answer ".to_vec(), true),
            GgufEmitResult::Continue
        );
        assert_eq!(
            filter.push_piece(&response, &mut output, b"<start_of".to_vec(), true),
            GgufEmitResult::Continue
        );
        assert_eq!(
            filter.push_piece(
                &response,
                &mut output,
                b"_turn>user\nignored".to_vec(),
                true
            ),
            GgufEmitResult::Stopped
        );

        assert_eq!(drain_chunks(&rx), "answer ");
    }

    #[test]
    fn stop_filter_flushes_text_without_stop_marker() {
        let (tx, rx) = mpsc::channel();
        let response = GgufResponse::Stream(tx);
        let mut output = Vec::new();
        let mut filter = GgufStopFilter::new();

        assert_eq!(
            filter.push_piece(&response, &mut output, b"hello".to_vec(), true),
            GgufEmitResult::Continue
        );
        assert!(filter.flush(&response, &mut output));

        assert_eq!(drain_chunks(&rx), "hello");
    }

    #[test]
    fn stop_filter_keeps_going_past_marker_below_floor() {
        // Below the min_tokens floor (may_stop = false), a turn-marker must not
        // retire the sequence: surrounding text is emitted, the marker bytes are
        // dropped, and generation continues.
        let (tx, rx) = mpsc::channel();
        let response = GgufResponse::Stream(tx);
        let mut output = Vec::new();
        let mut filter = GgufStopFilter::new();

        assert_eq!(
            filter.push_piece(&response, &mut output, b"answer<|im_end|>".to_vec(), false),
            GgufEmitResult::Continue
        );
        assert_eq!(
            filter.push_piece(&response, &mut output, b"more".to_vec(), false),
            GgufEmitResult::Continue
        );
        assert!(filter.flush(&response, &mut output));

        // The marker text never leaks; the text around it is preserved.
        assert_eq!(drain_chunks(&rx), "answermore");
    }

    #[test]
    fn stop_filter_drops_marker_below_floor_then_stops_at_floor() {
        // First marker (below floor) is dropped; once at the floor the next
        // marker latches the filter as usual.
        let (tx, rx) = mpsc::channel();
        let response = GgufResponse::Stream(tx);
        let mut output = Vec::new();
        let mut filter = GgufStopFilter::new();

        assert_eq!(
            filter.push_piece(&response, &mut output, b"a<|im_end|>b".to_vec(), false),
            GgufEmitResult::Continue
        );
        assert_eq!(
            filter.push_piece(&response, &mut output, b"c<|im_end|>d".to_vec(), true),
            GgufEmitResult::Stopped
        );
        // Subsequent pushes stay stopped.
        assert_eq!(
            filter.push_piece(&response, &mut output, b"e".to_vec(), true),
            GgufEmitResult::Stopped
        );

        assert_eq!(drain_chunks(&rx), "abc");
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
        self.metrics_snapshot()
    }

    fn health_check(&self) -> Result<(), EngineError> {
        Err(EngineError::backend(
            "GGUF support not compiled in (enable the 'gguf' feature)",
        ))
    }
}

#[cfg(all(test, feature = "gguf"))]
mod tests {
    use super::{highest_priority_index, GgufServingConfig};
    use std::time::Duration;

    #[test]
    fn highest_priority_index_prefers_lowest_value_then_earliest() {
        // Empty → None.
        assert_eq!(highest_priority_index(std::iter::empty()), None);

        // Single element.
        assert_eq!(highest_priority_index([3u8]), Some(0));

        // Latency-critical (0) jumps ahead of throughput (1), regardless of
        // arrival order.
        assert_eq!(highest_priority_index([1u8, 1, 0, 1]), Some(2));

        // Ties on priority break toward the earliest arrival (FIFO within a
        // level): two 0s at indices 1 and 3 → picks 1.
        assert_eq!(highest_priority_index([2u8, 0, 1, 0]), Some(1));

        // All equal → front (pure FIFO fallback).
        assert_eq!(highest_priority_index([5u8, 5, 5]), Some(0));
    }

    fn test_config(
        ctx_per_seq: u32,
        max_concurrent: usize,
        prefill_chunk_size: usize,
    ) -> GgufServingConfig {
        GgufServingConfig {
            max_concurrent,
            ctx_per_seq,
            queue_delay: Duration::ZERO,
            prefill_chunk_size,
            exact_prompt_kv_reuse: false,
            kv_bytes_per_cell: 0,
            uses_state_space_memory: false,
        }
    }

    #[test]
    fn gguf_batch_capacity_tracks_prefill_chunk_instead_of_full_ctx() {
        let config = test_config(2048, 1, 512);

        assert_eq!(config.total_ctx(), 2048);
        assert_eq!(config.n_batch(), 513);
    }

    #[test]
    fn gguf_batch_capacity_reserves_decode_slots_for_all_sequences() {
        let config = test_config(4096, 8, 128);

        assert_eq!(config.total_ctx(), 32_768);
        assert_eq!(config.n_batch(), 136);
    }

    // ── shared-KV architecture guard (classify_shared_kv_support) ──────────────

    use super::classify_shared_kv_support;

    /// Build (`has_key`, `pos_int`) closures over a fixed set of
    /// architecture-scoped metadata keys → values, mirroring how
    /// `gguf_shared_kv_disable_reason` prefixes lookups with `<arch>.`.
    fn classify_swa(
        arch: &str,
        n_swa: u32,
        allow_swa: bool,
        keys: &[(&str, &str)],
    ) -> Option<String> {
        let keys: Vec<(String, String)> = keys
            .iter()
            .map(|(k, v)| ((*k).to_string(), (*v).to_string()))
            .collect();
        let lookup = |suffix: &str| {
            keys.iter()
                .find(|(k, _)| k == suffix)
                .map(|(_, v)| v.clone())
        };
        let has_key = |suffix: &str| lookup(suffix).is_some();
        let pos_int = |suffix: &str| {
            lookup(suffix)
                .and_then(|v| v.trim().parse::<i64>().ok())
                .is_some_and(|n| n > 0)
        };
        classify_shared_kv_support(arch, n_swa, allow_swa, has_key, pos_int)
    }

    fn classify(arch: &str, n_swa: u32, keys: &[(&str, &str)]) -> Option<String> {
        classify_swa(arch, n_swa, false, keys)
    }

    #[test]
    fn shared_kv_allows_plain_causal_transformer() {
        // Qwen2 / TinyLlama-style: uniform KV, no SWA, no latent/recurrent state.
        assert!(classify("qwen2", 0, &[("attention.head_count_kv", "8")]).is_none());
    }

    #[test]
    fn shared_kv_rejects_unknown_architecture() {
        assert!(classify("", 0, &[]).is_some());
    }

    #[test]
    fn shared_kv_rejects_standard_sliding_window_via_metadata() {
        // Gemma-style: window advertised only through the GGUF key, n_swa not yet
        // surfaced (defends the metadata heuristic independently of n_swa).
        let reason = classify("gemma2", 0, &[("attention.sliding_window", "4096")])
            .expect("SWA model must be rejected");
        assert!(reason.contains("sliding-window"), "reason: {reason}");
    }

    #[test]
    fn shared_kv_rejects_chunked_attention_without_metadata_key() {
        // Llama 4 hard-codes swa_type=CHUNKED / n_swa=8192 when the
        // `attention.sliding_window` key is absent — only n_swa catches it.
        let reason =
            classify("llama4", 8192, &[]).expect("chunked-attention model must be rejected");
        assert!(reason.contains("n_swa=8192"), "reason: {reason}");
    }

    #[test]
    fn shared_kv_rejects_multi_head_latent_attention() {
        let reason = classify("deepseek2", 0, &[("attention.kv_lora_rank", "512")])
            .expect("MLA model must be rejected");
        assert!(reason.contains("latent"), "reason: {reason}");
    }

    #[test]
    fn shared_kv_rejects_recurrent_and_hybrid_state_space() {
        for (arch, key) in [
            ("mamba", "ssm.state_size"),
            ("jamba", "ssm.conv_kernel"),
            ("rwkv6", "wkv.head_size"),
        ] {
            let reason = classify(arch, 0, &[(key, "16")])
                .unwrap_or_else(|| panic!("{arch} must be rejected"));
            assert!(reason.contains("recurrent/hybrid"), "reason: {reason}");
        }
    }

    // ── state-space memory detection (feeds KV metrics, Phase 3) ────────────────

    use super::gguf_uses_state_space_memory;

    #[test]
    fn state_space_memory_detected_by_any_ssm_or_wkv_key() {
        let has = |present: &'static [&'static str]| move |suffix: &str| present.contains(&suffix);
        assert!(gguf_uses_state_space_memory(has(&["ssm.state_size"])));
        assert!(gguf_uses_state_space_memory(has(&["ssm.conv_kernel"])));
        assert!(gguf_uses_state_space_memory(has(&["wkv.head_size"])));
        // Hybrid archs (Jamba/Granite) carry both attention and ssm keys.
        assert!(gguf_uses_state_space_memory(has(&[
            "attention.head_count",
            "ssm.state_size"
        ])));
    }

    #[test]
    fn state_space_memory_not_detected_for_plain_attention() {
        assert!(!gguf_uses_state_space_memory(has_none));
        let has = |present: &'static [&'static str]| move |suffix: &str| present.contains(&suffix);
        assert!(!gguf_uses_state_space_memory(has(&[
            "attention.head_count",
            "attention.sliding_window"
        ])));
    }

    fn has_none(_: &str) -> bool {
        false
    }

    #[test]
    fn shared_kv_sliding_window_key_value_zero_is_not_swa() {
        // A present-but-zero window must not be treated as SWA on the metadata
        // path (n_swa is the source of truth and is 0 here).
        assert!(classify("qwen2", 0, &[("attention.sliding_window", "0")]).is_none());
    }

    #[test]
    fn shared_kv_allow_swa_admits_verified_gemma_family_only() {
        // Phase 1 opt-in admits SWA only for the quality-verified Gemma family.
        assert!(classify_swa("gemma2", 0, true, &[("attention.sliding_window", "4096")]).is_none());
        assert!(classify_swa("gemma3", 8192, true, &[]).is_none());
    }

    #[test]
    fn shared_kv_allow_swa_still_gates_unverified_families() {
        // Even with the opt-in set, non-allowlisted SWA archs stay native:
        // cohere2 regresses (NoPE), llama4/phi3 are not eval-verified yet.
        assert!(classify_swa(
            "cohere2",
            4096,
            true,
            &[("attention.sliding_window", "4096")]
        )
        .is_some());
        assert!(classify_swa("llama4", 8192, true, &[]).is_some());
        assert!(classify_swa("phi3", 0, true, &[("attention.sliding_window", "2048")]).is_some());
    }

    #[test]
    fn shared_kv_allow_swa_still_rejects_recurrent_and_mla() {
        // The SWA opt-in must not admit architectures the kernel cannot serve.
        assert!(classify_swa("deepseek2", 0, true, &[("attention.kv_lora_rank", "512")]).is_some());
        assert!(classify_swa("mamba", 0, true, &[("ssm.state_size", "16")]).is_some());
    }

    // ── SWA windowed KV ring math (Phase 2) ─────────────────────────────────────

    use super::{swa_window_blocks, windowed_layer_capacity};

    #[test]
    fn swa_window_blocks_covers_window_plus_ubatch() {
        // Gemma3 1B defaults: n_swa=512, n_batch=512+32, block_size=16.
        assert_eq!(swa_window_blocks(512, 544, 16), 67);
        // Gemma2: n_swa=4096.
        assert_eq!(swa_window_blocks(4096, 544, 16), 291);
        // Degenerate block size clamps to 1.
        assert_eq!(swa_window_blocks(4, 2, 0), 7);
    }

    #[test]
    fn windowed_layer_capacity_caps_only_windowed_layers() {
        assert_eq!(windowed_layer_capacity(500, Some(67)), 67);
        assert_eq!(windowed_layer_capacity(50, Some(67)), 50);
        assert_eq!(windowed_layer_capacity(500, None), 500);
    }

    /// The ring must never recycle a block that any live query can still read.
    ///
    /// Brute-force the invariant behind `swa_window_blocks`: after writing up
    /// to position `p_max`, every query in the current ubatch
    /// (`p_max - n_ubatch + 1 ..= p_max`) must find all keys in its standard
    /// window (`p - n_swa + 1 ..= p`, the loosest of the three window types)
    /// in logical blocks whose ring slot has not been overwritten — i.e.
    /// within the last `window_blocks` logical blocks written.
    #[test]
    fn swa_window_ring_never_recycles_live_blocks() {
        for &(block_size, n_swa, n_ubatch) in &[
            (4usize, 6usize, 3usize),
            (4, 8, 8),
            (16, 512, 544),
            (16, 4096, 544),
            (1, 5, 2),
            (7, 13, 29),
        ] {
            let wb = swa_window_blocks(n_swa, n_ubatch, block_size);
            for p_max in 0..(4 * (n_swa + n_ubatch) + 64) {
                let newest_block = p_max / block_size;
                for p in p_max.saturating_sub(n_ubatch - 1)..=p_max {
                    let win_start = p.saturating_sub(n_swa - 1);
                    let oldest_live_block = win_start / block_size;
                    assert!(
                        oldest_live_block + wb > newest_block,
                        "recycled live block: bs={block_size} n_swa={n_swa} \
                         n_ubatch={n_ubatch} wb={wb} p_max={p_max} p={p}"
                    );
                }
            }
        }
    }

    /// The block-table ring mapping (`pos % ring_len`) must keep all blocks in
    /// any `window_blocks`-sized span of logical positions distinct, and reuse
    /// exactly the slot from one ring revolution ago.
    #[test]
    fn swa_ring_mapping_wraps_without_collisions() {
        let wb = 5usize;
        let ring: Vec<u32> = (100..100 + wb as u32).collect();
        let entry = |pos: usize| ring[pos % ring.len()];
        for start in 0..3 * wb {
            let span: Vec<u32> = (start..start + wb).map(entry).collect();
            let mut dedup = span.clone();
            dedup.sort_unstable();
            dedup.dedup();
            assert_eq!(dedup.len(), wb, "collision within one window span");
        }
        for pos in wb..4 * wb {
            assert_eq!(
                entry(pos),
                entry(pos - wb),
                "slot not reused after one revolution"
            );
        }
    }

    // ── SSM recurrent-state checkpoint cache (Phase 4) ──────────────────────

    use super::{SsmStateCache, SsmStateEntry};
    use llama_cpp_2::token::LlamaToken;

    fn toks(ids: &[i32]) -> Vec<LlamaToken> {
        ids.iter().copied().map(LlamaToken).collect()
    }

    /// Insert an entry directly, bypassing the size/eviction policy, so lookup
    /// tests don't depend on insert behavior.
    fn put(cache: &mut SsmStateCache, hash: u64, n_tokens: usize, bytes: usize) {
        cache.total_bytes += bytes;
        cache.entries.insert(
            hash,
            SsmStateEntry {
                n_tokens,
                data: vec![0u8; bytes],
                last_used: cache.tick,
            },
        );
    }

    #[test]
    fn ssm_prefix_hashes_chain_and_align_to_chunks() {
        let cache = SsmStateCache::new(4, 1 << 20, 1 << 20);
        let a = cache.prefix_hashes(&toks(&[1, 2, 3, 4, 5, 6, 7, 8, 9]));
        // 9 tokens, chunk 4 -> checkpoints at 4 and 8 only.
        assert_eq!(a.len(), 2);
        // Shared prefix -> shared leading hash; divergence changes the rest.
        let b = cache.prefix_hashes(&toks(&[1, 2, 3, 4, 99, 6, 7, 8]));
        assert_eq!(a[0], b[0]);
        assert_ne!(a[1], b[1]);
    }

    #[test]
    fn ssm_lookup_picks_longest_strict_prefix_checkpoint() {
        let mut cache = SsmStateCache::new(4, 1 << 20, 1 << 20);
        let prompt = toks(&[1, 2, 3, 4, 5, 6, 7, 8, 9]);
        let hashes = cache.prefix_hashes(&prompt);
        put(&mut cache, hashes[0], 4, 10);
        put(&mut cache, hashes[1], 8, 10);

        // Longest checkpoint (8) is a strict prefix of the 9-token prompt.
        assert_eq!(cache.lookup_longest(&hashes, 9), Some((hashes[1], 8)));

        // For an 8-token prompt the position-8 checkpoint covers the WHOLE
        // prompt: no token would remain to decode for first-token logits, so
        // the shorter checkpoint must win.
        assert_eq!(cache.lookup_longest(&hashes, 8), Some((hashes[0], 4)));

        // Unrelated prompt: no hit.
        let other = cache.prefix_hashes(&toks(&[9, 9, 9, 9, 9, 9, 9, 9, 9]));
        assert_eq!(cache.lookup_longest(&other, 9), None);
    }

    #[test]
    fn ssm_insert_evicts_lru_and_caps_entry_size() {
        let mut cache = SsmStateCache::new(4, 100, 60);

        // Oversized entry (> max_entry_bytes) is refused outright.
        cache.insert(1, 4, vec![0u8; 61]);
        assert!(!cache.contains(1));

        cache.insert(2, 4, vec![0u8; 50]);
        cache.insert(3, 8, vec![0u8; 40]);
        assert_eq!(cache.total_bytes, 90);

        // Touch entry 2 so entry 3 becomes the LRU victim.
        let hashes = [2u64];
        assert!(cache.lookup_longest(&hashes, 5).is_some());

        cache.insert(4, 4, vec![0u8; 30]);
        assert!(cache.contains(2), "recently used entry evicted");
        assert!(
            !cache.contains(3),
            "LRU entry survived over-capacity insert"
        );
        assert!(cache.contains(4));
        assert_eq!(cache.total_bytes, 80);
    }

    #[test]
    fn ssm_insert_refreshes_existing_entry_without_double_count() {
        let mut cache = SsmStateCache::new(4, 100, 100);
        cache.insert(7, 4, vec![0u8; 40]);
        cache.insert(7, 4, vec![0u8; 40]);
        assert_eq!(cache.total_bytes, 40);
        cache.remove(7);
        assert_eq!(cache.total_bytes, 0);
        assert!(!cache.contains(7));
    }

    #[test]
    fn ssm_session_matches_longest_strict_token_prefix() {
        let mut cache = SsmStateCache::new(4, 1 << 20, 1 << 20);
        // A session holds up to two snapshots at arbitrary positions: the
        // post-prompt state and the post-retirement state.
        cache.insert_session("s1", toks(&[1, 2, 3, 4, 5]), vec![0u8; 10]);
        cache.insert_session("s1", toks(&[1, 2, 3, 4, 5, 6, 7]), vec![0u8; 10]);

        // Next turn resends the full history: the longer snapshot wins.
        let hit = cache.session_match("s1", &toks(&[1, 2, 3, 4, 5, 6, 7, 8, 9]));
        assert_eq!(hit.map(|(_, n)| n), Some(7));

        // Reply-seam retokenization drift kills only the longer snapshot; the
        // post-prompt one still matches.
        let hit = cache.session_match("s1", &toks(&[1, 2, 3, 4, 5, 99, 7, 8, 9]));
        assert_eq!(hit.map(|(_, n)| n), Some(5));

        // Identical prompt (no token left to decode): miss.
        assert_eq!(
            cache
                .session_match("s1", &toks(&[1, 2, 3, 4, 5, 6, 7]))
                .map(|(_, n)| n),
            Some(5)
        );
        assert_eq!(cache.session_match("s1", &toks(&[1, 2, 3, 4, 5])), None);
        // Unknown session: miss.
        assert_eq!(
            cache.session_match("s2", &toks(&[1, 2, 3, 4, 5, 6, 7, 8])),
            None
        );
    }

    #[test]
    fn ssm_session_slots_cap_and_share_byte_budget_with_checkpoints() {
        let mut cache = SsmStateCache::new(4, 100, 100);
        cache.insert_session("s1", toks(&[1, 2]), vec![0u8; 20]);
        cache.insert_session("s1", toks(&[1, 2, 3]), vec![0u8; 20]);
        assert_eq!(cache.total_bytes, 40);
        // Third snapshot exceeds the per-session slot cap: the oldest goes.
        cache.insert_session("s1", toks(&[1, 2, 3, 4]), vec![0u8; 20]);
        assert_eq!(cache.total_bytes, 40);
        assert!(!cache.session_is_current("s1", &toks(&[1, 2])));
        assert!(cache.session_is_current("s1", &toks(&[1, 2, 3])));
        assert!(cache.session_is_current("s1", &toks(&[1, 2, 3, 4])));
        // Re-inserting an identical snapshot only refreshes it.
        cache.insert_session("s1", toks(&[1, 2, 3, 4]), vec![0u8; 20]);
        assert_eq!(cache.total_bytes, 40);

        // A chunk checkpoint overflowing the shared budget evicts session
        // snapshots (LRU across both kinds) — only as many as needed: the
        // refreshed [1,2,3,4] snapshot survives at exactly full budget.
        cache.insert(11, 4, vec![0u8; 80]);
        assert!(cache.contains(11));
        assert!(!cache.session_is_current("s1", &toks(&[1, 2, 3])));
        assert!(cache.session_is_current("s1", &toks(&[1, 2, 3, 4])));
        assert_eq!(cache.total_bytes, 100);

        // And a session insert can evict LRU chunk checkpoints in turn.
        cache.insert_session("s2", toks(&[9, 9]), vec![0u8; 70]);
        assert!(!cache.contains(11));
        assert!(cache.session_is_current("s2", &toks(&[9, 9])));
        assert_eq!(cache.total_bytes, 70);
    }
}
