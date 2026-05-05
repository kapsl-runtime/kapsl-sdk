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
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

#[cfg(feature = "gguf-cuda-shared-kv")]
use cudarc::driver::{CudaDevice, CudaSlice, DevicePtr};
#[cfg(feature = "gguf-cuda-shared-kv")]
use kapsl_hal::cpu_block_store::CpuBlockStore;
#[cfg(feature = "gguf-cuda-shared-kv")]
use kapsl_hal::cross_device_scheduler::CrossDevicePoolScheduler;
#[cfg(feature = "gguf-cuda-shared-kv")]
use kapsl_hal::gpu_arena::{GpuBlockPool, GpuPoolHandle};
#[cfg(feature = "gguf-cuda-shared-kv")]
use kapsl_hal::prefix_cache::PrefixBlockCache;
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
#[cfg(feature = "gguf-cuda-shared-kv")]
use llama_cpp_sys_2::{
    llama_kapsl_kv_pool_desc, LLAMA_KAPSL_KV_DTYPE_F16,
};

// ─── Configuration ────────────────────────────────────────────────────────────

const MAX_CONCURRENT_DEFAULT: usize = 32;
const N_CTX_PER_SEQ_DEFAULT: u32 = 2048;
const GGUF_TARGET_CONCURRENCY_ENV: &str = "KAPSL_GGUF_TARGET_CONCURRENCY";
const GGUF_QUEUE_DELAY_US_DEFAULT: u64 = 1_000;
const GGUF_PREFILL_CHUNK_SIZE_DEFAULT: usize = 512;
// llama.cpp sequence-copy asserts unless the context uses a full/unified KV buffer.
// Keep this opt-in until we can detect that mode safely at runtime.
const GGUF_EXACT_PROMPT_KV_REUSE_DEFAULT: bool = false;

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
#[derive(Clone, Copy, Debug)]
struct GgufServingConfig {
    max_concurrent: usize,
    ctx_per_seq: u32,
    queue_delay: Duration,
    prefill_chunk_size: usize,
    exact_prompt_kv_reuse: bool,
    kv_bytes_per_cell: usize,
}

#[cfg(feature = "gguf")]
impl GgufServingConfig {
    fn from_model(model: &LlamaModel, n_ctx_train: u32) -> Self {
        let max_concurrent = max_concurrent();
        let ctx_per_seq = n_ctx_per_seq().min(n_ctx_train);
        let n_batch = ctx_per_seq + max_concurrent as u32;
        Self {
            max_concurrent,
            ctx_per_seq,
            queue_delay: gguf_queue_delay(),
            prefill_chunk_size: gguf_prefill_chunk_size().min(n_batch as usize).max(1),
            exact_prompt_kv_reuse: gguf_exact_prompt_kv_reuse(),
            kv_bytes_per_cell: estimate_kv_bytes_per_cell(model),
        }
    }

    fn total_ctx(self) -> usize {
        self.max_concurrent * self.ctx_per_seq as usize
    }

    fn n_batch(self) -> u32 {
        self.ctx_per_seq + self.max_concurrent as u32
    }
}

#[cfg(feature = "gguf")]
fn estimate_kv_bytes_per_cell(model: &LlamaModel) -> usize {
    let n_embd = model.n_embd().max(0) as usize;
    let n_head = model.n_head().max(1) as usize;
    let n_head_kv = model.n_head_kv().max(1) as usize;
    let n_embd_gqa = n_embd.saturating_mul(n_head_kv) / n_head;

    // llama.cpp reports this path as K/V f16 in the load log. Treat one KV cell
    // as K + V for every layer so Prometheus/admission logic has a comparable
    // capacity signal to the native/ONNX paths.
    model.n_layer().max(1) as usize * n_embd_gqa * 2 * std::mem::size_of::<u16>()
}

#[cfg(feature = "gguf-cuda-shared-kv")]
fn gguf_shared_kv_block_count(
    n_layers: usize,
    config: GgufServingConfig,
    block_size: usize,
) -> usize {
    if let Some(blocks) = std::env::var("KAPSL_GGUF_CUDA_SHARED_KV_POOL_BLOCKS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&v| v > 0)
    {
        return blocks.max(n_layers);
    }
    let blocks_per_seq = (config.ctx_per_seq as usize).div_ceil(block_size.max(1)).max(1);
    n_layers
        .saturating_mul(config.max_concurrent.max(1))
        .saturating_mul(blocks_per_seq)
        .max(n_layers)
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

// ─── Cross-device KV scheduler singleton ──────────────────────────────────────

#[cfg(feature = "gguf-cuda-shared-kv")]
static GGUF_KV_SCHEDULER: std::sync::OnceLock<
    std::sync::Mutex<CrossDevicePoolScheduler>
> = std::sync::OnceLock::new();

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
        .map(|v| !matches!(v.trim().to_ascii_lowercase().as_str(), "0" | "false" | "no" | "off"))
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

/// KV block data downloaded to CPU when the scheduler went idle.
///
/// Stored in `GgufSharedKvPoolState::evicted` until the next `reserve` call
/// restores the data back to freshly allocated GPU blocks.
#[cfg(feature = "gguf-cuda-shared-kv")]
struct CpuEvictedState {
    store:     CpuBlockStore,
    /// Slot indices: `slots[layer * n_logical + pos]`
    /// where `pos` is 0-indexed within the evicted (non-promoted) positions.
    slots:        Vec<u32>,
    /// Number of owned logical positions that were saved.
    n_logical:    usize,
    n_layers:     usize,
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
    device_id: usize,
    n_layers: usize,
    max_blocks_per_seq: usize,
    /// Optional prefix block cache. When present, `reserve_prefix` and
    /// `promote_prefix` callbacks are active.
    prefix_cache: Option<Arc<Mutex<PrefixBlockCache>>>,
    /// Stable model identity hash (set at pool construction time).
    model_fingerprint: u64,
    inner: Mutex<GgufSharedKvReservation>,
    /// CPU-side KV block backup, populated by `evict_to_cpu()`.
    /// Cleared by the next `reserve` call which uploads data back to GPU.
    evicted: Mutex<Option<CpuEvictedState>>,
    /// When true, `run_scheduler` calls `evict_to_cpu()` before blocking on
    /// the idle receive.  Controlled by `KAPSL_GGUF_EVICT_ON_IDLE`.
    evict_when_idle: bool,
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
    n_logical_blocks: usize,
}

#[cfg(feature = "gguf-cuda-shared-kv")]
unsafe impl Send for GgufSharedKvPool {}
#[cfg(feature = "gguf-cuda-shared-kv")]
unsafe impl Sync for GgufSharedKvPool {}

#[cfg(feature = "gguf-cuda-shared-kv")]
impl GgufSharedKvPool {
    fn new(
        handle:            GpuPoolHandle,
        device_id:         usize,
        n_layers:          usize,
        max_ctx:           usize,
        prefix_cache:      Option<Arc<Mutex<PrefixBlockCache>>>,
        model_fingerprint: u64,
    ) -> Self {
        let pool_arc           = handle.pool.clone();
        let block_size         = handle.pool.block_size().max(1);
        let max_blocks_per_seq = max_ctx.div_ceil(block_size).max(1);
        let has_prefix = prefix_cache.is_some();
        let evict_when_idle = std::env::var("KAPSL_GGUF_EVICT_ON_IDLE")
            .map(|v| !matches!(v.trim().to_ascii_lowercase().as_str(), "0" | "false" | "no" | "off"))
            .unwrap_or(false);

        let mut state = Box::new(GgufSharedKvPoolState {
            handle,
            device_id,
            n_layers,
            max_blocks_per_seq,
            prefix_cache,
            model_fingerprint,
            inner:           Mutex::new(GgufSharedKvReservation::default()),
            evicted:         Mutex::new(None),
            evict_when_idle,
        });
        let state_ptr = (&mut *state) as *mut GgufSharedKvPoolState;
        let pool = &state.handle.pool;
        let desc = Box::new(llama_kapsl_kv_pool_desc {
            user_data:                state_ptr.cast(),
            device_id:                device_id as u32,
            block_size:               pool.block_size() as u32,
            num_blocks:               pool.total_blocks() as u32,
            num_kv_heads:             pool.num_kv_heads() as u32,
            head_dim:                 pool.head_dim() as u32,
            dtype:                    LLAMA_KAPSL_KV_DTYPE_F16,
            device_base:              pool.device_base_ptr(),
            block_table_device:       std::ptr::null_mut(),
            block_table_layer_stride: max_blocks_per_seq as u32,
            n_layers:                 n_layers as u32,
            max_blocks_per_seq:       max_blocks_per_seq as u32,
            model_fingerprint,
            reserve:                  Some(gguf_kapsl_kv_reserve),
            release:                  Some(gguf_kapsl_kv_release),
            touch:                    Some(gguf_kapsl_kv_touch),
            reserve_prefix:  if has_prefix { Some(gguf_kapsl_kv_reserve_prefix) } else { None },
            promote_prefix:           None, // handled via promote_if_pending() from Rust
            needs_restore:            Some(gguf_kapsl_kv_needs_restore),
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
        let Some(cache) = &state.prefix_cache else { return };
        let mut inner = state.inner.lock().unwrap();
        let n_new = inner.n_new_logical;
        if n_new == 0 || inner.n_promoted_logical >= n_new { return; }

        let start     = inner.n_promoted_logical;
        let fp        = inner.model_fingerprint;
        let pool      = state.handle.pool.clone();
        let n_layers  = state.n_layers;
        let device_id = self.desc.device_id as usize;
        let mut c = match cache.lock() { Ok(c) => c, Err(_) => return };

        let mut promoted = start;
        for pos in start..n_new {
            let hash_idx = inner.n_prefix_hits + pos;
            let Some(&hash) = inner.all_hashes.get(hash_idx) else { break };

            // Build the block_ids Vec for this logical position (one block per layer).
            let block_ids: Vec<u32> = (0..n_layers)
                .map(|layer| inner.owned_blocks[layer * n_new + pos])
                .collect();

            if !c.insert(fp, hash, device_id, pool.clone(), block_ids, 1) {
                break; // cache full, all entries referenced — stop
            }
            promoted += 1;
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
        if state.evicted.lock().unwrap().is_some() { return false; }

        let mut inner      = state.inner.lock().unwrap();
        let n_new          = inner.n_new_logical;
        let n_promoted     = inner.n_promoted_logical;
        let n_owned        = n_new.saturating_sub(n_promoted);
        let n_logical_orig = inner.n_logical_blocks;

        if n_owned == 0 { return false; }

        let n_layers   = state.n_layers;
        let pool       = &state.handle.pool;

        // Download each un-promoted owned block to CPU.
        let n_slots    = n_owned * n_layers;
        let mut store  = CpuBlockStore::new(
            n_slots, pool.num_kv_heads(), pool.block_size(), pool.head_dim());
        let mut slots: Vec<u32> = Vec::with_capacity(n_slots);

        for layer in 0..n_layers {
            for pos in n_promoted..n_new {
                let block_id = match inner.owned_blocks.get(layer * n_new + pos) {
                    Some(&id) => id,
                    None      => return false,
                };
                match pool.download_block(block_id) {
                    Ok(data) => match store.store_block(&data) {
                        Ok(slot) => slots.push(slot),
                        Err(_)   => return false,
                    },
                    Err(_) => return false,
                }
            }
        }

        // Release prefix borrows + free all un-promoted GPU blocks.
        gguf_release_reservation_inner(
            &mut inner, &state.prefix_cache, pool, n_layers);
        drop(inner);

        *state.evicted.lock().unwrap() = Some(CpuEvictedState {
            store,
            slots,
            n_logical: n_owned,
            n_layers,
            n_logical_at_eviction: n_logical_orig,
        });

        log::info!(
            "[gguf] evicted {} owned positions ({} GPU blocks freed) to CPU",
            n_owned, n_owned * n_layers
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
        let mut inner = self.inner.lock().unwrap();
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
        let n_new    = inner.n_new_logical;
        let n_layers = self.n_layers;
        for layer in 0..n_layers {
            for pos in inner.n_promoted_logical..n_new {
                let block = inner.owned_blocks[layer * n_new + pos];
                self.handle.pool.free_block(block);
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
    if user_data.is_null() { return 0; }
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
    state:    &GgufSharedKvPoolState,
    n_needed: usize,
    out:      &mut Vec<u32>,
) -> bool {
    // First pass: allocate as many blocks as are immediately available.
    while out.len() < n_needed {
        match state.handle.pool.alloc_block() {
            Ok(b)  => out.push(b),
            Err(_) => break,
        }
    }
    if out.len() == n_needed { return true; }

    // Evict LRU zero-refcount prefix-cache entries to reclaim GPU blocks.
    let still_needed = n_needed - out.len();
    if let Some(cache) = &state.prefix_cache {
        if let Ok(mut c) = cache.lock() {
            let freed = c.evict_lru_for_device(state.device_id, still_needed);
            for (p, ids) in freed { for id in ids { p.free_block(id); } }
        }
    }

    // Second pass after eviction.
    while out.len() < n_needed {
        match state.handle.pool.alloc_block() {
            Ok(b)  => out.push(b),
            Err(_) => {
                for b in out.drain(..) { state.handle.pool.free_block(b); }
                return false;
            }
        }
    }
    true
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
    state:                   &GgufSharedKvPoolState,
    evicted:                 CpuEvictedState,
    logical_blocks:          usize,
    n_prefix_hits:           usize,
    prefix_hit_block_ids:    &[Vec<u32>], // per-hit, per-layer block ids; len == n_prefix_hits
    block_table_device_out:  *mut *mut u32,
    n_logical_out:           *mut u32,
    n_prefix_hits_out:       *mut u32,    // may be null for non-prefix path
) -> bool {
    let n_layers    = state.n_layers;
    let pool        = &state.handle.pool;
    let n_new       = logical_blocks.saturating_sub(n_prefix_hits);
    let n_restore   = n_new.min(evicted.n_logical);
    let n_fresh     = n_new.saturating_sub(n_restore);
    let total_phys  = n_new * n_layers;

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
                if pool.upload_block(block_id, &data[..half], &data[half..]).is_err() {
                    for b in new_blocks { pool.free_block(b); }
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
            let blk = block_ids.get(layer).copied()
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
            for b in new_blocks { pool.free_block(b); }
            *state.evicted.lock().unwrap() = Some(evicted);
            return false;
        }
    };
    let table_ptr = *table.device_ptr() as *mut u32;

    let mut inner = state.inner.lock().unwrap();
    gguf_release_reservation_inner(&mut inner, &state.prefix_cache, pool, n_layers);
    inner.owned_blocks       = new_blocks;
    inner.n_new_logical      = n_new;
    inner.n_promoted_logical = 0;
    inner.all_hashes         = Vec::new();
    inner.n_prefix_hits      = n_prefix_hits;
    inner.model_fingerprint  = state.model_fingerprint;
    inner.block_table        = Some(table);
    inner.n_logical_blocks   = logical_blocks;

    unsafe {
        *block_table_device_out = table_ptr;
        *n_logical_out          = logical_blocks as u32;
        if !n_prefix_hits_out.is_null() {
            *n_prefix_hits_out  = n_prefix_hits as u32;
        }
    }
    log::info!(
        "[gguf] restored {} positions from CPU ({} fresh) to GPU after eviction",
        n_restore, n_fresh
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
    let block_size     = state.handle.pool.block_size().max(1);
    let logical_blocks = (tokens_needed as usize).div_ceil(block_size).max(1);
    if logical_blocks > state.max_blocks_per_seq { return false; }

    // ── Restore path: re-upload CPU-evicted blocks ────────────────────────
    let evicted_opt = state.evicted.lock().unwrap().take();
    if let Some(evicted) = evicted_opt {
        return gguf_restore_from_cpu(
            state, evicted, logical_blocks,
            0, &[],
            block_table_device_out, n_blocks_out, std::ptr::null_mut(),
        );
    }

    // ── Fresh allocation ──────────────────────────────────────────────────
    let needed_physical = logical_blocks.saturating_mul(state.n_layers);
    if needed_physical > state.handle.cap() { return false; }

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
        Err(_) => { for b in new_blocks { state.handle.pool.free_block(b); } return false; }
    };
    let table_ptr = *table.device_ptr() as *mut u32;

    let mut inner = state.inner.lock().unwrap();
    gguf_release_reservation_inner(&mut inner, &state.prefix_cache, &state.handle.pool, state.n_layers);
    inner.owned_blocks       = new_blocks;
    inner.n_new_logical      = logical_blocks;
    inner.n_promoted_logical = 0;
    inner.all_hashes         = Vec::new();
    inner.n_prefix_hits      = 0;
    inner.model_fingerprint  = state.model_fingerprint;
    inner.block_table        = Some(table);
    inner.n_logical_blocks   = logical_blocks;

    *block_table_device_out = table_ptr;
    *n_blocks_out           = logical_blocks as u32;
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
    if user_data.is_null()
        || tokens.is_null()
        || block_table_device_out.is_null()
        || n_logical_blocks_out.is_null()
        || n_prefix_hits_out.is_null()
    {
        return false;
    }
    let state = &*(user_data as *mut GgufSharedKvPoolState);
    let Some(cache) = &state.prefix_cache else {
        // Prefix cache disabled — fall back to plain reserve.
        return gguf_kapsl_kv_reserve(
            user_data, _session_id, n_tokens,
            block_table_device_out, n_logical_blocks_out,
        );
    };

    let token_slice = std::slice::from_raw_parts(tokens, n_tokens as usize);
    let block_size  = state.handle.pool.block_size().max(1);
    let fp          = state.model_fingerprint;

    // 1. Compute chained hashes for all complete prefix blocks.
    let all_hashes = PrefixBlockCache::compute_prefix_hashes_i32(fp, token_slice, block_size);
    let n_logical  = (n_tokens as usize).div_ceil(block_size).max(1);
    if n_logical > state.max_blocks_per_seq { return false; }

    // 2. Lookup prefix cache — collect a contiguous run of hits.
    let hits = {
        let mut c = match cache.lock() { Ok(c) => c, Err(_) => return false };
        c.lookup(fp, &all_hashes)
    };
    let n_hits = hits.len();

    // 3. Check for CPU-evicted state: use restore path if available.
    let evicted_opt = state.evicted.lock().unwrap().take();
    if let Some(evicted) = evicted_opt {
        // Build the per-hit block-id lists for the restore helper.
        let hit_block_ids: Vec<Vec<u32>> = hits.iter()
            .map(|h| h.block_ids.clone())
            .collect();
        return gguf_restore_from_cpu(
            state, evicted, n_logical,
            n_hits, &hit_block_ids,
            block_table_device_out, n_logical_blocks_out, n_prefix_hits_out,
        );
    }

    // 4. Allocate fresh blocks for positions that missed the cache.
    let n_new       = n_logical.saturating_sub(n_hits);
    let needed_phys = n_new.saturating_mul(state.n_layers);
    if needed_phys > state.handle.cap() { return false; }

    let mut new_blocks: Vec<u32> = Vec::with_capacity(needed_phys);
    if !alloc_blocks_with_cache_eviction(state, needed_phys, &mut new_blocks) {
        // Release prefix borrows before failing.
        if let Ok(mut c) = cache.lock() {
            for h in &hits { c.release(fp, h.block_hash); }
        }
        return false;
    }

    // 5. Build the block table: cached blocks first, then new blocks.
    let mut host_table = vec![0u32; state.n_layers * state.max_blocks_per_seq];
    for layer in 0..state.n_layers {
        for (hit_pos, hit) in hits.iter().enumerate() {
            let blk = hit.block_ids.get(layer).copied()
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
            for b in new_blocks { state.handle.pool.free_block(b); }
            if let Ok(mut c) = cache.lock() {
                for h in &hits { c.release(fp, h.block_hash); }
            }
            return false;
        }
    };
    let table_ptr = *table.device_ptr() as *mut u32;

    let mut inner = state.inner.lock().unwrap();
    gguf_release_reservation_inner(&mut inner, &state.prefix_cache, &state.handle.pool, state.n_layers);
    inner.owned_blocks       = new_blocks;
    inner.n_new_logical      = n_new;
    inner.n_promoted_logical = 0;
    inner.all_hashes         = all_hashes;
    inner.n_prefix_hits      = n_hits;
    inner.model_fingerprint  = fp;
    inner.block_table        = Some(table);
    inner.n_logical_blocks   = n_logical;

    *block_table_device_out = table_ptr;
    *n_logical_blocks_out   = n_logical as u32;
    *n_prefix_hits_out      = n_hits as u32;
    true
}

// ── Release callback ──────────────────────────────────────────────────────────

/// Inline release logic, callable with explicit pool access.
#[cfg(feature = "gguf-cuda-shared-kv")]
fn gguf_release_reservation_inner(
    inner:        &mut GgufSharedKvReservation,
    prefix_cache: &Option<Arc<Mutex<PrefixBlockCache>>>,
    pool:         &Arc<GpuBlockPool>,
    n_layers:     usize,
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
unsafe extern "C" fn gguf_kapsl_kv_release(user_data: *mut std::ffi::c_void, _session_id: u64) {
    if user_data.is_null() { return; }
    let state = &*(user_data as *mut GgufSharedKvPoolState);
    let mut inner = state.inner.lock().unwrap();
    gguf_release_reservation_inner(&mut inner, &state.prefix_cache, &state.handle.pool, state.n_layers);
}

#[cfg(feature = "gguf-cuda-shared-kv")]
unsafe extern "C" fn gguf_kapsl_kv_touch(
    user_data: *mut std::ffi::c_void,
    _session_id: u64,
) -> bool {
    !user_data.is_null()
}

// ─── Backend ──────────────────────────────────────────────────────────────────

pub struct GgufBackend {
    #[cfg(feature = "gguf")]
    inner: Option<GgufInner>,
    metrics: Arc<Mutex<EngineMetrics>>,
    #[cfg(feature = "gguf-cuda-shared-kv")]
    device_id: usize,
    #[cfg(feature = "gguf-cuda-shared-kv")]
    pool_slot: Arc<Mutex<Option<GpuPoolHandle>>>,
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
            metrics: Arc::new(Mutex::new(EngineMetrics::new())),
            #[cfg(feature = "gguf-cuda-shared-kv")]
            device_id: 0,
            #[cfg(feature = "gguf-cuda-shared-kv")]
            pool_slot: Arc::new(Mutex::new(None)),
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
        }
    }

    #[cfg(feature = "gguf-cuda-shared-kv")]
    pub fn with_pool_handle(mut self, handle: GpuPoolHandle) -> Self {
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
    let pending_cells = pending
        .iter()
        .map(|pref| pref.next_token.min(pref.tokens.len()))
        .sum::<usize>();
    let active_cells = active
        .iter()
        .map(|seq| seq.pos.max(0) as usize)
        .sum::<usize>();
    let used_cells = pending_cells.saturating_add(active_cells).min(total_cells);
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
    #[cfg(feature = "gguf-cuda-shared-kv")] mut shared_kv_pool: Option<GgufSharedKvPool>,
) {
    let total_ctx = config.total_ctx();
    let n_ctx = match NonZeroU32::new(total_ctx as u32) {
        Some(v) => v,
        None => {
            log::error!("[gguf] invalid total_ctx=0");
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
    let batch_cap = n_batch as usize;
    let mut batch = LlamaBatch::new(batch_cap, 1);

    // seq_id pool: 0..max_concurrent are valid sequence identifiers for the KV cache.
    let mut available_ids: Vec<i32> = (0..config.max_concurrent as i32).rev().collect();
    let mut waiting: VecDeque<GgufRequest> = VecDeque::new();
    let mut pending: VecDeque<PendingPrefill> = VecDeque::new();
    let mut active: Vec<ActiveSeq> = Vec::with_capacity(config.max_concurrent);
    update_gguf_metrics(&metrics, config, &waiting, &pending, &active, 0);

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
        update_gguf_metrics(&metrics, config, &waiting, &pending, &active, 0);

        // ── 3. If completely idle, block for the next request ─────────────────
        if waiting.is_empty() && pending.is_empty() && active.is_empty() {
            #[cfg(feature = "gguf-cuda-shared-kv")]
            if let Some(pool) = shared_kv_pool.as_mut() {
                if pool.state.evict_when_idle {
                    pool.evict_to_cpu();
                }
                // Log cross-device pressure snapshot at debug level.
                let kv_heads  = pool.state.handle.pool.num_kv_heads();
                let head_dim  = pool.state.handle.pool.head_dim();
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
                    sched.registered_devices().iter()
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
            update_gguf_metrics(&metrics, config, &waiting, &pending, &active, 0);
            continue;
        }

        for pref in partial_prefills.drain(..) {
            pending.push_back(pref);
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
        update_gguf_metrics(&metrics, config, &waiting, &pending, &active, 0);
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

        let config = GgufServingConfig::from_model(&weights.model, weights.n_ctx_train);
        #[cfg(feature = "gguf-cuda-shared-kv")]
        let shared_kv_pool = {
            let n_layers = weights.model.n_layer().max(1) as usize;
            let n_embd = weights.model.n_embd().max(1) as usize;
            let n_head = weights.model.n_head().max(1) as usize;
            let n_head_kv = weights.model.n_head_kv().max(1) as usize;
            let head_dim = (n_embd / n_head).max(1);
            let block_size = 16usize;
            // Auto-select the least-loaded registered device when KAPSL_GGUF_AUTO_DEVICE=1.
            let effective_device_id = gguf_select_device(self.device_id, n_head_kv, head_dim);
            if effective_device_id != self.device_id {
                log::info!(
                    "[gguf] auto-device: selected device {} over {} (more free {n_head_kv}h×{head_dim}d blocks)",
                    effective_device_id, self.device_id,
                );
            }
            let handle = {
                let mut slot = self.pool_slot.lock().unwrap();
                if let Some(handle) = slot.as_ref() {
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
                        let num_blocks = gguf_shared_kv_block_count(n_layers, config, block_size);
                        let pool = Arc::new(
                            GpuBlockPool::new(device, num_blocks, block_size, n_head_kv, head_dim)
                                .map_err(|e| EngineError::backend(format!("shared KV pool: {e}")))?,
                        );
                        let handle = GpuPoolHandle::private(pool);
                        *slot = Some(handle.clone());
                        handle
                    }
                } else {
                    let device = CudaDevice::new(effective_device_id)
                        .map_err(|e| EngineError::backend(format!("CUDA: {e}")))?;
                    let num_blocks = gguf_shared_kv_block_count(n_layers, config, block_size);
                    let pool = Arc::new(
                        GpuBlockPool::new(device, num_blocks, block_size, n_head_kv, head_dim)
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
                    use std::hash::{Hash, Hasher};
                    use std::collections::hash_map::DefaultHasher;
                    let mut h = DefaultHasher::new();
                    n_layers.hash(&mut h);
                    n_head_kv.hash(&mut h);
                    head_dim.hash(&mut h);
                    n_embd.hash(&mut h);
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
                let prefix_cache = Some(Arc::new(Mutex::new(
                    PrefixBlockCache::new(prefix_cache_cap),
                )));
                log::info!(
                    "[gguf] Prefix KV cache enabled: capacity={} logical positions",
                    prefix_cache_cap
                );
                Some(GgufSharedKvPool::new(
                    handle,
                    effective_device_id,
                    n_layers,
                    config.total_ctx(),
                    prefix_cache,
                    model_fingerprint,
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
        std::thread::spawn(move || {
            run_scheduler(
                model_clone,
                backend_clone,
                rx,
                config,
                metrics,
                #[cfg(feature = "gguf-cuda-shared-kv")]
                shared_kv_pool,
            );
        });

        log::info!(
            "[gguf] Scheduler started: max_concurrent={}, ctx_per_seq={}",
            config.max_concurrent,
            config.ctx_per_seq
        );

        self.inner = Some(GgufInner {
            weights,
            request_tx: tx,
            max_concurrent: config.max_concurrent,
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
        if let Ok(mut metrics) = self.metrics.lock() {
            *metrics = EngineMetrics::new();
        }
        log::info!("[gguf] Backend unloaded");
    }

    fn metrics(&self) -> EngineMetrics {
        self.metrics_snapshot()
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
        self.metrics_snapshot()
    }

    fn health_check(&self) -> Result<(), EngineError> {
        Err(EngineError::backend(
            "GGUF support not compiled in (enable the 'gguf' feature)",
        ))
    }
}
