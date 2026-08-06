use async_trait::async_trait;
use base64::Engine as _;
use futures::stream::Stream;
use serde::{Deserialize, Serialize};
use std::borrow::Cow;
use std::fmt;
use std::pin::Pin;
use std::str::FromStr;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum EngineError {
    #[error("Backend error: {message}")]
    Backend {
        message: String,
        #[source]
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },
    #[error("Invalid input: {message}")]
    InvalidInput {
        message: String,
        #[source]
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },
    #[error("Model not loaded")]
    ModelNotLoaded,
    #[error("System overloaded: {message}")]
    Overloaded {
        message: String,
        #[source]
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },
    #[error("Model load error for {path}: {source}")]
    ModelLoadError {
        path: String,
        source: Box<dyn std::error::Error + Send + Sync>,
    },
    #[error("Inference error: {reason}")]
    InferenceError {
        reason: String,
        #[source]
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },
    #[error("Timeout: {message}")]
    TimeoutError {
        message: String,
        #[source]
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },
    #[error("Resource exhausted: {message}")]
    ResourceExhausted {
        message: String,
        #[source]
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },
    #[error("Cancelled: {message}")]
    Cancelled { message: String },
}

impl EngineError {
    pub fn backend(message: impl Into<String>) -> Self {
        EngineError::Backend {
            message: message.into(),
            source: None,
        }
    }

    pub fn invalid_input(message: impl Into<String>) -> Self {
        EngineError::InvalidInput {
            message: message.into(),
            source: None,
        }
    }

    pub fn invalid_input_with_source(
        message: impl Into<String>,
        source: impl std::error::Error + Send + Sync + 'static,
    ) -> Self {
        EngineError::InvalidInput {
            message: message.into(),
            source: Some(Box::new(source)),
        }
    }

    pub fn overloaded(message: impl Into<String>) -> Self {
        EngineError::Overloaded {
            message: message.into(),
            source: None,
        }
    }

    pub fn is_overloaded(&self) -> bool {
        matches!(self, EngineError::Overloaded { .. })
    }

    pub fn timeout(message: impl Into<String>) -> Self {
        EngineError::TimeoutError {
            message: message.into(),
            source: None,
        }
    }

    pub fn resource_exhausted(message: impl Into<String>) -> Self {
        EngineError::ResourceExhausted {
            message: message.into(),
            source: None,
        }
    }

    pub fn cancelled(message: impl Into<String>) -> Self {
        EngineError::Cancelled {
            message: message.into(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineMetrics {
    pub inference_time: f64,
    pub memory_usage: usize,
    pub gpu_utilization: f64,
    pub throughput: f64,
    pub batch_size: usize,
    pub queue_depth: usize,
    pub error_rate: f64,
    pub collected_at_ms: u64,
    pub kv_cache_bytes_used: usize,
    pub kv_cache_bytes_capacity: usize,
    pub kv_cache_blocks_total: usize,
    pub kv_cache_blocks_free: usize,
    pub kv_cache_sequences: usize,
    pub kv_cache_evicted_blocks: u64,
    pub kv_cache_evicted_sequences: u64,
    pub kv_cache_packed_layers: usize,
    pub kv_cache_cpu_offloaded_blocks: u64,
    pub prompt_tokens_total: u64,
    pub generated_tokens_total: u64,
    pub decode_steps_total: u64,
    pub decode_tokens_evaluated_total: u64,
    pub kv_partial_reuse_hits_total: u64,
    pub kv_partial_reuse_tokens_saved_total: u64,
    /// Operational health of the engine: 0 = healthy, 1 = degraded, 2 = dead.
    /// Reported from the cross-model scheduler's per-engine health state.
    pub engine_health: u8,
    pub onnx_session_pool_total: usize,
    pub onnx_session_pool_idle: usize,
    pub onnx_session_pool_waits_total: u64,
    pub onnx_session_pool_wait_seconds_total: f64,
}

impl EngineMetrics {
    pub fn new() -> Self {
        Self {
            inference_time: 0.0,
            memory_usage: 0,
            gpu_utilization: 0.0,
            throughput: 0.0,
            batch_size: 0,
            queue_depth: 0,
            error_rate: 0.0,
            collected_at_ms: Self::now_ms(),
            kv_cache_bytes_used: 0,
            kv_cache_bytes_capacity: 0,
            kv_cache_blocks_total: 0,
            kv_cache_blocks_free: 0,
            kv_cache_sequences: 0,
            kv_cache_evicted_blocks: 0,
            kv_cache_evicted_sequences: 0,
            kv_cache_packed_layers: 0,
            kv_cache_cpu_offloaded_blocks: 0,
            prompt_tokens_total: 0,
            generated_tokens_total: 0,
            decode_steps_total: 0,
            decode_tokens_evaluated_total: 0,
            kv_partial_reuse_hits_total: 0,
            kv_partial_reuse_tokens_saved_total: 0,
            engine_health: 0,
            onnx_session_pool_total: 0,
            onnx_session_pool_idle: 0,
            onnx_session_pool_waits_total: 0,
            onnx_session_pool_wait_seconds_total: 0.0,
        }
    }

    pub fn refresh_timestamp(&mut self) {
        self.collected_at_ms = Self::now_ms();
    }

    fn now_ms() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64
    }
}

impl Default for EngineMetrics {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TensorDtype {
    Float32,
    Float64,
    Float16,
    Int32,
    Int64,
    Uint8,
    Utf8,
}

impl TensorDtype {
    pub fn as_str(&self) -> &'static str {
        match self {
            TensorDtype::Float32 => "float32",
            TensorDtype::Float64 => "float64",
            TensorDtype::Float16 => "float16",
            TensorDtype::Int32 => "int32",
            TensorDtype::Int64 => "int64",
            TensorDtype::Uint8 => "uint8",
            TensorDtype::Utf8 => "string",
        }
    }

    pub fn size_bytes(&self) -> usize {
        match self {
            TensorDtype::Float32 => 4,
            TensorDtype::Float64 => 8,
            TensorDtype::Float16 => 2,
            TensorDtype::Int32 => 4,
            TensorDtype::Int64 => 8,
            TensorDtype::Uint8 => 1,
            TensorDtype::Utf8 => 1,
        }
    }
}

impl fmt::Display for TensorDtype {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

impl FromStr for TensorDtype {
    type Err = EngineError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.to_lowercase().as_str() {
            "float32" | "fp32" => Ok(TensorDtype::Float32),
            "float64" | "fp64" => Ok(TensorDtype::Float64),
            "float16" | "fp16" => Ok(TensorDtype::Float16),
            "int32" | "i32" => Ok(TensorDtype::Int32),
            "int64" | "i64" => Ok(TensorDtype::Int64),
            "uint8" | "u8" => Ok(TensorDtype::Uint8),
            "string" | "utf8" => Ok(TensorDtype::Utf8),
            other => Err(EngineError::InvalidInput {
                message: format!("Unsupported dtype: {}", other),
                source: None,
            }),
        }
    }
}

impl Serialize for TensorDtype {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_str(self.as_str())
    }
}

impl<'de> Deserialize<'de> for TensorDtype {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        TensorDtype::from_str(&value).map_err(serde::de::Error::custom)
    }
}

#[derive(Debug, Clone)]
pub struct BinaryTensorPacket {
    pub shape: Vec<i64>,
    pub dtype: TensorDtype,
    pub data: Vec<u8>,
}

impl serde::Serialize for BinaryTensorPacket {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeStruct;
        let human_readable = serializer.is_human_readable();
        // Must match the 4-field layout of BinaryTensorPacketPayload in the Deserialize impl
        // so that bincode round-trips correctly (derived Serialize only emits 3 fields,
        // causing a field-count mismatch on the bincode decode side).
        let mut state = serializer.serialize_struct("BinaryTensorPacket", 4)?;
        state.serialize_field("shape", &self.shape)?;
        state.serialize_field("dtype", &self.dtype)?;
        if human_readable {
            let data_base64 = base64::engine::general_purpose::STANDARD.encode(&self.data);
            state.serialize_field("data", &None::<&[u8]>)?;
            state.serialize_field("data_base64", &Some(data_base64.as_str()))?;
        } else {
            state.serialize_field("data", &Some(&self.data))?;
            state.serialize_field("data_base64", &None::<&str>)?;
        }
        state.end()
    }
}

impl<'de> Deserialize<'de> for BinaryTensorPacket {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        // Borrow `data_base64` as `&'de str` rather than allocating a `String`:
        // since we deserialize from a buffered byte slice (from_slice), serde can hand
        // us a reference directly into the JSON input, saving one large memcpy before decode.
        // Use Cow<str> for data_base64: borrows directly from the JSON buffer when
        // deserializing via from_slice (zero-copy), allocates when deserializing
        // from an owned Value (test/fallback path).
        #[derive(Deserialize)]
        struct BinaryTensorPacketPayload<'src> {
            shape: Vec<i64>,
            dtype: TensorDtype,
            #[serde(default)]
            data: Option<Vec<u8>>,
            #[serde(default, alias = "base64", borrow)]
            data_base64: Option<Cow<'src, str>>,
        }

        let payload = BinaryTensorPacketPayload::deserialize(deserializer)?;
        let data = match (payload.data, payload.data_base64) {
            (Some(data), None) => data,
            (None, Some(encoded)) => base64::engine::general_purpose::STANDARD
                .decode(encoded.as_bytes())
                .map_err(serde::de::Error::custom)?,
            (Some(_), Some(_)) => {
                return Err(serde::de::Error::custom(
                    "binary tensor payload must include only one of `data` or `data_base64`",
                ))
            }
            (None, None) => {
                return Err(serde::de::Error::custom(
                    "binary tensor payload must include `data` or `data_base64`",
                ))
            }
        };

        Ok(Self {
            shape: payload.shape,
            dtype: payload.dtype,
            data,
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BinaryTensorPacketRef<'a> {
    pub shape: Vec<i64>,
    pub dtype: TensorDtype,
    #[serde(borrow)]
    pub data: Cow<'a, [u8]>,
}

#[derive(Debug, Clone, Copy)]
pub struct TensorView<'a> {
    pub shape: &'a [i64],
    pub dtype: TensorDtype,
    pub data: &'a [u8],
}

impl BinaryTensorPacket {
    pub fn new(shape: Vec<i64>, dtype: TensorDtype, data: Vec<u8>) -> Result<Self, EngineError> {
        let packet = Self { shape, dtype, data };
        packet.validate()?;
        Ok(packet)
    }

    pub fn size_bytes(&self) -> usize {
        self.data.len()
    }

    pub fn tensor_elements(&self) -> Result<usize, EngineError> {
        shape_elements(&self.shape)
    }

    pub fn validate(&self) -> Result<(), EngineError> {
        let elements = self.tensor_elements()?;
        let expected = elements
            .checked_mul(self.dtype.size_bytes())
            .ok_or_else(|| EngineError::InvalidInput {
                message: "Data size overflow".to_string(),
                source: None,
            })?;

        if self.data.len() != expected {
            return Err(EngineError::InvalidInput {
                message: format!(
                    "Data length mismatch: expected {} bytes ({} {} values) but got {} bytes",
                    expected,
                    elements,
                    self.dtype,
                    self.data.len()
                ),
                source: None,
            });
        }

        Ok(())
    }

    pub fn view(&self) -> TensorView<'_> {
        TensorView {
            shape: &self.shape,
            dtype: self.dtype,
            data: &self.data,
        }
    }

}

impl<'a> BinaryTensorPacketRef<'a> {
    pub fn to_owned(self) -> BinaryTensorPacket {
        BinaryTensorPacket {
            shape: self.shape,
            dtype: self.dtype,
            data: self.data.into_owned(),
        }
    }
}

impl<'a> From<&'a BinaryTensorPacket> for BinaryTensorPacketRef<'a> {
    fn from(packet: &'a BinaryTensorPacket) -> Self {
        Self {
            shape: packet.shape.clone(),
            dtype: packet.dtype,
            data: Cow::Borrowed(&packet.data),
        }
    }
}

fn shape_elements(shape: &[i64]) -> Result<usize, EngineError> {
    if shape.is_empty() {
        return Ok(1);
    }

    let mut prod: usize = 1;
    for &dim in shape {
        if dim <= 0 {
            return Err(EngineError::InvalidInput {
                message: format!("Invalid shape dimension: {}", dim),
                source: None,
            });
        }
        prod = prod
            .checked_mul(dim as usize)
            .ok_or_else(|| EngineError::InvalidInput {
                message: "Shape multiplication overflow".to_string(),
                source: None,
            })?;
    }

    Ok(prod)
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RequestMetadata {
    #[serde(default)]
    pub request_id: Option<String>,
    #[serde(default)]
    pub timeout_ms: Option<u64>,
    #[serde(default)]
    pub priority: Option<u8>,
    #[serde(default)]
    pub force_cpu: Option<bool>,
    #[serde(default)]
    pub model_version: Option<String>,
    // Keep this field present for sequence-based formats such as bincode.
    // Omitting it shifts every following field and breaks request decoding.
    #[serde(default)]
    pub auth_token: Option<String>,

    // === Optional LLM overrides ===
    #[serde(default, alias = "max_tokens")]
    pub max_new_tokens: Option<u32>,
    #[serde(default, alias = "min_tokens")]
    pub min_new_tokens: Option<u32>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub top_k: Option<u32>,
    #[serde(default)]
    pub repetition_penalty: Option<f32>,
    #[serde(default)]
    pub seed: Option<u64>,
    #[serde(default, alias = "stop_ids")]
    pub stop_token_ids: Option<Vec<u32>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NamedTensor {
    pub name: String,
    pub tensor: BinaryTensorPacket,
}

#[derive(Debug, Clone, Default)]
pub struct CancellationToken {
    cancelled: Arc<AtomicBool>,
}

impl CancellationToken {
    pub fn new() -> Self {
        Self {
            cancelled: Arc::new(AtomicBool::new(false)),
        }
    }

    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::SeqCst);
    }

    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::SeqCst)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceRequest {
    pub input: BinaryTensorPacket,
    #[serde(default)]
    pub additional_inputs: Vec<NamedTensor>,
    #[serde(default)]
    pub session_id: Option<String>,
    #[serde(default)]
    pub metadata: Option<RequestMetadata>,
    #[serde(skip, default)]
    pub cancellation: Option<CancellationToken>,
}

impl InferenceRequest {
    pub fn new(input: BinaryTensorPacket) -> Self {
        Self {
            input,
            additional_inputs: Vec::new(),
            session_id: None,
            metadata: None,
            cancellation: None,
        }
    }

    pub fn with_session_id(mut self, session_id: impl Into<String>) -> Self {
        self.session_id = Some(session_id.into());
        self
    }

    pub fn with_metadata(mut self, metadata: RequestMetadata) -> Self {
        self.metadata = Some(metadata);
        self
    }

    pub fn add_input(&mut self, name: impl Into<String>, tensor: BinaryTensorPacket) {
        self.additional_inputs.push(NamedTensor {
            name: name.into(),
            tensor,
        });
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineModelInfo {
    pub input_names: Vec<String>,
    pub output_names: Vec<String>,
    pub input_shapes: Vec<Vec<i64>>,
    pub output_shapes: Vec<Vec<i64>>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub input_dtypes: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub output_dtypes: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub framework: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_version: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub peak_concurrency: Option<u32>,
}

pub type EngineStream = Pin<Box<dyn Stream<Item = Result<BinaryTensorPacket, EngineError>> + Send>>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BatchingMode {
    /// Do not intentionally coalesce requests in the outer scheduler.
    None,
    /// Stack compatible independent requests and execute them via `infer_batch`.
    RequestCoalescing,
    /// Backend owns active-sequence/token batching internally.
    Continuous,
    /// Backend or downstream service owns batching; the local scheduler should
    /// avoid request-level coalescing.
    Delegated,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BatchingPolicy {
    pub mode: BatchingMode,
    pub max_requests: usize,
    pub queue_delay_ms: Option<u64>,
    pub max_batched_tokens: Option<usize>,
    pub supports_priority: bool,
}

impl BatchingPolicy {
    pub fn none() -> Self {
        Self {
            mode: BatchingMode::None,
            max_requests: 1,
            queue_delay_ms: None,
            max_batched_tokens: None,
            supports_priority: false,
        }
    }

    pub fn request_coalescing(max_requests: usize) -> Self {
        Self {
            mode: BatchingMode::RequestCoalescing,
            max_requests: max_requests.max(1),
            queue_delay_ms: None,
            max_batched_tokens: None,
            supports_priority: false,
        }
    }

    pub fn continuous(max_requests: usize) -> Self {
        Self {
            mode: BatchingMode::Continuous,
            max_requests: max_requests.max(1),
            queue_delay_ms: None,
            max_batched_tokens: None,
            supports_priority: false,
        }
    }

    pub fn delegated() -> Self {
        Self {
            mode: BatchingMode::Delegated,
            max_requests: 1,
            queue_delay_ms: None,
            max_batched_tokens: None,
            supports_priority: false,
        }
    }

    pub fn with_queue_delay_ms(mut self, queue_delay_ms: u64) -> Self {
        self.queue_delay_ms = Some(queue_delay_ms);
        self
    }

    pub fn with_priority_support(mut self) -> Self {
        self.supports_priority = true;
        self
    }

    pub fn from_legacy(max_batch: usize, self_batches: bool) -> Self {
        if self_batches {
            Self::continuous(max_batch.max(1)).with_priority_support()
        } else if max_batch > 1 {
            Self::request_coalescing(max_batch)
        } else {
            Self::none()
        }
    }
}

#[async_trait]
pub trait Engine: Send + Sync {
    /// Load model weights and prepare runtime state.
    async fn load(&mut self, model_path: &std::path::Path) -> Result<(), EngineError>;

    /// Run a single inference request and return the output tensor.
    fn infer(&self, request: &InferenceRequest) -> Result<BinaryTensorPacket, EngineError>;

    /// Run a batch of inference requests.
    fn infer_batch(
        &self,
        requests: &[InferenceRequest],
    ) -> Result<Vec<BinaryTensorPacket>, EngineError> {
        requests.iter().map(|req| self.infer(req)).collect()
    }

    /// Maximum number of independent requests this engine can coalesce into a
    /// single batched execution via [`Engine::infer_batch`].
    ///
    /// Returns 1 when batching is unsupported or unprofitable (the default), so
    /// the scheduler dispatches requests one at a time. Backends that implement
    /// a real (non-serial) `infer_batch` — e.g. an ONNX model with a dynamic
    /// batch dimension — return a value > 1 so the scheduler's micro-batcher
    /// coalesces pending requests before dispatch.
    fn max_batch(&self) -> usize {
        1
    }

    /// Whether this backend performs its own internal batching over concurrent
    /// requests (e.g. an autoregressive LLM that continuously batches active
    /// sequences at the decode step).
    ///
    /// Returns `false` by default. When `true`, the scheduler must NOT coalesce
    /// this backend's requests via [`Engine::infer_batch`] — doing so would run
    /// a request-level batch to completion in one call and fight the backend's
    /// own continuous batcher, reintroducing head-of-line blocking. Instead the
    /// scheduler dispatches such requests individually and lets the backend
    /// multiplex them, gating concurrency by the backend's published occupancy
    /// (see [`Engine::metrics`]). A self-batching backend therefore keeps
    /// [`Engine::max_batch`] at 1.
    fn self_batches(&self) -> bool {
        false
    }

    /// Explicit batching/admission contract for this backend.
    ///
    /// The default preserves legacy behavior by deriving the policy from
    /// `max_batch()` and `self_batches()`. Backends and wrappers should override
    /// this when they need to preserve richer policy details.
    fn batching_policy(&self) -> BatchingPolicy {
        BatchingPolicy::from_legacy(self.max_batch(), self.self_batches())
    }

    /// Run a streaming inference request.
    fn infer_stream(&self, request: &InferenceRequest) -> EngineStream;

    /// Warm up the model runtime before serving requests.
    async fn warmup(&self) -> Result<(), EngineError> {
        Ok(())
    }

    /// Release any held resources.
    fn unload(&mut self);

    /// Report the latest metrics snapshot.
    fn metrics(&self) -> EngineMetrics;

    /// Report model metadata when available.
    fn model_info(&self) -> Option<EngineModelInfo> {
        None
    }

    /// Check if the model is healthy.
    fn health_check(&self) -> Result<(), EngineError>;

    /// Whether this backend supports live weight hot-swap.
    fn supports_swap(&self) -> bool {
        false
    }

    /// Whether this backend currently has weights staged in CPU RAM and ready
    /// for a `swap()` call.  Always `false` for backends that don't support swap.
    fn is_staged(&self) -> bool {
        false
    }

    /// Pre-load a model from `path` into CPU staging RAM so a future `swap()`
    /// only needs the PCIe transfer, not the disk read.
    async fn stage(&self, _path: &std::path::Path) -> Result<(), EngineError> {
        Err(EngineError::backend(
            "hot-swap staging not supported by this backend",
        ))
    }

    /// Atomically replace the live GPU weights with the previously staged model.
    /// All active sessions are invalidated; in-flight requests must be drained
    /// by the caller before calling this.
    async fn swap(&self) -> Result<(), EngineError> {
        Err(EngineError::backend(
            "hot-swap not supported by this backend",
        ))
    }
}

#[async_trait]
impl Engine for Box<dyn Engine> {
    async fn load(&mut self, model_path: &std::path::Path) -> Result<(), EngineError> {
        (**self).load(model_path).await
    }

    fn infer(&self, request: &InferenceRequest) -> Result<BinaryTensorPacket, EngineError> {
        (**self).infer(request)
    }

    fn infer_batch(
        &self,
        requests: &[InferenceRequest],
    ) -> Result<Vec<BinaryTensorPacket>, EngineError> {
        (**self).infer_batch(requests)
    }

    fn max_batch(&self) -> usize {
        (**self).max_batch()
    }

    fn self_batches(&self) -> bool {
        (**self).self_batches()
    }

    fn batching_policy(&self) -> BatchingPolicy {
        (**self).batching_policy()
    }

    fn infer_stream(&self, request: &InferenceRequest) -> EngineStream {
        (**self).infer_stream(request)
    }

    async fn warmup(&self) -> Result<(), EngineError> {
        (**self).warmup().await
    }

    fn unload(&mut self) {
        (**self).unload()
    }

    fn metrics(&self) -> EngineMetrics {
        (**self).metrics()
    }

    fn model_info(&self) -> Option<EngineModelInfo> {
        (**self).model_info()
    }

    fn health_check(&self) -> Result<(), EngineError> {
        (**self).health_check()
    }

    fn supports_swap(&self) -> bool {
        (**self).supports_swap()
    }

    fn is_staged(&self) -> bool {
        (**self).is_staged()
    }

    async fn stage(&self, path: &std::path::Path) -> Result<(), EngineError> {
        (**self).stage(path).await
    }

    async fn swap(&self) -> Result<(), EngineError> {
        (**self).swap().await
    }
}

pub type EngineHandle = Arc<dyn Engine>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn binary_tensor_packet_deserializes_from_data_array() {
        let payload = serde_json::json!({
            "shape": [1, 2],
            "dtype": "uint8",
            "data": [1, 2]
        });
        let packet: BinaryTensorPacket =
            serde_json::from_value(payload).expect("packet should deserialize");
        assert_eq!(packet.shape, vec![1, 2]);
        assert_eq!(packet.dtype, TensorDtype::Uint8);
        assert_eq!(packet.data, vec![1, 2]);
    }

    #[test]
    fn binary_tensor_packet_deserializes_from_data_base64() {
        let payload = serde_json::json!({
            "shape": [1, 4],
            "dtype": "uint8",
            "data_base64": "AQIDBA=="
        });
        let packet: BinaryTensorPacket =
            serde_json::from_value(payload).expect("packet should deserialize");
        assert_eq!(packet.shape, vec![1, 4]);
        assert_eq!(packet.dtype, TensorDtype::Uint8);
        assert_eq!(packet.data, vec![1, 2, 3, 4]);
    }

    #[test]
    fn binary_tensor_packet_serializes_json_as_data_base64() {
        let packet = BinaryTensorPacket {
            shape: vec![1, 4],
            dtype: TensorDtype::Uint8,
            data: vec![1, 2, 3, 4],
        };

        let payload = serde_json::to_value(packet).expect("packet should serialize");

        assert_eq!(payload["shape"], serde_json::json!([1, 4]));
        assert_eq!(payload["dtype"], serde_json::json!("uint8"));
        assert!(payload["data"].is_null());
        assert_eq!(payload["data_base64"], serde_json::json!("AQIDBA=="));
    }

    #[test]
    fn binary_tensor_packet_bincode_roundtrips_raw_data_layout() {
        let packet = BinaryTensorPacket {
            shape: vec![1, 4],
            dtype: TensorDtype::Uint8,
            data: vec![1, 2, 3, 4],
        };

        let encoded = bincode::serialize(&packet).expect("packet should serialize");
        let decoded: BinaryTensorPacket =
            bincode::deserialize(&encoded).expect("packet should deserialize");

        assert_eq!(decoded.shape, packet.shape);
        assert_eq!(decoded.dtype, packet.dtype);
        assert_eq!(decoded.data, packet.data);
    }

    #[test]
    fn binary_tensor_packet_deserializes_from_base64_alias() {
        let payload = serde_json::json!({
            "shape": [1, 3],
            "dtype": "uint8",
            "base64": "AQID"
        });
        let packet: BinaryTensorPacket =
            serde_json::from_value(payload).expect("packet should deserialize");
        assert_eq!(packet.shape, vec![1, 3]);
        assert_eq!(packet.dtype, TensorDtype::Uint8);
        assert_eq!(packet.data, vec![1, 2, 3]);
    }

    #[test]
    fn binary_tensor_packet_rejects_both_data_and_data_base64() {
        let payload = serde_json::json!({
            "shape": [1],
            "dtype": "uint8",
            "data": [1],
            "data_base64": "AQ=="
        });
        let err = serde_json::from_value::<BinaryTensorPacket>(payload)
            .expect_err("packet should fail deserialization");
        assert!(err
            .to_string()
            .contains("only one of `data` or `data_base64`"));
    }

    #[test]
    fn batching_policy_derives_from_legacy_capabilities() {
        assert_eq!(
            BatchingPolicy::from_legacy(1, false),
            BatchingPolicy::none()
        );

        let coalescing = BatchingPolicy::from_legacy(8, false);
        assert_eq!(coalescing.mode, BatchingMode::RequestCoalescing);
        assert_eq!(coalescing.max_requests, 8);

        let continuous = BatchingPolicy::from_legacy(1, true);
        assert_eq!(continuous.mode, BatchingMode::Continuous);
        assert_eq!(continuous.max_requests, 1);
        assert!(continuous.supports_priority);
    }

    #[test]
    fn inference_request_bincode_preserves_metadata_without_auth_token() {
        let mut metadata = RequestMetadata::default();
        metadata.priority = Some(0);
        metadata.max_new_tokens = Some(32);
        let request = InferenceRequest::new(BinaryTensorPacket {
            shape: vec![1],
            dtype: TensorDtype::Float32,
            data: 1.0f32.to_ne_bytes().to_vec(),
        })
        .with_metadata(metadata);

        let encoded = bincode::serialize(&request).expect("request should serialize");
        let decoded: InferenceRequest =
            bincode::deserialize(&encoded).expect("request should deserialize");
        let decoded_metadata = decoded.metadata.expect("metadata should round-trip");

        assert_eq!(decoded_metadata.priority, Some(0));
        assert_eq!(decoded_metadata.max_new_tokens, Some(32));
        assert_eq!(decoded_metadata.auth_token, None);
    }
}
