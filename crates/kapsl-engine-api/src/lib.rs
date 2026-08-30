use async_trait::async_trait;
use base64::Engine as _;
use futures::{stream::Stream, StreamExt};
use serde::{Deserialize, Serialize};
use std::borrow::Cow;
use std::fmt;
use std::future::Future;
use std::pin::Pin;
use std::str::FromStr;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll, Waker};
use std::time::{SystemTime, UNIX_EPOCH};
use thiserror::Error;

pub use kapsl_kv_abi::{KvBackendCapabilities, KvIntegrationTier, KvMetadataMode, KvTopology};

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

#[derive(Debug, Default)]
struct CancellationState {
    cancelled: AtomicBool,
    next_waiter_id: AtomicU64,
    waiters: Mutex<Vec<(u64, Waker)>>,
}

#[derive(Debug, Clone, Default)]
pub struct CancellationToken {
    state: Arc<CancellationState>,
}

impl CancellationToken {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn cancel(&self) {
        if self.state.cancelled.swap(true, Ordering::SeqCst) {
            return;
        }
        let waiters = {
            let mut waiters = self
                .state
                .waiters
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            std::mem::take(&mut *waiters)
        };
        for (_, waiter) in waiters {
            waiter.wake();
        }
    }

    pub fn is_cancelled(&self) -> bool {
        self.state.cancelled.load(Ordering::SeqCst)
    }

    /// Wait until cancellation without polling or retaining a dropped task.
    pub fn cancelled(&self) -> impl Future<Output = ()> + Send + 'static {
        CancellationWaiter {
            state: Arc::clone(&self.state),
            waiter_id: self.state.next_waiter_id.fetch_add(1, Ordering::Relaxed),
            registered: false,
        }
    }
}

struct CancellationWaiter {
    state: Arc<CancellationState>,
    waiter_id: u64,
    registered: bool,
}

impl Future for CancellationWaiter {
    type Output = ();

    fn poll(mut self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Self::Output> {
        if self.state.cancelled.load(Ordering::SeqCst) {
            return Poll::Ready(());
        }
        let mut waiters = self
            .state
            .waiters
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if self.state.cancelled.load(Ordering::SeqCst) {
            return Poll::Ready(());
        }
        if let Some((_, waiter)) = waiters
            .iter_mut()
            .find(|(waiter_id, _)| *waiter_id == self.waiter_id)
        {
            if !waiter.will_wake(context.waker()) {
                *waiter = context.waker().clone();
            }
        } else {
            waiters.push((self.waiter_id, context.waker().clone()));
        }
        drop(waiters);
        self.registered = true;
        Poll::Pending
    }
}

impl Drop for CancellationWaiter {
    fn drop(&mut self) {
        if !self.registered {
            return;
        }
        let mut waiters = self
            .state
            .waiters
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        waiters.retain(|(waiter_id, _)| *waiter_id != self.waiter_id);
    }
}

/// One-shot hook used by a self-scheduling backend to acquire request memory
/// only after it has assigned an execution slot.
///
/// The returned guard owns the runtime lease. Backends must hold it until the
/// slot's backing memory has been released; dropping it returns the reservation
/// to the runtime authority.
#[derive(Clone)]
pub struct RequestMemoryAdmission {
    inner: Arc<RequestMemoryAdmissionInner>,
}

struct RequestMemoryAdmissionInner {
    acquired: AtomicBool,
    acquire:
        Box<dyn Fn() -> Result<RequestMemoryAdmissionGuard, EngineError> + Send + Sync + 'static>,
}

impl RequestMemoryAdmission {
    pub fn new<F, T>(acquire: F) -> Self
    where
        F: Fn() -> Result<T, EngineError> + Send + Sync + 'static,
        T: Send + 'static,
    {
        Self {
            inner: Arc::new(RequestMemoryAdmissionInner {
                acquired: AtomicBool::new(false),
                acquire: Box::new(move || acquire().map(RequestMemoryAdmissionGuard::new)),
            }),
        }
    }

    /// Acquire this request's memory exactly once.
    pub fn acquire(&self) -> Result<RequestMemoryAdmissionGuard, EngineError> {
        if self.inner.acquired.swap(true, Ordering::AcqRel) {
            return Err(EngineError::backend(
                "request memory admission was acquired more than once",
            ));
        }
        (self.inner.acquire)()
    }
}

impl fmt::Debug for RequestMemoryAdmission {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("RequestMemoryAdmission")
            .field("acquired", &self.inner.acquired.load(Ordering::Acquire))
            .finish_non_exhaustive()
    }
}

/// Opaque request-lifetime memory lease returned by
/// [`RequestMemoryAdmission::acquire`].
pub struct RequestMemoryAdmissionGuard {
    _lease: Box<dyn Send + 'static>,
}

impl RequestMemoryAdmissionGuard {
    fn new<T: Send + 'static>(lease: T) -> Self {
        Self {
            _lease: Box::new(lease),
        }
    }
}

impl fmt::Debug for RequestMemoryAdmissionGuard {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("RequestMemoryAdmissionGuard")
            .finish_non_exhaustive()
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

/// Version of the protocol-native OpenAI request/response contract.
///
/// This protocol is carried by dedicated transport operation codes. It is not
/// appended to [`InferenceRequest`], whose sequence-serialized bincode layout
/// must remain readable by older clients and servers.
pub const OPENAI_WIRE_PROTOCOL_VERSION: u16 = 1;

/// Private managed-backend endpoint selected by Kapsl.
///
/// An enum deliberately prevents a caller from smuggling an arbitrary host or
/// path through the protocol-native fast path.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OpenAiWireEndpoint {
    ChatCompletions,
    Completions,
}

impl OpenAiWireEndpoint {
    pub const fn path(self) -> &'static str {
        match self {
            Self::ChatCompletions => "/v1/chat/completions",
            Self::Completions => "/v1/completions",
        }
    }
}

/// Expected upstream representation. Streaming responses are relayed as raw
/// SSE bytes and are never assumed to align with network chunks or events.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OpenAiWireFormat {
    Json,
    ServerSentEvents,
}

/// Internal policy metadata for a protocol-native OpenAI operation.
///
/// This deliberately excludes authorization and sampling fields. Transport
/// credentials live in a transport-specific envelope and are consumed before
/// this engine-facing type crosses the scheduler boundary.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct OpenAiWireMetadata {
    #[serde(default)]
    pub request_id: Option<String>,
    #[serde(default)]
    pub timeout_ms: Option<u64>,
    #[serde(default)]
    pub priority: Option<u8>,
}

/// A validated OpenAI operation after public ingress policy has run.
///
/// The body is the one normalized serialization forwarded to the managed
/// backend. Client authorization is absent from this type and is never placed
/// in `body`; the engine chooses its own private endpoint.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAiWireRequest {
    pub version: u16,
    pub endpoint: OpenAiWireEndpoint,
    pub format: OpenAiWireFormat,
    pub body: Vec<u8>,
    #[serde(default)]
    pub session_id: Option<String>,
    #[serde(default)]
    pub metadata: Option<OpenAiWireMetadata>,
    #[serde(skip, default)]
    pub cancellation: Option<CancellationToken>,
}

impl OpenAiWireRequest {
    pub fn new(endpoint: OpenAiWireEndpoint, format: OpenAiWireFormat, body: Vec<u8>) -> Self {
        Self {
            version: OPENAI_WIRE_PROTOCOL_VERSION,
            endpoint,
            format,
            body,
            session_id: None,
            metadata: None,
            cancellation: None,
        }
    }

    pub fn with_session_id(mut self, session_id: impl Into<String>) -> Self {
        self.session_id = Some(session_id.into());
        self
    }

    pub fn with_metadata(mut self, metadata: OpenAiWireMetadata) -> Self {
        self.metadata = Some(metadata);
        self
    }

    pub fn validate(&self, maximum_body_bytes: usize) -> Result<(), EngineError> {
        if self.version != OPENAI_WIRE_PROTOCOL_VERSION {
            return Err(EngineError::invalid_input(format!(
                "unsupported OpenAI wire protocol version {}; expected {}",
                self.version, OPENAI_WIRE_PROTOCOL_VERSION
            )));
        }
        if self.body.is_empty() {
            return Err(EngineError::invalid_input(
                "OpenAI wire request body must not be empty",
            ));
        }
        if self.body.len() > maximum_body_bytes {
            return Err(EngineError::resource_exhausted(format!(
                "OpenAI wire request body is {} bytes; maximum is {} bytes",
                self.body.len(),
                maximum_body_bytes
            )));
        }
        if self
            .session_id
            .as_deref()
            .is_some_and(|session| session.trim().is_empty())
        {
            return Err(EngineError::invalid_input(
                "OpenAI wire session ID must not be empty when present",
            ));
        }
        Ok(())
    }
}

/// Response headers that may cross the private managed-backend boundary.
/// Hop-by-hop and security-sensitive headers have no representation here.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OpenAiWireHeaderName {
    ContentType,
    CacheControl,
    RequestId,
    RetryAfter,
    ProcessingMilliseconds,
}

impl OpenAiWireHeaderName {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::ContentType => "content-type",
            Self::CacheControl => "cache-control",
            Self::RequestId => "x-request-id",
            Self::RetryAfter => "retry-after",
            Self::ProcessingMilliseconds => "openai-processing-ms",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OpenAiWireHeader {
    pub name: OpenAiWireHeaderName,
    pub value: Vec<u8>,
}

impl OpenAiWireHeader {
    pub fn new(name: OpenAiWireHeaderName, value: impl Into<Vec<u8>>) -> Result<Self, EngineError> {
        let value = value.into();
        if value.iter().any(|byte| matches!(byte, b'\r' | b'\n')) {
            return Err(EngineError::invalid_input(format!(
                "OpenAI wire response header '{}' contains a line break",
                name.as_str()
            )));
        }
        Ok(Self { name, value })
    }
}

/// Status and allowlisted headers emitted before either a JSON body or an SSE
/// byte stream.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OpenAiWireResponseHead {
    pub version: u16,
    pub status: u16,
    pub headers: Vec<OpenAiWireHeader>,
}

impl OpenAiWireResponseHead {
    pub fn new(status: u16, headers: Vec<OpenAiWireHeader>) -> Result<Self, EngineError> {
        if !(100..=599).contains(&status) {
            return Err(EngineError::invalid_input(format!(
                "OpenAI wire response status {status} is outside the HTTP range"
            )));
        }
        Ok(Self {
            version: OPENAI_WIRE_PROTOCOL_VERSION,
            status,
            headers,
        })
    }

    pub fn validate(&self) -> Result<(), EngineError> {
        if self.version != OPENAI_WIRE_PROTOCOL_VERSION {
            return Err(EngineError::invalid_input(format!(
                "unsupported OpenAI wire response version {}; expected {}",
                self.version, OPENAI_WIRE_PROTOCOL_VERSION
            )));
        }
        if !(100..=599).contains(&self.status) {
            return Err(EngineError::invalid_input(format!(
                "OpenAI wire response status {} is outside the HTTP range",
                self.status
            )));
        }
        for header in &self.headers {
            OpenAiWireHeader::new(header.name, header.value.clone())?;
        }
        Ok(())
    }
}

/// Complete protocol-native response used for non-streaming operations.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OpenAiWireResponse {
    pub head: OpenAiWireResponseHead,
    pub body: Vec<u8>,
}

/// Raw response body chunks for a protocol-native streaming operation.
pub type OpenAiWireStream =
    Pin<Box<dyn Stream<Item = Result<Vec<u8>, EngineError>> + Send + 'static>>;

pub struct OpenAiWireStreamResponse {
    pub head: OpenAiWireResponseHead,
    pub body: OpenAiWireStream,
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

/// A backend allocation that lives outside the runtime-owned device pool.
///
/// `allocation_id` identifies the physical allocation, rather than an engine
/// instance. Backends that share immutable weights between replicas must return
/// the same ID so the runtime charges those bytes once and reference-counts the
/// owners.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExternalDeviceMemory {
    pub allocation_id: String,
    pub device_id: usize,
    pub bytes: usize,
}

/// Planned or actual backend-owned device memory outside the shared pool.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExternalDeviceMemoryReport {
    #[serde(default)]
    pub allocations: Vec<ExternalDeviceMemory>,
}

impl ExternalDeviceMemoryReport {
    pub fn single(allocation_id: impl Into<String>, device_id: usize, bytes: usize) -> Self {
        Self {
            allocations: vec![ExternalDeviceMemory {
                allocation_id: allocation_id.into(),
                device_id,
                bytes,
            }],
        }
    }

    pub fn bytes_for_device(&self, device_id: usize) -> usize {
        self.allocations
            .iter()
            .filter(|allocation| allocation.device_id == device_id)
            .map(|allocation| allocation.bytes)
            .sum()
    }
}

/// Physical/accounting domain for backend-owned memory.
///
/// Unlike [`ExternalDeviceMemoryReport`], this represents host, page-locked,
/// mapped, CUDA, and other provider allocations without flattening them into a
/// CUDA device ID.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "kebab-case")]
pub enum MemoryDomain {
    Host,
    HostPinned {
        provider: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        device_id: Option<usize>,
    },
    HostMapped {
        provider: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        device_id: Option<usize>,
    },
    Cuda {
        device_id: usize,
    },
    Provider {
        provider: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        device_id: Option<usize>,
    },
}

/// Backend-neutral purpose of an allocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum MemoryAllocationClass {
    PersistentWeights,
    ModelSession,
    KvCache,
    TransientWorkspace,
    BlockTable,
    RequestTransient,
    ExternallyOwned,
}

/// Which component owns the physical allocation represented by a report row.
/// Runtime-managed rows are suballocated from a memory authority/pool; backend
/// rows must be admitted and reconciled as external allocations.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum MemoryAllocationSource {
    RuntimeManaged,
    #[default]
    BackendManaged,
}

/// One stable physical allocation or one accounting reservation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MemoryAllocation {
    pub allocation_id: String,
    pub domain: MemoryDomain,
    pub class: MemoryAllocationClass,
    #[serde(default)]
    pub source: MemoryAllocationSource,
    pub bytes: usize,
}

/// Planned or actual memory held by a backend.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct MemoryReport {
    #[serde(default)]
    pub allocations: Vec<MemoryAllocation>,
}

impl MemoryReport {
    pub fn single(
        allocation_id: impl Into<String>,
        domain: MemoryDomain,
        class: MemoryAllocationClass,
        bytes: usize,
    ) -> Self {
        Self {
            allocations: vec![MemoryAllocation {
                allocation_id: allocation_id.into(),
                domain,
                class,
                source: MemoryAllocationSource::BackendManaged,
                bytes,
            }],
        }
    }

    pub fn runtime(
        allocation_id: impl Into<String>,
        domain: MemoryDomain,
        class: MemoryAllocationClass,
        bytes: usize,
    ) -> Self {
        Self {
            allocations: vec![MemoryAllocation {
                allocation_id: allocation_id.into(),
                domain,
                class,
                source: MemoryAllocationSource::RuntimeManaged,
                bytes,
            }],
        }
    }

    pub fn push(&mut self, allocation: MemoryAllocation) -> &mut Self {
        self.allocations.push(allocation);
        self
    }

    pub fn extend(&mut self, other: Self) -> &mut Self {
        self.allocations.extend(other.allocations);
        self
    }

    pub fn bytes_for_domain(&self, domain: &MemoryDomain) -> usize {
        self.allocations
            .iter()
            .filter(|allocation| &allocation.domain == domain)
            .map(|allocation| allocation.bytes)
            .sum()
    }
}

impl From<ExternalDeviceMemoryReport> for MemoryReport {
    fn from(report: ExternalDeviceMemoryReport) -> Self {
        Self {
            allocations: report
                .allocations
                .into_iter()
                .map(|allocation| MemoryAllocation {
                    allocation_id: allocation.allocation_id,
                    domain: MemoryDomain::Cuda {
                        device_id: allocation.device_id,
                    },
                    class: MemoryAllocationClass::ExternallyOwned,
                    source: MemoryAllocationSource::BackendManaged,
                    bytes: allocation.bytes,
                })
                .collect(),
        }
    }
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
    /// Report how deeply this backend participates in Kapsl's KV memory plane.
    ///
    /// The default is deliberately conservative: ordinary inference backends
    /// and OpenAI-compatible endpoints are routable but remain unmanaged until
    /// they implement the versioned KV participant contract.
    fn kv_capabilities(&self) -> KvBackendCapabilities {
        KvBackendCapabilities::unmanaged()
    }

    /// Report the loaded model's cache-group topology when structured KV
    /// metadata is available. Opaque and unmanaged backends return `None`.
    fn kv_topology(&self) -> Option<KvTopology> {
        None
    }

    /// Report memory expected during `load` across every memory domain.
    ///
    /// Legacy backends automatically map their external CUDA report into this
    /// representation. New implementations should override this method when
    /// they also own host, pinned, mapped, or non-CUDA provider memory.
    fn planned_memory(&self, model_path: &std::path::Path) -> Result<MemoryReport, EngineError> {
        self.planned_external_device_memory(model_path)
            .map(Into::into)
    }

    /// Report external device allocations expected during `load`.
    ///
    /// The default is empty for CPU backends and legacy implementations. The
    /// runtime still observes CUDA free-memory deltas as a conservative fallback.
    fn planned_external_device_memory(
        &self,
        _model_path: &std::path::Path,
    ) -> Result<ExternalDeviceMemoryReport, EngineError> {
        Ok(ExternalDeviceMemoryReport::default())
    }

    /// Load model weights and prepare runtime state.
    async fn load(&mut self, model_path: &std::path::Path) -> Result<(), EngineError>;

    /// Report the external device allocations currently held by this backend.
    ///
    /// Backends should return an empty report after `unload` has synchronously
    /// released those allocations.
    fn actual_external_device_memory(&self) -> ExternalDeviceMemoryReport {
        ExternalDeviceMemoryReport::default()
    }

    /// Report memory currently retained by this backend across all domains.
    fn actual_memory(&self) -> MemoryReport {
        self.actual_external_device_memory().into()
    }

    /// Report the transient memory Kapsl should reserve while one request is
    /// active. The default covers materialized host input tensors; backends can
    /// add pinned staging, mapped buffers, outputs, and execution workspace.
    fn planned_request_memory(&self, request: &InferenceRequest) -> MemoryReport {
        let bytes = request
            .additional_inputs
            .iter()
            .map(|input| input.tensor.data.len())
            .fold(request.input.data.len(), usize::saturating_add);
        MemoryReport::single(
            "request:materialized-inputs",
            MemoryDomain::Host,
            MemoryAllocationClass::RequestTransient,
            bytes,
        )
    }

    /// Run one request under a runtime-provided memory admission hook.
    ///
    /// The default acquires immediately before inference. Self-scheduling
    /// backends may override this to enqueue the hook and acquire it only when
    /// they assign an execution slot, but must hold the returned guard until
    /// all backing memory for that slot has been released.
    fn infer_with_memory_admission(
        &self,
        request: &InferenceRequest,
        admission: RequestMemoryAdmission,
    ) -> Result<BinaryTensorPacket, EngineError> {
        let _guard = admission.acquire()?;
        self.infer(request)
    }

    /// Run a single inference request and return the output tensor.
    fn infer(&self, request: &InferenceRequest) -> Result<BinaryTensorPacket, EngineError>;

    /// Whether this engine accepts protocol-native OpenAI wire operations.
    fn supports_openai_wire(&self) -> bool {
        false
    }

    /// Report transient memory held while one protocol-native OpenAI request
    /// is active. The default accounts for the normalized serialized body;
    /// managed backends can add staging, block-table, or request-scoped KV
    /// rows when those are not already covered by persistent capacity.
    fn planned_openai_wire_request_memory(&self, request: &OpenAiWireRequest) -> MemoryReport {
        MemoryReport::single(
            "request:openai-wire-body",
            MemoryDomain::Host,
            MemoryAllocationClass::RequestTransient,
            request.body.len(),
        )
    }

    /// Run one protocol-native OpenAI request under a runtime-provided memory
    /// admission hook. The guard remains live until the complete response has
    /// been received from the backend.
    async fn infer_openai_wire_with_memory_admission(
        &self,
        request: &OpenAiWireRequest,
        admission: RequestMemoryAdmission,
    ) -> Result<OpenAiWireResponse, EngineError> {
        let _guard = admission.acquire()?;
        self.infer_openai_wire(request).await
    }

    /// Forward one validated OpenAI operation without translating its JSON
    /// response into tensors. Public ingress policy and replica selection run
    /// before this engine boundary.
    async fn infer_openai_wire(
        &self,
        _request: &OpenAiWireRequest,
    ) -> Result<OpenAiWireResponse, EngineError> {
        Err(EngineError::backend(
            "engine does not support protocol-native OpenAI requests",
        ))
    }

    /// Start one OpenAI SSE relay. Returning the response head separately lets
    /// the public route preserve an upstream non-2xx status before committing
    /// the downstream body stream.
    async fn infer_openai_wire_stream(
        &self,
        _request: &OpenAiWireRequest,
    ) -> Result<OpenAiWireStreamResponse, EngineError> {
        Err(EngineError::backend(
            "engine does not support protocol-native OpenAI streams",
        ))
    }

    /// Start one protocol-native OpenAI stream under a runtime-provided memory
    /// admission hook. The guard is captured by the returned stream and is
    /// released only on completion or downstream cancellation/drop.
    async fn infer_openai_wire_stream_with_memory_admission(
        &self,
        request: &OpenAiWireRequest,
        admission: RequestMemoryAdmission,
    ) -> Result<OpenAiWireStreamResponse, EngineError> {
        let guard = admission.acquire()?;
        let response = self.infer_openai_wire_stream(request).await?;
        Ok(OpenAiWireStreamResponse {
            head: response.head,
            body: Box::pin(response.body.map(move |item| {
                let _hold = &guard;
                item
            })),
        })
    }

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

    /// Stream one request under a runtime-provided memory admission hook.
    ///
    /// The default acquires before constructing the stream and retains the
    /// guard until the stream completes or is dropped. Self-scheduling
    /// backends can override this to acquire at their internal slot boundary.
    fn infer_stream_with_memory_admission(
        &self,
        request: &InferenceRequest,
        admission: RequestMemoryAdmission,
    ) -> EngineStream {
        let guard = match admission.acquire() {
            Ok(guard) => guard,
            Err(error) => return Box::pin(futures::stream::once(async move { Err(error) })),
        };
        Box::pin(self.infer_stream(request).map(move |item| {
            let _hold = &guard;
            item
        }))
    }

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
    fn kv_capabilities(&self) -> KvBackendCapabilities {
        (**self).kv_capabilities()
    }

    fn kv_topology(&self) -> Option<KvTopology> {
        (**self).kv_topology()
    }

    fn planned_memory(&self, model_path: &std::path::Path) -> Result<MemoryReport, EngineError> {
        (**self).planned_memory(model_path)
    }

    fn planned_external_device_memory(
        &self,
        model_path: &std::path::Path,
    ) -> Result<ExternalDeviceMemoryReport, EngineError> {
        (**self).planned_external_device_memory(model_path)
    }

    async fn load(&mut self, model_path: &std::path::Path) -> Result<(), EngineError> {
        (**self).load(model_path).await
    }

    fn actual_external_device_memory(&self) -> ExternalDeviceMemoryReport {
        (**self).actual_external_device_memory()
    }

    fn actual_memory(&self) -> MemoryReport {
        (**self).actual_memory()
    }

    fn planned_request_memory(&self, request: &InferenceRequest) -> MemoryReport {
        (**self).planned_request_memory(request)
    }

    fn infer_with_memory_admission(
        &self,
        request: &InferenceRequest,
        admission: RequestMemoryAdmission,
    ) -> Result<BinaryTensorPacket, EngineError> {
        (**self).infer_with_memory_admission(request, admission)
    }

    fn infer(&self, request: &InferenceRequest) -> Result<BinaryTensorPacket, EngineError> {
        (**self).infer(request)
    }

    fn supports_openai_wire(&self) -> bool {
        (**self).supports_openai_wire()
    }

    fn planned_openai_wire_request_memory(&self, request: &OpenAiWireRequest) -> MemoryReport {
        (**self).planned_openai_wire_request_memory(request)
    }

    async fn infer_openai_wire_with_memory_admission(
        &self,
        request: &OpenAiWireRequest,
        admission: RequestMemoryAdmission,
    ) -> Result<OpenAiWireResponse, EngineError> {
        (**self)
            .infer_openai_wire_with_memory_admission(request, admission)
            .await
    }

    async fn infer_openai_wire(
        &self,
        request: &OpenAiWireRequest,
    ) -> Result<OpenAiWireResponse, EngineError> {
        (**self).infer_openai_wire(request).await
    }

    async fn infer_openai_wire_stream(
        &self,
        request: &OpenAiWireRequest,
    ) -> Result<OpenAiWireStreamResponse, EngineError> {
        (**self).infer_openai_wire_stream(request).await
    }

    async fn infer_openai_wire_stream_with_memory_admission(
        &self,
        request: &OpenAiWireRequest,
        admission: RequestMemoryAdmission,
    ) -> Result<OpenAiWireStreamResponse, EngineError> {
        (**self)
            .infer_openai_wire_stream_with_memory_admission(request, admission)
            .await
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

    fn infer_stream_with_memory_admission(
        &self,
        request: &InferenceRequest,
        admission: RequestMemoryAdmission,
    ) -> EngineStream {
        (**self).infer_stream_with_memory_admission(request, admission)
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

    struct WireAdmissionTestEngine;

    #[async_trait]
    impl Engine for WireAdmissionTestEngine {
        async fn load(&mut self, _model_path: &std::path::Path) -> Result<(), EngineError> {
            Ok(())
        }

        fn infer(&self, _request: &InferenceRequest) -> Result<BinaryTensorPacket, EngineError> {
            Err(EngineError::backend("tensor path is not used"))
        }

        fn supports_openai_wire(&self) -> bool {
            true
        }

        async fn infer_openai_wire_stream(
            &self,
            _request: &OpenAiWireRequest,
        ) -> Result<OpenAiWireStreamResponse, EngineError> {
            Ok(OpenAiWireStreamResponse {
                head: OpenAiWireResponseHead::new(200, Vec::new())?,
                body: Box::pin(futures::stream::iter(vec![
                    Ok(b"first".to_vec()),
                    Ok(b"second".to_vec()),
                ])),
            })
        }

        fn infer_stream(&self, _request: &InferenceRequest) -> EngineStream {
            Box::pin(futures::stream::empty())
        }

        fn unload(&mut self) {}

        fn metrics(&self) -> EngineMetrics {
            EngineMetrics::default()
        }

        fn health_check(&self) -> Result<(), EngineError> {
            Ok(())
        }
    }

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
    fn default_kv_capabilities_are_explicitly_unmanaged() {
        let capabilities = KvBackendCapabilities::default();
        assert_eq!(capabilities.tier, KvIntegrationTier::UnmanagedEndpoint);
        assert!(capabilities.validate().is_ok());
    }

    #[test]
    fn inference_request_bincode_preserves_metadata_without_auth_token() {
        let metadata = RequestMetadata {
            priority: Some(0),
            max_new_tokens: Some(32),
            ..RequestMetadata::default()
        };
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

    #[test]
    fn openai_wire_request_has_an_independent_bincode_layout() {
        let cancellation = CancellationToken::new();
        let mut request = OpenAiWireRequest::new(
            OpenAiWireEndpoint::ChatCompletions,
            OpenAiWireFormat::ServerSentEvents,
            br#"{"model":"served","messages":[{"role":"user","content":"hello"}]}"#.to_vec(),
        )
        .with_session_id("principal:session")
        .with_metadata(OpenAiWireMetadata {
            request_id: Some("req-internal".to_string()),
            timeout_ms: Some(5000),
            priority: Some(1),
        });
        request.cancellation = Some(cancellation);
        request.validate(4096).unwrap();

        let encoded = bincode::serialize(&request).expect("wire request should serialize");
        let decoded: OpenAiWireRequest =
            bincode::deserialize(&encoded).expect("wire request should deserialize");
        assert_eq!(decoded.endpoint.path(), "/v1/chat/completions");
        assert_eq!(decoded.format, OpenAiWireFormat::ServerSentEvents);
        assert_eq!(decoded.body, request.body);
        assert_eq!(decoded.session_id.as_deref(), Some("principal:session"));
        assert_eq!(
            decoded.metadata.as_ref().and_then(|value| value.priority),
            Some(1)
        );
        let metadata_json = serde_json::to_value(decoded.metadata.as_ref().unwrap()).unwrap();
        assert_eq!(
            metadata_json
                .as_object()
                .unwrap()
                .keys()
                .map(String::as_str)
                .collect::<std::collections::BTreeSet<_>>(),
            ["priority", "request_id", "timeout_ms"]
                .into_iter()
                .collect()
        );
        assert!(metadata_json.get("auth_token").is_none());
        assert!(metadata_json.get("temperature").is_none());
        assert!(decoded.cancellation.is_none());
    }

    #[test]
    fn openai_wire_headers_are_allowlisted_and_reject_response_splitting() {
        let content_type = OpenAiWireHeader::new(
            OpenAiWireHeaderName::ContentType,
            b"text/event-stream".to_vec(),
        )
        .unwrap();
        let head = OpenAiWireResponseHead::new(429, vec![content_type]).unwrap();
        head.validate().unwrap();

        assert!(OpenAiWireHeader::new(
            OpenAiWireHeaderName::RequestId,
            b"safe\r\nx-injected: true".to_vec(),
        )
        .is_err());
        assert!(OpenAiWireResponseHead::new(42, Vec::new()).is_err());
    }

    #[test]
    fn request_memory_admission_is_one_shot_and_guard_owns_lease() {
        struct DropMarker(Arc<std::sync::atomic::AtomicUsize>);

        impl Drop for DropMarker {
            fn drop(&mut self) {
                self.0.fetch_add(1, Ordering::SeqCst);
            }
        }

        let acquisitions = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let releases = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let admission = RequestMemoryAdmission::new({
            let acquisitions = Arc::clone(&acquisitions);
            let releases = Arc::clone(&releases);
            move || {
                acquisitions.fetch_add(1, Ordering::SeqCst);
                Ok(DropMarker(Arc::clone(&releases)))
            }
        });

        let guard = admission
            .acquire()
            .expect("first acquisition should succeed");
        assert_eq!(acquisitions.load(Ordering::SeqCst), 1);
        assert_eq!(releases.load(Ordering::SeqCst), 0);
        assert!(admission.acquire().is_err());
        assert_eq!(acquisitions.load(Ordering::SeqCst), 1);

        drop(guard);
        assert_eq!(releases.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn openai_wire_stream_admission_guard_lives_until_stream_drop() {
        struct DropMarker(Arc<std::sync::atomic::AtomicUsize>);

        impl Drop for DropMarker {
            fn drop(&mut self) {
                self.0.fetch_add(1, Ordering::SeqCst);
            }
        }

        let releases = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let admission = RequestMemoryAdmission::new({
            let releases = Arc::clone(&releases);
            move || Ok(DropMarker(Arc::clone(&releases)))
        });
        let request = OpenAiWireRequest::new(
            OpenAiWireEndpoint::ChatCompletions,
            OpenAiWireFormat::ServerSentEvents,
            b"{}".to_vec(),
        );
        let mut response = WireAdmissionTestEngine
            .infer_openai_wire_stream_with_memory_admission(&request, admission)
            .await
            .unwrap();

        assert_eq!(releases.load(Ordering::SeqCst), 0);
        assert_eq!(response.body.next().await.unwrap().unwrap(), b"first");
        assert_eq!(releases.load(Ordering::SeqCst), 0);
        drop(response);
        assert_eq!(releases.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn memory_report_preserves_host_and_device_domains() {
        let report = MemoryReport {
            allocations: vec![
                MemoryAllocation {
                    allocation_id: "session".to_string(),
                    domain: MemoryDomain::Host,
                    class: MemoryAllocationClass::ModelSession,
                    source: MemoryAllocationSource::BackendManaged,
                    bytes: 10,
                },
                MemoryAllocation {
                    allocation_id: "staging".to_string(),
                    domain: MemoryDomain::HostPinned {
                        provider: "cuda".to_string(),
                        device_id: Some(2),
                    },
                    class: MemoryAllocationClass::TransientWorkspace,
                    source: MemoryAllocationSource::BackendManaged,
                    bytes: 20,
                },
                MemoryAllocation {
                    allocation_id: "weights".to_string(),
                    domain: MemoryDomain::Cuda { device_id: 2 },
                    class: MemoryAllocationClass::PersistentWeights,
                    source: MemoryAllocationSource::RuntimeManaged,
                    bytes: 30,
                },
            ],
        };
        assert_eq!(report.bytes_for_domain(&MemoryDomain::Host), 10);
        assert_eq!(
            report.bytes_for_domain(&MemoryDomain::Cuda { device_id: 2 }),
            30
        );
        let encoded = serde_json::to_value(&report).unwrap();
        let decoded: MemoryReport = serde_json::from_value(encoded).unwrap();
        assert_eq!(decoded, report);
    }

    #[test]
    fn legacy_device_report_maps_without_losing_identity() {
        let report = MemoryReport::from(ExternalDeviceMemoryReport::single("weights", 4, 99));
        assert_eq!(
            report,
            MemoryReport::single(
                "weights",
                MemoryDomain::Cuda { device_id: 4 },
                MemoryAllocationClass::ExternallyOwned,
                99,
            )
        );
    }

    #[test]
    fn legacy_serialized_allocation_defaults_to_backend_managed() {
        let allocation: MemoryAllocation = serde_json::from_value(serde_json::json!({
            "allocation_id": "legacy",
            "domain": { "kind": "host" },
            "class": "block-table",
            "bytes": 17
        }))
        .unwrap();
        assert_eq!(allocation.source, MemoryAllocationSource::BackendManaged);
        assert_eq!(allocation.class, MemoryAllocationClass::BlockTable);
    }
}
