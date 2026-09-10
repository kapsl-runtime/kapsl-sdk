//! Versioned process protocol; see the packaged README for normative lifecycle,
//! cancellation, admission, streaming and transport requirements.
//!
//! This crate contains wire values and bounded framing only. It neither starts
//! processes nor discovers Python, constructs backend arguments or implements
//! inference. KV participation uses the separate `kapsl-kv-abi` contract.

use std::collections::{BTreeMap, BTreeSet};
use std::io::{self, BufRead, Read, Write};

use kapsl_engine_api::{
    BinaryTensorPacket, EngineMetrics, EngineModelInfo, InferenceRequest, KvBackendCapabilities,
    KvTopology, MemoryReport, OpenAiWireRequest, OpenAiWireResponse, OpenAiWireResponseHead,
};
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use serde_json::Value;

pub const MANAGED_BACKEND_PROTOCOL_VERSION: u32 = 1;
pub const MANAGED_BACKEND_PROTOCOL_ID: &str = "kapsl-managed-v1";
/// Includes the terminating newline. The control reader never grows without
/// a bound while waiting for a delimiter. KV contents stay in the separately
/// negotiated shared-memory/CUDA IPC data plane.
pub const MAX_FRAME_BYTES: usize = 8 * 1024 * 1024;

mod validation;
pub use validation::StreamProgress;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InstanceIdentity {
    /// Engine-generated nonce, unique across process starts and reloads.
    pub instance_id: String,
    pub model_id: u32,
    pub replica_id: u32,
}

impl InstanceIdentity {
    pub fn validate(&self) -> Result<(), String> {
        if self.instance_id.is_empty()
            || self.instance_id.len() > 128
            || !self
                .instance_id
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_')
        {
            return Err(
                "instance_id must be a nonempty ASCII identifier of at most 128 bytes".into(),
            );
        }
        Ok(())
    }
}

/// Every command, response and stream event repeats the engine-bound owner and
/// process nonce. `id` is a nonzero, never-reused command correlation ID.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Envelope<T> {
    pub protocol_version: u32,
    pub owner: InstanceIdentity,
    pub id: u64,
    pub body: T,
}

impl<T> Envelope<T> {
    pub fn new(owner: InstanceIdentity, id: u64, body: T) -> Self {
        Self {
            protocol_version: MANAGED_BACKEND_PROTOCOL_VERSION,
            owner,
            id,
            body,
        }
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.protocol_version != MANAGED_BACKEND_PROTOCOL_VERSION {
            return Err(format!(
                "unsupported managed backend protocol version {}",
                self.protocol_version
            ));
        }
        if self.id == 0 {
            return Err("managed command IDs must be nonzero".into());
        }
        self.owner.validate()
    }

    pub fn validate_for(&self, owner: &InstanceIdentity, id: u64) -> Result<(), String> {
        self.validate()?;
        if self.owner != *owner || self.id != id {
            return Err(
                "managed message belongs to another process, model, replica or command".into(),
            );
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Method {
    Describe,
    Initialize,
    PlannedMemory,
    Load,
    PlannedRequestMemory,
    Infer,
    InferBatch,
    InferStream,
    Cancel,
    ActualMemory,
    Metrics,
    ModelInfo,
    BatchingPolicy,
    KvCapabilities,
    KvTopology,
    Health,
    Unload,
    Shutdown,
}

#[derive(Debug, Default, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Capabilities {
    pub batching: bool,
    pub streaming: bool,
    pub cancellation: bool,
    pub memory_reporting: bool,
    pub concurrent_inference: bool,
    pub openai_wire: bool,
    pub kv_participation: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Descriptor {
    pub schema_version: u32,
    pub protocol_version: u32,
    pub execution_mode: String,
    pub backend: String,
    pub profile: String,
    pub pack_version: String,
    pub formats: Vec<String>,
    pub model_types: Vec<String>,
    pub tasks: Vec<String>,
    pub capabilities: Capabilities,
    pub methods: BTreeSet<Method>,
}

impl Descriptor {
    pub fn validate(&self) -> Result<(), String> {
        if self.schema_version != 1
            || self.protocol_version != MANAGED_BACKEND_PROTOCOL_VERSION
            || self.execution_mode != "managed"
        {
            return Err(
                "incompatible managed descriptor schema, protocol or execution mode".into(),
            );
        }
        for value in [&self.backend, &self.profile, &self.pack_version] {
            if value.trim().is_empty() {
                return Err("managed descriptor identity must not be empty".into());
            }
        }
        for (label, values, required) in [
            ("formats", &self.formats, true),
            ("model_types", &self.model_types, false),
            ("tasks", &self.tasks, true),
        ] {
            let normalized: BTreeSet<_> = values
                .iter()
                .map(|s| s.trim().to_ascii_lowercase())
                .collect();
            if (required && values.is_empty())
                || normalized.contains("")
                || normalized.len() != values.len()
            {
                return Err(format!(
                    "managed descriptor {label} contains empty or duplicate values"
                ));
            }
        }
        let required = [
            Method::Describe,
            Method::Initialize,
            Method::PlannedMemory,
            Method::Load,
            Method::PlannedRequestMemory,
            Method::Infer,
            Method::ActualMemory,
            Method::Metrics,
            Method::ModelInfo,
            Method::BatchingPolicy,
            Method::Health,
            Method::Unload,
            Method::Shutdown,
        ];
        if !self.capabilities.memory_reporting || required.iter().any(|m| !self.methods.contains(m))
        {
            return Err(
                "managed descriptor is missing required lifecycle, inference or memory methods"
                    .into(),
            );
        }
        for (enabled, method) in [
            (self.capabilities.batching, Method::InferBatch),
            (self.capabilities.streaming, Method::InferStream),
            (self.capabilities.cancellation, Method::Cancel),
            (self.capabilities.kv_participation, Method::KvCapabilities),
            (self.capabilities.kv_participation, Method::KvTopology),
        ] {
            if enabled != self.methods.contains(&method) {
                return Err(format!(
                    "managed capability and {method:?} declaration disagree"
                ));
            }
        }
        if self.capabilities.streaming && !self.capabilities.cancellation {
            return Err("managed streaming requires cancellation".into());
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DeviceAssignment {
    /// Canonical hardware kind, such as `cpu` or `cuda`; no provider CLI syntax.
    pub kind: String,
    pub device_id: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Initialize {
    pub backend: String,
    pub profile: String,
    pub pack_version: String,
    pub devices: Vec<DeviceAssignment>,
    pub manifest: Value,
    pub options: BTreeMap<String, Value>,
    /// Opaque endpoint configuration for the separately negotiated KV ABI.
    /// This does not authorize inference, launch or changes to memory grants.
    pub kv_connection: Option<Value>,
}

/// An engine-issued reservation, bound to this envelope's instance and owner.
/// A model grant has no request IDs; a request/batch grant lists exactly the
/// IDs in its dispatch. Adapters may not mint, enlarge or transfer grants.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Admission {
    pub grant_id: String,
    pub request_ids: Vec<u64>,
    pub limits: MemoryReport,
}

impl Admission {
    pub fn validate_for(&self, request_ids: &[u64]) -> Result<(), String> {
        let requested: BTreeSet<_> = request_ids.iter().copied().collect();
        let admitted: BTreeSet<_> = self.request_ids.iter().copied().collect();
        if self.grant_id.trim().is_empty()
            || requested.contains(&0)
            || admitted.contains(&0)
            || requested.len() != request_ids.len()
            || admitted.len() != self.request_ids.len()
            || requested != admitted
        {
            return Err(
                "managed memory admission has missing, duplicate or foreign ownership".into(),
            );
        }
        validate_memory_report(&self.limits)
    }
}

fn validate_memory_report(report: &MemoryReport) -> Result<(), String> {
    let mut allocations = std::collections::HashSet::new();
    let mut total = 0usize;
    for row in &report.allocations {
        if row.allocation_id.trim().is_empty()
            || !allocations.insert((&row.domain, row.allocation_id.as_str()))
        {
            return Err("managed memory report contains invalid allocation identities".into());
        }
        total = total
            .checked_add(row.bytes)
            .ok_or("managed memory report byte count overflow")?;
    }
    Ok(())
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(
    tag = "kind",
    content = "request",
    rename_all = "snake_case",
    deny_unknown_fields
)]
pub enum Input {
    Tensors(InferenceRequest),
    OpenaiWire(OpenAiWireRequest),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Request {
    pub request_id: u64,
    pub input: Input,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "method", rename_all = "snake_case", deny_unknown_fields)]
pub enum Command {
    Describe,
    Initialize {
        config: Initialize,
    },
    PlannedMemory {
        model_path: String,
    },
    Load {
        model_path: String,
        admission: Admission,
    },
    PlannedRequestMemory {
        request: Request,
    },
    Infer {
        request: Request,
        admission: Admission,
    },
    InferBatch {
        requests: Vec<Request>,
        admission: Admission,
    },
    InferStream {
        request: Request,
        admission: Admission,
    },
    Cancel {
        request_ids: Vec<u64>,
    },
    ActualMemory,
    Metrics,
    ModelInfo,
    BatchingPolicy,
    KvCapabilities,
    KvTopology,
    Health,
    Unload,
    Shutdown,
}

impl Command {
    pub fn method(&self) -> Method {
        match self {
            Self::Describe => Method::Describe,
            Self::Initialize { .. } => Method::Initialize,
            Self::PlannedMemory { .. } => Method::PlannedMemory,
            Self::Load { .. } => Method::Load,
            Self::PlannedRequestMemory { .. } => Method::PlannedRequestMemory,
            Self::Infer { .. } => Method::Infer,
            Self::InferBatch { .. } => Method::InferBatch,
            Self::InferStream { .. } => Method::InferStream,
            Self::Cancel { .. } => Method::Cancel,
            Self::ActualMemory => Method::ActualMemory,
            Self::Metrics => Method::Metrics,
            Self::ModelInfo => Method::ModelInfo,
            Self::BatchingPolicy => Method::BatchingPolicy,
            Self::KvCapabilities => Method::KvCapabilities,
            Self::KvTopology => Method::KvTopology,
            Self::Health => Method::Health,
            Self::Unload => Method::Unload,
            Self::Shutdown => Method::Shutdown,
        }
    }

    pub fn validate(&self) -> Result<(), String> {
        match self {
            Self::Initialize { config } => config.validate(),
            Self::Infer { request, admission } | Self::InferStream { request, admission } => {
                request.validate()?;
                admission.validate_for(&[request.request_id])
            }
            Self::InferBatch {
                requests,
                admission,
            } => {
                if requests.is_empty() {
                    return Err("managed inference batches must not be empty".into());
                }
                for request in requests {
                    request.validate()?;
                }
                admission.validate_for(&requests.iter().map(|r| r.request_id).collect::<Vec<_>>())
            }
            Self::Load {
                model_path,
                admission,
            } => {
                if model_path.is_empty() {
                    return Err("model path must not be empty".into());
                }
                admission.validate_for(&[])
            }
            Self::Cancel { request_ids } => {
                if request_ids.is_empty()
                    || request_ids.contains(&0)
                    || request_ids.iter().collect::<BTreeSet<_>>().len() != request_ids.len()
                {
                    return Err("cancellation requires distinct nonzero request IDs".into());
                }
                Ok(())
            }
            Self::PlannedRequestMemory { request } => request.validate(),
            Self::PlannedMemory { model_path } if model_path.is_empty() => {
                Err("model path must not be empty".into())
            }
            _ => Ok(()),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ErrorCode {
    InvalidArgument,
    IncompatibleProtocol,
    Unsupported,
    NotReady,
    ResourceExhausted,
    Cancelled,
    BackendError,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Failure {
    pub code: ErrorCode,
    pub message: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(
    tag = "kind",
    content = "value",
    rename_all = "snake_case",
    deny_unknown_fields
)]
pub enum Output {
    Tensor(BinaryTensorPacket),
    OpenaiWire(OpenAiWireResponse),
    OpenaiHead(OpenAiWireResponseHead),
    /// Raw SSE body bytes; public credentials and hop-by-hop headers never
    /// cross this boundary. The head precedes all body chunks.
    OpenaiChunk(Vec<u8>),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ResultItem {
    pub request_id: u64,
    pub output: Output,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BatchingMode {
    None,
    RequestCoalescing,
    Continuous,
    Delegated,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BatchingPolicy {
    pub mode: BatchingMode,
    pub max_requests: usize,
    pub queue_delay_ms: Option<u64>,
    pub max_batched_tokens: Option<usize>,
    pub supports_priority: bool,
}

impl From<kapsl_engine_api::BatchingPolicy> for BatchingPolicy {
    fn from(policy: kapsl_engine_api::BatchingPolicy) -> Self {
        use kapsl_engine_api::BatchingMode as Mode;
        Self {
            mode: match policy.mode {
                Mode::None => BatchingMode::None,
                Mode::RequestCoalescing => BatchingMode::RequestCoalescing,
                Mode::Continuous => BatchingMode::Continuous,
                Mode::Delegated => BatchingMode::Delegated,
            },
            max_requests: policy.max_requests,
            queue_delay_ms: policy.queue_delay_ms,
            max_batched_tokens: policy.max_batched_tokens,
            supports_priority: policy.supports_priority,
        }
    }
}

impl TryFrom<BatchingPolicy> for kapsl_engine_api::BatchingPolicy {
    type Error = String;
    fn try_from(policy: BatchingPolicy) -> Result<Self, Self::Error> {
        if policy.max_requests == 0 || policy.max_batched_tokens == Some(0) {
            return Err("managed batching limits must be positive".into());
        }
        use kapsl_engine_api::BatchingMode as Mode;
        Ok(Self {
            mode: match policy.mode {
                BatchingMode::None => Mode::None,
                BatchingMode::RequestCoalescing => Mode::RequestCoalescing,
                BatchingMode::Continuous => Mode::Continuous,
                BatchingMode::Delegated => Mode::Delegated,
            },
            max_requests: policy.max_requests,
            queue_delay_ms: policy.queue_delay_ms,
            max_batched_tokens: policy.max_batched_tokens,
            supports_priority: policy.supports_priority,
        })
    }
}

/// All events for a stream use the original command ID and request ID. Sequence
/// starts at zero, increases by one, and `StreamEnd.next_sequence` identifies
/// the next unused value. Exactly one terminal response is permitted.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum Response {
    Descriptor {
        descriptor: Descriptor,
    },
    Ack,
    Memory {
        report: MemoryReport,
    },
    Metrics {
        metrics: EngineMetrics,
    },
    ModelInfo {
        model_info: EngineModelInfo,
    },
    BatchingPolicy {
        policy: BatchingPolicy,
    },
    KvCapabilities {
        capabilities: KvBackendCapabilities,
    },
    KvTopology {
        topology: KvTopology,
    },
    Health {
        healthy: bool,
        details: String,
    },
    Result {
        result: ResultItem,
    },
    BatchResult {
        results: Vec<ResultItem>,
    },
    StreamChunk {
        request_id: u64,
        sequence: u64,
        output: Output,
    },
    StreamEnd {
        request_id: u64,
        next_sequence: u64,
        error: Option<Failure>,
    },
    Cancelled {
        request_ids: Vec<u64>,
    },
    Error {
        error: Failure,
    },
}

pub type HostMessage = Envelope<Command>;
pub type AdapterMessage = Envelope<Response>;

/// Read one newline-delimited frame with a strict allocation bound. EOF between
/// frames returns None; an incomplete or oversized frame is a protocol error.
pub fn read_frame<T: DeserializeOwned>(
    reader: &mut impl BufRead,
) -> io::Result<Option<Envelope<T>>> {
    let mut bytes = Vec::new();
    let count = reader
        .take((MAX_FRAME_BYTES + 1) as u64)
        .read_until(b'\n', &mut bytes)?;
    if count == 0 {
        return Ok(None);
    }
    if count > MAX_FRAME_BYTES || bytes.last() != Some(&b'\n') {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "oversized or incomplete managed frame",
        ));
    }
    let envelope: Envelope<T> = serde_json::from_slice(&bytes)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
    envelope
        .validate()
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
    Ok(Some(envelope))
}

/// The caller serializes access to stdout/stdin for the entire write; logs use
/// stderr. Invalid/oversized messages leave the stream untouched.
pub fn write_frame<T: Serialize>(
    writer: &mut impl Write,
    envelope: &Envelope<T>,
) -> io::Result<()> {
    envelope
        .validate()
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidInput, e))?;
    struct Bounded(Vec<u8>);
    impl Write for Bounded {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
            if bytes.len() >= MAX_FRAME_BYTES.saturating_sub(self.0.len()) {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "managed frame exceeds size limit",
                ));
            }
            self.0.extend_from_slice(bytes);
            Ok(bytes.len())
        }
        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }
    let mut bytes = Bounded(Vec::new());
    serde_json::to_writer(&mut bytes, envelope).map_err(io::Error::other)?;
    bytes.0.push(b'\n');
    writer.write_all(&bytes.0)?;
    writer.flush()
}

#[cfg(test)]
mod tests;
