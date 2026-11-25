//! The contract between request transports and the schedulers that serve them.
//!
//! Transports (IPC, shared memory, gRPC) only need to hand a request to a
//! scheduler at some priority. Keeping that contract here lets them depend on
//! this crate alone instead of on `kapsl-scheduler` and everything it builds.

use crate::{
    BinaryTensorPacket, EngineError, EngineMetrics, EngineModelInfo, InferenceRequest,
    OpenAiWireRequest, OpenAiWireResponse, OpenAiWireStreamResponse,
};
use futures::Stream;
use std::pin::Pin;

/// How urgently a request should be served.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Priority {
    LatencyCritical,
    Throughput,
}

impl Priority {
    /// The value carried in request metadata: 0 = latency-critical, 1 = throughput.
    pub fn engine_metadata_value(self) -> u8 {
        match self {
            Self::LatencyCritical => 0,
            Self::Throughput => 1,
        }
    }
}

/// Trait that replica schedulers must implement
#[async_trait::async_trait]
pub trait ReplicaScheduler: Send + Sync {
    fn get_queue_depth(&self) -> (usize, usize);
    fn is_healthy(&self) -> bool;
    fn get_metrics(&self) -> EngineMetrics;
    fn model_info(&self) -> Option<EngineModelInfo> {
        None
    }
    async fn infer(
        &self,
        request: &InferenceRequest,
        priority: Priority,
        force_cpu: bool,
    ) -> Result<BinaryTensorPacket, EngineError>;

    async fn infer_openai_wire(
        &self,
        _request: OpenAiWireRequest,
        _priority: Priority,
        _force_cpu: bool,
    ) -> Result<OpenAiWireResponse, EngineError> {
        Err(EngineError::backend(
            "scheduler does not support protocol-native OpenAI requests",
        ))
    }

    async fn infer_openai_wire_stream(
        &self,
        _request: OpenAiWireRequest,
        _priority: Priority,
        _force_cpu: bool,
    ) -> Result<OpenAiWireStreamResponse, EngineError> {
        Err(EngineError::backend(
            "scheduler does not support protocol-native OpenAI streams",
        ))
    }

    async fn infer_stream(
        &self,
        request: InferenceRequest,
        priority: Priority,
        force_cpu: bool,
    ) -> Result<
        Pin<Box<dyn Stream<Item = Result<BinaryTensorPacket, EngineError>> + Send>>,
        EngineError,
    >;
}
