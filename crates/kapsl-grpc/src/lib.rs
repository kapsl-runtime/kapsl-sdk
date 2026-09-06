//! gRPC transport and protocol conversion, independent of any model backend.
//! Runtime authorization, model discovery, scheduling, and memory admission are
//! supplied by the engine through the two facade traits.

mod codec;
mod lifecycle;
mod service;

use std::{net::IpAddr, pin::Pin, sync::Arc};

use async_trait::async_trait;
use futures::Stream;
use kapsl_engine_api::{BinaryTensorPacket, EngineError, EngineModelInfo, InferenceRequest};
pub use service::{start_server, GrpcServerConfig, GrpcServerHandle};
pub use tonic;
use tonic::Status;

pub mod inference {
    tonic::include_proto!("inference");
}

pub mod kapsl {
    pub mod v1 {
        tonic::include_proto!("kapsl.v1");
    }
}

pub type EngineStream = Pin<Box<dyn Stream<Item = Result<BinaryTensorPacket, EngineError>> + Send>>;

#[derive(Clone, Debug)]
pub struct Model {
    pub id: u32,
    pub name: String,
    pub version: String,
    pub ready: bool,
    pub info: Option<EngineModelInfo>,
}

#[async_trait]
pub trait EngineFacade: Send + Sync + 'static {
    /// Return logical models, excluding internal replicas.
    fn models(&self) -> Vec<Model>;
    async fn infer(
        &self,
        model_id: u32,
        request: InferenceRequest,
    ) -> Result<BinaryTensorPacket, EngineError>;
    async fn infer_stream(
        &self,
        model_id: u32,
        request: InferenceRequest,
    ) -> Result<EngineStream, EngineError>;
}

pub trait RequestAuthorizer: Send + Sync + 'static {
    /// Called for every RPC, including discovery and health. The peer address
    /// comes from the accepted TCP connection, never forwarded metadata.
    fn authorize_reader(
        &self,
        authorization: Option<&str>,
        remote_ip: Option<IpAddr>,
    ) -> Result<(), Status>;

    /// Use the same credential namespace as the engine's other API adapters.
    /// Called only after authorize_reader succeeds.
    fn scope_session_id(
        &self,
        session_id: Option<&str>,
        authorization: Option<&str>,
    ) -> Option<String>;
}

#[derive(Clone)]
struct Dependencies {
    engine: Arc<dyn EngineFacade>,
    authorizer: Arc<dyn RequestAuthorizer>,
}

fn engine_status(error: EngineError) -> Status {
    match error {
        EngineError::InvalidInput { .. } => Status::invalid_argument(error.to_string()),
        EngineError::ModelNotLoaded => Status::not_found("Model not loaded"),
        EngineError::Overloaded { .. } | EngineError::ResourceExhausted { .. } => {
            Status::resource_exhausted(error.to_string())
        }
        EngineError::TimeoutError { .. } => Status::deadline_exceeded(error.to_string()),
        EngineError::Cancelled { .. } => Status::cancelled(error.to_string()),
        // Backend messages can contain paths, prompts, or credentials.
        _ => Status::internal("Inference backend failed"),
    }
}
