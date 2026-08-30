//! Lightweight connector contract for Kapsl RAG extensions.
//!
//! This crate intentionally contains only the wire protocol, manifest and
//! document types, plus the connector-side stdio server. Engine-side storage,
//! ingestion and WASM/sidecar hosting live in the separate `kapsl-rag` crate so
//! connector authors do not inherit those runtime dependencies.

pub mod manifest;
pub mod protocol;
pub mod server;
pub mod types;

pub use manifest::{ConnectorAuthMethod, ConnectorCapability, ConnectorManifest, ConnectorRuntime};
pub use protocol::{
    ConnectorRequest, ConnectorRequestKind, ConnectorResponse, ConnectorResponseKind,
};
pub use server::{
    read_frame, serve_io, serve_stdio, Connector, ConnectorError, MAX_CONNECTOR_REQUEST_BYTES,
    MAX_CONNECTOR_RESPONSE_BYTES,
};
pub use types::{
    ConnectorConfig, DocumentDelta, DocumentPayload, ExternalAcl, PromptTransformResult,
    SourceDescriptor, SyncCursor, SyncRequest,
};
