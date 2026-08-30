//! Engine-side implementation for Kapsl retrieval and connector hosting.
//!
//! Connector-facing protocol types remain in `kapsl-rag-sdk`; this crate owns
//! persistence, ingestion, extension installation and WASM/sidecar execution.

pub mod extension;
pub mod ingestion;
pub mod runtime;
pub mod storage;
pub mod vector;

mod validation;

/// Connector-facing manifest, protocol, and authoring API.
///
/// Re-exporting the lightweight SDK makes the host/connector boundary explicit
/// without forcing existing `kapsl-rag-sdk` consumers through a breaking crate
/// rename.
pub use kapsl_rag_sdk as connector_sdk;

pub use extension::{
    ConnectorRuntimeHandle, ExtensionManager, ExtensionRegistry, InstalledExtension,
};
pub use ingestion::{Chunker, DocumentParser, Embedder, IngestionPipeline, ParsedDocument};
pub use runtime::{ConnectorClient, ConnectorRuntime, PreopenDir, WasiPermissions};
pub use storage::{DocStore, FsDocStore};
pub use vector::{
    AccessControl, EmbeddedChunk, VectorQuery, VectorSearchResult, VectorStore, VectorStoreError,
};
