//! Kapsl's request transports and the server that runs them.
//!
//! - `transport`: transport-neutral framing, protocol, client and server
//!   interfaces, TCP, connection pooling and Windows named pipes.
//! - `ipc` (feature `ipc`): local socket/named-pipe and authenticated TCP
//!   servers and clients.
//! - `shm` (feature `shm`): shared-memory managers, allocators, queues and the
//!   shared-memory server.
//! - `hybrid` (feature `hybrid`, the default): types for the transport that
//!   pairs a socket control plane with a shared-memory data plane.
//! - `grpc` (feature `grpc`): Open Inference Protocol clients and the gRPC
//!   server, re-exported from `kapsl-grpc`.
//!
//! `transport`, `ipc` and `shm` were the `kapsl-transport`, `kapsl-ipc` and
//! `kapsl-shm` crates. Their paths here are the ones this crate already
//! re-exported them under.

/// Transport-neutral protocol, framing, client, and server interfaces.
pub mod transport;

/// Open Inference Protocol clients and Kapsl server-streaming gRPC transport.
#[cfg(feature = "grpc")]
pub mod grpc {
    pub use kapsl_grpc::*;
}

#[cfg(feature = "ipc")]
pub mod ipc;

#[cfg(feature = "shm")]
pub mod shm;

/// Types used by the socket-control/shared-memory-data hybrid transport.
#[cfg(feature = "hybrid")]
pub mod hybrid {
    pub use crate::ipc::{HybridMemory, HybridRequest, HybridResponse, HybridTensorLocation};
    pub use crate::shm::memory::{ShmManager, TensorHeader};
}

pub use transport::{
    RequestMetadata, ResponseMetadata, TransportClient, TransportError, TransportServer,
};

#[cfg(any(feature = "ipc", feature = "shm"))]
mod server;

#[cfg(any(feature = "ipc", feature = "shm"))]
pub use server::{CommunicationError, CommunicationServer, SchedulerLookup, SchedulerSnapshot};
