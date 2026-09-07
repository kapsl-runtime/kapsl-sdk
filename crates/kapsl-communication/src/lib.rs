//! Unified communication facade for Kapsl clients and runtime servers.
//!
//! The implementation crates remain independently usable. This facade offers
//! one dependency and one namespace while retaining feature-gated build
//! boundaries for local IPC and shared-memory transports.

/// Transport-neutral protocol, framing, client, and server interfaces.
pub mod transport {
    pub use kapsl_transport::*;
}

/// Open Inference Protocol clients and Kapsl server-streaming gRPC transport.
#[cfg(feature = "grpc")]
pub mod grpc {
    pub use kapsl_grpc::*;
}

/// Local socket/named-pipe and authenticated TCP transports.
#[cfg(feature = "ipc")]
pub mod ipc {
    pub use kapsl_ipc::*;
}

/// Shared-memory managers, allocators, queues, and server transport.
#[cfg(feature = "shm")]
pub mod shm {
    pub use kapsl_shm::*;
}

/// Types used by the socket-control/shared-memory-data hybrid transport.
#[cfg(feature = "hybrid")]
pub mod hybrid {
    pub use kapsl_ipc::{HybridMemory, HybridRequest, HybridResponse, HybridTensorLocation};
    pub use kapsl_shm::memory::{ShmManager, TensorHeader};
}

pub use kapsl_transport::{
    RequestMetadata, ResponseMetadata, TransportClient, TransportError, TransportServer,
};

#[cfg(any(feature = "ipc", feature = "shm"))]
mod server;

#[cfg(any(feature = "ipc", feature = "shm"))]
pub use server::{CommunicationError, CommunicationServer, SchedulerLookup, SchedulerSnapshot};
