use async_trait::async_trait;
use kapsl_scheduler::ReplicaScheduler;
use kapsl_transport::{TransportError, TransportServer};
use std::collections::HashMap;
use std::sync::Arc;
use thiserror::Error;

/// Dynamic lookup used by every server transport to resolve a model scheduler.
pub type SchedulerLookup =
    Arc<dyn Fn(u32) -> Option<Arc<dyn ReplicaScheduler + Send + Sync>> + Send + Sync>;

/// Snapshot used by the shared-memory server when constructing per-model pools.
pub type SchedulerSnapshot =
    Arc<dyn Fn() -> HashMap<u32, Arc<dyn ReplicaScheduler + Send + Sync>> + Send + Sync>;

#[derive(Debug, Error)]
pub enum CommunicationError {
    /// Shared-memory mapping creation failed.
    #[cfg(feature = "shm")]
    #[error(transparent)]
    SharedMemory(#[from] kapsl_shm::memory::ShmError),

    /// The selected transport cannot be built from the supplied settings.
    #[error("Invalid communication configuration: {0}")]
    InvalidConfiguration(String),
}

/// A constructed communication server and the endpoint advertised to clients.
#[derive(Clone)]
pub struct CommunicationServer {
    inner: Arc<dyn TransportServer>,
    endpoint: String,
}

impl CommunicationServer {
    /// Construct a local Unix-socket or Windows named-pipe server.
    #[cfg(feature = "ipc")]
    pub fn ipc(socket_path: &str, scheduler_lookup: SchedulerLookup) -> Self {
        Self {
            inner: Arc::new(kapsl_ipc::IpcServer::new_socket_with_lookup(
                socket_path,
                scheduler_lookup,
            )),
            endpoint: socket_path.to_string(),
        }
    }

    /// Construct an authenticated or loopback-only TCP server.
    #[cfg(feature = "ipc")]
    pub fn tcp(
        bind_addr: &str,
        port: u16,
        scheduler_lookup: SchedulerLookup,
        auth_token: Option<&str>,
    ) -> Self {
        let server = kapsl_ipc::TcpServer::new_with_lookup(bind_addr, port, scheduler_lookup);
        let server = match auth_token.map(str::trim).filter(|token| !token.is_empty()) {
            Some(token) => server.with_auth_token(token.to_string()),
            None => server,
        };
        Self {
            inner: Arc::new(server),
            endpoint: format!("{bind_addr}:{port}"),
        }
    }

    /// Construct a dedicated shared-memory request/response server.
    #[cfg(feature = "shm")]
    pub fn shared_memory(
        name: &str,
        size: usize,
        scheduler_lookup: SchedulerLookup,
        scheduler_snapshot: SchedulerSnapshot,
        metrics_registry: Option<Arc<prometheus::Registry>>,
    ) -> Self {
        Self {
            inner: Arc::new(kapsl_shm::ShmServer::new_with_lookup_and_registry(
                name,
                size,
                scheduler_lookup,
                scheduler_snapshot,
                metrics_registry,
            )),
            endpoint: name.to_string(),
        }
    }

    /// Construct socket control-plane transport backed by shared-memory tensors.
    #[cfg(feature = "hybrid")]
    pub fn hybrid(
        socket_path: &str,
        shm_name: &str,
        shm_size: usize,
        scheduler_lookup: SchedulerLookup,
    ) -> Result<Self, CommunicationError> {
        let shm = Arc::new(kapsl_shm::ShmManager::create(shm_name, shm_size)?);
        let server =
            kapsl_ipc::IpcServer::new_with_lookup(socket_path, scheduler_lookup, Some(shm));
        Ok(Self {
            inner: Arc::new(server),
            endpoint: format!("{socket_path} (shm: {shm_name})"),
        })
    }

    /// Return the endpoint string that should be advertised to clients.
    pub fn endpoint(&self) -> &str {
        &self.endpoint
    }

    /// Clone the type-erased server for storage in an application runtime.
    pub fn transport(&self) -> Arc<dyn TransportServer> {
        self.inner.clone()
    }
}

#[async_trait]
impl TransportServer for CommunicationServer {
    async fn run(&self) -> Result<(), TransportError> {
        self.inner.run().await
    }

    async fn shutdown(&self) -> Result<(), TransportError> {
        self.inner.shutdown().await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lookup() -> SchedulerLookup {
        Arc::new(|_| None)
    }

    #[cfg(feature = "ipc")]
    #[test]
    fn constructs_ipc_and_tcp_endpoints() {
        let ipc = CommunicationServer::ipc("/tmp/kapsl-test.sock", lookup());
        assert_eq!(ipc.endpoint(), "/tmp/kapsl-test.sock");

        let tcp = CommunicationServer::tcp("127.0.0.1", 9195, lookup(), None);
        assert_eq!(tcp.endpoint(), "127.0.0.1:9195");
    }

    #[cfg(feature = "shm")]
    #[test]
    fn constructs_shm_endpoint_without_allocating_until_run() {
        let snapshot: SchedulerSnapshot = Arc::new(HashMap::new);
        let server = CommunicationServer::shared_memory(
            "/kapsl-test-shm",
            1024 * 1024,
            lookup(),
            snapshot,
            None,
        );
        assert_eq!(server.endpoint(), "/kapsl-test-shm");
    }
}
