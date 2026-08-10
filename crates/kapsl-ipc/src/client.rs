use async_trait::async_trait;
use kapsl_engine_api::BinaryTensorPacket;
use kapsl_transport::connection_pool::{ConnectionFactory, ConnectionPool, PoolConfig};
use kapsl_transport::protocol::infer_over_stream;
use kapsl_transport::{TransportClient, TransportError};

#[cfg(unix)]
use tokio::net::UnixStream;

#[cfg(windows)]
use tokio::net::windows::named_pipe::{ClientOptions, NamedPipeClient};

// Platform-specific connection type alias
#[cfg(unix)]
type IpcConnection = UnixStream;

#[cfg(windows)]
type IpcConnection = NamedPipeClient;

/// Factory for creating IPC connections (Unix Domain Sockets on Unix, Named Pipes on Windows)
pub struct IpcConnectionFactory {
    socket_path: String,
}

impl IpcConnectionFactory {
    pub fn new(socket_path: String) -> Self {
        Self { socket_path }
    }
}

#[async_trait]
impl ConnectionFactory for IpcConnectionFactory {
    #[cfg(unix)]
    type Connection = UnixStream;
    #[cfg(windows)]
    type Connection = NamedPipeClient;
    type Error = std::io::Error;

    async fn connect(&self) -> Result<Self::Connection, Self::Error> {
        #[cfg(unix)]
        return UnixStream::connect(&self.socket_path).await;

        #[cfg(windows)]
        return ClientOptions::new().open(&self.socket_path);
    }

    async fn is_valid(&self, _conn: &Self::Connection) -> bool {
        // Similar to TCP, hard to check validity without I/O.
        true
    }
}

/// IPC Client implementation using connection pooling
pub struct IpcClient {
    pool: ConnectionPool<IpcConnection, IpcConnectionFactory>,
}

impl IpcClient {
    pub fn new(socket_path: String, pool_config: PoolConfig) -> Self {
        let factory = IpcConnectionFactory::new(socket_path);
        let pool = ConnectionPool::new(pool_config, factory);
        Self { pool }
    }
}

#[async_trait]
impl TransportClient for IpcClient {
    async fn infer(
        &self,
        model_id: u32,
        input: BinaryTensorPacket,
    ) -> Result<BinaryTensorPacket, TransportError> {
        let mut conn = self
            .pool
            .get()
            .await
            .map_err(|e| TransportError::Connection(e.to_string()))?;
        infer_over_stream(&mut *conn, model_id, input).await
    }
}
