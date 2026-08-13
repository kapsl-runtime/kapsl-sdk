use crate::connection_pool::{ConnectionFactory, ConnectionPool, PoolConfig};
use crate::{TransportClient, TransportError};
use async_trait::async_trait;
use kapsl_engine_api::{BinaryTensorPacket, InferenceRequest};
use tokio::net::TcpStream;

/// Factory for creating TCP connections
pub struct TcpConnectionFactory {
    host: String,
    port: u16,
}

impl TcpConnectionFactory {
    pub fn new(host: String, port: u16) -> Self {
        Self { host, port }
    }
}

#[async_trait]
impl ConnectionFactory for TcpConnectionFactory {
    type Connection = TcpStream;
    type Error = std::io::Error;

    async fn connect(&self) -> Result<Self::Connection, Self::Error> {
        let addr = format!("{}:{}", self.host, self.port);
        TcpStream::connect(addr).await
    }

    async fn is_valid(&self, _conn: &Self::Connection) -> bool {
        // Simple check: we can't easily check if a TCP stream is closed without reading/writing.
        // For now, assume it's valid if we have it.
        // A more robust check would be to peek, but TcpStream doesn't support peek easily.
        true
    }
}

/// TCP Client implementation using connection pooling
pub struct TcpClient {
    pool: ConnectionPool<TcpStream, TcpConnectionFactory>,
}

impl TcpClient {
    pub fn new(host: String, port: u16, pool_config: PoolConfig) -> Self {
        let factory = TcpConnectionFactory::new(host, port);
        let pool = ConnectionPool::new(pool_config, factory);
        Self { pool }
    }

    /// Send a complete request, including authentication and generation
    /// metadata required by authenticated TCP servers.
    pub async fn infer_request(
        &self,
        model_id: u32,
        request: &InferenceRequest,
    ) -> Result<BinaryTensorPacket, TransportError> {
        let mut conn = self
            .pool
            .get()
            .await
            .map_err(|error| TransportError::Connection(error.to_string()))?;
        crate::protocol::asynchronous::infer_request_over_stream(&mut *conn, model_id, request)
            .await
            .map_err(TransportError::from)
    }
}

#[async_trait]
impl TransportClient for TcpClient {
    async fn infer(
        &self,
        model_id: u32,
        input: BinaryTensorPacket,
    ) -> Result<BinaryTensorPacket, TransportError> {
        let request = InferenceRequest::new(input);
        TcpClient::infer_request(self, model_id, &request).await
    }

    async fn infer_request(
        &self,
        model_id: u32,
        request: &InferenceRequest,
    ) -> Result<BinaryTensorPacket, TransportError> {
        TcpClient::infer_request(self, model_id, request).await
    }
}
