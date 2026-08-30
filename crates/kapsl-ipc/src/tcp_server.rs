use async_trait::async_trait;
use kapsl_scheduler::ReplicaScheduler;
use kapsl_transport::{TransportError, TransportServer};
use std::collections::HashMap;
use std::net::IpAddr;
use std::sync::Arc;
use tokio::net::TcpListener;

/// TCP-based server for network communication across different computers
pub struct TcpServer {
    bind_addr: String,
    port: u16,
    scheduler_lookup: crate::server::SchedulerLookup,
    auth_token: Option<Arc<str>>,
}

impl TcpServer {
    pub fn new(
        bind_addr: &str,
        port: u16,
        schedulers: HashMap<u32, Arc<dyn ReplicaScheduler + Send + Sync>>,
    ) -> Self {
        let schedulers = Arc::new(schedulers);
        let scheduler_lookup: crate::server::SchedulerLookup =
            Arc::new(move |model_id| schedulers.get(&model_id).cloned());
        Self::new_with_lookup(bind_addr, port, scheduler_lookup)
    }

    pub fn new_with_lookup(
        bind_addr: &str,
        port: u16,
        scheduler_lookup: crate::server::SchedulerLookup,
    ) -> Self {
        Self {
            bind_addr: bind_addr.to_string(),
            port,
            scheduler_lookup,
            auth_token: None,
        }
    }

    /// Require an inference metadata token for every request handled by this
    /// server. A non-loopback listener is rejected unless this is configured.
    /// Protocol-native OpenAI operations remain disabled on non-loopback
    /// plaintext TCP even when native tensor inference is authenticated.
    pub fn with_auth_token(mut self, auth_token: impl Into<String>) -> Self {
        let auth_token = auth_token.into();
        if !auth_token.is_empty() {
            self.auth_token = Some(Arc::from(auth_token));
        }
        self
    }

    async fn run_internal(&self) -> std::io::Result<()> {
        let addr = format!("{}:{}", self.bind_addr, self.port);
        let listener = TcpListener::bind(&addr).await?;
        let bind_ip = listener.local_addr()?.ip();
        validate_tcp_exposure(bind_ip, self.auth_token.is_some())?;
        let wire_policy = openai_wire_policy(bind_ip);
        let scheduler_lookup = self.scheduler_lookup.clone();
        let auth_token = self.auth_token.clone();

        log::info!("TCP Server listening on {}", addr);
        log::info!("TCP Server bound to {}", addr);
        if wire_policy == crate::server::OpenAiWireTransportPolicy::PlaintextRemote {
            log::warn!(
                "Protocol-native OpenAI operations are disabled on non-loopback plaintext TCP listener {}",
                addr
            );
        }

        loop {
            let (stream, peer_addr) = listener.accept().await?;
            let scheduler_lookup = scheduler_lookup.clone();
            let auth_token = auth_token.clone();

            log::info!("New TCP connection from {}", peer_addr);

            tokio::spawn(async move {
                if let Err(e) = crate::server::handle_connection_with_wire_policy(
                    stream,
                    scheduler_lookup,
                    None,
                    auth_token,
                    wire_policy,
                )
                .await
                {
                    log::error!("Connection error: {}", e);
                }
                log::info!("TCP connection closed from {}", peer_addr);
            });
        }
    }
}

fn openai_wire_policy(bind_ip: IpAddr) -> crate::server::OpenAiWireTransportPolicy {
    if bind_ip.is_loopback() {
        crate::server::OpenAiWireTransportPolicy::Local
    } else {
        crate::server::OpenAiWireTransportPolicy::PlaintextRemote
    }
}

fn validate_tcp_exposure(bind_ip: IpAddr, authenticated: bool) -> std::io::Result<()> {
    if bind_ip.is_loopback() || authenticated {
        return Ok(());
    }

    Err(std::io::Error::new(
        std::io::ErrorKind::PermissionDenied,
        format!(
            "refusing unauthenticated TCP inference listener on non-loopback address {bind_ip}; configure a native-transport authentication token or bind to loopback"
        ),
    ))
}

#[async_trait]
impl TransportServer for TcpServer {
    async fn run(&self) -> Result<(), TransportError> {
        self.run_internal().await.map_err(TransportError::Io)
    }

    async fn shutdown(&self) -> Result<(), TransportError> {
        // TCP listeners don't need explicit cleanup
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{openai_wire_policy, validate_tcp_exposure};
    use crate::server::OpenAiWireTransportPolicy;

    #[test]
    fn unauthenticated_tcp_is_limited_to_loopback() {
        assert!(validate_tcp_exposure("127.0.0.1".parse().unwrap(), false).is_ok());
        assert!(validate_tcp_exposure("::1".parse().unwrap(), false).is_ok());
        assert!(validate_tcp_exposure("0.0.0.0".parse().unwrap(), false).is_err());
        assert!(validate_tcp_exposure("192.0.2.10".parse().unwrap(), false).is_err());
    }

    #[test]
    fn authenticated_tcp_may_bind_non_loopback() {
        assert!(validate_tcp_exposure("0.0.0.0".parse().unwrap(), true).is_ok());
    }

    #[test]
    fn openai_wire_is_limited_to_loopback_tcp() {
        assert_eq!(
            openai_wire_policy("127.0.0.1".parse().unwrap()),
            OpenAiWireTransportPolicy::Local
        );
        assert_eq!(
            openai_wire_policy("::1".parse().unwrap()),
            OpenAiWireTransportPolicy::Local
        );
        assert_eq!(
            openai_wire_policy("0.0.0.0".parse().unwrap()),
            OpenAiWireTransportPolicy::PlaintextRemote
        );
        assert_eq!(
            openai_wire_policy("192.0.2.10".parse().unwrap()),
            OpenAiWireTransportPolicy::PlaintextRemote
        );
    }
}
