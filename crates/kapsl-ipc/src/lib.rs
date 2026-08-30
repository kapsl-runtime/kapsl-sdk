//! Local IPC and authenticated TCP transports for Kapsl inference.

#[path = "client/connection.rs"]
pub mod client;
#[path = "protocol/wire.rs"]
pub mod protocol;
#[path = "server/ipc.rs"]
pub mod server;
#[path = "server/tcp.rs"]
pub mod tcp_server;

pub use protocol::*;
pub use server::IpcServer;
pub use tcp_server::TcpServer;
