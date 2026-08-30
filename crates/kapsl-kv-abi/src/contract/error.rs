//! Machine-readable KV contract errors.

use super::*;

#[derive(Debug, Clone, PartialEq, Eq, Error, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum KvContractError {
    #[error("KV ABI mismatch: host {host:?}, participant {participant:?}")]
    VersionMismatch {
        host: KvAbiVersion,
        participant: KvAbiVersion,
    },
    #[error("invalid KV capabilities: {message}")]
    InvalidCapabilities { message: String },
    #[error("invalid KV topology: {message}")]
    InvalidTopology { message: String },
    #[error("invalid KV request: {message}")]
    InvalidRequest { message: String },
    #[error("KV capacity exhausted: {message}")]
    CapacityExhausted { message: String },
    #[error("KV object not found: {message}")]
    NotFound { message: String },
    #[error("KV operation '{operation}' is unsupported")]
    Unsupported { operation: String },
    #[error("KV transport error: {message}")]
    Transport { message: String },
    #[error("KV participant error: {message}")]
    Internal { message: String },
}

impl KvContractError {
    pub fn invalid_capabilities(message: impl Into<String>) -> Self {
        Self::InvalidCapabilities {
            message: message.into(),
        }
    }

    pub fn invalid_topology(message: impl Into<String>) -> Self {
        Self::InvalidTopology {
            message: message.into(),
        }
    }

    pub fn invalid_request(message: impl Into<String>) -> Self {
        Self::InvalidRequest {
            message: message.into(),
        }
    }

    pub fn unsupported(operation: impl Into<String>) -> Self {
        Self::Unsupported {
            operation: operation.into(),
        }
    }
}
