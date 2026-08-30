use serde::{Deserialize, Serialize};

use crate::types::{
    ConnectorConfig, DocumentDelta, DocumentPayload, ExternalAcl, PromptTransformResult,
    SourceDescriptor,
};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConnectorRequest {
    pub id: String,
    #[serde(flatten)]
    pub kind: ConnectorRequestKind,
}

impl ConnectorRequest {
    /// Build a connector request with an explicit correlation id.
    pub fn new(id: impl Into<String>, kind: ConnectorRequestKind) -> Self {
        Self {
            id: id.into(),
            kind,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "method", content = "params")]
pub enum ConnectorRequestKind {
    ValidateConfig {
        config: ConnectorConfig,
    },
    AuthStart {
        config: ConnectorConfig,
    },
    AuthCallback {
        code: String,
        state: Option<String>,
    },
    ListSources {
        config: ConnectorConfig,
    },
    Sync {
        source_id: String,
        cursor: Option<String>,
    },
    FetchDocument {
        document_id: String,
    },
    TransformPrompt {
        config: ConnectorConfig,
        prompt: String,
    },
    ResolveAcl {
        acl: ExternalAcl,
    },
    Health,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConnectorResponse {
    pub id: String,
    #[serde(flatten)]
    pub kind: ConnectorResponseKind,
}

impl ConnectorResponse {
    /// Build a successful response correlated with a request id.
    pub fn ok(id: impl Into<String>, result: ConnectorResult) -> Self {
        Self {
            id: id.into(),
            kind: ConnectorResponseKind::Ok(result),
        }
    }

    /// Build an error response correlated with a request id.
    pub fn error(id: impl Into<String>, message: impl Into<String>, code: Option<String>) -> Self {
        Self {
            id: id.into(),
            kind: ConnectorResponseKind::Err(ConnectorErrorPayload {
                message: message.into(),
                code,
            }),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "status", content = "result")]
pub enum ConnectorResponseKind {
    Ok(ConnectorResult),
    Err(ConnectorErrorPayload),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", content = "value")]
pub enum ConnectorResult {
    Unit,
    AuthUrl(String),
    Sources(Vec<SourceDescriptor>),
    Deltas(Vec<DocumentDelta>),
    Document(DocumentPayload),
    PromptTransform(PromptTransformResult),
    Acl(ExternalAcl),
    Health(String),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConnectorErrorPayload {
    pub message: String,
    pub code: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn request_wire_shape_keeps_method_and_params_flattened() {
        let request = ConnectorRequest::new(
            "req-1",
            ConnectorRequestKind::ListSources {
                config: json!({"bucket": "documents"}),
            },
        );

        assert_eq!(
            serde_json::to_value(request).unwrap(),
            json!({
                "id": "req-1",
                "method": "ListSources",
                "params": {"config": {"bucket": "documents"}}
            })
        );
    }

    #[test]
    fn error_response_wire_shape_includes_correlation_and_code() {
        let response = ConnectorResponse::error(
            "req-2",
            "invalid configuration",
            Some("invalid_input".to_string()),
        );

        assert_eq!(
            serde_json::to_value(response).unwrap(),
            json!({
                "id": "req-2",
                "status": "Err",
                "result": {
                    "message": "invalid configuration",
                    "code": "invalid_input"
                }
            })
        );
    }
}
