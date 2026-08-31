use async_trait::async_trait;
use futures::executor::block_on;
use std::io::{self, BufRead, Write};

use crate::protocol::{ConnectorRequest, ConnectorRequestKind, ConnectorResponse, ConnectorResult};
use crate::types::{
    ConnectorConfig, DocumentDelta, DocumentPayload, ExternalAcl, PromptTransformResult,
    SourceDescriptor,
};

/// Maximum JSON request size accepted by a connector's stdio server.
pub const MAX_CONNECTOR_REQUEST_BYTES: usize = 4 * 1024 * 1024;
/// Maximum response frame read from a connector sidecar.
pub const MAX_CONNECTOR_RESPONSE_BYTES: usize = 64 * 1024 * 1024;

#[derive(thiserror::Error, Debug, PartialEq, Eq)]
pub enum ConnectorError {
    #[error("unsupported operation: {0}")]
    Unsupported(String),
    #[error("invalid input: {0}")]
    InvalidInput(String),
    #[error("io error: {0}")]
    Io(String),
    #[error("serialization error: {0}")]
    Serialization(String),
    #[error("internal error: {0}")]
    Internal(String),
}

impl ConnectorError {
    /// Stable machine-readable code returned in connector error responses.
    pub fn code(&self) -> &'static str {
        match self {
            Self::Unsupported(_) => "unsupported",
            Self::InvalidInput(_) => "invalid_input",
            Self::Io(_) => "io",
            Self::Serialization(_) => "serialization",
            Self::Internal(_) => "internal",
        }
    }
}

impl From<io::Error> for ConnectorError {
    fn from(err: io::Error) -> Self {
        ConnectorError::Io(err.to_string())
    }
}

impl From<serde_json::Error> for ConnectorError {
    fn from(err: serde_json::Error) -> Self {
        ConnectorError::Serialization(err.to_string())
    }
}

#[async_trait]
pub trait Connector: Send + Sync {
    async fn validate_config(&self, config: ConnectorConfig) -> Result<(), ConnectorError>;
    async fn auth_start(&self, _config: ConnectorConfig) -> Result<String, ConnectorError> {
        Err(ConnectorError::Unsupported("auth_start".to_string()))
    }
    async fn auth_callback(
        &self,
        _code: String,
        _state: Option<String>,
    ) -> Result<(), ConnectorError> {
        Err(ConnectorError::Unsupported("auth_callback".to_string()))
    }
    async fn list_sources(
        &self,
        config: ConnectorConfig,
    ) -> Result<Vec<SourceDescriptor>, ConnectorError>;
    async fn sync(
        &self,
        source_id: String,
        cursor: Option<String>,
    ) -> Result<Vec<DocumentDelta>, ConnectorError>;
    async fn fetch_document(&self, document_id: String) -> Result<DocumentPayload, ConnectorError>;
    async fn transform_prompt(
        &self,
        _config: ConnectorConfig,
        _prompt: String,
    ) -> Result<PromptTransformResult, ConnectorError> {
        Err(ConnectorError::Unsupported(
            "prompt transformation is not implemented".to_string(),
        ))
    }
    async fn resolve_acl(&self, _acl: ExternalAcl) -> Result<ExternalAcl, ConnectorError> {
        Err(ConnectorError::Unsupported("resolve_acl".to_string()))
    }
    async fn health(&self) -> Result<String, ConnectorError> {
        Ok("ok".to_string())
    }
}

pub fn serve_stdio<C: Connector>(connector: C) -> Result<(), ConnectorError> {
    let stdin = io::stdin();
    let stdout = io::stdout();
    serve_io(&connector, stdin.lock(), stdout.lock())
}

/// Serve newline-delimited connector requests over arbitrary buffered I/O.
///
/// This is the testable transport primitive behind [`serve_stdio`]. It is also
/// useful when embedding a connector in a custom process supervisor.
pub fn serve_io<C, R, W>(connector: &C, mut reader: R, mut writer: W) -> Result<(), ConnectorError>
where
    C: Connector,
    R: BufRead,
    W: Write,
{
    while let Some(line) = read_frame_with_limit(&mut reader, MAX_CONNECTOR_REQUEST_BYTES)? {
        if line.iter().all(|byte| byte.is_ascii_whitespace()) {
            continue;
        }
        let request: ConnectorRequest = serde_json::from_slice(&line)?;
        let response = block_on(dispatch(connector, request));
        serde_json::to_writer(&mut writer, &response)?;
        writer.write_all(b"\n")?;
        writer.flush()?;
    }
    Ok(())
}

/// Read one size-bounded newline-delimited protocol frame.
pub fn read_frame<R: BufRead>(reader: &mut R) -> Result<Option<Vec<u8>>, ConnectorError> {
    read_frame_with_limit(reader, MAX_CONNECTOR_RESPONSE_BYTES)
}

fn read_frame_with_limit<R: BufRead>(
    reader: &mut R,
    maximum_bytes: usize,
) -> Result<Option<Vec<u8>>, ConnectorError> {
    let mut line = Vec::new();
    loop {
        let available = reader.fill_buf()?;
        if available.is_empty() {
            return Ok((!line.is_empty()).then_some(line));
        }

        let consumed = available
            .iter()
            .position(|byte| *byte == b'\n')
            .map_or(available.len(), |position| position + 1);
        if line.len().saturating_add(consumed) > maximum_bytes {
            return Err(ConnectorError::InvalidInput(format!(
                "connector frame exceeds {maximum_bytes} bytes"
            )));
        }
        let finished = available[consumed - 1] == b'\n';
        line.extend_from_slice(&available[..consumed]);
        reader.consume(consumed);
        if finished {
            return Ok(Some(line));
        }
    }
}

async fn dispatch<C: Connector>(connector: &C, request: ConnectorRequest) -> ConnectorResponse {
    let id = request.id.clone();
    let result = match request.kind {
        ConnectorRequestKind::ValidateConfig { config } => connector
            .validate_config(config)
            .await
            .map(|_| ConnectorResult::Unit),
        ConnectorRequestKind::AuthStart { config } => connector
            .auth_start(config)
            .await
            .map(ConnectorResult::AuthUrl),
        ConnectorRequestKind::AuthCallback { code, state } => connector
            .auth_callback(code, state)
            .await
            .map(|_| ConnectorResult::Unit),
        ConnectorRequestKind::ListSources { config } => connector
            .list_sources(config)
            .await
            .map(ConnectorResult::Sources),
        ConnectorRequestKind::Sync { source_id, cursor } => connector
            .sync(source_id, cursor)
            .await
            .map(ConnectorResult::Deltas),
        ConnectorRequestKind::FetchDocument { document_id } => connector
            .fetch_document(document_id)
            .await
            .map(ConnectorResult::Document),
        ConnectorRequestKind::TransformPrompt { config, prompt } => connector
            .transform_prompt(config, prompt)
            .await
            .map(ConnectorResult::PromptTransform),
        ConnectorRequestKind::ResolveAcl { acl } => {
            connector.resolve_acl(acl).await.map(ConnectorResult::Acl)
        }
        ConnectorRequestKind::Health => connector.health().await.map(ConnectorResult::Health),
    };

    match result {
        Ok(value) => ConnectorResponse::ok(id, value),
        Err(err) => ConnectorResponse::error(id, err.to_string(), Some(err.code().to_string())),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::{ConnectorRequest, ConnectorRequestKind, ConnectorResponseKind};
    use crate::types::ConnectorConfig;
    use serde_json::json;
    use std::io::Cursor;

    struct MockConnector;

    #[async_trait]
    impl Connector for MockConnector {
        async fn validate_config(&self, _config: ConnectorConfig) -> Result<(), ConnectorError> {
            Ok(())
        }

        async fn list_sources(
            &self,
            _config: ConnectorConfig,
        ) -> Result<Vec<crate::types::SourceDescriptor>, ConnectorError> {
            Ok(Vec::new())
        }

        async fn sync(
            &self,
            _source_id: String,
            _cursor: Option<String>,
        ) -> Result<Vec<crate::types::DocumentDelta>, ConnectorError> {
            Ok(Vec::new())
        }

        async fn fetch_document(
            &self,
            _document_id: String,
        ) -> Result<crate::types::DocumentPayload, ConnectorError> {
            Err(ConnectorError::Unsupported("fetch".to_string()))
        }

        async fn transform_prompt(
            &self,
            _config: ConnectorConfig,
            prompt: String,
        ) -> Result<PromptTransformResult, ConnectorError> {
            Ok(PromptTransformResult {
                prompt: format!("<wrapped>{}</wrapped>", prompt),
            })
        }
    }

    #[test]
    fn dispatch_transform_prompt_returns_transformed_prompt() {
        let response = block_on(dispatch(
            &MockConnector,
            ConnectorRequest::new(
                "req-1",
                ConnectorRequestKind::TransformPrompt {
                    config: json!({"format":"custom"}),
                    prompt: "hello".to_string(),
                },
            ),
        ));

        match response.kind {
            ConnectorResponseKind::Ok(ConnectorResult::PromptTransform(result)) => {
                assert_eq!(result.prompt, "<wrapped>hello</wrapped>");
            }
            other => panic!("unexpected response: {:?}", other),
        }
    }

    #[test]
    fn optional_operations_return_a_stable_error_code() {
        let response = block_on(dispatch(
            &MockConnector,
            ConnectorRequest::new(
                "req-auth",
                ConnectorRequestKind::AuthStart { config: json!({}) },
            ),
        ));

        let ConnectorResponseKind::Err(error) = response.kind else {
            panic!("expected unsupported response")
        };
        assert_eq!(response.id, "req-auth");
        assert_eq!(error.code.as_deref(), Some("unsupported"));
    }

    #[test]
    fn serve_io_handles_multiple_requests_and_blank_lines() {
        let input = Cursor::new(
            b"\n{\"id\":\"req-1\",\"method\":\"Health\"}\n{\"id\":\"req-2\",\"method\":\"Health\"}\n",
        );
        let mut output = Vec::new();

        serve_io(&MockConnector, input, &mut output).unwrap();

        let responses = String::from_utf8(output).unwrap();
        let responses = responses
            .lines()
            .map(|line| serde_json::from_str::<ConnectorResponse>(line).unwrap())
            .collect::<Vec<_>>();
        assert_eq!(responses.len(), 2);
        assert_eq!(responses[0].id, "req-1");
        assert_eq!(responses[1].id, "req-2");
    }

    #[test]
    fn stdio_transport_rejects_oversized_requests() {
        let mut input = Cursor::new(vec![b'x'; MAX_CONNECTOR_REQUEST_BYTES + 1]);

        let error = read_frame_with_limit(&mut input, MAX_CONNECTOR_REQUEST_BYTES).unwrap_err();

        assert!(matches!(error, ConnectorError::InvalidInput(_)));
    }
}
