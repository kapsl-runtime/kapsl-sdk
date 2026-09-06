use kapsl_communication::transport::protocol::{asynchronous, CodecError, OP_INFER_STREAM};
use kapsl_engine_api::{
    BinaryTensorPacket, CancellationToken, InferenceRequest, NamedTensor, TensorDtype,
};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::collections::{HashMap, VecDeque};
use std::str::FromStr;
use std::sync::{Arc, Mutex};
#[cfg(unix)]
use tokio::net::UnixStream;
use tokio::{
    net::TcpStream,
    runtime::{Builder, Runtime},
};

use crate::native_stream::{deadline, with_deadline, ClientConnection, StreamIterator};
use crate::request_options::parse_options;

const DEFAULT_MAX_POOL_SIZE: usize = 8;
#[cfg(unix)]
const DEFAULT_SOCKET_ENDPOINT: &str = "/tmp/kapsl.sock";
#[cfg(windows)]
const DEFAULT_SOCKET_ENDPOINT: &str = r"\\.\pipe\kapsl";
const DEFAULT_TCP_HOST: &str = "127.0.0.1";
const DEFAULT_TCP_PORT: u16 = 9096;

type RawTensorInput = (Vec<i64>, String, Vec<u8>);
type AdditionalInputMap = HashMap<String, RawTensorInput>;

enum TransportProtocol {
    Socket,
    Tcp,
    Pipe,
}

enum ConnectionTarget {
    #[cfg(unix)]
    UnixSocket(String),
    #[cfg(windows)]
    NamedPipe(String),
    Tcp(String),
}

#[derive(Debug)]
pub(crate) enum ClientError {
    Io(std::io::Error),
    InvalidEndpoint(String),
    InvalidDtype(String),
    Serialization(String),
    Server(String),
    Timeout,
    Closed,
}

impl From<std::io::Error> for ClientError {
    fn from(value: std::io::Error) -> Self {
        Self::Io(value)
    }
}

impl From<CodecError> for ClientError {
    fn from(value: CodecError) -> Self {
        match value {
            CodecError::Io(err) => Self::Io(err),
            CodecError::Remote(message) => Self::Server(message),
            other => Self::Serialization(other.to_string()),
        }
    }
}

impl From<ClientError> for PyErr {
    fn from(value: ClientError) -> Self {
        match value {
            ClientError::Io(err) => {
                PyErr::new::<pyo3::exceptions::PyConnectionError, _>(err.to_string())
            }
            ClientError::InvalidEndpoint(msg) | ClientError::InvalidDtype(msg) => {
                PyErr::new::<pyo3::exceptions::PyValueError, _>(msg)
            }
            ClientError::Serialization(msg) | ClientError::Server(msg) => {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(msg)
            }
            ClientError::Timeout => {
                pyo3::exceptions::PyTimeoutError::new_err("Inference deadline exceeded")
            }
            ClientError::Closed => {
                pyo3::exceptions::PyRuntimeError::new_err("The client is closed")
            }
        }
    }
}

impl TransportProtocol {
    fn parse(raw: &str) -> Result<Self, ClientError> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "socket" | "unix" | "local" => Ok(Self::Socket),
            "tcp" => Ok(Self::Tcp),
            "pipe" | "named_pipe" | "named-pipe" => Ok(Self::Pipe),
            other => Err(ClientError::InvalidEndpoint(format!(
                "Unsupported protocol '{}'. Use one of: socket, tcp, pipe",
                other
            ))),
        }
    }
}

impl ConnectionTarget {
    fn default_local() -> Self {
        #[cfg(unix)]
        {
            Self::UnixSocket(DEFAULT_SOCKET_ENDPOINT.to_string())
        }
        #[cfg(windows)]
        {
            Self::NamedPipe(DEFAULT_SOCKET_ENDPOINT.to_string())
        }
    }

    fn protocol_name(&self) -> &'static str {
        match self {
            #[cfg(unix)]
            Self::UnixSocket(_) => "socket",
            #[cfg(windows)]
            Self::NamedPipe(_) => "pipe",
            Self::Tcp(_) => "tcp",
        }
    }

    fn endpoint_display(&self) -> String {
        match self {
            #[cfg(unix)]
            Self::UnixSocket(path) => format!("unix://{}", path),
            #[cfg(windows)]
            Self::NamedPipe(path) => format!("pipe://{}", path),
            Self::Tcp(addr) => format!("tcp://{}", addr),
        }
    }

    #[cfg(windows)]
    fn normalize_pipe_path(path: &str) -> String {
        if path.starts_with(r"\\.\pipe\") {
            path.to_string()
        } else {
            format!(r"\\.\pipe\{}", path)
        }
    }

    fn from_endpoint(endpoint: &str) -> Result<Self, ClientError> {
        let endpoint = endpoint.trim();
        if endpoint.is_empty() {
            return Err(ClientError::InvalidEndpoint(
                "Endpoint cannot be empty".to_string(),
            ));
        }

        if let Some(addr) = endpoint.strip_prefix("tcp://") {
            if addr.is_empty() {
                return Err(ClientError::InvalidEndpoint(
                    "tcp:// endpoint must include host:port".to_string(),
                ));
            }
            return Ok(Self::Tcp(addr.to_string()));
        }

        #[cfg(unix)]
        {
            if let Some(path) = endpoint.strip_prefix("unix://") {
                if path.is_empty() {
                    return Err(ClientError::InvalidEndpoint(
                        "unix:// endpoint must include a socket path".to_string(),
                    ));
                }
                return Ok(Self::UnixSocket(path.to_string()));
            }
            Ok(Self::UnixSocket(endpoint.to_string()))
        }

        #[cfg(windows)]
        {
            if endpoint.starts_with("unix://") {
                return Err(ClientError::InvalidEndpoint(
                    "unix:// endpoints are only supported on Unix".to_string(),
                ));
            }

            if let Some(pipe_path) = endpoint.strip_prefix("pipe://") {
                if pipe_path.is_empty() {
                    return Err(ClientError::InvalidEndpoint(
                        "pipe:// endpoint must include a pipe name".to_string(),
                    ));
                }
                return Ok(Self::NamedPipe(Self::normalize_pipe_path(pipe_path)));
            }

            Ok(Self::NamedPipe(endpoint.to_string()))
        }
    }

    fn from_options(
        endpoint: Option<&str>,
        protocol: Option<&str>,
        host: Option<&str>,
        port: Option<u16>,
        socket_path: Option<&str>,
        pipe_name: Option<&str>,
    ) -> Result<Self, ClientError> {
        fn normalize_opt(value: Option<&str>) -> Option<&str> {
            value.and_then(|raw| {
                let trimmed = raw.trim();
                if trimmed.is_empty() {
                    None
                } else {
                    Some(trimmed)
                }
            })
        }

        let endpoint = normalize_opt(endpoint);
        let protocol = normalize_opt(protocol);
        let host = normalize_opt(host);
        let socket_path = normalize_opt(socket_path);
        let pipe_name = normalize_opt(pipe_name);

        if let Some(protocol_raw) = protocol {
            let parsed = TransportProtocol::parse(protocol_raw)?;
            return match parsed {
                TransportProtocol::Tcp => {
                    if socket_path.is_some() || pipe_name.is_some() {
                        return Err(ClientError::InvalidEndpoint(
                            "protocol='tcp' cannot be combined with socket_path or pipe_name"
                                .to_string(),
                        ));
                    }

                    if let Some(endpoint_value) = endpoint {
                        if host.is_some() || port.is_some() {
                            return Err(ClientError::InvalidEndpoint(
                                "When endpoint is provided, do not also pass host/port".to_string(),
                            ));
                        }
                        if let Some(addr) = endpoint_value.strip_prefix("tcp://") {
                            if addr.is_empty() {
                                return Err(ClientError::InvalidEndpoint(
                                    "tcp:// endpoint must include host:port".to_string(),
                                ));
                            }
                            return Ok(Self::Tcp(addr.to_string()));
                        }
                        if endpoint_value.contains("://") {
                            return Err(ClientError::InvalidEndpoint(
                                "protocol='tcp' expects endpoint as host:port or tcp://host:port"
                                    .to_string(),
                            ));
                        }
                        return Ok(Self::Tcp(endpoint_value.to_string()));
                    }

                    let host = host.unwrap_or(DEFAULT_TCP_HOST);
                    let port = port.unwrap_or(DEFAULT_TCP_PORT);
                    Ok(Self::Tcp(format!("{}:{}", host, port)))
                }
                TransportProtocol::Socket => {
                    if host.is_some() || port.is_some() || pipe_name.is_some() {
                        return Err(ClientError::InvalidEndpoint(
                            "protocol='socket' cannot be combined with host/port/pipe_name"
                                .to_string(),
                        ));
                    }
                    if let Some(path_or_uri) = socket_path.or(endpoint) {
                        return Self::from_endpoint(path_or_uri);
                    }
                    Ok(Self::default_local())
                }
                TransportProtocol::Pipe => {
                    if host.is_some() || port.is_some() || socket_path.is_some() {
                        return Err(ClientError::InvalidEndpoint(
                            "protocol='pipe' cannot be combined with host/port/socket_path"
                                .to_string(),
                        ));
                    }
                    let value = pipe_name.or(endpoint).unwrap_or(DEFAULT_SOCKET_ENDPOINT);
                    #[cfg(unix)]
                    {
                        let _ = value;
                        Err(ClientError::InvalidEndpoint(
                            "protocol='pipe' is only supported on Windows".to_string(),
                        ))
                    }
                    #[cfg(windows)]
                    {
                        if let Some(raw) = value.strip_prefix("pipe://") {
                            if raw.is_empty() {
                                return Err(ClientError::InvalidEndpoint(
                                    "pipe:// endpoint must include a pipe name".to_string(),
                                ));
                            }
                            return Ok(Self::NamedPipe(Self::normalize_pipe_path(raw)));
                        }
                        if value.starts_with("tcp://") || value.starts_with("unix://") {
                            return Err(ClientError::InvalidEndpoint(
                                "protocol='pipe' expects a named pipe path".to_string(),
                            ));
                        }
                        Ok(Self::NamedPipe(Self::normalize_pipe_path(value)))
                    }
                }
            };
        }

        if let Some(endpoint_value) = endpoint {
            return Self::from_endpoint(endpoint_value);
        }

        if host.is_some() || port.is_some() {
            let host = host.unwrap_or(DEFAULT_TCP_HOST);
            let port = port.unwrap_or(DEFAULT_TCP_PORT);
            return Ok(Self::Tcp(format!("{}:{}", host, port)));
        }

        if let Some(path) = socket_path {
            return Self::from_endpoint(path);
        }

        if let Some(name) = pipe_name {
            #[cfg(unix)]
            {
                let _ = name;
                return Err(ClientError::InvalidEndpoint(
                    "pipe_name is only supported on Windows".to_string(),
                ));
            }
            #[cfg(windows)]
            {
                return Ok(Self::NamedPipe(Self::normalize_pipe_path(name)));
            }
        }

        Ok(Self::default_local())
    }
}

#[pyclass]
pub(crate) struct KapslClient {
    target: ConnectionTarget,
    max_pool_size: usize,
    connection_pool: Mutex<VecDeque<ClientConnection>>,
    api_token: Option<String>,
    runtime: Arc<Runtime>,
    closed: CancellationToken,
}

impl KapslClient {
    async fn connect_stream(&self) -> Result<ClientConnection, ClientError> {
        match &self.target {
            #[cfg(unix)]
            ConnectionTarget::UnixSocket(path) => Ok(Box::new(UnixStream::connect(path).await?)),
            #[cfg(windows)]
            ConnectionTarget::NamedPipe(path) => Ok(Box::new(
                kapsl_communication::transport::named_pipe::connect(path).await?,
            )),
            ConnectionTarget::Tcp(addr) => {
                let stream = TcpStream::connect(addr).await?;
                let _ = stream.set_nodelay(true);
                Ok(Box::new(stream))
            }
        }
    }

    async fn checkout_connection(&self) -> Result<ClientConnection, ClientError> {
        if let Ok(mut pool) = self.connection_pool.lock() {
            if let Some(stream) = pool.pop_front() {
                return Ok(stream);
            }
        }
        self.connect_stream().await
    }

    fn return_connection(&self, stream: ClientConnection) {
        if self.max_pool_size == 0 {
            return;
        }
        if let Ok(mut pool) = self.connection_pool.lock() {
            if !self.closed.is_cancelled() && pool.len() < self.max_pool_size {
                pool.push_back(stream);
            }
        }
    }

    fn parse_additional_inputs(raw: AdditionalInputMap) -> Result<Vec<NamedTensor>, ClientError> {
        raw.into_iter()
            .map(|(name, (shape, dtype_str, data))| {
                let dtype = TensorDtype::from_str(&dtype_str)
                    .map_err(|e| ClientError::InvalidDtype(e.to_string()))?;
                Ok(NamedTensor {
                    name,
                    tensor: BinaryTensorPacket { shape, dtype, data },
                })
            })
            .collect()
    }

    fn build_request(
        &self,
        shape: Vec<i64>,
        dtype: String,
        data: Vec<u8>,
        additional_inputs: Option<AdditionalInputMap>,
        session_id: Option<String>,
        options: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<InferenceRequest> {
        let dtype = TensorDtype::from_str(&dtype)
            .map_err(|error| PyErr::from(ClientError::InvalidDtype(error.to_string())))?;
        let additional_inputs =
            Self::parse_additional_inputs(additional_inputs.unwrap_or_default())
                .map_err(PyErr::from)?;

        Ok(InferenceRequest {
            input: BinaryTensorPacket { shape, dtype, data },
            additional_inputs,
            session_id,
            metadata: parse_options(options, self.api_token.as_deref())?,
            cancellation: None,
        })
    }

    fn run_infer(
        &self,
        model_id: u32,
        request: &InferenceRequest,
    ) -> Result<BinaryTensorPacket, ClientError> {
        let deadline = deadline(request.metadata.as_ref().and_then(|m| m.timeout_ms))?;
        self.runtime.block_on(async {
            tokio::select! {
                biased;
                _ = self.closed.cancelled() => Err(ClientError::Closed),
                result = with_deadline(deadline, async {
                    let mut stream = self.checkout_connection().await?;
                    let result = asynchronous::infer_request_over_stream(
                        stream.as_mut(), model_id, request,
                    ).await.map_err(ClientError::from);
                    if result.is_ok() || matches!(result, Err(ClientError::Server(_))) {
                        self.return_connection(stream);
                    }
                    result
                }) => result,
            }
        })
    }
}

#[pymethods]
#[allow(clippy::too_many_arguments, clippy::useless_conversion)]
impl KapslClient {
    #[new]
    #[pyo3(signature = (
        endpoint = None,
        *,
        protocol = None,
        host = None,
        port = None,
        socket_path = None,
        pipe_name = None,
        max_pool_size = DEFAULT_MAX_POOL_SIZE,
        api_token = None
    ))]
    // This mirrors the stable keyword-based Python constructor.
    #[allow(clippy::too_many_arguments)]
    fn new(
        endpoint: Option<String>,
        protocol: Option<String>,
        host: Option<String>,
        port: Option<u16>,
        socket_path: Option<String>,
        pipe_name: Option<String>,
        max_pool_size: usize,
        api_token: Option<String>,
    ) -> PyResult<Self> {
        let target = ConnectionTarget::from_options(
            endpoint.as_deref(),
            protocol.as_deref(),
            host.as_deref(),
            port,
            socket_path.as_deref(),
            pipe_name.as_deref(),
        )
        .map_err(PyErr::from)?;
        Ok(Self {
            target,
            max_pool_size,
            connection_pool: Mutex::new(VecDeque::new()),
            api_token,
            runtime: Arc::new(
                Builder::new_multi_thread()
                    .worker_threads(2)
                    .enable_all()
                    .build()?,
            ),
            closed: CancellationToken::new(),
        })
    }

    fn protocol(&self) -> String {
        self.target.protocol_name().to_string()
    }

    fn endpoint(&self) -> String {
        self.target.endpoint_display()
    }

    #[pyo3(signature = (model_id, shape, dtype, data, additional_inputs = None, session_id = None, *, options = None))]
    fn infer(
        &self,
        py: Python<'_>,
        model_id: u32,
        shape: Vec<i64>,
        dtype: String,
        data: Vec<u8>,
        additional_inputs: Option<AdditionalInputMap>,
        session_id: Option<String>,
        options: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Vec<u8>> {
        let request =
            self.build_request(shape, dtype, data, additional_inputs, session_id, options)?;
        py.detach(|| self.run_infer(model_id, &request))
            .map_err(PyErr::from)
            .map(|packet| packet.data)
    }

    /// Like `infer` but returns `(data, shape, dtype)` so the caller knows
    /// how to interpret the output bytes without hardcoding dimensions.
    /// Essential for models whose output shape varies (diffusion, video, TTS).
    #[pyo3(signature = (model_id, shape, dtype, data, additional_inputs = None, session_id = None, *, options = None))]
    fn infer_tensor(
        &self,
        py: Python<'_>,
        model_id: u32,
        shape: Vec<i64>,
        dtype: String,
        data: Vec<u8>,
        additional_inputs: Option<AdditionalInputMap>,
        session_id: Option<String>,
        options: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<(Vec<u8>, Vec<i64>, String)> {
        let request =
            self.build_request(shape, dtype, data, additional_inputs, session_id, options)?;
        py.detach(|| self.run_infer(model_id, &request))
            .map_err(PyErr::from)
            .map(|packet| (packet.data, packet.shape, packet.dtype.as_str().to_string()))
    }

    #[pyo3(signature = (model_id, shape, dtype, data, additional_inputs = None, session_id = None, *, options = None))]
    fn infer_stream(
        &self,
        py: Python<'_>,
        model_id: u32,
        shape: Vec<i64>,
        dtype: String,
        data: Vec<u8>,
        additional_inputs: Option<AdditionalInputMap>,
        session_id: Option<String>,
        options: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<StreamIterator> {
        let request =
            self.build_request(shape, dtype, data, additional_inputs, session_id, options)?;
        let deadline =
            deadline(request.metadata.as_ref().and_then(|m| m.timeout_ms)).map_err(PyErr::from)?;
        let stream = py
            .detach(|| {
                self.runtime.block_on(async {
                    tokio::select! {
                        biased;
                        _ = self.closed.cancelled() => Err(ClientError::Closed),
                        result = with_deadline(deadline, async {
                            let mut stream = self.connect_stream().await?;
                            asynchronous::write_request_value(
                                stream.as_mut(), model_id, OP_INFER_STREAM, &request,
                            ).await.map_err(ClientError::from)?;
                            Ok(stream)
                        }) => result,
                    }
                })
            })
            .map_err(PyErr::from)?;

        Ok(StreamIterator::new(
            self.runtime.clone(),
            stream,
            deadline,
            self.closed.clone(),
        ))
    }

    fn close(&self) {
        self.closed.cancel();
        self.connection_pool.lock().unwrap().clear();
    }

    #[getter]
    fn closed(&self) -> bool {
        self.closed.is_cancelled()
    }
}

impl Drop for KapslClient {
    fn drop(&mut self) {
        self.closed.cancel();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn codec_io_errors_remain_connection_errors() {
        let error = ClientError::from(CodecError::Io(std::io::Error::new(
            std::io::ErrorKind::BrokenPipe,
            "stale pooled connection",
        )));

        assert!(matches!(error, ClientError::Io(_)));
    }

    #[test]
    fn remote_codec_errors_remain_server_errors() {
        let error = ClientError::from(CodecError::Remote("model not found".to_string()));

        assert!(matches!(error, ClientError::Server(message) if message == "model not found"));
    }

    #[test]
    fn endpoint_options_build_tcp_targets() {
        let target = ConnectionTarget::from_options(
            None,
            Some("tcp"),
            Some("192.0.2.1"),
            Some(9000),
            None,
            None,
        )
        .expect("valid TCP target");

        assert_eq!(target.endpoint_display(), "tcp://192.0.2.1:9000");
        assert_eq!(target.protocol_name(), "tcp");
    }

    #[test]
    fn endpoint_options_reject_conflicting_tcp_inputs() {
        let result = ConnectionTarget::from_options(
            Some("tcp://127.0.0.1:9096"),
            Some("tcp"),
            Some("127.0.0.1"),
            None,
            None,
            None,
        );

        assert!(matches!(result, Err(ClientError::InvalidEndpoint(_))));
    }
}
