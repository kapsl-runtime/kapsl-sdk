//! Platform-native IPC server and inference request dispatch.

use crate::protocol::{
    HybridRequest, HybridResponse, HYBRID_SHM_PROTOCOL_VERSION, OP_HYBRID_INFER, OP_INFER,
    OP_INFER_STREAM, OP_OPENAI_WIRE, OP_OPENAI_WIRE_STREAM, STATUS_ERR, STATUS_OK,
    STATUS_OPENAI_WIRE_CHUNK, STATUS_OPENAI_WIRE_HEAD, STATUS_STREAM_CHUNK, STATUS_STREAM_END,
};
use async_trait::async_trait;
#[cfg(feature = "hybrid")]
use kapsl_engine_api::TensorDtype;
use kapsl_engine_api::{
    BinaryTensorPacket, CancellationToken, InferenceRequest, OpenAiWireFormat, OpenAiWireRequest,
};
use kapsl_scheduler::{Priority, ReplicaScheduler};
use kapsl_transport::protocol::{
    asynchronous as wire, decode_inference_request, CodecError, DEFAULT_MAX_FRAME_PAYLOAD_BYTES,
    MAX_OPENAI_WIRE_REQUEST_PAYLOAD_BYTES,
};
use kapsl_transport::{ResponseMetadata, TransportError, TransportServer};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite};

#[cfg(unix)]
use std::os::unix::fs::PermissionsExt;
#[cfg(windows)]
use tokio::net::windows::named_pipe::ServerOptions;
#[cfg(unix)]
use tokio::net::{UnixListener, UnixStream};

#[cfg(feature = "hybrid")]
use kapsl_shm::allocator::{SharedShmAllocator, SharedShmLease};
#[cfg(feature = "hybrid")]
use kapsl_shm::memory::{ShmManager, TensorHeader};

pub type SchedulerLookup =
    Arc<dyn Fn(u32) -> Option<Arc<dyn ReplicaScheduler + Send + Sync>> + Send + Sync>;

/// Location of an encoded tensor in a hybrid transport's shared memory.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HybridTensorLocation {
    /// Byte offset of the tensor header from the start of shared memory.
    pub offset: u64,
    /// Total encoded size of the tensor header and payload in bytes.
    pub size: u64,
    /// Process-shared lease token protecting this region.
    pub lease: u64,
}

/// Shared-memory operations required by the IPC hybrid opcode.
///
/// Keeping this interface transport-neutral lets socket/TCP-only builds avoid
/// linking a shared-memory implementation. Hybrid adapters must implement
/// lease validation and release explicitly.
pub trait HybridMemory: Send + Sync {
    /// Decode a tensor protected by a process-shared lease.
    fn read_leased_tensor(
        &self,
        location: HybridTensorLocation,
    ) -> Result<BinaryTensorPacket, String>;

    /// Encode an inference result and return the region advertised to the client.
    fn write_tensor(&self, tensor: &BinaryTensorPacket) -> Result<HybridTensorLocation, String>;

    /// Release an encoded tensor when its control-plane response could not be
    /// delivered.
    fn release_tensor(&self, location: HybridTensorLocation) -> bool;
}

#[cfg(feature = "hybrid")]
struct ShmHybridMemory {
    shm: Arc<ShmManager>,
    allocator: SharedShmAllocator,
}

#[cfg(feature = "hybrid")]
impl ShmHybridMemory {
    fn new(shm: Arc<ShmManager>) -> Self {
        let allocator =
            SharedShmAllocator::connect(shm.clone(), std::time::Duration::from_secs(30));
        Self { shm, allocator }
    }

    fn decode_tensor(
        &self,
        lease: SharedShmLease,
        encoded_size: usize,
    ) -> Result<BinaryTensorPacket, String> {
        let offset = lease.offset();
        let header_size = std::mem::size_of::<TensorHeader>();
        if encoded_size < header_size {
            return Err(format!(
                "Hybrid tensor region is too small: {encoded_size} bytes"
            ));
        }
        let header = unsafe {
            std::ptr::read_unaligned(self.shm.as_ptr().add(offset).cast::<TensorHeader>())
        };
        let rank = usize::try_from(header.ndim)
            .map_err(|_| "Hybrid tensor rank exceeds platform limits".to_string())?;
        if rank > header.shape.len() {
            return Err(format!(
                "Hybrid tensor rank {} exceeds maximum {}",
                rank,
                header.shape.len()
            ));
        }
        let data_size = usize::try_from(header.data_size)
            .map_err(|_| "Hybrid tensor data size exceeds platform limits".to_string())?;
        let total_size = header_size
            .checked_add(data_size)
            .ok_or_else(|| "Hybrid tensor encoded size overflow".to_string())?;
        if total_size > encoded_size || total_size > lease.capacity() {
            return Err(format!(
                "Hybrid tensor payload exceeds its lease: required={total_size}, declared={encoded_size}, capacity={}",
                lease.capacity()
            ));
        }
        let dtype = match header.dtype {
            0 => TensorDtype::Float32,
            1 => TensorDtype::Float64,
            2 => TensorDtype::Int32,
            3 => TensorDtype::Int64,
            value => return Err(format!("Unsupported hybrid tensor dtype code {value}")),
        };
        let data = unsafe {
            std::slice::from_raw_parts(self.shm.as_ptr().add(offset + header_size), data_size)
                .to_vec()
        };
        Ok(BinaryTensorPacket {
            shape: header.shape[..rank].to_vec(),
            dtype,
            data,
        })
    }
}

#[cfg(feature = "hybrid")]
impl HybridMemory for ShmHybridMemory {
    fn read_leased_tensor(
        &self,
        location: HybridTensorLocation,
    ) -> Result<BinaryTensorPacket, String> {
        let offset = usize::try_from(location.offset)
            .map_err(|_| "Hybrid tensor offset exceeds platform limits".to_string())?;
        let encoded_size = usize::try_from(location.size)
            .map_err(|_| "Hybrid tensor size exceeds platform limits".to_string())?;
        let lease = self
            .allocator
            .acquire(offset, encoded_size, location.lease)
            .ok_or_else(|| "Hybrid request carries an invalid or expired SHM lease".to_string())?;
        let result = self.decode_tensor(lease, encoded_size);
        let _ = self.allocator.release(lease);
        result
    }

    fn write_tensor(&self, tensor: &BinaryTensorPacket) -> Result<HybridTensorLocation, String> {
        if tensor.shape.len() > 8 {
            return Err(format!(
                "Hybrid tensor rank {} exceeds maximum 8",
                tensor.shape.len()
            ));
        }
        let header_size = std::mem::size_of::<TensorHeader>();
        let total_size = header_size
            .checked_add(tensor.data.len())
            .ok_or_else(|| "Hybrid output tensor size overflow".to_string())?;
        let dtype = match tensor.dtype {
            TensorDtype::Float32 => 0,
            TensorDtype::Float64 => 1,
            TensorDtype::Int32 => 2,
            TensorDtype::Int64 => 3,
            other => return Err(format!("Unsupported hybrid output dtype {other}")),
        };
        let lease = self.allocator.try_allocate(total_size).ok_or_else(|| {
            format!(
                "Hybrid SHM pool exhausted: required={total_size}, largest_slot={}",
                self.allocator.largest_slot_size()
            )
        })?;
        let mut shape = [0i64; 8];
        shape[..tensor.shape.len()].copy_from_slice(&tensor.shape);
        let header = TensorHeader {
            ndim: tensor.shape.len() as u32,
            dtype,
            _padding: [0; 3],
            shape,
            data_size: tensor.data.len() as u64,
        };

        unsafe {
            std::ptr::write_unaligned(
                self.shm.as_ptr().add(lease.offset()).cast::<TensorHeader>(),
                header,
            );
            std::ptr::copy_nonoverlapping(
                tensor.data.as_ptr(),
                self.shm.as_ptr().add(lease.offset() + header_size),
                tensor.data.len(),
            );
        }

        Ok(HybridTensorLocation {
            offset: lease.offset() as u64,
            size: total_size as u64,
            lease: lease.token(),
        })
    }

    fn release_tensor(&self, location: HybridTensorLocation) -> bool {
        let Ok(offset) = usize::try_from(location.offset) else {
            return false;
        };
        self.allocator.release_wire(offset, location.lease)
    }
}

// OpenAI ingress bodies are JSON/SSE control payloads, not arbitrary tensor
// storage. Keep their pre-auth allocation ceiling substantially below the
// tensor-frame maximum so an unauthenticated peer cannot force a 1 GiB
// allocation before the transport envelope's credential is checked.
const OPENAI_WIRE_OPERATION_LIMITS: &[(u32, usize)] = &[
    (OP_OPENAI_WIRE, MAX_OPENAI_WIRE_REQUEST_PAYLOAD_BYTES),
    (OP_OPENAI_WIRE_STREAM, MAX_OPENAI_WIRE_REQUEST_PAYLOAD_BYTES),
];
// A zero operation limit rejects every valid OpenAI wire envelope directly
// from its fixed header. This is used for plaintext TCP listeners exposed
// beyond loopback, before bearer credentials or prompt bytes are allocated or
// read from the socket.
const DISABLED_OPENAI_WIRE_OPERATION_LIMITS: &[(u32, usize)] =
    &[(OP_OPENAI_WIRE, 0), (OP_OPENAI_WIRE_STREAM, 0)];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum OpenAiWireTransportPolicy {
    Local,
    PlaintextRemote,
}

impl OpenAiWireTransportPolicy {
    fn operation_limits(self) -> &'static [(u32, usize)] {
        match self {
            Self::Local => OPENAI_WIRE_OPERATION_LIMITS,
            Self::PlaintextRemote => DISABLED_OPENAI_WIRE_OPERATION_LIMITS,
        }
    }

    fn rejects(self, operation: u32) -> bool {
        self == Self::PlaintextRemote && matches!(operation, OP_OPENAI_WIRE | OP_OPENAI_WIRE_STREAM)
    }
}

struct WireCancellationGuard(CancellationToken);

impl Drop for WireCancellationGuard {
    fn drop(&mut self) {
        self.0.cancel();
    }
}

async fn wait_for_peer_disconnect<T>(connection: &mut T) -> std::io::Result<()>
where
    T: AsyncRead + Unpin + ?Sized,
{
    let mut byte = [0u8; 1];
    match connection.read(&mut byte).await? {
        0 => Ok(()),
        _ => Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "IPC clients must not pipeline a request while a wire response is active",
        )),
    }
}

fn check_auth(request: &InferenceRequest, expected: Option<&str>) -> Option<String> {
    check_presented_auth(
        request
            .metadata
            .as_ref()
            .and_then(|metadata| metadata.auth_token.as_deref()),
        expected,
    )
}

fn check_presented_auth(presented: Option<&str>, expected: Option<&str>) -> Option<String> {
    let Some(expected_token) = expected else {
        return None; // auth not configured — allow all
    };
    if presented != Some(expected_token) {
        Some("Unauthorized".to_string())
    } else {
        None
    }
}

fn request_priority(request: &InferenceRequest, default: Priority) -> Priority {
    priority_value(
        request
            .metadata
            .as_ref()
            .and_then(|metadata| metadata.priority),
        default,
    )
}

async fn handle_tensor_request<T>(
    connection: &mut T,
    scheduler: &(dyn ReplicaScheduler + Send + Sync),
    mut request: InferenceRequest,
    streaming: bool,
) -> std::io::Result<()>
where
    T: AsyncRead + AsyncWrite + Unpin,
{
    use futures::StreamExt;
    let priority = request_priority(
        &request,
        if streaming {
            Priority::LatencyCritical
        } else {
            Priority::Throughput
        },
    );
    let force_cpu = request
        .metadata
        .as_ref()
        .and_then(|m| m.force_cpu)
        .unwrap_or(false);
    let timeout_ms = request.metadata.as_ref().and_then(|m| m.timeout_ms);
    let deadline = match timeout_ms {
        Some(0) => {
            write_error_response(connection, "timeout_ms must be positive").await?;
            return Ok(());
        }
        Some(ms) => {
            match tokio::time::Instant::now().checked_add(std::time::Duration::from_millis(ms)) {
                Some(at) => Some(at),
                None => {
                    write_error_response(connection, "timeout_ms is too large").await?;
                    return Ok(());
                }
            }
        }
        None => None,
    };
    let cancellation = CancellationToken::new();
    request.cancellation = Some(cancellation.clone());
    let _cancel_on_drop = WireCancellationGuard(cancellation);
    let (mut reader, mut writer) = tokio::io::split(connection);
    let operation = async {
        if streaming {
            match scheduler.infer_stream(request, priority, force_cpu).await {
                Ok(mut stream) => {
                    while let Some(result) = stream.next().await {
                        match result {
                            Ok(packet) => wire::write_response_value(
                                &mut writer,
                                STATUS_STREAM_CHUNK,
                                &packet,
                            )
                            .await
                            .map_err(codec_io)?,
                            Err(error) => {
                                write_error_response(&mut writer, &error.to_string()).await?;
                                return Ok(());
                            }
                        }
                    }
                    wire::write_response_bytes(&mut writer, STATUS_STREAM_END, &[])
                        .await
                        .map_err(codec_io)?;
                }
                Err(error) => write_error_response(&mut writer, &error.to_string()).await?,
            }
        } else {
            match scheduler.infer(&request, priority, force_cpu).await {
                Ok(packet) => wire::write_response_value(&mut writer, STATUS_OK, &packet)
                    .await
                    .map_err(codec_io)?,
                Err(error) => write_error_response(&mut writer, &error.to_string()).await?,
            }
        }
        Ok(())
    };
    tokio::select! {
        biased;
        _ = async {
            match deadline {
                Some(at) => tokio::time::sleep_until(at).await,
                None => std::future::pending().await,
            }
        } => Err(std::io::Error::new(
            std::io::ErrorKind::TimedOut, "Inference deadline exceeded",
        )),
        result = operation => result,
        result = wait_for_peer_disconnect(&mut reader) => result,
    }
}

fn openai_wire_priority(request: &OpenAiWireRequest, default: Priority) -> Priority {
    priority_value(
        request
            .metadata
            .as_ref()
            .and_then(|metadata| metadata.priority),
        default,
    )
}

fn priority_value(priority: Option<u8>, default: Priority) -> Priority {
    match priority {
        Some(0) => Priority::LatencyCritical,
        Some(_) => Priority::Throughput,
        None => default,
    }
}

fn codec_io(error: CodecError) -> std::io::Error {
    error.into_io_error()
}

fn inference_decode_message(error: CodecError) -> String {
    match error {
        CodecError::Deserialize(message) => format!("Deserialization error: {message}"),
        other => other.to_string(),
    }
}

pub struct IpcServer {
    socket_path: String,
    scheduler_lookup: SchedulerLookup,
    hybrid_memory: Option<Arc<dyn HybridMemory>>,
    auth_token: Option<Arc<str>>,
}

impl IpcServer {
    /// Construct a socket/named-pipe server without hybrid shared memory.
    pub fn new_socket(
        socket_path: &str,
        schedulers: HashMap<u32, Arc<dyn ReplicaScheduler + Send + Sync>>,
    ) -> Self {
        let schedulers = Arc::new(schedulers);
        let scheduler_lookup: SchedulerLookup =
            Arc::new(move |model_id| schedulers.get(&model_id).cloned());
        Self::new_socket_with_lookup(socket_path, scheduler_lookup)
    }

    /// Construct a socket/named-pipe server from a dynamic scheduler lookup.
    pub fn new_socket_with_lookup(socket_path: &str, scheduler_lookup: SchedulerLookup) -> Self {
        Self::new_with_lookup_and_hybrid_memory(socket_path, scheduler_lookup, None)
    }

    /// Construct an IPC server with an optional transport-neutral hybrid-memory adapter.
    pub fn new_with_lookup_and_hybrid_memory(
        socket_path: &str,
        scheduler_lookup: SchedulerLookup,
        hybrid_memory: Option<Arc<dyn HybridMemory>>,
    ) -> Self {
        Self {
            socket_path: socket_path.to_string(),
            scheduler_lookup,
            hybrid_memory,
            auth_token: None,
        }
    }

    /// Construct an IPC server with optional shared-memory tensor transfer.
    #[cfg(feature = "hybrid")]
    pub fn new(
        socket_path: &str,
        schedulers: HashMap<u32, Arc<dyn ReplicaScheduler + Send + Sync>>,
        shm_manager: Option<Arc<ShmManager>>,
    ) -> Self {
        let schedulers = Arc::new(schedulers);
        let scheduler_lookup: SchedulerLookup =
            Arc::new(move |model_id| schedulers.get(&model_id).cloned());
        Self::new_with_lookup(socket_path, scheduler_lookup, shm_manager)
    }

    /// Construct an IPC server with live scheduler lookup and optional SHM.
    #[cfg(feature = "hybrid")]
    pub fn new_with_lookup(
        socket_path: &str,
        scheduler_lookup: SchedulerLookup,
        shm_manager: Option<Arc<ShmManager>>,
    ) -> Self {
        let hybrid_memory = shm_manager
            .map(|manager| Arc::new(ShmHybridMemory::new(manager)) as Arc<dyn HybridMemory>);
        Self::new_with_lookup_and_hybrid_memory(socket_path, scheduler_lookup, hybrid_memory)
    }

    async fn run_internal(&self) -> std::io::Result<()> {
        let scheduler_lookup = self.scheduler_lookup.clone();
        let auth_token = self.auth_token.clone();

        #[cfg(unix)]
        {
            if std::path::Path::new(&self.socket_path).exists() {
                // Avoid clobbering a live socket from another runtime: if we can connect,
                // it is in-use and we should refuse to start.
                if UnixStream::connect(&self.socket_path).await.is_ok() {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::AddrInUse,
                        format!(
                            "IPC socket path {} is already in use. Is another kapsl runtime running? Use --socket to choose a different path.",
                            self.socket_path
                        ),
                    ));
                }

                // Stale socket (or leftover file) from a previous crash.
                std::fs::remove_file(&self.socket_path)?;
            }

            let listener = UnixListener::bind(&self.socket_path)?;
            std::fs::set_permissions(&self.socket_path, std::fs::Permissions::from_mode(0o600))?;
            log::info!("IPC Server listening on {}", self.socket_path);

            loop {
                let (stream, _) = listener.accept().await?;
                let scheduler_lookup = scheduler_lookup.clone();
                let hybrid_memory = self.hybrid_memory.clone();
                let auth_token = auth_token.clone();

                tokio::spawn(async move {
                    if let Err(e) =
                        handle_connection(stream, scheduler_lookup, hybrid_memory, auth_token).await
                    {
                        log::error!("Connection error: {}", e);
                    }
                });
            }
        }

        #[cfg(windows)]
        {
            loop {
                let server = ServerOptions::new().create(&self.socket_path)?;

                server.connect().await?;
                let scheduler_lookup = scheduler_lookup.clone();
                let hybrid_memory = self.hybrid_memory.clone();
                let auth_token = auth_token.clone();

                tokio::spawn(async move {
                    if let Err(e) =
                        handle_connection(server, scheduler_lookup, hybrid_memory, auth_token).await
                    {
                        log::error!("Connection error: {}", e);
                    }
                });
            }
        }
    }
}

#[async_trait]
impl TransportServer for IpcServer {
    async fn run(&self) -> Result<(), TransportError> {
        self.run_internal().await.map_err(TransportError::Io)
    }

    async fn shutdown(&self) -> Result<(), TransportError> {
        // Clean up socket file on shutdown
        #[cfg(unix)]
        {
            if std::path::Path::new(&self.socket_path).exists() {
                std::fs::remove_file(&self.socket_path).map_err(TransportError::Io)?;
            }
        }
        Ok(())
    }
}

pub(crate) async fn handle_connection<T>(
    connection: T,
    scheduler_lookup: SchedulerLookup,
    hybrid_memory: Option<Arc<dyn HybridMemory>>,
    auth_token: Option<Arc<str>>,
) -> std::io::Result<()>
where
    T: AsyncRead + AsyncWrite + Unpin,
{
    handle_connection_with_wire_policy(
        connection,
        scheduler_lookup,
        hybrid_memory,
        auth_token,
        OpenAiWireTransportPolicy::Local,
    )
    .await
}

pub(crate) async fn handle_connection_with_wire_policy<T>(
    mut connection: T,
    scheduler_lookup: SchedulerLookup,
    hybrid_memory: Option<Arc<dyn HybridMemory>>,
    auth_token: Option<Arc<str>>,
    wire_policy: OpenAiWireTransportPolicy,
) -> std::io::Result<()>
where
    T: AsyncRead + AsyncWrite + Unpin,
{
    loop {
        let Some(frame) = wire::read_request_frame_or_eof_with_operation_limits(
            &mut connection,
            DEFAULT_MAX_FRAME_PAYLOAD_BYTES,
            wire_policy.operation_limits(),
        )
        .await
        .map_err(codec_io)?
        else {
            return Ok(());
        };

        // The zero-sized limit above rejects every real envelope before its
        // payload is read. Keep this explicit check so even a malformed
        // zero-payload wire operation cannot enter dispatch on remote plaintext
        // TCP.
        if wire_policy.rejects(frame.header.op_code) {
            return Err(std::io::Error::new(
                std::io::ErrorKind::PermissionDenied,
                "protocol-native OpenAI operations require local IPC, loopback TCP, or a certified secure transport",
            ));
        }

        let model_id = frame.header.model_id;
        match frame.header.op_code {
            OP_OPENAI_WIRE_STREAM => {
                let envelope = match frame.decode_openai_wire_envelope() {
                    Ok(envelope)
                        if envelope.request.format == OpenAiWireFormat::ServerSentEvents =>
                    {
                        envelope
                    }
                    Ok(_) => {
                        write_error_response(
                            &mut connection,
                            "streaming OpenAI wire operation requires SSE format",
                        )
                        .await?;
                        continue;
                    }
                    Err(error) => {
                        write_error_response(&mut connection, &error.to_string()).await?;
                        continue;
                    }
                };
                if let Some(error) =
                    check_presented_auth(envelope.auth_token.as_deref(), auth_token.as_deref())
                {
                    write_error_response(&mut connection, &error).await?;
                    continue;
                }
                let mut request = envelope.request;
                let scheduler = match scheduler_lookup(model_id) {
                    Some(scheduler) => scheduler,
                    None => {
                        write_error_response(
                            &mut connection,
                            &format!("Model {model_id} not found"),
                        )
                        .await?;
                        continue;
                    }
                };

                let cancellation = CancellationToken::new();
                request.cancellation = Some(cancellation.clone());
                let _cancel_on_drop = WireCancellationGuard(cancellation);
                let priority = openai_wire_priority(&request, Priority::LatencyCritical);
                let operation = scheduler.infer_openai_wire_stream(request, priority, false);
                tokio::pin!(operation);
                let stream_result = tokio::select! {
                    result = &mut operation => result,
                    disconnected = wait_for_peer_disconnect(&mut connection) => {
                        return disconnected;
                    }
                };
                match stream_result {
                    Ok(mut response) => {
                        if let Err(error) = response.head.validate() {
                            write_error_response(&mut connection, &error.to_string()).await?;
                            continue;
                        }
                        wire::write_response_value(
                            &mut connection,
                            STATUS_OPENAI_WIRE_HEAD,
                            &response.head,
                        )
                        .await
                        .map_err(codec_io)?;

                        use futures::StreamExt;
                        let mut completed = true;
                        loop {
                            let next = response.body.next();
                            tokio::pin!(next);
                            let Some(chunk) = (tokio::select! {
                                chunk = &mut next => chunk,
                                disconnected = wait_for_peer_disconnect(&mut connection) => {
                                    return disconnected;
                                }
                            }) else {
                                break;
                            };
                            match chunk {
                                Ok(chunk) => wire::write_response_bytes(
                                    &mut connection,
                                    STATUS_OPENAI_WIRE_CHUNK,
                                    &chunk,
                                )
                                .await
                                .map_err(codec_io)?,
                                Err(error) => {
                                    write_error_response(&mut connection, &error.to_string())
                                        .await?;
                                    completed = false;
                                    break;
                                }
                            }
                        }
                        // Do not append End after an error frame: doing so
                        // leaves an unread frame that poisons connection reuse.
                        if completed {
                            wire::write_response_bytes(&mut connection, STATUS_STREAM_END, &[])
                                .await
                                .map_err(codec_io)?;
                        }
                    }
                    Err(error) => {
                        write_error_response(&mut connection, &error.to_string()).await?;
                    }
                }
            }
            OP_OPENAI_WIRE => {
                let envelope = match frame.decode_openai_wire_envelope() {
                    Ok(envelope) if envelope.request.format == OpenAiWireFormat::Json => envelope,
                    Ok(_) => {
                        write_error_response(
                            &mut connection,
                            "non-streaming OpenAI wire operation requires JSON format",
                        )
                        .await?;
                        continue;
                    }
                    Err(error) => {
                        write_error_response(&mut connection, &error.to_string()).await?;
                        continue;
                    }
                };
                if let Some(error) =
                    check_presented_auth(envelope.auth_token.as_deref(), auth_token.as_deref())
                {
                    write_error_response(&mut connection, &error).await?;
                    continue;
                }
                let mut request = envelope.request;
                let scheduler = match scheduler_lookup(model_id) {
                    Some(scheduler) => scheduler,
                    None => {
                        write_error_response(
                            &mut connection,
                            &format!("Model {model_id} not found"),
                        )
                        .await?;
                        continue;
                    }
                };

                let cancellation = CancellationToken::new();
                request.cancellation = Some(cancellation.clone());
                let _cancel_on_drop = WireCancellationGuard(cancellation);
                let priority = openai_wire_priority(&request, Priority::Throughput);
                let operation = scheduler.infer_openai_wire(request, priority, false);
                tokio::pin!(operation);
                let unary_result = tokio::select! {
                    result = &mut operation => result,
                    disconnected = wait_for_peer_disconnect(&mut connection) => {
                        return disconnected;
                    }
                };
                match unary_result {
                    Ok(response) => {
                        if let Err(error) = response.head.validate() {
                            write_error_response(&mut connection, &error.to_string()).await?;
                            continue;
                        }
                        wire::write_response_value(&mut connection, STATUS_OK, &response)
                            .await
                            .map_err(codec_io)?;
                    }
                    Err(error) => {
                        write_error_response(&mut connection, &error.to_string()).await?;
                    }
                }
            }
            OP_INFER | OP_INFER_STREAM => {
                let request = match decode_inference_request(&frame.payload) {
                    Ok(request) => request,
                    Err(error) => {
                        write_error_response(&mut connection, &inference_decode_message(error))
                            .await?;
                        continue;
                    }
                };
                if let Some(message) = check_auth(&request, auth_token.as_deref()) {
                    write_error_response(&mut connection, &message).await?;
                    continue;
                }
                let Some(scheduler) = scheduler_lookup(model_id) else {
                    write_error_response(&mut connection, &format!("Model {model_id} not found"))
                        .await?;
                    continue;
                };
                handle_tensor_request(
                    &mut connection,
                    scheduler.as_ref(),
                    request,
                    frame.header.op_code == OP_INFER_STREAM,
                )
                .await?;
            }
            OP_HYBRID_INFER => {
                let hybrid_req: HybridRequest = frame.deserialize().map_err(codec_io)?;
                if hybrid_req.protocol_version != HYBRID_SHM_PROTOCOL_VERSION {
                    write_error_response(
                        &mut connection,
                        &format!(
                            "Unsupported hybrid SHM protocol version {}; expected {}",
                            hybrid_req.protocol_version, HYBRID_SHM_PROTOCOL_VERSION
                        ),
                    )
                    .await?;
                    continue;
                }
                let Some(hybrid_memory) = hybrid_memory.as_ref() else {
                    write_error_response(&mut connection, "Hybrid memory is not configured")
                        .await?;
                    continue;
                };
                let packet = match hybrid_memory.read_leased_tensor(HybridTensorLocation {
                    offset: hybrid_req.shm_offset,
                    size: hybrid_req.shm_size,
                    lease: hybrid_req.shm_lease,
                }) {
                    Ok(packet) => packet,
                    Err(error) => {
                        write_error_response(&mut connection, &error).await?;
                        continue;
                    }
                };
                let request = InferenceRequest {
                    input: packet,
                    additional_inputs: Vec::new(),
                    session_id: None,
                    metadata: None,
                    cancellation: None,
                };
                let result = if let Some(scheduler) = scheduler_lookup(hybrid_req.metadata.model_id)
                {
                    scheduler
                        .infer(
                            &request,
                            Priority::Throughput,
                            hybrid_req.metadata.force_cpu,
                        )
                        .await
                } else {
                    Err(kapsl_engine_api::EngineError::ModelNotLoaded)
                };

                match result {
                    Ok(output) => match hybrid_memory.write_tensor(&output) {
                        Ok(location) => {
                            let response = HybridResponse {
                                metadata: ResponseMetadata {
                                    request_id: hybrid_req.metadata.request_id,
                                    status: STATUS_OK as u8,
                                    _padding: [0; 7],
                                    latency_ns: 0,
                                },
                                shm_offset: location.offset,
                                shm_size: location.size,
                                shm_lease: location.lease,
                                protocol_version: HYBRID_SHM_PROTOCOL_VERSION,
                            };
                            if let Err(error) =
                                wire::write_response_value(&mut connection, STATUS_OK, &response)
                                    .await
                            {
                                let _ = hybrid_memory.release_tensor(location);
                                return Err(codec_io(error));
                            }
                        }
                        Err(error) => {
                            write_error_response(&mut connection, &error).await?;
                        }
                    },
                    Err(error) => {
                        write_error_response(&mut connection, &error.to_string()).await?;
                    }
                }
            }
            _ => {
                wire::write_response_bytes(&mut connection, STATUS_ERR, &[])
                    .await
                    .map_err(codec_io)?;
            }
        }
    }
}

async fn write_error_response<T>(connection: &mut T, message: &str) -> std::io::Result<()>
where
    T: AsyncWrite + Unpin + ?Sized,
{
    wire::write_response_bytes(connection, STATUS_ERR, message.as_bytes())
        .await
        .map_err(codec_io)
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::Stream;
    use kapsl_engine_api::{
        EngineError, EngineMetrics, OpenAiWireEndpoint, OpenAiWireMetadata, OpenAiWireResponse,
        OpenAiWireResponseHead, OpenAiWireStreamResponse, RequestMetadata, TensorDtype,
    };
    use kapsl_transport::connection_pool::PoolConfig;
    use kapsl_transport::protocol::ResponseFrame;
    use kapsl_transport::tcp::TcpClient;
    use std::pin::Pin;
    use std::sync::Mutex;
    use tokio::io::{duplex, AsyncWriteExt, DuplexStream};

    struct PriorityRecordingScheduler {
        seen: Arc<Mutex<Vec<Priority>>>,
    }

    #[derive(Clone, Debug, PartialEq, Eq)]
    struct SeenWireRequest {
        priority: Priority,
        metadata_priority: Option<u8>,
        session_id: Option<String>,
        body: Vec<u8>,
    }

    struct WireRecordingScheduler {
        seen: Arc<Mutex<Vec<SeenWireRequest>>>,
    }

    struct IdleWireStreamScheduler {
        cancellation: Arc<Mutex<Option<CancellationToken>>>,
    }

    struct RecordingHybridMemory {
        input: BinaryTensorPacket,
        reads: Mutex<Vec<(u64, u64)>>,
        writes: Mutex<Vec<BinaryTensorPacket>>,
    }

    impl HybridMemory for RecordingHybridMemory {
        fn read_leased_tensor(
            &self,
            location: HybridTensorLocation,
        ) -> Result<BinaryTensorPacket, String> {
            self.reads
                .lock()
                .unwrap()
                .push((location.offset, location.size));
            Ok(self.input.clone())
        }

        fn write_tensor(
            &self,
            tensor: &BinaryTensorPacket,
        ) -> Result<HybridTensorLocation, String> {
            self.writes.lock().unwrap().push(tensor.clone());
            Ok(HybridTensorLocation {
                offset: 900,
                size: 120,
                lease: 91,
            })
        }

        fn release_tensor(&self, _location: HybridTensorLocation) -> bool {
            true
        }
    }

    #[async_trait::async_trait]
    impl ReplicaScheduler for PriorityRecordingScheduler {
        fn get_queue_depth(&self) -> (usize, usize) {
            (0, 0)
        }

        fn is_healthy(&self) -> bool {
            true
        }

        fn get_metrics(&self) -> EngineMetrics {
            EngineMetrics::default()
        }

        async fn infer(
            &self,
            request: &InferenceRequest,
            priority: Priority,
            _force_cpu: bool,
        ) -> Result<BinaryTensorPacket, EngineError> {
            self.seen.lock().unwrap().push(priority);
            Ok(request.input.clone())
        }

        async fn infer_stream(
            &self,
            request: InferenceRequest,
            priority: Priority,
            _force_cpu: bool,
        ) -> Result<
            Pin<Box<dyn Stream<Item = Result<BinaryTensorPacket, EngineError>> + Send>>,
            EngineError,
        > {
            self.seen.lock().unwrap().push(priority);
            Ok(Box::pin(futures::stream::once(
                async move { Ok(request.input) },
            )))
        }
    }

    #[async_trait::async_trait]
    impl ReplicaScheduler for WireRecordingScheduler {
        fn get_queue_depth(&self) -> (usize, usize) {
            (0, 0)
        }

        fn is_healthy(&self) -> bool {
            true
        }

        fn get_metrics(&self) -> EngineMetrics {
            EngineMetrics::default()
        }

        async fn infer(
            &self,
            _request: &InferenceRequest,
            _priority: Priority,
            _force_cpu: bool,
        ) -> Result<BinaryTensorPacket, EngineError> {
            Err(EngineError::backend("tensor inference not used"))
        }

        async fn infer_openai_wire(
            &self,
            request: OpenAiWireRequest,
            priority: Priority,
            _force_cpu: bool,
        ) -> Result<OpenAiWireResponse, EngineError> {
            self.record(&request, priority);
            Ok(OpenAiWireResponse {
                head: OpenAiWireResponseHead::new(201, Vec::new())?,
                body: b"{\"ok\":true}".to_vec(),
            })
        }

        async fn infer_openai_wire_stream(
            &self,
            request: OpenAiWireRequest,
            priority: Priority,
            _force_cpu: bool,
        ) -> Result<OpenAiWireStreamResponse, EngineError> {
            self.record(&request, priority);
            Ok(OpenAiWireStreamResponse {
                head: OpenAiWireResponseHead::new(200, Vec::new())?,
                body: Box::pin(futures::stream::iter(vec![
                    Ok(b"data: {\"part\":".to_vec()),
                    Ok(b"1}\n\ndata: [DONE]\n\n".to_vec()),
                ])),
            })
        }

        async fn infer_stream(
            &self,
            _request: InferenceRequest,
            _priority: Priority,
            _force_cpu: bool,
        ) -> Result<
            Pin<Box<dyn Stream<Item = Result<BinaryTensorPacket, EngineError>> + Send>>,
            EngineError,
        > {
            Err(EngineError::backend("tensor streaming not used"))
        }
    }

    #[async_trait::async_trait]
    impl ReplicaScheduler for IdleWireStreamScheduler {
        fn get_queue_depth(&self) -> (usize, usize) {
            (0, 0)
        }

        fn is_healthy(&self) -> bool {
            true
        }

        fn get_metrics(&self) -> EngineMetrics {
            EngineMetrics::default()
        }

        async fn infer(
            &self,
            _request: &InferenceRequest,
            _priority: Priority,
            _force_cpu: bool,
        ) -> Result<BinaryTensorPacket, EngineError> {
            Err(EngineError::backend("tensor inference not used"))
        }

        async fn infer_openai_wire_stream(
            &self,
            request: OpenAiWireRequest,
            _priority: Priority,
            _force_cpu: bool,
        ) -> Result<OpenAiWireStreamResponse, EngineError> {
            *self.cancellation.lock().unwrap() = request.cancellation;
            Ok(OpenAiWireStreamResponse {
                head: OpenAiWireResponseHead::new(200, Vec::new())?,
                body: Box::pin(futures::stream::pending()),
            })
        }

        async fn infer_stream(
            &self,
            _request: InferenceRequest,
            _priority: Priority,
            _force_cpu: bool,
        ) -> Result<
            Pin<Box<dyn Stream<Item = Result<BinaryTensorPacket, EngineError>> + Send>>,
            EngineError,
        > {
            Err(EngineError::backend("tensor streaming not used"))
        }
    }

    impl WireRecordingScheduler {
        fn record(&self, request: &OpenAiWireRequest, priority: Priority) {
            self.seen.lock().unwrap().push(SeenWireRequest {
                priority,
                metadata_priority: request
                    .metadata
                    .as_ref()
                    .and_then(|metadata| metadata.priority),
                session_id: request.session_id.clone(),
                body: request.body.clone(),
            });
        }
    }

    fn request_with_priority(priority: u8) -> InferenceRequest {
        let metadata = RequestMetadata {
            priority: Some(priority),
            ..RequestMetadata::default()
        };
        InferenceRequest::new(BinaryTensorPacket {
            shape: vec![1],
            dtype: TensorDtype::Float32,
            data: 1.0f32.to_ne_bytes().to_vec(),
        })
        .with_metadata(metadata)
    }

    async fn write_request(
        client: &mut DuplexStream,
        model_id: u32,
        op_code: u32,
        request: &InferenceRequest,
    ) {
        wire::write_request_value(client, model_id, op_code, request)
            .await
            .expect("write request frame");
    }

    async fn read_response(client: &mut DuplexStream) -> ResponseFrame {
        wire::read_response_frame(client, DEFAULT_MAX_FRAME_PAYLOAD_BYTES)
            .await
            .expect("read response frame")
    }

    fn request_header(model_id: u32, operation: u32, payload_size: u32) -> [u8; 12] {
        let mut header = [0u8; 12];
        header[0..4].copy_from_slice(&model_id.to_le_bytes());
        header[4..8].copy_from_slice(&operation.to_le_bytes());
        header[8..12].copy_from_slice(&payload_size.to_le_bytes());
        header
    }

    fn lookup_for(scheduler: Arc<dyn ReplicaScheduler + Send + Sync>) -> SchedulerLookup {
        Arc::new(move |model_id| (model_id == 7).then(|| scheduler.clone()))
    }

    #[cfg(feature = "hybrid")]
    #[test]
    fn shm_hybrid_memory_reads_validated_tensor_regions() {
        let name = format!("/kapsl-ipc-hybrid-memory-{}", std::process::id());
        let manager = match ShmManager::create(&name, 1024 * 1024) {
            Ok(manager) => Arc::new(manager),
            Err(error) => {
                eprintln!("Skipping shared memory test (mapping creation failed: {error})");
                return;
            }
        };
        let allocator =
            SharedShmAllocator::connect(manager.clone(), std::time::Duration::from_secs(30));
        let data = 3.0f32.to_ne_bytes();
        let header = TensorHeader {
            ndim: 2,
            dtype: 0,
            _padding: [0; 3],
            shape: [1, 1, 0, 0, 0, 0, 0, 0],
            data_size: data.len() as u64,
        };
        let header_size = std::mem::size_of::<TensorHeader>();
        let encoded_size = header_size + data.len();
        let lease = allocator
            .try_allocate(encoded_size)
            .expect("allocate input slot");
        unsafe {
            std::ptr::write_unaligned(
                manager.as_ptr().add(lease.offset()).cast::<TensorHeader>(),
                header,
            );
            std::ptr::copy_nonoverlapping(
                data.as_ptr(),
                manager.as_ptr().add(lease.offset() + header_size),
                data.len(),
            );
        }

        let adapter = ShmHybridMemory::new(manager);
        let packet = adapter
            .read_leased_tensor(HybridTensorLocation {
                offset: lease.offset() as u64,
                size: encoded_size as u64,
                lease: lease.token(),
            })
            .expect("read hybrid tensor");
        assert_eq!(packet.shape, vec![1, 1]);
        assert_eq!(packet.dtype, TensorDtype::Float32);
        assert_eq!(packet.data, data);

        let output = adapter.write_tensor(&packet).expect("write hybrid tensor");
        assert!(adapter
            .allocator
            .validate(output.offset as usize, output.size as usize, output.lease)
            .is_some());
        assert!(adapter
            .allocator
            .release_wire(output.offset as usize, output.lease));

        let before = adapter.allocator.snapshot().in_use_slots;
        let unsupported = BinaryTensorPacket {
            shape: vec![1],
            dtype: TensorDtype::Uint8,
            data: vec![1],
        };
        assert!(adapter.write_tensor(&unsupported).is_err());
        assert_eq!(adapter.allocator.snapshot().in_use_slots, before);
    }

    #[tokio::test]
    async fn plaintext_remote_policy_rejects_wire_before_payload_read_or_dispatch() {
        for (operation, payload_size, expected_kind) in [
            (OP_OPENAI_WIRE, 1024, std::io::ErrorKind::InvalidData),
            (
                OP_OPENAI_WIRE_STREAM,
                0,
                std::io::ErrorKind::PermissionDenied,
            ),
        ] {
            let lookup: SchedulerLookup =
                Arc::new(|_| panic!("remote plaintext wire request reached scheduler lookup"));
            let (mut client, server) = duplex(64);
            let task = tokio::spawn(handle_connection_with_wire_policy(
                server,
                lookup,
                None,
                Some(Arc::from("secret")),
                OpenAiWireTransportPolicy::PlaintextRemote,
            ));

            client
                .write_all(&request_header(7, operation, payload_size))
                .await
                .expect("write fixed request header only");
            let error = tokio::time::timeout(std::time::Duration::from_millis(200), task)
                .await
                .expect("remote wire request must be rejected from its header")
                .expect("connection task")
                .expect_err("remote plaintext wire request must fail");
            assert_eq!(error.kind(), expected_kind);
        }
    }

    #[tokio::test]
    async fn plaintext_remote_policy_preserves_authenticated_native_inference() {
        let seen = Arc::new(Mutex::new(Vec::new()));
        let scheduler: Arc<dyn ReplicaScheduler + Send + Sync> =
            Arc::new(PriorityRecordingScheduler { seen: seen.clone() });
        let (mut client, server) = duplex(4096);
        let task = tokio::spawn(handle_connection_with_wire_policy(
            server,
            lookup_for(scheduler),
            None,
            Some(Arc::from("secret")),
            OpenAiWireTransportPolicy::PlaintextRemote,
        ));
        let mut request = request_with_priority(1);
        request
            .metadata
            .as_mut()
            .expect("request metadata")
            .auth_token = Some("secret".to_string());

        let output = wire::infer_request_over_stream(&mut client, 7, &request)
            .await
            .expect("authenticated native inference remains available");
        assert_eq!(output.data, request.input.data);

        drop(client);
        task.await
            .expect("server task")
            .expect("connection handler");
        assert_eq!(*seen.lock().unwrap(), vec![Priority::Throughput]);
    }

    #[tokio::test]
    async fn transport_client_codec_round_trips_on_reused_connection() {
        let seen = Arc::new(Mutex::new(Vec::new()));
        let scheduler: Arc<dyn ReplicaScheduler + Send + Sync> =
            Arc::new(PriorityRecordingScheduler { seen: seen.clone() });
        let (mut client, server) = duplex(4096);
        let task = tokio::spawn(handle_connection(server, lookup_for(scheduler), None, None));
        let input = request_with_priority(1).input;

        for _ in 0..2 {
            let output =
                kapsl_transport::protocol::infer_over_stream(&mut client, 7, input.clone())
                    .await
                    .expect("inference round trip");
            assert_eq!(output.shape, input.shape);
            assert_eq!(output.dtype, input.dtype);
            assert_eq!(output.data, input.data);
        }

        drop(client);
        task.await
            .expect("server task")
            .expect("connection handler");
        assert_eq!(
            *seen.lock().unwrap(),
            vec![Priority::Throughput, Priority::Throughput]
        );
    }

    #[tokio::test]
    async fn hybrid_opcode_uses_injected_memory_adapter() {
        let input = BinaryTensorPacket {
            shape: vec![1],
            dtype: TensorDtype::Float32,
            data: 2.0f32.to_ne_bytes().to_vec(),
        };
        let memory = Arc::new(RecordingHybridMemory {
            input: input.clone(),
            reads: Mutex::new(Vec::new()),
            writes: Mutex::new(Vec::new()),
        });
        let scheduler: Arc<dyn ReplicaScheduler + Send + Sync> =
            Arc::new(PriorityRecordingScheduler {
                seen: Arc::new(Mutex::new(Vec::new())),
            });
        let (mut client, server) = duplex(4096);
        let hybrid_memory: Arc<dyn HybridMemory> = memory.clone();
        let task = tokio::spawn(handle_connection(
            server,
            lookup_for(scheduler),
            Some(hybrid_memory),
            None,
        ));
        let request = HybridRequest {
            metadata: kapsl_transport::RequestMetadata::new(44, 7, 1, false),
            shm_offset: 100,
            shm_size: 80,
            shm_lease: 55,
            protocol_version: HYBRID_SHM_PROTOCOL_VERSION,
        };

        wire::write_request_value(&mut client, 7, OP_HYBRID_INFER, &request)
            .await
            .expect("write hybrid request");
        let response = read_response(&mut client).await;
        assert_eq!(response.header.status, STATUS_OK);
        let response = response
            .deserialize::<HybridResponse>()
            .expect("decode hybrid response");
        assert_eq!(response.metadata.request_id, 44);
        assert_eq!(response.shm_offset, 900);
        assert_eq!(response.shm_size, 120);
        assert_eq!(response.shm_lease, 91);
        assert_eq!(response.protocol_version, HYBRID_SHM_PROTOCOL_VERSION);
        assert_eq!(*memory.reads.lock().unwrap(), vec![(100, 80)]);
        {
            let writes = memory.writes.lock().unwrap();
            assert_eq!(writes.len(), 1);
            assert_eq!(writes[0].shape, input.shape);
            assert_eq!(writes[0].dtype, input.dtype);
            assert_eq!(writes[0].data, input.data);
        }
        drop(client);
        task.await
            .expect("server task")
            .expect("connection handler");
    }

    #[tokio::test]
    async fn hybrid_opcode_rejects_stale_shared_memory_protocols() {
        let scheduler: Arc<dyn ReplicaScheduler + Send + Sync> =
            Arc::new(PriorityRecordingScheduler {
                seen: Arc::new(Mutex::new(Vec::new())),
            });
        let (mut client, server) = duplex(4096);
        let task = tokio::spawn(handle_connection(server, lookup_for(scheduler), None, None));
        let request = HybridRequest {
            metadata: kapsl_transport::RequestMetadata::new(45, 7, 1, false),
            shm_offset: 100,
            shm_size: 80,
            shm_lease: 55,
            protocol_version: HYBRID_SHM_PROTOCOL_VERSION - 1,
        };

        wire::write_request_value(&mut client, 7, OP_HYBRID_INFER, &request)
            .await
            .expect("write stale hybrid request");
        let response = read_response(&mut client).await;
        assert_eq!(response.header.status, STATUS_ERR);
        assert!(response
            .remote_error()
            .to_string()
            .contains("protocol version"));
        drop(client);
        task.await
            .expect("server task")
            .expect("connection handler");
    }

    #[tokio::test]
    async fn unary_codec_interop_reuses_connection_and_preserves_priority() {
        let seen = Arc::new(Mutex::new(Vec::new()));
        let scheduler: Arc<dyn ReplicaScheduler + Send + Sync> =
            Arc::new(PriorityRecordingScheduler { seen: seen.clone() });
        let (mut client, server) = duplex(4096);
        let task = tokio::spawn(handle_connection(server, lookup_for(scheduler), None, None));

        write_request(&mut client, 7, OP_INFER, &request_with_priority(0)).await;
        let response = read_response(&mut client).await;
        assert_eq!(response.header.status, STATUS_OK);
        response
            .deserialize::<BinaryTensorPacket>()
            .expect("response packet");

        write_request(&mut client, 7, OP_INFER, &request_with_priority(1)).await;
        let response = read_response(&mut client).await;
        assert_eq!(response.header.status, STATUS_OK);
        response
            .deserialize::<BinaryTensorPacket>()
            .expect("second response packet");

        drop(client);
        task.await
            .expect("server task")
            .expect("connection handler");
        assert_eq!(
            *seen.lock().unwrap(),
            vec![Priority::LatencyCritical, Priority::Throughput]
        );
    }

    #[tokio::test]
    async fn stream_uses_priority_stamped_by_delegating_scheduler() {
        let seen = Arc::new(Mutex::new(Vec::new()));
        let scheduler: Arc<dyn ReplicaScheduler + Send + Sync> =
            Arc::new(PriorityRecordingScheduler { seen: seen.clone() });
        let (mut client, server) = duplex(4096);
        let task = tokio::spawn(handle_connection(server, lookup_for(scheduler), None, None));

        write_request(&mut client, 7, OP_INFER_STREAM, &request_with_priority(1)).await;
        let response = read_response(&mut client).await;
        assert_eq!(response.header.status, STATUS_STREAM_CHUNK);
        response
            .deserialize::<BinaryTensorPacket>()
            .expect("stream packet");
        let response = read_response(&mut client).await;
        assert_eq!(response.header.status, STATUS_STREAM_END);
        assert!(response.payload.is_empty());

        drop(client);
        task.await
            .expect("server task")
            .expect("connection handler");
        assert_eq!(*seen.lock().unwrap(), vec![Priority::Throughput]);
    }

    #[tokio::test]
    async fn openai_wire_envelope_authenticates_before_dispatching_clean_policy_request() {
        use kapsl_transport::protocol::OpenAiWireStreamFrame;

        let seen = Arc::new(Mutex::new(Vec::new()));
        let scheduler: Arc<dyn ReplicaScheduler + Send + Sync> =
            Arc::new(WireRecordingScheduler { seen: seen.clone() });
        let (mut client, server) = duplex(16 * 1024);
        let task = tokio::spawn(handle_connection(
            server,
            lookup_for(scheduler),
            None,
            Some(Arc::from("secret")),
        ));

        let metadata = OpenAiWireMetadata {
            request_id: Some("request-42".to_string()),
            timeout_ms: Some(5000),
            priority: Some(0),
        };
        let unary = OpenAiWireRequest::new(
            OpenAiWireEndpoint::ChatCompletions,
            OpenAiWireFormat::Json,
            br#"{"model":"served","messages":[]}"#.to_vec(),
        )
        .with_session_id("session-42")
        .with_metadata(metadata.clone());
        let unauthenticated = wire::openai_wire_over_stream(&mut client, 7, &unary).await;
        assert!(matches!(
            unauthenticated,
            Err(CodecError::Remote(message)) if message == "Unauthorized"
        ));
        let response =
            wire::openai_wire_over_stream_authenticated(&mut client, 7, &unary, "secret")
                .await
                .expect("authenticated unary wire dispatch");
        assert_eq!(response.head.status, 201);
        assert_eq!(response.body, br#"{"ok":true}"#);

        let streaming = OpenAiWireRequest::new(
            OpenAiWireEndpoint::ChatCompletions,
            OpenAiWireFormat::ServerSentEvents,
            br#"{"model":"served","messages":[],"stream":true}"#.to_vec(),
        )
        .with_session_id("session-42")
        .with_metadata(OpenAiWireMetadata {
            priority: Some(1),
            ..metadata
        });
        wire::write_openai_wire_stream_request_authenticated(&mut client, 7, &streaming, "secret")
            .await
            .expect("write authenticated wire stream");
        assert!(matches!(
            wire::read_openai_wire_stream_frame(&mut client, DEFAULT_MAX_FRAME_PAYLOAD_BYTES)
                .await
                .unwrap(),
            OpenAiWireStreamFrame::Head(head) if head.status == 200
        ));
        assert_eq!(
            wire::read_openai_wire_stream_frame(&mut client, DEFAULT_MAX_FRAME_PAYLOAD_BYTES)
                .await
                .unwrap(),
            OpenAiWireStreamFrame::Chunk(b"data: {\"part\":".to_vec())
        );
        assert_eq!(
            wire::read_openai_wire_stream_frame(&mut client, DEFAULT_MAX_FRAME_PAYLOAD_BYTES)
                .await
                .unwrap(),
            OpenAiWireStreamFrame::Chunk(b"1}\n\ndata: [DONE]\n\n".to_vec())
        );
        assert!(matches!(
            wire::read_openai_wire_stream_frame(&mut client, DEFAULT_MAX_FRAME_PAYLOAD_BYTES)
                .await
                .unwrap(),
            OpenAiWireStreamFrame::End
        ));

        let seen = seen.lock().unwrap().clone();
        assert_eq!(seen.len(), 2);
        assert_eq!(seen[0].priority, Priority::LatencyCritical);
        assert_eq!(seen[0].metadata_priority, Some(0));
        assert_eq!(seen[0].session_id.as_deref(), Some("session-42"));
        assert_eq!(seen[0].body, unary.body);
        assert_eq!(seen[1].priority, Priority::Throughput);
        assert_eq!(seen[1].metadata_priority, Some(1));
        assert_eq!(seen[1].session_id.as_deref(), Some("session-42"));
        assert_eq!(seen[1].body, streaming.body);

        drop(client);
        task.await
            .expect("server task")
            .expect("connection handler");
    }

    #[tokio::test]
    async fn dropping_wire_stream_client_cancels_idle_server_request_and_exits_handler() {
        use kapsl_transport::protocol::OpenAiWireStreamFrame;

        let cancellation = Arc::new(Mutex::new(None));
        let scheduler: Arc<dyn ReplicaScheduler + Send + Sync> =
            Arc::new(IdleWireStreamScheduler {
                cancellation: cancellation.clone(),
            });
        let (mut client, server) = duplex(4096);
        let task = tokio::spawn(handle_connection(server, lookup_for(scheduler), None, None));
        let request = OpenAiWireRequest::new(
            OpenAiWireEndpoint::ChatCompletions,
            OpenAiWireFormat::ServerSentEvents,
            br#"{"model":"served","messages":[],"stream":true}"#.to_vec(),
        );

        wire::write_openai_wire_stream_request(&mut client, 7, &request)
            .await
            .expect("write wire stream request");
        assert!(matches!(
            wire::read_openai_wire_stream_frame(&mut client, DEFAULT_MAX_FRAME_PAYLOAD_BYTES)
                .await
                .expect("read wire response head"),
            OpenAiWireStreamFrame::Head(head) if head.status == 200
        ));
        let server_cancellation = cancellation
            .lock()
            .unwrap()
            .clone()
            .expect("server installs a local cancellation token");
        assert!(!server_cancellation.is_cancelled());

        drop(client);
        tokio::time::timeout(std::time::Duration::from_millis(500), task)
            .await
            .expect("disconnect should wake an idle stream handler")
            .expect("server task")
            .expect("connection handler exits cleanly");
        assert!(server_cancellation.is_cancelled());
    }

    #[tokio::test]
    async fn error_frame_does_not_poison_reused_connection() {
        let seen = Arc::new(Mutex::new(Vec::new()));
        let scheduler: Arc<dyn ReplicaScheduler + Send + Sync> =
            Arc::new(PriorityRecordingScheduler { seen: seen.clone() });
        let (mut client, server) = duplex(4096);
        let task = tokio::spawn(handle_connection(server, lookup_for(scheduler), None, None));

        write_request(&mut client, 999, OP_INFER, &request_with_priority(0)).await;
        let response = read_response(&mut client).await;
        assert_eq!(response.header.status, STATUS_ERR);
        assert_eq!(
            std::str::from_utf8(&response.payload).expect("UTF-8 error"),
            "Model 999 not found"
        );

        write_request(&mut client, 7, OP_INFER, &request_with_priority(1)).await;
        let response = read_response(&mut client).await;
        assert_eq!(response.header.status, STATUS_OK);
        response
            .deserialize::<BinaryTensorPacket>()
            .expect("response after error");

        drop(client);
        task.await
            .expect("server task")
            .expect("connection handler");
        assert_eq!(*seen.lock().unwrap(), vec![Priority::Throughput]);
    }

    #[tokio::test]
    async fn exported_tcp_client_interoperates_and_preserves_auth_metadata() {
        let seen = Arc::new(Mutex::new(Vec::new()));
        let scheduler: Arc<dyn ReplicaScheduler + Send + Sync> =
            Arc::new(PriorityRecordingScheduler { seen: seen.clone() });
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind test TCP listener");
        let port = listener.local_addr().expect("listener address").port();
        let lookup = lookup_for(scheduler);
        let task = tokio::spawn(async move {
            let (stream, _) = listener.accept().await.expect("accept test client");
            handle_connection(stream, lookup, None, Some(Arc::from("secret"))).await
        });

        let client = TcpClient::new("127.0.0.1".to_string(), port, PoolConfig::default());
        let mut request = request_with_priority(1);
        request
            .metadata
            .as_mut()
            .expect("request metadata")
            .auth_token = Some("secret".to_string());
        let output = client
            .infer_request(7, &request)
            .await
            .expect("authenticated TCP inference");
        assert_eq!(output.data, request.input.data);

        drop(client);
        task.abort();
        let _ = task.await;
        assert_eq!(*seen.lock().unwrap(), vec![Priority::Throughput]);
    }
}
