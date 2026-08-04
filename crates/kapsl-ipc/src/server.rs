use crate::protocol::{
    HybridRequest, HybridResponse, RequestHeader, ResponseHeader, OP_HYBRID_INFER, OP_INFER,
    OP_INFER_STREAM, STATUS_ERR, STATUS_OK, STATUS_STREAM_CHUNK, STATUS_STREAM_END,
};
use async_trait::async_trait;
use bincode;
use kapsl_engine_api::{BinaryTensorPacket, InferenceRequest, NamedTensor, TensorDtype};
use kapsl_scheduler::{Priority, ReplicaScheduler};
use kapsl_transport::{ResponseMetadata, TransportError, TransportServer};
use serde::Deserialize;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};

#[cfg(unix)]
use std::os::unix::fs::PermissionsExt;
#[cfg(windows)]
use tokio::net::windows::named_pipe::ServerOptions;
#[cfg(unix)]
use tokio::net::{UnixListener, UnixStream};

use kapsl_shm::memory::{ShmManager, TensorHeader};

pub type SchedulerLookup =
    Arc<dyn Fn(u32) -> Option<Arc<dyn ReplicaScheduler + Send + Sync>> + Send + Sync>;

#[derive(Debug, Deserialize)]
struct LegacyInferenceRequestV1 {
    input: BinaryTensorPacket,
    #[serde(default)]
    additional_inputs: Vec<NamedTensor>,
    #[serde(default)]
    session_id: Option<String>,
}

fn check_auth(request: &InferenceRequest, expected: Option<&str>) -> Option<String> {
    let Some(expected_token) = expected else {
        return None; // auth not configured — allow all
    };
    let presented = request
        .metadata
        .as_ref()
        .and_then(|m| m.auth_token.as_deref());
    if presented != Some(expected_token) {
        Some("Unauthorized".to_string())
    } else {
        None
    }
}

fn request_priority(request: &InferenceRequest, default: Priority) -> Priority {
    match request
        .metadata
        .as_ref()
        .and_then(|metadata| metadata.priority)
    {
        Some(0) => Priority::LatencyCritical,
        Some(_) => Priority::Throughput,
        None => default,
    }
}

fn decode_inference_request(payload: &[u8]) -> Result<InferenceRequest, String> {
    match bincode::deserialize::<InferenceRequest>(payload) {
        Ok(request) => Ok(request),
        Err(primary_err) => {
            if let Ok(legacy) = bincode::deserialize::<LegacyInferenceRequestV1>(payload) {
                return Ok(InferenceRequest {
                    input: legacy.input,
                    additional_inputs: legacy.additional_inputs,
                    session_id: legacy.session_id,
                    metadata: None,
                    cancellation: None,
                });
            }
            Err(format!("Deserialization error: {}", primary_err))
        }
    }
}

pub struct IpcServer {
    socket_path: String,
    scheduler_lookup: SchedulerLookup,
    shm_manager: Option<Arc<ShmManager>>,
    auth_token: Option<Arc<str>>,
}

impl IpcServer {
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

    pub fn new_with_lookup(
        socket_path: &str,
        scheduler_lookup: SchedulerLookup,
        shm_manager: Option<Arc<ShmManager>>,
    ) -> Self {
        Self {
            socket_path: socket_path.to_string(),
            scheduler_lookup,
            shm_manager,
            auth_token: None,
        }
    }

    /// Require every inference request to carry this token in
    /// `request.metadata.auth_token`. Requests without the token
    /// or with a wrong token receive `STATUS_ERR: Unauthorized`.
    pub fn with_auth_token(mut self, token: impl Into<String>) -> Self {
        self.auth_token = Some(Arc::from(token.into().as_str()));
        self
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
                let shm_manager = self.shm_manager.clone();
                let auth_token = auth_token.clone();

                tokio::spawn(async move {
                    if let Err(e) =
                        handle_connection(stream, scheduler_lookup, shm_manager, auth_token).await
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
                let shm_manager = self.shm_manager.clone();
                let auth_token = auth_token.clone();

                tokio::spawn(async move {
                    if let Err(e) =
                        handle_connection(server, scheduler_lookup, shm_manager, auth_token).await
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

    fn transport_type(&self) -> &'static str {
        "socket"
    }
}

pub(crate) async fn handle_connection<T>(
    mut connection: T,
    scheduler_lookup: SchedulerLookup,
    shm_manager: Option<Arc<ShmManager>>,
    auth_token: Option<Arc<str>>,
) -> std::io::Result<()>
where
    T: AsyncRead + AsyncWrite + Unpin,
{
    loop {
        // Read header as raw bytes (not bincode)
        let mut model_id_buf = [0u8; 4];
        if connection.read_exact(&mut model_id_buf).await.is_err() {
            return Ok(()); // Connection closed
        }
        let mut op_code_buf = [0u8; 4];
        connection.read_exact(&mut op_code_buf).await?;
        let mut payload_size_buf = [0u8; 4];
        connection.read_exact(&mut payload_size_buf).await?;

        let header = RequestHeader {
            model_id: u32::from_le_bytes(model_id_buf),
            op_code: u32::from_le_bytes(op_code_buf),
            payload_size: u32::from_le_bytes(payload_size_buf),
        };

        // Read payload
        let mut payload = vec![0u8; header.payload_size as usize];
        connection.read_exact(&mut payload).await?;

        match header.op_code {
            OP_INFER_STREAM => {
                // Deserialize request
                let request: InferenceRequest = match decode_inference_request(&payload) {
                    Ok(req) => req,
                    Err(error_msg) => {
                        let resp_header = ResponseHeader {
                            status: STATUS_ERR,
                            payload_size: error_msg.len() as u32,
                        };
                        connection
                            .write_all(&resp_header.status.to_le_bytes())
                            .await?;
                        connection
                            .write_all(&resp_header.payload_size.to_le_bytes())
                            .await?;
                        connection.write_all(error_msg.as_bytes()).await?;
                        continue;
                    }
                };

                if let Some(error_msg) = check_auth(&request, auth_token.as_deref()) {
                    let resp_header = ResponseHeader {
                        status: STATUS_ERR,
                        payload_size: error_msg.len() as u32,
                    };
                    connection
                        .write_all(&resp_header.status.to_le_bytes())
                        .await?;
                    connection
                        .write_all(&resp_header.payload_size.to_le_bytes())
                        .await?;
                    connection.write_all(error_msg.as_bytes()).await?;
                    continue;
                }

                // Get scheduler for model
                let scheduler = match scheduler_lookup(header.model_id) {
                    Some(s) => s,
                    None => {
                        let error_msg = format!("Model {} not found", header.model_id);
                        let resp_header = ResponseHeader {
                            status: STATUS_ERR,
                            payload_size: error_msg.len() as u32,
                        };
                        connection
                            .write_all(&resp_header.status.to_le_bytes())
                            .await?;
                        connection
                            .write_all(&resp_header.payload_size.to_le_bytes())
                            .await?;
                        connection.write_all(error_msg.as_bytes()).await?;
                        continue;
                    }
                };

                // Execute streaming inference
                let priority = request_priority(&request, Priority::LatencyCritical);
                let stream_result = scheduler.infer_stream(request, priority, false).await;

                use futures::StreamExt;
                match stream_result {
                    Ok(mut inference_stream) => {
                        while let Some(result) = inference_stream.next().await {
                            match result {
                                Ok(packet) => {
                                    // Serialize packet
                                    let response_bytes = match bincode::serialize(&packet) {
                                        Ok(b) => b,
                                        Err(e) => {
                                            log::error!("Serialization error: {}", e);
                                            break;
                                        }
                                    };

                                    // Send chunk header
                                    let response_header = ResponseHeader {
                                        status: STATUS_STREAM_CHUNK,
                                        payload_size: response_bytes.len() as u32,
                                    };

                                    connection
                                        .write_all(&response_header.status.to_le_bytes())
                                        .await?;
                                    connection
                                        .write_all(&response_header.payload_size.to_le_bytes())
                                        .await?;
                                    connection.write_all(&response_bytes).await?;
                                    connection.flush().await?;
                                }
                                Err(e) => {
                                    // Send error frame and stop
                                    let error_msg = e.to_string();
                                    let response_bytes = error_msg.as_bytes();
                                    let response_header = ResponseHeader {
                                        status: STATUS_ERR,
                                        payload_size: response_bytes.len() as u32,
                                    };
                                    connection
                                        .write_all(&response_header.status.to_le_bytes())
                                        .await?;
                                    connection
                                        .write_all(&response_header.payload_size.to_le_bytes())
                                        .await?;
                                    connection.write_all(response_bytes).await?;
                                    connection.flush().await?;
                                    break;
                                }
                            }
                        }

                        // Send End of Stream frame
                        let response_header = ResponseHeader {
                            status: STATUS_STREAM_END,
                            payload_size: 0,
                        };
                        connection
                            .write_all(&response_header.status.to_le_bytes())
                            .await?;
                        connection
                            .write_all(&response_header.payload_size.to_le_bytes())
                            .await?;
                        connection.flush().await?;
                    }
                    Err(e) => {
                        // Send error frame for initial failure
                        let error_msg = e.to_string();
                        let response_bytes = error_msg.as_bytes();
                        let response_header = ResponseHeader {
                            status: STATUS_ERR,
                            payload_size: response_bytes.len() as u32,
                        };
                        connection
                            .write_all(&response_header.status.to_le_bytes())
                            .await?;
                        connection
                            .write_all(&response_header.payload_size.to_le_bytes())
                            .await?;
                        connection.write_all(response_bytes).await?;
                        connection.flush().await?;
                    }
                }
            }
            OP_INFER => {
                // Find scheduler for model_id
                if let Some(scheduler) = scheduler_lookup(header.model_id) {
                    // Deserialize payload to InferenceRequest
                    let request: InferenceRequest = match decode_inference_request(&payload) {
                        Ok(req) => req,
                        Err(error_msg) => {
                            let resp_header = ResponseHeader {
                                status: STATUS_ERR,
                                payload_size: error_msg.len() as u32,
                            };
                            connection
                                .write_all(&resp_header.status.to_le_bytes())
                                .await?;
                            connection
                                .write_all(&resp_header.payload_size.to_le_bytes())
                                .await?;
                            connection.write_all(error_msg.as_bytes()).await?;
                            continue;
                        }
                    };

                    if let Some(error_msg) = check_auth(&request, auth_token.as_deref()) {
                        let resp_header = ResponseHeader {
                            status: STATUS_ERR,
                            payload_size: error_msg.len() as u32,
                        };
                        connection
                            .write_all(&resp_header.status.to_le_bytes())
                            .await?;
                        connection
                            .write_all(&resp_header.payload_size.to_le_bytes())
                            .await?;
                        connection.write_all(error_msg.as_bytes()).await?;
                        continue;
                    }

                    // Process. Direct IPC clients default to throughput; delegated
                    // engines carry the parent scheduler's resolved priority.
                    let priority = request_priority(&request, Priority::Throughput);
                    let result = scheduler.infer(&request, priority, false).await;

                    match result {
                        Ok(output) => {
                            let output_bytes =
                                bincode::serialize(&output).map_err(std::io::Error::other)?;

                            let resp_header = ResponseHeader {
                                status: STATUS_OK,
                                payload_size: output_bytes.len() as u32,
                            };

                            // Write header as raw bytes (not bincode)
                            connection
                                .write_all(&resp_header.status.to_le_bytes())
                                .await?;
                            connection
                                .write_all(&resp_header.payload_size.to_le_bytes())
                                .await?;
                            connection.write_all(&output_bytes).await?;
                        }
                        Err(e) => {
                            let error_msg = e.to_string();
                            let resp_header = ResponseHeader {
                                status: STATUS_ERR,
                                payload_size: error_msg.len() as u32,
                            };
                            connection
                                .write_all(&resp_header.status.to_le_bytes())
                                .await?;
                            connection
                                .write_all(&resp_header.payload_size.to_le_bytes())
                                .await?;
                            connection.write_all(error_msg.as_bytes()).await?;
                        }
                    }
                } else {
                    // Model not found
                    let error_msg = format!("Model {} not found", header.model_id);
                    let resp_header = ResponseHeader {
                        status: STATUS_ERR,
                        payload_size: error_msg.len() as u32,
                    };
                    connection
                        .write_all(&resp_header.status.to_le_bytes())
                        .await?;
                    connection
                        .write_all(&resp_header.payload_size.to_le_bytes())
                        .await?;
                    connection.write_all(error_msg.as_bytes()).await?;
                }
            }
            OP_HYBRID_INFER => {
                // Payload already read at line 131-132, just deserialize it
                // Deserialize HybridRequest
                let hybrid_req: HybridRequest = bincode::deserialize(&payload)
                    .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;

                if let Some(shm_manager) = &shm_manager {
                    let base_ptr = shm_manager.as_ptr();

                    // Read TensorHeader from SHM
                    let header_ptr = unsafe {
                        base_ptr.add(hybrid_req.shm_offset as usize) as *const TensorHeader
                    };
                    let tensor_header = unsafe { &*header_ptr };

                    // Read tensor data
                    let data_ptr = unsafe {
                        base_ptr.add(
                            hybrid_req.shm_offset as usize + std::mem::size_of::<TensorHeader>(),
                        )
                    };
                    let data_slice = unsafe {
                        std::slice::from_raw_parts(data_ptr, tensor_header.data_size as usize)
                    };

                    // Build InferenceRequest
                    let shape = tensor_header.shape[0..tensor_header.ndim as usize].to_vec();
                    let dtype = match tensor_header.dtype {
                        0 => TensorDtype::Float32,
                        1 => TensorDtype::Float64,
                        2 => TensorDtype::Int32,
                        3 => TensorDtype::Int64,
                        _ => TensorDtype::Float32,
                    };

                    let packet = BinaryTensorPacket {
                        shape,
                        dtype,
                        data: data_slice.to_vec(),
                    };

                    let request = InferenceRequest {
                        input: packet,
                        additional_inputs: Vec::new(),
                        session_id: None,
                        metadata: None,
                        cancellation: None,
                    };

                    // Perform inference
                    let result =
                        if let Some(scheduler) = scheduler_lookup(hybrid_req.metadata.model_id) {
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
                        Ok(output) => {
                            // Serialize output to BinaryTensorPacket
                            let packet = BinaryTensorPacket {
                                shape: output.shape,
                                dtype: output.dtype,
                                data: output.data,
                            };

                            // Calculate required size
                            let output_size =
                                std::mem::size_of::<TensorHeader>() + packet.data.len();

                            // Allocate output slot with bounds checking
                            // Use smaller slots (1MB) and more of them (400 slots from 512MB to 912MB)
                            static SERVER_SLOT_COUNTER: std::sync::atomic::AtomicUsize =
                                std::sync::atomic::AtomicUsize::new(0);
                            let slot = SERVER_SLOT_COUNTER
                                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                            let output_offset = 512 * 1024 * 1024 + (slot % 400) * 1_000_000; // 1MB slots, 400 slots

                            // Bounds check
                            let shm_size = shm_manager.size();
                            if output_offset + output_size > shm_size {
                                let error_msg = format!("Output would exceed SHM bounds: offset={}, size={}, shm_size={}",
                                    output_offset, output_size, shm_size);
                                let resp_header = ResponseHeader {
                                    status: STATUS_ERR,
                                    payload_size: error_msg.len() as u32,
                                };
                                connection
                                    .write_all(&resp_header.status.to_le_bytes())
                                    .await?;
                                connection
                                    .write_all(&resp_header.payload_size.to_le_bytes())
                                    .await?;
                                connection.write_all(error_msg.as_bytes()).await?;
                                continue;
                            }

                            // Write result to SHM
                            // Re-acquire base_ptr to avoid holding !Send raw pointer across await
                            let base_ptr = shm_manager.as_ptr();
                            unsafe {
                                // Write header
                                let out_header = TensorHeader {
                                    ndim: packet.shape.len() as u32,
                                    dtype: match packet.dtype {
                                        TensorDtype::Float32 => 0,
                                        TensorDtype::Float64 => 1,
                                        TensorDtype::Int32 => 2,
                                        TensorDtype::Int64 => 3,
                                        _ => 0,
                                    },
                                    _padding: [0; 3],
                                    shape: {
                                        let mut arr = [0i64; 8];
                                        for (i, &v) in packet.shape.iter().enumerate() {
                                            arr[i] = v;
                                        }
                                        arr
                                    },
                                    data_size: packet.data.len() as u64,
                                };

                                let hdr_ptr = base_ptr.add(output_offset) as *mut TensorHeader;
                                std::ptr::write(hdr_ptr, out_header);

                                let data_ptr = base_ptr
                                    .add(output_offset + std::mem::size_of::<TensorHeader>());
                                std::ptr::copy_nonoverlapping(
                                    packet.data.as_ptr(),
                                    data_ptr,
                                    packet.data.len(),
                                );
                            }

                            // Send HybridResponse
                            let resp = HybridResponse {
                                metadata: ResponseMetadata {
                                    request_id: hybrid_req.metadata.request_id,
                                    status: STATUS_OK as u8,
                                    _padding: [0; 7],
                                    latency_ns: 0,
                                },
                                shm_offset: output_offset as u64,
                                shm_size: (std::mem::size_of::<TensorHeader>() + packet.data.len())
                                    as u64,
                            };

                            let resp_bytes =
                                bincode::serialize(&resp).map_err(std::io::Error::other)?;

                            let resp_header = ResponseHeader {
                                status: STATUS_OK,
                                payload_size: resp_bytes.len() as u32,
                            };

                            connection
                                .write_all(&resp_header.status.to_le_bytes())
                                .await?;
                            connection
                                .write_all(&resp_header.payload_size.to_le_bytes())
                                .await?;
                            connection.write_all(&resp_bytes).await?;
                        }
                        Err(e) => {
                            let error_msg = e.to_string();
                            let resp_header = ResponseHeader {
                                status: STATUS_ERR,
                                payload_size: error_msg.len() as u32,
                            };
                            connection
                                .write_all(&resp_header.status.to_le_bytes())
                                .await?;
                            connection
                                .write_all(&resp_header.payload_size.to_le_bytes())
                                .await?;
                            connection.write_all(error_msg.as_bytes()).await?;
                        }
                    }
                } else {
                    let error_msg = "SHM Manager not configured".to_string();
                    let resp_header = ResponseHeader {
                        status: STATUS_ERR,
                        payload_size: error_msg.len() as u32,
                    };
                    connection
                        .write_all(&resp_header.status.to_le_bytes())
                        .await?;
                    connection
                        .write_all(&resp_header.payload_size.to_le_bytes())
                        .await?;
                    connection.write_all(error_msg.as_bytes()).await?;
                }
            }
            _ => {
                // Unsupported op
                let resp_header = ResponseHeader {
                    status: STATUS_ERR,
                    payload_size: 0,
                };
                // Write header as raw bytes (not bincode)
                connection
                    .write_all(&resp_header.status.to_le_bytes())
                    .await?;
                connection
                    .write_all(&resp_header.payload_size.to_le_bytes())
                    .await?;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::Stream;
    use kapsl_engine_api::{EngineError, EngineMetrics, RequestMetadata, TensorDtype};
    use std::pin::Pin;
    use std::sync::Mutex;
    use tokio::io::{duplex, AsyncReadExt, AsyncWriteExt, DuplexStream};

    struct PriorityRecordingScheduler {
        seen: Arc<Mutex<Vec<Priority>>>,
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

    fn request_with_priority(priority: u8) -> InferenceRequest {
        let mut metadata = RequestMetadata::default();
        metadata.priority = Some(priority);
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
        let payload = bincode::serialize(request).expect("request serialization");
        client
            .write_all(&model_id.to_le_bytes())
            .await
            .expect("write model id");
        client
            .write_all(&op_code.to_le_bytes())
            .await
            .expect("write op code");
        client
            .write_all(&(payload.len() as u32).to_le_bytes())
            .await
            .expect("write payload size");
        client.write_all(&payload).await.expect("write payload");
    }

    async fn read_response(client: &mut DuplexStream) -> (u32, Vec<u8>) {
        let mut header = [0u8; 8];
        client
            .read_exact(&mut header)
            .await
            .expect("read response header");
        let status = u32::from_le_bytes(header[0..4].try_into().unwrap());
        let payload_size = u32::from_le_bytes(header[4..8].try_into().unwrap());
        let mut payload = vec![0u8; payload_size as usize];
        client
            .read_exact(&mut payload)
            .await
            .expect("read response payload");
        (status, payload)
    }

    fn lookup_for(scheduler: Arc<dyn ReplicaScheduler + Send + Sync>) -> SchedulerLookup {
        Arc::new(move |model_id| (model_id == 7).then(|| scheduler.clone()))
    }

    #[tokio::test]
    async fn infer_uses_priority_stamped_by_delegating_scheduler() {
        let seen = Arc::new(Mutex::new(Vec::new()));
        let scheduler: Arc<dyn ReplicaScheduler + Send + Sync> =
            Arc::new(PriorityRecordingScheduler { seen: seen.clone() });
        let (mut client, server) = duplex(4096);
        let task = tokio::spawn(handle_connection(server, lookup_for(scheduler), None, None));

        write_request(&mut client, 7, OP_INFER, &request_with_priority(0)).await;
        let (status, payload) = read_response(&mut client).await;
        assert_eq!(status, STATUS_OK);
        bincode::deserialize::<BinaryTensorPacket>(&payload).expect("response packet");

        drop(client);
        task.await
            .expect("server task")
            .expect("connection handler");
        assert_eq!(*seen.lock().unwrap(), vec![Priority::LatencyCritical]);
    }

    #[tokio::test]
    async fn stream_uses_priority_stamped_by_delegating_scheduler() {
        let seen = Arc::new(Mutex::new(Vec::new()));
        let scheduler: Arc<dyn ReplicaScheduler + Send + Sync> =
            Arc::new(PriorityRecordingScheduler { seen: seen.clone() });
        let (mut client, server) = duplex(4096);
        let task = tokio::spawn(handle_connection(server, lookup_for(scheduler), None, None));

        write_request(&mut client, 7, OP_INFER_STREAM, &request_with_priority(1)).await;
        let (status, payload) = read_response(&mut client).await;
        assert_eq!(status, STATUS_STREAM_CHUNK);
        bincode::deserialize::<BinaryTensorPacket>(&payload).expect("stream packet");
        let (status, payload) = read_response(&mut client).await;
        assert_eq!(status, STATUS_STREAM_END);
        assert!(payload.is_empty());

        drop(client);
        task.await
            .expect("server task")
            .expect("connection handler");
        assert_eq!(*seen.lock().unwrap(), vec![Priority::Throughput]);
    }
}
