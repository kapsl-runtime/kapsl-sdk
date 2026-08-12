use crate::protocol::{
    HybridRequest, HybridResponse, OP_HYBRID_INFER, OP_INFER, OP_INFER_STREAM, STATUS_ERR,
    STATUS_OK, STATUS_STREAM_CHUNK, STATUS_STREAM_END,
};
use async_trait::async_trait;
use kapsl_engine_api::{BinaryTensorPacket, InferenceRequest, TensorDtype};
use kapsl_scheduler::{Priority, ReplicaScheduler};
use kapsl_transport::protocol::{
    asynchronous as wire, decode_inference_request, CodecError, DEFAULT_MAX_FRAME_PAYLOAD_BYTES,
};
use kapsl_transport::{ResponseMetadata, TransportError, TransportServer};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::io::{AsyncRead, AsyncWrite};

#[cfg(unix)]
use std::os::unix::fs::PermissionsExt;
#[cfg(windows)]
use tokio::net::windows::named_pipe::ServerOptions;
#[cfg(unix)]
use tokio::net::{UnixListener, UnixStream};

use kapsl_shm::memory::{ShmManager, TensorHeader};

pub type SchedulerLookup =
    Arc<dyn Fn(u32) -> Option<Arc<dyn ReplicaScheduler + Send + Sync>> + Send + Sync>;

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
        let Some(frame) =
            wire::read_request_frame_or_eof(&mut connection, DEFAULT_MAX_FRAME_PAYLOAD_BYTES)
                .await
                .map_err(codec_io)?
        else {
            return Ok(());
        };

        let model_id = frame.header.model_id;
        match frame.header.op_code {
            OP_INFER_STREAM => {
                let request = match decode_inference_request(&frame.payload) {
                    Ok(req) => req,
                    Err(error) => {
                        write_error_response(&mut connection, &inference_decode_message(error))
                            .await?;
                        continue;
                    }
                };

                if let Some(error_msg) = check_auth(&request, auth_token.as_deref()) {
                    write_error_response(&mut connection, &error_msg).await?;
                    continue;
                }

                let scheduler = match scheduler_lookup(model_id) {
                    Some(s) => s,
                    None => {
                        write_error_response(
                            &mut connection,
                            &format!("Model {model_id} not found"),
                        )
                        .await?;
                        continue;
                    }
                };

                let priority = request_priority(&request, Priority::LatencyCritical);
                let stream_result = scheduler.infer_stream(request, priority, false).await;

                use futures::StreamExt;
                match stream_result {
                    Ok(mut inference_stream) => {
                        while let Some(result) = inference_stream.next().await {
                            match result {
                                Ok(packet) => {
                                    wire::write_response_value(
                                        &mut connection,
                                        STATUS_STREAM_CHUNK,
                                        &packet,
                                    )
                                    .await
                                    .map_err(codec_io)?;
                                }
                                Err(e) => {
                                    write_error_response(&mut connection, &e.to_string()).await?;
                                    break;
                                }
                            }
                        }

                        wire::write_response_bytes(&mut connection, STATUS_STREAM_END, &[])
                            .await
                            .map_err(codec_io)?;
                    }
                    Err(e) => {
                        write_error_response(&mut connection, &e.to_string()).await?;
                    }
                }
            }
            OP_INFER => {
                if let Some(scheduler) = scheduler_lookup(model_id) {
                    let request = match decode_inference_request(&frame.payload) {
                        Ok(req) => req,
                        Err(error) => {
                            write_error_response(&mut connection, &inference_decode_message(error))
                                .await?;
                            continue;
                        }
                    };

                    if let Some(error_msg) = check_auth(&request, auth_token.as_deref()) {
                        write_error_response(&mut connection, &error_msg).await?;
                        continue;
                    }

                    let priority = request_priority(&request, Priority::Throughput);
                    let result = scheduler.infer(&request, priority, false).await;

                    match result {
                        Ok(output) => {
                            wire::write_response_value(&mut connection, STATUS_OK, &output)
                                .await
                                .map_err(codec_io)?;
                        }
                        Err(e) => {
                            write_error_response(&mut connection, &e.to_string()).await?;
                        }
                    }
                } else {
                    write_error_response(&mut connection, &format!("Model {model_id} not found"))
                        .await?;
                }
            }
            OP_HYBRID_INFER => {
                let hybrid_req: HybridRequest = frame.deserialize().map_err(codec_io)?;

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
                                write_error_response(&mut connection, &error_msg).await?;
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

                            wire::write_response_value(&mut connection, STATUS_OK, &resp)
                                .await
                                .map_err(codec_io)?;
                        }
                        Err(e) => {
                            write_error_response(&mut connection, &e.to_string()).await?;
                        }
                    }
                } else {
                    write_error_response(&mut connection, "SHM Manager not configured").await?;
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
    use kapsl_engine_api::{EngineError, EngineMetrics, RequestMetadata, TensorDtype};
    use kapsl_transport::connection_pool::PoolConfig;
    use kapsl_transport::protocol::ResponseFrame;
    use kapsl_transport::tcp::TcpClient;
    use std::pin::Pin;
    use std::sync::Mutex;
    use tokio::io::{duplex, DuplexStream};

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

    fn lookup_for(scheduler: Arc<dyn ReplicaScheduler + Send + Sync>) -> SchedulerLookup {
        Arc::new(move |model_id| (model_id == 7).then(|| scheduler.clone()))
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
