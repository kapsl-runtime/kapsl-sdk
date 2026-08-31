use kapsl_communication::ipc::protocol::{
    HybridRequest, HybridResponse, HYBRID_SHM_PROTOCOL_VERSION,
};
use kapsl_communication::shm::allocator::SharedShmAllocator;
use kapsl_communication::shm::memory::ShmManager;
use kapsl_communication::transport::protocol::{
    asynchronous, DEFAULT_MAX_FRAME_PAYLOAD_BYTES, OP_HYBRID_INFER, STATUS_OK,
};
use kapsl_communication::RequestMetadata;
use kapsl_engine_api::TensorDtype;
use pyo3::prelude::*;
use std::path::PathBuf;
use std::sync::Arc;
#[cfg(windows)]
use tokio::net::windows::named_pipe::{ClientOptions, NamedPipeClient, PipeMode};
#[cfg(unix)]
use tokio::net::UnixStream;
use tokio::runtime::{Builder, Runtime};
use tokio::sync::Mutex;

use crate::shm_tensor::{
    create_tensor_allocator, parse_shm_dtype, read_tensor, stage_tensor, WireLeaseReleaseGuard,
};

#[pyclass]
pub struct KapslHybridClient {
    shm: Arc<ShmManager>,
    allocator: Arc<SharedShmAllocator>,
    socket_path: PathBuf,
    #[cfg(unix)]
    streams: Mutex<Vec<UnixStream>>,
    #[cfg(windows)]
    streams: Mutex<Vec<NamedPipeClient>>,
    rt: Runtime,
}

#[pymethods]
#[allow(clippy::useless_conversion)]
impl KapslHybridClient {
    #[new]
    fn new(shm_name: String, socket_path: String) -> PyResult<Self> {
        let shm = ShmManager::connect(&shm_name)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyConnectionError, _>(e.to_string()))?;

        let shm_arc = Arc::new(shm);

        let allocator = create_tensor_allocator(&shm_arc).map_err(PyErr::from)?;
        let rt = Builder::new_multi_thread()
            .worker_threads(2)
            .enable_io()
            .build()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        Ok(Self {
            shm: shm_arc,
            allocator,
            socket_path: PathBuf::from(socket_path),
            streams: Mutex::new(Vec::new()),
            rt,
        })
    }

    #[pyo3(signature = (shape, dtype, data, *, model_id = 0))]
    fn infer(
        &self,
        py: Python<'_>,
        shape: Vec<i64>,
        dtype: String,
        data: Vec<u8>,
        model_id: u32,
    ) -> PyResult<Vec<u8>> {
        let request_id = self.shm.next_request_id();
        let dtype: TensorDtype = parse_shm_dtype(&dtype).map_err(PyErr::from)?;
        let mut staged =
            stage_tensor(&self.shm, &self.allocator, &shape, dtype, &data).map_err(PyErr::from)?;
        let metadata = RequestMetadata::new(request_id, model_id, 0, false);

        let request = HybridRequest {
            metadata,
            shm_offset: staged.offset() as u64,
            shm_size: staged.encoded_size() as u64,
            shm_lease: staged.lease_token(),
            protocol_version: HYBRID_SHM_PROTOCOL_VERSION,
        };
        // Once the control-plane write is attempted, only the server may free
        // the input. A failed connection leaves a bounded lease that the
        // process-shared allocator can reclaim after its TTL.
        staged.transfer_to_server();

        let runtime = &self.rt;
        let stream_pool = &self.streams;
        let socket_path = &self.socket_path;
        let response = py
            .detach(|| {
                runtime.block_on(async {
                    let mut stream = if let Some(stream) = stream_pool.lock().await.pop() {
                        stream
                    } else {
                        #[cfg(unix)]
                        let stream = UnixStream::connect(socket_path).await?;
                        #[cfg(windows)]
                        let stream = ClientOptions::new()
                            .pipe_mode(PipeMode::Byte)
                            .open(socket_path)
                            .map_err(|e| {
                                std::io::Error::new(
                                    std::io::ErrorKind::ConnectionRefused,
                                    format!(
                                        "Failed to open named pipe '{}': {}",
                                        socket_path.display(),
                                        e
                                    ),
                                )
                            })?;

                        stream
                    };
                    let result = async {
                        asynchronous::write_request_value(
                            &mut stream,
                            0,
                            OP_HYBRID_INFER,
                            &request,
                        )
                        .await
                        .map_err(std::io::Error::other)?;

                        let response = asynchronous::read_response_frame(
                            &mut stream,
                            DEFAULT_MAX_FRAME_PAYLOAD_BYTES,
                        )
                        .await
                        .map_err(std::io::Error::other)?;
                        if response.header.status != STATUS_OK {
                            return Err(std::io::Error::other(response.remote_error()));
                        }
                        response
                            .deserialize::<HybridResponse>()
                            .map_err(std::io::Error::other)
                    }
                    .await;
                    if result.is_ok() {
                        stream_pool.lock().await.push(stream);
                    }
                    result
                })
            })
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        let _response_lease = WireLeaseReleaseGuard::new(
            self.allocator.as_ref(),
            response.shm_offset,
            response.shm_lease,
        );
        if response.protocol_version != HYBRID_SHM_PROTOCOL_VERSION {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                "Unsupported hybrid SHM response version {}; expected {}",
                response.protocol_version, HYBRID_SHM_PROTOCOL_VERSION
            )));
        }
        let result_offset = usize::try_from(response.shm_offset)
            .map_err(|_| pyo3::exceptions::PyRuntimeError::new_err("SHM offset is too large"))?;
        let result_size = usize::try_from(response.shm_size)
            .map_err(|_| pyo3::exceptions::PyRuntimeError::new_err("SHM size is too large"))?;
        read_tensor(
            &self.shm,
            &self.allocator,
            result_offset,
            result_size,
            response.shm_lease,
        )
        .map_err(PyErr::from)
        .map(|packet| packet.data)
    }
}
