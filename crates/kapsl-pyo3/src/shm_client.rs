use kapsl_communication::shm::allocator::TieredShmAllocator;
use kapsl_communication::shm::memory::ShmManager;
use kapsl_communication::shm::ring_buffer::LockFreeRingBuffer;
use kapsl_communication::{RequestMetadata, ResponseMetadata};
use pyo3::prelude::*;
use std::sync::Arc;
use std::time::Duration;

use crate::shm_tensor::{
    create_tensor_allocator, next_request_id, parse_shm_dtype, read_error_message, read_tensor,
    stage_tensor, validate_region,
};

const SHM_QUEUE_CAPACITY: usize = 1024;
const RESPONSE_TIMEOUT: Duration = Duration::from_secs(5);
const RESPONSE_POLL_INTERVAL: Duration = Duration::from_millis(1);

/// Request entry in the shared memory queue
#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct ShmRequest {
    metadata: RequestMetadata,
    tensor_offset: u64,
    tensor_size: u64,
}

/// Response entry in the shared memory queue
#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct ShmResponse {
    metadata: ResponseMetadata,
    result_offset: u64,
    result_size: u64,
    error_offset: u64, // 0 if no error
}

/// PyO3 wrapper for shared memory client
#[pyclass]
pub struct KapslShmClient {
    shm: Arc<ShmManager>,
    allocator: Arc<TieredShmAllocator>,
    request_id_counter: u64,
}

#[pymethods]
#[allow(clippy::useless_conversion)]
impl KapslShmClient {
    #[new]
    fn new(shm_name: String) -> PyResult<Self> {
        let shm = ShmManager::connect(&shm_name)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyConnectionError, _>(e.to_string()))?;

        validate_region::<ShmRequest>(
            &shm,
            shm.request_queue_offset(),
            SHM_QUEUE_CAPACITY,
            "request queue",
        )
        .map_err(PyErr::from)?;
        validate_region::<ShmResponse>(
            &shm,
            shm.response_queue_offset(),
            SHM_QUEUE_CAPACITY,
            "response queue",
        )
        .map_err(PyErr::from)?;
        let allocator = create_tensor_allocator(&shm).map_err(PyErr::from)?;

        Ok(Self {
            shm: Arc::new(shm),
            allocator,
            request_id_counter: 1,
        })
    }

    #[pyo3(signature = (shape, dtype, data, *, model_id = 0))]
    fn infer(
        &mut self,
        py: Python<'_>,
        shape: Vec<i64>,
        dtype: String,
        data: Vec<u8>,
        model_id: u32,
    ) -> PyResult<Vec<u8>> {
        let request_id = next_request_id(&mut self.request_id_counter);
        let dtype = parse_shm_dtype(&dtype).map_err(PyErr::from)?;
        let staged =
            stage_tensor(&self.shm, &self.allocator, &shape, dtype, &data).map_err(PyErr::from)?;
        let metadata = RequestMetadata::new(request_id, model_id, 0, false);

        let request = ShmRequest {
            metadata,
            tensor_offset: staged.offset() as u64,
            tensor_size: staged.encoded_size() as u64,
        };

        // Push to request queue
        unsafe {
            let req_queue_offset = self.shm.request_queue_offset();
            let req_queue = LockFreeRingBuffer::<ShmRequest>::connect(
                self.shm.as_ptr().add(req_queue_offset) as *mut ShmRequest,
                SHM_QUEUE_CAPACITY,
            );

            req_queue
                .push(request)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        }

        // File-descriptor values are process-local and cannot be transferred by
        // writing the integer into shared memory. Polling is therefore used on
        // every platform, with the GIL released between checks.
        let start = std::time::Instant::now();
        loop {
            let response = unsafe {
                let resp_queue = LockFreeRingBuffer::<ShmResponse>::connect(
                    self.shm.as_ptr().add(self.shm.response_queue_offset()) as *mut ShmResponse,
                    SHM_QUEUE_CAPACITY,
                );
                resp_queue.pop()
            };

            if let Some(resp) = response {
                if resp.metadata.request_id == request_id {
                    if resp.metadata.status != 0 {
                        let error_msg = if resp.error_offset > 0 {
                            let error_offset =
                                usize::try_from(resp.error_offset).map_err(|_| {
                                    pyo3::exceptions::PyRuntimeError::new_err(
                                        "SHM error offset is too large",
                                    )
                                })?;
                            read_error_message(&self.shm, error_offset).map_err(PyErr::from)?
                        } else {
                            "Unknown error".to_string()
                        };
                        return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(error_msg));
                    }

                    let result_offset = usize::try_from(resp.result_offset).map_err(|_| {
                        pyo3::exceptions::PyRuntimeError::new_err("SHM result offset is too large")
                    })?;
                    let result_size = usize::try_from(resp.result_size).map_err(|_| {
                        pyo3::exceptions::PyRuntimeError::new_err("SHM result size is too large")
                    })?;
                    return read_tensor(&self.shm, result_offset, result_size)
                        .map_err(PyErr::from)
                        .map(|packet| packet.data);
                }
            }

            if start.elapsed() >= RESPONSE_TIMEOUT {
                return Err(PyErr::new::<pyo3::exceptions::PyTimeoutError, _>(
                    "Request timed out",
                ));
            }

            py.check_signals()?;
            py.detach(|| std::thread::sleep(RESPONSE_POLL_INTERVAL));
        }
    }
}
