use kapsl_communication::shm::allocator::SharedShmAllocator;
use kapsl_communication::shm::mailbox::ResponseMailboxRegistry;
use kapsl_communication::shm::memory::ShmManager;
use kapsl_communication::shm::protocol::{ShmRequest, SHM_PROTOCOL_VERSION, SHM_QUEUE_CAPACITY};
use kapsl_communication::shm::ring_buffer::LockFreeRingBuffer;
use kapsl_communication::RequestMetadata;
use pyo3::prelude::*;
use std::sync::Arc;
use std::time::Duration;

use crate::shm_tensor::{
    create_tensor_allocator, parse_shm_dtype, read_error_message, read_tensor, stage_tensor,
    validate_region, WireLeaseReleaseGuard,
};

const RESPONSE_TIMEOUT: Duration = Duration::from_secs(5);
const RESPONSE_POLL_INTERVAL: Duration = Duration::from_millis(1);

/// PyO3 wrapper for shared memory client
#[pyclass]
pub struct KapslShmClient {
    shm: Arc<ShmManager>,
    allocator: Arc<SharedShmAllocator>,
    response_mailboxes: ResponseMailboxRegistry,
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
        let shm = Arc::new(shm);
        let allocator = create_tensor_allocator(&shm).map_err(PyErr::from)?;
        let response_mailboxes = ResponseMailboxRegistry::connect(shm.clone());

        Ok(Self {
            shm,
            allocator,
            response_mailboxes,
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
        let dtype = parse_shm_dtype(&dtype).map_err(PyErr::from)?;
        let mut staged =
            stage_tensor(&self.shm, &self.allocator, &shape, dtype, &data).map_err(PyErr::from)?;
        let mailbox = self.response_mailboxes.claim(request_id).ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("All direct-SHM response mailboxes are busy")
        })?;
        let metadata = RequestMetadata::new(request_id, model_id, 0, false);

        let request = ShmRequest {
            metadata,
            tensor_offset: staged.offset() as u64,
            tensor_size: staged.encoded_size() as u64,
            tensor_lease: staged.lease_token(),
            response_mailbox: mailbox.index(),
            protocol_version: SHM_PROTOCOL_VERSION,
            _padding: [0; 2],
        };

        // Push to request queue
        unsafe {
            let req_queue_offset = self.shm.request_queue_offset();
            let req_queue = LockFreeRingBuffer::<ShmRequest>::connect(
                self.shm.as_ptr().add(req_queue_offset) as *mut ShmRequest,
                SHM_QUEUE_CAPACITY,
            );

            if let Err(error) = req_queue.push(request) {
                let _ = self.response_mailboxes.abort(mailbox);
                return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                    error.to_string(),
                ));
            }
        }
        staged.transfer_to_server();

        // File-descriptor values are process-local and cannot be transferred by
        // writing the integer into shared memory. Polling is therefore used on
        // every platform, with the GIL released between checks.
        let start = std::time::Instant::now();
        loop {
            let response = self.response_mailboxes.try_take(mailbox);

            if let Some(resp) = response {
                let _ = self.response_mailboxes.release(mailbox);
                let payload_offset = if resp.error_offset > 0 {
                    resp.error_offset
                } else {
                    resp.result_offset
                };
                let _payload_lease = WireLeaseReleaseGuard::new(
                    self.allocator.as_ref(),
                    payload_offset,
                    resp.payload_lease,
                );
                if resp.metadata.status != 0 {
                    let error_msg = if resp.error_offset > 0 {
                        let error_offset = usize::try_from(resp.error_offset).map_err(|_| {
                            pyo3::exceptions::PyRuntimeError::new_err(
                                "SHM error offset is too large",
                            )
                        })?;
                        read_error_message(
                            &self.shm,
                            &self.allocator,
                            error_offset,
                            resp.payload_lease,
                        )
                        .map_err(PyErr::from)?
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
                return read_tensor(
                    &self.shm,
                    &self.allocator,
                    result_offset,
                    result_size,
                    resp.payload_lease,
                )
                .map_err(PyErr::from)
                .map(|packet| packet.data);
            }

            if start.elapsed() >= RESPONSE_TIMEOUT {
                // This succeeds only while the server has not yet accepted the
                // request. Once processing starts, the server retains mailbox
                // ownership and will publish or reclaim it safely.
                let _ = self.response_mailboxes.abort(mailbox);
                return Err(PyErr::new::<pyo3::exceptions::PyTimeoutError, _>(
                    "Request timed out",
                ));
            }

            if let Err(error) = py.check_signals() {
                let _ = self.response_mailboxes.abort(mailbox);
                return Err(error);
            }
            py.detach(|| std::thread::sleep(RESPONSE_POLL_INTERVAL));
        }
    }
}
