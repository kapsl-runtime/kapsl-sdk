use crate::allocator::SharedShmAllocator;
use crate::mailbox::ResponseMailboxRegistry;
use crate::memory::{ShmManager, TensorHeader};
use crate::protocol::{ShmRequest, ShmResponse, SHM_PROTOCOL_VERSION, SHM_QUEUE_CAPACITY};
use crate::ring_buffer::LockFreeRingBuffer;
use async_trait::async_trait;
use kapsl_engine_api::{BinaryTensorPacket, InferenceRequest, TensorDtype};
use kapsl_scheduler::{Priority, ReplicaScheduler};
use kapsl_transport::{ResponseMetadata, TransportError, TransportServer};
use prometheus::Registry;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;
use tokio::time::{sleep, Duration};

mod metrics;
mod tensor_io;

use metrics::ShmPoolMetrics;
use tensor_io::*;

const TENSOR_SLOT_LEASE_TTL: Duration = Duration::from_secs(30);
const SHM_METRICS_REFRESH: Duration = Duration::from_secs(1);

/// Returns a processing mailbox to the registry if its Tokio task unwinds.
struct ProcessingMailboxGuard {
    mailboxes: ResponseMailboxRegistry,
    mailbox_index: u32,
    request_id: u64,
    completed: bool,
}

impl ProcessingMailboxGuard {
    fn new(mailboxes: ResponseMailboxRegistry, mailbox_index: u32, request_id: u64) -> Self {
        Self {
            mailboxes,
            mailbox_index,
            request_id,
            completed: false,
        }
    }

    fn complete(&mut self) {
        self.completed = true;
    }
}

impl Drop for ProcessingMailboxGuard {
    fn drop(&mut self) {
        if !self.completed
            && self
                .mailboxes
                .abandon_processing(self.mailbox_index, self.request_id)
        {
            log::warn!(
                "Released response mailbox {} after SHM request {} aborted",
                self.mailbox_index,
                self.request_id
            );
        }
    }
}

pub type SchedulerLookup =
    Arc<dyn Fn(u32) -> Option<Arc<dyn ReplicaScheduler + Send + Sync>> + Send + Sync>;
pub type SchedulerSnapshot =
    Arc<dyn Fn() -> HashMap<u32, Arc<dyn ReplicaScheduler + Send + Sync>> + Send + Sync>;

/// Shared-memory transport server backed by model schedulers.
///
/// The polling loop removes requests in queue order and dispatches each one in
/// its own Tokio task. Scheduler admission and backend capacity determine how
/// much inference work actually runs in parallel.
///
/// Each request owns a response mailbox and every tensor payload is protected
/// by a process-shared lease, so independent clients cannot consume or
/// overwrite one another's in-flight data.
pub struct ShmServer {
    shm_name: String,
    shm_size: usize,
    scheduler_lookup: SchedulerLookup,
    scheduler_snapshot: SchedulerSnapshot,
    metrics_registry: Option<Arc<Registry>>,
}

impl ShmServer {
    pub fn new(
        shm_name: &str,
        shm_size: usize,
        schedulers: HashMap<u32, Arc<dyn ReplicaScheduler + Send + Sync>>,
    ) -> Self {
        Self::new_with_registry(shm_name, shm_size, schedulers, None)
    }

    pub fn new_with_registry(
        shm_name: &str,
        shm_size: usize,
        schedulers: HashMap<u32, Arc<dyn ReplicaScheduler + Send + Sync>>,
        metrics_registry: Option<Arc<Registry>>,
    ) -> Self {
        let schedulers = Arc::new(schedulers);
        let scheduler_lookup: SchedulerLookup = {
            let schedulers = schedulers.clone();
            Arc::new(move |model_id| schedulers.get(&model_id).cloned())
        };
        let scheduler_snapshot: SchedulerSnapshot = Arc::new(move || schedulers.as_ref().clone());
        Self::new_with_lookup_and_registry(
            shm_name,
            shm_size,
            scheduler_lookup,
            scheduler_snapshot,
            metrics_registry,
        )
    }

    pub fn new_with_lookup(
        shm_name: &str,
        shm_size: usize,
        scheduler_lookup: SchedulerLookup,
        scheduler_snapshot: SchedulerSnapshot,
    ) -> Self {
        Self::new_with_lookup_and_registry(
            shm_name,
            shm_size,
            scheduler_lookup,
            scheduler_snapshot,
            None,
        )
    }

    pub fn new_with_lookup_and_registry(
        shm_name: &str,
        shm_size: usize,
        scheduler_lookup: SchedulerLookup,
        scheduler_snapshot: SchedulerSnapshot,
        metrics_registry: Option<Arc<Registry>>,
    ) -> Self {
        Self {
            shm_name: shm_name.to_string(),
            shm_size,
            scheduler_lookup,
            scheduler_snapshot,
            metrics_registry,
        }
    }

    /// Check if shared memory is available on this platform
    pub fn is_available() -> bool {
        cfg!(unix) || cfg!(windows)
    }

    async fn run_internal(&self) -> Result<(), TransportError> {
        // Create shared memory
        let shm = Arc::new(
            ShmManager::create(&self.shm_name, self.shm_size)
                .map_err(|e| TransportError::ServerError(e.to_string()))?,
        );

        let tensor_allocator = Arc::new(SharedShmAllocator::connect(
            shm.clone(),
            TENSOR_SLOT_LEASE_TTL,
        ));
        let response_mailboxes = ResponseMailboxRegistry::connect(shm.clone());
        let scheduler_lookup = self.scheduler_lookup.clone();
        let configured_models = (self.scheduler_snapshot)().len();
        let shm_pool_metrics = self.metrics_registry.as_ref().and_then(|registry| {
            match ShmPoolMetrics::register(registry) {
                Ok(metrics) => Some(Arc::new(metrics)),
                Err(e) => {
                    log::warn!("Failed to register SHM pool metrics: {}", e);
                    None
                }
            }
        });
        if let Some(metrics) = shm_pool_metrics.as_ref() {
            metrics.update_from_snapshot(tensor_allocator.snapshot());
        }
        log::info!(
            "SHM tensor pool configured: base={} layout=[{}] ttl_s={} mailboxes={} models={}",
            shm.tensor_pool_offset(),
            tensor_allocator.layout_summary(),
            TENSOR_SLOT_LEASE_TTL.as_secs(),
            shm.response_mailbox_count(),
            configured_models,
        );

        // Responses use the per-request mailbox array initialized by ShmManager.
        let req_queue_offset = shm.request_queue_offset();

        log::info!("Request queue offset: {}", req_queue_offset);

        let request_queue = unsafe {
            // Initialize the queues in shared memory (only once).
            LockFreeRingBuffer::<ShmRequest>::new(
                shm.as_ptr().add(req_queue_offset) as *mut ShmRequest,
                SHM_QUEUE_CAPACITY,
            )
        };

        log::info!("Shared memory server running on '{}'", self.shm_name);
        log::info!("Shared memory server listening on /{}", self.shm_name);
        log::info!("Starting request polling loop...");

        let mut poll_count = 0;
        let mut last_metrics_refresh = Instant::now();
        // Main server loop
        loop {
            if last_metrics_refresh.elapsed() >= SHM_METRICS_REFRESH {
                if let Some(metrics) = shm_pool_metrics.as_ref() {
                    metrics.update_from_snapshot(tensor_allocator.snapshot());
                }
                last_metrics_refresh = Instant::now();
            }

            // Poll request queue
            let request_opt = request_queue.pop();

            if poll_count % 10000 == 0 {
                log::debug!(
                    "Polled {} times, request: {:?}",
                    poll_count,
                    request_opt.is_some()
                );
            }
            poll_count += 1;

            if let Some(request) = request_opt {
                log::debug!("Received SHM request: {:?}", request);

                let scheduler_lookup = scheduler_lookup.clone();
                let shm = shm.clone();
                let tensor_allocator = tensor_allocator.clone();
                let response_mailboxes = response_mailboxes.clone();
                let shm_pool_metrics = shm_pool_metrics.clone();

                tokio::spawn(async move {
                    handle_request(
                        request,
                        shm,
                        tensor_allocator,
                        response_mailboxes,
                        scheduler_lookup,
                        shm_pool_metrics,
                    )
                    .await;
                });
            } else {
                // No requests, yield CPU briefly
                sleep(Duration::from_micros(10)).await;
            }
        }
    }
}

async fn handle_request(
    request: ShmRequest,
    shm: Arc<ShmManager>,
    tensor_allocator: Arc<SharedShmAllocator>,
    response_mailboxes: ResponseMailboxRegistry,
    scheduler_lookup: SchedulerLookup,
    shm_pool_metrics: Option<Arc<ShmPoolMetrics>>,
) {
    let request_id = request.metadata.request_id;
    if !response_mailboxes.begin_processing(request.response_mailbox, request_id) {
        let _ = tensor_allocator.release_wire(request.tensor_offset as usize, request.tensor_lease);
        log::warn!(
            "Rejected SHM request {} with unowned response mailbox {}",
            request_id,
            request.response_mailbox
        );
        return;
    }
    let mut processing_guard = ProcessingMailboxGuard::new(
        response_mailboxes.clone(),
        request.response_mailbox,
        request_id,
    );

    let start = Instant::now();
    if request.protocol_version != SHM_PROTOCOL_VERSION {
        let _ = tensor_allocator.release_wire(request.tensor_offset as usize, request.tensor_lease);
        if finish_request_with_error(
            shm.as_ref(),
            tensor_allocator.as_ref(),
            &response_mailboxes,
            request.response_mailbox,
            request_id,
            start,
            &format!(
                "Unsupported SHM protocol version {}; expected {}",
                request.protocol_version, SHM_PROTOCOL_VERSION
            ),
        ) {
            processing_guard.complete();
        }
        return;
    }

    let tensor = match take_request_tensor(shm.as_ref(), tensor_allocator.as_ref(), &request) {
        Ok(tensor) => tensor,
        Err(error) => {
            if finish_request_with_error(
                shm.as_ref(),
                tensor_allocator.as_ref(),
                &response_mailboxes,
                request.response_mailbox,
                request_id,
                start,
                &error,
            ) {
                processing_guard.complete();
            }
            return;
        }
    };

    let model_id = request.metadata.model_id;
    let Some(scheduler) = scheduler_lookup(model_id) else {
        if finish_request_with_error(
            shm.as_ref(),
            tensor_allocator.as_ref(),
            &response_mailboxes,
            request.response_mailbox,
            request_id,
            start,
            &format!("Model {model_id} not found"),
        ) {
            processing_guard.complete();
        }
        return;
    };
    let priority = if request.metadata.priority == 0 {
        Priority::LatencyCritical
    } else {
        Priority::Throughput
    };
    let inference_request = InferenceRequest {
        input: tensor,
        additional_inputs: Vec::new(),
        session_id: None,
        metadata: None,
        cancellation: None,
    };

    let response = match scheduler
        .infer(&inference_request, priority, request.metadata.force_cpu)
        .await
    {
        Ok(output) => match stage_response_tensor(shm.as_ref(), tensor_allocator.as_ref(), &output)
        {
            Ok((lease, result_size)) => ShmResponse {
                metadata: ResponseMetadata::success(request_id, start.elapsed().as_nanos() as u64),
                result_offset: lease.offset() as u64,
                result_size: result_size as u64,
                error_offset: 0,
                payload_lease: lease.token(),
            },
            Err(error) => {
                if let Some(metrics) = shm_pool_metrics.as_deref() {
                    metrics.on_exhausted();
                }
                error_response(
                    shm.as_ref(),
                    tensor_allocator.as_ref(),
                    request_id,
                    start.elapsed().as_nanos() as u64,
                    &error,
                )
            }
        },
        Err(error) => error_response(
            shm.as_ref(),
            tensor_allocator.as_ref(),
            request_id,
            start.elapsed().as_nanos() as u64,
            &error.to_string(),
        ),
    };

    if publish_or_release(
        tensor_allocator.as_ref(),
        &response_mailboxes,
        request.response_mailbox,
        request_id,
        response,
    ) {
        processing_guard.complete();
    }
    if let Some(metrics) = shm_pool_metrics.as_deref() {
        metrics.update_from_snapshot(tensor_allocator.snapshot());
    }
}

fn finish_request_with_error(
    shm: &ShmManager,
    allocator: &SharedShmAllocator,
    mailboxes: &ResponseMailboxRegistry,
    mailbox_index: u32,
    request_id: u64,
    started: Instant,
    message: &str,
) -> bool {
    let response = error_response(
        shm,
        allocator,
        request_id,
        started.elapsed().as_nanos() as u64,
        message,
    );
    publish_or_release(allocator, mailboxes, mailbox_index, request_id, response)
}

fn publish_or_release(
    allocator: &SharedShmAllocator,
    mailboxes: &ResponseMailboxRegistry,
    mailbox_index: u32,
    request_id: u64,
    response: ShmResponse,
) -> bool {
    if mailboxes.publish(mailbox_index, request_id, response) {
        return true;
    }

    let payload_offset = if response.error_offset > 0 {
        response.error_offset
    } else {
        response.result_offset
    };
    if payload_offset > 0 && response.payload_lease > 0 {
        let _ = allocator.release_wire(payload_offset as usize, response.payload_lease);
    }
    log::warn!(
        "Could not publish SHM response {} to mailbox {}",
        request_id,
        mailbox_index
    );
    false
}

#[async_trait]
impl TransportServer for ShmServer {
    async fn run(&self) -> Result<(), TransportError> {
        self.run_internal().await
    }

    async fn shutdown(&self) -> Result<(), TransportError> {
        // Shared memory will be cleaned up when the process exits
        Ok(())
    }
}

#[cfg(test)]
#[path = "tests.rs"]
mod tests;
