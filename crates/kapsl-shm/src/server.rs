use crate::allocator::{ShmAllocatorSnapshot, ShmPoolAllocator};
use crate::memory::{ShmManager, TensorHeader};
use crate::ring_buffer::LockFreeRingBuffer;
use async_trait::async_trait;
use kapsl_engine_api::{BinaryTensorPacket, InferenceRequest, TensorDtype};
use kapsl_scheduler::{Priority, ReplicaScheduler};
use kapsl_transport::{RequestMetadata, ResponseMetadata, TransportError, TransportServer};
use prometheus::{IntCounter, IntGauge, Opts, Registry};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;
use tokio::time::{sleep, Duration};

mod model_pool;
mod tensor_io;

#[cfg(test)]
use model_pool::{build_per_model_pool, derive_single_model_class_budgets};
use model_pool::{
    DynamicPerModelPool, DEFAULT_TENSOR_SLOT_LEASE_TTL_SECS, ERROR_LEN_PREFIX_BYTES,
    SHM_METRICS_REFRESH_SECS,
};
use tensor_io::*;

pub type SchedulerLookup =
    Arc<dyn Fn(u32) -> Option<Arc<dyn ReplicaScheduler + Send + Sync>> + Send + Sync>;
pub type SchedulerSnapshot =
    Arc<dyn Fn() -> HashMap<u32, Arc<dyn ReplicaScheduler + Send + Sync>> + Send + Sync>;

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

#[derive(Clone)]
struct ShmPoolMetrics {
    pool_in_use: IntGauge,
    pool_exhausted_total: IntCounter,
    pool_oldest_lease_ms: IntGauge,
}

impl ShmPoolMetrics {
    fn register(registry: &Arc<Registry>) -> Result<Self, prometheus::Error> {
        let pool_in_use = IntGauge::with_opts(Opts::new(
            "kapsl_shm_pool_in_use",
            "Number of currently leased SHM tensor slots",
        ))?;
        let pool_exhausted_total = IntCounter::with_opts(Opts::new(
            "kapsl_shm_pool_exhausted_total",
            "Total number of SHM tensor pool allocation failures",
        ))?;
        let pool_oldest_lease_ms = IntGauge::with_opts(Opts::new(
            "kapsl_shm_pool_oldest_lease_ms",
            "Age in milliseconds of the oldest active SHM slot lease",
        ))?;

        registry.register(Box::new(pool_in_use.clone()))?;
        registry.register(Box::new(pool_exhausted_total.clone()))?;
        registry.register(Box::new(pool_oldest_lease_ms.clone()))?;

        Ok(Self {
            pool_in_use,
            pool_exhausted_total,
            pool_oldest_lease_ms,
        })
    }

    fn update_from_snapshot(&self, snapshot: ShmAllocatorSnapshot) {
        self.pool_in_use
            .set(snapshot.in_use_slots.min(i64::MAX as usize) as i64);
        self.pool_oldest_lease_ms
            .set(snapshot.oldest_lease_ms.min(i64::MAX as u64) as i64);
    }

    fn on_exhausted(&self) {
        self.pool_exhausted_total.inc();
    }
}

/// Shared memory server implementing TransportServer
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

        let tensor_pool_bytes = shm.max_tensor_size();
        if tensor_pool_bytes <= std::mem::size_of::<TensorHeader>() {
            return Err(TransportError::ServerError(format!(
                "SHM tensor pool too small: {} bytes",
                tensor_pool_bytes
            )));
        }

        let lease_ttl = std::time::Duration::from_secs(DEFAULT_TENSOR_SLOT_LEASE_TTL_SECS);
        let initial_schedulers = (self.scheduler_snapshot)();
        let tensor_allocator = Arc::new(DynamicPerModelPool::new(
            &initial_schedulers,
            shm.tensor_pool_offset(),
            tensor_pool_bytes,
            lease_ttl,
        ));
        let scheduler_lookup = self.scheduler_lookup.clone();
        let scheduler_snapshot = self.scheduler_snapshot.clone();
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
            "SHM tensor pool configured (per-model sub-pools): base={} layout=[{}] ttl_s={}",
            shm.tensor_pool_offset(),
            tensor_allocator.layout_summary(),
            DEFAULT_TENSOR_SLOT_LEASE_TTL_SECS
        );

        // Initialize request and response queues
        let req_queue_offset = shm.request_queue_offset();
        let resp_queue_offset = shm.response_queue_offset();

        log::info!("Request queue offset: {}", req_queue_offset);
        log::info!("Response queue offset: {}", resp_queue_offset);

        unsafe {
            // Initialize the queues in shared memory (only once)
            LockFreeRingBuffer::<ShmRequest>::new(
                shm.as_ptr().add(req_queue_offset) as *mut ShmRequest,
                1024,
            );
            LockFreeRingBuffer::<ShmResponse>::new(
                shm.as_ptr().add(resp_queue_offset) as *mut ShmResponse,
                1024,
            );
        }

        log::info!("Shared memory server running on '{}'", self.shm_name);
        log::info!("Shared memory server listening on /{}", self.shm_name);
        log::info!("Starting request polling loop...");

        let mut poll_count = 0;
        let mut last_metrics_refresh = Instant::now();
        // Main server loop
        loop {
            if last_metrics_refresh.elapsed()
                >= std::time::Duration::from_secs(SHM_METRICS_REFRESH_SECS)
            {
                let schedulers = scheduler_snapshot();
                if let Some(layout) = tensor_allocator.refresh(&schedulers) {
                    log::info!(
                        "SHM scheduler registry changed; rebuilt per-model sub-pools: [{}]",
                        layout
                    );
                }
                if let Some(metrics) = shm_pool_metrics.as_ref() {
                    // Aggregate snapshot across all per-model sub-pools.
                    metrics.update_from_snapshot(tensor_allocator.snapshot());
                }
                last_metrics_refresh = Instant::now();
            }

            // Poll request queue
            let request_opt = unsafe {
                let req_queue: LockFreeRingBuffer<ShmRequest> = LockFreeRingBuffer::connect(
                    shm.as_ptr().add(req_queue_offset) as *mut ShmRequest,
                    1024,
                );
                req_queue.pop()
            };

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
                let shm_pool_metrics = shm_pool_metrics.clone();

                // Spawn task to handle request
                tokio::spawn(async move {
                    let start = Instant::now();

                    // Read tensor from shared memory
                    let tensor = unsafe {
                        read_tensor_from_shm(shm.as_ptr(), request.tensor_offset as usize)
                    };

                    // Process inference
                    let model_id = request.metadata.model_id;
                    if let Some(scheduler) = scheduler_lookup(model_id) {
                        let priority = if request.metadata.priority == 0 {
                            Priority::LatencyCritical
                        } else {
                            Priority::Throughput
                        };
                        let request_obj = InferenceRequest {
                            input: tensor,
                            additional_inputs: Vec::new(),
                            session_id: None, // SHM currently doesn't support session ID
                            metadata: None,
                            cancellation: None,
                        };
                        let result = scheduler
                            .infer(&request_obj, priority, request.metadata.force_cpu)
                            .await;

                        let latency_ns = start.elapsed().as_nanos() as u64;

                        match result {
                            Ok(output) => {
                                let result_size =
                                    std::mem::size_of::<TensorHeader>() + output.data.len();
                                let response = if let Some(result_offset) =
                                    allocate_pool_slot_for_model(
                                        tensor_allocator.as_ref(),
                                        model_id,
                                        result_size,
                                        shm_pool_metrics.as_deref(),
                                    ) {
                                    unsafe {
                                        write_tensor_to_shm(shm.as_ptr(), result_offset, &output);
                                    }
                                    ShmResponse {
                                        metadata: ResponseMetadata::success(
                                            request.metadata.request_id,
                                            latency_ns,
                                        ),
                                        result_offset: result_offset as u64,
                                        result_size: result_size as u64,
                                        error_offset: 0,
                                    }
                                } else {
                                    let msg = format!(
                                        "SHM tensor pool exhausted for model {} (required={} bytes, largest_slot={} bytes, layout={})",
                                        model_id,
                                        result_size,
                                        tensor_allocator.largest_slot_size_for_model(model_id),
                                        tensor_allocator.layout_summary()
                                    );
                                    log::warn!("{}", msg);
                                    error_response_for_model(
                                        shm.as_ptr(),
                                        tensor_allocator.as_ref(),
                                        model_id,
                                        request.metadata.request_id,
                                        latency_ns,
                                        shm_pool_metrics.as_deref(),
                                        &msg,
                                    )
                                };

                                unsafe {
                                    push_response_and_notify(shm.as_ref(), response);
                                }
                            }
                            Err(e) => {
                                let response = error_response_for_model(
                                    shm.as_ptr(),
                                    tensor_allocator.as_ref(),
                                    model_id,
                                    request.metadata.request_id,
                                    latency_ns,
                                    shm_pool_metrics.as_deref(),
                                    &e.to_string(),
                                );

                                unsafe {
                                    push_response_and_notify(shm.as_ref(), response);
                                }
                            }
                        }
                    } else {
                        let message = format!("Model {model_id} not found");
                        let response = error_response_for_model(
                            shm.as_ptr(),
                            tensor_allocator.as_ref(),
                            model_id,
                            request.metadata.request_id,
                            start.elapsed().as_nanos() as u64,
                            shm_pool_metrics.as_deref(),
                            &message,
                        );
                        unsafe {
                            push_response_and_notify(shm.as_ref(), response);
                        }
                    }
                });
            } else {
                // No requests, yield CPU briefly
                sleep(Duration::from_micros(10)).await;
            }
        }
    }
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
#[path = "server_tests.rs"]
mod server_tests;
