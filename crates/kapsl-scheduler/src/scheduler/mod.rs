use crate::gpu_executor::{GpuExecutor, WorkQueue};
use crate::mesh_routing::{MeshRouter, MeshRouterStats};
use crate::metrics::MetricsAccumulator;
use crate::priority::{stamp_engine_priority, stamp_openai_wire_priority, Priority};
use crate::request::Request;
use kapsl_engine_api::{
    BatchingMode, BinaryTensorPacket, CancellationToken, EngineError, EngineHandle,
    EngineModelInfo, InferenceRequest, OpenAiWireFormat, OpenAiWireRequest, OpenAiWireResponse,
    OpenAiWireStreamResponse,
};
use parking_lot::RwLock;
use std::sync::atomic::{AtomicUsize, Ordering};
use tokio::sync::oneshot;

use kapsl_hal::device_mesh::DeviceMesh;
use std::sync::Arc;
use std::time::{Duration, Instant};

mod admission;

use admission::{
    ActiveCountGuard, PriorityAdmission, PriorityAdmissionPermit, StreamAdmissionGuard,
    TrackedEngineStream, TrackedOpenAiWireStream,
};

/// Optional scheduler instrumentation hook. Implementations must remain cheap:
/// observations are emitted synchronously on request dispatch.
pub trait SchedulerObserver: Send + Sync {
    fn observe_queue_wait(&self, priority: Priority, operation: &'static str, elapsed: Duration);
}

pub(crate) type SharedSchedulerObserver = Arc<RwLock<Option<Arc<dyn SchedulerObserver>>>>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QueueOverflowPolicy {
    Block,
    DropNewest,
    DropOldest,
}

impl QueueOverflowPolicy {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Block => "block",
            Self::DropNewest => "drop_newest",
            Self::DropOldest => "drop_oldest",
        }
    }
}

/// Main scheduler that coordinates CPU and GPU execution
pub struct Scheduler {
    engines: Vec<EngineHandle>,
    cpu_pool: rayon::ThreadPool,
    // Vector of channels, one pair per worker
    gpu_high_priority_queues: Vec<WorkQueue>,
    gpu_low_priority_queues: Vec<WorkQueue>,
    // Maps each worker/queue index to the engine that owns that worker.
    worker_engine_indices: Vec<usize>,
    // Track active CPU inferences
    cpu_active_count: Arc<std::sync::atomic::AtomicUsize>,
    // Track in-flight GPU work (requests already dequeued from the channels, but not finished).
    gpu_in_flight_count: Arc<AtomicUsize>,
    // Track active GPU streams, which bypass the one-shot executor queue but
    // still need admission/backpressure accounting.
    gpu_stream_in_flight_count: Arc<AtomicUsize>,
    gpu_stream_admission: Arc<PriorityAdmission>,
    // Mesh router for topology-aware routing
    router: MeshRouter,
    max_micro_batch: usize,
    queue_overflow_policy: QueueOverflowPolicy,
    observer: SharedSchedulerObserver,
}

impl Drop for Scheduler {
    fn drop(&mut self) {
        // Executor tasks own engine handles. Closing their queues wakes the
        // tasks so model teardown can drop those handles and return backend
        // allocations to the runtime-owned device pool.
        for queue in self
            .gpu_high_priority_queues
            .iter()
            .chain(self.gpu_low_priority_queues.iter())
        {
            queue.close();
        }
        self.gpu_stream_admission.close();
    }
}

impl Scheduler {
    /// Builds a scheduler and starts one executor task per configured worker.
    ///
    /// The `_enable_fallback` argument is retained for source compatibility;
    /// replica failover is owned by [`crate::replica_pool::ReplicaPool`].
    ///
    /// # Panics
    ///
    /// Panics when no engines are supplied, `workers_per_device` is zero, the
    /// derived capacity overflows, the Rayon pool cannot be built, or this is
    /// called without an active Tokio runtime. A `cpu_workers` value of zero
    /// retains Rayon's automatic thread-count selection.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        engines: Vec<EngineHandle>,
        cpu_workers: usize,
        workers_per_device: usize,
        queue_size: usize,
        _enable_fallback: bool,
        max_micro_batch: usize,
        queue_delay_ms: u64,
        device_mesh: Option<Arc<DeviceMesh>>,
    ) -> Self {
        assert!(
            !engines.is_empty(),
            "Scheduler requires at least one engine"
        );
        assert!(
            workers_per_device > 0,
            "Scheduler requires at least one worker per device"
        );

        // Create CPU thread pool
        let cpu_pool = rayon::ThreadPoolBuilder::new()
            .num_threads(cpu_workers)
            .build()
            .expect("Failed to create CPU thread pool");

        let num_devices = engines.len();
        let total_workers = num_devices
            .checked_mul(workers_per_device)
            .expect("Scheduler worker count overflow");

        let mut gpu_high_priority_queues = Vec::with_capacity(total_workers);
        let mut gpu_low_priority_queues = Vec::with_capacity(total_workers);
        let mut worker_engine_indices = Vec::with_capacity(total_workers);
        let gpu_in_flight_count = Arc::new(AtomicUsize::new(0));
        let stream_capacity = total_workers
            .checked_mul(queue_size.max(1))
            .expect("Scheduler stream admission capacity overflow");
        let gpu_stream_admission = PriorityAdmission::new(stream_capacity);
        let observer: SharedSchedulerObserver = Arc::new(RwLock::new(None));

        for (engine_index, engine) in engines.iter().enumerate() {
            for _ in 0..workers_per_device {
                // Create GPU executor channels for this worker
                let high_queue = WorkQueue::new_observed(
                    queue_size,
                    Priority::LatencyCritical,
                    "translated",
                    observer.clone(),
                );
                let low_queue = WorkQueue::new_observed(
                    queue_size,
                    Priority::Throughput,
                    "translated",
                    observer.clone(),
                );

                gpu_high_priority_queues.push(high_queue.clone());
                gpu_low_priority_queues.push(low_queue.clone());
                worker_engine_indices.push(engine_index);

                // Spawn GPU executor for this worker
                let gpu_executor = GpuExecutor::new(
                    high_queue,
                    low_queue,
                    engine.clone(),
                    max_micro_batch,
                    queue_delay_ms,
                    gpu_in_flight_count.clone(),
                );
                tokio::spawn(gpu_executor.run());
            }
        }

        // Create mesh router for topology-aware routing
        let router = MeshRouter::new(device_mesh.clone(), total_workers);

        Self {
            engines,
            cpu_pool,
            gpu_high_priority_queues,
            gpu_low_priority_queues,
            worker_engine_indices,
            cpu_active_count: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
            gpu_in_flight_count,
            gpu_stream_in_flight_count: Arc::new(AtomicUsize::new(0)),
            gpu_stream_admission,
            router,
            max_micro_batch,
            queue_overflow_policy: QueueOverflowPolicy::Block,
            observer,
        }
    }

    pub fn with_observer(self, observer: Arc<dyn SchedulerObserver>) -> Self {
        *self.observer.write() = Some(observer);
        self
    }

    pub fn with_queue_overflow_policy(mut self, policy: QueueOverflowPolicy) -> Self {
        self.queue_overflow_policy = policy;
        self
    }

    pub fn queue_overflow_policy(&self) -> QueueOverflowPolicy {
        self.queue_overflow_policy
    }

    /// Get worker index using topology-aware routing
    fn get_worker_index(&self, session_id: &Option<String>) -> usize {
        self.router.route(session_id, None)
    }

    fn engine_for_worker(&self, worker_index: usize) -> EngineHandle {
        let engine_index = self.worker_engine_indices[worker_index];
        self.engines[engine_index].clone()
    }

    fn try_reserve_cpu_slot(&self) -> bool {
        let capacity = self.cpu_pool.current_num_threads();
        self.cpu_active_count
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |active| {
                (active < capacity).then_some(active + 1)
            })
            .is_ok()
    }

    fn dispatch_cpu(&self, request: Request, slot_reserved: bool) {
        let engine_idx = self.get_worker_index(&None) % self.engines.len();
        let engine = self.engines[engine_idx].clone();
        if !slot_reserved {
            self.cpu_active_count.fetch_add(1, Ordering::AcqRel);
        }
        let cpu_active_count = self.cpu_active_count.clone();

        self.cpu_pool.spawn(move || {
            let _active = ActiveCountGuard {
                counter: cpu_active_count,
            };
            let result = if request
                .input
                .cancellation
                .as_ref()
                .is_some_and(CancellationToken::is_cancelled)
            {
                Err(EngineError::cancelled("Request cancelled"))
            } else {
                engine.infer(&request.input)
            };
            let _ = request.response_tx.send(result);
        });
    }

    async fn acquire_gpu_stream_permit(
        &self,
        priority: Priority,
        cancellation: Option<&CancellationToken>,
    ) -> Result<PriorityAdmissionPermit, EngineError> {
        self.gpu_stream_admission
            .acquire(priority, self.queue_overflow_policy, cancellation)
            .await
    }

    fn observe_queue_wait(&self, priority: Priority, operation: &'static str, started: Instant) {
        if let Some(observer) = self.observer.read().as_ref() {
            observer.observe_queue_wait(priority, operation, started.elapsed());
        }
    }

    /// Get mesh routing statistics
    pub fn mesh_stats(&self) -> Option<MeshRouterStats> {
        self.router.mesh_stats()
    }

    /// Get the device mesh if available
    pub fn device_mesh(&self) -> Option<Arc<DeviceMesh>> {
        self.router.device_mesh()
    }

    pub async fn infer(
        &self,
        input: InferenceRequest,
        priority: Priority,
        force_cpu: bool,
    ) -> Result<BinaryTensorPacket, EngineError> {
        if input
            .cancellation
            .as_ref()
            .is_some_and(CancellationToken::is_cancelled)
        {
            return Err(EngineError::cancelled("Request cancelled"));
        }
        let (response_tx, response_rx) = oneshot::channel();

        // Determine worker index before moving input
        let worker_idx = if !force_cpu {
            self.get_worker_index(&input.session_id)
        } else {
            0
        };

        let request = Request { input, response_tx };

        if force_cpu {
            self.dispatch_cpu(request, false);

            response_rx
                .await
                .map_err(|_| EngineError::overloaded("Scheduler dropped request".to_string()))?
        } else {
            // Execute on GPU via priority queue
            let queue = match priority {
                Priority::LatencyCritical => &self.gpu_high_priority_queues[worker_idx],
                Priority::Throughput => &self.gpu_low_priority_queues[worker_idx],
            };

            match self.queue_overflow_policy {
                QueueOverflowPolicy::Block => {
                    queue.push_block(request).await;
                }
                QueueOverflowPolicy::DropNewest => {
                    if queue.try_push_drop_newest(request).is_err() {
                        return Err(EngineError::overloaded(format!(
                            "GPU queue full (policy={})",
                            self.queue_overflow_policy.as_str()
                        )));
                    }
                }
                QueueOverflowPolicy::DropOldest => {
                    if let Some(dropped) = queue.push_drop_oldest(request) {
                        let _ = dropped.response_tx.send(Err(EngineError::overloaded(
                            "GPU queue full: dropped oldest request (policy=drop_oldest)"
                                .to_string(),
                        )));
                    }
                }
            }

            response_rx.await.map_err(|_| EngineError::InferenceError {
                reason: "GPU execution failed".to_string(),
                source: None,
            })?
        }
    }

    /// Non-blocking infer: returns `Err(EngineError::overloaded)` immediately if
    /// the target queue (GPU) is full or the CPU pool is saturated, instead of
    /// waiting for capacity. Intended for background / cron callers that should
    /// skip a firing rather than blocking the async executor.
    pub async fn try_infer(
        &self,
        input: InferenceRequest,
        priority: Priority,
        force_cpu: bool,
    ) -> Result<BinaryTensorPacket, EngineError> {
        if force_cpu {
            if input
                .cancellation
                .as_ref()
                .is_some_and(CancellationToken::is_cancelled)
            {
                return Err(EngineError::cancelled("Request cancelled"));
            }
            // Atomically reserve capacity so concurrent non-blocking callers
            // cannot all observe the same free worker and overfill Rayon.
            if !self.try_reserve_cpu_slot() {
                return Err(EngineError::overloaded("CPU pool saturated".to_string()));
            }
            let (response_tx, response_rx) = oneshot::channel();
            self.dispatch_cpu(Request { input, response_tx }, true);
            return response_rx
                .await
                .map_err(|_| EngineError::overloaded("Scheduler dropped request".to_string()))?;
        }

        if input
            .cancellation
            .as_ref()
            .is_some_and(CancellationToken::is_cancelled)
        {
            return Err(EngineError::cancelled("Request cancelled"));
        }

        // GPU path: attempt a non-blocking push.
        let (response_tx, response_rx) = oneshot::channel();
        let worker_idx = self.get_worker_index(&input.session_id);
        let request = Request { input, response_tx };

        let queue = match priority {
            Priority::LatencyCritical => &self.gpu_high_priority_queues[worker_idx],
            Priority::Throughput => &self.gpu_low_priority_queues[worker_idx],
        };

        queue
            .try_push_drop_newest(request)
            .map_err(|_| EngineError::overloaded("GPU queue full".to_string()))?;

        response_rx.await.map_err(|_| EngineError::InferenceError {
            reason: "GPU execution failed".to_string(),
            source: None,
        })?
    }
    pub fn get_queue_depth(&self) -> (usize, usize) {
        let cpu_depth = self
            .cpu_active_count
            .load(std::sync::atomic::Ordering::Relaxed);

        let mut gpu_total = 0usize;
        for (high_queue, low_queue) in self
            .gpu_high_priority_queues
            .iter()
            .zip(self.gpu_low_priority_queues.iter())
        {
            gpu_total = gpu_total.saturating_add(high_queue.len());
            gpu_total = gpu_total.saturating_add(low_queue.len());
        }
        gpu_total = gpu_total.saturating_add(
            self.gpu_in_flight_count
                .load(std::sync::atomic::Ordering::Relaxed),
        );
        gpu_total = gpu_total.saturating_add(
            self.gpu_stream_in_flight_count
                .load(std::sync::atomic::Ordering::Relaxed),
        );
        (cpu_depth, gpu_total)
    }

    pub fn is_healthy(&self) -> bool {
        for (high_queue, low_queue) in self
            .gpu_high_priority_queues
            .iter()
            .zip(self.gpu_low_priority_queues.iter())
        {
            let high_capacity = high_queue.capacity();
            let low_capacity = low_queue.capacity();
            let high_depth = high_queue.len();
            let low_depth = low_queue.len();

            if high_capacity > 0
                && low_capacity > 0
                && ((high_depth as f64 / high_capacity as f64) >= 0.8
                    || (low_depth as f64 / low_capacity as f64) >= 0.8)
            {
                return false;
            }
        }
        true
    }
}

#[async_trait::async_trait]
impl crate::replica_pool::ReplicaScheduler for Scheduler {
    fn get_queue_depth(&self) -> (usize, usize) {
        self.get_queue_depth()
    }

    fn is_healthy(&self) -> bool {
        self.is_healthy()
    }

    fn get_metrics(&self) -> kapsl_engine_api::EngineMetrics {
        let mut aggregate = MetricsAccumulator::default();
        let mut delegated_capacity = 0usize;
        let mut has_delegated_capacity = false;

        for engine in &self.engines {
            let metrics = engine.metrics();
            aggregate.add(&metrics);
            let batching = engine.batching_policy();
            if matches!(
                batching.mode,
                BatchingMode::Continuous | BatchingMode::Delegated
            ) {
                has_delegated_capacity = true;
                delegated_capacity = delegated_capacity.saturating_add(batching.max_requests);
            }
        }

        let (cpu_q, gpu_q) = self.get_queue_depth();
        let mut metrics = aggregate.finish();
        metrics.batch_size = if has_delegated_capacity {
            delegated_capacity.max(1)
        } else {
            self.max_micro_batch
        };
        metrics.queue_depth = cpu_q.saturating_add(gpu_q);
        metrics
    }

    fn model_info(&self) -> Option<EngineModelInfo> {
        self.engines.iter().find_map(|engine| engine.model_info())
    }

    async fn infer(
        &self,
        request: &InferenceRequest,
        priority: Priority,
        force_cpu: bool,
    ) -> Result<BinaryTensorPacket, EngineError> {
        self.infer(request.clone(), priority, force_cpu).await
    }

    async fn infer_openai_wire(
        &self,
        mut request: OpenAiWireRequest,
        priority: Priority,
        force_cpu: bool,
    ) -> Result<OpenAiWireResponse, EngineError> {
        if request.format != OpenAiWireFormat::Json {
            return Err(EngineError::invalid_input(
                "non-streaming OpenAI wire inference requires JSON format",
            ));
        }
        request.validate(usize::MAX)?;
        if request
            .cancellation
            .as_ref()
            .is_some_and(|token| token.is_cancelled())
        {
            return Err(EngineError::cancelled("Request cancelled"));
        }
        if !self.is_healthy() {
            return Err(EngineError::overloaded("Scheduler overloaded".to_string()));
        }

        let worker_idx = if force_cpu {
            0
        } else {
            self.get_worker_index(&request.session_id)
        };
        let engine = self.engine_for_worker(worker_idx);
        if !engine.supports_openai_wire() {
            return Err(EngineError::backend(
                "selected engine does not support protocol-native OpenAI requests",
            ));
        }
        let batching_policy = engine.batching_policy();
        if matches!(
            batching_policy.mode,
            BatchingMode::Continuous | BatchingMode::Delegated
        ) && batching_policy.supports_priority
        {
            stamp_openai_wire_priority(&mut request, priority);
        }

        let queue_started = Instant::now();
        let (counter, permit) = if force_cpu {
            (self.cpu_active_count.clone(), None)
        } else {
            (
                self.gpu_stream_in_flight_count.clone(),
                Some(
                    self.acquire_gpu_stream_permit(priority, request.cancellation.as_ref())
                        .await?,
                ),
            )
        };
        self.observe_queue_wait(priority, "wire", queue_started);
        if request
            .cancellation
            .as_ref()
            .is_some_and(|token| token.is_cancelled())
        {
            return Err(EngineError::cancelled(
                "Request cancelled while awaiting admission",
            ));
        }
        let _guard = StreamAdmissionGuard::new(counter, permit);
        let response = engine.infer_openai_wire(&request).await?;
        response.head.validate()?;
        Ok(response)
    }

    async fn infer_openai_wire_stream(
        &self,
        mut request: OpenAiWireRequest,
        priority: Priority,
        force_cpu: bool,
    ) -> Result<OpenAiWireStreamResponse, EngineError> {
        if request.format != OpenAiWireFormat::ServerSentEvents {
            return Err(EngineError::invalid_input(
                "streaming OpenAI wire inference requires SSE format",
            ));
        }
        request.validate(usize::MAX)?;
        if request
            .cancellation
            .as_ref()
            .is_some_and(|token| token.is_cancelled())
        {
            return Err(EngineError::cancelled("Request cancelled"));
        }
        if !self.is_healthy() {
            return Err(EngineError::overloaded("Scheduler overloaded".to_string()));
        }

        let worker_idx = if force_cpu {
            0
        } else {
            self.get_worker_index(&request.session_id)
        };
        let engine = self.engine_for_worker(worker_idx);
        if !engine.supports_openai_wire() {
            return Err(EngineError::backend(
                "selected engine does not support protocol-native OpenAI streams",
            ));
        }
        let batching_policy = engine.batching_policy();
        if matches!(
            batching_policy.mode,
            BatchingMode::Continuous | BatchingMode::Delegated
        ) && batching_policy.supports_priority
        {
            stamp_openai_wire_priority(&mut request, priority);
        }

        let queue_started = Instant::now();
        let (counter, permit) = if force_cpu {
            (self.cpu_active_count.clone(), None)
        } else {
            (
                self.gpu_stream_in_flight_count.clone(),
                Some(
                    self.acquire_gpu_stream_permit(priority, request.cancellation.as_ref())
                        .await?,
                ),
            )
        };
        self.observe_queue_wait(priority, "wire", queue_started);
        if request
            .cancellation
            .as_ref()
            .is_some_and(|token| token.is_cancelled())
        {
            return Err(EngineError::cancelled(
                "Request cancelled while awaiting admission",
            ));
        }
        let guard = StreamAdmissionGuard::new(counter, permit);
        let response = engine.infer_openai_wire_stream(&request).await?;
        response.head.validate()?;
        Ok(OpenAiWireStreamResponse {
            head: response.head,
            body: Box::pin(TrackedOpenAiWireStream {
                inner: response.body,
                _guard: guard,
            }),
        })
    }

    async fn infer_stream(
        &self,
        mut request: InferenceRequest,
        priority: Priority,
        force_cpu: bool,
    ) -> Result<
        std::pin::Pin<
            Box<dyn futures::Stream<Item = Result<BinaryTensorPacket, EngineError>> + Send>,
        >,
        EngineError,
    > {
        if request
            .cancellation
            .as_ref()
            .is_some_and(|token| token.is_cancelled())
        {
            return Err(EngineError::cancelled("Request cancelled"));
        }

        if !self.is_healthy() {
            return Err(EngineError::overloaded("Scheduler overloaded".to_string()));
        }

        let worker_idx = if !force_cpu {
            self.get_worker_index(&request.session_id)
        } else {
            0
        };

        let engine = self.engine_for_worker(worker_idx);

        let batching_policy = engine.batching_policy();
        if matches!(
            batching_policy.mode,
            BatchingMode::Continuous | BatchingMode::Delegated
        ) && batching_policy.supports_priority
        {
            stamp_engine_priority(&mut request, priority);
        }

        let (counter, permit) = if force_cpu {
            (self.cpu_active_count.clone(), None)
        } else {
            (
                self.gpu_stream_in_flight_count.clone(),
                Some(
                    self.acquire_gpu_stream_permit(priority, request.cancellation.as_ref())
                        .await?,
                ),
            )
        };
        if request
            .cancellation
            .as_ref()
            .is_some_and(|token| token.is_cancelled())
        {
            return Err(EngineError::cancelled(
                "Request cancelled while awaiting admission",
            ));
        }
        let guard = StreamAdmissionGuard::new(counter, permit);
        let stream = engine.infer_stream(&request);
        Ok(Box::pin(TrackedEngineStream {
            inner: stream,
            _guard: guard,
        }))
    }
}

#[cfg(test)]
mod tests;
