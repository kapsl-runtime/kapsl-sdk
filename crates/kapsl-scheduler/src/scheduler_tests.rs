use super::*;
use crate::replica_pool::ReplicaScheduler;
use crate::request::Request;
use crate::scheduler::QueueOverflowPolicy;
use async_trait::async_trait;
use kapsl_engine_api::{
    BatchingPolicy, BinaryTensorPacket, CancellationToken, Engine, EngineError, EngineMetrics,
    EngineStream, InferenceRequest, OpenAiWireEndpoint, OpenAiWireFormat, OpenAiWireMetadata,
    OpenAiWireRequest, OpenAiWireResponse, OpenAiWireResponseHead, OpenAiWireStreamResponse,
    TensorDtype,
};
use std::sync::atomic::AtomicUsize;
use std::sync::Arc;
use tokio::sync::oneshot;

struct MockEngine {
    metrics: EngineMetrics,
}

impl MockEngine {
    fn new(metrics: EngineMetrics) -> Self {
        Self { metrics }
    }
}

#[async_trait]
impl Engine for MockEngine {
    async fn load(&mut self, _model_path: &std::path::Path) -> Result<(), EngineError> {
        Ok(())
    }

    fn infer(&self, request: &InferenceRequest) -> Result<BinaryTensorPacket, EngineError> {
        Ok(request.input.clone())
    }

    fn infer_stream(&self, request: &InferenceRequest) -> EngineStream {
        let output = Ok(request.input.clone());
        Box::pin(futures::stream::once(async move { output }))
    }

    fn unload(&mut self) {}

    fn metrics(&self) -> EngineMetrics {
        self.metrics.clone()
    }

    fn health_check(&self) -> Result<(), EngineError> {
        Ok(())
    }
}

/// Engine that advertises batching (`max_batch() > 1`) and records the size of
/// every dispatched call, so tests can assert the executor coalesced requests.
struct BatchRecordingEngine {
    calls: Arc<std::sync::Mutex<Vec<usize>>>,
    max_batch: usize,
    self_batches: bool,
    policy_queue_delay_ms: Option<u64>,
}

impl BatchRecordingEngine {
    fn new(calls: Arc<std::sync::Mutex<Vec<usize>>>, max_batch: usize) -> Self {
        Self {
            calls,
            max_batch,
            self_batches: false,
            policy_queue_delay_ms: None,
        }
    }

    fn new_with_policy_delay(
        calls: Arc<std::sync::Mutex<Vec<usize>>>,
        max_batch: usize,
        queue_delay_ms: u64,
    ) -> Self {
        Self {
            calls,
            max_batch,
            self_batches: false,
            policy_queue_delay_ms: Some(queue_delay_ms),
        }
    }

    /// A backend that advertises batching capacity yet self-batches internally,
    /// so the executor must dispatch each request individually anyway.
    fn new_self_batching(calls: Arc<std::sync::Mutex<Vec<usize>>>, max_batch: usize) -> Self {
        Self {
            calls,
            max_batch,
            self_batches: true,
            policy_queue_delay_ms: None,
        }
    }
}

#[async_trait]
impl Engine for BatchRecordingEngine {
    async fn load(&mut self, _model_path: &std::path::Path) -> Result<(), EngineError> {
        Ok(())
    }

    fn infer(&self, request: &InferenceRequest) -> Result<BinaryTensorPacket, EngineError> {
        self.calls.lock().unwrap().push(1);
        Ok(request.input.clone())
    }

    fn infer_batch(
        &self,
        requests: &[InferenceRequest],
    ) -> Result<Vec<BinaryTensorPacket>, EngineError> {
        self.calls.lock().unwrap().push(requests.len());
        Ok(requests.iter().map(|req| req.input.clone()).collect())
    }

    fn max_batch(&self) -> usize {
        self.max_batch
    }

    fn self_batches(&self) -> bool {
        self.self_batches
    }

    fn batching_policy(&self) -> BatchingPolicy {
        let policy = BatchingPolicy::from_legacy(self.max_batch, self.self_batches);
        match self.policy_queue_delay_ms {
            Some(queue_delay_ms) => policy.with_queue_delay_ms(queue_delay_ms),
            None => policy,
        }
    }

    fn infer_stream(&self, request: &InferenceRequest) -> EngineStream {
        let output = Ok(request.input.clone());
        Box::pin(futures::stream::once(async move { output }))
    }

    fn unload(&mut self) {}

    fn metrics(&self) -> EngineMetrics {
        EngineMetrics::default()
    }

    fn health_check(&self) -> Result<(), EngineError> {
        Ok(())
    }
}

/// Self-batching engine whose `infer` blocks until released and whose reported
/// KV occupancy is toggleable, so tests can drive the executor's
/// occupancy-driven admission gate deterministically.
struct GatedEngine {
    started: Arc<AtomicUsize>,
    release: Arc<(std::sync::Mutex<bool>, std::sync::Condvar)>,
    saturated: Arc<std::sync::atomic::AtomicBool>,
    kv_total: usize,
}

impl GatedEngine {
    fn new(kv_total: usize) -> Self {
        Self {
            started: Arc::new(AtomicUsize::new(0)),
            release: Arc::new((std::sync::Mutex::new(false), std::sync::Condvar::new())),
            saturated: Arc::new(std::sync::atomic::AtomicBool::new(true)),
            kv_total,
        }
    }

    fn release_all(&self) {
        let (lock, cvar) = &*self.release;
        *lock.lock().unwrap() = true;
        cvar.notify_all();
    }
}

#[async_trait]
impl Engine for GatedEngine {
    async fn load(&mut self, _model_path: &std::path::Path) -> Result<(), EngineError> {
        Ok(())
    }

    fn infer(&self, request: &InferenceRequest) -> Result<BinaryTensorPacket, EngineError> {
        self.started
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        let (lock, cvar) = &*self.release;
        let mut released = lock.lock().unwrap();
        while !*released {
            released = cvar.wait(released).unwrap();
        }
        Ok(request.input.clone())
    }

    fn self_batches(&self) -> bool {
        true
    }

    fn infer_stream(&self, request: &InferenceRequest) -> EngineStream {
        let output = Ok(request.input.clone());
        Box::pin(futures::stream::once(async move { output }))
    }

    fn unload(&mut self) {}

    fn metrics(&self) -> EngineMetrics {
        let free = if self.saturated.load(std::sync::atomic::Ordering::SeqCst) {
            0
        } else {
            self.kv_total
        };
        EngineMetrics {
            kv_cache_blocks_total: self.kv_total,
            kv_cache_blocks_free: free,
            ..EngineMetrics::default()
        }
    }

    fn health_check(&self) -> Result<(), EngineError> {
        Ok(())
    }
}

/// Self-batching engine that records the `metadata.priority` seen on infer and
/// stream calls, so tests can assert schedulers stamp the resolved queue
/// priority before handing work to an internal batcher.
struct PriorityRecordingEngine {
    seen: Arc<std::sync::Mutex<Vec<Option<u8>>>>,
    policy: BatchingPolicy,
}

impl PriorityRecordingEngine {
    fn continuous(seen: Arc<std::sync::Mutex<Vec<Option<u8>>>>) -> Self {
        Self {
            seen,
            policy: BatchingPolicy::continuous(8).with_priority_support(),
        }
    }

    fn delegated(seen: Arc<std::sync::Mutex<Vec<Option<u8>>>>) -> Self {
        Self {
            seen,
            policy: BatchingPolicy::delegated().with_priority_support(),
        }
    }
}

#[async_trait]
impl Engine for PriorityRecordingEngine {
    async fn load(&mut self, _model_path: &std::path::Path) -> Result<(), EngineError> {
        Ok(())
    }

    fn infer(&self, request: &InferenceRequest) -> Result<BinaryTensorPacket, EngineError> {
        let priority = request.metadata.as_ref().and_then(|m| m.priority);
        self.seen.lock().unwrap().push(priority);
        Ok(request.input.clone())
    }

    fn infer_batch(
        &self,
        _requests: &[InferenceRequest],
    ) -> Result<Vec<BinaryTensorPacket>, EngineError> {
        panic!("priority-aware self/delegated backend must not be request-coalesced")
    }

    fn self_batches(&self) -> bool {
        true
    }

    fn supports_openai_wire(&self) -> bool {
        true
    }

    async fn infer_openai_wire(
        &self,
        request: &OpenAiWireRequest,
    ) -> Result<OpenAiWireResponse, EngineError> {
        self.seen.lock().unwrap().push(
            request
                .metadata
                .as_ref()
                .and_then(|metadata| metadata.priority),
        );
        Ok(OpenAiWireResponse {
            head: OpenAiWireResponseHead::new(200, Vec::new())?,
            body: request.body.clone(),
        })
    }

    async fn infer_openai_wire_stream(
        &self,
        request: &OpenAiWireRequest,
    ) -> Result<OpenAiWireStreamResponse, EngineError> {
        self.seen.lock().unwrap().push(
            request
                .metadata
                .as_ref()
                .and_then(|metadata| metadata.priority),
        );
        Ok(OpenAiWireStreamResponse {
            head: OpenAiWireResponseHead::new(200, Vec::new())?,
            body: Box::pin(futures::stream::iter(vec![
                Ok(b"data: one\n\n".to_vec()),
                Ok(b"data: [DONE]\n\n".to_vec()),
            ])),
        })
    }

    fn batching_policy(&self) -> BatchingPolicy {
        self.policy
    }

    fn infer_stream(&self, request: &InferenceRequest) -> EngineStream {
        let priority = request.metadata.as_ref().and_then(|m| m.priority);
        self.seen.lock().unwrap().push(priority);
        let output = Ok(request.input.clone());
        Box::pin(futures::stream::once(async move { output }))
    }

    fn unload(&mut self) {}

    fn metrics(&self) -> EngineMetrics {
        EngineMetrics::default()
    }

    fn health_check(&self) -> Result<(), EngineError> {
        Ok(())
    }
}

struct PendingStreamEngine;

#[async_trait]
impl Engine for PendingStreamEngine {
    async fn load(&mut self, _model_path: &std::path::Path) -> Result<(), EngineError> {
        Ok(())
    }

    fn infer(&self, request: &InferenceRequest) -> Result<BinaryTensorPacket, EngineError> {
        Ok(request.input.clone())
    }

    fn infer_stream(&self, _request: &InferenceRequest) -> EngineStream {
        Box::pin(futures::stream::pending::<
            Result<BinaryTensorPacket, EngineError>,
        >())
    }

    fn unload(&mut self) {}

    fn metrics(&self) -> EngineMetrics {
        EngineMetrics::default()
    }

    fn health_check(&self) -> Result<(), EngineError> {
        Ok(())
    }
}

fn make_inference_request(session_id: Option<&str>) -> InferenceRequest {
    let input = BinaryTensorPacket {
        shape: vec![1],
        dtype: TensorDtype::Float32,
        data: vec![0, 0, 0, 0],
    };
    let request = InferenceRequest::new(input);
    match session_id {
        Some(id) => request.with_session_id(id),
        None => request,
    }
}

fn make_openai_wire_request(format: OpenAiWireFormat) -> OpenAiWireRequest {
    OpenAiWireRequest::new(
        OpenAiWireEndpoint::ChatCompletions,
        format,
        br#"{"model":"test","messages":[]}"#.to_vec(),
    )
    .with_session_id("wire-session")
    .with_metadata(OpenAiWireMetadata {
        request_id: Some("wire-request".to_string()),
        ..Default::default()
    })
}

fn make_request(session_id: Option<&str>) -> Request {
    let (response_tx, _response_rx) = oneshot::channel();
    Request {
        input: make_inference_request(session_id),
        response_tx,
    }
}

fn build_scheduler_for_queue_tests(
    engines: Vec<EngineHandle>,
    queue_size: usize,
    cpu_active: usize,
) -> Scheduler {
    let cpu_pool = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .expect("Failed to create CPU thread pool");

    let high_queue = crate::gpu_executor::WorkQueue::new(queue_size);
    let low_queue = crate::gpu_executor::WorkQueue::new(queue_size);

    use crate::mesh_routing::MeshRouter;

    Scheduler {
        engines: engines.clone(),
        cpu_pool,
        gpu_high_priority_queues: vec![high_queue],
        gpu_low_priority_queues: vec![low_queue],
        _enable_fallback: false,
        cpu_active_count: Arc::new(AtomicUsize::new(cpu_active)),
        gpu_in_flight_count: Arc::new(AtomicUsize::new(0)),
        gpu_stream_in_flight_count: Arc::new(AtomicUsize::new(0)),
        gpu_stream_admission: PriorityAdmission::new(queue_size.max(1)),
        device_mesh: None,
        router: MeshRouter::new(None, 1),
        max_micro_batch: 1,
        queue_overflow_policy: QueueOverflowPolicy::Block,
    }
}

async fn wait_for_wire_admission_waiters(scheduler: &Scheduler, expected: (usize, usize)) {
    tokio::time::timeout(std::time::Duration::from_secs(5), async {
        loop {
            if scheduler.gpu_stream_admission.waiter_counts() == expected {
                return;
            }
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap_or_else(|_| {
        panic!(
            "timed out waiting for wire admission state {:?}; observed {:?}",
            expected,
            scheduler.gpu_stream_admission.waiter_counts()
        )
    });
}

#[tokio::test]
async fn test_get_worker_index_round_robin() {
    let engine_handle: EngineHandle = Arc::new(MockEngine::new(EngineMetrics::default()));
    let scheduler = Scheduler::new(vec![engine_handle], 1, 3, 8, false, 1, 0, None);

    let indices = (0..4)
        .map(|_| scheduler.get_worker_index(&None))
        .collect::<Vec<_>>();

    assert_eq!(indices, vec![0, 1, 2, 0]);
}

#[tokio::test]
async fn test_get_worker_index_sticky_session() {
    let engine_handle: EngineHandle = Arc::new(MockEngine::new(EngineMetrics::default()));
    let scheduler = Scheduler::new(vec![engine_handle], 1, 4, 8, false, 1, 0, None);

    let session_id = Some("session-1".to_string());
    let first = scheduler.get_worker_index(&session_id);
    let second = scheduler.get_worker_index(&session_id);

    assert_eq!(first, second);
    assert!(first < scheduler.gpu_high_priority_queues.len());
}

#[tokio::test]
async fn test_drop_scheduler_releases_executor_engine_handles() {
    let engine = Arc::new(MockEngine::new(EngineMetrics::default()));
    let engine_handle: EngineHandle = engine.clone();
    let scheduler = Scheduler::new(vec![engine_handle], 1, 1, 8, false, 1, 0, None);

    drop(scheduler);
    for _ in 0..100 {
        if Arc::strong_count(&engine) == 1 {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(1)).await;
    }

    assert_eq!(
        Arc::strong_count(&engine),
        1,
        "executor task retained its engine after scheduler teardown"
    );
}

#[test]
fn test_is_healthy_threshold() {
    let engine_handle: EngineHandle = Arc::new(MockEngine::new(EngineMetrics::default()));
    let scheduler = build_scheduler_for_queue_tests(vec![engine_handle], 10, 0);

    for _ in 0..7 {
        assert!(scheduler.gpu_high_priority_queues[0]
            .try_push_drop_newest(make_request(None))
            .is_ok());
    }
    assert!(scheduler.is_healthy());

    assert!(scheduler.gpu_high_priority_queues[0]
        .try_push_drop_newest(make_request(None))
        .is_ok());
    assert!(!scheduler.is_healthy());
}

#[test]
fn test_get_queue_depth_counts_cpu_and_gpu() {
    let engine_handle: EngineHandle = Arc::new(MockEngine::new(EngineMetrics::default()));
    let scheduler = build_scheduler_for_queue_tests(vec![engine_handle], 5, 2);

    assert!(scheduler.gpu_high_priority_queues[0]
        .try_push_drop_newest(make_request(None))
        .is_ok());
    assert!(scheduler.gpu_low_priority_queues[0]
        .try_push_drop_newest(make_request(None))
        .is_ok());
    assert!(scheduler.gpu_low_priority_queues[0]
        .try_push_drop_newest(make_request(None))
        .is_ok());

    let (cpu_depth, gpu_depth) = scheduler.get_queue_depth();
    assert_eq!(cpu_depth, 2);
    assert_eq!(gpu_depth, 3);
}

#[test]
fn test_metrics_aggregation() {
    let engine_a: EngineHandle = Arc::new(MockEngine::new(EngineMetrics {
        memory_usage: 10,
        gpu_utilization: 0.2,
        throughput: 5.0,
        kv_cache_cpu_offloaded_blocks: 2,
        ..EngineMetrics::default()
    }));
    let engine_b: EngineHandle = Arc::new(MockEngine::new(EngineMetrics {
        memory_usage: 20,
        gpu_utilization: 0.6,
        throughput: 7.0,
        kv_cache_cpu_offloaded_blocks: 3,
        ..EngineMetrics::default()
    }));
    let scheduler = build_scheduler_for_queue_tests(vec![engine_a, engine_b], 5, 1);

    assert!(scheduler.gpu_high_priority_queues[0]
        .try_push_drop_newest(make_request(None))
        .is_ok());
    assert!(scheduler.gpu_low_priority_queues[0]
        .try_push_drop_newest(make_request(None))
        .is_ok());

    let metrics = ReplicaScheduler::get_metrics(&scheduler);

    assert_eq!(metrics.memory_usage, 30);
    assert_eq!(metrics.throughput, 12.0);
    assert_eq!(metrics.queue_depth, 3);
    assert_eq!(metrics.kv_cache_cpu_offloaded_blocks, 5);
    assert!((metrics.gpu_utilization - 0.4).abs() < 1e-6);
}

#[tokio::test]
async fn test_gpu_executor_coalesces_capable_backend() {
    use crate::gpu_executor::{GpuExecutor, WorkQueue};
    use std::time::Duration;

    let calls = Arc::new(std::sync::Mutex::new(Vec::<usize>::new()));
    let engine: EngineHandle = Arc::new(BatchRecordingEngine::new(calls.clone(), 8));

    let high = WorkQueue::new(64);
    let low = WorkQueue::new(64);
    let in_flight = Arc::new(AtomicUsize::new(0));
    let executor = GpuExecutor::new(high.clone(), low.clone(), engine, 8, 2, in_flight);

    // Enqueue a burst of latency-critical requests before the executor starts,
    // so its first greedy drain sees all of them already queued.
    let mut receivers = Vec::new();
    for _ in 0..6 {
        let (response_tx, response_rx) = oneshot::channel();
        high.push_block(Request {
            input: make_inference_request(None),
            response_tx,
        })
        .await;
        receivers.push(response_rx);
    }

    tokio::spawn(executor.run());

    for rx in receivers {
        let output = tokio::time::timeout(Duration::from_secs(5), rx)
            .await
            .expect("executor did not respond in time")
            .expect("response channel dropped")
            .expect("inference failed");
        assert_eq!(output.shape, vec![1]);
    }

    let recorded = calls.lock().unwrap().clone();
    assert_eq!(
        recorded.iter().sum::<usize>(),
        6,
        "every request must be dispatched exactly once, got {:?}",
        recorded
    );
    assert!(
        recorded.iter().any(|&n| n > 1),
        "expected the executor to coalesce a capable backend, got {:?}",
        recorded
    );
}

#[tokio::test]
async fn test_gpu_executor_uses_policy_queue_delay_for_request_coalescing_backend() {
    use crate::gpu_executor::{GpuExecutor, WorkQueue};
    use std::time::Duration;

    let calls = Arc::new(std::sync::Mutex::new(Vec::<usize>::new()));
    let engine: EngineHandle = Arc::new(BatchRecordingEngine::new_with_policy_delay(
        calls.clone(),
        8,
        60,
    ));

    let high = WorkQueue::new(64);
    let low = WorkQueue::new(64);
    // Simulate concurrent load so the adaptive coalescing window is allowed to
    // wait for a near-term straggler.
    let in_flight = Arc::new(AtomicUsize::new(1));
    let executor = GpuExecutor::new(high.clone(), low.clone(), engine, 8, 0, in_flight);

    let (tx1, rx1) = oneshot::channel();
    high.push_block(Request {
        input: make_inference_request(None),
        response_tx: tx1,
    })
    .await;

    tokio::spawn(executor.run());

    tokio::time::sleep(Duration::from_millis(10)).await;
    let (tx2, rx2) = oneshot::channel();
    high.push_block(Request {
        input: make_inference_request(None),
        response_tx: tx2,
    })
    .await;

    for rx in [rx1, rx2] {
        tokio::time::timeout(Duration::from_secs(5), rx)
            .await
            .expect("executor did not respond in time")
            .expect("response channel dropped")
            .expect("inference failed");
    }

    let recorded = calls.lock().unwrap().clone();
    assert!(
        recorded.contains(&2),
        "policy queue_delay_ms should allow the executor to coalesce the straggler, got {:?}",
        recorded
    );
}

#[tokio::test]
async fn test_gpu_executor_never_coalesces_self_batching_backend() {
    use crate::gpu_executor::{GpuExecutor, WorkQueue};
    use std::time::Duration;

    // A backend advertising max_batch() == 8 but self_batches() == true must
    // never be routed through infer_batch: it multiplexes internally, so every
    // request has to be dispatched individually (recorded call size 1).
    let calls = Arc::new(std::sync::Mutex::new(Vec::<usize>::new()));
    let engine: EngineHandle = Arc::new(BatchRecordingEngine::new_self_batching(calls.clone(), 8));

    let high = WorkQueue::new(64);
    let low = WorkQueue::new(64);
    let in_flight = Arc::new(AtomicUsize::new(0));
    let executor = GpuExecutor::new(high.clone(), low.clone(), engine, 8, 2, in_flight);

    // Burst on the low-priority queue too, where the single-dispatch path would
    // otherwise accumulate a serial infer_batch group.
    let mut receivers = Vec::new();
    for _ in 0..6 {
        let (response_tx, response_rx) = oneshot::channel();
        low.push_block(Request {
            input: make_inference_request(None),
            response_tx,
        })
        .await;
        receivers.push(response_rx);
    }

    tokio::spawn(executor.run());

    for rx in receivers {
        tokio::time::timeout(Duration::from_secs(5), rx)
            .await
            .expect("executor did not respond in time")
            .expect("response channel dropped")
            .expect("inference failed");
    }

    let recorded = calls.lock().unwrap().clone();
    assert_eq!(
        recorded.iter().sum::<usize>(),
        6,
        "every request must be dispatched exactly once, got {:?}",
        recorded
    );
    assert!(
        recorded.iter().all(|&n| n == 1),
        "self-batching backend must never be coalesced, got {:?}",
        recorded
    );
}

#[tokio::test]
async fn test_gpu_executor_admission_gates_saturated_self_batching_backend() {
    use crate::gpu_executor::{GpuExecutor, WorkQueue};
    use std::sync::atomic::Ordering;
    use std::time::Duration;

    // Free-KV headroom = 50%: with kv_total=100, free=0 (saturated) blocks
    // admission, free=100 (unsaturated) allows it.
    let gated = Arc::new(GatedEngine::new(100));
    let engine: EngineHandle = gated.clone();

    let high = WorkQueue::new(64);
    let low = WorkQueue::new(64);
    let in_flight = Arc::new(AtomicUsize::new(0));
    let executor = GpuExecutor::new(high.clone(), low.clone(), engine, 8, 2, in_flight)
        .with_admission_min_free_pct(50);

    // First request: admitted even though saturated, because nothing is in
    // flight yet (liveness escape). Its infer() then blocks, holding in_flight.
    let (tx1, _rx1) = oneshot::channel();
    high.push_block(Request {
        input: make_inference_request(None),
        response_tx: tx1,
    })
    .await;

    tokio::spawn(executor.run());

    // Wait until req1 is actually running inside infer().
    let deadline = std::time::Instant::now() + Duration::from_secs(5);
    while gated.started.load(Ordering::SeqCst) < 1 {
        assert!(std::time::Instant::now() < deadline, "req1 never started");
        tokio::time::sleep(Duration::from_millis(2)).await;
    }

    // Second request arrives while the backend is saturated AND req1 is in
    // flight: it must be held in the queue, not dispatched.
    let (tx2, rx2) = oneshot::channel();
    high.push_block(Request {
        input: make_inference_request(None),
        response_tx: tx2,
    })
    .await;

    tokio::time::sleep(Duration::from_millis(40)).await;
    assert_eq!(
        gated.started.load(Ordering::SeqCst),
        1,
        "saturated self-batching backend must not admit a second request"
    );

    // Free the KV cache: admission should now let req2 through even though req1
    // is still in flight (proves the gate keys on occupancy, not just in_flight).
    gated.saturated.store(false, Ordering::SeqCst);
    while gated.started.load(Ordering::SeqCst) < 2 {
        assert!(
            std::time::Instant::now() < deadline,
            "req2 not admitted after KV freed"
        );
        tokio::time::sleep(Duration::from_millis(2)).await;
    }

    // Drain: release both blocked infers and confirm req2 completes.
    gated.release_all();
    tokio::time::timeout(Duration::from_secs(5), rx2)
        .await
        .expect("req2 did not respond in time")
        .expect("response channel dropped")
        .expect("inference failed");
}

#[tokio::test]
async fn test_gpu_executor_stamps_queue_priority_for_self_batching_backend() {
    use crate::gpu_executor::{GpuExecutor, WorkQueue};
    use std::time::Duration;

    let seen = Arc::new(std::sync::Mutex::new(Vec::<Option<u8>>::new()));
    let engine: EngineHandle = Arc::new(PriorityRecordingEngine::continuous(seen.clone()));

    let high = WorkQueue::new(64);
    let low = WorkQueue::new(64);
    let in_flight = Arc::new(AtomicUsize::new(0));
    let executor = GpuExecutor::new(high.clone(), low.clone(), engine, 8, 2, in_flight);

    // One request on each queue, priority left unset so we observe the stamp.
    let (htx, hrx) = oneshot::channel();
    high.push_block(Request {
        input: make_inference_request(None),
        response_tx: htx,
    })
    .await;
    let (ltx, lrx) = oneshot::channel();
    low.push_block(Request {
        input: make_inference_request(None),
        response_tx: ltx,
    })
    .await;

    tokio::spawn(executor.run());

    for rx in [hrx, lrx] {
        tokio::time::timeout(Duration::from_secs(5), rx)
            .await
            .expect("executor did not respond in time")
            .expect("response channel dropped")
            .expect("inference failed");
    }

    let mut recorded = seen.lock().unwrap().clone();
    recorded.sort();
    assert_eq!(
        recorded,
        vec![Some(0), Some(1)],
        "high-queue request must be stamped priority 0 and low-queue 1, got {:?}",
        recorded
    );
}

#[tokio::test]
async fn test_gpu_executor_forwards_priority_without_coalescing_delegated_backend() {
    use crate::gpu_executor::{GpuExecutor, WorkQueue};
    use std::time::Duration;

    let seen = Arc::new(std::sync::Mutex::new(Vec::<Option<u8>>::new()));
    let engine: EngineHandle = Arc::new(PriorityRecordingEngine::delegated(seen.clone()));

    let high = WorkQueue::new(64);
    let low = WorkQueue::new(64);
    let in_flight = Arc::new(AtomicUsize::new(0));
    let executor = GpuExecutor::new(high.clone(), low.clone(), engine, 8, 2, in_flight);

    let mut receivers = Vec::new();
    let (high_tx, high_rx) = oneshot::channel();
    high.push_block(Request {
        input: make_inference_request(None),
        response_tx: high_tx,
    })
    .await;
    receivers.push(high_rx);

    for _ in 0..4 {
        let (response_tx, response_rx) = oneshot::channel();
        low.push_block(Request {
            input: make_inference_request(None),
            response_tx,
        })
        .await;
        receivers.push(response_rx);
    }

    tokio::spawn(executor.run());

    for rx in receivers {
        tokio::time::timeout(Duration::from_secs(5), rx)
            .await
            .expect("executor did not respond in time")
            .expect("response channel dropped")
            .expect("inference failed");
    }

    let mut recorded = seen.lock().unwrap().clone();
    recorded.sort();
    assert_eq!(
        recorded,
        vec![Some(0), Some(1), Some(1), Some(1), Some(1)],
        "delegated requests must be dispatched singly with parent queue priority, got {:?}",
        recorded
    );
}

#[tokio::test]
async fn test_scheduler_stream_stamps_priority_for_self_batching_backend() {
    let seen = Arc::new(std::sync::Mutex::new(Vec::<Option<u8>>::new()));
    let engine: EngineHandle = Arc::new(PriorityRecordingEngine::continuous(seen.clone()));
    let scheduler = build_scheduler_for_queue_tests(vec![engine], 8, 0);

    let latency_stream = ReplicaScheduler::infer_stream(
        &scheduler,
        make_inference_request(None),
        Priority::LatencyCritical,
        false,
    )
    .await
    .expect("latency stream should be admitted");

    let throughput_stream = ReplicaScheduler::infer_stream(
        &scheduler,
        make_inference_request(None),
        Priority::Throughput,
        false,
    )
    .await
    .expect("throughput stream should be admitted");

    drop(latency_stream);
    drop(throughput_stream);

    let mut recorded = seen.lock().unwrap().clone();
    recorded.sort();
    assert_eq!(
        recorded,
        vec![Some(0), Some(1)],
        "streaming requests must be stamped with the resolved scheduler priority, got {:?}",
        recorded
    );
}

#[tokio::test]
async fn test_scheduler_stream_stamps_priority_for_delegated_backend() {
    let seen = Arc::new(std::sync::Mutex::new(Vec::<Option<u8>>::new()));
    let engine: EngineHandle = Arc::new(PriorityRecordingEngine::delegated(seen.clone()));
    let scheduler = build_scheduler_for_queue_tests(vec![engine], 8, 0);

    let latency_stream = ReplicaScheduler::infer_stream(
        &scheduler,
        make_inference_request(None),
        Priority::LatencyCritical,
        false,
    )
    .await
    .expect("latency stream should be admitted");

    let throughput_stream = ReplicaScheduler::infer_stream(
        &scheduler,
        make_inference_request(None),
        Priority::Throughput,
        false,
    )
    .await
    .expect("throughput stream should be admitted");

    drop(latency_stream);
    drop(throughput_stream);

    let mut recorded = seen.lock().unwrap().clone();
    recorded.sort();
    assert_eq!(
        recorded,
        vec![Some(0), Some(1)],
        "delegated streams must carry the resolved scheduler priority, got {:?}",
        recorded
    );
}

#[tokio::test]
async fn test_scheduler_wire_paths_stamp_priority_and_hold_bounded_stream_admission() {
    use futures::StreamExt;

    let seen = Arc::new(std::sync::Mutex::new(Vec::<Option<u8>>::new()));
    let engine: EngineHandle = Arc::new(PriorityRecordingEngine::delegated(seen.clone()));
    let scheduler = build_scheduler_for_queue_tests(vec![engine], 1, 0)
        .with_queue_overflow_policy(QueueOverflowPolicy::DropNewest);

    let response = ReplicaScheduler::infer_openai_wire(
        &scheduler,
        make_openai_wire_request(OpenAiWireFormat::Json),
        Priority::Throughput,
        false,
    )
    .await
    .expect("wire request should be admitted");
    assert_eq!(response.body, br#"{"model":"test","messages":[]}"#);
    assert_eq!(scheduler.get_queue_depth(), (0, 0));

    let mut response = ReplicaScheduler::infer_openai_wire_stream(
        &scheduler,
        make_openai_wire_request(OpenAiWireFormat::ServerSentEvents),
        Priority::LatencyCritical,
        false,
    )
    .await
    .expect("wire stream should be admitted");
    assert_eq!(scheduler.get_queue_depth(), (0, 1));
    assert_eq!(
        response.body.next().await.unwrap().unwrap(),
        b"data: one\n\n"
    );

    let second = ReplicaScheduler::infer_openai_wire_stream(
        &scheduler,
        make_openai_wire_request(OpenAiWireFormat::ServerSentEvents),
        Priority::Throughput,
        false,
    )
    .await;
    match second {
        Ok(_) => panic!("second wire stream should be rejected while its slot is held"),
        Err(error) => assert!(error.is_overloaded()),
    }
    drop(response);
    assert_eq!(scheduler.get_queue_depth(), (0, 0));

    assert_eq!(*seen.lock().unwrap(), vec![Some(1), Some(0)]);
}

#[tokio::test]
async fn test_scheduler_wire_admission_prioritizes_latency_over_queued_throughput() {
    let seen = Arc::new(std::sync::Mutex::new(Vec::<Option<u8>>::new()));
    let engine: EngineHandle = Arc::new(PriorityRecordingEngine::delegated(seen.clone()));
    let scheduler = Arc::new(build_scheduler_for_queue_tests(vec![engine], 1, 0));

    let held = ReplicaScheduler::infer_openai_wire_stream(
        scheduler.as_ref(),
        make_openai_wire_request(OpenAiWireFormat::ServerSentEvents),
        Priority::Throughput,
        false,
    )
    .await
    .expect("first stream should hold the only admission slot");
    assert_eq!(scheduler.gpu_stream_admission.available_permits(), 0);

    let low_scheduler = Arc::clone(&scheduler);
    let low = tokio::spawn(async move {
        ReplicaScheduler::infer_openai_wire(
            low_scheduler.as_ref(),
            make_openai_wire_request(OpenAiWireFormat::Json),
            Priority::Throughput,
            false,
        )
        .await
    });
    wait_for_wire_admission_waiters(scheduler.as_ref(), (0, 1)).await;

    let high_scheduler = Arc::clone(&scheduler);
    let high = tokio::spawn(async move {
        ReplicaScheduler::infer_openai_wire_stream(
            high_scheduler.as_ref(),
            make_openai_wire_request(OpenAiWireFormat::ServerSentEvents),
            Priority::LatencyCritical,
            false,
        )
        .await
    });
    wait_for_wire_admission_waiters(scheduler.as_ref(), (1, 1)).await;

    drop(held);
    let high_response = tokio::time::timeout(std::time::Duration::from_secs(5), high)
        .await
        .expect("latency-critical request should receive the released slot")
        .expect("latency-critical task should not panic")
        .expect("latency-critical request should be admitted");
    assert!(
        !low.is_finished(),
        "queued throughput request must remain blocked while the high-priority stream owns the slot"
    );
    assert_eq!(*seen.lock().unwrap(), vec![Some(1), Some(0)]);

    drop(high_response);
    tokio::time::timeout(std::time::Duration::from_secs(5), low)
        .await
        .expect("throughput request should run after the high-priority stream drops")
        .expect("throughput task should not panic")
        .expect("throughput request should be admitted");
    assert_eq!(*seen.lock().unwrap(), vec![Some(1), Some(0), Some(1)]);
    assert_eq!(scheduler.gpu_stream_admission.available_permits(), 1);
}

#[tokio::test]
async fn test_scheduler_wire_admission_cancellation_and_drop_do_not_leak_capacity() {
    let seen = Arc::new(std::sync::Mutex::new(Vec::<Option<u8>>::new()));
    let engine: EngineHandle = Arc::new(PriorityRecordingEngine::delegated(seen));
    let scheduler = Arc::new(build_scheduler_for_queue_tests(vec![engine], 1, 0));

    let held = ReplicaScheduler::infer_openai_wire_stream(
        scheduler.as_ref(),
        make_openai_wire_request(OpenAiWireFormat::ServerSentEvents),
        Priority::Throughput,
        false,
    )
    .await
    .expect("first stream should hold the only admission slot");

    let cancelled_scheduler = Arc::clone(&scheduler);
    let cancelled = tokio::spawn(async move {
        ReplicaScheduler::infer_openai_wire(
            cancelled_scheduler.as_ref(),
            make_openai_wire_request(OpenAiWireFormat::Json),
            Priority::Throughput,
            false,
        )
        .await
    });
    wait_for_wire_admission_waiters(scheduler.as_ref(), (0, 1)).await;
    cancelled.abort();
    assert!(
        cancelled
            .await
            .expect_err("queued task should be cancelled")
            .is_cancelled(),
        "aborted admission task should report cancellation"
    );
    wait_for_wire_admission_waiters(scheduler.as_ref(), (0, 0)).await;
    assert_eq!(scheduler.gpu_stream_admission.available_permits(), 0);

    drop(held);
    let response = tokio::time::timeout(
        std::time::Duration::from_secs(5),
        ReplicaScheduler::infer_openai_wire(
            scheduler.as_ref(),
            make_openai_wire_request(OpenAiWireFormat::Json),
            Priority::LatencyCritical,
            false,
        ),
    )
    .await
    .expect("capacity should be reusable after cancellation and stream drop")
    .expect("replacement request should be admitted");
    assert_eq!(response.head.status, 200);
    assert_eq!(scheduler.gpu_stream_admission.available_permits(), 1);
}

#[tokio::test]
async fn test_scheduler_wire_token_cancelled_while_queued_never_dispatches() {
    let seen = Arc::new(std::sync::Mutex::new(Vec::<Option<u8>>::new()));
    let engine: EngineHandle = Arc::new(PriorityRecordingEngine::delegated(seen.clone()));
    let scheduler = Arc::new(build_scheduler_for_queue_tests(vec![engine], 1, 0));

    let held = ReplicaScheduler::infer_openai_wire_stream(
        scheduler.as_ref(),
        make_openai_wire_request(OpenAiWireFormat::ServerSentEvents),
        Priority::Throughput,
        false,
    )
    .await
    .expect("first stream should hold the only admission slot");

    let cancellation = CancellationToken::new();
    let mut request = make_openai_wire_request(OpenAiWireFormat::Json);
    request.cancellation = Some(cancellation.clone());
    let queued_scheduler = Arc::clone(&scheduler);
    let queued = tokio::spawn(async move {
        ReplicaScheduler::infer_openai_wire(
            queued_scheduler.as_ref(),
            request,
            Priority::Throughput,
            false,
        )
        .await
    });
    wait_for_wire_admission_waiters(scheduler.as_ref(), (0, 1)).await;

    cancellation.cancel();
    let error = tokio::time::timeout(std::time::Duration::from_secs(5), queued)
        .await
        .expect("cancellation should wake the queued admission immediately")
        .expect("queued task should not panic")
        .expect_err("a token cancelled while queued must not dispatch");

    assert!(matches!(error, EngineError::Cancelled { .. }));
    assert_eq!(seen.lock().unwrap().len(), 1);
    assert_eq!(scheduler.gpu_stream_admission.available_permits(), 0);
    drop(held);
    assert_eq!(scheduler.gpu_stream_admission.available_permits(), 1);
}

#[tokio::test]
async fn test_scheduler_stream_admission_tracks_depth_and_releases_on_drop() {
    let engine: EngineHandle = Arc::new(PendingStreamEngine);
    let scheduler = build_scheduler_for_queue_tests(vec![engine], 1, 0)
        .with_queue_overflow_policy(QueueOverflowPolicy::DropNewest);

    let stream = ReplicaScheduler::infer_stream(
        &scheduler,
        make_inference_request(None),
        Priority::Throughput,
        false,
    )
    .await
    .expect("first stream should be admitted");

    assert_eq!(scheduler.get_queue_depth(), (0, 1));

    let second = ReplicaScheduler::infer_stream(
        &scheduler,
        make_inference_request(None),
        Priority::Throughput,
        false,
    )
    .await;
    let err = match second {
        Ok(_) => panic!("second stream should be rejected while the only slot is held"),
        Err(err) => err,
    };
    assert!(err.is_overloaded());

    drop(stream);
    assert_eq!(scheduler.get_queue_depth(), (0, 0));

    let stream = ReplicaScheduler::infer_stream(
        &scheduler,
        make_inference_request(None),
        Priority::Throughput,
        false,
    )
    .await
    .expect("released stream slot should admit the next stream");
    drop(stream);
}

#[tokio::test]
async fn test_gpu_executor_no_batch_for_non_capable_backend() {
    use crate::gpu_executor::{GpuExecutor, WorkQueue};
    use std::time::Duration;

    // MockEngine keeps the default max_batch() == 1, so high-priority requests
    // must stay on the single-dispatch path even when a burst is queued.
    let engine: EngineHandle = Arc::new(MockEngine::new(EngineMetrics::default()));
    let high = WorkQueue::new(64);
    let low = WorkQueue::new(64);
    let in_flight = Arc::new(AtomicUsize::new(0));
    let executor = GpuExecutor::new(high.clone(), low.clone(), engine, 8, 2, in_flight);

    let mut receivers = Vec::new();
    for _ in 0..4 {
        let (response_tx, response_rx) = oneshot::channel();
        high.push_block(Request {
            input: make_inference_request(None),
            response_tx,
        })
        .await;
        receivers.push(response_rx);
    }

    tokio::spawn(executor.run());

    for rx in receivers {
        tokio::time::timeout(Duration::from_secs(5), rx)
            .await
            .expect("executor did not respond in time")
            .expect("response channel dropped")
            .expect("inference failed");
    }
}
