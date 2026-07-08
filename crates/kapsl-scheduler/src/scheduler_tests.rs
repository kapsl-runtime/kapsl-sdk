use super::*;
use crate::replica_pool::ReplicaScheduler;
use crate::request::Request;
use crate::scheduler::QueueOverflowPolicy;
use async_trait::async_trait;
use kapsl_engine_api::{
    BinaryTensorPacket, Engine, EngineError, EngineMetrics, EngineStream, InferenceRequest,
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
}

impl BatchRecordingEngine {
    fn new(calls: Arc<std::sync::Mutex<Vec<usize>>>, max_batch: usize) -> Self {
        Self {
            calls,
            max_batch,
            self_batches: false,
        }
    }

    /// A backend that advertises batching capacity yet self-batches internally,
    /// so the executor must dispatch each request individually anyway.
    fn new_self_batching(calls: Arc<std::sync::Mutex<Vec<usize>>>, max_batch: usize) -> Self {
        Self {
            calls,
            max_batch,
            self_batches: true,
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

    let scheduler = Scheduler {
        engines: engines.clone(),
        cpu_pool,
        gpu_high_priority_queues: vec![high_queue],
        gpu_low_priority_queues: vec![low_queue],
        _enable_fallback: false,
        cpu_active_count: Arc::new(AtomicUsize::new(cpu_active)),
        gpu_in_flight_count: Arc::new(AtomicUsize::new(0)),
        device_mesh: None,
        router: MeshRouter::new(None, 1),
        max_micro_batch: 1,
        queue_overflow_policy: QueueOverflowPolicy::Block,
    };

    scheduler
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
