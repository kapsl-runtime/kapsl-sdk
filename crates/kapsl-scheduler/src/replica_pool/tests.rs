use super::*;
use futures::StreamExt;
use kapsl_engine_api::{
    BinaryTensorPacket, OpenAiWireEndpoint, OpenAiWireFormat, OpenAiWireResponseHead, TensorDtype,
};

struct MockScheduler {
    queue_depth: (usize, usize),
    healthy: bool,
    metrics: kapsl_engine_api::EngineMetrics,
}

impl MockScheduler {
    fn new(queue_depth: (usize, usize), healthy: bool) -> Self {
        Self {
            queue_depth,
            healthy,
            metrics: kapsl_engine_api::EngineMetrics {
                queue_depth: queue_depth.0 + queue_depth.1,
                ..kapsl_engine_api::EngineMetrics::default()
            },
        }
    }

    fn with_metrics(
        queue_depth: (usize, usize),
        healthy: bool,
        metrics: kapsl_engine_api::EngineMetrics,
    ) -> Self {
        Self {
            queue_depth,
            healthy,
            metrics,
        }
    }
}

#[async_trait::async_trait]
impl ReplicaScheduler for MockScheduler {
    fn get_queue_depth(&self) -> (usize, usize) {
        self.queue_depth
    }

    fn is_healthy(&self) -> bool {
        self.healthy
    }

    async fn infer(
        &self,
        request: &InferenceRequest,
        _priority: Priority,
        _force_cpu: bool,
    ) -> Result<BinaryTensorPacket, EngineError> {
        if !self.healthy {
            return Err(EngineError::InferenceError {
                reason: "Unhealthy replica".to_string(),
                source: None,
            });
        }
        Ok(request.input.clone())
    }

    async fn infer_openai_wire(
        &self,
        request: OpenAiWireRequest,
        _priority: Priority,
        _force_cpu: bool,
    ) -> Result<OpenAiWireResponse, EngineError> {
        if !self.healthy {
            return Err(EngineError::backend("Unhealthy replica"));
        }
        Ok(OpenAiWireResponse {
            head: OpenAiWireResponseHead::new(200, Vec::new())?,
            body: request.body,
        })
    }

    async fn infer_openai_wire_stream(
        &self,
        request: OpenAiWireRequest,
        _priority: Priority,
        _force_cpu: bool,
    ) -> Result<OpenAiWireStreamResponse, EngineError> {
        if !self.healthy {
            return Err(EngineError::backend("Unhealthy replica"));
        }
        Ok(OpenAiWireStreamResponse {
            head: OpenAiWireResponseHead::new(200, Vec::new())?,
            body: Box::pin(futures::stream::once(async move { Ok(request.body) })),
        })
    }

    fn get_metrics(&self) -> kapsl_engine_api::EngineMetrics {
        self.metrics.clone()
    }

    async fn infer_stream(
        &self,
        request: InferenceRequest,
        _priority: Priority,
        _force_cpu: bool,
    ) -> Result<
        std::pin::Pin<
            Box<dyn futures::Stream<Item = Result<BinaryTensorPacket, EngineError>> + Send>,
        >,
        EngineError,
    > {
        if !self.healthy {
            return Err(EngineError::InferenceError {
                reason: "Unhealthy replica".to_string(),
                source: None,
            });
        }
        let result = Ok(request.input.clone());
        Ok(Box::pin(futures::stream::once(async move { result })))
    }
}

#[derive(Debug)]
struct WireDispatchSource(&'static str);

impl std::fmt::Display for WireDispatchSource {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.0)
    }
}

impl std::error::Error for WireDispatchSource {}

#[derive(Clone, Copy)]
enum WireDispatchFailure {
    None,
    Unary,
    Stream,
}

struct WireDispatchScheduler {
    failure: WireDispatchFailure,
    unary_calls: AtomicUsize,
    stream_calls: AtomicUsize,
}

impl WireDispatchScheduler {
    fn new(failure: WireDispatchFailure) -> Self {
        Self {
            failure,
            unary_calls: AtomicUsize::new(0),
            stream_calls: AtomicUsize::new(0),
        }
    }
}

#[async_trait::async_trait]
impl ReplicaScheduler for WireDispatchScheduler {
    fn get_queue_depth(&self) -> (usize, usize) {
        (0, 0)
    }

    fn is_healthy(&self) -> bool {
        true
    }

    fn get_metrics(&self) -> kapsl_engine_api::EngineMetrics {
        kapsl_engine_api::EngineMetrics::default()
    }

    async fn infer(
        &self,
        request: &InferenceRequest,
        _priority: Priority,
        _force_cpu: bool,
    ) -> Result<BinaryTensorPacket, EngineError> {
        Ok(request.input.clone())
    }

    async fn infer_openai_wire(
        &self,
        request: OpenAiWireRequest,
        _priority: Priority,
        _force_cpu: bool,
    ) -> Result<OpenAiWireResponse, EngineError> {
        self.unary_calls.fetch_add(1, Ordering::SeqCst);
        if matches!(self.failure, WireDispatchFailure::Unary) {
            return Err(EngineError::TimeoutError {
                message: "ambiguous unary failure after dispatch".to_string(),
                source: Some(Box::new(WireDispatchSource("unary transport reset"))),
            });
        }

        Ok(OpenAiWireResponse {
            head: OpenAiWireResponseHead::new(200, Vec::new())?,
            body: request.body,
        })
    }

    async fn infer_openai_wire_stream(
        &self,
        request: OpenAiWireRequest,
        _priority: Priority,
        _force_cpu: bool,
    ) -> Result<OpenAiWireStreamResponse, EngineError> {
        self.stream_calls.fetch_add(1, Ordering::SeqCst);
        if matches!(self.failure, WireDispatchFailure::Stream) {
            return Err(EngineError::Backend {
                message: "ambiguous stream failure after dispatch".to_string(),
                source: Some(Box::new(WireDispatchSource("stream header reset"))),
            });
        }

        Ok(OpenAiWireStreamResponse {
            head: OpenAiWireResponseHead::new(200, Vec::new())?,
            body: Box::pin(futures::stream::once(async move { Ok(request.body) })),
        })
    }

    async fn infer_stream(
        &self,
        request: InferenceRequest,
        _priority: Priority,
        _force_cpu: bool,
    ) -> Result<
        std::pin::Pin<
            Box<dyn futures::Stream<Item = Result<BinaryTensorPacket, EngineError>> + Send>,
        >,
        EngineError,
    > {
        Ok(Box::pin(futures::stream::once(
            async move { Ok(request.input) },
        )))
    }
}

fn assert_preserved_wire_error(
    error: EngineError,
    expected_message: &str,
    expected_source: &str,
    expect_timeout: bool,
) {
    let (message, source) = match error {
        EngineError::TimeoutError { message, source } if expect_timeout => (message, source),
        EngineError::Backend { message, source } if !expect_timeout => (message, source),
        other => panic!("wire error changed variant: {other:?}"),
    };
    assert_eq!(message, expected_message);
    let source = source.expect("wire error source must be preserved");
    let source = source
        .downcast_ref::<WireDispatchSource>()
        .expect("wire error source type must be preserved");
    assert_eq!(source.0, expected_source);
}

#[tokio::test]
async fn openai_wire_unary_ambiguous_error_is_not_replayed_and_is_preserved() {
    let pool = ReplicaPool::new(PoolStrategy::RoundRobin);
    let failing = Arc::new(WireDispatchScheduler::new(WireDispatchFailure::Unary));
    let fallback = Arc::new(WireDispatchScheduler::new(WireDispatchFailure::None));
    pool.add_replica(0, failing.clone());
    pool.add_replica(1, fallback.clone());

    let request = OpenAiWireRequest::new(
        OpenAiWireEndpoint::ChatCompletions,
        OpenAiWireFormat::Json,
        b"unary".to_vec(),
    );
    let error = match pool
        .infer_openai_wire(request, Priority::Throughput, false)
        .await
    {
        Ok(_) => panic!("ambiguous unary failure unexpectedly succeeded"),
        Err(error) => error,
    };

    assert_preserved_wire_error(
        error,
        "ambiguous unary failure after dispatch",
        "unary transport reset",
        true,
    );
    assert_eq!(failing.unary_calls.load(Ordering::SeqCst), 1);
    assert_eq!(fallback.unary_calls.load(Ordering::SeqCst), 0);
    assert_eq!(pool.stats()[0].requests_total, 1);
    assert_eq!(pool.stats()[1].requests_total, 0);
}

#[tokio::test]
async fn openai_wire_stream_ambiguous_error_is_not_replayed_and_is_preserved() {
    let pool = ReplicaPool::new(PoolStrategy::RoundRobin);
    let failing = Arc::new(WireDispatchScheduler::new(WireDispatchFailure::Stream));
    let fallback = Arc::new(WireDispatchScheduler::new(WireDispatchFailure::None));
    pool.add_replica(0, failing.clone());
    pool.add_replica(1, fallback.clone());

    let request = OpenAiWireRequest::new(
        OpenAiWireEndpoint::ChatCompletions,
        OpenAiWireFormat::ServerSentEvents,
        b"stream".to_vec(),
    );
    let error = match pool
        .infer_openai_wire_stream(request, Priority::LatencyCritical, false)
        .await
    {
        Ok(_) => panic!("ambiguous stream failure unexpectedly succeeded"),
        Err(error) => error,
    };

    assert_preserved_wire_error(
        error,
        "ambiguous stream failure after dispatch",
        "stream header reset",
        false,
    );
    assert_eq!(failing.stream_calls.load(Ordering::SeqCst), 1);
    assert_eq!(fallback.stream_calls.load(Ordering::SeqCst), 0);
    assert_eq!(pool.stats()[0].requests_total, 1);
    assert_eq!(pool.stats()[1].requests_total, 0);
}

#[tokio::test]
async fn test_round_robin_distribution() {
    let pool = ReplicaPool::new(PoolStrategy::RoundRobin);

    // Add 3 replicas
    for i in 0..3 {
        pool.add_replica(i, Arc::new(MockScheduler::new((0, 0), true)));
    }

    let request = InferenceRequest::new(BinaryTensorPacket {
        shape: vec![1, 1],
        dtype: TensorDtype::Float32,
        data: vec![0, 0, 0, 0],
    });

    // Execute 9 requests
    for _ in 0..9 {
        let _ = pool
            .execute(request.clone(), Priority::Throughput, false)
            .await;
    }

    // Verify distribution is even (each should have 3 requests)
    let stats = pool.stats();
    for stat in stats {
        assert_eq!(stat.requests_total, 3);
    }
}

#[test]
fn pool_metrics_sum_delegated_capacity_across_replicas() {
    let pool = ReplicaPool::new(PoolStrategy::RoundRobin);
    for (replica_id, capacity) in [(0, 7), (1, 5)] {
        pool.add_replica(
            replica_id,
            Arc::new(MockScheduler::with_metrics(
                (0, 0),
                true,
                kapsl_engine_api::EngineMetrics {
                    batch_size: capacity,
                    ..Default::default()
                },
            )),
        );
    }

    assert_eq!(ReplicaScheduler::get_metrics(&pool).batch_size, 12);
}

#[test]
fn empty_pool_is_not_healthy() {
    let pool = ReplicaPool::<MockScheduler>::new(PoolStrategy::RoundRobin);

    assert!(!ReplicaScheduler::is_healthy(&pool));
    assert_eq!(ReplicaScheduler::get_queue_depth(&pool), (0, 0));
}

#[tokio::test]
async fn test_least_loaded_selection() {
    let pool = ReplicaPool::new(PoolStrategy::LeastLoaded);

    // Add replicas with different queue depths
    pool.add_replica(0, Arc::new(MockScheduler::new((10, 5), true)));
    pool.add_replica(1, Arc::new(MockScheduler::new((2, 1), true)));
    pool.add_replica(2, Arc::new(MockScheduler::new((5, 3), true)));

    let request = InferenceRequest::new(BinaryTensorPacket {
        shape: vec![1, 1],
        dtype: TensorDtype::Float32,
        data: vec![0, 0, 0, 0],
    });

    // Execute request - should go to replica 1 (lowest queue depth of 3)
    let _ = pool.execute(request, Priority::Throughput, false).await;

    let stats = pool.stats();
    // Replica 1 should have received the request
    assert_eq!(stats[1].requests_total, 1);
    assert_eq!(stats[0].requests_total, 0);
    assert_eq!(stats[2].requests_total, 0);
}

#[tokio::test]
async fn test_sticky_routing() {
    let pool = ReplicaPool::new(PoolStrategy::Sticky);

    // Add 3 replicas
    for i in 0..3 {
        pool.add_replica(i, Arc::new(MockScheduler::new((0, 0), true)));
    }

    let request = InferenceRequest::new(BinaryTensorPacket {
        shape: vec![1, 1],
        dtype: TensorDtype::Float32,
        data: vec![0, 0, 0, 0],
    })
    .with_session_id("session123");

    // Execute same session multiple times
    for _ in 0..5 {
        let _ = pool
            .execute(request.clone(), Priority::Throughput, false)
            .await;
    }

    let stats = pool.stats();
    // All requests should go to the same replica (whichever the hash maps to)
    let total_requests: u64 = stats.iter().map(|s| s.requests_total).sum();
    assert_eq!(total_requests, 5);

    // One replica should have all 5 requests
    assert!(stats.iter().any(|s| s.requests_total == 5));
}

#[tokio::test]
async fn test_failover() {
    let pool = ReplicaPool::new(PoolStrategy::RoundRobin);

    // Add unhealthy and healthy replicas
    pool.add_replica(0, Arc::new(MockScheduler::new((0, 0), false)));
    pool.add_replica(1, Arc::new(MockScheduler::new((0, 0), true)));

    let request = InferenceRequest::new(BinaryTensorPacket {
        shape: vec![1, 1],
        dtype: TensorDtype::Float32,
        data: vec![0, 0, 0, 0],
    });

    // First request goes to replica 0 (unhealthy), should failover to replica 1
    let result = pool.execute(request, Priority::Throughput, false).await;
    assert!(result.is_ok());

    let stats = pool.stats();
    // Replica 1 should have received the failover request
    assert_eq!(stats[1].requests_total, 1);
}

#[tokio::test]
async fn test_streaming_failover() {
    let pool = ReplicaPool::new(PoolStrategy::RoundRobin);

    // Add unhealthy and healthy replicas
    pool.add_replica(0, Arc::new(MockScheduler::new((0, 0), false)));
    pool.add_replica(1, Arc::new(MockScheduler::new((0, 0), true)));

    let request = InferenceRequest::new(BinaryTensorPacket {
        shape: vec![1, 1],
        dtype: TensorDtype::Float32,
        data: vec![0, 0, 0, 0],
    });

    // First attempt should pick replica 0 (Round Robin) and failover to replica 1
    let result = pool
        .infer_stream(request.clone(), Priority::LatencyCritical, false)
        .await;

    assert!(
        result.is_ok(),
        "Streaming request should succeed via failover"
    );
    let mut stream = result.unwrap();

    // Consume stream to verify it works
    let item = stream.next().await;
    assert!(item.is_some());
    assert!(item.unwrap().is_ok());

    let stats = pool.stats();
    // Replica 1 should have received the request (Replica 0 failed)
    assert!(stats[1].requests_total >= 1);
}

#[tokio::test]
async fn openai_wire_paths_preserve_replica_selection_and_startup_failover() {
    let pool = ReplicaPool::new(PoolStrategy::RoundRobin);
    pool.add_replica(0, Arc::new(MockScheduler::new((0, 0), false)));
    pool.add_replica(1, Arc::new(MockScheduler::new((0, 0), true)));

    let unary = OpenAiWireRequest::new(
        OpenAiWireEndpoint::ChatCompletions,
        OpenAiWireFormat::Json,
        b"unary-body".to_vec(),
    )
    .with_session_id("session-a")
    .with_metadata(kapsl_engine_api::OpenAiWireMetadata {
        request_id: Some("wire-request".to_string()),
        ..Default::default()
    });
    let response = pool
        .infer_openai_wire(unary, Priority::Throughput, false)
        .await
        .expect("unary wire request should fail over");
    assert_eq!(response.body, b"unary-body");

    // Advance round-robin back to the unhealthy replica and verify stream
    // startup can fail over before a response head is committed.
    let advance = OpenAiWireRequest::new(
        OpenAiWireEndpoint::ChatCompletions,
        OpenAiWireFormat::Json,
        b"advance".to_vec(),
    );
    pool.infer_openai_wire(advance, Priority::Throughput, false)
        .await
        .expect("healthy replica should accept the intervening request");
    let streaming = OpenAiWireRequest::new(
        OpenAiWireEndpoint::ChatCompletions,
        OpenAiWireFormat::ServerSentEvents,
        b"raw-sse".to_vec(),
    )
    .with_session_id("session-a");
    let mut response = pool
        .infer_openai_wire_stream(streaming, Priority::LatencyCritical, false)
        .await
        .expect("wire stream should fail over during startup");
    assert_eq!(response.body.next().await.unwrap().unwrap(), b"raw-sse");
    assert_eq!(pool.stats()[1].requests_total, 3);
}

#[tokio::test]
async fn test_queue_depth_aggregation() {
    let pool = ReplicaPool::new(PoolStrategy::RoundRobin);

    // Add replicas with different queue depths
    pool.add_replica(0, Arc::new(MockScheduler::new((10, 5), true)));
    pool.add_replica(1, Arc::new(MockScheduler::new((2, 1), true)));
    pool.add_replica(2, Arc::new(MockScheduler::new((5, 3), true)));

    // Total high: 10 + 2 + 5 = 17
    // Total low: 5 + 1 + 3 = 9
    let (high, low) = pool.get_queue_depth();
    assert_eq!(high, 17);
    assert_eq!(low, 9);
}

#[tokio::test]
async fn test_least_loaded_prefers_lower_kv_pressure_when_paged_metrics_exist() {
    let pool = ReplicaPool::new(PoolStrategy::LeastLoaded);

    let high_pressure_metrics = kapsl_engine_api::EngineMetrics {
        kv_cache_blocks_total: 100,
        kv_cache_blocks_free: 4,
        kv_cache_bytes_capacity: 1_000,
        kv_cache_bytes_used: 960,
        ..kapsl_engine_api::EngineMetrics::default()
    };
    let low_pressure_metrics = kapsl_engine_api::EngineMetrics {
        kv_cache_blocks_total: 100,
        kv_cache_blocks_free: 32,
        kv_cache_bytes_capacity: 1_000,
        kv_cache_bytes_used: 680,
        ..kapsl_engine_api::EngineMetrics::default()
    };

    pool.add_replica(
        0,
        Arc::new(MockScheduler::with_metrics(
            (0, 0),
            true,
            high_pressure_metrics,
        )),
    );
    pool.add_replica(
        1,
        Arc::new(MockScheduler::with_metrics(
            (1, 0),
            true,
            low_pressure_metrics,
        )),
    );

    let request = InferenceRequest::new(BinaryTensorPacket {
        shape: vec![1, 1],
        dtype: TensorDtype::Float32,
        data: vec![0, 0, 0, 0],
    });

    let _ = pool.execute(request, Priority::Throughput, false).await;

    let stats = pool.stats();
    assert_eq!(stats[0].requests_total, 0);
    assert_eq!(stats[1].requests_total, 1);
}

#[tokio::test]
async fn test_failover_prefers_more_kv_headroom() {
    let pool = ReplicaPool::new(PoolStrategy::RoundRobin);

    let failing_metrics = kapsl_engine_api::EngineMetrics::default();
    let worse_headroom = kapsl_engine_api::EngineMetrics {
        kv_cache_blocks_total: 100,
        kv_cache_blocks_free: 8,
        kv_cache_bytes_capacity: 1_000,
        kv_cache_bytes_used: 920,
        ..kapsl_engine_api::EngineMetrics::default()
    };
    let better_headroom = kapsl_engine_api::EngineMetrics {
        kv_cache_blocks_total: 100,
        kv_cache_blocks_free: 40,
        kv_cache_bytes_capacity: 1_000,
        kv_cache_bytes_used: 600,
        ..kapsl_engine_api::EngineMetrics::default()
    };

    pool.add_replica(
        0,
        Arc::new(MockScheduler::with_metrics((0, 0), false, failing_metrics)),
    );
    pool.add_replica(
        1,
        Arc::new(MockScheduler::with_metrics((0, 0), true, worse_headroom)),
    );
    pool.add_replica(
        2,
        Arc::new(MockScheduler::with_metrics((0, 0), true, better_headroom)),
    );

    let request = InferenceRequest::new(BinaryTensorPacket {
        shape: vec![1, 1],
        dtype: TensorDtype::Float32,
        data: vec![0, 0, 0, 0],
    });

    let result = pool.execute(request, Priority::Throughput, false).await;
    assert!(result.is_ok());

    let stats = pool.stats();
    assert_eq!(stats[1].requests_total, 0);
    assert_eq!(stats[2].requests_total, 1);
}
