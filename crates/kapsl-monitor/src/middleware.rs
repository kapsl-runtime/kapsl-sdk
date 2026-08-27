use async_trait::async_trait;
use futures::Stream;
use kapsl_engine_api::Engine;
use kapsl_engine_api::{
    BatchingPolicy, BinaryTensorPacket, EngineError, EngineModelInfo, InferenceRequest,
    KvBackendCapabilities, KvTopology, MemoryReport, OpenAiWireRequest, OpenAiWireResponse,
    OpenAiWireStreamResponse, RequestMemoryAdmission,
};
use std::collections::VecDeque;
use std::pin::Pin;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll};
use std::time::Instant;

use crate::metrics::KapslMetrics;

pub struct MonitoringMiddleware<E: Engine> {
    inner: E,
    metrics: KapslMetrics,
    model_id: String,
    version: String,
    auto_tune: Arc<ConcurrencyAutoTuneState>,
}

struct MetricStream<T> {
    inner: Pin<Box<dyn Stream<Item = Result<T, EngineError>> + Send>>,
    metrics: KapslMetrics,
    model_id: String,
    version: String,
    start: Instant,
    finished: bool,
    saw_error: bool,
    first_token_seen: bool,
    auto_tune: Arc<ConcurrencyAutoTuneState>,
}

const PEAK_CONCURRENCY_WINDOW_ENV: &str = "KAPSL_PEAK_CONCURRENCY_WINDOW";
const PEAK_CONCURRENCY_SAMPLE_STRIDE_ENV: &str = "KAPSL_PEAK_CONCURRENCY_SAMPLE_STRIDE";
const DEFAULT_PEAK_CONCURRENCY_WINDOW: usize = 512;
const DEFAULT_PEAK_CONCURRENCY_SAMPLE_STRIDE: u64 = 1;

#[derive(Debug)]
struct ConcurrencyAutoTuneState {
    in_flight: AtomicUsize,
    sample_counter: AtomicU64,
    samples: Mutex<VecDeque<u32>>,
    window: usize,
    sample_stride: u64,
}

impl ConcurrencyAutoTuneState {
    fn new() -> Self {
        let window = std::env::var(PEAK_CONCURRENCY_WINDOW_ENV)
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok())
            .unwrap_or(DEFAULT_PEAK_CONCURRENCY_WINDOW)
            .max(1);
        let sample_stride = std::env::var(PEAK_CONCURRENCY_SAMPLE_STRIDE_ENV)
            .ok()
            .and_then(|v| v.trim().parse::<u64>().ok())
            .unwrap_or(DEFAULT_PEAK_CONCURRENCY_SAMPLE_STRIDE)
            .max(1);
        Self {
            in_flight: AtomicUsize::new(0),
            sample_counter: AtomicU64::new(0),
            samples: Mutex::new(VecDeque::with_capacity(window)),
            window,
            sample_stride,
        }
    }

    fn on_request_start(&self) {
        let active = self
            .in_flight
            .fetch_add(1, Ordering::Relaxed)
            .saturating_add(1);
        let count = self
            .sample_counter
            .fetch_add(1, Ordering::Relaxed)
            .saturating_add(1);
        if count.is_multiple_of(self.sample_stride) {
            let mut samples = self
                .samples
                .lock()
                .unwrap_or_else(|poison| poison.into_inner());
            if samples.len() >= self.window {
                samples.pop_front();
            }
            samples.push_back(active.min(u32::MAX as usize) as u32);
        }
    }

    fn on_request_end(&self) {
        let _ = self.in_flight.fetch_sub(1, Ordering::Relaxed);
    }

    fn estimated_peak_concurrency(&self) -> Option<u32> {
        let mut values: Vec<u32> = {
            let samples = self
                .samples
                .lock()
                .unwrap_or_else(|poison| poison.into_inner());
            if samples.is_empty() {
                return None;
            }
            samples.iter().copied().collect()
        };
        values.sort_unstable();
        let idx = ((values.len() - 1) * 95) / 100;
        Some(values[idx].max(1))
    }
}

impl<T> MetricStream<T> {
    fn new(
        inner: Pin<Box<dyn Stream<Item = Result<T, EngineError>> + Send>>,
        metrics: KapslMetrics,
        model_id: String,
        version: String,
        start: Instant,
        auto_tune: Arc<ConcurrencyAutoTuneState>,
    ) -> Self {
        Self {
            inner,
            metrics,
            model_id,
            version,
            start,
            finished: false,
            saw_error: false,
            first_token_seen: false,
            auto_tune,
        }
    }

    fn finish(&mut self) {
        if self.finished {
            return;
        }

        let status = if self.saw_error { "err" } else { "ok" };
        let elapsed = self.start.elapsed().as_secs_f64();

        self.metrics
            .inference_count
            .with_label_values(&[self.model_id.as_str(), status])
            .inc();
        self.metrics
            .inference_latency
            .with_label_values(&[self.model_id.as_str(), self.version.as_str(), status])
            .observe(elapsed);
        self.metrics
            .active_inferences
            .with_label_values(&[self.model_id.as_str()])
            .dec();
        self.auto_tune.on_request_end();

        self.finished = true;
    }
}

impl<T> Stream for MetricStream<T> {
    type Item = Result<T, EngineError>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();
        match this.inner.as_mut().poll_next(cx) {
            Poll::Ready(Some(Ok(item))) => {
                if !this.first_token_seen {
                    this.first_token_seen = true;
                    let ttft = this.start.elapsed().as_secs_f64();
                    this.metrics
                        .ttft_latency
                        .with_label_values(&[this.model_id.as_str(), this.version.as_str(), "ok"])
                        .observe(ttft);
                    this.metrics
                        .model_ttft_ms
                        .with_label_values(&[this.model_id.as_str()])
                        .set(ttft * 1000.0);
                }
                Poll::Ready(Some(Ok(item)))
            }
            Poll::Ready(Some(Err(err))) => {
                if !this.first_token_seen {
                    this.first_token_seen = true;
                    let ttft = this.start.elapsed().as_secs_f64();
                    this.metrics
                        .ttft_latency
                        .with_label_values(&[this.model_id.as_str(), this.version.as_str(), "err"])
                        .observe(ttft);
                    this.metrics
                        .model_ttft_ms
                        .with_label_values(&[this.model_id.as_str()])
                        .set(ttft * 1000.0);
                }
                this.saw_error = true;
                Poll::Ready(Some(Err(err)))
            }
            Poll::Ready(None) => {
                this.finish();
                Poll::Ready(None)
            }
            Poll::Pending => Poll::Pending,
        }
    }
}

impl<T> Drop for MetricStream<T> {
    fn drop(&mut self) {
        if !self.finished {
            self.saw_error = true;
            self.finish();
        }
    }
}

impl<E: Engine> MonitoringMiddleware<E> {
    pub fn new(
        inner: E,
        model_id: String,
        version: String,
        registry: &std::sync::Arc<prometheus::Registry>,
    ) -> Self {
        Self {
            inner,
            metrics: KapslMetrics::new(registry),
            model_id,
            version,
            auto_tune: Arc::new(ConcurrencyAutoTuneState::new()),
        }
    }

    pub fn new_with_metrics(
        inner: E,
        model_id: String,
        version: String,
        metrics: KapslMetrics,
    ) -> Self {
        Self {
            inner,
            metrics,
            model_id,
            version,
            auto_tune: Arc::new(ConcurrencyAutoTuneState::new()),
        }
    }

    pub fn registry(&self) -> &std::sync::Arc<prometheus::Registry> {
        &self.metrics.registry
    }

    /// Record scheduler dispatch cardinality, not the tensor's leading
    /// dimension. Before preprocessing, that dimension can be an encoded image
    /// byte count or a PCM sample count rather than a model batch axis.
    fn observe_dispatch_size(&self, request_count: usize) {
        debug_assert!(request_count > 0);
        self.metrics
            .batch_size_hist
            .with_label_values(&[self.model_id.as_str()])
            .observe(request_count as f64);
    }

    fn begin_wire_observation(&self) -> Instant {
        self.metrics
            .active_inferences
            .with_label_values(&[self.model_id.as_str()])
            .inc();
        self.auto_tune.on_request_start();
        self.observe_dispatch_size(1);
        Instant::now()
    }

    fn finish_wire_observation(&self, start: Instant, succeeded: bool) {
        let status = if succeeded { "ok" } else { "err" };
        self.metrics
            .inference_count
            .with_label_values(&[self.model_id.as_str(), status])
            .inc();
        self.metrics
            .inference_latency
            .with_label_values(&[self.model_id.as_str(), self.version.as_str(), status])
            .observe(start.elapsed().as_secs_f64());
        self.metrics
            .active_inferences
            .with_label_values(&[self.model_id.as_str()])
            .dec();
        self.auto_tune.on_request_end();
    }

    async fn observe_wire_unary<F>(&self, operation: F) -> Result<OpenAiWireResponse, EngineError>
    where
        F: std::future::Future<Output = Result<OpenAiWireResponse, EngineError>>,
    {
        let start = self.begin_wire_observation();
        let result = operation.await;
        self.finish_wire_observation(start, result.is_ok());
        result
    }

    async fn observe_wire_stream<F>(
        &self,
        operation: F,
    ) -> Result<OpenAiWireStreamResponse, EngineError>
    where
        F: std::future::Future<Output = Result<OpenAiWireStreamResponse, EngineError>>,
    {
        let start = self.begin_wire_observation();
        match operation.await {
            Ok(response) => Ok(OpenAiWireStreamResponse {
                head: response.head,
                body: Box::pin(MetricStream::new(
                    response.body,
                    self.metrics.clone(),
                    self.model_id.clone(),
                    self.version.clone(),
                    start,
                    self.auto_tune.clone(),
                )),
            }),
            Err(error) => {
                self.finish_wire_observation(start, false);
                Err(error)
            }
        }
    }
}

#[async_trait]
impl<E: Engine> Engine for MonitoringMiddleware<E> {
    fn kv_capabilities(&self) -> KvBackendCapabilities {
        self.inner.kv_capabilities()
    }

    fn kv_topology(&self) -> Option<KvTopology> {
        self.inner.kv_topology()
    }

    fn planned_memory(
        &self,
        model_path: &std::path::Path,
    ) -> Result<kapsl_engine_api::MemoryReport, EngineError> {
        self.inner.planned_memory(model_path)
    }

    fn planned_external_device_memory(
        &self,
        model_path: &std::path::Path,
    ) -> Result<kapsl_engine_api::ExternalDeviceMemoryReport, EngineError> {
        self.inner.planned_external_device_memory(model_path)
    }

    async fn load(&mut self, model_path: &std::path::Path) -> Result<(), EngineError> {
        self.inner.load(model_path).await
    }

    fn actual_external_device_memory(&self) -> kapsl_engine_api::ExternalDeviceMemoryReport {
        self.inner.actual_external_device_memory()
    }

    fn actual_memory(&self) -> kapsl_engine_api::MemoryReport {
        self.inner.actual_memory()
    }

    fn planned_request_memory(&self, request: &InferenceRequest) -> kapsl_engine_api::MemoryReport {
        self.inner.planned_request_memory(request)
    }

    fn infer(&self, request: &InferenceRequest) -> Result<BinaryTensorPacket, EngineError> {
        let start_active = Instant::now();
        self.metrics
            .active_inferences
            .with_label_values(&[self.model_id.as_str()])
            .inc();
        self.auto_tune.on_request_start();

        self.observe_dispatch_size(1);

        let start = Instant::now();
        let result = self.inner.infer(request);
        let elapsed = start.elapsed().as_secs_f64();

        match &result {
            Ok(_) => {
                self.metrics
                    .inference_count
                    .with_label_values(&[self.model_id.as_str(), "ok"])
                    .inc();
                self.metrics
                    .inference_latency
                    .with_label_values(&[self.model_id.as_str(), self.version.as_str(), "ok"])
                    .observe(elapsed);
            }
            Err(_) => {
                self.metrics
                    .inference_count
                    .with_label_values(&[self.model_id.as_str(), "err"])
                    .inc();
                self.metrics
                    .inference_latency
                    .with_label_values(&[self.model_id.as_str(), self.version.as_str(), "err"])
                    .observe(elapsed);
            }
        }

        let _active_elapsed = start_active.elapsed().as_secs_f64();

        self.metrics
            .active_inferences
            .with_label_values(&[self.model_id.as_str()])
            .dec();
        self.auto_tune.on_request_end();

        result
    }

    fn supports_openai_wire(&self) -> bool {
        self.inner.supports_openai_wire()
    }

    fn planned_openai_wire_request_memory(&self, request: &OpenAiWireRequest) -> MemoryReport {
        self.inner.planned_openai_wire_request_memory(request)
    }

    async fn infer_openai_wire_with_memory_admission(
        &self,
        request: &OpenAiWireRequest,
        admission: RequestMemoryAdmission,
    ) -> Result<OpenAiWireResponse, EngineError> {
        self.observe_wire_unary(
            self.inner
                .infer_openai_wire_with_memory_admission(request, admission),
        )
        .await
    }

    async fn infer_openai_wire(
        &self,
        request: &OpenAiWireRequest,
    ) -> Result<OpenAiWireResponse, EngineError> {
        self.observe_wire_unary(self.inner.infer_openai_wire(request))
            .await
    }

    async fn infer_openai_wire_stream(
        &self,
        request: &OpenAiWireRequest,
    ) -> Result<OpenAiWireStreamResponse, EngineError> {
        self.observe_wire_stream(self.inner.infer_openai_wire_stream(request))
            .await
    }

    async fn infer_openai_wire_stream_with_memory_admission(
        &self,
        request: &OpenAiWireRequest,
        admission: RequestMemoryAdmission,
    ) -> Result<OpenAiWireStreamResponse, EngineError> {
        self.observe_wire_stream(
            self.inner
                .infer_openai_wire_stream_with_memory_admission(request, admission),
        )
        .await
    }

    fn infer_batch(
        &self,
        requests: &[InferenceRequest],
    ) -> Result<Vec<BinaryTensorPacket>, EngineError> {
        if requests.is_empty() {
            return self.inner.infer_batch(requests);
        }

        let model_id = self.model_id.as_str();
        self.observe_dispatch_size(requests.len());
        for _ in requests {
            self.metrics
                .active_inferences
                .with_label_values(&[model_id])
                .inc();
            self.auto_tune.on_request_start();
        }

        let start = Instant::now();
        let result = self.inner.infer_batch(requests);
        let elapsed = start.elapsed().as_secs_f64();

        let status = if result.is_ok() { "ok" } else { "err" };
        for _ in requests {
            self.metrics
                .inference_count
                .with_label_values(&[model_id, status])
                .inc();
            self.metrics
                .inference_latency
                .with_label_values(&[model_id, self.version.as_str(), status])
                .observe(elapsed);
            self.metrics
                .active_inferences
                .with_label_values(&[model_id])
                .dec();
            self.auto_tune.on_request_end();
        }

        result
    }

    fn max_batch(&self) -> usize {
        self.inner.max_batch()
    }

    fn self_batches(&self) -> bool {
        self.inner.self_batches()
    }

    fn batching_policy(&self) -> BatchingPolicy {
        self.inner.batching_policy()
    }

    fn infer_stream(
        &self,
        request: &InferenceRequest,
    ) -> std::pin::Pin<
        Box<dyn futures::stream::Stream<Item = Result<BinaryTensorPacket, EngineError>> + Send>,
    > {
        self.metrics
            .active_inferences
            .with_label_values(&[self.model_id.as_str()])
            .inc();
        self.auto_tune.on_request_start();
        self.observe_dispatch_size(1);

        let start = Instant::now();
        let inner = self.inner.infer_stream(request);
        let wrapped = MetricStream::new(
            inner,
            self.metrics.clone(),
            self.model_id.clone(),
            self.version.clone(),
            start,
            self.auto_tune.clone(),
        );

        Box::pin(wrapped)
    }

    fn unload(&mut self) {
        self.inner.unload()
    }

    fn metrics(&self) -> kapsl_engine_api::EngineMetrics {
        self.inner.metrics()
    }

    fn model_info(&self) -> Option<EngineModelInfo> {
        let mut info = self.inner.model_info()?;
        if let Some(observed_p95) = self.auto_tune.estimated_peak_concurrency() {
            info.peak_concurrency = Some(info.peak_concurrency.unwrap_or(1).max(observed_p95));
        }
        Some(info)
    }

    fn health_check(&self) -> Result<(), EngineError> {
        self.inner.health_check()
    }

    fn supports_swap(&self) -> bool {
        self.inner.supports_swap()
    }

    fn is_staged(&self) -> bool {
        self.inner.is_staged()
    }

    async fn stage(&self, path: &std::path::Path) -> Result<(), EngineError> {
        self.inner.stage(path).await
    }

    async fn swap(&self) -> Result<(), EngineError> {
        self.inner.swap().await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::StreamExt;
    use kapsl_engine_api::EngineError;
    use kapsl_engine_api::TensorDtype;
    use prometheus::{Encoder, TextEncoder};

    struct MockEngine;

    fn raw_media_request(payload_len: i64) -> InferenceRequest {
        InferenceRequest::new(BinaryTensorPacket {
            shape: vec![payload_len],
            dtype: TensorDtype::Uint8,
            data: vec![0],
        })
    }

    fn metrics_text(registry: &prometheus::Registry) -> String {
        let mut output = Vec::new();
        TextEncoder::new()
            .encode(&registry.gather(), &mut output)
            .expect("encode metrics");
        String::from_utf8(output).expect("metrics are UTF-8")
    }

    #[async_trait]
    impl Engine for MockEngine {
        fn kv_capabilities(&self) -> KvBackendCapabilities {
            KvBackendCapabilities::opaque_connected()
        }

        fn planned_memory(
            &self,
            _model_path: &std::path::Path,
        ) -> Result<kapsl_engine_api::MemoryReport, EngineError> {
            Ok(kapsl_engine_api::MemoryReport::single(
                "mock:planned",
                kapsl_engine_api::MemoryDomain::Host,
                kapsl_engine_api::MemoryAllocationClass::ModelSession,
                11,
            ))
        }

        async fn load(&mut self, _model_path: &std::path::Path) -> Result<(), EngineError> {
            Ok(())
        }

        fn actual_memory(&self) -> kapsl_engine_api::MemoryReport {
            kapsl_engine_api::MemoryReport::single(
                "mock:actual",
                kapsl_engine_api::MemoryDomain::Host,
                kapsl_engine_api::MemoryAllocationClass::ModelSession,
                22,
            )
        }

        fn planned_request_memory(
            &self,
            _request: &InferenceRequest,
        ) -> kapsl_engine_api::MemoryReport {
            kapsl_engine_api::MemoryReport::single(
                "mock:request",
                kapsl_engine_api::MemoryDomain::Host,
                kapsl_engine_api::MemoryAllocationClass::RequestTransient,
                33,
            )
        }

        fn infer(&self, request: &InferenceRequest) -> Result<BinaryTensorPacket, EngineError> {
            Ok(request.input.clone())
        }

        fn supports_openai_wire(&self) -> bool {
            true
        }

        fn planned_openai_wire_request_memory(&self, request: &OpenAiWireRequest) -> MemoryReport {
            MemoryReport::single(
                "mock:wire-request",
                kapsl_engine_api::MemoryDomain::Host,
                kapsl_engine_api::MemoryAllocationClass::RequestTransient,
                request.body.len(),
            )
        }

        async fn infer_openai_wire(
            &self,
            request: &OpenAiWireRequest,
        ) -> Result<OpenAiWireResponse, EngineError> {
            Ok(OpenAiWireResponse {
                head: kapsl_engine_api::OpenAiWireResponseHead::new(200, Vec::new())?,
                body: request.body.clone(),
            })
        }

        async fn infer_openai_wire_stream(
            &self,
            request: &OpenAiWireRequest,
        ) -> Result<OpenAiWireStreamResponse, EngineError> {
            Ok(OpenAiWireStreamResponse {
                head: kapsl_engine_api::OpenAiWireResponseHead::new(200, Vec::new())?,
                body: Box::pin(futures::stream::once({
                    let body = request.body.clone();
                    async move { Ok(body) }
                })),
            })
        }

        fn infer_stream(
            &self,
            request: &InferenceRequest,
        ) -> std::pin::Pin<
            Box<dyn futures::stream::Stream<Item = Result<BinaryTensorPacket, EngineError>> + Send>,
        > {
            let result = Ok(request.input.clone());
            Box::pin(futures::stream::once(async move { result }))
        }

        fn unload(&mut self) {}

        fn metrics(&self) -> kapsl_engine_api::EngineMetrics {
            kapsl_engine_api::EngineMetrics::default()
        }

        fn health_check(&self) -> Result<(), EngineError> {
            Ok(())
        }
    }

    #[tokio::test]
    async fn test_monitoring_middleware() {
        let engine = MockEngine;
        let registry = std::sync::Arc::new(prometheus::Registry::new());
        let middleware = MonitoringMiddleware::new(
            engine,
            "test_model".to_string(),
            "v1".to_string(),
            &registry,
        );

        let input = BinaryTensorPacket {
            shape: vec![1, 1],
            dtype: TensorDtype::Float32,
            data: vec![0, 0, 0, 0],
        };
        let request = InferenceRequest {
            input,
            additional_inputs: Vec::new(),
            session_id: None,
            metadata: None,
            cancellation: None,
        };

        let result = middleware.infer(&request);

        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn openai_wire_methods_delegate_and_stream_metrics_finish_on_drop() {
        let registry = std::sync::Arc::new(prometheus::Registry::new());
        let middleware =
            MonitoringMiddleware::new(MockEngine, "wire".to_string(), "v1".to_string(), &registry);
        let unary = OpenAiWireRequest::new(
            kapsl_engine_api::OpenAiWireEndpoint::ChatCompletions,
            kapsl_engine_api::OpenAiWireFormat::Json,
            b"unary".to_vec(),
        );
        assert!(middleware.supports_openai_wire());
        assert_eq!(
            middleware
                .planned_openai_wire_request_memory(&unary)
                .bytes_for_domain(&kapsl_engine_api::MemoryDomain::Host),
            5
        );
        assert_eq!(
            middleware.infer_openai_wire(&unary).await.unwrap().body,
            b"unary"
        );

        let streaming = OpenAiWireRequest::new(
            kapsl_engine_api::OpenAiWireEndpoint::ChatCompletions,
            kapsl_engine_api::OpenAiWireFormat::ServerSentEvents,
            b"stream".to_vec(),
        );
        let mut response = middleware
            .infer_openai_wire_stream(&streaming)
            .await
            .unwrap();
        assert_eq!(response.body.next().await.unwrap().unwrap(), b"stream");
        drop(response);

        let text = metrics_text(&registry);
        assert!(text.contains("kapsl_inference_total"));
        assert!(text.contains("model=\"wire\""));
    }

    #[test]
    fn cross_domain_memory_reports_are_delegated_without_loss() {
        let registry = std::sync::Arc::new(prometheus::Registry::new());
        let middleware = MonitoringMiddleware::new(
            MockEngine,
            "memory".to_string(),
            "v1".to_string(),
            &registry,
        );
        let request = raw_media_request(1);

        assert_eq!(
            middleware
                .planned_memory(std::path::Path::new("model"))
                .unwrap()
                .allocations[0]
                .bytes,
            11
        );
        assert_eq!(middleware.actual_memory().allocations[0].bytes, 22);
        assert_eq!(
            middleware.kv_capabilities(),
            KvBackendCapabilities::opaque_connected()
        );
        assert_eq!(
            middleware.planned_request_memory(&request).allocations[0].bytes,
            33
        );
    }

    #[test]
    fn single_raw_media_request_records_dispatch_size_one() {
        let registry = std::sync::Arc::new(prometheus::Registry::new());
        let middleware = MonitoringMiddleware::new(
            MockEngine,
            "vision".to_string(),
            "v1".to_string(),
            &registry,
        );

        middleware.infer(&raw_media_request(173_131)).unwrap();

        let metrics = metrics_text(&registry);
        assert!(metrics.contains(r#"kapsl_batch_size_bucket{model="vision",le="1"} 1"#));
        assert!(metrics.contains(r#"kapsl_batch_size_sum{model="vision"} 1"#));
        assert!(metrics.contains(r#"kapsl_batch_size_count{model="vision"} 1"#));
    }

    #[test]
    fn infer_batch_records_one_observation_for_request_group() {
        let registry = std::sync::Arc::new(prometheus::Registry::new());
        let middleware =
            MonitoringMiddleware::new(MockEngine, "audio".to_string(), "v1".to_string(), &registry);
        let requests = (0..4)
            .map(|_| raw_media_request(264_000))
            .collect::<Vec<_>>();

        middleware.infer_batch(&requests).unwrap();

        let metrics = metrics_text(&registry);
        assert!(metrics.contains(r#"kapsl_batch_size_bucket{model="audio",le="4"} 1"#));
        assert!(metrics.contains(r#"kapsl_batch_size_sum{model="audio"} 4"#));
        assert!(metrics.contains(r#"kapsl_batch_size_count{model="audio"} 1"#));
    }

    #[tokio::test]
    async fn streaming_raw_media_request_records_dispatch_size_one() {
        let registry = std::sync::Arc::new(prometheus::Registry::new());
        let middleware = MonitoringMiddleware::new(
            MockEngine,
            "streaming-audio".to_string(),
            "v1".to_string(),
            &registry,
        );

        let mut stream = middleware.infer_stream(&raw_media_request(264_000));
        while stream.next().await.is_some() {}

        let metrics = metrics_text(&registry);
        assert!(metrics.contains(r#"kapsl_batch_size_bucket{model="streaming-audio",le="1"} 1"#));
        assert!(metrics.contains(r#"kapsl_batch_size_sum{model="streaming-audio"} 1"#));
        assert!(metrics.contains(r#"kapsl_batch_size_count{model="streaming-audio"} 1"#));
    }
}

#[cfg(test)]
#[path = "middleware_tests.rs"]
mod middleware_tests;
