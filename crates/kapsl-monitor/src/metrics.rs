use prometheus::{
    GaugeVec, HistogramOpts, HistogramVec, IntCounterVec, IntGaugeVec, Opts, Registry,
};
use std::sync::Arc;

#[derive(Debug, Clone)]
pub struct KapslMetrics {
    pub registry: Arc<Registry>,
    pub inference_latency: HistogramVec,
    pub ttft_latency: HistogramVec,
    pub inference_count: IntCounterVec,
    pub active_inferences: IntGaugeVec,
    pub batch_size_hist: HistogramVec,
    pub queue_wait_hist: HistogramVec,
    // Pool-specific metrics
    pub pool_active_replicas: IntGaugeVec,
    pub pool_queue_depth_high: IntGaugeVec,
    pub pool_queue_depth_low: IntGaugeVec,
    pub pool_healthy_replicas: IntGaugeVec,
    // KV cache metrics
    pub kv_cache_bytes_used: IntGaugeVec,
    pub kv_cache_bytes_capacity: IntGaugeVec,
    pub kv_cache_blocks_total: IntGaugeVec,
    pub kv_cache_blocks_free: IntGaugeVec,
    pub kv_cache_sequences: IntGaugeVec,
    pub kv_cache_evicted_blocks: IntGaugeVec,
    pub kv_cache_evicted_sequences: IntGaugeVec,
    pub kv_cache_packed_layers: IntGaugeVec,
    pub kv_cache_cpu_offloaded_blocks: IntGaugeVec,
    pub prompt_tokens_total: IntGaugeVec,
    pub generated_tokens_total: IntGaugeVec,
    pub decode_steps_total: IntGaugeVec,
    pub decode_tokens_evaluated_total: IntGaugeVec,
    pub kv_partial_reuse_hits_total: IntGaugeVec,
    pub kv_partial_reuse_tokens_saved_total: IntGaugeVec,
    /// Per-model engine health: 0 = healthy, 1 = degraded, 2 = dead.
    pub engine_health: IntGaugeVec,
    pub onnx_session_pool_total: IntGaugeVec,
    pub onnx_session_pool_idle: IntGaugeVec,
    pub onnx_session_pool_waits_total: IntGaugeVec,
    pub onnx_session_pool_wait_seconds_total: GaugeVec,
    // Runtime-owned per-device memory authority metrics.
    pub device_memory_budget_bytes: IntGaugeVec,
    pub device_memory_pooled_bytes: IntGaugeVec,
    pub device_memory_planned_external_bytes: IntGaugeVec,
    pub device_memory_external_bytes: IntGaugeVec,
    pub device_memory_available_bytes: IntGaugeVec,
    /// Most recent time-to-first-token (milliseconds) per model. Surfaced on the
    /// runtime `/api/models` so control-plane autoscalers can scale on TTFT SLOs.
    pub model_ttft_ms: GaugeVec,
}

impl KapslMetrics {
    pub fn new(registry: &Arc<Registry>) -> Self {
        let inference_latency = HistogramVec::new(
            HistogramOpts::new(
                "kapsl_inference_latency_seconds",
                "Inference latency (seconds)",
            )
            .buckets(vec![
                0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0,
            ]),
            &["model", "version", "status"],
        )
        .unwrap();

        let ttft_latency = HistogramVec::new(
            HistogramOpts::new("kapsl_ttft_seconds", "Time to first token (seconds)").buckets(
                vec![0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0],
            ),
            &["model", "version", "status"],
        )
        .unwrap();

        let inference_count = IntCounterVec::new(
            Opts::new("kapsl_inference_total", "Number of inferences"),
            &["model", "status"],
        )
        .unwrap();

        let active_inferences = IntGaugeVec::new(
            Opts::new("kapsl_active_inferences", "Active running inferences"),
            &["model"],
        )
        .unwrap();

        let batch_size_hist = HistogramVec::new(
            HistogramOpts::new("kapsl_batch_size", "Number of requests per engine dispatch")
                .buckets(vec![1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0]),
            &["model"],
        )
        .unwrap();

        let queue_wait_hist = HistogramVec::new(
            HistogramOpts::new("kapsl_queue_wait_seconds", "Scheduler queue wait time"),
            &["model"],
        )
        .unwrap();

        // Pool-specific metrics
        let pool_active_replicas = IntGaugeVec::new(
            Opts::new(
                "kapsl_pool_active_replicas",
                "Number of active replicas in the pool",
            ),
            &["model"],
        )
        .unwrap();

        let pool_queue_depth_high = IntGaugeVec::new(
            Opts::new(
                "kapsl_pool_queue_depth_high",
                "High priority queue depth across all replicas",
            ),
            &["model"],
        )
        .unwrap();

        let pool_queue_depth_low = IntGaugeVec::new(
            Opts::new(
                "kapsl_pool_queue_depth_low",
                "Low priority queue depth across all replicas",
            ),
            &["model"],
        )
        .unwrap();

        let pool_healthy_replicas = IntGaugeVec::new(
            Opts::new(
                "kapsl_pool_healthy_replicas",
                "Number of healthy replicas in the pool",
            ),
            &["model"],
        )
        .unwrap();

        let kv_cache_bytes_used = IntGaugeVec::new(
            Opts::new("kapsl_kv_cache_bytes_used", "KV cache bytes currently used"),
            &["model"],
        )
        .unwrap();
        let kv_cache_bytes_capacity = IntGaugeVec::new(
            Opts::new(
                "kapsl_kv_cache_bytes_capacity",
                "KV cache total byte capacity",
            ),
            &["model"],
        )
        .unwrap();
        let kv_cache_blocks_total = IntGaugeVec::new(
            Opts::new("kapsl_kv_cache_blocks_total", "KV cache total blocks"),
            &["model"],
        )
        .unwrap();
        let kv_cache_blocks_free = IntGaugeVec::new(
            Opts::new("kapsl_kv_cache_blocks_free", "KV cache free blocks"),
            &["model"],
        )
        .unwrap();
        let kv_cache_sequences = IntGaugeVec::new(
            Opts::new(
                "kapsl_kv_cache_sequences",
                "Active sequences tracked in KV cache",
            ),
            &["model"],
        )
        .unwrap();
        let kv_cache_evicted_blocks = IntGaugeVec::new(
            Opts::new(
                "kapsl_kv_cache_evicted_blocks_total",
                "Total evicted KV blocks",
            ),
            &["model"],
        )
        .unwrap();
        let kv_cache_evicted_sequences = IntGaugeVec::new(
            Opts::new(
                "kapsl_kv_cache_evicted_sequences_total",
                "Total evicted KV sequences",
            ),
            &["model"],
        )
        .unwrap();
        let kv_cache_packed_layers = IntGaugeVec::new(
            Opts::new(
                "kapsl_kv_cache_packed_layers",
                "Number of packed KV layers cached",
            ),
            &["model"],
        )
        .unwrap();
        let kv_cache_cpu_offloaded_blocks = IntGaugeVec::new(
            Opts::new(
                "kapsl_kv_cache_cpu_offloaded_blocks",
                "KV cache blocks currently offloaded to CPU memory",
            ),
            &["model"],
        )
        .unwrap();
        let prompt_tokens_total = IntGaugeVec::new(
            Opts::new(
                "kapsl_prompt_tokens_total",
                "Cumulative prompt tokens processed by the engine",
            ),
            &["model"],
        )
        .unwrap();
        let generated_tokens_total = IntGaugeVec::new(
            Opts::new(
                "kapsl_generated_tokens_total",
                "Cumulative generated tokens produced by the engine",
            ),
            &["model"],
        )
        .unwrap();
        let decode_steps_total = IntGaugeVec::new(
            Opts::new(
                "kapsl_decode_steps_total",
                "Cumulative decode reserve steps observed by the engine",
            ),
            &["model"],
        )
        .unwrap();
        let decode_tokens_evaluated_total = IntGaugeVec::new(
            Opts::new(
                "kapsl_decode_tokens_evaluated_total",
                "Cumulative logical decode tokens evaluated by the engine",
            ),
            &["model"],
        )
        .unwrap();
        let kv_partial_reuse_hits_total = IntGaugeVec::new(
            Opts::new(
                "kapsl_kv_partial_reuse_hits_total",
                "Cumulative same-session partial KV reuse hits",
            ),
            &["model"],
        )
        .unwrap();
        let kv_partial_reuse_tokens_saved_total = IntGaugeVec::new(
            Opts::new(
                "kapsl_kv_partial_reuse_tokens_saved_total",
                "Cumulative decode tokens avoided by same-session partial KV reuse",
            ),
            &["model"],
        )
        .unwrap();
        let engine_health = IntGaugeVec::new(
            Opts::new(
                "kapsl_engine_health",
                "Per-model engine health: 0 = healthy, 1 = degraded, 2 = dead",
            ),
            &["model"],
        )
        .unwrap();
        let onnx_session_pool_total = IntGaugeVec::new(
            Opts::new(
                "kapsl_onnx_session_pool_total",
                "Total ONNX Runtime sessions allocated across session pools",
            ),
            &["model"],
        )
        .unwrap();
        let onnx_session_pool_idle = IntGaugeVec::new(
            Opts::new(
                "kapsl_onnx_session_pool_idle",
                "Idle ONNX Runtime sessions available across session pools",
            ),
            &["model"],
        )
        .unwrap();
        let onnx_session_pool_waits_total = IntGaugeVec::new(
            Opts::new(
                "kapsl_onnx_session_pool_waits_total",
                "Cumulative waits for an ONNX Runtime session",
            ),
            &["model"],
        )
        .unwrap();
        let onnx_session_pool_wait_seconds_total = GaugeVec::new(
            Opts::new(
                "kapsl_onnx_session_pool_wait_seconds_total",
                "Cumulative seconds spent waiting for an ONNX Runtime session",
            ),
            &["model"],
        )
        .unwrap();
        let model_ttft_ms = GaugeVec::new(
            Opts::new(
                "kapsl_model_ttft_ms",
                "Most recent time-to-first-token in milliseconds per model",
            ),
            &["model"],
        )
        .unwrap();
        let device_memory_budget_bytes = IntGaugeVec::new(
            Opts::new(
                "kapsl_device_memory_budget_bytes",
                "Runtime admission budget per CUDA device",
            ),
            &["device"],
        )
        .unwrap();
        let device_memory_pooled_bytes = IntGaugeVec::new(
            Opts::new(
                "kapsl_device_memory_pooled_bytes",
                "Bytes owned by the runtime CUDA pool per device",
            ),
            &["device"],
        )
        .unwrap();
        let device_memory_planned_external_bytes = IntGaugeVec::new(
            Opts::new(
                "kapsl_device_memory_planned_external_bytes",
                "External CUDA bytes currently reserved for in-progress loads",
            ),
            &["device"],
        )
        .unwrap();
        let device_memory_external_bytes = IntGaugeVec::new(
            Opts::new(
                "kapsl_device_memory_external_bytes",
                "Backend and library CUDA bytes charged outside the runtime pool",
            ),
            &["device"],
        )
        .unwrap();
        let device_memory_available_bytes = IntGaugeVec::new(
            Opts::new(
                "kapsl_device_memory_available_bytes",
                "Uncommitted runtime admission budget per CUDA device",
            ),
            &["device"],
        )
        .unwrap();

        registry
            .register(Box::new(inference_latency.clone()))
            .expect("Failed to register inference_latency");
        registry
            .register(Box::new(ttft_latency.clone()))
            .expect("Failed to register ttft_latency");
        registry
            .register(Box::new(model_ttft_ms.clone()))
            .expect("Failed to register model_ttft_ms");
        registry
            .register(Box::new(inference_count.clone()))
            .expect("Failed to register inference_count");
        registry
            .register(Box::new(active_inferences.clone()))
            .expect("Failed to register active_inferences");
        registry
            .register(Box::new(batch_size_hist.clone()))
            .expect("Failed to register batch_size_hist");
        registry
            .register(Box::new(queue_wait_hist.clone()))
            .expect("Failed to register queue_wait_hist");
        registry
            .register(Box::new(pool_active_replicas.clone()))
            .expect("Failed to register pool_active_replicas");
        registry
            .register(Box::new(pool_queue_depth_high.clone()))
            .expect("Failed to register pool_queue_depth_high");
        registry
            .register(Box::new(pool_queue_depth_low.clone()))
            .expect("Failed to register pool_queue_depth_low");
        registry
            .register(Box::new(pool_healthy_replicas.clone()))
            .expect("Failed to register pool_healthy_replicas");
        registry
            .register(Box::new(kv_cache_bytes_used.clone()))
            .expect("Failed to register kv_cache_bytes_used");
        registry
            .register(Box::new(kv_cache_bytes_capacity.clone()))
            .expect("Failed to register kv_cache_bytes_capacity");
        registry
            .register(Box::new(kv_cache_blocks_total.clone()))
            .expect("Failed to register kv_cache_blocks_total");
        registry
            .register(Box::new(kv_cache_blocks_free.clone()))
            .expect("Failed to register kv_cache_blocks_free");
        registry
            .register(Box::new(kv_cache_sequences.clone()))
            .expect("Failed to register kv_cache_sequences");
        registry
            .register(Box::new(kv_cache_evicted_blocks.clone()))
            .expect("Failed to register kv_cache_evicted_blocks");
        registry
            .register(Box::new(kv_cache_evicted_sequences.clone()))
            .expect("Failed to register kv_cache_evicted_sequences");
        registry
            .register(Box::new(kv_cache_packed_layers.clone()))
            .expect("Failed to register kv_cache_packed_layers");
        registry
            .register(Box::new(kv_cache_cpu_offloaded_blocks.clone()))
            .expect("Failed to register kv_cache_cpu_offloaded_blocks");
        registry
            .register(Box::new(prompt_tokens_total.clone()))
            .expect("Failed to register prompt_tokens_total");
        registry
            .register(Box::new(generated_tokens_total.clone()))
            .expect("Failed to register generated_tokens_total");
        registry
            .register(Box::new(decode_steps_total.clone()))
            .expect("Failed to register decode_steps_total");
        registry
            .register(Box::new(decode_tokens_evaluated_total.clone()))
            .expect("Failed to register decode_tokens_evaluated_total");
        registry
            .register(Box::new(kv_partial_reuse_hits_total.clone()))
            .expect("Failed to register kv_partial_reuse_hits_total");
        registry
            .register(Box::new(kv_partial_reuse_tokens_saved_total.clone()))
            .expect("Failed to register kv_partial_reuse_tokens_saved_total");
        registry
            .register(Box::new(engine_health.clone()))
            .expect("Failed to register engine_health");
        registry
            .register(Box::new(onnx_session_pool_total.clone()))
            .expect("Failed to register onnx_session_pool_total");
        registry
            .register(Box::new(onnx_session_pool_idle.clone()))
            .expect("Failed to register onnx_session_pool_idle");
        registry
            .register(Box::new(onnx_session_pool_waits_total.clone()))
            .expect("Failed to register onnx_session_pool_waits_total");
        registry
            .register(Box::new(onnx_session_pool_wait_seconds_total.clone()))
            .expect("Failed to register onnx_session_pool_wait_seconds_total");
        registry
            .register(Box::new(device_memory_budget_bytes.clone()))
            .expect("Failed to register device_memory_budget_bytes");
        registry
            .register(Box::new(device_memory_pooled_bytes.clone()))
            .expect("Failed to register device_memory_pooled_bytes");
        registry
            .register(Box::new(device_memory_planned_external_bytes.clone()))
            .expect("Failed to register device_memory_planned_external_bytes");
        registry
            .register(Box::new(device_memory_external_bytes.clone()))
            .expect("Failed to register device_memory_external_bytes");
        registry
            .register(Box::new(device_memory_available_bytes.clone()))
            .expect("Failed to register device_memory_available_bytes");

        Self {
            registry: registry.clone(),
            inference_latency,
            ttft_latency,
            inference_count,
            active_inferences,
            batch_size_hist,
            queue_wait_hist,
            pool_active_replicas,
            pool_queue_depth_high,
            pool_queue_depth_low,
            pool_healthy_replicas,
            kv_cache_bytes_used,
            kv_cache_bytes_capacity,
            kv_cache_blocks_total,
            kv_cache_blocks_free,
            kv_cache_sequences,
            kv_cache_evicted_blocks,
            kv_cache_evicted_sequences,
            kv_cache_packed_layers,
            kv_cache_cpu_offloaded_blocks,
            prompt_tokens_total,
            generated_tokens_total,
            decode_steps_total,
            decode_tokens_evaluated_total,
            kv_partial_reuse_hits_total,
            kv_partial_reuse_tokens_saved_total,
            engine_health,
            onnx_session_pool_total,
            onnx_session_pool_idle,
            onnx_session_pool_waits_total,
            onnx_session_pool_wait_seconds_total,
            device_memory_budget_bytes,
            device_memory_pooled_bytes,
            device_memory_planned_external_bytes,
            device_memory_external_bytes,
            device_memory_available_bytes,
            model_ttft_ms,
        }
    }

    pub fn set_kv_cache_metrics(&self, model: &str, metrics: &kapsl_engine_api::EngineMetrics) {
        self.kv_cache_bytes_used
            .with_label_values(&[model])
            .set(metrics.kv_cache_bytes_used as i64);
        self.kv_cache_bytes_capacity
            .with_label_values(&[model])
            .set(metrics.kv_cache_bytes_capacity as i64);
        self.kv_cache_blocks_total
            .with_label_values(&[model])
            .set(metrics.kv_cache_blocks_total as i64);
        self.kv_cache_blocks_free
            .with_label_values(&[model])
            .set(metrics.kv_cache_blocks_free as i64);
        self.kv_cache_sequences
            .with_label_values(&[model])
            .set(metrics.kv_cache_sequences as i64);
        self.kv_cache_evicted_blocks
            .with_label_values(&[model])
            .set(metrics.kv_cache_evicted_blocks as i64);
        self.kv_cache_evicted_sequences
            .with_label_values(&[model])
            .set(metrics.kv_cache_evicted_sequences as i64);
        self.kv_cache_packed_layers
            .with_label_values(&[model])
            .set(metrics.kv_cache_packed_layers as i64);
        self.kv_cache_cpu_offloaded_blocks
            .with_label_values(&[model])
            .set(metrics.kv_cache_cpu_offloaded_blocks as i64);
        self.prompt_tokens_total
            .with_label_values(&[model])
            .set(metrics.prompt_tokens_total as i64);
        self.generated_tokens_total
            .with_label_values(&[model])
            .set(metrics.generated_tokens_total as i64);
        self.decode_steps_total
            .with_label_values(&[model])
            .set(metrics.decode_steps_total as i64);
        self.decode_tokens_evaluated_total
            .with_label_values(&[model])
            .set(metrics.decode_tokens_evaluated_total as i64);
        self.kv_partial_reuse_hits_total
            .with_label_values(&[model])
            .set(metrics.kv_partial_reuse_hits_total as i64);
        self.kv_partial_reuse_tokens_saved_total
            .with_label_values(&[model])
            .set(metrics.kv_partial_reuse_tokens_saved_total as i64);
        self.engine_health
            .with_label_values(&[model])
            .set(metrics.engine_health as i64);
        self.onnx_session_pool_total
            .with_label_values(&[model])
            .set(metrics.onnx_session_pool_total as i64);
        self.onnx_session_pool_idle
            .with_label_values(&[model])
            .set(metrics.onnx_session_pool_idle as i64);
        self.onnx_session_pool_waits_total
            .with_label_values(&[model])
            .set(metrics.onnx_session_pool_waits_total as i64);
        self.onnx_session_pool_wait_seconds_total
            .with_label_values(&[model])
            .set(metrics.onnx_session_pool_wait_seconds_total);
    }
}

#[cfg(test)]
#[path = "metrics_tests.rs"]
mod metrics_tests;
