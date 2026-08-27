use prometheus::{
    GaugeVec, HistogramOpts, HistogramVec, IntCounterVec, IntGaugeVec, Opts, Registry,
};
use std::{
    collections::{HashMap, HashSet},
    sync::{Arc, Mutex},
};

/// Pool-wide values exported for one CUDA device.
///
/// This transport type intentionally contains only monitoring primitives, so
/// runtime callers can populate it without exposing allocator or HAL types to
/// this crate. `fragmentation_ratio` is expected to be in the inclusive range
/// `0.0..=1.0`, where zero means that all free space is contiguous.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct GpuDevicePoolMetrics {
    pub allocated_bytes: u64,
    pub live_allocations: u64,
    pub free_bytes: u64,
    pub free_ranges: u64,
    pub largest_free_range_bytes: u64,
    pub fragmentation_ratio: f64,
    pub owners: Vec<GpuDevicePoolOwnerMetrics>,
}

/// Per-owner values exported for one CUDA device pool.
///
/// `owner` is a stable, bounded identity such as `onnx`, `gguf_kv:42`, or
/// `native_kv:42`. It deliberately combines owner kind and model identity into
/// one label instead of creating additional high-cardinality label dimensions.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct GpuDevicePoolOwnerMetrics {
    pub owner: String,
    pub usage_bytes: u64,
    pub guaranteed_bytes: u64,
    pub max_bytes: u64,
    pub admitted: bool,
    pub allocatable_bytes: u64,
}

/// Managed-vLLM-specific observability kept separate from backend-neutral
/// engine metrics. Labels are bounded runtime identities: model ID, replica
/// ID, CUDA ordinal, bridge mode, and a fixed stage/error vocabulary.
#[derive(Debug, Clone)]
pub struct ManagedVllmMetrics {
    pub kv_requested_bytes: IntGaugeVec,
    pub kv_granted_bytes: IntGaugeVec,
    pub kv_minimum_bytes: IntGaugeVec,
    pub kv_backing_bytes: IntGaugeVec,
    pub kv_logical_leased_bytes: IntGaugeVec,
    pub kv_blocks_total: IntGaugeVec,
    pub kv_blocks_allocated: IntGaugeVec,
    pub kv_blocks_active: IntGaugeVec,
    pub kv_blocks_idle: IntGaugeVec,
    pub kv_quarantine_bytes: IntGaugeVec,
    pub effective_target_concurrency: IntGaugeVec,
    pub provisional_reservation_age_seconds: GaugeVec,
    pub provisional_reservation_state: IntGaugeVec,
    pub restart_generation: IntGaugeVec,
    pub planning_reductions_total: IntCounterVec,
    pub planning_rejections_total: IntCounterVec,
    pub bridge_stage_seconds: HistogramVec,
    pub bridge_requests_total: IntCounterVec,
    pub bridge_relayed_bytes_total: IntCounterVec,
    pub bridge_relayed_chunks_total: IntCounterVec,
    pub bridge_active_streams: IntGaugeVec,
    pub bridge_connection_attempts_total: IntCounterVec,
    pub bridge_open_connections: IntGaugeVec,
    pub bridge_cancellations_total: IntCounterVec,
    pub bridge_upstream_errors_total: IntCounterVec,
}

impl ManagedVllmMetrics {
    fn new(registry: &Registry) -> Self {
        let memory_labels = &["model", "replica", "device"];
        let replica_labels = &["model", "replica"];
        let bridge_labels = &["model", "replica", "mode"];
        let kv_requested_bytes = IntGaugeVec::new(
            Opts::new(
                "kapsl_managed_vllm_kv_requested_bytes",
                "Desired exact vLLM KV bytes per replica and CUDA device",
            ),
            memory_labels,
        )
        .unwrap();
        let kv_granted_bytes = IntGaugeVec::new(
            Opts::new(
                "kapsl_managed_vllm_kv_granted_bytes",
                "MemoryAuthority-granted vLLM KV bytes per replica and CUDA device",
            ),
            memory_labels,
        )
        .unwrap();
        let kv_minimum_bytes = IntGaugeVec::new(
            Opts::new(
                "kapsl_managed_vllm_kv_minimum_bytes",
                "Minimum bytes required for one maximum-length vLLM sequence",
            ),
            memory_labels,
        )
        .unwrap();
        let kv_backing_bytes = IntGaugeVec::new(
            Opts::new(
                "kapsl_managed_vllm_kv_backing_bytes",
                "Physical CUDA-IPC KV backing bytes",
            ),
            memory_labels,
        )
        .unwrap();
        let kv_logical_leased_bytes = IntGaugeVec::new(
            Opts::new(
                "kapsl_managed_vllm_kv_logical_leased_bytes",
                "Logical request-leased bytes inside the physical vLLM KV backing",
            ),
            memory_labels,
        )
        .unwrap();
        let block_metric = |name: &str, help: &str| {
            IntGaugeVec::new(Opts::new(name, help), memory_labels).unwrap()
        };
        let kv_blocks_total = block_metric(
            "kapsl_managed_vllm_kv_blocks_total",
            "Total certified block capacity of the vLLM backing",
        );
        let kv_blocks_allocated = block_metric(
            "kapsl_managed_vllm_kv_blocks_allocated",
            "Blocks physically allocated in the fixed vLLM backing",
        );
        let kv_blocks_active = block_metric(
            "kapsl_managed_vllm_kv_blocks_active",
            "Blocks covered by active logical request leases",
        );
        let kv_blocks_idle = block_metric(
            "kapsl_managed_vllm_kv_blocks_idle",
            "Immediately reusable idle blocks inside the vLLM backing",
        );
        let kv_quarantine_bytes = IntGaugeVec::new(
            Opts::new(
                "kapsl_managed_vllm_kv_quarantine_bytes",
                "Bytes retained after an ambiguous shared-pool release",
            ),
            memory_labels,
        )
        .unwrap();
        let effective_target_concurrency = IntGaugeVec::new(
            Opts::new(
                "kapsl_managed_vllm_effective_target_concurrency",
                "Whole maximum-length sequences admitted by the exact KV grant",
            ),
            replica_labels,
        )
        .unwrap();
        let provisional_reservation_age_seconds = GaugeVec::new(
            Opts::new(
                "kapsl_managed_vllm_provisional_reservation_age_seconds",
                "Age of an unconsumed exact-KV reservation",
            ),
            replica_labels,
        )
        .unwrap();
        let provisional_reservation_state = IntGaugeVec::new(
            Opts::new(
                "kapsl_managed_vllm_provisional_reservation_state",
                "One-hot provisional reservation state",
            ),
            &["model", "replica", "state"],
        )
        .unwrap();
        let restart_generation = IntGaugeVec::new(
            Opts::new(
                "kapsl_managed_vllm_restart_generation",
                "Current supervised managed-vLLM process generation",
            ),
            replica_labels,
        )
        .unwrap();
        let planning_reductions_total = IntCounterVec::new(
            Opts::new(
                "kapsl_managed_vllm_planning_reductions_total",
                "Exact-KV planning reductions by reason",
            ),
            &["model", "replica", "reason"],
        )
        .unwrap();
        let planning_rejections_total = IntCounterVec::new(
            Opts::new(
                "kapsl_managed_vllm_planning_rejections_total",
                "Exact-KV planning or authority rejections by reason",
            ),
            &["model", "replica", "reason"],
        )
        .unwrap();
        let bridge_stage_seconds = HistogramVec::new(
            HistogramOpts::new(
                "kapsl_managed_vllm_bridge_stage_seconds",
                "Managed-vLLM bridge latency by bounded processing stage",
            )
            .buckets(vec![
                0.0001, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0,
                5.0, 10.0,
            ]),
            &["model", "replica", "mode", "stage"],
        )
        .unwrap();
        let bridge_requests_total = IntCounterVec::new(
            Opts::new(
                "kapsl_managed_vllm_bridge_requests_total",
                "Managed-vLLM bridge requests by mode and streaming shape",
            ),
            &["model", "replica", "mode", "stream"],
        )
        .unwrap();
        let bridge_relayed_bytes_total = IntCounterVec::new(
            Opts::new(
                "kapsl_managed_vllm_bridge_relayed_bytes_total",
                "Bytes relayed from managed vLLM",
            ),
            bridge_labels,
        )
        .unwrap();
        let bridge_relayed_chunks_total = IntCounterVec::new(
            Opts::new(
                "kapsl_managed_vllm_bridge_relayed_chunks_total",
                "Response chunks relayed from managed vLLM",
            ),
            bridge_labels,
        )
        .unwrap();
        let bridge_active_streams = IntGaugeVec::new(
            Opts::new(
                "kapsl_managed_vllm_bridge_active_streams",
                "Currently active managed-vLLM response streams",
            ),
            bridge_labels,
        )
        .unwrap();
        let bridge_connection_attempts_total = IntCounterVec::new(
            Opts::new(
                "kapsl_managed_vllm_bridge_connection_attempts_total",
                "New private upstream connection attempts by client implementation",
            ),
            &["model", "replica", "client"],
        )
        .unwrap();
        let bridge_open_connections = IntGaugeVec::new(
            Opts::new(
                "kapsl_managed_vllm_bridge_open_connections",
                "Private upstream connections currently owned by the bridge",
            ),
            &["model", "replica", "client"],
        )
        .unwrap();
        let bridge_cancellations_total = IntCounterVec::new(
            Opts::new(
                "kapsl_managed_vllm_bridge_cancellations_total",
                "Managed-vLLM upstream operations cancelled by downstream ownership",
            ),
            bridge_labels,
        )
        .unwrap();
        let bridge_upstream_errors_total = IntCounterVec::new(
            Opts::new(
                "kapsl_managed_vllm_bridge_upstream_errors_total",
                "Managed-vLLM upstream errors by bounded category",
            ),
            &["model", "replica", "mode", "kind"],
        )
        .unwrap();

        macro_rules! register {
            ($collector:ident) => {
                registry
                    .register(Box::new($collector.clone()))
                    .unwrap_or_else(|error| {
                        panic!("Failed to register {}: {error}", stringify!($collector))
                    });
            };
        }
        register!(kv_requested_bytes);
        register!(kv_granted_bytes);
        register!(kv_minimum_bytes);
        register!(kv_backing_bytes);
        register!(kv_logical_leased_bytes);
        register!(kv_blocks_total);
        register!(kv_blocks_allocated);
        register!(kv_blocks_active);
        register!(kv_blocks_idle);
        register!(kv_quarantine_bytes);
        register!(effective_target_concurrency);
        register!(provisional_reservation_age_seconds);
        register!(provisional_reservation_state);
        register!(restart_generation);
        register!(planning_reductions_total);
        register!(planning_rejections_total);
        register!(bridge_stage_seconds);
        register!(bridge_requests_total);
        register!(bridge_relayed_bytes_total);
        register!(bridge_relayed_chunks_total);
        register!(bridge_active_streams);
        register!(bridge_connection_attempts_total);
        register!(bridge_open_connections);
        register!(bridge_cancellations_total);
        register!(bridge_upstream_errors_total);

        Self {
            kv_requested_bytes,
            kv_granted_bytes,
            kv_minimum_bytes,
            kv_backing_bytes,
            kv_logical_leased_bytes,
            kv_blocks_total,
            kv_blocks_allocated,
            kv_blocks_active,
            kv_blocks_idle,
            kv_quarantine_bytes,
            effective_target_concurrency,
            provisional_reservation_age_seconds,
            provisional_reservation_state,
            restart_generation,
            planning_reductions_total,
            planning_rejections_total,
            bridge_stage_seconds,
            bridge_requests_total,
            bridge_relayed_bytes_total,
            bridge_relayed_chunks_total,
            bridge_active_streams,
            bridge_connection_attempts_total,
            bridge_open_connections,
            bridge_cancellations_total,
            bridge_upstream_errors_total,
        }
    }
}

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
    // Runtime-owned CUDA device-pool allocator metrics.
    pub gpu_device_pool_allocated_bytes: IntGaugeVec,
    pub gpu_device_pool_live_allocations: IntGaugeVec,
    pub gpu_device_pool_free_bytes: IntGaugeVec,
    pub gpu_device_pool_free_ranges: IntGaugeVec,
    pub gpu_device_pool_largest_free_range_bytes: IntGaugeVec,
    pub gpu_device_pool_fragmentation_ratio: GaugeVec,
    pub gpu_device_pool_owner_usage_bytes: IntGaugeVec,
    pub gpu_device_pool_owner_quota_guaranteed_bytes: IntGaugeVec,
    pub gpu_device_pool_owner_quota_max_bytes: IntGaugeVec,
    pub gpu_device_pool_owner_admitted: IntGaugeVec,
    pub gpu_device_pool_owner_allocatable_bytes: IntGaugeVec,
    pub managed_vllm: ManagedVllmMetrics,
    gpu_device_pool_owner_labels: Arc<Mutex<HashMap<String, HashSet<String>>>>,
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
        let gpu_device_pool_allocated_bytes = IntGaugeVec::new(
            Opts::new(
                "kapsl_gpu_device_pool_allocated_bytes",
                "Bytes currently allocated from the runtime-owned CUDA device pool",
            ),
            &["device"],
        )
        .unwrap();
        let gpu_device_pool_live_allocations = IntGaugeVec::new(
            Opts::new(
                "kapsl_gpu_device_pool_live_allocations",
                "Number of live allocations in the runtime-owned CUDA device pool",
            ),
            &["device"],
        )
        .unwrap();
        let gpu_device_pool_free_bytes = IntGaugeVec::new(
            Opts::new(
                "kapsl_gpu_device_pool_free_bytes",
                "Bytes currently free in the runtime-owned CUDA device pool",
            ),
            &["device"],
        )
        .unwrap();
        let gpu_device_pool_free_ranges = IntGaugeVec::new(
            Opts::new(
                "kapsl_gpu_device_pool_free_ranges",
                "Number of disjoint free ranges in the runtime-owned CUDA device pool",
            ),
            &["device"],
        )
        .unwrap();
        let gpu_device_pool_largest_free_range_bytes = IntGaugeVec::new(
            Opts::new(
                "kapsl_gpu_device_pool_largest_free_range_bytes",
                "Size of the largest free range in the runtime-owned CUDA device pool",
            ),
            &["device"],
        )
        .unwrap();
        let gpu_device_pool_fragmentation_ratio = GaugeVec::new(
            Opts::new(
                "kapsl_gpu_device_pool_fragmentation_ratio",
                "CUDA device-pool external fragmentation from 0 (contiguous) to 1",
            ),
            &["device"],
        )
        .unwrap();
        let gpu_device_pool_owner_usage_bytes = IntGaugeVec::new(
            Opts::new(
                "kapsl_gpu_device_pool_owner_usage_bytes",
                "CUDA device-pool bytes currently used by an owner",
            ),
            &["device", "owner"],
        )
        .unwrap();
        let gpu_device_pool_owner_quota_guaranteed_bytes = IntGaugeVec::new(
            Opts::new(
                "kapsl_gpu_device_pool_owner_quota_guaranteed_bytes",
                "CUDA device-pool bytes guaranteed to an owner while admitted",
            ),
            &["device", "owner"],
        )
        .unwrap();
        let gpu_device_pool_owner_quota_max_bytes = IntGaugeVec::new(
            Opts::new(
                "kapsl_gpu_device_pool_owner_quota_max_bytes",
                "Maximum CUDA device-pool bytes an owner may use",
            ),
            &["device", "owner"],
        )
        .unwrap();
        let gpu_device_pool_owner_admitted = IntGaugeVec::new(
            Opts::new(
                "kapsl_gpu_device_pool_owner_admitted",
                "Whether a CUDA device-pool owner is admitted (1) or not (0)",
            ),
            &["device", "owner"],
        )
        .unwrap();
        let gpu_device_pool_owner_allocatable_bytes = IntGaugeVec::new(
            Opts::new(
                "kapsl_gpu_device_pool_owner_allocatable_bytes",
                "CUDA device-pool bytes currently allocatable by an owner after policy and fragmentation",
            ),
            &["device", "owner"],
        )
        .unwrap();
        let managed_vllm = ManagedVllmMetrics::new(registry);

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
        registry
            .register(Box::new(gpu_device_pool_allocated_bytes.clone()))
            .expect("Failed to register gpu_device_pool_allocated_bytes");
        registry
            .register(Box::new(gpu_device_pool_live_allocations.clone()))
            .expect("Failed to register gpu_device_pool_live_allocations");
        registry
            .register(Box::new(gpu_device_pool_free_bytes.clone()))
            .expect("Failed to register gpu_device_pool_free_bytes");
        registry
            .register(Box::new(gpu_device_pool_free_ranges.clone()))
            .expect("Failed to register gpu_device_pool_free_ranges");
        registry
            .register(Box::new(gpu_device_pool_largest_free_range_bytes.clone()))
            .expect("Failed to register gpu_device_pool_largest_free_range_bytes");
        registry
            .register(Box::new(gpu_device_pool_fragmentation_ratio.clone()))
            .expect("Failed to register gpu_device_pool_fragmentation_ratio");
        registry
            .register(Box::new(gpu_device_pool_owner_usage_bytes.clone()))
            .expect("Failed to register gpu_device_pool_owner_usage_bytes");
        registry
            .register(Box::new(
                gpu_device_pool_owner_quota_guaranteed_bytes.clone(),
            ))
            .expect("Failed to register gpu_device_pool_owner_quota_guaranteed_bytes");
        registry
            .register(Box::new(gpu_device_pool_owner_quota_max_bytes.clone()))
            .expect("Failed to register gpu_device_pool_owner_quota_max_bytes");
        registry
            .register(Box::new(gpu_device_pool_owner_admitted.clone()))
            .expect("Failed to register gpu_device_pool_owner_admitted");
        registry
            .register(Box::new(gpu_device_pool_owner_allocatable_bytes.clone()))
            .expect("Failed to register gpu_device_pool_owner_allocatable_bytes");

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
            gpu_device_pool_allocated_bytes,
            gpu_device_pool_live_allocations,
            gpu_device_pool_free_bytes,
            gpu_device_pool_free_ranges,
            gpu_device_pool_largest_free_range_bytes,
            gpu_device_pool_fragmentation_ratio,
            gpu_device_pool_owner_usage_bytes,
            gpu_device_pool_owner_quota_guaranteed_bytes,
            gpu_device_pool_owner_quota_max_bytes,
            gpu_device_pool_owner_admitted,
            gpu_device_pool_owner_allocatable_bytes,
            managed_vllm,
            gpu_device_pool_owner_labels: Arc::new(Mutex::new(HashMap::new())),
            model_ttft_ms,
        }
    }

    /// Export the cache, token, health, and session-pool fields from one engine
    /// snapshot to their model-scoped Prometheus collectors.
    ///
    /// This is the canonical [`EngineMetrics`](kapsl_engine_api::EngineMetrics)
    /// mapping for those collectors. Runtime callers should pass the complete
    /// snapshot here rather than assigning individual collectors themselves.
    pub fn set_engine_metrics(&self, model: &str, metrics: &kapsl_engine_api::EngineMetrics) {
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

    /// Backwards-compatible name for [`Self::set_engine_metrics`].
    pub fn set_kv_cache_metrics(&self, model: &str, metrics: &kapsl_engine_api::EngineMetrics) {
        self.set_engine_metrics(model, metrics);
    }

    /// Replace the live CUDA device-pool snapshot for `device`.
    ///
    /// Owner rows omitted from a later snapshot are removed from the
    /// Prometheus registry, preventing unloaded model identities from leaving
    /// stale label series behind. Values wider than Prometheus' signed integer
    /// gauge representation are saturated instead of wrapping.
    pub fn set_gpu_device_pool_metrics(&self, device: &str, snapshot: &GpuDevicePoolMetrics) {
        let mut tracked = self
            .gpu_device_pool_owner_labels
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let current_owners: HashSet<_> = snapshot
            .owners
            .iter()
            .map(|owner| owner.owner.clone())
            .collect();

        if let Some(previous_owners) = tracked.get(device) {
            for stale_owner in previous_owners.difference(&current_owners) {
                self.remove_gpu_device_pool_owner_metrics(device, stale_owner);
            }
        }

        let device_labels = &[device];
        self.gpu_device_pool_allocated_bytes
            .with_label_values(device_labels)
            .set(prometheus_i64(snapshot.allocated_bytes));
        self.gpu_device_pool_live_allocations
            .with_label_values(device_labels)
            .set(prometheus_i64(snapshot.live_allocations));
        self.gpu_device_pool_free_bytes
            .with_label_values(device_labels)
            .set(prometheus_i64(snapshot.free_bytes));
        self.gpu_device_pool_free_ranges
            .with_label_values(device_labels)
            .set(prometheus_i64(snapshot.free_ranges));
        self.gpu_device_pool_largest_free_range_bytes
            .with_label_values(device_labels)
            .set(prometheus_i64(snapshot.largest_free_range_bytes));
        self.gpu_device_pool_fragmentation_ratio
            .with_label_values(device_labels)
            .set(normalize_fragmentation(snapshot.fragmentation_ratio));

        for owner in &snapshot.owners {
            let owner_labels = &[device, owner.owner.as_str()];
            self.gpu_device_pool_owner_usage_bytes
                .with_label_values(owner_labels)
                .set(prometheus_i64(owner.usage_bytes));
            self.gpu_device_pool_owner_quota_guaranteed_bytes
                .with_label_values(owner_labels)
                .set(prometheus_i64(owner.guaranteed_bytes));
            self.gpu_device_pool_owner_quota_max_bytes
                .with_label_values(owner_labels)
                .set(prometheus_i64(owner.max_bytes));
            self.gpu_device_pool_owner_admitted
                .with_label_values(owner_labels)
                .set(i64::from(owner.admitted));
            self.gpu_device_pool_owner_allocatable_bytes
                .with_label_values(owner_labels)
                .set(prometheus_i64(owner.allocatable_bytes));
        }

        tracked.insert(device.to_owned(), current_owners);
    }

    /// Remove all pool-wide and per-owner Prometheus series for `device`.
    pub fn remove_gpu_device_pool_metrics(&self, device: &str) {
        let mut tracked = self
            .gpu_device_pool_owner_labels
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if let Some(owners) = tracked.remove(device) {
            for owner in owners {
                self.remove_gpu_device_pool_owner_metrics(device, &owner);
            }
        }

        let labels = &[device];
        let _ = self
            .gpu_device_pool_allocated_bytes
            .remove_label_values(labels);
        let _ = self
            .gpu_device_pool_live_allocations
            .remove_label_values(labels);
        let _ = self.gpu_device_pool_free_bytes.remove_label_values(labels);
        let _ = self.gpu_device_pool_free_ranges.remove_label_values(labels);
        let _ = self
            .gpu_device_pool_largest_free_range_bytes
            .remove_label_values(labels);
        let _ = self
            .gpu_device_pool_fragmentation_ratio
            .remove_label_values(labels);
    }

    fn remove_gpu_device_pool_owner_metrics(&self, device: &str, owner: &str) {
        let labels = &[device, owner];
        let _ = self
            .gpu_device_pool_owner_usage_bytes
            .remove_label_values(labels);
        let _ = self
            .gpu_device_pool_owner_quota_guaranteed_bytes
            .remove_label_values(labels);
        let _ = self
            .gpu_device_pool_owner_quota_max_bytes
            .remove_label_values(labels);
        let _ = self
            .gpu_device_pool_owner_admitted
            .remove_label_values(labels);
        let _ = self
            .gpu_device_pool_owner_allocatable_bytes
            .remove_label_values(labels);
    }
}

fn prometheus_i64(value: u64) -> i64 {
    i64::try_from(value).unwrap_or(i64::MAX)
}

fn normalize_fragmentation(value: f64) -> f64 {
    if value.is_finite() {
        value.clamp(0.0, 1.0)
    } else {
        0.0
    }
}

#[cfg(test)]
#[path = "metrics_tests.rs"]
mod metrics_tests;
