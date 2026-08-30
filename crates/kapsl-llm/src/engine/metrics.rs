//! Runtime metrics recorded by LLM engines.

#[derive(Default, Clone)]
pub struct LLMMetrics {
    pub total_inference_time: f64,
    /// True only while authoritative ONNX KV values are retained in CUDA
    /// OrtValues. This is internal reporting state rather than a public metric.
    pub kv_cache_device_resident: bool,
    pub kv_cache_bytes_used: usize,
    pub kv_cache_bytes_capacity: usize,
    /// Physical host KV bytes retained by the fallback cache, including its
    /// free-list/backing storage.
    pub kv_cache_host_bytes_retained: usize,
    /// Physical CUDA KV bytes currently retained in provider values.
    pub kv_cache_device_bytes_retained: usize,
    /// Conservative authority reservation required before one new sequence
    /// can allocate or grow KV state.
    pub kv_cache_request_reservation_bytes: usize,
    /// Additional host bytes needed if a device-resident sequence migrates to
    /// the fallback cache. Paged backing is preleased at model load, so only a
    /// dense fallback needs a request-lifetime reservation here.
    pub kv_cache_host_fallback_reservation_bytes: usize,
    pub kv_cache_blocks_total: usize,
    pub kv_cache_blocks_free: usize,
    pub kv_cache_sequences: usize,
    pub kv_cache_evicted_blocks: u64,
    pub kv_cache_evicted_sequences: u64,
    pub kv_cache_packed_layers: usize,
    /// Blocks currently sitting in the CPU offload store (paged mode only).
    pub kv_cache_cpu_offloaded_blocks: u64,
    /// allocate_sequence calls that reused a cached prefix (paged mode only).
    pub kv_cache_prefix_reuse_hits: u64,
    /// Cumulative prompt tokens skipped via prefix reuse (paged mode only).
    pub kv_cache_prefix_reuse_tokens_saved: u64,
}
