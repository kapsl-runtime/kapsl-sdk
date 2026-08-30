use kapsl_engine_api::EngineMetrics;

/// Overflow-safe aggregation shared by scheduler and replica-pool metrics.
///
/// Additive counters and capacities are summed, GPU utilization is averaged,
/// and health uses the worst reported state. Fields without an established
/// aggregation contract retain their `EngineMetrics::default` values.
#[derive(Default)]
pub(crate) struct MetricsAccumulator {
    metrics: EngineMetrics,
    samples: usize,
}

impl MetricsAccumulator {
    pub(crate) fn add(&mut self, sample: &EngineMetrics) {
        self.samples = self.samples.saturating_add(1);
        self.metrics.memory_usage = self
            .metrics
            .memory_usage
            .saturating_add(sample.memory_usage);
        self.metrics.gpu_utilization += sample.gpu_utilization;
        self.metrics.throughput += sample.throughput;
        self.metrics.batch_size = self.metrics.batch_size.saturating_add(sample.batch_size);
        self.metrics.kv_cache_bytes_used = self
            .metrics
            .kv_cache_bytes_used
            .saturating_add(sample.kv_cache_bytes_used);
        self.metrics.kv_cache_bytes_capacity = self
            .metrics
            .kv_cache_bytes_capacity
            .saturating_add(sample.kv_cache_bytes_capacity);
        self.metrics.kv_cache_blocks_total = self
            .metrics
            .kv_cache_blocks_total
            .saturating_add(sample.kv_cache_blocks_total);
        self.metrics.kv_cache_blocks_free = self
            .metrics
            .kv_cache_blocks_free
            .saturating_add(sample.kv_cache_blocks_free);
        self.metrics.kv_cache_sequences = self
            .metrics
            .kv_cache_sequences
            .saturating_add(sample.kv_cache_sequences);
        self.metrics.kv_cache_evicted_blocks = self
            .metrics
            .kv_cache_evicted_blocks
            .saturating_add(sample.kv_cache_evicted_blocks);
        self.metrics.kv_cache_evicted_sequences = self
            .metrics
            .kv_cache_evicted_sequences
            .saturating_add(sample.kv_cache_evicted_sequences);
        self.metrics.kv_cache_packed_layers = self
            .metrics
            .kv_cache_packed_layers
            .saturating_add(sample.kv_cache_packed_layers);
        self.metrics.kv_cache_cpu_offloaded_blocks = self
            .metrics
            .kv_cache_cpu_offloaded_blocks
            .saturating_add(sample.kv_cache_cpu_offloaded_blocks);
        self.metrics.prompt_tokens_total = self
            .metrics
            .prompt_tokens_total
            .saturating_add(sample.prompt_tokens_total);
        self.metrics.generated_tokens_total = self
            .metrics
            .generated_tokens_total
            .saturating_add(sample.generated_tokens_total);
        self.metrics.decode_steps_total = self
            .metrics
            .decode_steps_total
            .saturating_add(sample.decode_steps_total);
        self.metrics.decode_tokens_evaluated_total = self
            .metrics
            .decode_tokens_evaluated_total
            .saturating_add(sample.decode_tokens_evaluated_total);
        self.metrics.kv_partial_reuse_hits_total = self
            .metrics
            .kv_partial_reuse_hits_total
            .saturating_add(sample.kv_partial_reuse_hits_total);
        self.metrics.kv_partial_reuse_tokens_saved_total = self
            .metrics
            .kv_partial_reuse_tokens_saved_total
            .saturating_add(sample.kv_partial_reuse_tokens_saved_total);
        self.metrics.engine_health = self.metrics.engine_health.max(sample.engine_health);
        self.metrics.onnx_session_pool_total = self
            .metrics
            .onnx_session_pool_total
            .saturating_add(sample.onnx_session_pool_total);
        self.metrics.onnx_session_pool_idle = self
            .metrics
            .onnx_session_pool_idle
            .saturating_add(sample.onnx_session_pool_idle);
        self.metrics.onnx_session_pool_waits_total = self
            .metrics
            .onnx_session_pool_waits_total
            .saturating_add(sample.onnx_session_pool_waits_total);
        self.metrics.onnx_session_pool_wait_seconds_total +=
            sample.onnx_session_pool_wait_seconds_total;
    }

    pub(crate) fn finish(mut self) -> EngineMetrics {
        if self.samples == 0 {
            return EngineMetrics::default();
        }
        let samples = self.samples as f64;
        self.metrics.gpu_utilization /= samples;
        self.metrics
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aggregation_averages_utilization_and_saturates_counters() {
        let mut accumulator = MetricsAccumulator::default();
        accumulator.add(&EngineMetrics {
            gpu_utilization: 0.5,
            memory_usage: usize::MAX,
            engine_health: 1,
            ..EngineMetrics::default()
        });
        accumulator.add(&EngineMetrics {
            gpu_utilization: 0.9,
            memory_usage: 1,
            engine_health: 2,
            ..EngineMetrics::default()
        });

        let metrics = accumulator.finish();
        assert_eq!(metrics.gpu_utilization, 0.7);
        assert_eq!(metrics.memory_usage, usize::MAX);
        assert_eq!(metrics.engine_health, 2);
    }
}
