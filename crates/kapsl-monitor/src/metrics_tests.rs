use super::KapslMetrics;
use kapsl_engine_api::EngineMetrics;
use prometheus::{Encoder, Registry, TextEncoder};
use std::sync::Arc;

#[test]
fn set_kv_cache_metrics_exports_cpu_offload_blocks() {
    let registry = Arc::new(Registry::new());
    let metrics = KapslMetrics::new(&registry);

    metrics.set_kv_cache_metrics(
        "model-7",
        &EngineMetrics {
            kv_cache_bytes_used: 1024,
            kv_cache_bytes_capacity: 4096,
            kv_cache_blocks_total: 64,
            kv_cache_blocks_free: 48,
            kv_cache_sequences: 3,
            kv_cache_evicted_blocks: 2,
            kv_cache_evicted_sequences: 1,
            kv_cache_packed_layers: 5,
            kv_cache_cpu_offloaded_blocks: 9,
            ..EngineMetrics::default()
        },
    );

    let mut buf = Vec::new();
    TextEncoder::new()
        .encode(&registry.gather(), &mut buf)
        .expect("encode metrics");
    let text = String::from_utf8(buf).expect("utf8 metrics");

    assert!(text.contains(r#"kapsl_kv_cache_bytes_used{model="model-7"} 1024"#));
    assert!(text.contains(r#"kapsl_kv_cache_blocks_free{model="model-7"} 48"#));
    assert!(text.contains(r#"kapsl_kv_cache_cpu_offloaded_blocks{model="model-7"} 9"#));
}
