use super::KapslMetrics;
use kapsl_engine_api::EngineMetrics;
use prometheus::{Encoder, Registry, TextEncoder};
use std::sync::Arc;

#[test]
fn set_engine_metrics_exports_cache_and_session_fields() {
    let registry = Arc::new(Registry::new());
    let metrics = KapslMetrics::new(&registry);

    metrics.set_engine_metrics(
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
            onnx_session_pool_total: 2,
            onnx_session_pool_idle: 1,
            onnx_session_pool_waits_total: 4,
            onnx_session_pool_wait_seconds_total: 1.25,
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
    assert!(text.contains(r#"kapsl_onnx_session_pool_total{model="model-7"} 2"#));
    assert!(text.contains(r#"kapsl_onnx_session_pool_idle{model="model-7"} 1"#));
    assert!(text.contains(r#"kapsl_onnx_session_pool_waits_total{model="model-7"} 4"#));
    assert!(text.contains(r#"kapsl_onnx_session_pool_wait_seconds_total{model="model-7"} 1.25"#));
}
