use super::{GpuDevicePoolMetrics, GpuDevicePoolOwnerMetrics, KapslMetrics};
use kapsl_engine_api::EngineMetrics;
use prometheus::{Encoder, Registry, TextEncoder};
use std::sync::Arc;

fn gather_text(registry: &Registry) -> String {
    let mut buf = Vec::new();
    TextEncoder::new()
        .encode(&registry.gather(), &mut buf)
        .expect("encode metrics");
    String::from_utf8(buf).expect("utf8 metrics")
}

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

    let text = gather_text(&registry);

    assert!(text.contains(r#"kapsl_kv_cache_bytes_used{model="model-7"} 1024"#));
    assert!(text.contains(r#"kapsl_kv_cache_blocks_free{model="model-7"} 48"#));
    assert!(text.contains(r#"kapsl_kv_cache_cpu_offloaded_blocks{model="model-7"} 9"#));
    assert!(text.contains(r#"kapsl_onnx_session_pool_total{model="model-7"} 2"#));
    assert!(text.contains(r#"kapsl_onnx_session_pool_idle{model="model-7"} 1"#));
    assert!(text.contains(r#"kapsl_onnx_session_pool_waits_total{model="model-7"} 4"#));
    assert!(text.contains(r#"kapsl_onnx_session_pool_wait_seconds_total{model="model-7"} 1.25"#));
}

#[test]
fn set_engine_metrics_saturates_unsigned_values_for_prometheus() {
    let registry = Arc::new(Registry::new());
    let metrics = KapslMetrics::new(&registry);

    metrics.set_engine_metrics(
        "oversized",
        &EngineMetrics {
            kv_cache_bytes_used: usize::MAX,
            prompt_tokens_total: u64::MAX,
            onnx_session_pool_waits_total: u64::MAX,
            ..EngineMetrics::default()
        },
    );

    assert_eq!(
        metrics
            .kv_cache_bytes_used
            .with_label_values(&["oversized"])
            .get(),
        i64::MAX
    );
    assert_eq!(
        metrics
            .prompt_tokens_total
            .with_label_values(&["oversized"])
            .get(),
        i64::MAX
    );
    assert_eq!(
        metrics
            .onnx_session_pool_waits_total
            .with_label_values(&["oversized"])
            .get(),
        i64::MAX
    );
}

#[test]
fn set_gpu_device_pool_metrics_exports_pool_and_owner_fields() {
    let registry = Arc::new(Registry::new());
    let metrics = KapslMetrics::new(&registry);

    metrics.set_gpu_device_pool_metrics(
        "0",
        &GpuDevicePoolMetrics {
            allocated_bytes: 3072,
            live_allocations: 3,
            free_bytes: 1024,
            free_ranges: 2,
            largest_free_range_bytes: 768,
            fragmentation_ratio: 0.25,
            owners: vec![GpuDevicePoolOwnerMetrics {
                owner: "gguf_kv:42".to_owned(),
                usage_bytes: 2048,
                guaranteed_bytes: 1024,
                max_bytes: 3072,
                admitted: true,
                allocatable_bytes: 1024,
            }],
        },
    );

    let text = gather_text(&registry);
    assert!(text.contains(r#"kapsl_gpu_device_pool_allocated_bytes{device="0"} 3072"#));
    assert!(text.contains(r#"kapsl_gpu_device_pool_live_allocations{device="0"} 3"#));
    assert!(text.contains(r#"kapsl_gpu_device_pool_free_bytes{device="0"} 1024"#));
    assert!(text.contains(r#"kapsl_gpu_device_pool_free_ranges{device="0"} 2"#));
    assert!(text.contains(r#"kapsl_gpu_device_pool_largest_free_range_bytes{device="0"} 768"#));
    assert!(text.contains(r#"kapsl_gpu_device_pool_fragmentation_ratio{device="0"} 0.25"#));
    assert!(text.contains(
        r#"kapsl_gpu_device_pool_owner_usage_bytes{device="0",owner="gguf_kv:42"} 2048"#
    ));
    assert!(text.contains(
        r#"kapsl_gpu_device_pool_owner_quota_guaranteed_bytes{device="0",owner="gguf_kv:42"} 1024"#
    ));
    assert!(text.contains(
        r#"kapsl_gpu_device_pool_owner_quota_max_bytes{device="0",owner="gguf_kv:42"} 3072"#
    ));
    assert!(
        text.contains(r#"kapsl_gpu_device_pool_owner_admitted{device="0",owner="gguf_kv:42"} 1"#)
    );
    assert!(text.contains(
        r#"kapsl_gpu_device_pool_owner_allocatable_bytes{device="0",owner="gguf_kv:42"} 1024"#
    ));
}

#[test]
fn replacing_snapshot_removes_stale_owner_series_across_clones() {
    let registry = Arc::new(Registry::new());
    let metrics = KapslMetrics::new(&registry);
    metrics.set_gpu_device_pool_metrics(
        "3",
        &GpuDevicePoolMetrics {
            owners: vec![
                GpuDevicePoolOwnerMetrics {
                    owner: "onnx".to_owned(),
                    usage_bytes: 10,
                    ..GpuDevicePoolOwnerMetrics::default()
                },
                GpuDevicePoolOwnerMetrics {
                    owner: "native_kv:7".to_owned(),
                    usage_bytes: 20,
                    ..GpuDevicePoolOwnerMetrics::default()
                },
            ],
            ..GpuDevicePoolMetrics::default()
        },
    );

    metrics.clone().set_gpu_device_pool_metrics(
        "3",
        &GpuDevicePoolMetrics {
            owners: vec![GpuDevicePoolOwnerMetrics {
                owner: "onnx".to_owned(),
                usage_bytes: 30,
                ..GpuDevicePoolOwnerMetrics::default()
            }],
            ..GpuDevicePoolMetrics::default()
        },
    );

    let text = gather_text(&registry);
    assert!(
        text.contains(r#"kapsl_gpu_device_pool_owner_usage_bytes{device="3",owner="onnx"} 30"#)
    );
    assert!(!text.contains(r#"owner="native_kv:7""#));
}

#[test]
fn remove_gpu_device_pool_metrics_clears_device_and_owner_series() {
    let registry = Arc::new(Registry::new());
    let metrics = KapslMetrics::new(&registry);
    metrics.set_gpu_device_pool_metrics(
        "2",
        &GpuDevicePoolMetrics {
            allocated_bytes: 1,
            owners: vec![GpuDevicePoolOwnerMetrics {
                owner: "onnx".to_owned(),
                usage_bytes: 1,
                ..GpuDevicePoolOwnerMetrics::default()
            }],
            ..GpuDevicePoolMetrics::default()
        },
    );

    metrics.remove_gpu_device_pool_metrics("2");
    metrics.remove_gpu_device_pool_metrics("2");

    let text = gather_text(&registry);
    assert!(!text.contains(r#"kapsl_gpu_device_pool_allocated_bytes{device="2"}"#));
    assert!(!text.contains(r#"device="2",owner="onnx""#));
}

#[test]
fn pool_metric_values_are_safe_for_prometheus_gauges() {
    let registry = Arc::new(Registry::new());
    let metrics = KapslMetrics::new(&registry);
    metrics.set_gpu_device_pool_metrics(
        "9",
        &GpuDevicePoolMetrics {
            allocated_bytes: u64::MAX,
            fragmentation_ratio: f64::INFINITY,
            ..GpuDevicePoolMetrics::default()
        },
    );

    assert_eq!(
        metrics
            .gpu_device_pool_allocated_bytes
            .with_label_values(&["9"])
            .get(),
        i64::MAX
    );
    let text = gather_text(&registry);
    assert!(text.contains(r#"kapsl_gpu_device_pool_fragmentation_ratio{device="9"} 0"#));
}

#[test]
fn managed_vllm_metrics_export_memory_lifecycle_and_bridge_stages() {
    let registry = Arc::new(Registry::new());
    let metrics = KapslMetrics::new(&registry);
    let memory = &["qwen", "2", "0"];
    metrics
        .managed_vllm
        .kv_granted_bytes
        .with_label_values(memory)
        .set(4096);
    metrics
        .managed_vllm
        .kv_blocks_active
        .with_label_values(memory)
        .set(3);
    metrics
        .managed_vllm
        .provisional_reservation_state
        .with_label_values(&["qwen", "2", "active"])
        .set(1);
    metrics
        .managed_vllm
        .bridge_stage_seconds
        .with_label_values(&["qwen", "2", "wire", "scheduler_queue"])
        .observe(0.002);
    metrics
        .managed_vllm
        .bridge_relayed_bytes_total
        .with_label_values(&["qwen", "2", "wire"])
        .inc_by(512);
    metrics
        .managed_vllm
        .bridge_connection_attempts_total
        .with_label_values(&["qwen", "2", "async"])
        .inc();

    let text = gather_text(&registry);
    assert!(text.contains(
        r#"kapsl_managed_vllm_kv_granted_bytes{device="0",model="qwen",replica="2"} 4096"#
    ));
    assert!(
        text.contains(
            r#"kapsl_managed_vllm_kv_blocks_active{device="0",model="qwen",replica="2"} 3"#
        )
    );
    assert!(text.contains(
        r#"kapsl_managed_vllm_provisional_reservation_state{model="qwen",replica="2",state="active"} 1"#
    ));
    assert!(text.contains(
        r#"kapsl_managed_vllm_bridge_stage_seconds_count{mode="wire",model="qwen",replica="2",stage="scheduler_queue"} 1"#
    ));
    assert!(text.contains(
        r#"kapsl_managed_vllm_bridge_relayed_bytes_total{mode="wire",model="qwen",replica="2"} 512"#
    ));
    assert!(text.contains(
        r#"kapsl_managed_vllm_bridge_connection_attempts_total{client="async",model="qwen",replica="2"} 1"#
    ));
}
