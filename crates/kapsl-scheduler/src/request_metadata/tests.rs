use super::{determine_priority, RequestMetadata};
use crate::priority::Priority;

#[test]
fn priority_override_wins() {
    let meta = RequestMetadata {
        priority: 0,
        sla_deadline: Some(10_000),
        batch_size: 32,
        input_size_bytes: Some(10_000_000),
        estimated_flops: Some(10_000_000_000),
    };

    assert_eq!(determine_priority(&meta), Priority::LatencyCritical);
}

#[test]
fn sla_deadline_triggers_latency() {
    let meta = RequestMetadata {
        priority: 1,
        sla_deadline: Some(200),
        batch_size: 16,
        input_size_bytes: Some(10_000_000),
        estimated_flops: Some(10_000_000_000),
    };

    assert_eq!(determine_priority(&meta), Priority::LatencyCritical);
}

#[test]
fn light_workload_by_input_size() {
    let meta = RequestMetadata {
        priority: 1,
        sla_deadline: None,
        batch_size: 16,
        input_size_bytes: Some(500_000),
        estimated_flops: Some(10_000_000_000),
    };

    assert_eq!(determine_priority(&meta), Priority::LatencyCritical);
}

#[test]
fn light_workload_by_flops() {
    let meta = RequestMetadata {
        priority: 1,
        sla_deadline: None,
        batch_size: 16,
        input_size_bytes: Some(10_000_000),
        estimated_flops: Some(1_000_000_000),
    };

    assert_eq!(determine_priority(&meta), Priority::LatencyCritical);
}

#[test]
fn light_workload_by_combined_size() {
    let meta = RequestMetadata {
        priority: 1,
        sla_deadline: None,
        batch_size: 4,
        input_size_bytes: Some(500_000),
        estimated_flops: Some(10_000_000_000),
    };

    assert_eq!(determine_priority(&meta), Priority::LatencyCritical);
}

#[test]
fn small_batch_triggers_latency() {
    let meta = RequestMetadata {
        priority: 1,
        sla_deadline: None,
        batch_size: 4,
        input_size_bytes: Some(10_000_000),
        estimated_flops: Some(10_000_000_000),
    };

    assert_eq!(determine_priority(&meta), Priority::LatencyCritical);
}

#[test]
fn heavy_workload_defaults_to_throughput() {
    let meta = RequestMetadata {
        priority: 1,
        sla_deadline: None,
        batch_size: 8,
        input_size_bytes: Some(10_000_000),
        estimated_flops: Some(10_000_000_000),
    };

    assert_eq!(determine_priority(&meta), Priority::Throughput);
}

#[test]
fn combined_work_estimate_saturates_instead_of_overflowing() {
    let meta = RequestMetadata {
        priority: 1,
        sla_deadline: None,
        batch_size: usize::MAX,
        input_size_bytes: Some(usize::MAX),
        estimated_flops: Some(u64::MAX),
    };

    assert_eq!(determine_priority(&meta), Priority::Throughput);
}
