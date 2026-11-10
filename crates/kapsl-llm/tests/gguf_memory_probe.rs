//! Opt-in probe: loads a real GGUF through `GgufBackend` and prints the planned
//! and actual memory rows, so the scratch estimate can be calibrated against
//! llama.cpp's real compute buffer on whatever device this host has.
//!
//!   KAPSL_PROBE_GGUF=/path/model.gguf cargo test -p kapsl-llm --features gguf \
//!     --test gguf_memory_probe -- --ignored --nocapture
//!
//! Context sizing follows the production env knobs (KAPSL_GGUF_MAX_CONCURRENT,
//! KAPSL_GGUF_CTX_PER_SEQ, KAPSL_GGUF_PREFILL_CHUNK_SIZE).
#![cfg(feature = "gguf")]

use kapsl_engine_api::{Engine, MemoryAllocationClass, MemoryReport};
use kapsl_llm::GgufBackend;
use std::path::PathBuf;

fn class_bytes(report: &MemoryReport, class: MemoryAllocationClass) -> usize {
    report
        .allocations
        .iter()
        .filter(|allocation| allocation.class == class)
        .map(|allocation| allocation.bytes)
        .sum()
}

#[tokio::test]
#[ignore = "needs KAPSL_PROBE_GGUF and a real device"]
async fn planned_scratch_covers_actual_compute_buffer() {
    let path = PathBuf::from(
        std::env::var("KAPSL_PROBE_GGUF").expect("set KAPSL_PROBE_GGUF to a .gguf path"),
    );
    let mut backend = GgufBackend::new();
    let planned = backend.planned_memory(&path).expect("planned memory");
    backend.load(&path).await.expect("load");
    let actual = backend.actual_memory();

    for (label, report) in [("planned", &planned), ("actual", &actual)] {
        for allocation in &report.allocations {
            println!(
                "PROBE {label} class={:?} domain={:?} bytes={} ({:.2} MiB)",
                allocation.class,
                allocation.domain,
                allocation.bytes,
                allocation.bytes as f64 / (1024.0 * 1024.0)
            );
        }
    }

    let planned_scratch = class_bytes(&planned, MemoryAllocationClass::TransientWorkspace);
    let actual_scratch = class_bytes(&actual, MemoryAllocationClass::TransientWorkspace);
    assert!(actual_scratch > 0, "backend must report its compute buffer");
    assert!(
        planned_scratch >= actual_scratch,
        "planned scratch {planned_scratch} must cover actual compute buffer {actual_scratch}"
    );
    // The engine reconciles a planned reservation to the post-load report by
    // allocation id, so scratch and native KV must keep the same id across
    // planning and the report (otherwise the plan and the actual are charged
    // side by side).
    for class in [
        MemoryAllocationClass::TransientWorkspace,
        MemoryAllocationClass::KvCache,
    ] {
        let ids = |report: &MemoryReport| -> Vec<String> {
            let mut ids: Vec<String> = report
                .allocations
                .iter()
                .filter(|allocation| allocation.class == class)
                .map(|allocation| allocation.allocation_id.clone())
                .collect();
            ids.sort();
            ids.dedup();
            ids
        };
        assert_eq!(
            ids(&planned),
            ids(&actual),
            "planned and reported allocation ids differ for {class:?}"
        );
    }
    backend.unload();
}
