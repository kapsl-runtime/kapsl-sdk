//! Manual measurement: report versus process memory for real ONNX models.
//!
//! Ignored by default (needs model files). Loads the models in order in one
//! process and prints one `ONNXMEM` line per step:
//!
//! ```text
//! KAPSL_ONNX_MEMORY_MODELS="/path/a.onnx|1,3,416,416;/path/b.onnx|1,3,224,224" \
//! KAPSL_ONNX_MEMORY_PROVIDER=cpu|coreml \
//!   cargo test -p kapsl-backends --test onnx_memory_real_models -- --ignored --nocapture
//! ```
//!
//! `rss` is the process resident size, `footprint` is macOS `footprint`'s
//! physical footprint (includes memory RSS does not see), both in bytes.

use kapsl_backends::onnx::ExecutionProvider;
use kapsl_backends::OnnxBackend;
use kapsl_engine_api::{
    BinaryTensorPacket, Engine, InferenceRequest, MemoryDomain, MemoryReport, TensorDtype,
};
use std::path::Path;

fn rss() -> usize {
    let out = std::process::Command::new("ps")
        .args(["-o", "rss=", "-p", &std::process::id().to_string()])
        .output()
        .expect("ps");
    String::from_utf8_lossy(&out.stdout)
        .trim()
        .parse::<usize>()
        .unwrap_or(0)
        * 1024
}

/// Physical footprint from macOS `footprint`, if the tool is available.
fn footprint() -> Option<usize> {
    let out = std::process::Command::new("footprint")
        .args(["-p", &std::process::id().to_string(), "--noCategories"])
        .output()
        .ok()?;
    let text = String::from_utf8_lossy(&out.stdout);
    let line = text.lines().find(|l| l.contains("Footprint:"))?;
    let value = line.split("Footprint:").nth(1)?.trim();
    let mut parts = value.split_whitespace();
    let number: f64 = parts.next()?.parse().ok()?;
    let unit = parts.next()?;
    let scale = match unit {
        "KB" => 1024.0,
        "MB" => 1024.0 * 1024.0,
        "GB" => 1024.0 * 1024.0 * 1024.0,
        _ => 1.0,
    };
    Some((number * scale) as usize)
}

fn rows(report: &MemoryReport) -> String {
    report
        .allocations
        .iter()
        .map(|a| {
            let domain = match &a.domain {
                MemoryDomain::Host => "host".to_string(),
                other => format!("{other:?}"),
            };
            format!("{domain}/{:?}={}", a.class, a.bytes)
        })
        .collect::<Vec<_>>()
        .join(",")
}

fn total(report: &MemoryReport) -> usize {
    report.allocations.iter().map(|a| a.bytes).sum()
}

#[tokio::test]
#[ignore = "needs model files; see module docs"]
async fn report_versus_process_memory() {
    let models = std::env::var("KAPSL_ONNX_MEMORY_MODELS").expect("KAPSL_ONNX_MEMORY_MODELS");
    let provider = match std::env::var("KAPSL_ONNX_MEMORY_PROVIDER").as_deref() {
        Ok("coreml") => ExecutionProvider::CoreML,
        _ => ExecutionProvider::CPU,
    };
    let mut keep = Vec::new();
    for spec in models.split(';') {
        let (path, shape) = spec.split_once('|').expect("path|shape");
        let shape: Vec<i64> = shape.split(',').map(|d| d.parse().unwrap()).collect();
        let name = Path::new(path)
            .file_name()
            .unwrap()
            .to_string_lossy()
            .to_string();
        let file = std::fs::metadata(path).unwrap().len() as usize;
        let (rss0, fp0) = (rss(), footprint());
        let mut backend = OnnxBackend::builder().with_provider(provider).build();
        let plan = backend.planned_memory(Path::new(path)).unwrap();
        backend.load(Path::new(path)).await.unwrap();
        let (rss1, fp1) = (rss(), footprint());
        let after_load = backend.actual_memory();
        let len: usize = shape.iter().product::<i64>() as usize;
        let request = InferenceRequest::new(BinaryTensorPacket {
            shape: shape.clone(),
            dtype: TensorDtype::Float32,
            data: vec![0.5f32; len]
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect(),
        });
        backend.infer(&request).unwrap();
        let (rss2, fp2) = (rss(), footprint());
        let after_warmup = backend.actual_memory();
        for _ in 0..20 {
            backend.infer(&request).unwrap();
        }
        let (rss3, fp3) = (rss(), footprint());
        let after_runs = backend.actual_memory();
        let d = |a: usize, b: usize| b as i64 - a as i64;
        let dfp = |a: Option<usize>, b: Option<usize>| match (a, b) {
            (Some(a), Some(b)) => d(a, b).to_string(),
            _ => "n/a".into(),
        };
        println!(
            "ONNXMEM model={name} provider={provider:?} file={file} plan_total={} plan=[{}]",
            total(&plan),
            rows(&plan)
        );
        println!("ONNXMEM model={name} step=after_load reported_total={} rss_delta={} footprint_delta={} rows=[{}]", total(&after_load), d(rss0, rss1), dfp(fp0, fp1), rows(&after_load));
        println!("ONNXMEM model={name} step=after_warmup reported_total={} rss_delta={} footprint_delta={} rows=[{}]", total(&after_warmup), d(rss0, rss2), dfp(fp0, fp2), rows(&after_warmup));
        println!("ONNXMEM model={name} step=after_20_runs reported_total={} rss_delta={} footprint_delta={} rows=[{}]", total(&after_runs), d(rss0, rss3), dfp(fp0, fp3), rows(&after_runs));
        keep.push(backend);
    }
    drop(keep);
}
