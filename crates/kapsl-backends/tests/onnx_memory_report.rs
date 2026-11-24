//! ONNX models report measured memory, not their plan.
//!
//! * Real ORT on the identity fixtures (CPU): the report after load and after
//!   runs comes from the session, differs from the plan and moves with the
//!   arena.
//! * Every task wrapper forwards `planned_memory` / `actual_memory` to the
//!   inner engine (they used to forward only the legacy external report, which
//!   is empty without a CUDA pool, so the engine never saw a measurement).

use async_trait::async_trait;
use kapsl_backends::onnx::ExecutionProvider;
use kapsl_backends::{
    OnnxBackend, OnnxClassifyBackend, OnnxEmbedBackend, OnnxTranscribeBackend, PreprocessBackend,
    Preprocessor,
};
use kapsl_engine_api::{
    BinaryTensorPacket, Engine, EngineError, EngineMetrics, InferenceRequest, MemoryAllocation,
    MemoryAllocationClass, MemoryAllocationSource, MemoryDomain, MemoryReport, TensorDtype,
};
use std::path::{Path, PathBuf};

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name)
}

fn f32_packet(shape: Vec<i64>, vals: &[f32]) -> BinaryTensorPacket {
    BinaryTensorPacket {
        shape,
        dtype: TensorDtype::Float32,
        data: vals.iter().flat_map(|v| v.to_le_bytes()).collect(),
    }
}

fn bytes(report: &MemoryReport, class: MemoryAllocationClass) -> Option<usize> {
    report
        .allocations
        .iter()
        .find(|a| a.domain == MemoryDomain::Host && a.class == class)
        .map(|a| a.bytes)
}

#[tokio::test]
async fn cpu_actual_memory_is_measured_and_differs_from_the_plan() {
    let model = fixture("identity_logits.onnx");
    let mut backend = OnnxBackend::builder()
        .with_provider(ExecutionProvider::CPU)
        .build();
    let plan = backend.planned_memory(&model).expect("plan");
    // The plan is the file size (123 bytes) on the host and nothing else.
    assert_eq!(plan.allocations.len(), 1);
    backend.load(&model).await.expect("load");

    let after_load = backend.actual_memory();
    let session = bytes(&after_load, MemoryAllocationClass::ModelSession).expect("session row");
    // Without a clean resident delta the floor is the file size; with one the
    // delta is measured. Either way the row is derived from the session.
    assert!(session > 0);
    // Before any run the arena has not been sized: the workspace row is the
    // planned fallback (max(file / 4, 64 MiB)), not a false zero.
    let workspace0 =
        bytes(&after_load, MemoryAllocationClass::TransientWorkspace).expect("workspace row");
    assert_eq!(workspace0, 64 * 1024 * 1024);

    let request = InferenceRequest::new(f32_packet(vec![1, 3], &[1.0, 2.0, 3.0]));
    for _ in 0..3 {
        backend.infer(&request).expect("infer");
    }
    let after_run = backend.actual_memory();
    assert_eq!(
        bytes(&after_run, MemoryAllocationClass::ModelSession),
        Some(session),
        "the load-time footprint is fixed"
    );
    let workspace1 =
        bytes(&after_run, MemoryAllocationClass::TransientWorkspace).expect("workspace row");
    assert!(
        workspace1 < workspace0,
        "after runs the row is the measured arena growth ({workspace1}), not the fallback"
    );
    assert_ne!(after_run, plan);

    backend.unload();
    assert!(
        backend.actual_memory().allocations.is_empty(),
        "nothing after unload"
    );
}

#[derive(Default)]
struct FixedMemoryEngine;

fn marker(id: &str, bytes: usize) -> MemoryReport {
    MemoryReport {
        allocations: vec![MemoryAllocation {
            allocation_id: id.to_string(),
            domain: MemoryDomain::Host,
            class: MemoryAllocationClass::ModelSession,
            source: MemoryAllocationSource::BackendManaged,
            bytes,
        }],
    }
}

#[async_trait]
impl Engine for FixedMemoryEngine {
    fn planned_memory(&self, _model_path: &Path) -> Result<MemoryReport, EngineError> {
        Ok(marker("planned", 111))
    }
    async fn load(&mut self, _path: &Path) -> Result<(), EngineError> {
        Ok(())
    }
    fn actual_memory(&self) -> MemoryReport {
        marker("actual", 222)
    }
    fn infer(&self, _req: &InferenceRequest) -> Result<BinaryTensorPacket, EngineError> {
        Ok(f32_packet(vec![1], &[0.0]))
    }
    fn infer_stream(
        &self,
        req: &InferenceRequest,
    ) -> std::pin::Pin<
        Box<dyn futures::stream::Stream<Item = Result<BinaryTensorPacket, EngineError>> + Send>,
    > {
        let result = self.infer(req);
        Box::pin(futures::stream::once(async move { result }))
    }
    fn unload(&mut self) {}
    fn metrics(&self) -> EngineMetrics {
        EngineMetrics::default()
    }
    fn health_check(&self) -> Result<(), EngineError> {
        Ok(())
    }
}

struct NoopPreprocessor;
impl Preprocessor for NoopPreprocessor {
    fn name(&self) -> &'static str {
        "noop"
    }
    fn apply(&self, input: &BinaryTensorPacket) -> Result<BinaryTensorPacket, EngineError> {
        Ok(input.clone())
    }
}

fn assert_forwards(name: &str, engine: &dyn Engine) {
    let plan = engine.planned_memory(Path::new("model.onnx")).expect(name);
    assert_eq!(
        plan,
        marker("planned", 111),
        "{name} must forward planned_memory"
    );
    assert_eq!(
        engine.actual_memory(),
        marker("actual", 222),
        "{name} must forward actual_memory"
    );
}

#[test]
fn every_onnx_task_wrapper_forwards_planned_and_actual_memory() {
    assert_forwards(
        "classify",
        &OnnxClassifyBackend::new(Box::new(FixedMemoryEngine), true),
    );
    assert_forwards(
        "embed",
        &OnnxEmbedBackend::new(Box::new(FixedMemoryEngine), true),
    );
    assert_forwards(
        "transcribe",
        &OnnxTranscribeBackend::new(Box::new(FixedMemoryEngine), Default::default()),
    );
    assert_forwards(
        "preprocess",
        &PreprocessBackend::new(Box::new(FixedMemoryEngine), Box::new(NoopPreprocessor)),
    );
    let detect_config: kapsl_backends::onnx_detect::DetectConfig =
        serde_yaml::from_str("num_classes: 3").expect("detect config");
    assert_forwards(
        "detect",
        &kapsl_backends::OnnxDetectBackend::new(Box::new(FixedMemoryEngine), detect_config)
            .expect("detect backend"),
    );
}

/// CoreML on Apple silicon runs in unified memory: its plan is a host plan, so
/// plan and measurement land in the same domain (no `metal` provider rows).
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[test]
fn coreml_on_apple_silicon_is_planned_in_host_memory() {
    let model = fixture("identity_logits.onnx");
    let backend = OnnxBackend::builder()
        .with_provider(ExecutionProvider::CoreML)
        .build();
    let plan = backend.planned_memory(&model).expect("plan");
    assert!(!plan.allocations.is_empty());
    assert!(
        plan.allocations
            .iter()
            .all(|a| a.domain == MemoryDomain::Host),
        "{plan:?}"
    );
    assert!(plan
        .allocations
        .iter()
        .any(|a| a.class == MemoryAllocationClass::TransientWorkspace));
}
