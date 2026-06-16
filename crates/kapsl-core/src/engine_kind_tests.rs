use super::*;

#[test]
fn classifies_known_frameworks() {
    assert_eq!(EngineKind::from_framework("gguf"), EngineKind::GgufGenerate);
    assert_eq!(EngineKind::from_framework("llm"), EngineKind::OnnxGenerate);
    assert_eq!(EngineKind::from_framework("native"), EngineKind::Native);
    assert_eq!(
        EngineKind::from_framework("safetensors"),
        EngineKind::Native
    );
    assert_eq!(EngineKind::from_framework("onnx"), EngineKind::OnnxForward);
}

#[test]
fn is_case_and_whitespace_insensitive() {
    assert_eq!(EngineKind::from_framework("  GGUF "), EngineKind::GgufGenerate);
    assert_eq!(EngineKind::from_framework("LLM"), EngineKind::OnnxGenerate);
    assert_eq!(EngineKind::from_framework("SafeTensors"), EngineKind::Native);
}

#[test]
fn unknown_frameworks_fall_through_to_onnx_forward() {
    // Preserves the legacy `else` arm: pytorch/tensorflow/unknown -> stateless ONNX.
    assert_eq!(EngineKind::from_framework("pytorch"), EngineKind::OnnxForward);
    assert_eq!(EngineKind::from_framework("tensorflow"), EngineKind::OnnxForward);
    assert_eq!(EngineKind::from_framework("totally-made-up"), EngineKind::OnnxForward);
    assert_eq!(EngineKind::from_framework(""), EngineKind::OnnxForward);
}

#[test]
fn predicate_helpers() {
    assert!(EngineKind::GgufGenerate.is_generative());
    assert!(EngineKind::OnnxGenerate.is_generative());
    assert!(!EngineKind::Native.is_generative());
    assert!(!EngineKind::OnnxForward.is_generative());

    assert!(EngineKind::GgufGenerate.is_gguf());
    assert!(!EngineKind::OnnxGenerate.is_gguf());

    // Only the legacy "llm" path gets LLM scheduler tuning / pipeline handling.
    assert!(EngineKind::OnnxGenerate.is_onnx_generate());
    assert!(!EngineKind::GgufGenerate.is_onnx_generate());

    assert!(EngineKind::OnnxGenerate.uses_onnx_session());
    assert!(EngineKind::OnnxForward.uses_onnx_session());
    assert!(!EngineKind::GgufGenerate.uses_onnx_session());
    assert!(!EngineKind::Native.uses_onnx_session());
}

#[test]
fn resolve_reads_manifest_framework() {
    let manifest = Manifest {
        project_name: "m".into(),
        framework: "gguf".into(),
        version: "1.0.0".into(),
        created_at: "0".into(),
        model_file: "model.gguf".into(),
        metadata: None,
        hardware_requirements: Default::default(),
        cron_jobs: Vec::new(),
    };
    assert_eq!(EngineKind::resolve(&manifest), EngineKind::GgufGenerate);
}
