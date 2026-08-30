//! Tests for manifest framework and task classification.

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
    assert_eq!(
        EngineKind::from_framework("  GGUF "),
        EngineKind::GgufGenerate
    );
    assert_eq!(EngineKind::from_framework("LLM"), EngineKind::OnnxGenerate);
    assert_eq!(
        EngineKind::from_framework("SafeTensors"),
        EngineKind::Native
    );
}

#[test]
fn unknown_frameworks_fail_closed() {
    assert_eq!(
        EngineKind::from_framework("pytorch"),
        EngineKind::Unsupported
    );
    assert_eq!(
        EngineKind::from_framework("tensorflow"),
        EngineKind::Unsupported
    );
    assert_eq!(
        EngineKind::from_framework("totally-made-up"),
        EngineKind::Unsupported
    );
    assert_eq!(EngineKind::from_framework(""), EngineKind::Unsupported);
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
    assert!(!EngineKind::Unsupported.uses_onnx_session());
}

/// Build a manifest from the axes under test (model_file controls the
/// extension cross-check in `validate`).
fn mk(
    framework: &str,
    format: Option<&str>,
    model_type: Option<&str>,
    task: Option<&str>,
    model_file: &str,
) -> Manifest {
    Manifest {
        project_name: "m".into(),
        framework: framework.into(),
        version: "1.0.0".into(),
        created_at: "0".into(),
        model_file: model_file.into(),
        format: format.map(str::to_string),
        model_type: model_type.map(str::to_string),
        task: task.map(str::to_string),
        metadata: None,
        hardware_requirements: Default::default(),
        cron_jobs: Vec::new(),
    }
}

#[test]
fn resolve_reads_manifest_framework() {
    let manifest = mk("gguf", None, None, None, "model.gguf");
    assert_eq!(EngineKind::resolve(&manifest), EngineKind::GgufGenerate);
}

#[test]
fn legacy_only_manifests_resolve_unchanged() {
    assert_eq!(
        EngineKind::resolve(&mk("llm", None, None, None, "m.onnx")),
        EngineKind::OnnxGenerate
    );
    assert_eq!(
        EngineKind::resolve(&mk("onnx", None, None, None, "m.onnx")),
        EngineKind::OnnxForward
    );
    assert_eq!(
        EngineKind::resolve(&mk("safetensors", None, None, None, "m.safetensors")),
        EngineKind::Native
    );
}

#[test]
fn new_axes_take_precedence_over_framework() {
    // framework says onnx (forward), but task=generate -> ONNX generative.
    let m = mk(
        "onnx",
        Some("onnx"),
        Some("causal-lm"),
        Some("generate"),
        "m.onnx",
    );
    assert_eq!(EngineKind::resolve(&m), EngineKind::OnnxGenerate);

    // onnx embedding -> dedicated (not-yet-implemented) embed engine.
    let m = mk(
        "onnx",
        Some("onnx"),
        Some("embedding"),
        Some("embed"),
        "m.onnx",
    );
    assert_eq!(EngineKind::resolve(&m), EngineKind::OnnxEmbed);
}

#[test]
fn resolve_maps_onnx_tasks_to_distinct_engines() {
    let embed = mk(
        "onnx",
        Some("onnx"),
        Some("embedding"),
        Some("embed"),
        "m.onnx",
    );
    assert_eq!(EngineKind::resolve(&embed), EngineKind::OnnxEmbed);

    let classify = mk(
        "onnx",
        Some("onnx"),
        Some("seq-classifier"),
        Some("classify"),
        "m.onnx",
    );
    assert_eq!(EngineKind::resolve(&classify), EngineKind::OnnxClassify);
}

#[test]
fn all_current_cells_are_implemented() {
    for kind in [
        EngineKind::GgufGenerate,
        EngineKind::OnnxGenerate,
        EngineKind::Native,
        EngineKind::OnnxForward,
        EngineKind::OnnxEmbed,
        EngineKind::OnnxClassify,
    ] {
        assert!(kind.is_implemented(), "{kind:?} should have a backend");
    }
    assert!(!EngineKind::Unsupported.is_implemented());
    assert!(EngineKind::OnnxClassify.uses_onnx_session());
}

#[test]
fn effective_axes_infer_from_framework() {
    let m = mk("llm", None, None, None, "m.onnx");
    assert_eq!(super::effective_format(&m), "onnx");
    assert_eq!(super::effective_model_type(&m), "causal-lm");
    assert_eq!(super::effective_task(&m), "generate");

    let m = mk("gguf", None, None, None, "m.gguf");
    assert_eq!(super::effective_format(&m), "gguf");
    assert_eq!(super::effective_task(&m), "generate");
}

#[test]
fn validate_accepts_legacy_manifests() {
    assert!(EngineKind::validate(&mk("gguf", None, None, None, "model.gguf")).is_ok());
    assert!(EngineKind::validate(&mk("llm", None, None, None, "model.onnx")).is_ok());
    assert!(EngineKind::validate(&mk("onnx", None, None, None, "model.onnx")).is_ok());
    assert!(
        EngineKind::validate(&mk("safetensors", None, None, None, "model.safetensors")).is_ok()
    );
}

#[test]
fn validate_rejects_unsupported_legacy_frameworks_before_onnx_dispatch() {
    for (framework, model_file) in [
        ("pytorch", "model.pt"),
        ("tensorflow", "saved_model.pb"),
        ("totally-made-up", "model.onnx"),
        ("", "model.onnx"),
    ] {
        let err = EngineKind::validate(&mk(framework, None, None, None, model_file))
            .expect_err("unsupported framework must fail closed");
        assert!(err.contains("unsupported framework"), "unexpected: {err}");
    }
}

#[test]
fn validate_rejects_non_onnx_weight_extensions() {
    for model_file in ["model.pt", "model.pth", "saved_model.pb", "model.bin"] {
        let err = EngineKind::validate(&mk("onnx", None, None, None, model_file))
            .expect_err("unsupported extension must not reach ONNX Runtime");
        assert!(
            err.contains("will not pass") || err.contains("unsupported extension"),
            "unexpected: {err}"
        );
    }
}

#[test]
fn explicit_converted_format_can_override_source_framework_label() {
    assert!(EngineKind::validate(&mk(
        "pytorch",
        Some("onnx"),
        Some("opaque"),
        Some("forward"),
        "converted.onnx"
    ))
    .is_ok());
}

#[test]
fn validate_rejects_llm_tag_on_gguf_file() {
    // The gemma footgun: framework=llm but the file is a .gguf.
    let err = EngineKind::validate(&mk("llm", None, None, None, "gemma-2b.gguf")).unwrap_err();
    assert!(err.contains("gguf"), "unexpected: {err}");
}

#[test]
fn validate_rejects_non_causal_gguf() {
    let err = EngineKind::validate(&mk(
        "gguf",
        Some("gguf"),
        Some("embedding"),
        Some("embed"),
        "m.gguf",
    ))
    .unwrap_err();
    assert!(err.contains("causal-lm"), "unexpected: {err}");
}

#[test]
fn validate_rejects_incoherent_task_for_model_type() {
    // causal-lm cannot classify.
    assert!(EngineKind::validate(&mk(
        "onnx",
        Some("onnx"),
        Some("causal-lm"),
        Some("classify"),
        "m.onnx"
    ))
    .is_err());
    // embedding model cannot generate.
    assert!(EngineKind::validate(&mk(
        "onnx",
        Some("onnx"),
        Some("embedding"),
        Some("generate"),
        "m.onnx"
    ))
    .is_err());
}

#[test]
fn validate_rejects_unknown_vocabulary() {
    assert!(EngineKind::validate(&mk("onnx", Some("tensorrt"), None, None, "m.onnx")).is_err());
    assert!(EngineKind::validate(&mk("onnx", None, Some("diffusion"), None, "m.onnx")).is_err());
    assert!(EngineKind::validate(&mk("onnx", None, None, Some("teleport"), "m.onnx")).is_err());
}

#[test]
fn validate_accepts_onnx_embedding() {
    assert!(EngineKind::validate(&mk(
        "onnx",
        Some("onnx"),
        Some("embedding"),
        Some("embed"),
        "m.onnx"
    ))
    .is_ok());
}
