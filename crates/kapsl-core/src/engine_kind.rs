//! Centralized classification of a manifest's `framework` string into the
//! runtime engine family it selects.
//!
//! Historically the `framework` field was compared against string literals
//! (`"gguf"`, `"llm"`, `"native"`, `"safetensors"`, …) in many places across the
//! backend factory and the CLI. That scattered, stringly-typed dispatch made it
//! easy to mis-handle a value and hard to reason about which models are
//! generative. [`EngineKind`] is the single place that interprets `framework`;
//! every dispatch site classifies through it instead of comparing strings.
//!
//! This is intentionally **behavior-preserving** over the legacy mapping: any
//! unrecognized framework resolves to [`EngineKind::OnnxForward`], exactly as the
//! old `else` arm fell through to a stateless ONNX session. Richer validation
//! (e.g. rejecting an `llm` tag on a GGUF file) is layered on top separately.

use crate::loader::Manifest;

/// The runtime engine family selected by a package's `framework`.
///
/// This names *what kind of engine* runs the model, independent of the concrete
/// backend chosen for the available hardware/features (e.g. `GgufGenerate` may be
/// served by the native-CUDA GGUF backend or the llama.cpp one).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EngineKind {
    /// GGUF causal-LM, generative. Tokenizer is embedded in the `.gguf` file.
    /// Legacy framework: `"gguf"`.
    GgufGenerate,
    /// ONNX causal-LM, generative (autoregressive decode loop). Requires an
    /// external `tokenizer.json`. Legacy framework: `"llm"`.
    OnnxGenerate,
    /// safetensors weights run with custom CUDA kernels.
    /// Legacy framework: `"native"` / `"safetensors"`.
    Native,
    /// A plain ONNX graph run as a single stateless forward pass (tensors in,
    /// tensors out). Legacy framework: `"onnx"` (and anything unrecognized).
    OnnxForward,
    /// ONNX encoder run for embeddings (forward pass + pooling). Selected by
    /// `format=onnx`, `task=embed`.
    OnnxEmbed,
    /// ONNX classifier (forward pass + softmax over logits). Selected by
    /// `format=onnx`, `task=classify`.
    OnnxClassify,
    /// ONNX object detector (forward pass + box decode + non-max suppression).
    /// Emits `[num_detections, 6]` = `[x1, y1, x2, y2, score, class_id]`.
    /// Selected by `format=onnx`, `task=detect`.
    OnnxDetect,
    /// ONNX CTC speech recognizer (forward pass over acoustic features +
    /// greedy CTC decode). Emits `[num_tokens]` token ids. Selected by
    /// `format=onnx`, `task=transcribe`. Encoder-decoder ASR (Whisper) is out of
    /// scope: it is an autoregressive decode loop, not a single forward pass.
    OnnxTranscribe,
}

impl EngineKind {
    /// Classify a raw `framework` string. Case- and whitespace-insensitive.
    ///
    /// Unrecognized values map to [`EngineKind::OnnxForward`] to preserve the
    /// legacy fall-through behavior (`onnx`, `pytorch`, `tensorflow`, and any
    /// unknown tag all went to the stateless ONNX path).
    pub fn from_framework(framework: &str) -> Self {
        match framework.trim().to_ascii_lowercase().as_str() {
            "gguf" => Self::GgufGenerate,
            "llm" => Self::OnnxGenerate,
            "native" | "safetensors" => Self::Native,
            _ => Self::OnnxForward,
        }
    }

    /// Classify the engine family selected by a manifest.
    ///
    /// Prefers the orthogonal `format`/`model_type`/`task` axes when present and
    /// falls back to the legacy `framework`. Infallible and best-effort (it
    /// always yields *some* family for dispatch); use [`EngineKind::validate`]
    /// to reject incoherent declarations.
    pub fn resolve(manifest: &Manifest) -> Self {
        // Fast path: no new axes set -> exactly the legacy classification.
        if manifest.format.is_none() && manifest.model_type.is_none() && manifest.task.is_none() {
            return Self::from_framework(&manifest.framework);
        }
        let format = effective_format(manifest);
        let task = effective_task(manifest);
        match format.as_str() {
            "gguf" => Self::GgufGenerate,
            "safetensors" => Self::Native,
            // onnx (and anything else) dispatches on the requested task.
            _ => match task.as_str() {
                "generate" => Self::OnnxGenerate,
                "embed" => Self::OnnxEmbed,
                "classify" => Self::OnnxClassify,
                "detect" => Self::OnnxDetect,
                "transcribe" => Self::OnnxTranscribe,
                _ => Self::OnnxForward,
            },
        }
    }

    /// Validate that a manifest's axes form a coherent, supported combination.
    ///
    /// This is where the historically-silent mistakes become clear errors:
    /// e.g. a `.gguf` model tagged `framework = "llm"`, or a GGUF model declared
    /// with a non-causal `model_type`. Returns `Ok(())` for legacy manifests
    /// that resolve to a supported engine, so existing packages keep loading.
    pub fn validate(manifest: &Manifest) -> Result<(), String> {
        // Reject explicitly-set values outside the known vocabularies.
        check_known("format", manifest.format.as_deref(), VALID_FORMATS)?;
        check_known(
            "model_type",
            manifest.model_type.as_deref(),
            VALID_MODEL_TYPES,
        )?;
        check_known("task", manifest.task.as_deref(), VALID_TASKS)?;

        let format = effective_format(manifest);
        let model_type = effective_model_type(manifest);
        let task = effective_task(manifest);

        // The declared format must not contradict the model file's extension.
        if let Some(ext_format) = format_from_model_file(&manifest.model_file) {
            if ext_format != format {
                return Err(format!(
                    "model_file `{}` is a {} model but the package resolves to format `{}` \
                     (framework={:?}, format={:?}); set format/framework to `{}`",
                    manifest.model_file,
                    ext_format,
                    format,
                    manifest.framework,
                    manifest.format,
                    ext_format
                ));
            }
        }

        // Task must be valid for the model type.
        let task_ok = match model_type.as_str() {
            "causal-lm" => matches!(task.as_str(), "generate" | "embed" | "forward"),
            "embedding" => task == "embed",
            "seq-classifier" => task == "classify",
            "detector" => task == "detect",
            "asr" => task == "transcribe",
            "seq2seq" => matches!(task.as_str(), "generate" | "forward"),
            // opaque graphs only support a raw forward pass.
            _ => task == "forward",
        };
        if !task_ok {
            return Err(format!(
                "task `{}` is not valid for model_type `{}`",
                task, model_type
            ));
        }

        // GGUF is currently causal-LM/generate only (tokenizer embedded; decode loop).
        if format == "gguf" && (model_type != "causal-lm" || task != "generate") {
            return Err(format!(
                "GGUF models are served as causal-lm/generate only; got model_type=`{}`, task=`{}`. \
                 (A GGUF embedding/classifier backend does not exist yet.)",
                model_type, task
            ));
        }

        Ok(())
    }

    /// Whether this engine performs autoregressive text generation.
    pub fn is_generative(&self) -> bool {
        matches!(self, Self::GgufGenerate | Self::OnnxGenerate)
    }

    /// Whether this engine loads a GGUF model (embedded tokenizer).
    pub fn is_gguf(&self) -> bool {
        matches!(self, Self::GgufGenerate)
    }

    /// Whether this engine is the ONNX **generative** path (`LLMBackend`): the
    /// one that requires an external tokenizer and gets LLM scheduler tuning,
    /// gguf auto-sizing hints, and pipeline-parallel handling. Legacy `"llm"`.
    pub fn is_onnx_generate(&self) -> bool {
        matches!(self, Self::OnnxGenerate)
    }

    /// Whether this engine runs an ONNX Runtime session.
    pub fn uses_onnx_session(&self) -> bool {
        matches!(
            self,
            Self::OnnxGenerate
                | Self::OnnxForward
                | Self::OnnxEmbed
                | Self::OnnxClassify
                | Self::OnnxDetect
                | Self::OnnxTranscribe
        )
    }

    /// Whether a backend exists for this engine kind. All current cells are
    /// implemented; this stays as the extension point — a new variant added
    /// without a backend gets a `=> false` arm here, and the runtime rejects it
    /// with a clear "not implemented yet" error rather than silently running the
    /// wrong path. The match is exhaustive so a new variant won't compile until
    /// its status is declared.
    pub fn is_implemented(&self) -> bool {
        match self {
            Self::GgufGenerate
            | Self::OnnxGenerate
            | Self::Native
            | Self::OnnxForward
            | Self::OnnxEmbed
            | Self::OnnxClassify
            | Self::OnnxDetect
            | Self::OnnxTranscribe => true,
        }
    }

    /// Stable label for logs/diagnostics.
    pub fn label(&self) -> &'static str {
        match self {
            Self::GgufGenerate => "gguf-generate",
            Self::OnnxGenerate => "onnx-generate",
            Self::Native => "native",
            Self::OnnxForward => "onnx-forward",
            Self::OnnxEmbed => "onnx-embed",
            Self::OnnxClassify => "onnx-classify",
            Self::OnnxDetect => "onnx-detect",
            Self::OnnxTranscribe => "onnx-transcribe",
        }
    }
}

/// Known model file formats / loaders.
pub const VALID_FORMATS: &[&str] = &["onnx", "gguf", "safetensors"];
/// Known model capability classes.
pub const VALID_MODEL_TYPES: &[&str] = &[
    "causal-lm",
    "embedding",
    "seq-classifier",
    "detector",
    "asr",
    "seq2seq",
    "opaque",
];
/// Known serving operations.
pub const VALID_TASKS: &[&str] = &[
    "generate",
    "embed",
    "classify",
    "detect",
    "transcribe",
    "rerank",
    "forward",
];

fn check_known(field: &str, value: Option<&str>, allowed: &[&str]) -> Result<(), String> {
    if let Some(v) = value {
        let v = v.trim().to_ascii_lowercase();
        if !allowed.iter().any(|a| *a == v) {
            return Err(format!(
                "unknown {} `{}`; expected one of {:?}",
                field, v, allowed
            ));
        }
    }
    Ok(())
}

fn norm(value: &Option<String>) -> Option<String> {
    value
        .as_deref()
        .map(|v| v.trim().to_ascii_lowercase())
        .filter(|v| !v.is_empty())
}

/// Effective model file format: the explicit `format`, else inferred from the
/// legacy `framework`.
pub fn effective_format(manifest: &Manifest) -> String {
    if let Some(f) = norm(&manifest.format) {
        return f;
    }
    match manifest.framework.trim().to_ascii_lowercase().as_str() {
        "gguf" => "gguf",
        "native" | "safetensors" => "safetensors",
        // "llm", "onnx", and anything unknown are ONNX-format (or treated as such).
        _ => "onnx",
    }
    .to_string()
}

/// Effective model capability class: explicit `model_type`, else inferred from
/// the legacy `framework`.
pub fn effective_model_type(manifest: &Manifest) -> String {
    if let Some(t) = norm(&manifest.model_type) {
        return t;
    }
    match manifest.framework.trim().to_ascii_lowercase().as_str() {
        "gguf" | "llm" => "causal-lm",
        // onnx / safetensors / native / unknown: we don't model the architecture.
        _ => "opaque",
    }
    .to_string()
}

/// Effective serving operation: explicit `task`, else defaulted from the
/// effective model type.
pub fn effective_task(manifest: &Manifest) -> String {
    if let Some(t) = norm(&manifest.task) {
        return t;
    }
    match effective_model_type(manifest).as_str() {
        "causal-lm" => "generate",
        "embedding" => "embed",
        "seq-classifier" => "classify",
        "detector" => "detect",
        "asr" => "transcribe",
        "seq2seq" => "generate",
        _ => "forward",
    }
    .to_string()
}

/// Confident format implied by a model file's extension, if recognized.
fn format_from_model_file(model_file: &str) -> Option<&'static str> {
    let ext = std::path::Path::new(model_file)
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    match ext.as_str() {
        "gguf" => Some("gguf"),
        "onnx" => Some("onnx"),
        "safetensors" => Some("safetensors"),
        _ => None,
    }
}

#[cfg(test)]
#[path = "engine_kind_tests.rs"]
mod engine_kind_tests;
