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
    pub fn resolve(manifest: &Manifest) -> Self {
        Self::from_framework(&manifest.framework)
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

    /// Whether this engine runs an ONNX Runtime session (generative or forward).
    pub fn uses_onnx_session(&self) -> bool {
        matches!(self, Self::OnnxGenerate | Self::OnnxForward)
    }

    /// Stable label for logs/diagnostics.
    pub fn label(&self) -> &'static str {
        match self {
            Self::GgufGenerate => "gguf-generate",
            Self::OnnxGenerate => "onnx-generate",
            Self::Native => "native",
            Self::OnnxForward => "onnx-forward",
        }
    }
}

#[path = "engine_kind_tests.rs"]
mod engine_kind_tests;
