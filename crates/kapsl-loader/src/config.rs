//! HuggingFace `config.json` parser.
//!
//! Supports LLaMA-style architectures (LLaMA, Mistral, Qwen, Phi-3).
//! Unknown fields are ignored so the parser stays forward-compatible.

use serde::{Deserialize, Serialize};
use std::path::Path;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ConfigError {
    #[error("Failed to read config.json: {0}")]
    Io(#[from] std::io::Error),
    #[error("Failed to parse config.json: {0}")]
    Parse(#[from] serde_json::Error),
    #[error("Config missing required field: {0}")]
    MissingField(&'static str),
}

/// Architecture family detected from `architectures` field.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ArchFamily {
    LLaMA,
    Mistral,
    Phi,
    Qwen,
    Unknown(String),
}

/// Normalised model configuration extracted from `config.json`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Total hidden (embedding) dimension.
    pub hidden_size: usize,
    /// FFN intermediate dimension (after gating).
    pub intermediate_size: usize,
    /// Number of transformer layers.
    pub num_hidden_layers: usize,
    /// Number of query heads.
    pub num_attention_heads: usize,
    /// Number of key/value heads. Equal to `num_attention_heads` for MHA;
    /// smaller for GQA (e.g. Mistral uses 8 KV heads for 32 query heads).
    #[serde(default)]
    pub num_key_value_heads: Option<usize>,
    /// Vocabulary size.
    pub vocab_size: usize,
    /// RMS-norm epsilon.
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f64,
    /// RoPE base frequency.
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f64,
    /// Maximum context length.
    #[serde(default = "default_max_position_embeddings")]
    pub max_position_embeddings: usize,
    /// Architecture strings from config (e.g. ["LlamaForCausalLM"]).
    #[serde(default)]
    pub architectures: Vec<String>,
}

fn default_rms_norm_eps() -> f64 {
    1e-5
}
fn default_rope_theta() -> f64 {
    10_000.0
}
fn default_max_position_embeddings() -> usize {
    4096
}

impl ModelConfig {
    /// Load from a `config.json` file path.
    pub fn from_file(path: &Path) -> Result<Self, ConfigError> {
        let content = std::fs::read_to_string(path)?;
        let cfg: Self = serde_json::from_str(&content)?;
        Ok(cfg)
    }

    /// Infer config from a model directory (looks for `config.json`).
    pub fn from_model_dir(dir: &Path) -> Result<Self, ConfigError> {
        Self::from_file(&dir.join("config.json"))
    }

    /// Per-head dimension derived from hidden size and head count.
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    /// Effective number of KV heads (GQA support).
    pub fn num_kv_heads(&self) -> usize {
        self.num_key_value_heads
            .unwrap_or(self.num_attention_heads)
    }

    /// GQA group size: how many query heads share one KV head.
    pub fn gqa_groups(&self) -> usize {
        self.num_attention_heads / self.num_kv_heads()
    }

    /// Detected architecture family.
    pub fn arch_family(&self) -> ArchFamily {
        for arch in &self.architectures {
            let lower = arch.to_ascii_lowercase();
            if lower.contains("llama") {
                return ArchFamily::LLaMA;
            }
            if lower.contains("mistral") {
                return ArchFamily::Mistral;
            }
            if lower.contains("phi") {
                return ArchFamily::Phi;
            }
            if lower.contains("qwen") {
                return ArchFamily::Qwen;
            }
        }
        ArchFamily::Unknown(self.architectures.first().cloned().unwrap_or_default())
    }
}
