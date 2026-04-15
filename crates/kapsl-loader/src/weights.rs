//! CPU-side weight tensors loaded from safetensors.
//!
//! Weights are kept as raw byte buffers plus dtype/shape metadata.
//! The native backend uploads them to GPU on first use.

use half::{bf16, f16};
use std::sync::Arc;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum WeightError {
    #[error("Tensor '{0}' not found in safetensors file")]
    Missing(String),
    #[error("Shape mismatch for '{name}': expected {expected:?}, got {actual:?}")]
    ShapeMismatch {
        name: String,
        expected: Vec<usize>,
        actual: Vec<usize>,
    },
    #[error("Unsupported dtype '{0}' for tensor '{1}'")]
    UnsupportedDtype(String, String),
}

/// Element type of a tensor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DType {
    F32,
    F16,
    BF16,
    I8,
    U8,
}

impl DType {
    /// Bytes per element.
    pub fn byte_size(self) -> usize {
        match self {
            DType::F32 => 4,
            DType::F16 | DType::BF16 => 2,
            DType::I8 | DType::U8 => 1,
        }
    }

    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "F32" => Some(DType::F32),
            "F16" => Some(DType::F16),
            "BF16" => Some(DType::BF16),
            "I8" => Some(DType::I8),
            "U8" => Some(DType::U8),
            _ => None,
        }
    }
}

/// A named tensor loaded from safetensors (CPU memory).
#[derive(Clone)]
pub struct TensorData {
    /// Raw bytes (dtype-encoded, little-endian).
    pub bytes: Arc<Vec<u8>>,
    pub dtype: DType,
    pub shape: Vec<usize>,
}

impl TensorData {
    pub fn new(bytes: Vec<u8>, dtype: DType, shape: Vec<usize>) -> Self {
        Self {
            bytes: Arc::new(bytes),
            dtype,
            shape,
        }
    }

    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    /// View as f16 slice (panics if dtype is not F16).
    pub fn as_f16(&self) -> &[f16] {
        assert_eq!(self.dtype, DType::F16);
        let ptr = self.bytes.as_ptr() as *const f16;
        unsafe { std::slice::from_raw_parts(ptr, self.numel()) }
    }

    /// View as bf16 slice (panics if dtype is not BF16).
    pub fn as_bf16(&self) -> &[bf16] {
        assert_eq!(self.dtype, DType::BF16);
        let ptr = self.bytes.as_ptr() as *const bf16;
        unsafe { std::slice::from_raw_parts(ptr, self.numel()) }
    }

    /// Convert to f16 regardless of source dtype (allocates).
    pub fn to_f16_vec(&self) -> Vec<f16> {
        match self.dtype {
            DType::F16 => self.as_f16().to_vec(),
            DType::BF16 => self
                .as_bf16()
                .iter()
                .map(|v| f16::from_f32(v.to_f32()))
                .collect(),
            DType::F32 => {
                let ptr = self.bytes.as_ptr() as *const f32;
                let f32s = unsafe { std::slice::from_raw_parts(ptr, self.numel()) };
                f32s.iter().map(|&v| f16::from_f32(v)).collect()
            }
            _ => panic!("Cannot convert {:?} tensor to f16", self.dtype),
        }
    }
}

impl std::fmt::Debug for TensorData {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "TensorData({:?} {:?}, {} bytes)",
            self.dtype,
            self.shape,
            self.bytes.len()
        )
    }
}

/// Weights for a single transformer layer.
#[derive(Debug)]
pub struct LayerWeights {
    /// Pre-attention RMSNorm scale [hidden_size].
    pub input_layernorm: TensorData,
    /// Q projection [num_heads * head_dim, hidden_size].
    pub q_proj: TensorData,
    /// K projection [num_kv_heads * head_dim, hidden_size].
    pub k_proj: TensorData,
    /// V projection [num_kv_heads * head_dim, hidden_size].
    pub v_proj: TensorData,
    /// Output projection [hidden_size, num_heads * head_dim].
    pub o_proj: TensorData,
    /// Post-attention RMSNorm scale [hidden_size].
    pub post_attention_layernorm: TensorData,
    /// FFN gate projection [intermediate_size, hidden_size].
    pub gate_proj: TensorData,
    /// FFN up projection [intermediate_size, hidden_size].
    pub up_proj: TensorData,
    /// FFN down projection [hidden_size, intermediate_size].
    pub down_proj: TensorData,
}

/// All weights for a model loaded from safetensors.
#[derive(Debug)]
pub struct ModelWeights {
    pub config: crate::config::ModelConfig,
    /// Token embeddings [vocab_size, hidden_size].
    pub embed_tokens: TensorData,
    /// Per-layer weights in order from layer 0 to N-1.
    pub layers: Vec<LayerWeights>,
    /// Final RMSNorm scale [hidden_size].
    pub norm: TensorData,
    /// LM head [vocab_size, hidden_size]. May alias embed_tokens (tied weights).
    pub lm_head: TensorData,
}

impl ModelWeights {
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }
}
