//! CPU-side weight tensors loaded from safetensors.
//!
//! Weights are kept as raw byte buffers plus dtype/shape metadata.
//! The native backend uploads them to GPU on first use.

use half::{bf16, f16};
use std::sync::Arc;
use thiserror::Error;

#[cfg(target_endian = "little")]
fn assert_little_endian() {}

#[cfg(target_endian = "big")]
fn assert_little_endian() {
    panic!("zero-copy tensor views require a little-endian target");
}

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
///
/// Quantized variants keep GGML's own spelling (`Q8_0`, `Q4_K`) so the names
/// match the GGUF type table and the kernels that consume them, rather than
/// Rust's camel case.
#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DType {
    F32,
    F16,
    BF16,
    I8,
    U8,
    /// Q8_0: 34 raw bytes per 32 elements — [f16 scale, i8 qs[32]]
    Q8_0,
    /// Q4_K: 144 raw bytes per 256 elements — [f16 d, f16 dmin, u8 scales[12], u8 qs[128]]
    Q4_K,
}

impl DType {
    /// Bytes per element for non-quantized types. Panics for quantized types — use raw_bytes_for_numel.
    pub fn byte_size(self) -> usize {
        match self {
            DType::F32 => 4,
            DType::F16 | DType::BF16 => 2,
            DType::I8 | DType::U8 => 1,
            DType::Q8_0 | DType::Q4_K => panic!(
                "byte_size() not valid for quantized dtype {:?} — use raw_bytes_for_numel()",
                self
            ),
        }
    }

    /// Total encoded byte count for `numel` elements.
    ///
    /// Quantized tensors must contain complete GGML blocks. The multiplication
    /// is checked so corrupt or impossible shapes fail instead of wrapping.
    pub fn raw_bytes_for_numel(self, numel: usize) -> usize {
        match self {
            DType::Q8_0 => {
                assert_eq!(numel % 32, 0, "Q8_0 tensors require 32-element blocks");
                (numel / 32)
                    .checked_mul(34)
                    .expect("Q8_0 tensor byte size overflow")
            }
            DType::Q4_K => {
                assert_eq!(numel % 256, 0, "Q4_K tensors require 256-element blocks");
                (numel / 256)
                    .checked_mul(144)
                    .expect("Q4_K tensor byte size overflow")
            }
            _ => numel
                .checked_mul(self.byte_size())
                .expect("tensor byte size overflow"),
        }
    }

    /// Parse a safetensors dtype spelling. This intentionally returns `Option`
    /// because callers attach the tensor name to unsupported-dtype errors.
    #[allow(clippy::should_implement_trait)]
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
        self.shape
            .iter()
            .try_fold(1usize, |count, dimension| count.checked_mul(*dimension))
            .expect("tensor element count overflow")
    }

    fn assert_dense_byte_len(&self, element_size: usize) {
        let expected = self
            .numel()
            .checked_mul(element_size)
            .expect("tensor byte size overflow");
        assert_eq!(
            self.bytes.len(),
            expected,
            "tensor byte length does not match dtype and shape"
        );
    }

    /// View the raw buffer as an f16 slice without allocating.
    ///
    /// Panics when the dtype, byte length, platform endianness, or allocation
    /// alignment does not permit a valid zero-copy view. Use [`Self::to_f16_vec`]
    /// when an alignment-independent conversion is required.
    pub fn as_f16(&self) -> &[f16] {
        assert_eq!(self.dtype, DType::F16);
        assert_little_endian();
        self.assert_dense_byte_len(2);
        // SAFETY: every `u16` bit pattern is a valid `f16`. `align_to` reports
        // any unaligned prefix/suffix, which are rejected below.
        let (prefix, values, suffix) = unsafe { self.bytes.as_slice().align_to::<f16>() };
        assert!(
            prefix.is_empty() && suffix.is_empty() && values.len() == self.numel(),
            "F16 tensor buffer is not aligned for a zero-copy view"
        );
        values
    }

    /// View the raw buffer as a bf16 slice without allocating.
    ///
    /// The same validation and alignment requirements as [`Self::as_f16`]
    /// apply.
    pub fn as_bf16(&self) -> &[bf16] {
        assert_eq!(self.dtype, DType::BF16);
        assert_little_endian();
        self.assert_dense_byte_len(2);
        // SAFETY: every `u16` bit pattern is a valid `bf16`; see `as_f16`.
        let (prefix, values, suffix) = unsafe { self.bytes.as_slice().align_to::<bf16>() };
        assert!(
            prefix.is_empty() && suffix.is_empty() && values.len() == self.numel(),
            "BF16 tensor buffer is not aligned for a zero-copy view"
        );
        values
    }

    /// Convert a floating-point tensor to f16 using little-endian decoding.
    ///
    /// This allocates but does not depend on the alignment of the byte buffer.
    /// Panics for non-floating-point dtypes or inconsistent shape metadata.
    pub fn to_f16_vec(&self) -> Vec<f16> {
        match self.dtype {
            DType::F16 => {
                self.assert_dense_byte_len(2);
                self.bytes
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .map(|bytes| f16::from_bits(u16::from_le_bytes(*bytes)))
                    .collect()
            }
            DType::BF16 => {
                self.assert_dense_byte_len(2);
                self.bytes
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .map(|bytes| {
                        let value = bf16::from_bits(u16::from_le_bytes(*bytes));
                        f16::from_f32(value.to_f32())
                    })
                    .collect()
            }
            DType::F32 => {
                self.assert_dense_byte_len(4);
                self.bytes
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|bytes| f16::from_f32(f32::from_le_bytes(*bytes)))
                    .collect()
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
    /// Number of transformer layers represented by this weight set.
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn u16_bytes(values: impl IntoIterator<Item = u16>) -> Vec<u8> {
        values.into_iter().flat_map(u16::to_le_bytes).collect()
    }

    #[test]
    fn converts_f16_bytes_without_relying_on_buffer_alignment() {
        let expected = [f16::from_f32(1.5), f16::from_f32(-2.0)];
        let tensor = TensorData::new(
            u16_bytes(expected.iter().map(|value| value.to_bits())),
            DType::F16,
            vec![2],
        );

        assert_eq!(tensor.to_f16_vec(), expected);
    }

    #[test]
    fn converts_bf16_and_f32_little_endian_bytes() {
        let bf16_values = [bf16::from_f32(1.5), bf16::from_f32(-2.0)];
        let bf16_tensor = TensorData::new(
            u16_bytes(bf16_values.iter().map(|value| value.to_bits())),
            DType::BF16,
            vec![2],
        );
        assert_eq!(
            bf16_tensor.to_f16_vec(),
            vec![f16::from_f32(1.5), f16::from_f32(-2.0)]
        );

        let f32_tensor = TensorData::new(
            [1.5_f32, -2.0]
                .into_iter()
                .flat_map(f32::to_le_bytes)
                .collect(),
            DType::F32,
            vec![2],
        );
        assert_eq!(
            f32_tensor.to_f16_vec(),
            vec![f16::from_f32(1.5), f16::from_f32(-2.0)]
        );
    }

    #[test]
    fn quantized_byte_counts_require_complete_blocks() {
        assert_eq!(DType::Q8_0.raw_bytes_for_numel(64), 68);
        assert_eq!(DType::Q4_K.raw_bytes_for_numel(512), 288);
    }

    #[test]
    #[should_panic(expected = "Q8_0 tensors require 32-element blocks")]
    fn rejects_partial_quantized_blocks() {
        DType::Q8_0.raw_bytes_for_numel(33);
    }
}
