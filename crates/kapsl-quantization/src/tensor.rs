use half::f16;
use std::sync::Arc;

#[derive(Debug, Clone)]
pub enum QuantizedTensor {
    Gptq(GptqTensor),
    Awq(AwqTensor),
    Int8(Int8Tensor),
    Int4(Int4Tensor),
}

#[derive(Debug, Clone)]
pub struct GptqTensor {
    pub qweight: Arc<Vec<u32>>,
    pub qzeros: Arc<Vec<u32>>,
    pub scales: Arc<Vec<f16>>,
    pub g_idx: Option<Arc<Vec<u32>>>,
    pub bias: Option<Arc<Vec<f16>>>,
    pub shape: Vec<usize>,
}

#[derive(Debug, Clone)]
pub struct AwqTensor {
    pub qweight: Arc<Vec<u32>>,
    pub qzeros: Arc<Vec<u32>>,
    pub scales: Arc<Vec<f16>>,
    pub shape: Vec<usize>,
}

#[derive(Debug, Clone)]
pub struct Int8Tensor {
    pub weight: Arc<Vec<i8>>,
    pub scale: f32,
    pub zero_point: i32,
    /// `true` for symmetric (absmax) — values are signed i8 in [-127, 127].
    /// `false` for asymmetric (min-max) — values are unsigned [0, 255] bit-cast
    /// into i8; dequantize via `q as u8 as i32`.
    pub symmetric: bool,
    pub shape: Vec<usize>,
}

/// Int4 grouped-quantization tensor.
///
/// Two 4-bit values are packed per byte (lo nibble = first, hi nibble = second).
/// Each group of `group_size` weights shares one f16 scale centered at 8
/// (i.e. the unsigned range [0, 15] is shifted to [-7.5, 7.5]).
#[derive(Debug, Clone)]
pub struct Int4Tensor {
    /// Packed 4-bit weights: two values per byte.
    pub packed: Vec<u8>,
    /// Per-group f16 scale factors.
    pub scales: Arc<Vec<f16>>,
    /// Number of weights per scale group.
    pub group_size: usize,
    pub shape: Vec<usize>,
}
