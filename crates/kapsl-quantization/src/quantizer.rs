//! Weight quantization: convert float32 tensors to quantized formats.
//!
//! Reduces model size on disk and VRAM usage at load time. Complements
//! [`kv_cache`](super::kv_cache) which reduces runtime memory during inference.
//!
//! # Supported formats
//!
//! | Format | Bits/weight | Technique |
//! |--------|------------|-----------|
//! | Int8 symmetric  | 8 | absmax per-tensor scale |
//! | Int8 asymmetric | 8 | min-max per-tensor scale + zero-point |
//! | Int4 grouped    | 4 | absmax per-group (group_size=128) |
//!
//! # Example
//!
//! ```rust
//! use kapsl_quantization::quantizer::{quantize_int8_symmetric, quantize_int4_grouped};
//!
//! let weights = vec![0.1f32, -0.5, 0.3, 1.2, -0.8, 0.0];
//! let tensor = quantize_int8_symmetric(&weights, &[2, 3]);
//!
//! let tensor4 = quantize_int4_grouped(&weights, &[2, 3], 128);
//! ```

use crate::tensor::{Int4Tensor, Int8Tensor, QuantizedTensor};
use anyhow::{anyhow, Result};
use half::f16;
use std::sync::Arc;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Maximum positive value of a signed 8-bit integer.
const I8_MAX: f32 = 127.0;

/// Scale of unsigned 8-bit range used for asymmetric quantization.
const U8_RANGE: f32 = 255.0;

/// Number of values per group for Int4 grouped quantization.
pub const DEFAULT_GROUP_SIZE: usize = 128;

/// Maximum value encodable in 4 bits (unsigned).
const U4_MAX: f32 = 15.0;

// ---------------------------------------------------------------------------
// Int8 symmetric (absmax)
// ---------------------------------------------------------------------------

/// Quantize `weights` to Int8 using symmetric (absmax) per-tensor scaling.
///
/// `scale = max(|w|) / 127`
/// `q = clamp(round(w / scale), -127, 127)`
///
/// ~4× size reduction vs f32. Zero-point is always 0.
pub fn quantize_int8_symmetric(weights: &[f32], shape: &[usize]) -> QuantizedTensor {
    let absmax = weights
        .iter()
        .copied()
        .fold(0.0f32, |acc, v| acc.max(v.abs()));

    let scale = if absmax == 0.0 { 1.0 } else { absmax / I8_MAX };

    let quantized: Vec<i8> = weights
        .iter()
        .map(|&w| (w / scale).round().clamp(-I8_MAX, I8_MAX) as i8)
        .collect();

    QuantizedTensor::Int8(Int8Tensor {
        weight: Arc::new(quantized),
        scale,
        zero_point: 0,
        symmetric: true,
        shape: shape.to_vec(),
    })
}

// ---------------------------------------------------------------------------
// Int8 asymmetric (min-max)
// ---------------------------------------------------------------------------

/// Quantize `weights` to Int8 using asymmetric (min-max) per-tensor scaling.
///
/// `scale = (max - min) / 255`
/// `zero_point = round(-min / scale)`
/// `q = clamp(round(w / scale) + zero_point, 0, 255)` stored as i8 offset
///
/// Better for one-sided distributions (e.g. post-ReLU activations).
pub fn quantize_int8_asymmetric(weights: &[f32], shape: &[usize]) -> QuantizedTensor {
    let min = weights.iter().copied().fold(f32::INFINITY, f32::min);
    let max = weights.iter().copied().fold(f32::NEG_INFINITY, f32::max);

    let range = max - min;
    let scale = if range == 0.0 { 1.0 } else { range / U8_RANGE };
    let zero_point = (-min / scale).round() as i32;

    // Clamp to [0, 255], then bit-cast to i8 so the full unsigned range fits
    // in the i8 storage. Dequantization reverses with `q as u8 as i32`.
    let quantized: Vec<i8> = weights
        .iter()
        .map(|&w| {
            let q = (w / scale).round() as i32 + zero_point;
            q.clamp(0, 255) as u8 as i8
        })
        .collect();

    QuantizedTensor::Int8(Int8Tensor {
        weight: Arc::new(quantized),
        scale,
        zero_point,
        symmetric: false,
        shape: shape.to_vec(),
    })
}

// ---------------------------------------------------------------------------
// Int4 grouped (absmax per group)
// ---------------------------------------------------------------------------

/// Quantize `weights` to Int4 using per-group absmax scaling.
///
/// Weights are split into groups of `group_size`. Each group gets its own
/// f16 scale factor. Two 4-bit values are packed per byte.
///
/// ~8× size reduction vs f32. Maintains better accuracy than per-tensor Int4
/// by limiting the quantization range to a local group.
pub fn quantize_int4_grouped(
    weights: &[f32],
    shape: &[usize],
    group_size: usize,
) -> Result<QuantizedTensor> {
    if group_size == 0 {
        return Err(anyhow!("group_size must be > 0"));
    }
    if weights.is_empty() {
        return Err(anyhow!("weights must not be empty"));
    }

    let mut packed: Vec<u8> = Vec::with_capacity(weights.len() / 2 + 1);
    let mut scales: Vec<f16> = Vec::with_capacity(weights.len().div_ceil(group_size));

    for group in weights.chunks(group_size) {
        let absmax = group.iter().copied().fold(0.0f32, |a, v| a.max(v.abs()));
        let scale = if absmax == 0.0 { 1.0 } else { absmax / (U4_MAX / 2.0) };
        scales.push(f16::from_f32(scale));

        // Quantize group: map [-absmax, absmax] → [0, 15], center at 8
        let indices: Vec<u8> = group
            .iter()
            .map(|&w| {
                let q = (w / scale + U4_MAX / 2.0).round().clamp(0.0, U4_MAX) as u8;
                q
            })
            .collect();

        // Pack pairs of 4-bit values into bytes
        for pair in indices.chunks(2) {
            let lo = pair[0] & 0xF;
            let hi = if pair.len() > 1 { pair[1] & 0xF } else { 0 };
            packed.push(lo | (hi << 4));
        }
    }

    Ok(QuantizedTensor::Int4(Int4Tensor {
        packed,
        scales: Arc::new(scales),
        group_size,
        shape: shape.to_vec(),
    }))
}

// ---------------------------------------------------------------------------
// Dequantization helpers (for verification / inference fallback)
// ---------------------------------------------------------------------------

/// Dequantize an [`Int8Tensor`] back to f32.
///
/// For symmetric tensors (`zero_point == 0`) the signed value is used directly.
/// For asymmetric tensors (`zero_point != 0`) the bit pattern is reinterpreted
/// as unsigned (`q as u8 as i32`) before the zero-point shift is applied.
pub fn dequantize_int8(tensor: &Int8Tensor) -> Vec<f32> {
    tensor
        .weight
        .iter()
        .map(|&q| {
            let q_int = if tensor.symmetric {
                q as i32
            } else {
                // Asymmetric: stored as unsigned bit-cast into i8.
                q as u8 as i32
            };
            (q_int - tensor.zero_point) as f32 * tensor.scale
        })
        .collect()
}

/// Dequantize an [`Int4Tensor`] back to f32.
pub fn dequantize_int4(tensor: &Int4Tensor) -> Vec<f32> {
    let mut out = Vec::with_capacity(tensor.shape.iter().product());
    let center = U4_MAX / 2.0;

    for (group_idx, scale) in tensor.scales.iter().enumerate() {
        let scale_f32 = scale.to_f32();
        let byte_start = group_idx * tensor.group_size / 2;
        let byte_end = (byte_start + tensor.group_size / 2).min(tensor.packed.len());

        for &byte in &tensor.packed[byte_start..byte_end] {
            let lo = (byte & 0xF) as f32;
            let hi = ((byte >> 4) & 0xF) as f32;
            out.push((lo - center) * scale_f32);
            out.push((hi - center) * scale_f32);
        }
    }

    out.truncate(tensor.shape.iter().product());
    out
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const TOLERANCE_INT8: f32 = 0.015;
    const TOLERANCE_INT4: f32 = 0.15;

    fn max_abs_error(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max)
    }

    #[test]
    fn int8_symmetric_roundtrip() {
        let weights: Vec<f32> = (0..256).map(|i| (i as f32 - 128.0) / 128.0).collect();
        let tensor = quantize_int8_symmetric(&weights, &[256]);
        let QuantizedTensor::Int8(ref t) = tensor else { panic!() };
        let recovered = dequantize_int8(t);
        assert!(max_abs_error(&weights, &recovered) < TOLERANCE_INT8);
    }

    #[test]
    fn int8_symmetric_all_zeros() {
        let weights = vec![0.0f32; 64];
        let tensor = quantize_int8_symmetric(&weights, &[64]);
        let QuantizedTensor::Int8(ref t) = tensor else { panic!() };
        assert!(t.weight.iter().all(|&q| q == 0));
    }

    #[test]
    fn int8_asymmetric_positive_weights() {
        let weights: Vec<f32> = (0..128).map(|i| i as f32 / 128.0).collect();
        let tensor = quantize_int8_asymmetric(&weights, &[128]);
        let QuantizedTensor::Int8(ref t) = tensor else { panic!() };
        let recovered = dequantize_int8(t);
        assert!(max_abs_error(&weights, &recovered) < TOLERANCE_INT8);
    }

    #[test]
    fn int4_grouped_roundtrip() {
        let weights: Vec<f32> = (0..256).map(|i| (i as f32 - 128.0) / 128.0).collect();
        let tensor = quantize_int4_grouped(&weights, &[256], 128).unwrap();
        let QuantizedTensor::Int4(ref t) = tensor else { panic!() };
        let recovered = dequantize_int4(t);
        assert!(max_abs_error(&weights, &recovered) < TOLERANCE_INT4);
    }

    #[test]
    fn int4_grouped_rejects_zero_group_size() {
        let weights = vec![1.0f32; 128];
        assert!(quantize_int4_grouped(&weights, &[128], 0).is_err());
    }

    #[test]
    fn int4_grouped_packed_size() {
        let weights = vec![0.5f32; 256];
        let tensor = quantize_int4_grouped(&weights, &[256], 128).unwrap();
        let QuantizedTensor::Int4(ref t) = tensor else { panic!() };
        // 256 values / 2 = 128 bytes packed
        assert_eq!(t.packed.len(), 128);
        // 256 / 128 = 2 groups → 2 scales
        assert_eq!(t.scales.len(), 2);
    }

    #[test]
    fn int8_shape_preserved() {
        let weights = vec![1.0f32; 32];
        let tensor = quantize_int8_symmetric(&weights, &[4, 8]);
        let QuantizedTensor::Int8(ref t) = tensor else { panic!() };
        assert_eq!(t.shape, vec![4, 8]);
    }
}
