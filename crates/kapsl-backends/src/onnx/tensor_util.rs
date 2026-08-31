//! Tensor-packing helpers shared by the ONNX task backends.
//!
//! Each of the `onnx_{classify,detect,embed,transcribe}` backends decodes raw
//! little-endian tensor bytes and repacks its results the same way, so the
//! conversions live here rather than being copied per backend.

use kapsl_engine_api::{BinaryTensorPacket, TensorDtype};

/// Clamp a (possibly dynamic, i.e. negative) ONNX dimension to a usable extent.
pub(crate) fn dim_usize(d: i64) -> usize {
    d.max(0) as usize
}

/// Reinterpret little-endian bytes as f32 values, ignoring a trailing partial element.
pub(crate) fn bytes_to_f32(data: &[u8]) -> Vec<f32> {
    data.as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect()
}

/// Pack f32 values into a `Float32` tensor packet.
pub(crate) fn f32_packet(shape: Vec<i64>, values: Vec<f32>) -> BinaryTensorPacket {
    let mut data = Vec::with_capacity(values.len() * 4);
    for v in &values {
        data.extend_from_slice(&v.to_le_bytes());
    }
    BinaryTensorPacket {
        shape,
        dtype: TensorDtype::Float32,
        data,
    }
}

/// Pack i32 values into an `Int32` tensor packet.
pub(crate) fn i32_packet(shape: Vec<i64>, values: Vec<i32>) -> BinaryTensorPacket {
    let mut data = Vec::with_capacity(values.len() * 4);
    for v in &values {
        data.extend_from_slice(&v.to_le_bytes());
    }
    BinaryTensorPacket {
        shape,
        dtype: TensorDtype::Int32,
        data,
    }
}
