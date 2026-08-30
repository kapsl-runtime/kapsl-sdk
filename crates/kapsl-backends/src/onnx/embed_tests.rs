//! Tests for ONNX embedding post-processing.

use super::*;
use kapsl_engine_api::NamedTensor;

fn approx(a: &[f32], b: &[f32]) {
    assert_eq!(a.len(), b.len(), "length mismatch: {a:?} vs {b:?}");
    for (x, y) in a.iter().zip(b) {
        assert!((x - y).abs() < 1e-5, "values differ: {a:?} vs {b:?}");
    }
}

#[test]
fn masked_mean_pool_averages_active_tokens() {
    // batch=1, seq=2, dim=2: rows [1,2] and [3,4].
    let hidden = vec![1.0, 2.0, 3.0, 4.0];
    // Both tokens active -> mean = [2, 3].
    approx(
        &masked_mean_pool(&hidden, 1, 2, 2, &[1.0, 1.0]),
        &[2.0, 3.0],
    );
    // Second token masked out -> just the first row [1, 2].
    approx(
        &masked_mean_pool(&hidden, 1, 2, 2, &[1.0, 0.0]),
        &[1.0, 2.0],
    );
}

#[test]
fn masked_mean_pool_handles_all_padding_without_nan() {
    let hidden = vec![5.0, 6.0];
    let out = masked_mean_pool(&hidden, 1, 1, 2, &[0.0]);
    assert!(out.iter().all(|v| v.is_finite()), "got non-finite: {out:?}");
}

#[test]
fn l2_normalize_rows_makes_unit_vectors() {
    let mut v = vec![3.0, 4.0, 0.0, 0.0];
    l2_normalize_rows(&mut v, 2, 2);
    approx(&v[0..2], &[0.6, 0.8]);
    // A zero row stays finite (clamped denominator).
    assert!(v[2..4].iter().all(|x| x.is_finite()));
}

fn mask_request(mask: &[i64]) -> InferenceRequest {
    let mut data = Vec::new();
    for m in mask {
        data.extend_from_slice(&m.to_le_bytes());
    }
    let mask_tensor = BinaryTensorPacket {
        shape: vec![1, mask.len() as i64],
        dtype: TensorDtype::Int64,
        data,
    };
    let mut req = InferenceRequest::new(BinaryTensorPacket {
        shape: vec![1, mask.len() as i64],
        dtype: TensorDtype::Int64,
        data: Vec::new(),
    });
    req.additional_inputs = vec![NamedTensor {
        name: "attention_mask".to_string(),
        tensor: mask_tensor,
    }];
    req
}

#[test]
fn embed_from_output_pools_3d_hidden_states_with_mask() {
    // [batch=1, seq=2, dim=2] hidden states.
    let hidden = f32_packet(vec![1, 2, 2], vec![1.0, 2.0, 3.0, 4.0]);

    // Both active, no normalization -> mean [2, 3].
    let out = embed_from_output(&hidden, &mask_request(&[1, 1]), false).unwrap();
    assert_eq!(out.shape, vec![1, 2]);
    approx(&bytes_to_f32(&out.data), &[2.0, 3.0]);

    // Second token padded -> first row only [1, 2].
    let out = embed_from_output(&hidden, &mask_request(&[1, 0]), false).unwrap();
    approx(&bytes_to_f32(&out.data), &[1.0, 2.0]);
}

#[test]
fn embed_from_output_normalizes_when_requested() {
    let hidden = f32_packet(vec![1, 1, 2], vec![3.0, 4.0]);
    let out = embed_from_output(&hidden, &mask_request(&[1]), true).unwrap();
    approx(&bytes_to_f32(&out.data), &[0.6, 0.8]);
}

#[test]
fn embed_from_output_passes_through_prepooled_2d() {
    // Model already emits a pooled [batch=1, dim=2] vector; only normalize it.
    let pooled = f32_packet(vec![1, 2], vec![3.0, 4.0]);
    let req = InferenceRequest::new(BinaryTensorPacket {
        shape: vec![1, 2],
        dtype: TensorDtype::Int64,
        data: Vec::new(),
    });
    let out = embed_from_output(&pooled, &req, true).unwrap();
    assert_eq!(out.shape, vec![1, 2]);
    approx(&bytes_to_f32(&out.data), &[0.6, 0.8]);
}

#[test]
fn embed_from_output_rejects_non_float32() {
    let bad = BinaryTensorPacket {
        shape: vec![1, 2],
        dtype: TensorDtype::Int64,
        data: vec![0u8; 16],
    };
    let req = InferenceRequest::new(bad.clone());
    assert!(embed_from_output(&bad, &req, true).is_err());
}
