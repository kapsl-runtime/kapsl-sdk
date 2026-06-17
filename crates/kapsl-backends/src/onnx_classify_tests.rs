use super::*;

fn approx(a: &[f32], b: &[f32]) {
    assert_eq!(a.len(), b.len(), "length mismatch: {a:?} vs {b:?}");
    for (x, y) in a.iter().zip(b) {
        assert!((x - y).abs() < 1e-5, "values differ: {a:?} vs {b:?}");
    }
}

#[test]
fn softmax_rows_is_a_probability_distribution() {
    // Two rows; equal logits -> uniform.
    let mut v = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
    softmax_rows(&mut v, 2, 3);
    approx(&v[0..3], &[1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]);
    approx(&v[3..6], &[1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]);
    assert!((v[0..3].iter().sum::<f32>() - 1.0).abs() < 1e-5);
}

#[test]
fn softmax_rows_is_numerically_stable_for_large_logits() {
    let mut v = vec![1000.0, 1001.0];
    softmax_rows(&mut v, 1, 2);
    assert!(v.iter().all(|x| x.is_finite()), "overflowed: {v:?}");
    assert!((v.iter().sum::<f32>() - 1.0).abs() < 1e-5);
    assert!(v[1] > v[0]);
}

#[test]
fn classify_applies_softmax_to_logits() {
    // logits [batch=1, classes=3].
    let logits = f32_packet(vec![1, 3], vec![1.0, 2.0, 3.0]);
    let out = classify_from_output(&logits, true).unwrap();
    assert_eq!(out.shape, vec![1, 3]);
    let p = bytes_to_f32(&out.data);
    assert!((p.iter().sum::<f32>() - 1.0).abs() < 1e-5);
    // Monotonic: larger logit -> larger probability.
    assert!(p[0] < p[1] && p[1] < p[2]);
}

#[test]
fn classify_passthrough_when_softmax_disabled() {
    let probs = f32_packet(vec![1, 2], vec![0.3, 0.7]);
    let out = classify_from_output(&probs, false).unwrap();
    approx(&bytes_to_f32(&out.data), &[0.3, 0.7]);
}

#[test]
fn classify_promotes_1d_output_to_single_row() {
    let logits = f32_packet(vec![2], vec![0.0, 0.0]);
    let out = classify_from_output(&logits, true).unwrap();
    assert_eq!(out.shape, vec![1, 2]);
    approx(&bytes_to_f32(&out.data), &[0.5, 0.5]);
}

#[test]
fn classify_rejects_non_float32() {
    let bad = BinaryTensorPacket {
        shape: vec![1, 2],
        dtype: TensorDtype::Int64,
        data: vec![0u8; 16],
    };
    assert!(classify_from_output(&bad, true).is_err());
}

#[test]
fn classify_rejects_bad_rank() {
    let bad = f32_packet(vec![1, 2, 2], vec![1.0, 2.0, 3.0, 4.0]);
    assert!(classify_from_output(&bad, true).is_err());
}
