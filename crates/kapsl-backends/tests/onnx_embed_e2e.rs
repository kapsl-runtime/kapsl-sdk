//! End-to-end test for the ONNX embedding backend: it runs a *real* ONNX
//! Runtime session (a tiny Identity encoder fixture whose output equals its
//! input) through `OnnxEmbedBackend`, so the assertions cover the actual
//! session-run -> masked-mean-pool -> output wiring, not just the pooling math
//! on synthetic packets (which the unit tests already cover).
//!
//! The fixture is `tests/fixtures/identity_embed.onnx` (Identity:
//! `hidden[batch,seq,2] -> last_hidden_state[batch,seq,2]`), generated with the
//! `onnx` Python helper.

use kapsl_backends::onnx::ExecutionProvider;
use kapsl_backends::{OnnxBackend, OnnxEmbedBackend};
use kapsl_engine_api::{BinaryTensorPacket, Engine, InferenceRequest, NamedTensor, TensorDtype};
use std::path::PathBuf;

fn f32_packet(shape: Vec<i64>, vals: &[f32]) -> BinaryTensorPacket {
    let mut data = Vec::new();
    for v in vals {
        data.extend_from_slice(&v.to_le_bytes());
    }
    BinaryTensorPacket {
        shape,
        dtype: TensorDtype::Float32,
        data,
    }
}

fn i64_packet(shape: Vec<i64>, vals: &[i64]) -> BinaryTensorPacket {
    let mut data = Vec::new();
    for v in vals {
        data.extend_from_slice(&v.to_le_bytes());
    }
    BinaryTensorPacket {
        shape,
        dtype: TensorDtype::Int64,
        data,
    }
}

fn to_f32(p: &BinaryTensorPacket) -> Vec<f32> {
    p.data
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect()
}

fn mask(values: &[i64]) -> Vec<NamedTensor> {
    vec![NamedTensor {
        name: "attention_mask".to_string(),
        tensor: i64_packet(vec![1, values.len() as i64], values),
    }]
}

#[tokio::test]
async fn onnx_embed_end_to_end_pools_real_session_output() {
    let model =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/identity_embed.onnx");

    let mut inner = OnnxBackend::builder()
        .with_provider(ExecutionProvider::CPU)
        .build();
    inner.load(&model).await.expect("load identity fixture");
    let backend = OnnxEmbedBackend::new(Box::new(inner), false);

    // hidden [1,2,2] = rows [1,2] and [3,4].
    let mut req = InferenceRequest::new(f32_packet(vec![1, 2, 2], &[1.0, 2.0, 3.0, 4.0]));

    // Both tokens active -> masked mean is [2, 3].
    req.additional_inputs = mask(&[1, 1]);
    let out = backend.infer(&req).expect("infer (both active)");
    assert_eq!(out.shape, vec![1, 2]);
    let v = to_f32(&out);
    assert!(
        (v[0] - 2.0).abs() < 1e-4 && (v[1] - 3.0).abs() < 1e-4,
        "expected [2,3], got {v:?}"
    );

    // Second token padded out -> just the first row [1, 2].
    req.additional_inputs = mask(&[1, 0]);
    let out = backend.infer(&req).expect("infer (masked)");
    let v = to_f32(&out);
    assert!(
        (v[0] - 1.0).abs() < 1e-4 && (v[1] - 2.0).abs() < 1e-4,
        "expected [1,2], got {v:?}"
    );
}

#[tokio::test]
async fn onnx_embed_end_to_end_normalizes_to_unit_norm() {
    let model =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/identity_embed.onnx");

    let mut inner = OnnxBackend::builder()
        .with_provider(ExecutionProvider::CPU)
        .build();
    inner.load(&model).await.expect("load identity fixture");
    let backend = OnnxEmbedBackend::new(Box::new(inner), true);

    // Single active token [3,4] -> normalized to [0.6, 0.8] (norm 5).
    let mut req = InferenceRequest::new(f32_packet(vec![1, 1, 2], &[3.0, 4.0]));
    req.additional_inputs = mask(&[1]);
    let out = backend.infer(&req).expect("infer");
    let v = to_f32(&out);
    let norm = (v[0] * v[0] + v[1] * v[1]).sqrt();
    assert!((norm - 1.0).abs() < 1e-4, "expected unit norm, got {norm}");
    assert!(
        (v[0] - 0.6).abs() < 1e-4 && (v[1] - 0.8).abs() < 1e-4,
        "expected [0.6,0.8], got {v:?}"
    );
}
