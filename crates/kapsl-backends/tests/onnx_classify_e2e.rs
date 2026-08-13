//! End-to-end test for the ONNX classification backend: runs a tiny Identity
//! logits fixture (`logits_in[batch,3] -> logits[batch,3]`) through
//! `OnnxClassifyBackend`, exercising the real session-run -> softmax wiring
//! against exact expected probabilities (the unit tests cover softmax on
//! synthetic packets).

use kapsl_backends::onnx::ExecutionProvider;
use kapsl_backends::{OnnxBackend, OnnxClassifyBackend};
use kapsl_engine_api::{BinaryTensorPacket, Engine, InferenceRequest, TensorDtype};
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

fn to_f32(p: &BinaryTensorPacket) -> Vec<f32> {
    p.data
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

#[tokio::test]
async fn onnx_classify_end_to_end_softmaxes_real_session_output() {
    let model =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/identity_logits.onnx");

    let mut inner = OnnxBackend::builder()
        .with_provider(ExecutionProvider::CPU)
        .build();
    inner.load(&model).await.expect("load logits fixture");
    let backend = OnnxClassifyBackend::new(Box::new(inner), true);

    // logits [1, 3] = [1, 2, 3] -> softmax.
    let req = InferenceRequest::new(f32_packet(vec![1, 3], &[1.0, 2.0, 3.0]));
    let out = backend.infer(&req).expect("infer");
    assert_eq!(out.shape, vec![1, 3]);

    let p = to_f32(&out);
    // softmax([1,2,3]) = [0.09003, 0.24473, 0.66524].
    let expect = [0.09003, 0.24473, 0.66524];
    for (got, want) in p.iter().zip(expect) {
        assert!((got - want).abs() < 1e-4, "got {p:?}, want {expect:?}");
    }
    assert!(
        (p.iter().sum::<f32>() - 1.0).abs() < 1e-5,
        "not a distribution: {p:?}"
    );
}
