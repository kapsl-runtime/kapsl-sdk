//! Real-GPU regression for kapsl-engine#250. On CUDA, every distinct input
//! length used to build another ORT session, each holding its own device copy
//! of the weights, until the fixed device pool ran out. The CUDA provider must
//! now serve every length from the one prewarmed session pool.
//!
//! Needs an NVIDIA GPU with the CUDA 12 and cuDNN 9 runtime libraries:
//! `cargo test -p kapsl-backends --test onnx_cuda_session_buckets -- --ignored`

use kapsl_backends::onnx::ExecutionProvider;
use kapsl_backends::OnnxBackend;
use kapsl_engine_api::{BinaryTensorPacket, Engine, InferenceRequest, TensorDtype};
use std::path::PathBuf;

fn f32_packet(shape: Vec<i64>, values: &[f32]) -> BinaryTensorPacket {
    BinaryTensorPacket {
        shape,
        dtype: TensorDtype::Float32,
        data: values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect(),
    }
}

fn to_f32(packet: &BinaryTensorPacket) -> Vec<f32> {
    packet
        .data
        .as_chunks::<4>()
        .0
        .iter()
        .map(|bytes| f32::from_le_bytes(*bytes))
        .collect()
}

#[tokio::test]
#[ignore = "needs an NVIDIA GPU and the ORT CUDA provider"]
async fn cuda_serves_every_input_length_from_one_session_pool() {
    // Identity: hidden[batch,seq,2] -> last_hidden_state[batch,seq,2].
    let model =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/identity_embed.onnx");
    let mut backend = OnnxBackend::builder()
        .with_provider(ExecutionProvider::CUDA)
        .with_max_bucket_sessions(4)
        .build();
    backend
        .load(&model)
        .await
        .expect("load identity fixture on CUDA");

    // With buckets, lengths 2, 100, 200 and 300 would each get a session
    // (seq is rounded up to 64); 60 shares the first bucket.
    for seq in [2_i64, 100, 200, 300, 60] {
        let values = (0..seq * 2).map(|value| value as f32).collect::<Vec<_>>();
        let request = InferenceRequest::new(f32_packet(vec![1, seq, 2], &values));
        let output = backend.infer(&request).expect("infer on CUDA");
        assert_eq!(output.shape, vec![1, seq, 2]);
        assert_eq!(to_f32(&output), values);
    }

    assert_eq!(backend.metrics().onnx_session_pool_total, 1);
}
