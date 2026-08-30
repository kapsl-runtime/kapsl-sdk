//! Tests for ONNX classification post-processing.

use super::*;
use async_trait::async_trait;
use kapsl_engine_api::{
    BatchingMode, BatchingPolicy, Engine, EngineMetrics, EngineStream, InferenceRequest,
};
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

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

struct BatchCapableInner {
    single_calls: Arc<AtomicUsize>,
    batch_calls: Arc<AtomicUsize>,
}

#[async_trait]
impl Engine for BatchCapableInner {
    async fn load(&mut self, _model_path: &Path) -> Result<(), EngineError> {
        Ok(())
    }

    fn infer(&self, _request: &InferenceRequest) -> Result<BinaryTensorPacket, EngineError> {
        self.single_calls.fetch_add(1, Ordering::Relaxed);
        Ok(f32_packet(vec![1, 2], vec![0.25, 0.75]))
    }

    fn infer_batch(
        &self,
        requests: &[InferenceRequest],
    ) -> Result<Vec<BinaryTensorPacket>, EngineError> {
        self.batch_calls.fetch_add(1, Ordering::Relaxed);
        Ok(requests
            .iter()
            .map(|_| f32_packet(vec![1, 2], vec![0.25, 0.75]))
            .collect())
    }

    fn max_batch(&self) -> usize {
        8
    }

    fn batching_policy(&self) -> BatchingPolicy {
        BatchingPolicy::request_coalescing(8).with_queue_delay_ms(3)
    }

    fn infer_stream(&self, request: &InferenceRequest) -> EngineStream {
        let result = self.infer(request);
        Box::pin(futures::stream::once(async move { result }))
    }

    fn unload(&mut self) {}

    fn metrics(&self) -> EngineMetrics {
        EngineMetrics::new()
    }

    fn health_check(&self) -> Result<(), EngineError> {
        Ok(())
    }
}

#[test]
fn classify_wrapper_preserves_inner_batching_policy_and_uses_infer_batch() {
    let single_calls = Arc::new(AtomicUsize::new(0));
    let batch_calls = Arc::new(AtomicUsize::new(0));
    let backend = OnnxClassifyBackend::new(
        Box::new(BatchCapableInner {
            single_calls: single_calls.clone(),
            batch_calls: batch_calls.clone(),
        }),
        false,
    );

    assert_eq!(backend.max_batch(), 8);
    let policy = backend.batching_policy();
    assert_eq!(policy.mode, BatchingMode::RequestCoalescing);
    assert_eq!(policy.max_requests, 8);
    assert_eq!(policy.queue_delay_ms, Some(3));

    let request = InferenceRequest::new(BinaryTensorPacket {
        shape: vec![1, 2],
        dtype: TensorDtype::Float32,
        data: vec![0; 8],
    });
    let outputs = backend
        .infer_batch(&[request.clone(), request])
        .expect("batched classification should succeed");

    assert_eq!(outputs.len(), 2);
    assert_eq!(batch_calls.load(Ordering::Relaxed), 1);
    assert_eq!(single_calls.load(Ordering::Relaxed), 0);
    for output in outputs {
        approx(&bytes_to_f32(&output.data), &[0.25, 0.75]);
    }
}
