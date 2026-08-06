//! ONNX classification backend (`EngineKind::OnnxClassify`).
//!
//! Classification is the same ONNX forward pass as [`crate::onnx::OnnxBackend`]
//! producing per-class logits `[batch, num_classes]`, followed by a softmax so
//! callers get a probability distribution they can `argmax`. This backend wraps
//! an inner ONNX engine and post-processes its output.
//!
//! Most classifier exports emit raw logits, so softmax is applied by default.
//! Models that already apply softmax in-graph set
//! `metadata.classify.apply_softmax = false` to pass the output through.

use crate::tensor_util::{bytes_to_f32, dim_usize, f32_packet};
use async_trait::async_trait;
use kapsl_engine_api::{
    BatchingPolicy, BinaryTensorPacket, Engine, EngineError, EngineMetrics, EngineModelInfo,
    EngineStream, InferenceRequest, TensorDtype,
};
use std::path::Path;

/// Wraps an inner ONNX engine and turns per-class logits into probabilities.
pub struct OnnxClassifyBackend {
    inner: Box<dyn Engine>,
    apply_softmax: bool,
}

impl OnnxClassifyBackend {
    pub fn new(inner: Box<dyn Engine>, apply_softmax: bool) -> Self {
        Self {
            inner,
            apply_softmax,
        }
    }
}

#[async_trait]
impl Engine for OnnxClassifyBackend {
    async fn load(&mut self, model_path: &Path) -> Result<(), EngineError> {
        self.inner.load(model_path).await
    }

    fn infer(&self, request: &InferenceRequest) -> Result<BinaryTensorPacket, EngineError> {
        let output = self.inner.infer(request)?;
        classify_from_output(&output, self.apply_softmax)
    }

    fn infer_batch(
        &self,
        requests: &[InferenceRequest],
    ) -> Result<Vec<BinaryTensorPacket>, EngineError> {
        let outputs = self.inner.infer_batch(requests)?;
        outputs
            .into_iter()
            .map(|output| classify_from_output(&output, self.apply_softmax))
            .collect()
    }

    fn max_batch(&self) -> usize {
        self.inner.max_batch()
    }

    fn self_batches(&self) -> bool {
        self.inner.self_batches()
    }

    fn batching_policy(&self) -> BatchingPolicy {
        self.inner.batching_policy()
    }

    fn infer_stream(&self, request: &InferenceRequest) -> EngineStream {
        let result = self.infer(request);
        Box::pin(futures::stream::once(async move { result }))
    }

    fn unload(&mut self) {
        self.inner.unload();
    }

    fn metrics(&self) -> EngineMetrics {
        self.inner.metrics()
    }

    fn model_info(&self) -> Option<EngineModelInfo> {
        self.inner.model_info()
    }

    fn health_check(&self) -> Result<(), EngineError> {
        self.inner.health_check()
    }
}

/// Turn an ONNX classifier output into per-class probabilities.
fn classify_from_output(
    output: &BinaryTensorPacket,
    apply_softmax: bool,
) -> Result<BinaryTensorPacket, EngineError> {
    if output.dtype != TensorDtype::Float32 {
        return Err(EngineError::backend(format!(
            "classifier output dtype {:?} is not supported (expected float32)",
            output.dtype
        )));
    }
    let mut values = bytes_to_f32(&output.data);

    // Normalize to [batch, classes]: a 1-D [classes] output is a single row.
    let (batch, classes, shape) = match output.shape.as_slice() {
        [classes] => (1usize, dim_usize(*classes), vec![1, *classes]),
        [batch, classes] => (dim_usize(*batch), dim_usize(*classes), output.shape.clone()),
        other => {
            return Err(EngineError::backend(format!(
                "classifier expects a 1-D [classes] or 2-D [batch, classes] output, got shape {:?}",
                other
            )));
        }
    };

    let expected = batch * classes;
    if values.len() != expected {
        return Err(EngineError::backend(format!(
            "classifier output has {} values but shape {:?} implies {}",
            values.len(),
            output.shape,
            expected
        )));
    }

    if apply_softmax {
        softmax_rows(&mut values, batch, classes);
    }
    Ok(f32_packet(shape, values))
}

/// Numerically stable softmax over each `classes`-length row, in place.
fn softmax_rows(v: &mut [f32], rows: usize, classes: usize) {
    for r in 0..rows {
        let row = &mut v[r * classes..r * classes + classes];
        let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0f32;
        for x in row.iter_mut() {
            *x = (*x - max).exp();
            sum += *x;
        }
        let sum = sum.max(1e-12);
        for x in row.iter_mut() {
            *x /= sum;
        }
    }
}

#[cfg(test)]
#[path = "onnx_classify_tests.rs"]
mod onnx_classify_tests;
