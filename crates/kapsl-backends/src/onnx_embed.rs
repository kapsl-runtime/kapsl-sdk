//! ONNX embedding backend (`EngineKind::OnnxEmbed`).
//!
//! Embeddings are the same ONNX encoder forward pass as [`crate::onnx::OnnxBackend`],
//! followed by a pooling step that collapses the per-token hidden states into one
//! vector per input. This backend wraps an inner ONNX engine and post-processes
//! its output:
//!
//! - If the model already emits a pooled vector (a 2-D `[batch, dim]` output),
//!   it is passed through unchanged (only optionally normalized) — re-pooling it
//!   would be wrong.
//! - Otherwise the 3-D `[batch, seq, dim]` hidden states are **masked-mean
//!   pooled** using the request's `attention_mask` (padding tokens excluded), the
//!   default pooling for the mainstream sentence-embedding models.
//!
//! The result is L2-normalized by default (so cosine similarity == dot product,
//! which is what vector stores expect); pass `normalize = false` to disable.

use crate::tensor_util::{bytes_to_f32, dim_usize, f32_packet};
use async_trait::async_trait;
use kapsl_engine_api::{
    BatchingPolicy, BinaryTensorPacket, Engine, EngineError, EngineMetrics, EngineModelInfo,
    EngineStream, InferenceRequest, TensorDtype,
};
use std::path::Path;

/// Wraps an inner ONNX engine and turns its raw output into a pooled embedding.
pub struct OnnxEmbedBackend {
    inner: Box<dyn Engine>,
    normalize: bool,
}

impl OnnxEmbedBackend {
    pub fn new(inner: Box<dyn Engine>, normalize: bool) -> Self {
        Self { inner, normalize }
    }
}

#[async_trait]
impl Engine for OnnxEmbedBackend {
    async fn load(&mut self, model_path: &Path) -> Result<(), EngineError> {
        self.inner.load(model_path).await
    }

    fn infer(&self, request: &InferenceRequest) -> Result<BinaryTensorPacket, EngineError> {
        let output = self.inner.infer(request)?;
        embed_from_output(&output, request, self.normalize)
    }

    fn infer_batch(
        &self,
        requests: &[InferenceRequest],
    ) -> Result<Vec<BinaryTensorPacket>, EngineError> {
        let outputs = self.inner.infer_batch(requests)?;
        if outputs.len() != requests.len() {
            return Err(EngineError::backend(format!(
                "embedding batch result length mismatch: expected {}, got {}",
                requests.len(),
                outputs.len()
            )));
        }
        outputs
            .into_iter()
            .zip(requests.iter())
            .map(|(output, request)| embed_from_output(&output, request, self.normalize))
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
        // Embeddings are not streamed; emit the single pooled result as a
        // one-item stream (mirrors OnnxBackend::infer_stream).
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

/// Turn an ONNX encoder output into a pooled embedding packet.
fn embed_from_output(
    output: &BinaryTensorPacket,
    request: &InferenceRequest,
    normalize: bool,
) -> Result<BinaryTensorPacket, EngineError> {
    if output.dtype != TensorDtype::Float32 {
        return Err(EngineError::backend(format!(
            "embedding output dtype {:?} is not supported (expected float32)",
            output.dtype
        )));
    }
    let values = bytes_to_f32(&output.data);

    match output.shape.as_slice() {
        // Already pooled: pass through (only normalize).
        [batch, dim] => {
            let (batch, dim) = (dim_usize(*batch), dim_usize(*dim));
            let mut pooled = values;
            if normalize {
                l2_normalize_rows(&mut pooled, batch, dim);
            }
            Ok(f32_packet(vec![batch as i64, dim as i64], pooled))
        }
        // Token hidden states: masked-mean pool over the sequence.
        [batch, seq, dim] => {
            let (batch, seq, dim) = (dim_usize(*batch), dim_usize(*seq), dim_usize(*dim));
            let expected = batch * seq * dim;
            if values.len() != expected {
                return Err(EngineError::backend(format!(
                    "embedding output has {} values but shape {:?} implies {}",
                    values.len(),
                    output.shape,
                    expected
                )));
            }
            let mask = extract_attention_mask(request, batch, seq);
            let mut pooled = masked_mean_pool(&values, batch, seq, dim, &mask);
            if normalize {
                l2_normalize_rows(&mut pooled, batch, dim);
            }
            Ok(f32_packet(vec![batch as i64, dim as i64], pooled))
        }
        other => Err(EngineError::backend(format!(
            "embedding expects a 2-D [batch, dim] or 3-D [batch, seq, dim] output, got shape {:?}",
            other
        ))),
    }
}

/// Masked mean over the sequence axis. `mask` is `batch * seq` of 0/1 weights;
/// padding tokens (weight 0) are excluded. Returns `batch * dim`.
fn masked_mean_pool(
    hidden: &[f32],
    batch: usize,
    seq: usize,
    dim: usize,
    mask: &[f32],
) -> Vec<f32> {
    let mut out = vec![0f32; batch * dim];
    for b in 0..batch {
        let mut denom = 0f32;
        for s in 0..seq {
            let w = mask.get(b * seq + s).copied().unwrap_or(1.0);
            if w == 0.0 {
                continue;
            }
            denom += w;
            let hbase = (b * seq + s) * dim;
            let obase = b * dim;
            for d in 0..dim {
                out[obase + d] += w * hidden[hbase + d];
            }
        }
        let denom = denom.max(1e-9);
        let obase = b * dim;
        for d in 0..dim {
            out[obase + d] /= denom;
        }
    }
    out
}

/// L2-normalize each `dim`-length row in place.
fn l2_normalize_rows(v: &mut [f32], rows: usize, dim: usize) {
    for r in 0..rows {
        let base = r * dim;
        let row = &mut v[base..base + dim];
        let norm = row.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
        for x in row.iter_mut() {
            *x /= norm;
        }
    }
}

/// Pull the `attention_mask` from the request's additional inputs as `batch*seq`
/// weights. Missing or wrong-sized masks fall back to all-ones (plain mean),
/// matching OnnxBackend's behavior when no mask is supplied.
fn extract_attention_mask(request: &InferenceRequest, batch: usize, seq: usize) -> Vec<f32> {
    let expected = batch * seq;
    for named in &request.additional_inputs {
        if named.name.contains("attention_mask") {
            let mask = packet_to_f32(&named.tensor);
            if mask.len() == expected {
                return mask;
            }
        }
    }
    vec![1.0; expected]
}

/// Interpret a tensor packet as f32 weights, accepting the integer/float dtypes
/// an attention mask is commonly stored in.
fn packet_to_f32(packet: &BinaryTensorPacket) -> Vec<f32> {
    match packet.dtype {
        TensorDtype::Float32 => bytes_to_f32(&packet.data),
        TensorDtype::Int64 => packet
            .data
            .chunks_exact(8)
            .map(|b| i64::from_le_bytes(b.try_into().unwrap()) as f32)
            .collect(),
        TensorDtype::Int32 => packet
            .data
            .chunks_exact(4)
            .map(|b| i32::from_le_bytes(b.try_into().unwrap()) as f32)
            .collect(),
        _ => Vec::new(),
    }
}

#[cfg(test)]
#[path = "onnx_embed_tests.rs"]
mod onnx_embed_tests;
