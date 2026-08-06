//! ONNX CTC speech-recognition backend (`EngineKind::OnnxTranscribe`).
//!
//! Targets the CTC family of ASR models (Wav2Vec2, Conformer-CTC, …): a single
//! forward pass over acoustic features produces per-frame class logits
//! `[batch, time, vocab]`, which this backend collapses into a token sequence
//! with greedy CTC decoding (argmax per frame, merge consecutive duplicate
//! labels, drop the blank token).
//!
//! Output is `[num_tokens]` `Int32` token ids. Turning ids into text needs the
//! model's vocabulary and is the caller's job — the engine stays tensors-in /
//! tokens-out, mirroring the generative backends.
//!
//! Encoder-decoder ASR (Whisper) is intentionally out of scope: it is an
//! autoregressive decode loop, closer to the generative backends than to a
//! stateless post-processor.
//!
//! Configured from the manifest's `metadata.transcribe` block (all optional):
//!
//! ```yaml
//! metadata:
//!   transcribe:
//!     blank_id: 0            # CTC blank index (default 0; some vocabs use last)
//!     collapse_repeats: true # standard CTC repeat merge (default true)
//! ```

use crate::tensor_util::{bytes_to_f32, dim_usize, i32_packet};
use async_trait::async_trait;
use kapsl_engine_api::{
    BatchingPolicy, BinaryTensorPacket, Engine, EngineError, EngineMetrics, EngineModelInfo,
    EngineStream, InferenceRequest, TensorDtype,
};
use serde::Deserialize;
use std::path::Path;

fn default_collapse_repeats() -> bool {
    true
}

/// Parsed `metadata.transcribe` block. All fields default, so the block is
/// optional for a plain blank-0 CTC model.
#[derive(Debug, Clone, Deserialize)]
pub struct TranscribeConfig {
    /// Index of the CTC blank symbol in the vocabulary. Default `0`.
    #[serde(default)]
    pub blank_id: usize,
    /// Merge consecutive identical frame labels before dropping blanks (standard
    /// CTC). Disable to emit the raw per-frame argmax minus blanks.
    #[serde(default = "default_collapse_repeats")]
    pub collapse_repeats: bool,
}

impl Default for TranscribeConfig {
    fn default() -> Self {
        Self {
            blank_id: 0,
            collapse_repeats: default_collapse_repeats(),
        }
    }
}

/// Wraps an inner ONNX engine and greedily CTC-decodes its logits.
pub struct OnnxTranscribeBackend {
    inner: Box<dyn Engine>,
    cfg: TranscribeConfig,
}

impl OnnxTranscribeBackend {
    pub fn new(inner: Box<dyn Engine>, cfg: TranscribeConfig) -> Self {
        Self { inner, cfg }
    }
}

#[async_trait]
impl Engine for OnnxTranscribeBackend {
    async fn load(&mut self, model_path: &Path) -> Result<(), EngineError> {
        self.inner.load(model_path).await
    }

    fn infer(&self, request: &InferenceRequest) -> Result<BinaryTensorPacket, EngineError> {
        let output = self.inner.infer(request)?;
        transcribe_from_output(&output, &self.cfg)
    }

    fn infer_batch(
        &self,
        requests: &[InferenceRequest],
    ) -> Result<Vec<BinaryTensorPacket>, EngineError> {
        let outputs = self.inner.infer_batch(requests)?;
        if outputs.len() != requests.len() {
            return Err(EngineError::backend(format!(
                "transcription batch result length mismatch: expected {}, got {}",
                requests.len(),
                outputs.len()
            )));
        }
        outputs
            .into_iter()
            .map(|output| transcribe_from_output(&output, &self.cfg))
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

/// Greedy CTC decode of a `[time, vocab]` logit grid into token ids.
fn transcribe_from_output(
    output: &BinaryTensorPacket,
    cfg: &TranscribeConfig,
) -> Result<BinaryTensorPacket, EngineError> {
    if output.dtype != TensorDtype::Float32 {
        return Err(EngineError::backend(format!(
            "CTC output dtype {:?} is not supported (expected float32)",
            output.dtype
        )));
    }
    let values = bytes_to_f32(&output.data);

    // Accept [time, vocab] or [1, time, vocab].
    let (time, vocab) = match output.shape.as_slice() {
        [t, v] => (dim_usize(*t), dim_usize(*v)),
        [1, t, v] => (dim_usize(*t), dim_usize(*v)),
        other => {
            return Err(EngineError::backend(format!(
                "CTC decoder expects a 2-D [time, vocab] or 3-D [1, time, vocab] output, \
                 got shape {:?}",
                other
            )));
        }
    };
    if vocab == 0 {
        return Err(EngineError::backend("CTC output has zero-width vocab"));
    }
    if cfg.blank_id >= vocab {
        return Err(EngineError::backend(format!(
            "CTC blank_id {} is out of range for vocab size {}",
            cfg.blank_id, vocab
        )));
    }
    if values.len() != time * vocab {
        return Err(EngineError::backend(format!(
            "CTC output has {} values but shape implies {}x{}={}",
            values.len(),
            time,
            vocab,
            time * vocab
        )));
    }

    let mut tokens: Vec<i32> = Vec::new();
    let mut prev: Option<usize> = None;
    for t in 0..time {
        let row = &values[t * vocab..t * vocab + vocab];
        let arg = argmax(row);

        // Merge consecutive identical frame labels (blank included, so a repeat
        // separated by a blank frame still yields two tokens).
        if cfg.collapse_repeats && prev == Some(arg) {
            continue;
        }
        prev = Some(arg);

        if arg != cfg.blank_id {
            tokens.push(arg as i32);
        }
    }

    Ok(i32_packet(vec![tokens.len() as i64], tokens))
}

/// Index of the maximum element (first on ties). `row` is non-empty by caller.
fn argmax(row: &[f32]) -> usize {
    let mut best = 0usize;
    let mut best_v = row[0];
    for (i, &v) in row.iter().enumerate().skip(1) {
        if v > best_v {
            best_v = v;
            best = i;
        }
    }
    best
}

#[cfg(test)]
#[path = "onnx_transcribe_tests.rs"]
mod onnx_transcribe_tests;
