//! Server-side input preprocessing stage for ONNX backends.
//!
//! The task cells in this crate ([`crate::onnx_embed`], [`crate::onnx_classify`])
//! are output *post*-processors: they wrap an inner ONNX engine and transform its
//! result. Text models never needed an input stage because tokenization already
//! produces a well-formed integer tensor upstream.
//!
//! Non-text models (vision, audio) are different: the model-specific, expensive
//! front-end lives on the *input* — decode a JPEG and normalize it into an
//! `f32[N,C,H,W]` tensor, or turn a waveform into a log-mel spectrogram. This
//! module adds that missing stage as a symmetric *pre*-processor wrapper.
//!
//! [`PreprocessBackend`] wraps an inner [`Engine`] together with a
//! [`Preprocessor`]. On each request it transforms `request.input` (typically a
//! `Uint8` packet of raw encoded file bytes) into the tensor the model expects,
//! then delegates to the inner engine unchanged. Everything else — batching
//! capability, metrics, model info, hot-swap — passes straight through, so a
//! preprocessed ONNX classifier still composes with
//! [`crate::factory::maybe_wrap_onnx_postprocess`]: the pipeline becomes
//! `preprocess -> onnx forward -> postprocess`.

pub mod vision;

#[cfg(test)]
mod vision_tests;

use async_trait::async_trait;
use kapsl_engine_api::{
    BinaryTensorPacket, CancellationToken, Engine, EngineError, EngineMetrics, EngineModelInfo,
    EngineStream, InferenceRequest,
};
use std::path::Path;

pub use vision::VisionPreprocessor;

/// Transforms a raw client input packet into the tensor a model expects.
///
/// Implementations are pure and stateless with respect to a request: given the
/// same input packet they must produce the same output tensor. They run on the
/// CPU before the request reaches the accelerator-bound inner engine.
pub trait Preprocessor: Send + Sync {
    /// Transform `input` into the model's input tensor. `input` carries whatever
    /// the client sent (e.g. a `Uint8` 1-D packet of encoded image bytes); the
    /// returned packet must match the model's expected dtype, shape, and layout.
    fn apply(&self, input: &BinaryTensorPacket) -> Result<BinaryTensorPacket, EngineError>;

    /// Stable label for logs/diagnostics.
    fn name(&self) -> &'static str;
}

/// Wraps an inner engine so every request's input is transformed by a
/// [`Preprocessor`] before inference.
pub struct PreprocessBackend {
    inner: Box<dyn Engine>,
    pre: Box<dyn Preprocessor>,
}

impl PreprocessBackend {
    pub fn new(inner: Box<dyn Engine>, pre: Box<dyn Preprocessor>) -> Self {
        Self { inner, pre }
    }

    /// Build a request identical to `request` but with its input replaced by the
    /// preprocessed tensor. Additional inputs, session id, metadata, and the
    /// cancellation token are preserved.
    fn transform(&self, request: &InferenceRequest) -> Result<InferenceRequest, EngineError> {
        let input = self.pre.apply(&request.input)?;
        let mut transformed = request.clone();
        transformed.input = input;
        Ok(transformed)
    }
}

#[async_trait]
impl Engine for PreprocessBackend {
    async fn load(&mut self, model_path: &Path) -> Result<(), EngineError> {
        self.inner.load(model_path).await
    }

    fn infer(&self, request: &InferenceRequest) -> Result<BinaryTensorPacket, EngineError> {
        let transformed = self.transform(request)?;
        self.inner.infer(&transformed)
    }

    async fn infer_async(
        &self,
        request: &InferenceRequest,
    ) -> Result<BinaryTensorPacket, EngineError> {
        let transformed = self.transform(request)?;
        self.inner.infer_async(&transformed).await
    }

    fn infer_batch(
        &self,
        requests: &[InferenceRequest],
    ) -> Result<Vec<BinaryTensorPacket>, EngineError> {
        let transformed = requests
            .iter()
            .map(|r| self.transform(r))
            .collect::<Result<Vec<_>, _>>()?;
        self.inner.infer_batch(&transformed)
    }

    async fn infer_batch_async(
        &self,
        requests: &[InferenceRequest],
    ) -> Result<Vec<BinaryTensorPacket>, EngineError> {
        let transformed = requests
            .iter()
            .map(|r| self.transform(r))
            .collect::<Result<Vec<_>, _>>()?;
        self.inner.infer_batch_async(&transformed).await
    }

    fn infer_stream(&self, request: &InferenceRequest) -> EngineStream {
        match self.transform(request) {
            Ok(transformed) => self.inner.infer_stream(&transformed),
            Err(e) => Box::pin(futures::stream::once(async move { Err(e) })),
        }
    }

    fn infer_with_cancellation(
        &self,
        request: &InferenceRequest,
        cancellation: &CancellationToken,
    ) -> Result<BinaryTensorPacket, EngineError> {
        if cancellation.is_cancelled() {
            return Err(EngineError::Cancelled {
                message: "Request cancelled".to_string(),
            });
        }
        let transformed = self.transform(request)?;
        self.inner.infer_with_cancellation(&transformed, cancellation)
    }

    fn max_batch(&self) -> usize {
        self.inner.max_batch()
    }

    fn self_batches(&self) -> bool {
        self.inner.self_batches()
    }

    async fn warmup(&self) -> Result<(), EngineError> {
        self.inner.warmup().await
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

    fn supports_swap(&self) -> bool {
        self.inner.supports_swap()
    }

    fn is_staged(&self) -> bool {
        self.inner.is_staged()
    }

    async fn stage(&self, path: &Path) -> Result<(), EngineError> {
        self.inner.stage(path).await
    }

    async fn swap(&self) -> Result<(), EngineError> {
        self.inner.swap().await
    }
}
