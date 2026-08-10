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
//! `Uint8` packet of raw encoded file bytes) into the tensor the model expects
//! and may add named inputs derived from that tensor, then delegates to the
//! inner engine. Everything else — batching capability, metrics, model info,
//! hot-swap — passes straight through, so a preprocessed ONNX classifier still composes with
//! [`crate::factory::maybe_wrap_onnx_postprocess`]: the pipeline becomes
//! `preprocess -> onnx forward -> postprocess`.

pub mod audio;
pub mod vision;

#[cfg(test)]
mod audio_tests;
#[cfg(test)]
mod preprocess_tests;
#[cfg(test)]
mod vision_tests;

use async_trait::async_trait;
use kapsl_engine_api::{
    BatchingPolicy, BinaryTensorPacket, Engine, EngineError, EngineMetrics, EngineModelInfo,
    EngineStream, InferenceRequest, NamedTensor,
};
use std::path::Path;

pub use audio::AudioPreprocessor;
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

    /// Derive named model inputs from the tensor returned by [`Self::apply`].
    ///
    /// Most preprocessors only replace the primary input. Audio front-ends can
    /// additionally use this hook to emit the feature-frame count expected by
    /// multi-input acoustic models, without making clients duplicate framing
    /// arithmetic. A derived tensor takes precedence over a client-supplied
    /// tensor with the same name.
    fn derived_inputs(
        &self,
        _output: &BinaryTensorPacket,
    ) -> Result<Vec<NamedTensor>, EngineError> {
        Ok(Vec::new())
    }

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
    /// preprocessed tensor. Additional inputs are preserved unless a
    /// preprocessor derives a fresh tensor with the same name; session id,
    /// metadata, and the cancellation token are always preserved.
    fn transform(&self, request: &InferenceRequest) -> Result<InferenceRequest, EngineError> {
        let input = self.pre.apply(&request.input)?;
        let derived_inputs = self.pre.derived_inputs(&input)?;
        let mut transformed = request.clone();
        transformed.input = input;
        for derived in derived_inputs {
            // The manifest owns derived inputs. Remove any stale value a legacy
            // client supplied, then append the value computed from the actual
            // emitted tensor.
            transformed
                .additional_inputs
                .retain(|candidate| candidate.name != derived.name);
            transformed.additional_inputs.push(derived);
        }
        Ok(transformed)
    }
}

#[async_trait]
impl Engine for PreprocessBackend {
    fn planned_external_device_memory(
        &self,
        model_path: &Path,
    ) -> Result<kapsl_engine_api::ExternalDeviceMemoryReport, EngineError> {
        self.inner.planned_external_device_memory(model_path)
    }

    async fn load(&mut self, model_path: &Path) -> Result<(), EngineError> {
        self.inner.load(model_path).await
    }

    fn actual_external_device_memory(&self) -> kapsl_engine_api::ExternalDeviceMemoryReport {
        self.inner.actual_external_device_memory()
    }

    fn infer(&self, request: &InferenceRequest) -> Result<BinaryTensorPacket, EngineError> {
        let transformed = self.transform(request)?;
        self.inner.infer(&transformed)
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

    fn infer_stream(&self, request: &InferenceRequest) -> EngineStream {
        match self.transform(request) {
            Ok(transformed) => self.inner.infer_stream(&transformed),
            Err(e) => Box::pin(futures::stream::once(async move { Err(e) })),
        }
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
