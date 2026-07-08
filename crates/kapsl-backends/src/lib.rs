pub mod engine_pool;
pub mod factory;
#[cfg(feature = "gguf-native")]
pub mod gguf_native;
#[cfg(feature = "native")]
pub mod native;
pub mod onnx;
pub mod onnx_classify;
pub mod onnx_detect;
pub mod onnx_embed;
pub mod onnx_transcribe;
#[cfg(feature = "onnx-cuda-pool")]
pub mod ort_pool_allocator;
pub mod preprocess;
mod provider_compat;

pub use engine_pool::{EnginePool, EnginePoolConfig};
pub use factory::{BackendFactory, OnnxRuntimeTuning};
#[cfg(feature = "gguf-native")]
pub use gguf_native::GgufNativeBackend;
#[cfg(feature = "native")]
pub use native::NativeBackend;
pub use onnx::OnnxBackend;
pub use onnx_classify::OnnxClassifyBackend;
pub use onnx_detect::OnnxDetectBackend;
pub use onnx_embed::OnnxEmbedBackend;
pub use onnx_transcribe::OnnxTranscribeBackend;
pub use preprocess::{PreprocessBackend, Preprocessor, VisionPreprocessor};
