pub mod engine_pool;
#[path = "onnx/env.rs"]
mod env_util;
pub mod factory;
#[cfg(feature = "gguf-native")]
#[path = "native/gguf.rs"]
pub mod gguf_native;
#[cfg(feature = "native")]
pub mod native;
pub mod onnx;
#[path = "onnx/classify.rs"]
pub mod onnx_classify;
#[path = "onnx/detect.rs"]
pub mod onnx_detect;
#[path = "onnx/embed.rs"]
pub mod onnx_embed;
#[path = "onnx/transcribe.rs"]
pub mod onnx_transcribe;
#[cfg(feature = "onnx-cuda-pool")]
#[path = "onnx/pool_allocator.rs"]
pub mod ort_pool_allocator;
pub mod preprocess;
#[path = "onnx/provider_compat.rs"]
mod provider_compat;
#[path = "onnx/tensor_util.rs"]
mod tensor_util;

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
pub use preprocess::{AudioPreprocessor, PreprocessBackend, Preprocessor, VisionPreprocessor};
