//! LLM inference engines, scheduling, KV-cache management, and prompt tooling.

#[cfg(all(feature = "onnx-directml", not(target_os = "windows")))]
compile_error!("the `onnx-directml` feature is supported only on Windows");

#[cfg(feature = "onnx")]
#[path = "engine/allocation_scope.rs"]
pub mod allocation_scope;
#[path = "cache/block_manager.rs"]
pub mod block_manager;
#[cfg(feature = "onnx")]
#[path = "engine/onnx.rs"]
pub mod engine;
#[path = "gguf/backend.rs"]
pub mod gguf_backend;
#[path = "scheduling/global_scheduler.rs"]
pub mod global_scheduler;
#[path = "cache/kv_cache.rs"]
pub mod kv_cache;
#[cfg(feature = "onnx")]
#[path = "engine/backend.rs"]
pub mod llm_backend;
#[path = "engine/metrics.rs"]
pub mod llm_metrics;
#[path = "model/model_paths.rs"]
pub mod model_paths;
#[path = "model/prompt_adapter.rs"]
pub mod prompt_adapter;
#[path = "cache/radix_tree.rs"]
pub mod radix_tree;
#[path = "retrieval/rag.rs"]
pub mod rag;
#[path = "scheduling/scheduler.rs"]
pub mod scheduler;
#[path = "scheduling/sequence.rs"]
pub mod sequence;

pub use gguf_backend::GgufBackend;
