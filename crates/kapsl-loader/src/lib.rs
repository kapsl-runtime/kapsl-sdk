//! Validated model configuration and weight loading for GGUF and safetensors.

#[path = "model/config.rs"]
pub mod config;
#[path = "formats/gguf/mod.rs"]
pub mod gguf_loader;
#[path = "formats/safetensors.rs"]
pub mod loader;
#[path = "model/weights.rs"]
pub mod weights;

pub use config::ModelConfig;
pub use gguf_loader::{load_gguf_weights, GgufError};
pub use loader::{load_safetensors, LoadError};
pub use weights::{DType, LayerWeights, ModelWeights, TensorData};
