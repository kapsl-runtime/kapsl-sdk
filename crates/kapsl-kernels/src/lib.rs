//! CPU reference implementations and optional CUDA kernels for Kapsl inference.

#[path = "cpu/attention.rs"]
pub mod attention;
#[path = "cpu/backend.rs"]
pub mod backend;
#[path = "cpu/mlp.rs"]
pub mod mlp;
#[cfg(feature = "cuda")]
#[path = "cuda/compiler.rs"]
mod nvrtc_util;
pub mod quant;

pub use backend::{create_backend, CpuBackend};

#[cfg(feature = "cuda")]
pub use attention::cuda as cuda_kernels;
#[cfg(feature = "cuda")]
pub use quant::cuda as cuda_quant_kernels;
