//! Quantized kernel implementations.

#[cfg(feature = "cuda")]
#[path = "../cuda/quant.rs"]
pub mod cuda;
