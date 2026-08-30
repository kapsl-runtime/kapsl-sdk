//! CPU kernel-backend factory.

use crate::attention::CpuAttention;
use crate::mlp::CpuMlp;
use kapsl_hal::kernel::{AttentionKernel, KernelBackend, MlpKernel};

/// Trait-based backend composed from the portable CPU reference kernels.
#[derive(Debug)]
pub struct CpuBackend;

impl KernelBackend for CpuBackend {
    fn attention(&self) -> Box<dyn AttentionKernel> {
        Box::new(CpuAttention)
    }

    fn mlp(&self) -> Box<dyn MlpKernel> {
        Box::new(CpuMlp)
    }
}

/// Creates the portable CPU backend.
///
/// Optimized CUDA kernels use the explicit launch APIs exported through
/// `crate::cuda_kernels` rather than this trait-based factory.
pub fn create_backend() -> Box<dyn KernelBackend> {
    log::info!("Creating CPU reference kernel backend");
    Box::new(CpuBackend)
}
