//! NVRTC compilation and module loading for the runtime-compiled CUDA
//! kernels in `cuda/kernels/*.cu`.

use cudarc::driver::{CudaDevice, CudaFunction};
use cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions};
use std::sync::Arc;

/// NVRTC options pointing at the local CUDA headers.
///
/// Honours `CUDA_PATH` then `CUDA_HOME`, falling back to the default
/// `/usr/local/cuda` install prefix.
fn cuda_compile_opts() -> CompileOptions {
    let cuda_include = std::env::var("CUDA_PATH")
        .or_else(|_| std::env::var("CUDA_HOME"))
        .map(|p| format!("{p}/include"))
        .unwrap_or_else(|_| "/usr/local/cuda/include".to_string());
    CompileOptions {
        include_paths: vec![cuda_include],
        ..Default::default()
    }
}

/// One `.cu` source file, compiled by NVRTC into a single CUDA module.
pub(crate) struct KernelModule {
    /// Name the loaded module is cached under on each `CudaDevice`.
    pub name: &'static str,
    /// CUDA C++ source, `include_str!`-ed from `cuda/kernels/`.
    pub src: &'static str,
    /// Every `extern "C"` kernel that `src` exports.
    pub kernels: &'static [&'static str],
}

impl KernelModule {
    /// Returns `kernel` from this module on `device`, compiling and loading
    /// the module on first use. Later calls hit cudarc's per-device cache.
    pub fn get_func(&self, device: &Arc<CudaDevice>, kernel: &str) -> Result<CudaFunction, String> {
        if let Some(func) = device.get_func(self.name, kernel) {
            return Ok(func);
        }
        let ptx = compile_ptx_with_opts(self.src, cuda_compile_opts())
            .map_err(|e| format!("NVRTC compile of {} failed: {e}", self.name))?;
        device
            .load_ptx(ptx, self.name, self.kernels)
            .map_err(|e| format!("PTX load of {} failed: {e}", self.name))?;
        device
            .get_func(self.name, kernel)
            .ok_or_else(|| format!("kernel {kernel} not found in module {}", self.name))
    }
}
