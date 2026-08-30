//! Shared NVRTC compilation settings for the runtime-compiled CUDA kernels.

use cudarc::nvrtc::CompileOptions;

/// NVRTC options pointing at the local CUDA headers.
///
/// Honours `CUDA_PATH` then `CUDA_HOME`, falling back to the default
/// `/usr/local/cuda` install prefix.
pub(crate) fn cuda_compile_opts() -> CompileOptions {
    let cuda_include = std::env::var("CUDA_PATH")
        .or_else(|_| std::env::var("CUDA_HOME"))
        .map(|p| format!("{p}/include"))
        .unwrap_or_else(|_| "/usr/local/cuda/include".to_string());
    CompileOptions {
        include_paths: vec![cuda_include],
        ..Default::default()
    }
}
