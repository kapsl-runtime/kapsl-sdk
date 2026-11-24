//! Tensor-core attention kernels written with CuTe (CUTLASS).
//!
//! Unlike the NVRTC kernels in `kernels/*.cu`, these are compiled to PTX by
//! `build.rs` with nvcc and embedded in the binary: CuTe is a header-only C++
//! library, so compiling at run time would need CUTLASS on every user's
//! machine and seconds of compile time on first use.
//!
//! The PTX targets sm_80. Launchers return `Ok(false)` without launching when
//! the device or the head dim is not supported, so the caller can fall back to
//! the NVRTC kernel.

use crate::attention::cuda::PrefillAttnParams;
use cudarc::driver::sys::CUdevice_attribute;
use cudarc::driver::{CudaDevice, CudaFunction, LaunchAsync, LaunchConfig};
use cudarc::nvrtc::Ptx;
use half::f16;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, OnceLock};

const MODULE: &str = "kapsl_cute_prefill_attn";
static PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/prefill_attention.ptx"));

/// One entry point per head dim; keep in sync with the
/// `KAPSL_PREFILL_ATTENTION_ENTRY` list in `kernels/cute/prefill_attention.cu`.
const HEAD_DIMS: &[u32] = &[32, 64, 96, 128];
const KERNELS: &[&str] = &[
    "prefill_attention_cute_d32",
    "prefill_attention_cute_d64",
    "prefill_attention_cute_d96",
    "prefill_attention_cute_d128",
];

// Must match `PrefillTraits` in the kernel source.
const BLOCK_M: u32 = 64;
const BLOCK_N: u32 = 64;
const THREADS: u32 = 128;

/// Oldest compute capability with the tensor-core MMA the kernels use.
const MIN_COMPUTE_MAJOR: i32 = 8;

/// Set when the precompiled PTX fails to load, so we stop retrying.
static LOAD_FAILED: AtomicBool = AtomicBool::new(false);

/// `KAPSL_DISABLE_CUTE=1` forces the NVRTC kernels, e.g. to compare the two.
fn disabled_by_env() -> bool {
    static DISABLED: OnceLock<bool> = OnceLock::new();
    *DISABLED.get_or_init(|| {
        std::env::var("KAPSL_DISABLE_CUTE").is_ok_and(|v| !v.is_empty() && v != "0")
    })
}

/// The kernel for `head_dim` if this device can run it.
fn kernel_for(device: &Arc<CudaDevice>, head_dim: u32) -> Option<&'static str> {
    if LOAD_FAILED.load(Ordering::Relaxed) || disabled_by_env() {
        return None;
    }
    let kernel = KERNELS[HEAD_DIMS.iter().position(|&d| d == head_dim)?];
    let major = device
        .attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)
        .ok()?;
    (major >= MIN_COMPUTE_MAJOR).then_some(kernel)
}

fn get_func(device: &Arc<CudaDevice>, kernel: &str) -> Result<CudaFunction, String> {
    if let Some(func) = device.get_func(MODULE, kernel) {
        return Ok(func);
    }
    device
        .load_ptx(Ptx::from_src(PTX), MODULE, KERNELS)
        .map_err(|e| format!("PTX load of {MODULE} failed: {e}"))?;
    device
        .get_func(MODULE, kernel)
        .ok_or_else(|| format!("kernel {kernel} not found in module {MODULE}"))
}

/// Causal prefill attention on tensor cores.
///
/// Returns `Ok(true)` if the kernel was launched and `Ok(false)` if this
/// device or head dim is not supported, in which case nothing was written.
pub fn launch_prefill_attention(
    device: &Arc<CudaDevice>,
    p: &mut PrefillAttnParams<'_>,
) -> Result<bool, String> {
    if p.seq_len == 0 {
        return Ok(false);
    }
    let Some(kernel) = kernel_for(device, p.head_dim) else {
        return Ok(false);
    };
    let func = match get_func(device, kernel) {
        Ok(func) => func,
        Err(e) => {
            // For example a driver older than the toolkit that built the PTX.
            LOAD_FAILED.store(true, Ordering::Relaxed);
            log::warn!("CuTe kernels disabled, using the NVRTC kernels instead: {e}");
            return Ok(false);
        }
    };

    let cfg = LaunchConfig {
        grid_dim: (p.seq_len.div_ceil(BLOCK_M), p.num_q_heads, 1),
        block_dim: (THREADS, 1, 1),
        // One K tile and one V tile, whatever the sequence length.
        shared_mem_bytes: 2 * BLOCK_N * p.head_dim * std::mem::size_of::<f16>() as u32,
    };
    unsafe {
        func.launch(
            cfg,
            (
                &mut *p.out,
                p.q,
                p.k,
                p.v,
                p.scale,
                p.seq_len as i32,
                p.num_q_heads as i32,
                p.num_kv_heads as i32,
            ),
        )
        .map_err(|e| format!("prefill_attn (cute) launch: {e}"))?;
    }
    Ok(true)
}
