//! Fused dequantize-GEMV kernels for Q8_0 and Q4_K weight matrices.
//!
//! The CUDA source lives in `kernels/quant_gemv.cu`. Both kernels compute
//! `out[m + b*M] = Σ_k W_quant[m,k] · x[k + b*K]`, one warp per output row.

#[cfg(feature = "cuda")]
mod inner {
    use crate::nvrtc_util::KernelModule;
    use cudarc::driver::{CudaDevice, CudaSlice, LaunchAsync, LaunchConfig};
    use half::f16;
    use std::sync::Arc;

    static QUANT_GEMV: KernelModule = KernelModule {
        name: "kapsl_quant_gemv",
        src: include_str!("kernels/quant_gemv.cu"),
        kernels: &["q8_0_gemv", "q4_k_gemv"],
    };

    // ── Public launch params ──────────────────────────────────────────────────

    pub struct QuantGemvParams<'a> {
        /// Output tensor: [B * M] f16 (column-major [M, B]).
        pub out: &'a mut CudaSlice<f16>,
        /// Quantized weight matrix: raw Q8_0 or Q4_K bytes, M rows × K cols.
        pub w: &'a CudaSlice<u8>,
        /// Input tensor: [B * K] f16 (column-major [K, B]).
        pub x: &'a CudaSlice<f16>,
        pub m: u32,
        pub k: u32,
        pub b: u32,
    }

    /// Rows per thread block; must match `Q80_ROWS` / `Q4K_ROWS` in the .cu file.
    const ROWS: u32 = 4;

    pub fn launch_q8_0_gemv(
        device: &Arc<CudaDevice>,
        p: &mut QuantGemvParams,
    ) -> Result<(), String> {
        let f = QUANT_GEMV.get_func(device, "q8_0_gemv")?;
        let cfg = LaunchConfig {
            grid_dim: (p.m.div_ceil(ROWS), p.b, 1),
            block_dim: (32, ROWS, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            f.launch(
                cfg,
                (&mut *p.out, p.w, p.x, p.m as i32, p.k as i32, p.b as i32),
            )
            .map_err(|e| format!("q8_0_gemv launch: {e}"))
        }
    }

    pub fn launch_q4_k_gemv(
        device: &Arc<CudaDevice>,
        p: &mut QuantGemvParams,
    ) -> Result<(), String> {
        let f = QUANT_GEMV.get_func(device, "q4_k_gemv")?;
        let cfg = LaunchConfig {
            grid_dim: (p.m.div_ceil(ROWS), p.b, 1),
            block_dim: (32, ROWS, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            f.launch(
                cfg,
                (&mut *p.out, p.w, p.x, p.m as i32, p.k as i32, p.b as i32),
            )
            .map_err(|e| format!("q4_k_gemv launch: {e}"))
        }
    }
}

#[cfg(feature = "cuda")]
pub use inner::{launch_q4_k_gemv, launch_q8_0_gemv, QuantGemvParams};
