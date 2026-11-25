//! CUDA kernels for attention, normalisation, RoPE, KV-cache writes and
//! greedy sampling.
//!
//! The CUDA sources live in `kernels/*.cu` and are embedded with
//! `include_str!`. Each file is compiled via NVRTC on first use and launched
//! through cudarc. The paged-attention kernel reads K/V from a GPU block pool
//! using a per-sequence block table — no gather or CPU-GPU copies needed.
//!
//! Block pool layout: `[num_blocks, 2, num_kv_heads, block_size, head_dim]`
//! where dim 1 = 0 for key and 1 for value.

#[cfg(feature = "cuda")]
mod sources {
    use crate::nvrtc_util::KernelModule;

    pub(super) static PAGED_ATTENTION: KernelModule = KernelModule {
        name: "kapsl_paged_attn",
        src: include_str!("kernels/paged_attention.cu"),
        kernels: &["paged_attention_v1"],
    };

    pub(super) static PREFILL_ATTENTION: KernelModule = KernelModule {
        name: "kapsl_prefill_attn",
        src: include_str!("kernels/prefill_attention.cu"),
        kernels: &["prefill_attention"],
    };

    pub(super) static RMS_NORM: KernelModule = KernelModule {
        name: "kapsl_rmsnorm",
        src: include_str!("kernels/rms_norm.cu"),
        kernels: &["rms_norm"],
    };

    pub(super) static ELEMENTWISE: KernelModule = KernelModule {
        name: "kapsl_elementwise",
        src: include_str!("kernels/elementwise.cu"),
        kernels: &["fused_swiglu", "residual_add"],
    };

    pub(super) static ROPE: KernelModule = KernelModule {
        name: "kapsl_rope",
        src: include_str!("kernels/rope.cu"),
        kernels: &["rope_forward", "batch_rope_forward", "batch_decode_rope"],
    };

    pub(super) static KV_CACHE: KernelModule = KernelModule {
        name: "kapsl_kv_cache",
        src: include_str!("kernels/kv_cache.cu"),
        kernels: &["write_kv_to_pool", "batch_kv_write"],
    };

    pub(super) static ARGMAX: KernelModule = KernelModule {
        name: "kapsl_argmax",
        src: include_str!("kernels/argmax.cu"),
        kernels: &["argmax_f16", "batch_argmax_f16"],
    };
}

#[cfg(feature = "cuda")]
mod inner {
    use super::sources::{
        ELEMENTWISE, KV_CACHE, PAGED_ATTENTION, PREFILL_ATTENTION, RMS_NORM, ROPE,
    };
    use cudarc::driver::{CudaDevice, CudaSlice, CudaView, CudaViewMut, LaunchAsync, LaunchConfig};
    use half::f16;
    use std::sync::Arc;

    // ─── Paged Attention (decode) ───────────────────────────────────────

    /// Parameters for a single paged-attention launch.
    pub struct PagedAttentionParams<'a> {
        /// Output:  [batch, num_q_heads, head_dim], device-allocated f16.
        pub out: &'a mut CudaSlice<f16>,
        /// Query:   [batch, num_q_heads, head_dim], device f16.
        pub q: &'a CudaSlice<f16>,
        /// KV pool: [num_blocks, 2, num_kv_heads, block_size, head_dim], device f16.
        pub kv_cache: CudaView<'a, f16>,
        /// Block tables: [batch, max_blocks_per_seq], device i32.
        pub block_tables: &'a CudaSlice<i32>,
        /// Context lengths: [batch], device i32.
        pub context_lens: &'a CudaSlice<i32>,
        pub scale: f32,
        pub batch_size: u32,
        pub num_q_heads: u32,
        pub num_kv_heads: u32,
        pub head_dim: u32,
        pub block_size: u32,
        pub max_blocks_per_seq: u32,
    }

    /// Launch the paged attention CUDA kernel.
    pub fn launch_paged_attention(
        device: &Arc<CudaDevice>,
        params: &mut PagedAttentionParams<'_>,
    ) -> Result<(), String> {
        let func = PAGED_ATTENTION.get_func(device, "paged_attention_v1")?;

        // Thread block size: 256 is a good default for Ampere/Hopper.
        let threads = 256u32;

        // Shared memory: head_dim floats (query) + max_context_len floats (scores)
        //                + 1024 floats (reduction workspace).
        // We conservatively allocate for the maximum context length across the batch.
        // The host should pass a reasonable upper bound via max_blocks_per_seq * block_size.
        let max_ctx = params.max_blocks_per_seq * params.block_size;
        let shared_bytes = (params.head_dim + max_ctx + 1024) * 4;

        let cfg = LaunchConfig {
            grid_dim: (params.batch_size, params.num_q_heads, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: shared_bytes,
        };

        unsafe {
            func.launch(
                cfg,
                (
                    &mut *params.out,
                    params.q,
                    &params.kv_cache,
                    params.block_tables,
                    params.context_lens,
                    params.scale,
                    params.num_q_heads as i32,
                    params.num_kv_heads as i32,
                    params.head_dim as i32,
                    params.block_size as i32,
                    params.max_blocks_per_seq as i32,
                ),
            )
            .map_err(|e| format!("kernel launch failed: {e}"))?;
        }
        Ok(())
    }

    // ─── RMS Norm ───────────────────────────────────────────────────────

    pub struct RmsNormParams<'a> {
        pub out: &'a mut CudaSlice<f16>,
        pub input: CudaView<'a, f16>,
        pub weight: &'a CudaSlice<f16>,
        pub rows: u32,
        pub dim: u32,
        pub eps: f32,
    }

    pub fn launch_rms_norm(
        device: &Arc<CudaDevice>,
        params: &mut RmsNormParams<'_>,
    ) -> Result<(), String> {
        let func = RMS_NORM.get_func(device, "rms_norm")?;

        let threads = 256u32.min(params.dim);
        let shared_bytes = threads * 4;
        let cfg = LaunchConfig {
            grid_dim: (params.rows, 1, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: shared_bytes,
        };
        unsafe {
            func.launch(
                cfg,
                (
                    &mut *params.out,
                    &params.input,
                    params.weight,
                    params.dim as i32,
                    params.eps,
                ),
            )
            .map_err(|e| format!("rmsnorm launch: {e}"))?;
        }
        Ok(())
    }

    // ─── Fused SwiGLU ───────────────────────────────────────────────────

    /// Fused SwiGLU activation: out = silu(gate) * up
    pub fn launch_fused_swiglu(
        device: &Arc<CudaDevice>,
        out: &mut CudaSlice<f16>,
        gate: &CudaSlice<f16>,
        up: &CudaSlice<f16>,
        n: u32,
    ) -> Result<(), String> {
        let func = ELEMENTWISE.get_func(device, "fused_swiglu")?;

        let threads = 256u32;
        let blocks = n.div_ceil(threads);
        let cfg = LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            func.launch(cfg, (out, gate, up, n as i32))
                .map_err(|e| format!("swiglu launch: {e}"))?;
        }
        Ok(())
    }

    // ─── RoPE (Rotary Position Embeddings) ─────────────────────────────

    pub struct RopeParams<'a> {
        pub q: &'a mut CudaSlice<f16>,
        pub k: &'a mut CudaSlice<f16>,
        pub num_q_heads: u32,
        pub num_kv_heads: u32,
        pub head_dim: u32,
        pub position: u32,
        pub theta: f32,
    }

    pub fn launch_rope(
        device: &Arc<CudaDevice>,
        params: &mut RopeParams<'_>,
    ) -> Result<(), String> {
        let func = ROPE.get_func(device, "rope_forward")?;

        let max_heads = params.num_q_heads.max(params.num_kv_heads);
        let half_dim = params.head_dim / 2;
        let cfg = LaunchConfig {
            grid_dim: (max_heads, 1, 1),
            block_dim: (half_dim, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            func.launch(
                cfg,
                (
                    &mut *params.q,
                    &mut *params.k,
                    params.num_q_heads as i32,
                    params.num_kv_heads as i32,
                    params.head_dim as i32,
                    params.position as i32,
                    params.theta,
                ),
            )
            .map_err(|e| format!("rope launch: {e}"))?;
        }
        Ok(())
    }

    // ─── KV Cache Write ─────────────────────────────────────────────────

    pub struct KvWriteParams<'a> {
        pub kv_cache: CudaViewMut<'a, f16>,
        pub k_vec: &'a CudaSlice<f16>,
        pub v_vec: &'a CudaSlice<f16>,
        pub physical_block: u32,
        pub pos_in_block: u32,
        pub num_kv_heads: u32,
        pub block_size: u32,
        pub head_dim: u32,
    }

    pub fn launch_kv_write(
        device: &Arc<CudaDevice>,
        params: &mut KvWriteParams<'_>,
    ) -> Result<(), String> {
        let func = KV_CACHE.get_func(device, "write_kv_to_pool")?;

        let cfg = LaunchConfig {
            grid_dim: (params.num_kv_heads, 1, 1),
            block_dim: (params.head_dim, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            func.launch(
                cfg,
                (
                    &mut params.kv_cache,
                    params.k_vec,
                    params.v_vec,
                    params.physical_block as i32,
                    params.pos_in_block as i32,
                    params.num_kv_heads as i32,
                    params.block_size as i32,
                    params.head_dim as i32,
                ),
            )
            .map_err(|e| format!("kv_write launch: {e}"))?;
        }
        Ok(())
    }

    // ─── Residual Add ───────────────────────────────────────────────────

    pub fn launch_residual_add(
        device: &Arc<CudaDevice>,
        out: &mut CudaSlice<f16>,
        a: &CudaSlice<f16>,
        b: &CudaSlice<f16>,
        n: u32,
    ) -> Result<(), String> {
        let func = ELEMENTWISE.get_func(device, "residual_add")?;

        let threads = 256u32;
        let blocks = n.div_ceil(threads);
        let cfg = LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            func.launch(cfg, (out, a, b, n as i32))
                .map_err(|e| format!("residual_add launch: {e}"))?;
        }
        Ok(())
    }

    // ─── Batch RoPE ─────────────────────────────────────────────────────
    // Applies rotary embeddings to [N, num_heads, head_dim] Q and K.

    pub struct BatchRopeParams<'a> {
        pub q: &'a mut CudaSlice<f16>,
        pub k: &'a mut CudaSlice<f16>,
        pub seq_len: u32,
        pub num_q_heads: u32,
        pub num_kv_heads: u32,
        pub head_dim: u32,
        pub position_offset: u32,
        pub theta: f32,
    }

    pub fn launch_batch_rope(
        device: &Arc<CudaDevice>,
        p: &mut BatchRopeParams<'_>,
    ) -> Result<(), String> {
        let func = ROPE.get_func(device, "batch_rope_forward")?;

        let max_heads = p.num_q_heads.max(p.num_kv_heads);
        let cfg = LaunchConfig {
            grid_dim: (p.seq_len, max_heads, 1),
            block_dim: (p.head_dim / 2, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            func.launch(
                cfg,
                (
                    &mut *p.q,
                    &mut *p.k,
                    p.num_q_heads as i32,
                    p.num_kv_heads as i32,
                    p.head_dim as i32,
                    p.position_offset as i32,
                    p.theta,
                ),
            )
            .map_err(|e| format!("batch_rope launch: {e}"))?;
        }
        Ok(())
    }

    // ─── Prefill Attention ───────────────────────────────────────────────
    // Causal self-attention for the full prompt in one kernel launch.

    pub struct PrefillAttnParams<'a> {
        pub out: &'a mut CudaSlice<f16>,
        pub q: &'a CudaSlice<f16>,
        pub k: &'a CudaSlice<f16>,
        pub v: &'a CudaSlice<f16>,
        pub scale: f32,
        pub seq_len: u32,
        pub num_q_heads: u32,
        pub num_kv_heads: u32,
        pub head_dim: u32,
    }

    /// Launch prefill attention with the best kernel this build and device
    /// support: the CuTe tensor-core kernel (`cute` feature, sm_80+, supported
    /// head dim) or else the scalar NVRTC kernel. The two agree to within f16
    /// rounding, not bit for bit.
    pub fn launch_prefill_attention(
        device: &Arc<CudaDevice>,
        p: &mut PrefillAttnParams<'_>,
    ) -> Result<(), String> {
        #[cfg(feature = "cute")]
        if launch_prefill_attention_cute(device, p)? {
            return Ok(());
        }
        launch_prefill_attention_scalar(device, p)
    }

    #[cfg(feature = "cute")]
    pub use crate::cute::launch_prefill_attention as launch_prefill_attention_cute;

    /// The NVRTC prefill-attention kernel: runs on any CUDA GPU and any head
    /// dim, without tensor cores. Its shared memory grows with `seq_len`.
    pub fn launch_prefill_attention_scalar(
        device: &Arc<CudaDevice>,
        p: &mut PrefillAttnParams<'_>,
    ) -> Result<(), String> {
        let func = PREFILL_ATTENTION.get_func(device, "prefill_attention")?;

        let threads = 256u32;
        let shared = (p.head_dim + p.seq_len + 1024) * 4;
        let cfg = LaunchConfig {
            grid_dim: (p.seq_len, p.num_q_heads, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: shared,
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
                    p.head_dim as i32,
                ),
            )
            .map_err(|e| format!("prefill_attn launch: {e}"))?;
        }
        Ok(())
    }

    // ─── Batch KV Write ──────────────────────────────────────────────────
    // Writes N tokens' K and V to the paged block pool in one launch.

    pub struct BatchKvWriteParams<'a> {
        pub kv_cache: CudaViewMut<'a, f16>,
        pub k: &'a CudaSlice<f16>,
        pub v: &'a CudaSlice<f16>,
        pub physical_blocks: &'a CudaSlice<i32>,
        pub pos_in_blocks: &'a CudaSlice<i32>,
        pub seq_len: u32,
        pub num_kv_heads: u32,
        pub block_size: u32,
        pub head_dim: u32,
    }

    pub fn launch_batch_kv_write(
        device: &Arc<CudaDevice>,
        p: &mut BatchKvWriteParams<'_>,
    ) -> Result<(), String> {
        let func = KV_CACHE.get_func(device, "batch_kv_write")?;

        let cfg = LaunchConfig {
            grid_dim: (p.seq_len, p.num_kv_heads, 1),
            block_dim: (p.head_dim, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            func.launch(
                cfg,
                (
                    &mut p.kv_cache,
                    p.k,
                    p.v,
                    p.physical_blocks,
                    p.pos_in_blocks,
                    p.num_kv_heads as i32,
                    p.block_size as i32,
                    p.head_dim as i32,
                ),
            )
            .map_err(|e| format!("batch_kv_write launch: {e}"))?;
        }
        Ok(())
    }
}

// ── GPU greedy-sampling (argmax) ─────────────────────────────────────────

#[cfg(feature = "cuda")]
mod argmax_inner {
    use super::sources::{ARGMAX, ROPE};
    use cudarc::driver::{CudaDevice, CudaSlice, LaunchAsync, LaunchConfig};
    use half::f16;
    use std::sync::Arc;

    /// Parallel argmax over a single row of f16 logits stored in GPU memory.
    /// Only one u32 is transferred back to CPU.
    pub struct ArgmaxParams<'a> {
        pub input: &'a CudaSlice<f16>,
        pub output: &'a mut CudaSlice<u32>,
        pub vocab_size: u32,
    }

    pub fn launch_argmax(device: &Arc<CudaDevice>, p: &mut ArgmaxParams<'_>) -> Result<(), String> {
        let func = ARGMAX.get_func(device, "argmax_f16")?;
        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            func.launch(cfg, (p.input, &mut *p.output, p.vocab_size as i32))
                .map_err(|e| format!("argmax launch: {e}"))?;
        }
        Ok(())
    }

    // ─── Batch Decode RoPE ───────────────────────────────────────────────
    // Per-sequence RoPE for a decode batch.  Each sequence contributes exactly
    // one token whose absolute position differs from all others.

    pub struct BatchDecodeRopeParams<'a> {
        pub q: &'a mut CudaSlice<f16>,
        pub k: &'a mut CudaSlice<f16>,
        pub positions: &'a CudaSlice<i32>,
        pub batch_size: u32,
        pub num_q_heads: u32,
        pub num_kv_heads: u32,
        pub head_dim: u32,
        pub theta: f32,
    }

    pub fn launch_batch_decode_rope(
        device: &Arc<CudaDevice>,
        p: &mut BatchDecodeRopeParams<'_>,
    ) -> Result<(), String> {
        let func = ROPE.get_func(device, "batch_decode_rope")?;

        let max_heads = p.num_q_heads.max(p.num_kv_heads);
        let cfg = LaunchConfig {
            grid_dim: (p.batch_size, max_heads, 1),
            block_dim: (p.head_dim / 2, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            func.launch(
                cfg,
                (
                    &mut *p.q,
                    &mut *p.k,
                    p.positions,
                    p.num_q_heads as i32,
                    p.num_kv_heads as i32,
                    p.head_dim as i32,
                    p.theta,
                ),
            )
            .map_err(|e| format!("batch_decode_rope launch: {e}"))?;
        }
        Ok(())
    }

    // ─── Batch Argmax ────────────────────────────────────────────────────
    // GPU greedy sampling over a batch of logit rows.

    pub struct BatchArgmaxParams<'a> {
        pub input: &'a CudaSlice<f16>,
        pub output: &'a mut CudaSlice<u32>,
        pub batch_size: u32,
        pub vocab_size: u32,
    }

    pub fn launch_batch_argmax(
        device: &Arc<CudaDevice>,
        p: &mut BatchArgmaxParams<'_>,
    ) -> Result<(), String> {
        let func = ARGMAX.get_func(device, "batch_argmax_f16")?;
        let cfg = LaunchConfig {
            grid_dim: (p.batch_size, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            func.launch(cfg, (p.input, &mut *p.output, p.vocab_size as i32))
                .map_err(|e| format!("batch_argmax launch: {e}"))?;
        }
        Ok(())
    }
}

#[cfg(feature = "cuda")]
pub use argmax_inner::{
    launch_argmax, launch_batch_argmax, launch_batch_decode_rope, ArgmaxParams, BatchArgmaxParams,
    BatchDecodeRopeParams,
};
#[cfg(feature = "cuda")]
pub use inner::*;
