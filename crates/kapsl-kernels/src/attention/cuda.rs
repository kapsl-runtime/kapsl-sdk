//! CUDA paged-attention kernel.
//!
//! Compiles the embedded CUDA source via NVRTC at first use, then launches
//! the kernel through cudarc. The kernel reads K/V from a GPU block pool
//! using a per-sequence block table — no gather or CPU-GPU copies needed.
//!
//! Block pool layout: `[num_blocks, 2, num_kv_heads, block_size, head_dim]`
//! where dim 1 = 0 for key and 1 for value.

#[cfg(feature = "cuda")]
mod inner {
    use cudarc::driver::{CudaDevice, CudaFunction, CudaSlice, LaunchAsync, LaunchConfig};
    use cudarc::nvrtc::compile_ptx;
    use half::f16;
    use std::sync::{Arc, OnceLock};

    /// Compiled PTX module, lazily initialised.
    struct KernelModule {
        device: Arc<CudaDevice>,
    }

    static MODULE_NAME: &str = "kapsl_paged_attn";
    static KERNEL_NAME: &str = "paged_attention_v1";

    /// CUDA kernel source — compiled at runtime via NVRTC.
    const KERNEL_SRC: &str = r#"
#include <cuda_fp16.h>

// Each thread block handles one (batch, head) pair.
// Threads cooperate across key positions using shared memory for scores.
//
// Block pool layout: [num_blocks, 2, num_kv_heads, block_size, head_dim]
//   dim1=0 → key, dim1=1 → value
//
// Grid:  (batch_size, num_query_heads, 1)
// Block: (THREADS_PER_BLOCK, 1, 1)

extern "C"
__global__ void paged_attention_v1(
    __half* __restrict__       out,            // [batch, num_q_heads, head_dim]
    const __half* __restrict__ q,              // [batch, num_q_heads, head_dim]
    const __half* __restrict__ kv_cache,       // [num_blocks, 2, num_kv_heads, block_size, head_dim]
    const int* __restrict__    block_tables,   // [batch, max_blocks_per_seq]
    const int* __restrict__    context_lens,   // [batch]
    const float               scale,
    const int                  num_q_heads,
    const int                  num_kv_heads,
    const int                  head_dim,
    const int                  block_size,
    const int                  max_blocks_per_seq
) {
    const int batch_idx = blockIdx.x;
    const int head_idx  = blockIdx.y;
    const int tid       = threadIdx.x;
    const int nthreads  = blockDim.x;

    const int ctx_len = context_lens[batch_idx];
    if (ctx_len <= 0) return;

    // GQA: map query head → KV head.
    const int kv_head_idx = head_idx * num_kv_heads / num_q_heads;

    // Strides inside the block pool.
    const int kv_head_stride = block_size * head_dim;
    const int kv_type_stride = num_kv_heads * kv_head_stride;
    const int block_stride   = 2 * kv_type_stride;

    // Pointer to this batch/head's query vector.
    const __half* q_ptr = q + (batch_idx * num_q_heads + head_idx) * head_dim;

    // Pointer to this batch's block table.
    const int* bt = block_tables + batch_idx * max_blocks_per_seq;

    // ─── Pass 1: compute attention scores ─────────────────────────────

    // Shared memory: first head_dim floats = query (f32), then ctx_len scores.
    extern __shared__ float smem[];
    float* q_smem     = smem;
    float* score_smem = smem + head_dim;

    // Load query into shared memory (collaborative).
    for (int d = tid; d < head_dim; d += nthreads) {
        q_smem[d] = __half2float(q_ptr[d]);
    }
    __syncthreads();

    // Each thread processes a strided set of context positions.
    float thread_max = -1e20f;
    for (int pos = tid; pos < ctx_len; pos += nthreads) {
        const int block_idx_logical = pos / block_size;
        const int pos_in_block      = pos % block_size;
        const int phys_block        = bt[block_idx_logical];

        // Key pointer for this position: kv_cache[phys_block, 0, kv_head, pos_in_block, :]
        const __half* k_ptr = kv_cache
            + phys_block * block_stride
            + 0 * kv_type_stride
            + kv_head_idx * kv_head_stride
            + pos_in_block * head_dim;

        float dot = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            dot += q_smem[d] * __half2float(k_ptr[d]);
        }
        dot *= scale;
        score_smem[pos] = dot;
        if (dot > thread_max) thread_max = dot;
    }
    __syncthreads();

    // ─── Reduction: find global max ───────────────────────────────────
    // Use shared memory for block-level reduction.
    __shared__ float reduce_smem[1024];
    reduce_smem[tid] = thread_max;
    __syncthreads();
    for (int s = nthreads / 2; s > 0; s >>= 1) {
        if (tid < s && reduce_smem[tid + s] > reduce_smem[tid]) {
            reduce_smem[tid] = reduce_smem[tid + s];
        }
        __syncthreads();
    }
    const float global_max = reduce_smem[0];

    // ─── Pass 2: exp(score - max) and sum ─────────────────────────────
    float thread_sum = 0.0f;
    for (int pos = tid; pos < ctx_len; pos += nthreads) {
        float val = expf(score_smem[pos] - global_max);
        score_smem[pos] = val;
        thread_sum += val;
    }
    __syncthreads();

    reduce_smem[tid] = thread_sum;
    __syncthreads();
    for (int s = nthreads / 2; s > 0; s >>= 1) {
        if (tid < s) reduce_smem[tid] += reduce_smem[tid + s];
        __syncthreads();
    }
    const float inv_sum = (reduce_smem[0] > 0.0f) ? (1.0f / reduce_smem[0]) : 0.0f;

    // ─── Pass 3: weighted sum of values ───────────────────────────────

    __half* out_ptr = out + (batch_idx * num_q_heads + head_idx) * head_dim;

    for (int d = tid; d < head_dim; d += nthreads) {
        float acc = 0.0f;
        for (int pos = 0; pos < ctx_len; pos++) {
            const int block_idx_logical = pos / block_size;
            const int pos_in_block      = pos % block_size;
            const int phys_block        = bt[block_idx_logical];

            // Value pointer: kv_cache[phys_block, 1, kv_head, pos_in_block, d]
            const __half* v_ptr = kv_cache
                + phys_block * block_stride
                + 1 * kv_type_stride
                + kv_head_idx * kv_head_stride
                + pos_in_block * head_dim;

            acc += score_smem[pos] * __half2float(v_ptr[d]);
        }
        out_ptr[d] = __float2half(acc * inv_sum);
    }
}
"#;

    /// Ensure the PTX module is loaded on `device`. Idempotent.
    fn ensure_module(device: &Arc<CudaDevice>) -> Result<(), String> {
        // cudarc caches modules by name, so second call is a no-op.
        if device.get_func(MODULE_NAME, KERNEL_NAME).is_some() {
            return Ok(());
        }
        let ptx = compile_ptx(KERNEL_SRC).map_err(|e| format!("NVRTC compile failed: {e}"))?;
        device
            .load_ptx(ptx, MODULE_NAME, &[KERNEL_NAME])
            .map_err(|e| format!("PTX load failed: {e}"))?;
        Ok(())
    }

    /// Parameters for a single paged-attention launch.
    pub struct PagedAttentionParams<'a> {
        /// Output:  [batch, num_q_heads, head_dim], device-allocated f16.
        pub out: &'a mut CudaSlice<f16>,
        /// Query:   [batch, num_q_heads, head_dim], device f16.
        pub q: &'a CudaSlice<f16>,
        /// KV pool: [num_blocks, 2, num_kv_heads, block_size, head_dim], device f16.
        pub kv_cache: &'a CudaSlice<f16>,
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
        ensure_module(device)?;

        let func = device
            .get_func(MODULE_NAME, KERNEL_NAME)
            .ok_or("kernel function not found after load")?;

        // Thread block size: 256 is a good default for Ampere/Hopper.
        let threads = 256u32;

        // Shared memory: head_dim floats (query) + max_context_len floats (scores)
        //                + 1024 floats (reduction workspace).
        // We conservatively allocate for the maximum context length across the batch.
        // The host should pass a reasonable upper bound via max_blocks_per_seq * block_size.
        let max_ctx = params.max_blocks_per_seq * params.block_size;
        let shared_bytes = ((params.head_dim + max_ctx + 1024) * 4) as u32;

        let cfg = LaunchConfig {
            grid_dim: (params.batch_size, params.num_q_heads, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: shared_bytes,
        };

        unsafe {
            func.clone()
                .launch(
                    cfg,
                    (
                        params.out,
                        params.q,
                        params.kv_cache,
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

    /// RMS-norm CUDA kernel source.
    const RMSNORM_SRC: &str = r#"
#include <cuda_fp16.h>

extern "C"
__global__ void rms_norm(
    __half* __restrict__       out,    // [rows, dim]
    const __half* __restrict__ input,  // [rows, dim]
    const __half* __restrict__ weight, // [dim]
    const int                  dim,
    const float                eps
) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;

    const __half* x = input + row * dim;
    __half*       o = out   + row * dim;

    // Compute sum of squares.
    extern __shared__ float smem[];
    float local_ss = 0.0f;
    for (int i = tid; i < dim; i += nthreads) {
        float val = __half2float(x[i]);
        local_ss += val * val;
    }
    smem[tid] = local_ss;
    __syncthreads();

    for (int s = nthreads / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }

    float rms = rsqrtf(smem[0] / (float)dim + eps);

    for (int i = tid; i < dim; i += nthreads) {
        float val = __half2float(x[i]) * rms * __half2float(weight[i]);
        o[i] = __float2half(val);
    }
}
"#;

    static RMSNORM_MODULE: &str = "kapsl_rmsnorm";
    static RMSNORM_KERNEL: &str = "rms_norm";

    pub struct RmsNormParams<'a> {
        pub out: &'a mut CudaSlice<f16>,
        pub input: &'a CudaSlice<f16>,
        pub weight: &'a CudaSlice<f16>,
        pub rows: u32,
        pub dim: u32,
        pub eps: f32,
    }

    pub fn launch_rms_norm(
        device: &Arc<CudaDevice>,
        params: &mut RmsNormParams<'_>,
    ) -> Result<(), String> {
        if device.get_func(RMSNORM_MODULE, RMSNORM_KERNEL).is_none() {
            let ptx =
                compile_ptx(RMSNORM_SRC).map_err(|e| format!("NVRTC rmsnorm compile: {e}"))?;
            device
                .load_ptx(ptx, RMSNORM_MODULE, &[RMSNORM_KERNEL])
                .map_err(|e| format!("PTX load: {e}"))?;
        }
        let func = device
            .get_func(RMSNORM_MODULE, RMSNORM_KERNEL)
            .ok_or("rmsnorm not found")?;

        let threads = 256u32.min(params.dim);
        let shared_bytes = threads * 4;
        let cfg = LaunchConfig {
            grid_dim: (params.rows, 1, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: shared_bytes,
        };
        unsafe {
            func.clone()
                .launch(
                    cfg,
                    (
                        params.out,
                        params.input,
                        params.weight,
                        params.dim as i32,
                        params.eps,
                    ),
                )
                .map_err(|e| format!("rmsnorm launch: {e}"))?;
        }
        Ok(())
    }

    /// Fused SwiGLU activation kernel: out = silu(gate) * up
    const SWIGLU_SRC: &str = r#"
#include <cuda_fp16.h>

extern "C"
__global__ void fused_swiglu(
    __half* __restrict__       out,
    const __half* __restrict__ gate,
    const __half* __restrict__ up,
    const int                  n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float g = __half2float(gate[idx]);
        float u = __half2float(up[idx]);
        float silu = g / (1.0f + expf(-g));
        out[idx] = __float2half(silu * u);
    }
}
"#;

    static SWIGLU_MODULE: &str = "kapsl_swiglu";
    static SWIGLU_KERNEL: &str = "fused_swiglu";

    pub fn launch_fused_swiglu(
        device: &Arc<CudaDevice>,
        out: &mut CudaSlice<f16>,
        gate: &CudaSlice<f16>,
        up: &CudaSlice<f16>,
        n: u32,
    ) -> Result<(), String> {
        if device.get_func(SWIGLU_MODULE, SWIGLU_KERNEL).is_none() {
            let ptx =
                compile_ptx(SWIGLU_SRC).map_err(|e| format!("NVRTC swiglu compile: {e}"))?;
            device
                .load_ptx(ptx, SWIGLU_MODULE, &[SWIGLU_KERNEL])
                .map_err(|e| format!("PTX load: {e}"))?;
        }
        let func = device
            .get_func(SWIGLU_MODULE, SWIGLU_KERNEL)
            .ok_or("swiglu not found")?;

        let threads = 256u32;
        let blocks = (n + threads - 1) / threads;
        let cfg = LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            func.clone()
                .launch(cfg, (out, gate, up, n as i32))
                .map_err(|e| format!("swiglu launch: {e}"))?;
        }
        Ok(())
    }

    // ─── RoPE (Rotary Position Embeddings) ─────────────────────────────

    const ROPE_SRC: &str = r#"
#include <cuda_fp16.h>

// Apply rotary position embeddings to Q and K vectors.
//
// Each thread handles one (head, pair) combination.
// Grid:  (num_heads, 1, 1)
// Block: (half_head_dim, 1, 1)   — one thread per cosine/sine pair
//
// For each consecutive pair (x[2i], x[2i+1]):
//   freq = position * theta^(-2i / head_dim)
//   (x[2i], x[2i+1]) = (x[2i]*cos - x[2i+1]*sin, x[2i]*sin + x[2i+1]*cos)

extern "C"
__global__ void rope_forward(
    __half* __restrict__       q,          // [num_q_heads, head_dim]
    __half* __restrict__       k,          // [num_kv_heads, head_dim]
    const int                  num_q_heads,
    const int                  num_kv_heads,
    const int                  head_dim,
    const int                  position,
    const float                theta_base
) {
    const int head = blockIdx.x;
    const int pair = threadIdx.x;
    const int half_dim = head_dim / 2;

    if (pair >= half_dim) return;

    float freq = (float)position * powf(theta_base, -2.0f * (float)pair / (float)head_dim);
    float cos_val = cosf(freq);
    float sin_val = sinf(freq);

    // Apply to Q head
    if (head < num_q_heads) {
        int idx = head * head_dim + pair * 2;
        float x0 = __half2float(q[idx]);
        float x1 = __half2float(q[idx + 1]);
        q[idx]     = __float2half(x0 * cos_val - x1 * sin_val);
        q[idx + 1] = __float2half(x0 * sin_val + x1 * cos_val);
    }

    // Apply to K head (only if this head index is within KV head range)
    if (head < num_kv_heads) {
        int idx = head * head_dim + pair * 2;
        float x0 = __half2float(k[idx]);
        float x1 = __half2float(k[idx + 1]);
        k[idx]     = __float2half(x0 * cos_val - x1 * sin_val);
        k[idx + 1] = __float2half(x0 * sin_val + x1 * cos_val);
    }
}
"#;

    static ROPE_MODULE: &str = "kapsl_rope";
    static ROPE_KERNEL: &str = "rope_forward";

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
        if device.get_func(ROPE_MODULE, ROPE_KERNEL).is_none() {
            let ptx = compile_ptx(ROPE_SRC).map_err(|e| format!("NVRTC rope compile: {e}"))?;
            device
                .load_ptx(ptx, ROPE_MODULE, &[ROPE_KERNEL])
                .map_err(|e| format!("PTX load: {e}"))?;
        }
        let func = device
            .get_func(ROPE_MODULE, ROPE_KERNEL)
            .ok_or("rope not found")?;

        let max_heads = params.num_q_heads.max(params.num_kv_heads);
        let half_dim = params.head_dim / 2;
        let cfg = LaunchConfig {
            grid_dim: (max_heads, 1, 1),
            block_dim: (half_dim, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            func.clone()
                .launch(
                    cfg,
                    (
                        params.q,
                        params.k,
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

    const KV_WRITE_SRC: &str = r#"
#include <cuda_fp16.h>

// Write a single token's K and V vectors into the paged block pool.
//
// Block pool layout: [num_blocks, 2, num_kv_heads, block_size, head_dim]
//
// Grid:  (num_kv_heads, 1, 1)
// Block: (head_dim, 1, 1)

extern "C"
__global__ void write_kv_to_pool(
    __half* __restrict__       kv_cache,       // block pool storage
    const __half* __restrict__ k_vec,          // [num_kv_heads, head_dim]
    const __half* __restrict__ v_vec,          // [num_kv_heads, head_dim]
    const int                  physical_block,
    const int                  pos_in_block,
    const int                  num_kv_heads,
    const int                  block_size,
    const int                  head_dim
) {
    const int kv_head = blockIdx.x;
    const int d       = threadIdx.x;
    if (kv_head >= num_kv_heads || d >= head_dim) return;

    const int kv_head_stride = block_size * head_dim;
    const int kv_type_stride = num_kv_heads * kv_head_stride;
    const int block_stride   = 2 * kv_type_stride;

    int base = physical_block * block_stride;
    int head_offset = kv_head * kv_head_stride + pos_in_block * head_dim + d;

    // Write key: kv_cache[block, 0, kv_head, pos, d]
    kv_cache[base + 0 * kv_type_stride + head_offset] = k_vec[kv_head * head_dim + d];
    // Write value: kv_cache[block, 1, kv_head, pos, d]
    kv_cache[base + 1 * kv_type_stride + head_offset] = v_vec[kv_head * head_dim + d];
}
"#;

    static KV_WRITE_MODULE: &str = "kapsl_kv_write";
    static KV_WRITE_KERNEL: &str = "write_kv_to_pool";

    pub struct KvWriteParams<'a> {
        pub kv_cache: &'a mut CudaSlice<f16>,
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
        if device.get_func(KV_WRITE_MODULE, KV_WRITE_KERNEL).is_none() {
            let ptx =
                compile_ptx(KV_WRITE_SRC).map_err(|e| format!("NVRTC kv_write compile: {e}"))?;
            device
                .load_ptx(ptx, KV_WRITE_MODULE, &[KV_WRITE_KERNEL])
                .map_err(|e| format!("PTX load: {e}"))?;
        }
        let func = device
            .get_func(KV_WRITE_MODULE, KV_WRITE_KERNEL)
            .ok_or("kv_write not found")?;

        let cfg = LaunchConfig {
            grid_dim: (params.num_kv_heads, 1, 1),
            block_dim: (params.head_dim, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            func.clone()
                .launch(
                    cfg,
                    (
                        params.kv_cache,
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

    const RESIDUAL_ADD_SRC: &str = r#"
#include <cuda_fp16.h>

// Elementwise: out[i] = a[i] + b[i]
// Grid/block: standard 1D covering n elements.

extern "C"
__global__ void residual_add(
    __half* __restrict__       out,
    const __half* __restrict__ a,
    const __half* __restrict__ b,
    const int                  n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = __float2half(__half2float(a[idx]) + __half2float(b[idx]));
    }
}
"#;

    static RESIDUAL_MODULE: &str = "kapsl_residual";
    static RESIDUAL_KERNEL: &str = "residual_add";

    pub fn launch_residual_add(
        device: &Arc<CudaDevice>,
        out: &mut CudaSlice<f16>,
        a: &CudaSlice<f16>,
        b: &CudaSlice<f16>,
        n: u32,
    ) -> Result<(), String> {
        if device.get_func(RESIDUAL_MODULE, RESIDUAL_KERNEL).is_none() {
            let ptx = compile_ptx(RESIDUAL_ADD_SRC)
                .map_err(|e| format!("NVRTC residual compile: {e}"))?;
            device
                .load_ptx(ptx, RESIDUAL_MODULE, &[RESIDUAL_KERNEL])
                .map_err(|e| format!("PTX load: {e}"))?;
        }
        let func = device
            .get_func(RESIDUAL_MODULE, RESIDUAL_KERNEL)
            .ok_or("residual_add not found")?;

        let threads = 256u32;
        let blocks = (n + threads - 1) / threads;
        let cfg = LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            func.clone()
                .launch(cfg, (out, a, b, n as i32))
                .map_err(|e| format!("residual_add launch: {e}"))?;
        }
        Ok(())
    }

    // ─── Batch RoPE ─────────────────────────────────────────────────────
    // Applies rotary embeddings to [N, num_heads, head_dim] Q and K.
    // Grid: (seq_len, max(num_q_heads, num_kv_heads), 1)
    // Block: (head_dim/2, 1, 1)

    const BATCH_ROPE_SRC: &str = r#"
#include <cuda_fp16.h>

extern "C"
__global__ void batch_rope_forward(
    __half* __restrict__       q,
    __half* __restrict__       k,
    const int                  num_q_heads,
    const int                  num_kv_heads,
    const int                  head_dim,
    const int                  position_offset,
    const float                theta_base
) {
    const int seq_pos  = blockIdx.x;
    const int head     = blockIdx.y;
    const int pair     = threadIdx.x;
    const int half_dim = head_dim / 2;
    if (pair >= half_dim) return;

    const int position = position_offset + seq_pos;
    const float freq    = (float)position
                        * powf(theta_base, -2.0f * (float)pair / (float)head_dim);
    const float cos_val = cosf(freq);
    const float sin_val = sinf(freq);

    if (head < num_q_heads) {
        const int base = seq_pos * num_q_heads * head_dim + head * head_dim + pair * 2;
        const float x0 = __half2float(q[base]);
        const float x1 = __half2float(q[base + 1]);
        q[base]     = __float2half(x0 * cos_val - x1 * sin_val);
        q[base + 1] = __float2half(x0 * sin_val + x1 * cos_val);
    }
    if (head < num_kv_heads) {
        const int base = seq_pos * num_kv_heads * head_dim + head * head_dim + pair * 2;
        const float x0 = __half2float(k[base]);
        const float x1 = __half2float(k[base + 1]);
        k[base]     = __float2half(x0 * cos_val - x1 * sin_val);
        k[base + 1] = __float2half(x0 * sin_val + x1 * cos_val);
    }
}
"#;

    static BATCH_ROPE_MODULE: &str = "kapsl_batch_rope";
    static BATCH_ROPE_KERNEL: &str = "batch_rope_forward";

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
        if device.get_func(BATCH_ROPE_MODULE, BATCH_ROPE_KERNEL).is_none() {
            let ptx = compile_ptx(BATCH_ROPE_SRC)
                .map_err(|e| format!("NVRTC batch_rope compile: {e}"))?;
            device
                .load_ptx(ptx, BATCH_ROPE_MODULE, &[BATCH_ROPE_KERNEL])
                .map_err(|e| format!("PTX load: {e}"))?;
        }
        let func = device
            .get_func(BATCH_ROPE_MODULE, BATCH_ROPE_KERNEL)
            .ok_or("batch_rope not found")?;

        let max_heads = p.num_q_heads.max(p.num_kv_heads);
        let cfg = LaunchConfig {
            grid_dim: (p.seq_len, max_heads, 1),
            block_dim: (p.head_dim / 2, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            func.clone()
                .launch(
                    cfg,
                    (
                        p.q,
                        p.k,
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
    // Q/K/V are contiguous [N, num_heads, head_dim] buffers; no paging.
    //
    // Grid:  (seq_len, num_q_heads, 1)
    // Block: (256, 1, 1)
    // Shared: (head_dim + seq_len + 1024) * 4 bytes

    const PREFILL_ATTN_SRC: &str = r#"
#include <cuda_fp16.h>

extern "C"
__global__ void prefill_attention(
    __half* __restrict__       out,
    const __half* __restrict__ q,
    const __half* __restrict__ k,
    const __half* __restrict__ v,
    const float               scale,
    const int                  seq_len,
    const int                  num_q_heads,
    const int                  num_kv_heads,
    const int                  head_dim
) {
    const int query_pos = blockIdx.x;
    const int head_idx  = blockIdx.y;
    const int tid       = threadIdx.x;
    const int nthreads  = blockDim.x;

    const int kv_head       = head_idx * num_kv_heads / num_q_heads;
    const int ctx_len       = query_pos + 1;
    const int q_row_stride  = num_q_heads  * head_dim;
    const int kv_row_stride = num_kv_heads * head_dim;

    const __half* q_ptr = q + query_pos * q_row_stride + head_idx * head_dim;

    extern __shared__ float smem[];
    float* q_smem     = smem;
    float* score_smem = smem + head_dim;

    for (int d = tid; d < head_dim; d += nthreads)
        q_smem[d] = __half2float(q_ptr[d]);
    __syncthreads();

    float thread_max = -1e20f;
    for (int pos = tid; pos < ctx_len; pos += nthreads) {
        const __half* k_ptr = k + pos * kv_row_stride + kv_head * head_dim;
        float dot = 0.0f;
        for (int d = 0; d < head_dim; d++)
            dot += q_smem[d] * __half2float(k_ptr[d]);
        dot *= scale;
        score_smem[pos] = dot;
        if (dot > thread_max) thread_max = dot;
    }
    __syncthreads();

    __shared__ float reduce[1024];
    reduce[tid] = thread_max;
    __syncthreads();
    for (int s = nthreads / 2; s > 0; s >>= 1) {
        if (tid < s && reduce[tid + s] > reduce[tid]) reduce[tid] = reduce[tid + s];
        __syncthreads();
    }
    const float gmax = reduce[0];

    float thread_sum = 0.0f;
    for (int pos = tid; pos < ctx_len; pos += nthreads) {
        const float val = expf(score_smem[pos] - gmax);
        score_smem[pos] = val;
        thread_sum += val;
    }
    __syncthreads();

    reduce[tid] = thread_sum;
    __syncthreads();
    for (int s = nthreads / 2; s > 0; s >>= 1) {
        if (tid < s) reduce[tid] += reduce[tid + s];
        __syncthreads();
    }
    const float inv_sum = (reduce[0] > 0.0f) ? (1.0f / reduce[0]) : 0.0f;

    __half* out_ptr = out + query_pos * q_row_stride + head_idx * head_dim;
    for (int d = tid; d < head_dim; d += nthreads) {
        float acc = 0.0f;
        for (int pos = 0; pos < ctx_len; pos++) {
            const __half* v_ptr = v + pos * kv_row_stride + kv_head * head_dim;
            acc += score_smem[pos] * __half2float(v_ptr[d]);
        }
        out_ptr[d] = __float2half(acc * inv_sum);
    }
}
"#;

    static PREFILL_ATTN_MODULE: &str = "kapsl_prefill_attn";
    static PREFILL_ATTN_KERNEL: &str = "prefill_attention";

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

    pub fn launch_prefill_attention(
        device: &Arc<CudaDevice>,
        p: &mut PrefillAttnParams<'_>,
    ) -> Result<(), String> {
        if device
            .get_func(PREFILL_ATTN_MODULE, PREFILL_ATTN_KERNEL)
            .is_none()
        {
            let ptx = compile_ptx(PREFILL_ATTN_SRC)
                .map_err(|e| format!("NVRTC prefill_attn compile: {e}"))?;
            device
                .load_ptx(ptx, PREFILL_ATTN_MODULE, &[PREFILL_ATTN_KERNEL])
                .map_err(|e| format!("PTX load: {e}"))?;
        }
        let func = device
            .get_func(PREFILL_ATTN_MODULE, PREFILL_ATTN_KERNEL)
            .ok_or("prefill_attention not found")?;

        let threads = 256u32;
        let shared =
            ((p.head_dim + p.seq_len + 1024) * 4) as u32;
        let cfg = LaunchConfig {
            grid_dim: (p.seq_len, p.num_q_heads, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: shared,
        };
        unsafe {
            func.clone()
                .launch(
                    cfg,
                    (
                        p.out,
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
    //
    // Grid:  (N, num_kv_heads, 1)
    // Block: (head_dim, 1, 1)

    const BATCH_KV_WRITE_SRC: &str = r#"
#include <cuda_fp16.h>

extern "C"
__global__ void batch_kv_write(
    __half* __restrict__       kv_cache,
    const __half* __restrict__ k,
    const __half* __restrict__ v,
    const int* __restrict__    physical_blocks,
    const int* __restrict__    pos_in_blocks,
    const int                  num_kv_heads,
    const int                  block_size,
    const int                  head_dim
) {
    const int seq_pos = blockIdx.x;
    const int kv_head = blockIdx.y;
    const int d       = threadIdx.x;
    if (d >= head_dim) return;

    const int phys = physical_blocks[seq_pos];
    const int pos  = pos_in_blocks[seq_pos];

    const int kv_head_stride = block_size * head_dim;
    const int kv_type_stride = num_kv_heads * kv_head_stride;
    const int block_stride   = 2 * kv_type_stride;
    const int src_row_stride = num_kv_heads * head_dim;

    const int base     = phys * block_stride;
    const int head_off = kv_head * kv_head_stride + pos * head_dim + d;
    const int src_off  = seq_pos * src_row_stride + kv_head * head_dim + d;

    kv_cache[base + 0 * kv_type_stride + head_off] = k[src_off];
    kv_cache[base + 1 * kv_type_stride + head_off] = v[src_off];
}
"#;

    static BATCH_KV_WRITE_MODULE: &str = "kapsl_batch_kv_write";
    static BATCH_KV_WRITE_KERNEL: &str = "batch_kv_write";

    pub struct BatchKvWriteParams<'a> {
        pub kv_cache: &'a mut CudaSlice<f16>,
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
        if device
            .get_func(BATCH_KV_WRITE_MODULE, BATCH_KV_WRITE_KERNEL)
            .is_none()
        {
            let ptx = compile_ptx(BATCH_KV_WRITE_SRC)
                .map_err(|e| format!("NVRTC batch_kv_write compile: {e}"))?;
            device
                .load_ptx(ptx, BATCH_KV_WRITE_MODULE, &[BATCH_KV_WRITE_KERNEL])
                .map_err(|e| format!("PTX load: {e}"))?;
        }
        let func = device
            .get_func(BATCH_KV_WRITE_MODULE, BATCH_KV_WRITE_KERNEL)
            .ok_or("batch_kv_write not found")?;

        let cfg = LaunchConfig {
            grid_dim: (p.seq_len, p.num_kv_heads, 1),
            block_dim: (p.head_dim, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            func.clone()
                .launch(
                    cfg,
                    (
                        p.kv_cache,
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
    use cudarc::driver::{CudaDevice, CudaSlice, LaunchAsync, LaunchConfig};
    use cudarc::nvrtc::compile_ptx;
    use half::f16;
    use std::sync::Arc;

    /// Parallel argmax over a single row of f16 logits stored in GPU memory.
    ///
    /// Grid: (1,1,1), Block: (256,1,1).  Each thread sweeps vocab/256 elements
    /// (serial phase), then the 256 partial winners are reduced in shared memory
    /// (parallel phase).  Only one u32 is transferred back to CPU.
    const ARGMAX_SRC: &str = r#"
#include <cuda_fp16.h>

extern "C"
__global__ void argmax_f16(
    const __half* __restrict__ input,
    unsigned int* __restrict__ output,
    const int                  vocab_size
) {
    __shared__ float       s_val[256];
    __shared__ unsigned int s_idx[256];

    const int tid    = threadIdx.x;
    const int stride = blockDim.x;

    float        best_val = -1.0e30f;
    unsigned int best_idx = 0;

    for (int i = tid; i < vocab_size; i += stride) {
        float v = __half2float(input[i]);
        if (v > best_val) { best_val = v; best_idx = (unsigned int)i; }
    }

    s_val[tid] = best_val;
    s_idx[tid] = best_idx;
    __syncthreads();

    for (int s = stride >> 1; s > 0; s >>= 1) {
        if (tid < s && s_val[tid + s] > s_val[tid]) {
            s_val[tid] = s_val[tid + s];
            s_idx[tid] = s_idx[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) output[0] = s_idx[0];
}
"#;

    static ARGMAX_MODULE: &str = "kapsl_argmax";
    static ARGMAX_KERNEL: &str = "argmax_f16";

    pub struct ArgmaxParams<'a> {
        pub input:      &'a CudaSlice<f16>,
        pub output:     &'a mut CudaSlice<u32>,
        pub vocab_size: u32,
    }

    pub fn launch_argmax(
        device: &Arc<CudaDevice>,
        p: &mut ArgmaxParams<'_>,
    ) -> Result<(), String> {
        if device.get_func(ARGMAX_MODULE, ARGMAX_KERNEL).is_none() {
            let ptx = compile_ptx(ARGMAX_SRC)
                .map_err(|e| format!("NVRTC argmax compile: {e}"))?;
            device.load_ptx(ptx, ARGMAX_MODULE, &[ARGMAX_KERNEL])
                .map_err(|e| format!("PTX load: {e}"))?;
        }
        let func = device.get_func(ARGMAX_MODULE, ARGMAX_KERNEL).ok_or("argmax not found")?;
        let cfg = LaunchConfig { grid_dim: (1, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
        unsafe {
            func.clone()
                .launch(cfg, (p.input, p.output, p.vocab_size as i32))
                .map_err(|e| format!("argmax launch: {e}"))?;
        }
        Ok(())
    }
}

#[cfg(feature = "cuda")]
pub use inner::*;
#[cfg(feature = "cuda")]
pub use argmax_inner::{launch_argmax, ArgmaxParams};
