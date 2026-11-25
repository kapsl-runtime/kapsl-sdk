// Paged-attention decode kernel.
//
// Compiled at runtime via NVRTC; launched from `src/cuda/attention.rs`.

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
