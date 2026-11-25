// Causal self-attention for the full prompt in one kernel launch.
//
// Compiled at runtime via NVRTC; launched from `src/cuda/attention.rs`.
//
// Q/K/V are contiguous [N, num_heads, head_dim] buffers; no paging.
//
// Grid:   (seq_len, num_q_heads, 1)
// Block:  (256, 1, 1)
// Shared: (head_dim + seq_len + 1024) * 4 bytes

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
