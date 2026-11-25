// GPU greedy sampling (argmax over f16 logits).
//
// Compiled at runtime via NVRTC; launched from `src/cuda/attention.rs`.
//
// Each thread sweeps vocab/256 elements (serial phase), then the 256 partial
// winners are reduced in shared memory (parallel phase). Only the winning
// index is transferred back to the CPU.

#include <cuda_fp16.h>

// Single row of logits.
//
// Grid:  (1, 1, 1)
// Block: (256, 1, 1)
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

// Batch of logit rows.
//
// Grid:   (batch_size, 1, 1)
// Block:  (256, 1, 1)
// Input:  logits[batch, vocab_size]   (row-major)
// Output: argmax[batch]
extern "C"
__global__ void batch_argmax_f16(
    const __half* __restrict__ input,      // [batch, vocab_size]
    unsigned int* __restrict__ output,     // [batch]
    const int                  vocab_size
) {
    __shared__ float        s_val[256];
    __shared__ unsigned int s_idx[256];

    const int batch_idx = blockIdx.x;
    const int tid       = threadIdx.x;
    const int stride    = blockDim.x;

    const __half* row = input + (long long)batch_idx * vocab_size;

    float        best_val = -1.0e30f;
    unsigned int best_idx = 0;

    for (int i = tid; i < vocab_size; i += stride) {
        float v = __half2float(row[i]);
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

    if (tid == 0) output[batch_idx] = s_idx[0];
}
