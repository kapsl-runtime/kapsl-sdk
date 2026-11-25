// RMS normalisation.
//
// Compiled at runtime via NVRTC; launched from `src/cuda/attention.rs`.
//
// Grid:   (rows, 1, 1)
// Block:  (min(256, dim), 1, 1)
// Shared: blockDim.x * 4 bytes

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
