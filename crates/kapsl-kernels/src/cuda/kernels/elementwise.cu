// Elementwise f16 kernels: fused SwiGLU and residual add.
//
// Compiled at runtime via NVRTC; launched from `src/cuda/attention.rs`.
//
// Grid/block: standard 1D covering n elements.

#include <cuda_fp16.h>

// out = silu(gate) * up
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

// out[i] = a[i] + b[i]
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
