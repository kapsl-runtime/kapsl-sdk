// Fused dequantize-GEMV kernels for Q8_0 and Q4_K weight matrices.
//
// Compiled at runtime via NVRTC; launched from `src/cuda/quant.rs`.
//
// Both kernels compute:  out[m + b*M] = Σ_k  W_quant[m,k] · x[k + b*K]
//
// Grid:  (ceil(M / ROWS_PER_BLK), B)
// Block: (32 = one warp, ROWS_PER_BLK)
// Each warp handles one output row; threads stride across K-blocks then
// warp-shuffle reduce.

#include <cuda_fp16.h>

// ── helpers ──────────────────────────────────────────────────────────────────

__device__ __forceinline__ void scale_min_k4(
    int j, const unsigned char* q,
    unsigned char* sc_out, unsigned char* m_out)
{
    if (j < 4) {
        *sc_out = q[j] & 63;
        *m_out  = q[j + 4] & 63;
    } else {
        *sc_out = (q[j + 4] & 0x0f) | ((q[j - 4] >> 6) << 4);
        *m_out  = (q[j + 4] >> 4)   | ((q[j]     >> 6) << 4);
    }
}

// ── Q8_0 batched GEMV ────────────────────────────────────────────────────────
//
// W stored row-major as Q8_0 blocks: [f16 scale, i8 qs[32]]  — 34 bytes/block
// x layout:   x[b * K + k]      (row b, feature k)
// out layout: out[b * M + m]    (row b, output m)

#define Q80_ROWS 4

extern "C" __global__ void q8_0_gemv(
    __half* __restrict__         out,
    const unsigned char* __restrict__ W,
    const __half* __restrict__   x,
    int M, int K, int B)
{
    int row = blockIdx.x * Q80_ROWS + (int)threadIdx.y;
    int b   = (int)blockIdx.y;
    int tid = (int)threadIdx.x;
    if (row >= M || b >= B) return;

    int K_blocks = K / 32;
    const unsigned char* W_row = W + (long long)row * K_blocks * 34;
    const __half* x_b = x + (long long)b * K;

    float acc = 0.0f;
    for (int blk = tid; blk < K_blocks; blk += 32) {
        const unsigned char* bp = W_row + blk * 34;
        unsigned short scale_bits = (unsigned short)bp[0] | ((unsigned short)bp[1] << 8);
        float d = __half2float(__ushort_as_half(scale_bits));
        const signed char* qs = (const signed char*)(bp + 2);
        const __half* xp = x_b + blk * 32;
        float s = 0.0f;
        for (int i = 0; i < 32; i++)
            s += (float)qs[i] * __half2float(xp[i]);
        acc += d * s;
    }

    // Warp-level reduce across the 32 lanes
    for (int off = 16; off > 0; off >>= 1)
        acc += __shfl_down_sync(0xffffffff, acc, off);

    if (tid == 0)
        out[(long long)b * M + row] = __float2half(acc);
}

// ── Q4_K batched GEMV ────────────────────────────────────────────────────────
//
// W stored row-major as Q4_K superblocks:
//   [f16 d, f16 dmin, u8 scales[12], u8 qs[128]]  — 144 bytes/superblock
//   Each superblock covers 256 output elements in 4 groups of 64.
//   Within each group, 32 qs bytes encode 64 4-bit values (lo/hi nibble).

#define Q4K_ROWS 4

extern "C" __global__ void q4_k_gemv(
    __half* __restrict__         out,
    const unsigned char* __restrict__ W,
    const __half* __restrict__   x,
    int M, int K, int B)
{
    int row = blockIdx.x * Q4K_ROWS + (int)threadIdx.y;
    int b   = (int)blockIdx.y;
    int tid = (int)threadIdx.x;
    if (row >= M || b >= B) return;

    int K_sbs = K / 256;
    const unsigned char* W_row = W + (long long)row * K_sbs * 144;
    const __half* x_b = x + (long long)b * K;

    float acc = 0.0f;
    for (int sb = tid; sb < K_sbs; sb += 32) {
        const unsigned char* blk = W_row + sb * 144;
        float d    = __half2float(__ushort_as_half(
                         (unsigned short)blk[0] | ((unsigned short)blk[1] << 8)));
        float dmin = __half2float(__ushort_as_half(
                         (unsigned short)blk[2] | ((unsigned short)blk[3] << 8)));
        const unsigned char* sc = blk + 4;
        const unsigned char* qs = blk + 16;
        const __half* xp = x_b + sb * 256;

        int is = 0, qi = 0, oi = 0;
        for (int g = 0; g < 4; g++) {
            unsigned char sc0, m0, sc1, m1;
            scale_min_k4(is,     sc, &sc0, &m0);
            scale_min_k4(is + 1, sc, &sc1, &m1);
            float d1 = d * sc0, m1v = dmin * m0;
            float d2 = d * sc1, m2v = dmin * m1;
            for (int l = 0; l < 32; l++) {
                acc += (d1 * (float)(qs[qi+l] & 0xf) - m1v) * __half2float(xp[oi + l]);
                acc += (d2 * (float)(qs[qi+l] >> 4)  - m2v) * __half2float(xp[oi + 32 + l]);
            }
            qi += 32; is += 2; oi += 64;
        }
    }

    for (int off = 16; off > 0; off >>= 1)
        acc += __shfl_down_sync(0xffffffff, acc, off);

    if (tid == 0)
        out[(long long)b * M + row] = __float2half(acc);
}
