// Rotary position embeddings (RoPE) for Q and K.
//
// Compiled at runtime via NVRTC; launched from `src/cuda/attention.rs`.
//
// For each consecutive pair (x[2i], x[2i+1]):
//   freq = position * theta^(-2i / head_dim)
//   (x[2i], x[2i+1]) = (x[2i]*cos - x[2i+1]*sin, x[2i]*sin + x[2i+1]*cos)

#include <cuda_fp16.h>

// Single token: Q is [num_q_heads, head_dim], K is [num_kv_heads, head_dim].
//
// Each thread handles one (head, pair) combination.
// Grid:  (max(num_q_heads, num_kv_heads), 1, 1)
// Block: (half_head_dim, 1, 1)   — one thread per cosine/sine pair
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

// Prefill: [N, num_heads, head_dim] Q and K at consecutive positions
// starting from position_offset.
//
// Grid:  (seq_len, max(num_q_heads, num_kv_heads), 1)
// Block: (head_dim/2, 1, 1)
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

// Decode batch: each sequence contributes exactly one token whose absolute
// position differs from all others.
//
// Grid:  (batch_size, max(num_q_heads, num_kv_heads), 1)
// Block: (head_dim / 2, 1, 1)
extern "C"
__global__ void batch_decode_rope(
    __half* __restrict__  q,               // [batch, num_q_heads,  head_dim]
    __half* __restrict__  k,               // [batch, num_kv_heads, head_dim]
    const int* __restrict__ positions,     // [batch]
    const int             num_q_heads,
    const int             num_kv_heads,
    const int             head_dim,
    const float           theta
) {
    const int batch_idx = blockIdx.x;
    const int head_idx  = blockIdx.y;
    const int pair      = threadIdx.x;   // 0 .. head_dim/2 - 1
    if (pair >= head_dim / 2) return;

    const int pos = positions[batch_idx];
    const float freq = __powf(theta, -2.0f * (float)pair / (float)head_dim);
    const float angle = (float)pos * freq;
    const float cos_val = cosf(angle);
    const float sin_val = sinf(angle);

    // Rotate Q (if head_idx < num_q_heads)
    if (head_idx < num_q_heads) {
        const int base = (batch_idx * num_q_heads + head_idx) * head_dim + pair * 2;
        const float x0 = __half2float(q[base]);
        const float x1 = __half2float(q[base + 1]);
        q[base]     = __float2half(x0 * cos_val - x1 * sin_val);
        q[base + 1] = __float2half(x0 * sin_val + x1 * cos_val);
    }

    // Rotate K (if head_idx < num_kv_heads)
    if (head_idx < num_kv_heads) {
        const int base = (batch_idx * num_kv_heads + head_idx) * head_dim + pair * 2;
        const float x0 = __half2float(k[base]);
        const float x1 = __half2float(k[base + 1]);
        k[base]     = __float2half(x0 * cos_val - x1 * sin_val);
        k[base + 1] = __float2half(x0 * sin_val + x1 * cos_val);
    }
}
