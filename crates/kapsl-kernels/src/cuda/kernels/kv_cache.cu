// Writes K/V vectors into the paged block pool.
//
// Compiled at runtime via NVRTC; launched from `src/cuda/attention.rs`.
//
// Block pool layout: [num_blocks, 2, num_kv_heads, block_size, head_dim]
//   dim1=0 → key, dim1=1 → value

#include <cuda_fp16.h>

// Write a single token's K and V vectors.
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

// Write N tokens' K and V vectors in one launch.
//
// Grid:  (N, num_kv_heads, 1)
// Block: (head_dim, 1, 1)
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
