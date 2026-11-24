// Causal prefill self-attention on tensor cores, written with CuTe (CUTLASS).
//
// Compiled ahead of time by `build.rs` with nvcc (needs sm_80+); launched from
// `src/cuda/cute.rs`. Same inputs and outputs as `../prefill_attention.cu`,
// which remains the fallback for older GPUs and unsupported head dims.
//
// Q/K/V are contiguous [N, num_heads, head_dim] f16 buffers; no paging.
//
// Algorithm (FlashAttention-style online softmax):
//   One thread block owns kBlockM consecutive query rows of one head and walks
//   the keys in tiles of kBlockN. Per tile:
//       S  = scale * Q K^T          tensor-core MMA, f32 accumulate
//       S  = causal_mask(S); running row max / row sum; P = exp(S - max)
//       O += P V                    tensor-core MMA, f32 accumulate
//   and finally O /= row_sum. Shared memory holds one K tile and one V tile,
//   so it does not grow with the sequence length.
//
// Launch contract (must match `src/cuda/cute.rs`):
//   Grid:   (ceil(seq_len / kBlockM), num_q_heads, 1)
//   Block:  (kNThreads, 1, 1)
//   Shared: kSmemElems * sizeof(half) bytes, dynamic
//
// Not done yet (correctness first): swizzled shared memory + ldmatrix loads,
// cp.async for the global loads, head dims above 128.

#include <cmath>

#include <cute/tensor.hpp>
#include <cute/atom/mma_atom.hpp>

// The host emulation test (tests/cute_host) swaps in an emulated tensor-core
// op with the same register layouts; everything else is shared.
#ifndef KAPSL_CUTE_MMA_OP
#define KAPSL_CUTE_MMA_OP cute::SM80_16x8x16_F32F16F16F32_TN
#endif

namespace kapsl {

using namespace cute;

template <int kHeadDim_>
struct PrefillTraits {
  using Element = cute::half_t;

  static constexpr int kHeadDim  = kHeadDim_;
  static constexpr int kNWarps   = 4;
  static constexpr int kNThreads = kNWarps * 32;
  static constexpr int kBlockM   = kNWarps * 16;  // query rows per thread block
  static constexpr int kBlockN   = 64;            // keys per tile

  static_assert(kHeadDim % 16 == 0, "head_dim must be a multiple of the MMA K extent");
  static_assert(kHeadDim <= 128, "Q is held in registers; larger head dims need Q in smem");
  static_assert(kBlockN % 16 == 0, "tile width must be a multiple of the MMA K extent");
  static_assert(kBlockM <= kBlockN, "the Q tile is staged in the K tile's buffer");

  // Each warp owns 16 query rows and every column of the tile.
  using TiledMma = TiledMMA<MMA_Atom<KAPSL_CUTE_MMA_OP>,
                            Layout<Shape<Int<kNWarps>, _1, _1>>,
                            Tile<Int<kBlockM>, _16, _16>>;

  // Shared-memory tiles, row-major [row, head_dim].
  using QLayout  = Layout<Shape<Int<kBlockM>, Int<kHeadDim>>, Stride<Int<kHeadDim>, _1>>;
  using KLayout  = Layout<Shape<Int<kBlockN>, Int<kHeadDim>>, Stride<Int<kHeadDim>, _1>>;
  // The V tile viewed as [head_dim, row]: the B operand of O += P V.
  using VtLayout = Layout<Shape<Int<kHeadDim>, Int<kBlockN>>, Stride<_1, Int<kHeadDim>>>;

  static constexpr int kTileElems = kBlockN * kHeadDim;
  static constexpr int kSmemElems = 2 * kTileElems;  // K tile + V tile
};

// In an SM80 16x8 accumulator fragment each thread holds, per atom, two rows
// of two adjacent columns, in the order
//     value v = c + 2 * r      (c: column in the pair, r: row in the pair)
// and the four threads of a "quad" (lanes 4g .. 4g+3) hold the same two rows.
// The loops below rely on that; tests/cute_host asserts it for every thread.
constexpr int kFragRows = 2;
constexpr int kFragCols = 2;

// Reduce a per-thread partial across the four threads that share its rows.
CUTE_DEVICE float quad_max(float x) {
  x = fmaxf(x, __shfl_xor_sync(0xffffffffu, x, 2));
  x = fmaxf(x, __shfl_xor_sync(0xffffffffu, x, 1));
  return x;
}

CUTE_DEVICE float quad_sum(float x) {
  x += __shfl_xor_sync(0xffffffffu, x, 2);
  x += __shfl_xor_sync(0xffffffffu, x, 1);
  return x;
}

template <class Traits>
CUTE_DEVICE void prefill_attention_block(
    typename Traits::Element*       out,   // [seq_len, num_q_heads,  head_dim]
    const typename Traits::Element* q,     // [seq_len, num_q_heads,  head_dim]
    const typename Traits::Element* k,     // [seq_len, num_kv_heads, head_dim]
    const typename Traits::Element* v,     // [seq_len, num_kv_heads, head_dim]
    const float scale,
    const int   seq_len,
    const int   num_q_heads,
    const int   num_kv_heads,
    typename Traits::Element*       smem)  // kSmemElems elements, 16-byte aligned
{
  using Element = typename Traits::Element;
  constexpr int kHeadDim  = Traits::kHeadDim;
  constexpr int kBlockM   = Traits::kBlockM;
  constexpr int kBlockN   = Traits::kBlockN;
  constexpr int kNThreads = Traits::kNThreads;

  const int tid  = threadIdx.x;
  const int m0   = blockIdx.x * kBlockM;  // first query row of this block
  const int head = blockIdx.y;

  // GQA: map query head -> KV head.
  const int kv_head = head * num_kv_heads / num_q_heads;

  const long long q_row_stride  = (long long)num_q_heads * kHeadDim;
  const long long kv_row_stride = (long long)num_kv_heads * kHeadDim;
  const long long q_head_off    = (long long)head * kHeadDim;
  const long long kv_head_off   = (long long)kv_head * kHeadDim;

  const Element zero(0.0f);

  Element* smem_k = smem;
  Element* smem_v = smem + Traits::kTileElems;

  Tensor sQ  = make_tensor(make_smem_ptr(smem_k), typename Traits::QLayout{});   // (M, D)
  Tensor sK  = make_tensor(make_smem_ptr(smem_k), typename Traits::KLayout{});   // (N, D)
  Tensor sVt = make_tensor(make_smem_ptr(smem_v), typename Traits::VtLayout{});  // (D, N)

  typename Traits::TiledMma tiled_mma;
  auto thr_mma = tiled_mma.get_thread_slice(tid);

  // (m, n) / (m, d) coordinates of every accumulator element this thread owns.
  Tensor cS   = make_identity_tensor(Shape<Int<kBlockM>, Int<kBlockN>>{});
  Tensor cO   = make_identity_tensor(Shape<Int<kBlockM>, Int<kHeadDim>>{});
  Tensor tScS = thr_mma.partition_C(cS);  // (MMA, MMA_M, MMA_N)
  Tensor tOcO = thr_mma.partition_C(cO);  // (MMA, MMA_M, MMA_D)

  // ─── Q tile: global -> shared -> registers ──────────────────────────
  for (int i = tid; i < kBlockM * kHeadDim; i += kNThreads) {
    const int row = m0 + i / kHeadDim;
    if (row < seq_len) {
      smem_k[i] = q[row * q_row_stride + q_head_off + i % kHeadDim];
    } else {
      smem_k[i] = zero;
    }
  }
  __syncthreads();

  Tensor tSsQ = thr_mma.partition_A(sQ);           // (MMA, MMA_M, MMA_K)
  Tensor tSrQ = thr_mma.partition_fragment_A(sQ);  // registers
  copy(tSsQ, tSrQ);
  __syncthreads();  // the K tile reuses this buffer

  // ─── Per-thread state ───────────────────────────────────────────────
  Tensor tSsK  = thr_mma.partition_B(sK);            // (MMA, MMA_N, MMA_K)
  Tensor tSrK  = thr_mma.partition_fragment_B(sK);
  Tensor tOsVt = thr_mma.partition_B(sVt);           // (MMA, MMA_D, MMA_N/2)
  Tensor tOrVt = thr_mma.partition_fragment_B(sVt);

  Tensor acc_s = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{});   // (MMA, MMA_M, MMA_N)
  Tensor acc_o = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kHeadDim>>{});  // (MMA, MMA_M, MMA_D)
  // P as the A operand of O += P V: (MMA, MMA_M, MMA_N/2), f16 registers.
  Tensor tOrP  = make_tensor<Element>(shape(thr_mma.partition_A(cS)));

  constexpr int kMmaM = decltype(size<1>(acc_s))::value;
  constexpr int kMmaN = decltype(size<2>(acc_s))::value;
  constexpr int kMmaD = decltype(size<2>(acc_o))::value;
  static_assert(decltype(size<0>(acc_s))::value == kFragRows * kFragCols);
  static_assert(decltype(size<0>(tOrP))::value == 2 * kFragRows * kFragCols);
  static_assert(decltype(size<2>(tOrP))::value * 2 == kMmaN);

  float row_max[kMmaM * kFragRows];
  float row_sum[kMmaM * kFragRows];
  CUTE_UNROLL
  for (int i = 0; i < kMmaM * kFragRows; ++i) {
    row_max[i] = -INFINITY;
    row_sum[i] = 0.0f;
  }
  clear(acc_o);

  // Causal: this block needs keys 0 ..= its last valid query row.
  const int last_row = (m0 + kBlockM < seq_len ? m0 + kBlockM : seq_len) - 1;
  const int n_tiles  = last_row / kBlockN + 1;

  for (int tile = 0; tile < n_tiles; ++tile) {
    const int n0 = tile * kBlockN;  // first key of this tile

    // ─── K and V tiles: global -> shared ──────────────────────────────
    for (int i = tid; i < kBlockN * kHeadDim; i += kNThreads) {
      const int row = n0 + i / kHeadDim;
      if (row < seq_len) {
        const long long off = row * kv_row_stride + kv_head_off + i % kHeadDim;
        smem_k[i] = k[off];
        smem_v[i] = v[off];
      } else {
        smem_k[i] = zero;
        smem_v[i] = zero;
      }
    }
    __syncthreads();

    // ─── S = Q K^T ────────────────────────────────────────────────────
    clear(acc_s);
    CUTE_UNROLL
    for (int kk = 0; kk < size<2>(tSrQ); ++kk) {
      copy(tSsK(_, _, kk), tSrK(_, _, kk));
      gemm(tiled_mma, tSrQ(_, _, kk), tSrK(_, _, kk), acc_s);
    }

    // ─── Scale, causal mask, online softmax ───────────────────────────
    CUTE_UNROLL
    for (int mi = 0; mi < kMmaM; ++mi) {
      CUTE_UNROLL
      for (int r = 0; r < kFragRows; ++r) {
        const int ri  = mi * kFragRows + r;
        const int row = m0 + get<0>(tScS(kFragCols * r, mi, 0));

        float local_max = -INFINITY;
        CUTE_UNROLL
        for (int nj = 0; nj < kMmaN; ++nj) {
          CUTE_UNROLL
          for (int c = 0; c < kFragCols; ++c) {
            const int vi  = c + kFragCols * r;
            const int col = n0 + get<1>(tScS(vi, mi, nj));
            float s = acc_s(vi, mi, nj) * scale;
            if (col > row || col >= seq_len) s = -INFINITY;
            acc_s(vi, mi, nj) = s;
            local_max = fmaxf(local_max, s);
          }
        }

        // Every row has an unmasked key in tile 0 (key 0), so new_max is finite.
        const float new_max = fmaxf(row_max[ri], quad_max(local_max));
        const float rescale = expf(row_max[ri] - new_max);  // 0 on the first tile
        row_max[ri] = new_max;

        float local_sum = 0.0f;
        CUTE_UNROLL
        for (int nj = 0; nj < kMmaN; ++nj) {
          CUTE_UNROLL
          for (int c = 0; c < kFragCols; ++c) {
            const int vi = c + kFragCols * r;
            const float p = expf(acc_s(vi, mi, nj) - new_max);
            acc_s(vi, mi, nj) = p;
            local_sum += p;
          }
        }
        row_sum[ri] = row_sum[ri] * rescale + local_sum;

        CUTE_UNROLL
        for (int dj = 0; dj < kMmaD; ++dj) {
          CUTE_UNROLL
          for (int c = 0; c < kFragCols; ++c) {
            acc_o(c + kFragCols * r, mi, dj) *= rescale;
          }
        }
      }
    }

    // ─── P: f32 accumulator fragment -> f16 A-operand fragment ────────
    // The A fragment of one 16-wide K atom is the accumulator fragments of
    // its two 8-wide N atoms, back to back.
    CUTE_UNROLL
    for (int mi = 0; mi < kMmaM; ++mi) {
      CUTE_UNROLL
      for (int nk = 0; nk < kMmaN / 2; ++nk) {
        CUTE_UNROLL
        for (int h = 0; h < 2; ++h) {
          CUTE_UNROLL
          for (int vi = 0; vi < kFragRows * kFragCols; ++vi) {
            tOrP(vi + kFragRows * kFragCols * h, mi, nk) = Element(acc_s(vi, mi, 2 * nk + h));
          }
        }
      }
    }

    // ─── O += P V ─────────────────────────────────────────────────────
    CUTE_UNROLL
    for (int kk = 0; kk < size<2>(tOrP); ++kk) {
      copy(tOsVt(_, _, kk), tOrVt(_, _, kk));
      gemm(tiled_mma, tOrP(_, _, kk), tOrVt(_, _, kk), acc_o);
    }

    __syncthreads();  // everyone is done with this tile before it is overwritten
  }

  // ─── Normalise and write O ──────────────────────────────────────────
  CUTE_UNROLL
  for (int mi = 0; mi < kMmaM; ++mi) {
    CUTE_UNROLL
    for (int r = 0; r < kFragRows; ++r) {
      const int   ri      = mi * kFragRows + r;
      const float sum     = quad_sum(row_sum[ri]);
      const float inv_sum = sum > 0.0f ? 1.0f / sum : 0.0f;
      const int   row     = m0 + get<0>(tOcO(kFragCols * r, mi, 0));
      if (row < seq_len) {
        Element* out_row = out + row * q_row_stride + q_head_off;
        CUTE_UNROLL
        for (int dj = 0; dj < kMmaD; ++dj) {
          CUTE_UNROLL
          for (int c = 0; c < kFragCols; ++c) {
            const int vi = c + kFragCols * r;
            out_row[get<1>(tOcO(vi, mi, dj))] = Element(acc_o(vi, mi, dj) * inv_sum);
          }
        }
      }
    }
  }
}

}  // namespace kapsl

#if defined(__CUDACC__)

// One entry point per supported head dim; keep in sync with
// `HEAD_DIMS` and `KERNELS` in `src/cuda/cute.rs`.
#define KAPSL_PREFILL_ATTENTION_ENTRY(HEAD_DIM)                                        \
  extern "C" __global__ void prefill_attention_cute_d##HEAD_DIM(                       \
      __half* out, const __half* q, const __half* k, const __half* v,                  \
      const float scale, const int seq_len,                                            \
      const int num_q_heads, const int num_kv_heads)                                   \
  {                                                                                    \
    extern __shared__ char smem[];                                                     \
    kapsl::prefill_attention_block<kapsl::PrefillTraits<HEAD_DIM>>(                    \
        reinterpret_cast<cute::half_t*>(out),                                          \
        reinterpret_cast<const cute::half_t*>(q),                                      \
        reinterpret_cast<const cute::half_t*>(k),                                      \
        reinterpret_cast<const cute::half_t*>(v),                                      \
        scale, seq_len, num_q_heads, num_kv_heads,                                     \
        reinterpret_cast<cute::half_t*>(smem));                                        \
  }

KAPSL_PREFILL_ATTENTION_ENTRY(32)
KAPSL_PREFILL_ATTENTION_ENTRY(64)
KAPSL_PREFILL_ATTENTION_ENTRY(96)
KAPSL_PREFILL_ATTENTION_ENTRY(128)

#endif  // __CUDACC__
