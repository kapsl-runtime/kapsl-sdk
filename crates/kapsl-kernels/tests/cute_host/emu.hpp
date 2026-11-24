// Host emulation of one CUDA thread block, for testing CuTe kernels without a
// GPU.
//
// Every CUDA thread is an OS thread. `__syncthreads`, warp shuffles and the
// tensor-core MMA instruction are rendezvous points built on barriers, so a
// kernel whose control flow is not uniform across a warp/block hangs here
// instead of silently passing.
//
// The MMA instruction is emulated from CuTe's own thread/value layouts for the
// SM80 atom. That checks a kernel's logic against CuTe's model of the
// hardware. It does not check CuTe itself, or the PTX that nvcc generates.
#pragma once

#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include <cute/tensor.hpp>
#include <cute/atom/mma_atom.hpp>

namespace kapsl_emu {

constexpr int kWarpSize = 32;

// Cyclic barrier: wait() returns once `parties` threads have called it.
class Barrier {
 public:
  explicit Barrier(int parties) : parties_(parties) {}

  void wait() {
    std::unique_lock<std::mutex> lock(mu_);
    const std::uint64_t generation = generation_;
    if (++waiting_ == parties_) {
      waiting_ = 0;
      ++generation_;
      cv_.notify_all();
    } else {
      cv_.wait(lock, [&] { return generation != generation_; });
    }
  }

 private:
  std::mutex              mu_;
  std::condition_variable cv_;
  const int               parties_;
  int                     waiting_    = 0;
  std::uint64_t           generation_ = 0;
};

// Exchange area for one warp's collective operations.
struct Warp {
  Barrier       barrier{kWarpSize};
  float         shfl[kWarpSize];
  std::uint16_t a[kWarpSize][8];  // A registers, as f16 bit patterns
  std::uint16_t b[kWarpSize][4];  // B registers, as f16 bit patterns
  float         c[kWarpSize][4];
};

struct Block {
  explicit Block(int nthreads) : sync(nthreads) {
    for (int w = 0; w < nthreads / kWarpSize; ++w) warps.push_back(std::make_unique<Warp>());
  }
  Barrier                            sync;
  std::vector<std::unique_ptr<Warp>> warps;
};

struct Uint3 {
  unsigned x = 0, y = 0, z = 0;
};

inline thread_local Block* tl_block = nullptr;

}  // namespace kapsl_emu

// CUDA built-ins, as seen by device code.
inline thread_local kapsl_emu::Uint3 threadIdx, blockIdx, blockDim, gridDim;

namespace kapsl_emu {

inline Warp& this_warp() { return *tl_block->warps[threadIdx.x / kWarpSize]; }
inline int   this_lane() { return int(threadIdx.x % kWarpSize); }

inline float half_bits_to_float(std::uint16_t bits) {
  return float(cutlass::half_t::bitcast(bits));
}

// Emulated `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32`:
//   D[m][n] = C[m][n] + sum_k A[m][k] * B[n][k]      (M=16, N=8, K=16)
// with each lane's registers placed by CuTe's layouts for the real op.
struct EmuSM80_16x8x16_F32F16F16F32_TN {
  using DRegisters = float[4];
  using ARegisters = std::uint32_t[4];
  using BRegisters = std::uint32_t[2];
  using CRegisters = float[4];

  static void fma(float& d0, float& d1, float& d2, float& d3,
                  std::uint32_t const& a0, std::uint32_t const& a1,
                  std::uint32_t const& a2, std::uint32_t const& a3,
                  std::uint32_t const& b0, std::uint32_t const& b1,
                  float const& c0, float const& c1, float const& c2, float const& c3) {
    using Real = cute::MMA_Traits<cute::SM80_16x8x16_F32F16F16F32_TN>;

    Warp&     warp = this_warp();
    const int lane = this_lane();

    // D aliases C in `gemm(mma, A, B, C)`, so read the inputs before writing.
    const std::uint32_t a_regs[4] = {a0, a1, a2, a3};
    const std::uint32_t b_regs[2] = {b0, b1};
    const float         c_regs[4] = {c0, c1, c2, c3};
    std::memcpy(warp.a[lane], a_regs, sizeof(a_regs));
    std::memcpy(warp.b[lane], b_regs, sizeof(b_regs));
    std::memcpy(warp.c[lane], c_regs, sizeof(c_regs));
    warp.barrier.wait();  // every lane has deposited its registers

    // Reassemble the full operands. Layouts map (thread, value) to a
    // column-major index into (M, K), (N, K) and (M, N) respectively.
    float A[16][16];
    float B[8][16];
    for (int t = 0; t < kWarpSize; ++t) {
      for (int v = 0; v < 8; ++v) {
        const int idx = int(Real::ALayout{}(t, v));
        A[idx % 16][idx / 16] = half_bits_to_float(warp.a[t][v]);
      }
      for (int v = 0; v < 4; ++v) {
        const int idx = int(Real::BLayout{}(t, v));
        B[idx % 8][idx / 8] = half_bits_to_float(warp.b[t][v]);
      }
    }

    float d[4];
    for (int v = 0; v < 4; ++v) {
      const int idx = int(Real::CLayout{}(lane, v));
      const int m   = idx % 16;
      const int n   = idx / 16;
      float acc = c_regs[v];
      for (int k = 0; k < 16; ++k) acc += A[m][k] * B[n][k];
      d[v] = acc;
    }
    warp.barrier.wait();  // every lane has read the exchange area

    d0 = d[0];
    d1 = d[1];
    d2 = d[2];
    d3 = d[3];
  }
};

// Run `kernel()` once per CUDA thread for every block of `grid`.
// `per_block()` runs before each block starts (e.g. to reset shared memory).
template <class PerBlock, class Kernel>
void launch(Uint3 grid, int block_threads, PerBlock&& per_block, Kernel&& kernel) {
  for (unsigned by = 0; by < grid.y; ++by) {
    for (unsigned bx = 0; bx < grid.x; ++bx) {
      per_block();
      Block                    block(block_threads);
      std::vector<std::thread> threads;
      threads.reserve(block_threads);
      for (int t = 0; t < block_threads; ++t) {
        threads.emplace_back([&, t] {
          threadIdx = {unsigned(t), 0, 0};
          blockIdx  = {bx, by, 0};
          blockDim  = {unsigned(block_threads), 1, 1};
          gridDim   = grid;
          tl_block  = &block;
          kernel();
        });
      }
      for (auto& th : threads) th.join();
    }
  }
}

}  // namespace kapsl_emu

inline void __syncthreads() { kapsl_emu::tl_block->sync.wait(); }

inline float __shfl_xor_sync(unsigned /*mask*/, float value, int lane_mask) {
  kapsl_emu::Warp& warp = kapsl_emu::this_warp();
  const int        lane = kapsl_emu::this_lane();
  warp.shfl[lane] = value;
  warp.barrier.wait();
  const float other = warp.shfl[lane ^ lane_mask];
  warp.barrier.wait();
  return other;
}

namespace cute {

// Same thread/value layouts and value types as the real SM80 op.
template <>
struct MMA_Traits<kapsl_emu::EmuSM80_16x8x16_F32F16F16F32_TN>
    : MMA_Traits<SM80_16x8x16_F32F16F16F32_TN> {};

}  // namespace cute
