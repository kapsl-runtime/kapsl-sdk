// CPU test of the CuTe prefill-attention kernel: runs the real kernel source
// under the thread-block emulator in emu.hpp and compares it with a
// double-precision reference. Needs a host C++17 compiler, the CUTLASS
// submodule and the CUDA toolkit *headers*; no GPU and no nvcc.
//
// From the repository root:
//
//   g++ -std=c++17 -O2 -pthread \
//       -I third_party/cutlass/include -isystem "$CUDA_PATH/include" \
//       crates/kapsl-kernels/tests/cute_host/prefill_attention_test.cpp \
//       -o target/prefill_attention_test && target/prefill_attention_test
//
// What this does and does not cover is described at the top of emu.hpp.

#include "emu.hpp"

#define KAPSL_CUTE_MMA_OP kapsl_emu::EmuSM80_16x8x16_F32F16F16F32_TN
#include "../../src/cuda/kernels/cute/prefill_attention.cu"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

namespace {

using Half = cute::half_t;

const Half kNaN = Half::bitcast(0x7e00);

int g_failures = 0;

#define CHECK(cond, ...)                \
  do {                                  \
    if (!(cond)) {                      \
      ++g_failures;                     \
      std::printf("  FAIL: " __VA_ARGS__); \
      std::printf("\n");                \
    }                                   \
  } while (0)

// The kernel hard-codes three facts about the SM80 accumulator fragment.
// Check them, for every thread, against CuTe's own partitioning.
template <class Traits>
void check_layout_assumptions() {
  using namespace cute;
  constexpr int M = Traits::kBlockM, N = Traits::kBlockN;
  auto cS = make_identity_tensor(Shape<Int<M>, Int<N>>{});
  typename Traits::TiledMma mma;

  std::vector<int> seen(M * N, 0);
  int before = g_failures;
  for (int tid = 0; tid < Traits::kNThreads; ++tid) {
    auto thr = mma.get_thread_slice(tid);
    auto tC  = thr.partition_C(cS);  // (4, MMA_M, MMA_N) -> (m, n)
    auto tA  = thr.partition_A(cS);  // (8, MMA_M, MMA_N/2) -> (m, k)
    const int mma_m = int(size<1>(tC)), mma_n = int(size<2>(tC));

    for (int mi = 0; mi < mma_m; ++mi) {
      for (int r = 0; r < kapsl::kFragRows; ++r) {
        // 1. value v = c + 2r: fixed r means one row, whatever c and the atom.
        const int row = int(get<0>(tC(kapsl::kFragCols * r, mi, 0)));
        for (int nj = 0; nj < mma_n; ++nj) {
          for (int c = 0; c < kapsl::kFragCols; ++c) {
            auto crd = tC(c + kapsl::kFragCols * r, mi, nj);
            CHECK(int(get<0>(crd)) == row, "thread %d: row varies within (mi=%d, r=%d)", tid, mi, r);
            ++seen[int(get<0>(crd)) * N + int(get<1>(crd))];
          }
        }
        // 2. the quad (lanes differing in the low two bits) shares its rows.
        for (int other : {tid ^ 1, tid ^ 2}) {
          auto tC2 = mma.get_thread_slice(other).partition_C(cS);
          CHECK(int(get<0>(tC2(kapsl::kFragCols * r, mi, 0))) == row,
                "threads %d and %d do not share row (mi=%d, r=%d)", tid, other, mi, r);
        }
      }
      // 3. A fragment of K atom nk == accumulator fragments of N atoms 2nk, 2nk+1.
      for (int nk = 0; nk < mma_n / 2; ++nk) {
        for (int h = 0; h < 2; ++h) {
          for (int vi = 0; vi < 4; ++vi) {
            auto a = tA(vi + 4 * h, mi, nk);
            auto c = tC(vi, mi, 2 * nk + h);
            CHECK(int(get<0>(a)) == int(get<0>(c)) && int(get<1>(a)) == int(get<1>(c)),
                  "thread %d: A/C fragment mismatch at (vi=%d, h=%d, nk=%d)", tid, vi, h, nk);
          }
        }
      }
    }
  }
  // Every (m, n) of the tile is owned by exactly one (thread, value).
  CHECK(std::all_of(seen.begin(), seen.end(), [](int n) { return n == 1; }),
        "accumulator fragments do not tile the %dx%d block exactly once", M, N);
  std::printf("layout assumptions d=%-3d : %s\n", Traits::kHeadDim,
              g_failures == before ? "ok" : "FAILED");
}

struct Case {
  int   seq_len, num_q_heads, num_kv_heads;
  float input_range;  // Q and K are uniform in [-range, range]
};

// Rows past `seq_len` are NaN: any read of them poisons the output.
std::vector<Half> random_rows(std::mt19937& rng, int rows, int row_elems, float range, int guard_rows) {
  std::uniform_real_distribution<float> dist(-range, range);
  std::vector<Half> buf(size_t(rows + guard_rows) * row_elems, kNaN);
  for (size_t i = 0; i < size_t(rows) * row_elems; ++i) buf[i] = Half(dist(rng));
  return buf;
}

template <class Traits>
double run_case(const Case& tc, unsigned seed) {
  constexpr int D = Traits::kHeadDim;
  constexpr int kGuardRows = 3;
  const int N = tc.seq_len, HQ = tc.num_q_heads, HKV = tc.num_kv_heads;
  const float scale = 1.0f / std::sqrt(float(D));

  std::mt19937 rng(seed);
  auto q = random_rows(rng, N, HQ * D, tc.input_range, kGuardRows);
  auto k = random_rows(rng, N, HKV * D, tc.input_range, kGuardRows);
  auto v = random_rows(rng, N, HKV * D, 1.0f, kGuardRows);
  std::vector<Half> out(size_t(N + kGuardRows) * HQ * D, kNaN);

  // Shared memory: one buffer per block, 16-byte aligned, NaN before each block
  // so that reading a tile before writing it is caught too.
  std::vector<Half> smem_storage(Traits::kSmemElems + 8);
  Half* smem = smem_storage.data();
  while (reinterpret_cast<std::uintptr_t>(smem) % 16 != 0) ++smem;

  const kapsl_emu::Uint3 grid{unsigned((N + Traits::kBlockM - 1) / Traits::kBlockM), unsigned(HQ), 1};
  kapsl_emu::launch(
      grid, Traits::kNThreads,
      [&] { std::fill(smem, smem + Traits::kSmemElems, kNaN); },
      [&] {
        kapsl::prefill_attention_block<Traits>(out.data(), q.data(), k.data(), v.data(),
                                                scale, N, HQ, HKV, smem);
      });

  // Reference: causal attention in double precision from the same f16 inputs.
  double max_err = 0.0;
  bool   finite  = true;
  std::vector<double> score(N), ref(D);
  for (int i = 0; i < N; ++i) {
    for (int h = 0; h < HQ; ++h) {
      const int   kvh  = h * HKV / HQ;
      const Half* qrow = &q[(size_t(i) * HQ + h) * D];
      double best = -INFINITY;
      for (int j = 0; j <= i; ++j) {
        const Half* krow = &k[(size_t(j) * HKV + kvh) * D];
        double dot = 0.0;
        for (int d = 0; d < D; ++d) dot += double(float(qrow[d])) * double(float(krow[d]));
        score[j] = dot * double(scale);
        best = std::max(best, score[j]);
      }
      double sum = 0.0;
      for (int j = 0; j <= i; ++j) sum += (score[j] = std::exp(score[j] - best));
      std::fill(ref.begin(), ref.end(), 0.0);
      for (int j = 0; j <= i; ++j) {
        const Half* vrow = &v[(size_t(j) * HKV + kvh) * D];
        for (int d = 0; d < D; ++d) ref[d] += score[j] / sum * double(float(vrow[d]));
      }
      const Half* orow = &out[(size_t(i) * HQ + h) * D];
      for (int d = 0; d < D; ++d) {
        const double got = double(float(orow[d]));
        if (!std::isfinite(got)) finite = false;
        max_err = std::max(max_err, std::fabs(got - ref[d]));
      }
    }
  }

  bool guard_intact = true;
  for (size_t i = size_t(N) * HQ * D; i < out.size(); ++i) guard_intact &= out[i].raw() == kNaN.raw();

  // Outputs are O(1) and stored as f16, with P rounded to f16 before P*V.
  constexpr double kTolerance = 4e-3;
  const bool ok = finite && guard_intact && max_err < kTolerance;
  std::printf("d=%-3d seq_len=%-4d heads=%d/%d range=%.0f : max_abs_err=%.2e %s%s%s\n",
              D, N, HQ, HKV, tc.input_range, max_err, ok ? "ok" : "FAILED",
              finite ? "" : " [non-finite output]", guard_intact ? "" : " [wrote past seq_len]");
  if (!ok) ++g_failures;
  return max_err;
}

template <int D>
void run_head_dim(const std::vector<Case>& cases) {
  using Traits = kapsl::PrefillTraits<D>;
  check_layout_assumptions<Traits>();
  unsigned seed = 1000 + D;
  for (const Case& tc : cases) run_case<Traits>(tc, seed++);
}

}  // namespace

int main(int argc, char** argv) {
  const bool quick = argc > 1 && std::string(argv[1]) == "--quick";

  // seq_len values straddle the 64-row block / 64-key tile boundaries.
  const std::vector<Case> full = {
      {1, 2, 2, 2},   {2, 2, 2, 2},   {17, 4, 2, 2},  {63, 2, 2, 2},
      {64, 2, 1, 2},  {65, 3, 1, 2},  {129, 4, 2, 2}, {200, 2, 2, 2},
      {150, 2, 1, 8},  // peaky softmax: large dynamic range across tiles
  };
  const std::vector<Case> brief = {{1, 2, 2, 2}, {65, 4, 2, 2}, {150, 2, 1, 8}};

  run_head_dim<64>(quick ? brief : full);
  run_head_dim<128>(brief);
  run_head_dim<32>(brief);
  run_head_dim<96>(brief);

  std::printf("%s\n", g_failures == 0 ? "ALL PASSED" : "FAILURES");
  return g_failures == 0 ? 0 : 1;
}
