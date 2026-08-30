# llama.cpp Shared KV Pool Fork Plan

> Status: the integration is implemented and the application-level `cuda`
> profile now enables `gguf-cuda-shared-kv`. The explicit `gguf-cuda` feature
> remains the static-KV rollback. The milestones below retain the original
> rollout context and should not be read as the current feature-selection state.
> The backend-neutral contract and fork exit strategy now live in
> [Backend-neutral KV Integration](./backend-kv-integration.md).

This is the integration contract for making llama.cpp use Kapsl's GPU-wide KV
pool instead of allocating a private, fixed-size KV cache per context.

## Goal

Keep llama.cpp's optimized GGUF decode path, CUDA kernels, tokenizer, sampler,
and quantized weight handling, but replace its static per-context KV allocation
with Kapsl-owned paged KV blocks.

The target advantage over vLLM is not single-model attention math. The target is
fleet-level memory utilization: many GGUF models and sessions sharing one GPU
KV budget with dynamic admission, eviction, restore, and prefix reuse.

## Current Kapsl Surface

Kapsl already has the core pool pieces in `kapsl-hal`:

- `GpuBlockPool`: owns device storage with layout
  `[num_blocks, 2, num_kv_heads, block_size, head_dim]` in f16.
- `CrossDevicePoolScheduler`: GPU-wide admission, migration, CPU offload, and
  per-geometry pool registration.
- `PrefixBlockCache`: refcounted prompt-prefix KV block reuse.

The llama.cpp fork must consume this existing block table and storage. It should
not introduce a competing allocator in C++.

## Fork Strategy

Do not start by rewriting llama.cpp attention or replacing all graph code.
Implement a new memory backend inside llama.cpp:

```text
llama_memory_i
  ├─ llama_kv_cache          existing static KV cache
  ├─ llama_kv_cache_iswa     existing iSWA KV cache
  └─ llama_kv_cache_kapsl    new external paged KV cache
```

Recent llama.cpp already routes decode through `llama_memory_i`, and
`llama_kv_cache` is the existing implementation to mirror. The fork should add a
Kapsl implementation that preserves the same graph-facing behavior while its
physical storage comes from Kapsl.

## Proposed C ABI

Add an optional external KV pool descriptor to `include/llama.h`.

```c
struct llama_kapsl_kv_pool_desc {
    void * user_data;

    uint32_t device_id;
    uint32_t block_size;
    uint32_t num_blocks;
    uint32_t num_kv_heads;
    uint32_t head_dim;
    uint32_t dtype;        // start with f16 only

    void * device_base;    // base pointer for [block, kv, head, token, dim]
    uint32_t * block_table_device;
    uint32_t block_table_stride;

    bool (*reserve)(
        void * user_data,
        uint64_t session_id,
        uint32_t tokens_needed,
        uint32_t ** block_table_device_out,
        uint32_t * n_blocks_out);

    void (*release)(
        void * user_data,
        uint64_t session_id);

    bool (*touch)(
        void * user_data,
        uint64_t session_id);
};
```

Then extend `llama_context_params` with:

```c
struct llama_kapsl_kv_pool_desc * kapsl_kv_pool;
uint64_t kapsl_session_id;
```

If `kapsl_kv_pool == NULL`, llama.cpp keeps its current behavior. If it is set,
context creation chooses `llama_kv_cache_kapsl`.

## llama.cpp Fork Work

Expected files in the fork:

- `include/llama.h`: add the Kapsl KV descriptor and context params.
- `src/llama-model.cpp`: select `llama_kv_cache_kapsl` from
  `llama_model::create_memory`.
- `src/llama-kv-cache-kapsl.h/.cpp`: implement `llama_memory_i`.
- `src/llama-context.cpp`: wire params through context construction and memory
  updates.
- `ggml/src/ggml-cuda/*`: update CUDA attention/KV-copy kernels only where they
  currently assume contiguous KV cells.

Current local fork progress:

- Added `llama_kapsl_kv_pool_desc` and `kapsl_session_id` to
  `llama_context_params`.
- Added `llama_kv_cache_kapsl` as a new `llama_memory_i` backend selected when
  `kapsl_kv_pool` is set.
- Added reserve/release callback plumbing so llama.cpp can request Kapsl KV
  blocks and receive a device block table.
- Added `ggml_backend_cuda_buffer_from_device_ptr` so ggml can wrap
  Kapsl-owned CUDA memory without freeing it.
- Added `GpuBlockPool::device_base_ptr()` on the Kapsl side for the FFI
  descriptor.
- Refactored attention graph construction to depend on
  `llama_kv_cache_graph_context` instead of concrete `llama_kv_cache_context`,
  so Kapsl can provide a graph-facing KV context without unsafe concrete casts.
- Added a non-owning ggml CUDA buffer bridge inside
  `llama_kv_cache_kapsl_context`, allowing Kapsl-owned pool memory to be
  represented as physical K/V tensors without transferring ownership to ggml.
- Tightened the ABI so the device block table is explicitly layer-strided:
  `[n_layers, max_blocks_per_seq]`.
- Added `GGML_OP_KAPSL_KV_WRITE`, a CUDA-only ggml op that writes current layer
  K/V tensors into the Kapsl paged pool via the physical block table.
- Wired `llama_kv_cache_kapsl_context::cpy_k/cpy_v` to emit the new write op
  and `build_input_*_idxs` to pass logical token positions into that op.
- Added `GGML_OP_KAPSL_PAGED_ATTN`, a correctness-first CUDA paged-attention op
  that reads K/V directly from Kapsl physical blocks and returns the normal
  attention output shape for llama.cpp's output projection.
- Changed `llama_kv_cache_kapsl::init_batch()` to split ubatches, reserve by
  highest logical token position, and return `LLAMA_MEMORY_STATUS_SUCCESS` for
  the supported path.

The remaining blocker is CUDA validation and optimization. The first paged
attention kernel is intentionally naive and serializes each output element over
the context length. It is for correctness smoke testing only; after cluster CUDA
build validation, replace it with a tiled/flash-style paged kernel before using
it for throughput comparisons.

Phase 1 should support only:

- CUDA.
- f16 KV.
- Single GPU.
- Causal decoder-only models.
- No CPU restore inside llama.cpp.
- No prefix reuse inside llama.cpp.

Kapsl remains responsible for reserve, release, migration, prefix cache policy,
and CPU offload. The fork only reads and writes the physical blocks it is given.

### Session isolation

Runtime-owned KV allocations are synchronously zeroed before ownership is
published, so a recycled physical extent does not expose the previous owner's
cache contents. Cross-session prefix reuse is disabled by default because its
live blocks are keyed by model and token hashes rather than an authenticated
security domain. A trusted single-tenant deployment may explicitly opt in with
`KAPSL_GGUF_ALLOW_CROSS_SESSION_PREFIX_CACHE=1` and optionally size it with
`KAPSL_GGUF_PREFIX_CACHE_BLOCKS`.

## Kapsl Runtime Work

Add a new backend feature rather than changing the existing baseline:

```text
gguf-cuda             current llama.cpp-backed static KV path
gguf-cuda-shared-kv   llama.cpp fork + Kapsl KV pool
gguf-native           Kapsl CUDA kernels and Kapsl-owned KV
```

Expected Kapsl files:

- `crates/kapsl-hal/src/memory/gpu_arena.rs`: expose stable device base pointer
  and block geometry for FFI.
- `crates/kapsl-hal/src/memory/cross_device_scheduler.rs`: provide FFI-safe
  reserve, release, and touch wrappers around existing scheduler operations.
- `crates/kapsl-llm/src/gguf_backend.rs`: create llama contexts with
  `llama_kapsl_kv_pool_desc` when `gguf-cuda-shared-kv` is enabled.
- `crates/kapsl-backends/Cargo.toml` and `crates/kapsl-llm/Cargo.toml`: patch
  `llama-cpp-2` / `llama-cpp-sys-2` to the forked source.

## Milestones

1. Static external pool smoke test:
   llama.cpp uses Kapsl-allocated KV blocks for one model/session. No dynamic
   eviction. Output must match the existing `gguf-cuda` backend for greedy
   deterministic prompts.

2. Dynamic reserve/free:
   multiple llama.cpp contexts allocate from one Kapsl `GpuBlockPool`; finished
   sessions release blocks immediately. No per-context KV preallocation remains.

3. Scheduler admission:
   route requests through `CrossDevicePoolScheduler`, enforce soft/hard KV
   limits, and expose pool metrics.

4. Prefix reuse:
   integrate `PrefixBlockCache` so shared complete prompt blocks are borrowed
   instead of recomputed.

5. CPU offload/restore:
   let inactive sessions move to `CpuBlockStore` and restore to GPU on demand.

6. Multi-GPU:
   enable cross-device placement and migration once single-GPU throughput is
   within target.

## Acceptance Criteria

- Correctness: greedy deterministic output matches current llama.cpp for the
  same GGUF, seed, prompt, and sampling params.
- Throughput: single-model decode is within 5-10% of existing llama.cpp CUDA
  before enabling eviction or prefix reuse.
- Memory: no fixed `n_ctx` KV allocation per context in shared-KV mode.
- Utilization: idle sessions/models return blocks to the GPU-wide pool.
- Isolation: existing `gguf-cuda` remains available as the rollback backend.

## Build Integration

The cluster build for this path can select the shared-KV feature directly:

```bash
cargo build -p kapsl --release --no-default-features --features gguf-cuda-shared-kv
```

The separate native-kernel GGUF backend is built with:

```bash
cargo build -p kapsl --release --no-default-features --features gguf-native
```

The stable `cuda` feature maps to `gguf-cuda-shared-kv`. Build with the
explicit `gguf-cuda` feature to use the static-KV rollback backend; it does not
map to `gguf-native`.
