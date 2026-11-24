# GGUF memory accounting

How `kapsl-llm`'s GGUF backend tells the runtime's memory authority what a
loaded model really holds on the device, and how it plans that before loading.

## What a loaded GGUF model holds

| Part | Planned from | Reported as (after load) |
| --- | --- | --- |
| Weights | GGUF file size | `PersistentWeights` per device, `LlamaModel::device_memory()` |
| KV cache | layers x kv heads x head dim x 2 x f16 x total context | `KvCache` |
| Compute buffer | `estimate_compute_buffer_bytes` (below) | `TransientWorkspace` per device, `LlamaContext::device_compute_memory()` |

The compute buffer is llama.cpp's scheduler graph buffer. It is allocated per
context on top of weights and KV, and is not small: about 0.6 GiB for a
Qwen2.5 model at the default context settings.

It is final only once the scheduler has been reserved. llama.cpp reserves in the
context constructor and again lazily on the first decode whenever the sampler
set changed, and the backend installs a sampler per sequence slot. The
scheduler thread therefore calls `LlamaContext::reserve_compute_buffers()`
(`llama_context_sched_reserve`) after installing samplers and publishes the
result before the model reports ready, so the first report is the steady state.

Both C entry points are in the kapsl llama.cpp fork
(`third_party/llama.cpp-kapsl`): `llama_context_device_compute_memory` and
`llama_context_sched_reserve`. The allocation id of the reported row is the same
`gguf-scratch:<model path>` id used at planning time, so the engine reconciles
the planned row to the measured one.

## Planning formula

```
n_ubatch = prefill_chunk + max_concurrent        (GgufServingConfig::n_batch)
n_ctx    = max_concurrent * ctx_per_seq

raw     = 8 * n_vocab * n_ubatch
        + 4 * (n_embd + n_ff) * n_ubatch
        + 256 * n_ctx
planned = max(raw * 17/16, 64 MiB)
```

* `8 * n_vocab * n_ubatch`: with backend sampling the reserve graph holds the
  f32 logits for every micro-batch row plus the sampler's working copy. This is
  more than 98% of the buffer for the Qwen2.5 family (151,936-token vocabulary).
* `4 * (n_embd + n_ff) * n_ubatch`: an upper bound for one hidden and one FFN
  activation row per token.
* `256 * n_ctx`: attention mask and position bookkeeping, fitted (~230 B/cell).
* `17/16`: 6.25% headroom so the estimate stays an upper bound across backends.

`n_vocab`, `n_embd` and `n_ff` come from the GGUF header
(`vocab_size`, `hidden_size`, `intermediate_size`). If the header cannot be read,
planning falls back to the old `max(file_size / 8, 256 MiB)` rather than failing.

Env knobs that change the inputs: `KAPSL_GGUF_MAX_CONCURRENT`,
`KAPSL_GGUF_CTX_PER_SEQ`, `KAPSL_GGUF_PREFILL_CHUNK_SIZE`.

## Error against measurements

Measured on Metal (kapsl llama.cpp fork, Qwen2.5 Q4_K_M). Planned is always at
or above measured; "raw" is the formula before the 17/16 margin.

| Model | n_ubatch | n_ctx | Measured MiB | Raw | Planned MiB | Planned vs measured |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.5B | 544 | 65536 | 637.3 | +3.3% | 699.7 | +9.8% |
| 0.5B | 520 | 16384 | 609.2 | +1.5% | 656.8 | +7.8% |
| 0.5B | 514 | 4096 | 602.2 | +1.0% | 646.1 | +7.3% |
| 0.5B | 136 | 16384 | 160.3 | +2.7% | 174.9 | +9.1% |
| 0.5B | 1032 | 16384 | 1208.5 | +1.2% | 1299.4 | +7.5% |
| 0.5B | 520 | 4096 | 607.9 | +1.2% | 653.6 | +7.5% |
| 0.5B | 520 | 65536 | 621.4 | +1.4% | 669.6 | +7.8% |
| 3B | 544 | 65536 | 639.7 | +5.3% | 715.8 | +11.9% |
| 3B | 520 | 16384 | 611.5 | +3.5% | 672.2 | +9.9% |
| 3B | 514 | 4096 | 604.4 | +3.0% | 661.3 | +9.4% |
| 3B | 136 | 16384 | 160.9 | +4.7% | 178.9 | +11.2% |
| 3B | 1032 | 16384 | 1213.0 | +3.2% | 1329.9 | +9.6% |
| 3B | 520 | 4096 | 612.5 | +2.8% | 669.0 | +9.2% |
| 3B | 520 | 65536 | 623.7 | +3.4% | 685.0 | +9.8% |

Planned is +7.3% to +11.9% above measured across these rows. The 0.5B and 3B
models differ by under 1% in measured size because they share a vocabulary; the
3B overshoot is the activation upper bound.

One CUDA point exists: an RTX 3090 with the 0.5B model measured 616.26 MiB. The
run's `KAPSL_GGUF_*` settings were not recorded; under the defaults the raw
formula is +6.9% and the planned value +13.5%. No CUDA measurement of the 3B
model exists yet.

## Re-measuring

`crates/kapsl-llm/tests/gguf_memory_probe.rs` loads a GGUF through the real
backend, prints planned and actual rows, and asserts planned scratch covers the
actual compute buffer:

```bash
KAPSL_PROBE_GGUF=/path/model.gguf cargo test -p kapsl-llm --features gguf \
  --test gguf_memory_probe -- --ignored --nocapture
```

On a CUDA build also confirm the reported `CUDA0` value equals llama.cpp's own
`CUDA0 compute buffer size` log line, which rules out pinned host buffers being
counted as device memory. Update `COMPUTE_BUFFER_MEASUREMENTS` in
`gguf/backend.rs` with any new data.

## Logging

llama.cpp/ggml output is routed through `tracing`
(`llama_cpp_2::send_logs_to_tracing`) when the backend initialises, so a host
subscriber's filter (`RUST_LOG`) applies to it. Set
`KAPSL_GGUF_NATIVE_LLAMA_LOGS=1` to keep llama.cpp's own unfiltered stderr
logging, for example in a host that installs no `tracing` subscriber.

## Where each row is counted (placement)

Rows are counted where the memory physically lives, by `gguf/placement.rs`:

| llama.cpp device | domain |
| --- | --- |
| CPU, accelerator, integrated GPU, or any device reporting `unified_memory` (Apple Silicon Metal, Jetson-class CUDA) | `Host` |
| CUDA GPU matching exactly one engine CUDA device by **PCI bus id** | `Cuda{engine id}` |
| ROCm/HIP, Vulkan discrete, SYCL, OpenCL, MUSA | `Provider{<backend>, ordinal within the backend}` (accounting only) |
| anything that cannot be mapped (no PCI match, two engine devices with one PCI id, unknown type) | `Provider{"unattributed"}` plus a warning, never `cuda:0` |

Per device the backend reports weights (`llama_model_device_memory`), KV cache
(`llama_context_device_context_memory`) and compute buffers
(`llama_context_device_compute_memory`); buffers that live on no listed device
(CPU weights, KV of layers kept on the CPU by partial offload, host compute
buffers) come from `llama_context_host_memory`. Rows with the same domain and
class are merged; a class split over several domains uses the base allocation id
for the first domain (by name) and `base@domain` for the others.

Planning uses the same rule: the offloaded fraction `n_gpu_layers / n_layers`
of weights, KV and scratch goes to the primary device's domain, the rest to
`Host`. The host share of the scratch carries 25% headroom because the estimate
above is fitted to device-resident compute buffers and CPU compute buffers
measured 1.10x to 1.18x larger (Qwen2.5-0.5B, `KAPSL_GGUF_N_GPU_LAYERS=0`).

The engine's inventory is registered with `register_engine_devices` (done by
the backend factory); without it only a lone CUDA device is mapped.

Known limits: the fork reports SYCL and OpenCL devices as discrete `GPU` with no
PCI id and no `unified_memory`, so an Intel integrated GPU used through SYCL is
counted as `Provider{sycl, n}` instead of `Host`; the CUDA, HIP, MUSA and Vulkan
edits that feed `unified_memory` have not been compiled outside CPU and Metal.
