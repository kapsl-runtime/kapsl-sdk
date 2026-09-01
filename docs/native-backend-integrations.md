# Native backend integrations

Kapsl Engine is the memory-governance and orchestration layer. Backend-specific
execution belongs behind versioned integration contracts and is released
independently from the engine.

## Repository boundary

The target ownership model is:

```text
kapsl-engine
  routing, scheduling, model lifecycle, memory authority, signed-pack loader

kapsl-sdk
  kapsl-backend-abi, kapsl-kv-abi, shared clients and public contract fixtures

kapsl-integrations
  integrations/ort, integrations/vllm, integrations/sglang,
  integrations/llama-cpp, backend packaging and conformance drivers
```

`kapsl-integrations` is one integration monorepo, not one repository per
backend. Engine and integration releases consume published SDK packages. They
must not depend on sibling checkouts, Git submodules, or committed Cargo path
patches.

## Two independent contracts

`kapsl-backend-abi` is the native inference lifecycle: discovery, planning,
load, inference, cancellation, reporting, unload, and governed device-allocation
callbacks. `kapsl-kv-abi` is the optional KV participant contract used only by
backends with cache state to govern.

An ORT image classifier implements the native backend ABI without claiming a KV
tier. An autoregressive ORT adapter may implement both contracts. Python-native
servers such as vLLM generally use an out-of-process inference boundary and the
transport-neutral KV control client instead of the native C function table.

## ORT runtime topology

Moving ORT to `kapsl-integrations` changes source, build, package, and release
ownership. It does not move ORT out of the Kapsl process:

```text
Kapsl GpuDevicePool
        |
        | KapslBackendHostV1 allocation callbacks
        v
in-process ORT adapter -> OrtAllocator -> ORT execution provider
```

Bulk tensors cross the plugin boundary as borrowed views. JSON is limited to
configuration and small reports. The governed profile registers the host-backed
allocator before constructing any ORT session and enables the environment
allocator on those sessions. The adapter library stays loaded until all model
instances, tensors, sessions, and allocator registrations are gone.

An optional out-of-process ORT worker is a later isolation mode. It would need a
separate CUDA IPC/VMM design and is not part of the extraction migration.

## Published LLM provider features

The published `kapsl-llm` crate exposes `onnx` as the portable CPU generation
contract. Integration packs must disable default features and select exactly
one reviewed profile: `onnx`, `onnx-cuda`, `onnx-tensorrt`, `onnx-coreml`,
`onnx-directml`, `onnx-openvino`, or `onnx-rocm`. `onnx-cuda-pool` includes
`onnx-cuda` and the Kapsl allocator hooks.

The crate's default feature set retains the target-neutral historical provider
set for embedded callers during migration. DirectML is an explicit Windows-only
feature because enabling it in a cross-platform default makes Linux binaries
fail to link. The default is not a pack profile. An external adapter must never
inherit it and then claim a CPU-only artifact.

## Migration sequence

1. Capture the embedded ORT functional, memory, and performance baseline.
2. Publish the backend-neutral ABI and cross-language contract from `kapsl-sdk`.
3. Create and secure `kapsl-integrations` with independent release workflows.
4. Move the existing vLLM package without behavior changes to prove repository
   packaging and release plumbing.
5. Add one generic native-pack host to `kapsl-engine`; do not add an ORT-specific
   second plugin loader.
6. Implement ORT CPU forward execution in `kapsl-integrations` and prove output,
   batching, load/unload, and request-memory parity.
7. Move embedding, classification, detection, transcription, and their
   backend-specific tensor preparation and postprocessing.
8. Move the custom CUDA allocator into the adapter and forward it to the
   engine-owned device pool through the native host callbacks.
9. Add separately identified CUDA, TensorRT, and ONNX-generation profiles.
10. Switch signed backend selection to the adapter after parity passes, retain
    an explicit rollback for one stable cycle, then remove embedded ORT code and
    the engine's direct `ort` dependency.
11. Move llama.cpp integration work and add SGLang in the same integrations
    repository.

Do not combine the extraction with an ORT, CUDA, TensorRT, model-layout, or
performance-tuning upgrade.

## Baseline and acceptance gates

The initial host baseline on the SDK `develop` lineage is:

```text
KAPSL_LLAMA_CPP_DIR=third_party/llama.cpp-kapsl \
  cargo test -p kapsl-backends --lib onnx

72 passed; 0 failed

KAPSL_LLAMA_CPP_DIR=third_party/llama.cpp-kapsl \
  cargo test -p kapsl-llm --lib engine::engine_tests::tests

23 passed; 0 failed
```

Before the adapter becomes the default, compare the embedded and packed paths
with identical model artifacts and configuration. Require:

- equivalent outputs within existing numerical tolerances;
- no full tensor encoded through JSON or Base64;
- no additional host/device transfer;
- equivalent batching and session counts;
- no statistically meaningful throughput or latency regression;
- the same Kapsl-owned allocator coverage and bounded provider-private memory;
- correct concurrent pressure with another governed backend;
- complete memory reclamation after unload and reload;
- device-resident `past.*` and `present.*` state for ONNX generation.

Real GPU conformance runs only in an official stable-release workflow, never on
ordinary pull requests, branch pushes, beta tags, or prereleases. Cleanup runs
unconditionally and the release verifies removal of its JIT runner, VM, boot
disk, and firewall before publishing or allowlisting a profile.

## Failure and rollback policy

- ABI, profile, provider, or allocator mismatch fails before model activation.
- CUDA and TensorRT profiles never silently fall back to CPU.
- A governed adapter never silently falls back to an ungoverned allocator.
- Embedded and packed ORT runtimes are not loaded in the same process.
- Rollback selects the previous signed pack or previous engine release; it does
  not weaken the active deployment's memory-governance requirement.
