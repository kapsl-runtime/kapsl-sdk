# Kapsl native backend ABI

`kapsl-backend-abi` is the stable C boundary between a Kapsl runtime and an
in-process native inference backend pack. It contains no backend implementation
and has no dependency on ONNX Runtime, llama.cpp, CUDA, or a Rust async runtime.

The backend-neutral `KapslBackendApiV1` contract covers:

- adapter discovery and capability negotiation;
- model planning, load, health, unload, and shutdown;
- borrowed numeric or UTF-8 tensor inputs and adapter-owned tensor results;
- batching, streaming, and cancellation when advertised;
- structured memory, metrics, model, batching, and optional KV reports;
- runtime-owned device allocations supplied through host callbacks.

Every table must advertise memory reporting and at least one execution target.
TensorRT tables also advertise CUDA, and governed-device-allocation tables may
only be used by CUDA-capable packs. Hosts reject contradictory tables before
adapter initialization.

The existing llama.cpp v1 declarations are preserved and re-exported for source
and binary-layout compatibility while native packs migrate to the neutral API.

## Ownership and lifetime

- Request tensors are borrowed only for the duration of the synchronous ABI
  call. A backend must not retain their pointers.
- Inference results belong to the backend until the host calls the matching
  `release_result` or `release_batch_result` function.
- JSON report and error buffers belong to the backend until the host calls the
  same function table's `free_buffer` function.
- The host callback table and its context remain valid until backend `shutdown`
  returns.
- Governed device allocations must be returned with the exact allocation ID and
  pointer supplied by the host.
- A backend must synchronize outstanding device work before unload or shutdown
  permits governed storage to be recycled.

No Rust-owned value, trait object, collection, future, or unwinding exception may
cross the ABI. Adapter entrypoints must catch panics and translate them to
`KAPSL_STATUS_PANIC`.

## ORT integration

An ORT backend pack remains in the Kapsl process. Its custom `OrtAllocator`
forwards allocation and free operations to `KapslBackendHostV1`, preserving the
runtime-owned `GpuDevicePool` path without RPC, CUDA IPC, or tensor serialization.
The ORT adapter implementation lives outside this crate and is independently
packaged and released.

## Release policy

Every release is built and tested from an exact `kapsl-backend-abi-vX.Y.Z`
stable tag whose commit is already present on `main`. The crates.io publication
job is protected by the `crates-io-backend-abi` GitHub environment. Branch,
beta, release-candidate, and mismatched tags cannot publish this crate.
