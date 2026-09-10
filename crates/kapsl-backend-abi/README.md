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

## Optional host contracts

`KapslBackendHostExtensionsV1` appends exact name/version discovery after the
published host and scoped-allocator prefixes. The prefixes keep their layout,
ABI version and behavior. Adapters check `struct_size` before reading the tail,
then validate the queried table's own size, version and callbacks. Unsupported
contracts return null; the host never substitutes another version. Tables and
contexts remain borrowed through adapter shutdown.

`kapsl-kv-abi` publishes the `kapsl-kv-native` extension for shared KV. This
lookup mechanism contains no KV layout, backend name or backend launch policy.
CPU hosts may expose extension discovery without device allocator callbacks;
adapters independently require the capabilities they actually use.

The separate `kapsl-managed-abi` crate defines lifecycle and inference for
supervised processes. Neither protocol uses KV messages to launch backends.

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

Because crates.io cannot configure a trusted publisher before a crate exists,
the `0.1.0` job is a one-time bootstrap guarded by that environment and its API
token. After `0.1.0` is published, register `kapsl-runtime/kapsl-sdk`,
`publish-backend-abi.yml`, and the `crates-io-backend-abi` environment as the
crate's trusted publisher. Every later stable release obtains a short-lived
crates.io credential through GitHub OIDC; the bootstrap job cannot publish any
version other than `0.1.0`.
