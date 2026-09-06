# Python SDK compatibility audit

Checked on 2026-09-07 against SDK main `6b511d6`, the gRPC release commit
`6742576`, and engine integration commit `e3223df`.

The Python SDK is behind both the current SDK source and the new gRPC API.
Updating its version alone will not provide compatibility across all transports.

## Published package and source

- [PyPI kapsl-sdk 0.1.23](https://pypi.org/project/kapsl-sdk/0.1.23/) was uploaded
  on 2026-04-08. This was the latest PyPI release when checked.
- Both checked-in Python manifests still declare `0.1.16`. The Python release
  workflow substitutes the version from `v*` release tags, so this is metadata
  drift rather than evidence that the PyPI version should be downgraded.
- The published macOS ARM64 wheel was installed and imported under Python 3.14.
  Its actual exported classes and method signatures were inspected. The source
  distribution was also compared with current Rust/PyO3 source.
- Rust crate versions and Python package versions are independent. The gap is
  in implementation and wire compatibility, not just their version numbers.

## Feature coverage

| Capability | PyPI 0.1.23 | Current Python source |
| --- | --- | --- |
| Native socket/TCP inference and streaming | Exposed | Exposed, using the shared transport codec and releasing the GIL during blocking operations |
| Tensor output metadata, named extra inputs, session IDs | Exposed by `KapslClient` | Exposed by `KapslClient` |
| Native token authentication | Constructor accepts a token; authenticated streaming fails against the checked engine | Uses the current request metadata layout; a new wheel has not been published |
| SHM/hybrid model selection and concurrent allocation | Old layout; model ID is fixed to zero | Adds `model_id`, allocation leases, and direct-SHM response mailboxes |
| gRPC discovery, unary inference, and typed server streaming | Absent | Absent from the Python package; separate proto generation and examples are available |
| Per-request timeout, priority, force-CPU, and generation overrides | No Python arguments | No Python arguments |
| Explicit stream `cancel()` / `close()`, async iteration, typed stream packets | Absent; synchronous iteration returns packet data | Absent; synchronous iteration returns packet data |
| Health, model discovery, and governance control helpers | Absent | Absent |

Scheduling and memory governance run inside the engine for requests that reach
its governed serving path. Python does not need to duplicate that policy.
Dedicated Python helpers are useful for inspecting or controlling the runtime,
and for exposing transport options that the server already supports.

## Confirmed native authentication incompatibility

The installed PyPI wheel was tested against an isolated, model-free runtime
with a loopback native TCP listener and `KAPSL_TCP_AUTH_TOKEN` configured.
Calling `infer_stream` with the matching `api_token` returned `Unauthorized`.

Capturing that request and decoding it with the exact published
`kapsl-engine-api 0.3.0` and `kapsl-transport 0.3.0` used by the engine confirmed:

1. Strict decoding as the current `InferenceRequest` fails with
   `io error: unexpected end of file`.
2. The compatibility decoder accepts the pre-metadata request prefix and
   returns `metadata: None`, discarding the provided authentication token.
3. The old metadata layout lacks `min_new_tokens`. Re-encoding the same request
   with current metadata adds the missing optional field.
4. Replaying the current encoding with the same token passes authentication and
   reaches model lookup, returning the expected `Model 987654 not found`.

Relevant source:

- [Current request metadata](../crates/kapsl-engine-api/src/lib.rs)
- [Transport compatibility decoder](../crates/kapsl-transport/src/protocol.rs)
- [Current Python request construction](../crates/kapsl-pyo3/src/client.rs)

This failure concerns the native framed transport. The new gRPC API sends
credentials in request metadata and uses its shared engine authorization
adapter. Its authorization and streaming tests pass.

The model-free probe verifies decoding and authorization, not successful model
inference. The native unary path looks up the model before checking credentials,
so a unary `Model not found` result alone is not proof that authentication works.

## SHM release coordination

Current Python source uses the new shared-memory allocation and mailbox
contracts. The engine branch still resolves the published `kapsl-ipc 0.3.0` and
`kapsl-shm 0.3.0`, whose request layouts predate the lease, protocol-version,
and mailbox fields. These source changes have not been published as updated
transport crates. `kapsl-communication` is also not yet published on crates.io.

A Python rebuild from main must therefore be coordinated with compatible
transport crate releases and engine dependency updates. Publishing the current
Python source by itself would not synchronize SHM/hybrid clients and servers.
SHM/hybrid interoperability was assessed from source layouts, not exercised
with a live shared-memory model in this audit.

## Recommended follow-up

1. Fix and regression-test native metadata compatibility using requests from
   the released Python wheel. Preserve credentials and policy fields through
   supported legacy decoders; avoid silently discarding metadata.
2. Release the matching transport/SHM changes and update the engine together
   with new Python wheels. Test native, SHM, and hybrid client/server pairs.
3. Package the generated gRPC clients behind an optional Python dependency,
   with token/TLS configuration, deadlines, cancellation, and streaming helpers.
4. Add Python request options for the supported scheduler and generation
   controls, then publish a new Python version with wheel-level compatibility
   checks against supported engine releases.

This audit did not modify or publish the Python package. The separate
[`kapsl-grpc 0.3.0` release](https://crates.io/crates/kapsl-grpc/0.3.0) was
published and the engine now resolves it from crates.io. The packaged crate's
18 tests and the engine's 452 tests passed after that dependency change.
