# Managed backend protocol v1

`kapsl-managed-v1` is the versioned lifecycle and inference boundary between
the engine and a supervised integration adapter. This crate contains the Rust
wire definitions and a bounded codec. It contains no backend compute or launch
implementation. Other languages implement the same JSON protocol.

The engine selects and verifies a signed pack, launches its executable
entrypoint directly with the argument `--kapsl-managed-protocol=1`, and owns its
process group, deadlines, restart policy and memory admission. The adapter owns
language/runtime discovery, backend command construction, configuration
translation, dependencies and any child backend processes. Every child stays
in the supervised process group. The signed pack must be self-contained for
offline execution; startup must not download mutable dependencies.

## Transport and identity

Commands use stdin; responses and stream events use stdout. Each frame is one
UTF-8 JSON object followed by LF, at most 8 MiB including LF. Logs use stderr.
Writers serialize whole frames so concurrent operations cannot interleave
bytes. Readers reject partial, malformed, oversized and unknown-version
frames. EOF between frames means the adapter has exited, not successful
completion of pending requests. Backpressure must not make cancellation or
shutdown unreachable: adapters keep their command reader responsive while
inference and streaming are running.

Every envelope contains `protocol_version: 1`, an engine-issued `owner`
(`instance_id`, `model_id`, `replica_id`), a nonzero command `id`, and `body`.
One process instance belongs to exactly one model/replica. The nonce changes
on every process start. Command IDs and inference request IDs are never reused
within that instance. All integers are exact; implementations must not round
64-bit IDs through floating-point numbers.

Responses repeat the complete owner and command ID. Hosts reject foreign or
stale messages before delivering output or changing accounting. Batch and
stream responses also repeat the engine-issued inference request IDs. An
adapter cannot authorize work by choosing its own IDs.

## Discovery and capabilities

`describe` returns `Descriptor`, including the protocol/schema version,
`execution_mode: "managed"`, backend/profile/version, formats, model types,
tasks, capabilities and implemented methods. The host compares these with the
selected signed descriptor before initialization. Required lifecycle, inference,
memory, reporting and shutdown methods must all be present. Optional batching,
streaming, cancellation and KV methods must agree exactly with capabilities.
Streaming requires cancellation. An advertised method returning `unsupported`
is a conformance failure. An unsupported explicit backend pin fails selection;
the host never substitutes another adapter or embedded backend.

The descriptor is an assertion to verify, not qualification by itself. Host
tests must load the actual adapter process and exercise every advertised
method, including failure and cleanup behavior.

## Lifecycle

The legal lifecycle is:

```text
spawn → describe → initialize → planned_memory → admit → load → inference
                                                        ↑          │
                                                        └─ unload ─┘
                                      any state → shutdown → process exit
```

`initialize` binds the signed backend/profile/version, assigned devices,
manifest and opaque adapter options. It may prepare control state but must not
allocate model/device memory. The engine admits process startup memory before
launch. `planned_memory` and `planned_request_memory` are read-only; they must
not allocate the resources they are estimating.

`load` requires an engine-issued model admission grant. A successful reply
means the model is usable and its live memory is reportable. Loading an already
loaded instance fails. Inference before load or after unload fails. Health and
reporting remain callable during recovery.

Unload waits for inference, cancellation and device work to finish. Its success
acknowledgment means model/request resources and mappings have been released;
the process can load again using new grant and request IDs. Failed load or
unload leaves the instance unavailable for inference until cleanup succeeds.
Failures must never report successful reclamation. `shutdown` is valid even
after failed initialization/load and drains all owned children/resources before
acknowledgment and process exit.

The engine retains grants until corresponding operations are quiescent. If an
adapter hangs, crashes or violates the protocol, supervision terminates its
entire process group, verifies termination, and completes required device/IPC
synchronization before reclaiming memory. A timeout alone is not proof that
memory is free. Restarts use a new instance nonce and require fresh admission.

## Inference, admission and cancellation

Tensor requests, results, memory reports, model information and metrics use the
JSON representations published in `kapsl-engine-api 0.3.0`. Tensor bytes use its
base64 encoding. Public OpenAI wire requests are an optional capability, with
translation owned by the adapter. Public credentials never enter this private
protocol. Oversized tensor messages fail explicitly; they are not silently
split or routed through another backend.

Each infer/stream operation carries a request ID and engine-issued admission
grant. A batch grant lists exactly its distinct request IDs. Model grants have
no request IDs. Grants contain per-allocation limits; only the engine may mint,
resize, transfer or release them. The adapter must reject missing or mismatched
grants and remain within admitted limits. Planned/live/request reports retain
allocation identities and domains. Engine accounting and observed process/device
usage remain authoritative; an adapter reporting fewer bytes does not free an
allocation or an IPC mapping.

Batch results preserve input order and request IDs. The host may overlap calls
only when `concurrent_inference` is advertised. Cancellation remains callable
concurrently even for adapters that serialize inference.

`cancel` names distinct live request IDs. The adapter signals the matching work
and returns `cancelled` only after that work is quiescent. Each original
operation must also produce its terminal response; cancellation acknowledgment
does not authorize the host to discard or release another request's resources.
Cancelling an already completed request is idempotent. An unknown/foreign ID
fails, and no new allocation may be attributed to cancelled work.

## Streaming

Each `stream_chunk` repeats the request ID and a sequence number starting at
zero. Sequence numbers increase by exactly one. A `stream_end` terminates the
operation and contains `next_sequence`, the next unused value, and optional
failure. No chunk or second terminal event may follow it. A pre-dispatch error
may use `error`; after any chunk, termination uses `stream_end`.

Tensor streams emit tensor outputs. OpenAI wire streams emit exactly one
`openai_head` followed by zero or more `openai_chunk` byte payloads. The host
validates response headers/statuses using the published OpenAI wire contract.
Consumer drop triggers cancellation and drains the terminal exchange before
releasing admission. The command reader must keep processing cancellation
while the output writer is under backpressure.

## KV and shared memory

KV participation is negotiated independently through `kapsl-kv-abi` and the
engine-supplied `kv_connection`. Its registration, grants, leases, epochs,
shared-memory/CUDA IPC mappings, resize, completion and detach rules continue
to apply. This process protocol does not replace those rules or use KV messages
to launch backends or carry general inference. Shared KV stays in its existing
data plane; tensor/metadata frames do not serialize KV contents.

## Method results

| Command | Successful response |
| --- | --- |
| `describe` | `descriptor` |
| `initialize`, `load`, `unload`, `shutdown` | `ack` |
| `planned_memory`, `planned_request_memory`, `actual_memory` | `memory` |
| `infer` | `result` |
| `infer_batch` | `batch_result` |
| `infer_stream` | zero or more `stream_chunk`, then `stream_end` |
| `cancel` | `cancelled` |
| `metrics`, `model_info`, `batching_policy` | corresponding report |
| `kv_capabilities`, `kv_topology` | corresponding KV report |
| `health` | `health` |

Every command may return a structured `error`. Receivers validate response kind,
correlation, request ownership and lifecycle; syntactically valid JSON alone is
insufficient. Unknown methods/versions fail explicitly, with no fallback.

## Release and conformance

Publish this contract from an immutable stable SDK tag before engine or
integration consumers select its exact released version. Protocol changes
require a new negotiated version when existing messages or behavior change.
PR conformance is host-only: no GPU provisioning or performance thresholds.
Engine neutrality additionally requires signed fake native and managed packs
to install, select, load, exercise and unload using an existing engine binary.

Publication uses `publish-host-contracts.yml` with a matching
`kapsl-managed-abi-vX.Y.Z` tag already merged into `main`, under the protected
`crates-io-backend-abi` environment. Version 0.1.0 has a one-time token bootstrap
because crates.io cannot configure a trusted publisher before the crate exists.
After that publication, register this repository, workflow and environment as
the crate's trusted publisher; later versions use OIDC. No branch, mismatched
tag or prerelease can publish.
