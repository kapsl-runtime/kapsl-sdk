# Backend-neutral KV integration

Kapsl's backend boundary has two independent parts:

```text
Kapsl backend = inference adapter + versioned KV participant
```

The inference adapter translates requests, streaming output, cancellation, and
health checks. The KV participant is what distinguishes a Kapsl backend from a
generic model server: it joins the device-wide memory authority so Kapsl can
coordinate KV capacity across models, replicas, and backend runtimes.

An OpenAI-compatible endpoint only provides the first half. It is useful as a
compatibility path, but it is not presented as deep Kapsl integration.

## Integration tiers

| Tier | KV owner | What Kapsl controls | Intended use |
|---|---|---|---|
| `unmanaged_endpoint` | none visible | routing only | generic OpenAI-compatible vLLM/TGI endpoints |
| `kv_connected` | backend or runtime | admission, capacity leases, lifecycle, global budgets; optional prefix/offload operations | minimum official backend tier |
| `shared_pool` | Kapsl runtime | all of the above plus physical blocks consumed directly by attention | flagship integration |

Capability reporting is runtime state, not a build-time label. For example, a
GGUF model that cannot use the external llama.cpp path reports
`unmanaged_endpoint` after falling back to llama.cpp's native cache. It must not
continue reporting `shared_pool` merely because the binary was compiled with
that feature.

## Opaque mode

Opaque mode belongs inside `kv_connected`. It does **not** mean "send prompts to
an opaque HTTP server and infer memory use from metrics."

In opaque mode:

- Kapsl grants a logical capacity lease before backend allocation;
- Kapsl owns admission and the device-wide budget;
- the backend exposes allocation granularity and bytes per allocation for each
  opaque cache group, so logical leases remain byte-accountable;
- every pool names its physical host/CUDA/provider domains, and replicated
  tensor-parallel pools are charged independently on every listed device;
- cache groups can name a shared physical pool, preventing hybrid allocators
  such as vLLM HMA from being double-charged;
- the backend reports commit, participant heartbeat/per-lease touch, and
  release lifecycle events;
- the backend keeps physical block IDs and tensor layout private;
- unsupported operations such as restore or prefix lookup are not advertised.

This is a practical bridge for vLLM and TGI because their allocators can remain
native while Kapsl gains real global policy control. It is not the endpoint:
`shared_pool` is the mode in which backend attention reads and writes
Kapsl-owned blocks.

## Versioned contract

The `kapsl-kv-abi` crate owns the backend-neutral semantic contract:

- major/minor compatibility negotiation;
- tier, ownership, metadata, feature, and transport capabilities;
- structured cache topology with multiple cache groups;
- full-attention, sliding-window, recurrent, and backend-defined policies;
- logical reservation requests and revocable leases;
- prefix, commit, eviction, restore, touch, and release operations;
- transport-neutral block handles;
- newline-delimited JSON control envelopes for out-of-process connectors.

Structured topology is deliberately not modeled after the current llama.cpp C
descriptor. A model may advertise several cache groups with different
geometries, so hybrid attention/SSM backends do not inherit llama.cpp's uniform
K/V limitation. Serialized handles never contain an unchecked raw pointer;
in-process adapters map a validated pool ID and block index to local storage.

## Backend paths

### llama.cpp

The existing fork remains the active `shared_pool` data plane. Its Rust bridge
now translates the live descriptor geometry into the neutral topology and only
advertises shared-pool capabilities while that path is active.

The fork is transitional. "No fork" means Kapsl should not maintain a permanent
copy of the whole backend; it does not mean zero backend-specific code. The exit
path is to upstream a small external-KV hook in llama.cpp and keep the Kapsl
adapter out of tree. Until that hook can bind Kapsl blocks and call the reserve,
commit, touch, and release lifecycle safely, removing the fork would also remove
the main product advantage.

### vLLM

`integrations/vllm` is an out-of-tree `KVConnectorBase_V1` package. Its first
phase registers as `kv_connected` with opaque metadata and obtains a Kapsl
capacity lease from vLLM's one-shot `on_new_request` hook. It reports no external
prefix hits and performs no KV copies, so it cannot accidentally skip compute
for data it did not restore.

Kapsl Runtime exposes the matching local Unix control listener with
`--kv-control-socket`. Participant reservations enter the same process-wide
memory authority as built-in backends, and expiring leases are reclaimed when
their connector heartbeat stops. The socket is an admission/lifecycle control
plane; it does not turn opaque backend memory into a shared pool.

The next vLLM phase is a worker data plane using layer-wise CUDA IPC or NIXL.
Only after attention can consume runtime-owned blocks should it advertise
`shared_pool` and `direct_attention_access`.

### TGI and other engines

Reuse the contract and implement two small compatibility surfaces:

1. request/stream/cancellation translation for inference;
2. a KV participant or coordinator client at the backend's allocator hooks.

If allocator hooks are unavailable, the backend remains an
`unmanaged_endpoint`; an HTTP adapter alone is never promoted to an official KV
backend.

## Failure policy

- Registration and ABI negotiation fail closed for a backend configured as
  KV-connected.
- Admission denial occurs before backend KV allocation whenever the backend
  exposes a one-shot request hook.
- A capability is advertised only when its data path is live.
- Unknown major versions, malformed topology, stale block generations, and
  missing leases are hard errors.
- Native backend fallback is allowed only when deployment policy permits
  unmanaged operation; production deployments can require `kv_connected` or
  `shared_pool` once that policy is wired into runtime configuration.

This boundary keeps broad backend compatibility, but the product promise stays
narrow and testable: Kapsl is the GPU-wide KV memory authority.
