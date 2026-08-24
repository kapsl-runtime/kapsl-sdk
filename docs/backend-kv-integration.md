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

ABI 1.1 added the first out-of-process `shared_pool` ownership boundary:

- registration returns an epoch-bound receipt containing one isolated physical
  binding for every logical pool and memory domain;
- leases name runtime pool bindings, block indices, and generations rather
  than serializing process-local pointers;
- a shared-pool release must prove the backend has synchronized access before
  blocks can be recycled;
- a lease that expires without that proof is quarantined, not returned to the
  allocator;
- transport-specific provisioners retain the actual allocation and must zero
  each assigned block before publishing its handle.

ABI 1.2 adds explicit block-selection ownership. `runtime_leased` pools publish
generation-checked physical indices and require a synchronized release fence.
`participant_managed` pools let a backend such as vLLM keep its proven native
block allocator: Kapsl owns and exports the backing allocation and grants
aggregate capacity, but lease block arrays remain empty because backend block
tables select the indices. This avoids pretending that two independent block
allocators can safely choose identical IDs.

The runtime does not export its process-wide CUDA allocator backing: doing so
would grant one external worker access to allocations belonging to other
models. A CUDA IPC or NIXL provider must create an isolated exportable binding
for the participant. Until such a provider is configured, an external
`shared_pool` registration fails closed.

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

`integrations/vllm` is an out-of-tree `KVConnectorBase_V1` package. Its default
mode registers as `kv_connected` with opaque metadata and obtains a Kapsl
capacity lease from vLLM's one-shot `on_new_request` hook. It reports no
external prefix hits and performs no KV copies, so it cannot accidentally skip
compute for data it did not restore.

Kapsl Runtime exposes the matching local Unix control listener with
`--kv-control-socket`. Participant reservations enter the same process-wide
memory authority as built-in backends, and expiring leases are reclaimed when
their connector heartbeat stops.

On Linux CUDA builds, the runtime listener also installs an isolated CUDA IPC
provisioner. The connector's opt-in `shared_pool` mode registers structured
topology, imports the receipt's device binding in each worker, and substitutes
that backing at vLLM's packed raw-allocation seam before attention tensors are
created. Attention therefore reads and writes Kapsl-owned memory directly.
vLLM remains responsible for its native block IDs, while Kapsl owns physical
capacity and request-level admission. The compatibility hook is deliberately
narrow and signature-checked because it tracks an experimental upstream API.
Prefix restore/offload and pipeline-parallel partitions remain later phases.

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
