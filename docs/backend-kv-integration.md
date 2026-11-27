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

ABI 1.3 adds an external attachment gate. Registration now means
**provisioned**, not usable. Registration declares an exact adapter/backend
profile, which the runtime checks against its deployment allowlist before it
allocates or exports memory. Each worker then imports its receipt binding,
constructs backend-native cache tensors, and repeats that profile with exact
layer views as bounded offsets inside the binding. The runtime checks the epoch,
byte size, topology coverage, and parallel shard.
Only after every binding is attached can the scheduler activate the participant;
reservations fail before activation. Detach is a separate synchronized action
and cannot race a live lease. Raw process-local pointers never enter the wire
contract.

Attachment evidence prevents accidental partial wiring, but it is not by itself
proof that a backend attention kernel dereferenced those tensors. A profile is
eligible for a deployment allowlist only after the hardware conformance suite
performs causal backend-native KV write and attention-read probes. See
[`backend-kv-conformance.md`](backend-kv-conformance.md).

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

## Striped elastic pools (ABI 1.6)

An elastic `cuda_vmm` binding normally backs one contiguous virtual range in
the participant, and its segments cover that range's mapped prefix. A CUDA
allocation maps only from offset zero and unmaps only whole, so one segment
backs one contiguous piece of one buffer. A backend that keeps its cache in
several buffers indexed by the same slot, such as SGLang with one K and one V
tensor per layer, would have to rebuild them as views of one range. A striped
binding backs each buffer instead:

- `elastic.stripe_count` is the number of ranges (stripes), from 2 to
  `KV_MAX_POOL_STRIPES` (253). A single range omits it; an explicit 1 is
  refused, so each layout has one form.
- Stripes share the block geometry. `bytes_per_block` is the total over all
  stripes and must divide by the stripe count. Each stripe holds an equal part
  of every block, and block `j` sits at the same offset in every stripe. The
  minimum, mapped and maximum block counts apply to every stripe, and each
  stripe's sizes must be multiples of the allocation granularity.
- Each segment names its `stripe` (omitted for stripe 0) and its offset is
  relative to that stripe. Every stripe's segments densely cover the same
  mapped prefix, so no segment crosses into another stripe. Segment IDs and
  handle indices are unique across the binding.
- A resize operation carries the binding's `stripe_count` and changes every
  stripe by the same tail: worker phases carry segments covering the changed
  range in each stripe.
- Attachment views name their stripe and are unique per group, layer and
  stripe. `imported_bytes` and `mapped_bytes` remain totals over all stripes.
- A participant opts in with the `striped_elastic_pool` feature, which needs
  ABI 1.6 and `live_pool_resize`. It then receives only striped bindings, and
  a participant without the feature never receives one.

Use stripes when the backend's attention already reads separate buffers and
changing how it allocates them would be invasive. Prefer one range when the
backend can take strided views of a single allocation: each growth step then
maps one segment instead of one per stripe. All handles of a response travel in
one `SCM_RIGHTS` message, and Linux accepts at most 253 descriptors per message,
which sets the stripe bound; a response covering several tensor-parallel
bindings must stay within it as well. Segments already address a stripe and an
offset, so mapping pages anywhere in a stripe later only relaxes the
dense-prefix rule.

A 1.6 host still accepts 1.5 participants and answers every request in the
version the request carried (`KvAbiVersion::negotiate`), so a 1.5 participant
that checks the response version is unaffected.

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

[`kapsl-integrations/integrations/vllm`](https://github.com/kapsl-runtime/kapsl-integrations/tree/main/integrations/vllm)
is an out-of-tree `KVConnectorBase_V1` package. Its default
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
created. Workers report those exact aliases; the scheduler cannot obtain a
request lease until every binding is attached and runtime activation succeeds.
Only vLLM builds that pass the backend-native GPU probes may be placed on the
runtime's shared-pool profile allowlist.
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
