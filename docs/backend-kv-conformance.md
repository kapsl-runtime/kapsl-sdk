# Shared-pool backend conformance

`shared_pool` means a backend's real attention implementation consumes memory
owned and accounted for by Kapsl. API compatibility, successful CUDA IPC import,
or a tensor with the expected shape is not sufficient evidence.

Conformance is tied to an exact tuple:

```text
(adapter_id, adapter_version, backend_version, profile_id)
```

The profile identifies one allocator seam, KV layout, and attention integration.
Changing any tuple field invalidates the result. GPU architecture, driver/CUDA
version, element type, block size, attention backend, and parallel topology are
recorded as matrix dimensions even though they are not wire identity fields.

## Required gates

1. **Contract.** Reject incompatible ABI versions, incomplete topology, wrong
   ownership, missing bindings, stale epochs, duplicate shards, and unbounded
   views. An HTTP-only endpoint must remain `unmanaged_endpoint`.
2. **Allocator attachment.** Import an isolated Kapsl allocation through the
   advertised transport. Construct every backend KV tensor through the real
   allocator seam, then prove the registered layer tensors share the imported
   storage and remain inside their negotiated byte ranges. Deny or poison the
   backend's normal KV allocation path so a hidden second cache fails the test.
3. **Backend-native write.** Run the backend's actual KV population/cache-write
   kernel with deterministic K/V inputs and a selected block table. Observe the
   expected bytes from the exporting Kapsl allocation. Guard regions around the
   negotiated views must remain unchanged.
4. **Backend-native read.** Run the backend's actual paged/recurrent attention
   implementation against those blocks and compare with a reference result.
   Mutate a selected Kapsl-owned K or V block, synchronize, and run attention
   again. The result must change to the second reference value. This causal
   mutation is what distinguishes attention proof from pointer inspection.
5. **Lifecycle.** Exercise prefill, at least two decode steps, block reuse, the
   maximum negotiated block index, cancellation, and capacity exhaustion.
   Reservations before complete attachment or after deactivation must fail.
   Detach must fail with a live lease and succeed only after the backend's GPU
   work is fenced and all KV views are destroyed.
6. **Parallel coverage.** Repeat attachment and read/write probes for every
   tensor/pipeline shard and cache group. Each physical binding must be consumed
   by exactly its negotiated rank; one successful rank cannot certify the rest.

The runtime attachment operation enforces gates 1 and 2 at each startup. Gates
3–6 run in a Linux GPU conformance job against the exact backend build. Only a
tuple with all gates passing belongs in the runtime's
`--kv-shared-pool-profile` allowlist. That flag is operator authorization, not
cryptographic remote attestation; access to the mode-`0600` local control socket
remains part of the deployment trust boundary.

## CI split

The ordinary SDK suite runs without vLLM or CUDA and checks serialization,
fail-closed state transitions, topology coverage, and tensor alias/span logic.
A hardware job must run the backend-specific write/read/lifecycle driver under
CUDA error checking for every supported matrix entry. Results should retain:

- the exact four-field profile tuple and backend package/build digest;
- Kapsl Runtime, GPU, driver, CUDA, and framework versions;
- dtype, cache geometry, attention implementation, and parallel topology;
- per-gate pass/fail results and logs.

vLLM's current profile is `vllm-v1-packed-cuda-ipc`: the out-of-tree hook owns
the allocation seam and startup attachment checks. It must remain experimental
until its backend-native GPU job passes gates 3–6. llama.cpp needs the same
tests against an upstream external-buffer hook before the maintained fork can be
removed. TGI remains `unmanaged_endpoint` until its serving engine offers a
safe allocator hook and passes this matrix.
