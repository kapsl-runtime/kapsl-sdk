# Kapsl KV ABI

This crate defines KV-memory participation: registration, admission, capacity,
leases, ownership, shared mappings, completion, resize and detach. The existing
transport-neutral JSON contract and its version negotiation remain unchanged.
It does not define backend launching or general inference.

## Native shared KV v1

`native::KapslNativeKvHostV1` and `include/kapsl_kv_native_v1.h` publish a C
function table for in-process shared KV. An adapter queries `kapsl-kv-native`
version 1 through `KapslBackendHostExtensionsV1` from `kapsl-backend-abi` and
validates size, version, every required callback and the device/model/replica
binding before use. Missing or incompatible tables fail initialization when
shared KV is required. No Rust-owned value or trait crosses a shared library
boundary; pointer-bearing values in this extension never travel over JSON/IPC.

The engine owns storage and admission. It creates a pool view only during an
admitted model lifecycle operation and validates requested geometry and device
before allocation. The geometry's requested block count includes all layers;
one physical block stores K and V for one layer. Element sizes/layout support
are explicit host capabilities: an unsupported layout fails without allocating
or changing the requested geometry.

The host issues a nonzero pool ID unique for its process lifetime. Every later
operation checks that the ID belongs to this host's model/replica. Retired IDs
never alias new pools, including when physical device addresses are reused.
The pool descriptor's addressable range is an address space, not a grant to
access another owner's blocks. The block table contains only admitted blocks
reserved for this view.

`reserve` identifies an active engine request, model, replica, sequence and
sequence slot. The engine rejects missing/foreign/cancelled requests, a slot
still owned by another sequence, shrink attempts and quota violations before
allocation. Sequence ID zero is valid; request ID zero is not. Sequence IDs
are never reused within a pool, so a late release cannot free a later request.
Sequence slots may be reused after successful release. The integration adapter
must bind each scheduler slot to the engine request before any reservation and
keep that binding until all device use and callbacks for the sequence finish.
It must not infer request ownership from callback order or a reusable slot ID.
Reservation growth preserves existing blocks. `KAPSL_KV_RESERVE_UPLOAD` publishes the table
before return; otherwise the adapter calls `commit` before device work reads
it. Failed growth/upload leaves previous reservations intact, after safely
synchronizing and reclaiming any temporary allocation.

The tuple of pool, sequence and request retains allocation attribution until
release. Adapters cannot rebind a live sequence to another request. Cancellation
revokes new growth, while matching release remains permitted. `touch` verifies
a live reservation without changing ownership or capacity. `pool_bytes` reports
the view's live owned blocks and table storage, not the full shared address space.

`release` synchronizes device use before returning sequence blocks. `destroy_pool`
drains and synchronizes all remaining references before releasing the view.
Invalid IDs and double frees fail. Failed synchronization/free retains storage
and accounting for retry; it cannot acknowledge successful reclamation. Errors
return a status, leave output arguments untouched, and use engine diagnostics
for details. They never return a host allocation for an adapter to free.

The host table/context remain alive through adapter shutdown. After successful
pool destruction, its borrowed pointers cannot be used. If an adapter leaks a
pool, the engine reclaims it only after calls are quiescent and synchronization
succeeds; unsafe-to-reclaim storage stays charged and quarantined. Adapter
unload/reload and process teardown use the generic backend lifecycle.

## Validation

Rust tests validate versions, bounded geometry, ownership headers, required
callbacks and layouts. `scripts/test-backend-abi-header.sh` checks both C11 and
C++17 declarations. These are contract tests, not proof of an engine host or
GPU implementation. Consumers must also execute native host tests for valid
and invalid ownership, cancellation, synchronization failure and cleanup.

## Publication

The `publish-host-contracts.yml` workflow publishes from an exact
`kapsl-kv-abi-vX.Y.Z` tag already merged into `main`. It uses OIDC through the
protected `crates-io-backend-abi` environment. Before publishing through this
workflow, configure `kapsl-runtime/kapsl-sdk`, `publish-host-contracts.yml`, and
that environment as the crate's trusted publisher. Branches, prereleases and
mismatched tags cannot publish. Engine/integration consumers use the published
crate version, without a temporary patch or cross-repository path dependency.
