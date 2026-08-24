# Kapsl vLLM connector

This is an out-of-tree vLLM V1 KV connector. It proves the backend adapter
boundary without carrying a vLLM fork.

Status: the package supports an opaque admission mode and an experimental,
opt-in Linux CUDA shared-pool mode. Both fail closed unless they can register
with Kapsl Runtime and obtain a lease before vLLM allocation. A shared-pool
participant additionally cannot activate until every worker proves that its
registered KV tensors alias its Kapsl binding and its exact adapter/backend
profile is allowlisted by the deployment. The first certifiable profile pins
vLLM's `FLASH_ATTN` implementation; automatic backend selection is rejected in
shared mode because it could choose a reader that was never tested. The profile
is declared during registration so an unapproved build is rejected before CUDA
memory is provisioned, then repeated in worker attachment evidence.

The default `opaque` mode implements Kapsl's **KV-connected opaque** tier:

- it registers a versioned KV participant;
- Kapsl grants a capacity lease when vLLM admits a request;
- vLLM's cache-group page sizes are converted into an opaque byte-accounting
  model, without exposing block handles;
- completion and release are reported back to Kapsl;
- vLLM retains its private block layout and performs attention normally.

The opt-in `shared_pool` mode implements the first direct-attention data plane:

- Kapsl Runtime allocates and accounts for one isolated physical KV backing per
  advertised CUDA device, then exports it with CUDA IPC;
- each vLLM worker imports its device's binding before KV tensor construction;
- vLLM's attention tensors are views of the Kapsl-owned backing, so there is no
  copy and no second PyTorch KV allocation;
- vLLM keeps its native block allocator and block tables, while Kapsl grants
  aggregate request capacity through participant-managed leases.

This is ABI 1.3's `participant_managed` shared-pool mode. Empty block arrays in
its request leases are intentional: vLLM selects native block indices, while
Kapsl owns the physical allocation and the total capacity budget. Opaque mode
does **not** advertise shared-pool access. Neither mode currently advertises
prefix restore or KV transfer.

## Install and configure

Install this package in the same environment as vLLM, then configure vLLM's
external connector module:

```json
{
  "kv_connector": "KapslConnectorV1",
  "kv_role": "kv_both",
  "kv_connector_module_path": "kapsl_vllm_connector",
  "kv_connector_extra_config": {
    "kapsl_control_endpoint": "unix:///run/kapsl/kv-control.sock",
    "kapsl_participant_id": "vllm-qwen-worker",
    "kapsl_model_fingerprint": "sha256:<model-manifest-digest>",
    "kapsl_kv_mode": "opaque",
    "kapsl_memory_domains": [
      {"kind": "cuda", "device_id": 0}
    ],
    "kapsl_lease_ttl_ms": 30000
  }
}
```

To make vLLM consume Kapsl-owned CUDA memory directly, select `shared_pool`:

```json
{
  "kv_connector": "KapslConnectorV1",
  "kv_role": "kv_both",
  "kv_connector_module_path": "kapsl_vllm_connector",
  "kv_connector_extra_config": {
    "kapsl_control_endpoint": "unix:///run/kapsl/kv-control.sock",
    "kapsl_participant_id": "vllm-qwen-worker",
    "kapsl_model_fingerprint": "sha256:<model-manifest-digest>",
    "kapsl_kv_mode": "shared_pool",
    "kapsl_memory_domains": [
      {"kind": "cuda", "device_id": 0},
      {"kind": "cuda", "device_id": 2}
    ],
    "kapsl_rank_device_map": {"0": 0, "1": 2},
    "kapsl_lease_ttl_ms": 30000
  }
}
```

The vLLM engine configuration must also select `FLASH_ATTN` explicitly. A
different backend (including FlashInfer or Triton attention), automatic
selection, or a per-cache-kind override needs a separate profile and native
probe before it can use `shared_pool`.

Start Kapsl Runtime with the matching listener and TTL:

```bash
kapsl run \
  --kv-control-socket /run/kapsl/kv-control.sock \
  --kv-control-lease-ttl-ms 30000 \
  --kv-shared-pool-profile \
    'kapsl-vllm-connector,0.5.0,<vllm-version>,vllm-v1-packed-cuda-ipc/flash-attn'
```

Create `/run/kapsl` with ownership suitable for the runtime user first. The
runtime creates the socket with mode `0600`; the transport is local-only.
The profile tuple must match the installed vLLM version exactly and should be
configured only after that build passes the GPU conformance matrix. Without an
allowlisted profile, opaque participants continue to work but `shared_pool`
registration fails closed.

`kapsl_control_endpoint`, `kapsl_model_fingerprint`, and
`kapsl_memory_domains` are required. Device IDs are CUDA ordinals in Kapsl
Runtime's device-visibility namespace. For tensor parallelism, list every domain that
contains a replica of the vLLM block pool; Kapsl charges and allocates the pool
once on each listed device. When there is more than one domain, the explicit
`kapsl_rank_device_map` maps each vLLM global rank to the corresponding Kapsl
device ID. Do not infer these IDs from an unrelated `CUDA_VISIBLE_DEVICES`
mapping.

The connector sends one participant heartbeat at one third of
`kapsl_lease_ttl_ms`; that operation renews all of its live request leases, so
heartbeat traffic does not grow with request concurrency. The TTL value must not exceed the runtime's
`--kv-control-lease-ttl-ms` maximum. Registration, admission, heartbeat, and
release failures are hard errors, so a deployment cannot be mislabeled as
KV-connected while running as an unmanaged vLLM endpoint.

CUDA placements require a Kapsl Runtime build with `gpu-device-pool`. On Linux,
that build installs the isolated CUDA IPC provisioner when the control socket
is enabled. A CPU-only or non-Linux runtime still supports opaque domains when
it has bounded authority, but rejects `shared_pool` registration.

The connector targets vLLM's experimental `KVConnectorBase_V1` API. Because
that upstream interface changes, the worker hook checks the packed
`allocate_kv_cache` signature before changing it and restores every imported
module at shutdown or on partial installation failure. An unknown allocation
shape is a startup error, not an opaque fallback. Shared mode currently
supports CUDA, packed vLLM KV tensors, tensor parallelism on one host, and no
pipeline/data/decode-context parallel partitions or vLLM sleep mode. The Kapsl
wire contract and client have no vLLM dependency.

The shared-pool startup lifecycle is `register -> attach every worker ->
activate -> reserve`. `register_kv_caches` verifies the complete layer set,
storage base, imported byte size, and every tensor's bounded byte span before
the worker sends attachment evidence. The scheduler's first admission attempts
activation; if any rank did not attach, no request lease is issued. The wire ABI
also defines synchronized detach, but this connector does not send it until
vLLM exposes a teardown callback that guarantees every model-owned KV view has
been destroyed. A crash or ambiguous teardown intentionally retains the
exported backing instead of risking a use-after-free.

## Test

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

These host tests cover the wire contract, activation ordering, and exact tensor
alias/span checks. Production certification additionally requires the
backend-native CUDA attention probes in
[`../../docs/backend-kv-conformance.md`](../../docs/backend-kv-conformance.md);
passing only the host tests is not sufficient reason to allowlist a profile.

## Linux/CUDA certification

Run Kapsl Runtime with the exact profile provisionally allowlisted only inside
an isolated certification job, then execute:

```bash
kapsl-vllm-flash-attn-probe \
  --endpoint unix:///run/kapsl/kv-control.sock \
  --devices 0,1 \
  --adapter-build-id sha256:<adapter-wheel-digest> \
  --backend-build-id sha256:<vllm-wheel-digest> \
  --runtime-build-id sha256:<runtime-binary-digest> \
  --report artifacts/vllm-kv-report.json \
  --allowlist-output artifacts/allowlist-profile.txt
```

The process launches one worker per listed CUDA ordinal. Every worker imports
its Kapsl binding through the production allocator hook while the normal
`torch.zeros` KV path is poisoned. It then runs vLLM's native
`reshape_and_cache_flash` writer and paged `flash_attn_varlen_func` reader,
checks per-block guards, compares prefill and two decode steps with a reference,
mutates Kapsl-owned V storage and requires a causal output change, overwrites a
reused block, and reads the maximum block index. Rank zero also checks
pre-attachment activation, lease heartbeat renewal, cancellation, capacity
exhaustion, live-lease detach rejection, and post-deactivation admission
rejection. All CUDA work is fenced and all tensor views are destroyed before
detach.

The report is always retained. `allowlist-profile.txt` is removed before the
run and recreated only if all six gates pass on every rank. The engine repo's
opt-in **vLLM Shared-Pool Conformance** workflow builds and hashes the runtime,
installs an exact vLLM wheel, runs host contract tests, enables synchronous CUDA
error reporting, executes this probe, and uploads both artifacts. The generated
tuple is still operator authorization rather than remote attestation; review
the report and matrix before deploying it.
