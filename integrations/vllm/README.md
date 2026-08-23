# Kapsl vLLM connector

This is an out-of-tree vLLM V1 KV connector. It proves the backend adapter
boundary without carrying a vLLM fork.

Status: the package, versioned wire client, and Kapsl Runtime coordinator
listener are implemented. The connector intentionally fails closed unless it
can register with that listener and obtain a lease before vLLM allocation.

The current connector implements Kapsl's **KV-connected opaque** tier:

- it registers a versioned KV participant;
- Kapsl grants a capacity lease when vLLM admits a request;
- vLLM's cache-group page sizes are converted into an opaque byte-accounting
  model, without exposing block handles;
- completion and release are reported back to Kapsl;
- vLLM retains its private block layout and performs attention normally.

It deliberately does **not** advertise prefix restore, KV transfer, or
shared-pool access. Those require worker-side CUDA/NIXL integration and will
graduate the connector toward `shared_pool`; silently pretending that a no-op
connector has those features would defeat Kapsl's product boundary.

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
    "kapsl_memory_domains": [
      {"kind": "cuda", "device_id": 0}
    ],
    "kapsl_lease_ttl_ms": 30000
  }
}
```

Start Kapsl Runtime with the matching listener and TTL:

```bash
kapsl run \
  --kv-control-socket /run/kapsl/kv-control.sock \
  --kv-control-lease-ttl-ms 30000
```

Create `/run/kapsl` with ownership suitable for the runtime user first. The
runtime creates the socket with mode `0600`; the transport is local-only.

`kapsl_control_endpoint`, `kapsl_model_fingerprint`, and
`kapsl_memory_domains` are required. Device IDs are the CUDA device ordinals as
seen by Kapsl Runtime. For tensor parallelism, list every physical domain that
contains a replica of the vLLM block pool; Kapsl charges the pool stride once
on each listed device. Do not guess or infer these IDs from an unrelated
`CUDA_VISIBLE_DEVICES` mapping.

The connector sends one participant heartbeat at one third of
`kapsl_lease_ttl_ms`; that operation renews all of its live request leases, so
heartbeat traffic does not grow with request concurrency. The TTL value must not exceed the runtime's
`--kv-control-lease-ttl-ms` maximum. Registration, admission, heartbeat, and
release failures are hard errors, so a deployment cannot be mislabeled as
KV-connected while running as an unmanaged vLLM endpoint.

CUDA placements require a Kapsl Runtime build with its CUDA memory authority
enabled. A CPU-only runtime rejects the registration instead of accepting an
unbounded device claim.

The connector targets vLLM's experimental `KVConnectorBase_V1` API. Because
that upstream interface changes, compatibility is isolated to `connector.py`;
the Kapsl wire contract and client have no vLLM dependency.

## Test

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```
