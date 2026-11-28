# kapsl-sdk

Python client SDK for [kapsl-runtime](https://kapsl.ai) — the Rust-native AI model inference engine.

Supports native socket/TCP, shared memory, hybrid IPC, and optional gRPC inference
with synchronous and asynchronous streaming, plus an HTTP client for the
engine's management API and OpenAI-compatible routes.

## Install

```bash
pip install kapsl-sdk
```

Pre-compiled abi3 wheels are available for Linux, macOS, and Windows on Python 3.10+.

Native, shared-memory and gRPC clients need engine 0.2.8 or later, which
introduced the current transport protocols. Older native request layouts and
SHM protocols are rejected. Coming from 0.1.x, see the
[0.2.0 migration guide](docs/python-sdk-0.2.md).

## What's new in 0.3.0

Version 0.3.0 adds `KapslHttpClient`, a standard-library HTTP client for the
engine's API on port 9095, with typed results. The native, SHM, hybrid and
gRPC protocols are unchanged from 0.2.0, so 0.3.0 works with every engine that
0.2.0 supports and existing code keeps working.

| Feature | Minimum engine |
|---------|----------------|
| Health, hardware, `/metrics`, model list and details, start/stop/remove, autoscaling policy | 0.2.8 |
| Access management: login, users, API keys, role tokens | 0.2.8 |
| `/v1/models`, chat completions (streaming and non-streaming), embeddings | 0.2.8 |
| `start_model(wait=True)`: `active` or `denied` with the memory-admission request per domain, or the unavailable CUDA device | 0.3.0 |
| Idle unloading: `start_model(idle_unload_seconds=...)`, and `idle_unloaded`, `reloads_on_request`, `idle_unload_seconds` on `ModelStatus` | 0.3.0 |
| `SystemStats.cuda`, `gpu_sampling`, `gpu_sampling_age_secs`, `MemoryDomainUsage.authority_available_bytes` | 0.3.0 |
| `role_token_status()` without token values (older engines return the values; the SDK keeps only whether each is set) | 0.3.0 |

Fields an older engine does not send are `None` or `False`. Fields the SDK does
not model yet stay available in each result's `extra` mapping.

Engine 0.3.0 behaviour that the SDK reports as-is:

- OpenAI chat completions return `finish_reason: "stop"` even when
  `max_tokens` cut the reply short, except for managed vLLM models, whose
  responses are relayed unchanged.
- With `--idle-reload`, gRPC reports an idle-unloaded model as not ready and
  rejects its requests instead of reloading it. HTTP and native transports
  reload it.
- `GET /api/models/{id}` answers an unknown ID with HTTP 200 and an error body;
  `get_model` raises `KapslHttpError` with `status == 200`.

## HTTP management and OpenAI-compatible API

```python
from kapsl_sdk import KapslHttpClient

with KapslHttpClient("http://127.0.0.1:9095", api_token="admin-token") as http:
    result = http.start_model(
        "/models/assistant.aimod", wait=True, idle_unload_seconds=900,
        timeout_ms=600_000,
    )
    if result.denied:
        print(result.reason, [(r.domain, r.bytes) for r in result.requested])

    for model in http.list_models():
        print(model.id, model.name, model.status, model.idle_unloaded)

    stats = http.system_stats()
    print(stats.pressure_state, stats.memory_authority.domain("cuda:0"))

    reply = http.chat_completion(
        "assistant", [{"role": "user", "content": "Hello"}], max_tokens=64,
        session_id="conversation-1",
    )
    print(reply.text, reply.finish_reason)

    with http.chat_completion_stream(
        "assistant", [{"role": "user", "content": "Hello"}], include_usage=True,
    ) as stream:
        for chunk in stream:
            print(chunk.text, end="", flush=True)
```

Errors raise `KapslHttpError` with `status`, the engine's `reason` (such as
`memory_admission` or `idle_reload_memory_unavailable`), and
`retry_after_seconds` when the engine suggests a retry. With `--idle-reload`,
the first request after an idle unload waits for the model to load, so give it
a timeout that covers a cold start. See the
[API reference](docs/api-reference.md#kapslhttpclient) for every method.

## gRPC

Install the optional dependencies with `pip install 'kapsl-sdk[grpc]'`.
The Python package includes generated protocol classes; consumers do not need
`protoc` or `grpcio-tools`.

```python
from kapsl_sdk import KapslGrpcClient

with KapslGrpcClient("127.0.0.1:9097", api_token="your-token") as client:
    print(client.list_models())
    with client.infer_stream_tensors(
        "my-model", [1], "string", b"Hello", timeout_ms=30_000,
        max_new_tokens=128,
    ) as stream:
        for tensor in stream:
            print(tensor.data.decode("utf-8"), end="", flush=True)
```

Enable the engine with `--features grpc-server` and `--grpc-port 9097`.
See [Python gRPC usage](docs/python-sdk-0.2.md) and the [gRPC service contract](docs/grpc.md).

## Quick start

```python
from kapsl_sdk import KapslClient

client = KapslClient()  # connects to /tmp/kapsl.sock by default

# Streaming LLM inference
prompt = "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n"

for chunk in client.infer_stream(model_id=0, shape=[1, 1], dtype="string", data=prompt.encode()):
    print(chunk.decode("utf-8"), end="", flush=True)
```

```python
import numpy as np

# Standard tensor inference
data = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
result = client.infer(model_id=0, shape=[1, 4], dtype="float32", data=data.tobytes())
output = np.frombuffer(result, dtype=np.float32)
```

## Transports

| Client | Transport | Use case |
|--------|-----------|----------|
| `KapslHttpClient` | HTTP | Management API, statistics, access control, OpenAI-compatible chat and embeddings |
| `KapslGrpcClient` / `AsyncKapslGrpcClient` | gRPC | Typed service integrations and asynchronous streaming |
| `KapslClient` | Unix socket / TCP | Default — local or remote |
| `KapslShmClient` | Shared memory | Lowest latency, co-located only |
| `KapslHybridClient` | Socket control + SHM data | Production throughput |

```python
from kapsl_sdk import KapslClient, KapslShmClient, KapslHybridClient

# TCP
client = KapslClient("tcp://192.168.1.10:9096")

# Shared memory (same machine only)
client = KapslShmClient("kapsl-shm-default")

# Hybrid
client = KapslHybridClient("kapsl-shm-default", "/tmp/kapsl.sock")
```

## Authentication

```python
client = KapslClient(api_token="your-token")
```

`KapslHttpClient` and the gRPC clients send `api_token` as a bearer token.
Reader tokens cover inference, listings and statistics; model lifecycle,
scaling updates, `/metrics` and access management need an admin token.

## Docs

- [Installation](./docs/installation.md)
- [Quick Start](./docs/quickstart.md)
- [Client Types](./docs/client-types.md)
- [Inference](./docs/inference.md)
- [Streaming](./docs/streaming.md)
- [Authentication](./docs/authentication.md)
- [Python 0.2.0 migration and gRPC](./docs/python-sdk-0.2.md)
- [Backend-neutral KV Integration](./docs/backend-kv-integration.md)

## Requirements

- Python 3.10+
- A running `kapsl-runtime` instance ([install guide](https://downloads.kapsl.net/install.sh))
