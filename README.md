# kapsl-sdk

Python client SDK for [kapsl-runtime](https://kapsl.ai) — the Rust-native AI model inference engine.

Supports native socket/TCP, shared memory, hybrid IPC, and optional gRPC inference
with synchronous and asynchronous streaming.

## Install

```bash
pip install kapsl-sdk
```

Pre-compiled abi3 wheels are available for Linux, macOS, and Windows on Python 3.9+.

The 0.2.0 source requires the matching engine transport update. Older native
request layouts and SHM protocols are rejected. See the
[0.2.0 migration guide](docs/python-sdk-0.2.md) before upgrading.

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

- Python 3.9+
- A running `kapsl-runtime` instance ([install guide](https://downloads.kapsl.net/install.sh))
