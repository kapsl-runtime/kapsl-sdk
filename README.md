# kapsl-sdk

Python client SDK for [kapsl-runtime](https://kapsl.ai) — the Rust-native AI model inference engine.

Supports socket, TCP, shared-memory, and hybrid transports with a simple Python API.

## Install

```bash
pip install kapsl-sdk
```

Pre-compiled abi3 wheels are available for Linux, macOS, and Windows on Python 3.9+.

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
| `KapslClient` | Unix socket / TCP | Default — local or remote |
| `KapslShmClient` | Shared memory | Lowest latency, co-located only |
| `KapslHybridClient` | Socket control + SHM data | Production throughput |

```python
from kapsl_sdk import KapslClient, KapslShmClient, KapslHybridClient

# TCP
client = KapslClient("tcp://192.168.1.10:9096")

# Shared memory (same machine only)
client = KapslShmClient()

# Hybrid
client = KapslHybridClient()
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

## Requirements

- Python 3.9+
- A running `kapsl-runtime` instance ([install guide](https://downloads.kapsl.net/install.sh))
