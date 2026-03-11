# Quick Start

This guide shows the minimal steps to run inference against a `kapsl-runtime` instance.

## Prerequisites

- `kapsl-sdk` installed (`pip install kapsl-sdk`)
- `kapsl-runtime` running locally (defaults: Unix socket at `/tmp/kapsl.sock`, port 9096 for TCP)

## 1. Connect and infer

```python
import numpy as np
from kapsl_sdk import KapslClient

# Connect to the default local socket
client = KapslClient()

# Prepare a float32 tensor — e.g. a 1×4 vector
data = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)

# Run inference against model ID 0
result_bytes = client.infer(
    model_id=0,
    shape=[1, 4],
    dtype="float32",
    data=data.tobytes(),
)

# Interpret the result
output = np.frombuffer(result_bytes, dtype=np.float32)
print(output)
```

## 2. Get output shape alongside data

Use `infer_tensor` when you do not know the output shape in advance:

```python
data_bytes, shape, dtype = client.infer_tensor(
    model_id=0,
    shape=[1, 4],
    dtype="float32",
    data=data.tobytes(),
)

output = np.frombuffer(data_bytes, dtype=np.dtype(dtype)).reshape(shape)
print(output)
```

## 3. Connect over TCP

```python
# Remote runtime or a local TCP endpoint
client = KapslClient("tcp://192.168.1.10:9096")

# Or using keyword arguments
client = KapslClient(host="192.168.1.10", port=9096)
```

## 4. Authenticated connection

```python
client = KapslClient(
    "tcp://192.168.1.10:9096",
    api_token="your-api-token",
)
```

Every request sent by this client will include the token automatically.

## Next steps

- [Client Types](./client-types.md) — pick the right transport for your workload
- [Inference](./inference.md) — multi-input models, session IDs
- [Streaming](./streaming.md) — LLM token streaming
- [Authentication](./authentication.md) — token configuration
