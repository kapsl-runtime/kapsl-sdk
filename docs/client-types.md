# Client Types

`kapsl-sdk` provides three client classes, each optimised for a different deployment topology.

## KapslClient

General-purpose client. Works over Unix socket (Linux/macOS), Windows named pipe, or TCP.

```python
from kapsl_sdk import KapslClient

# Default: Unix socket /tmp/kapsl.sock (Linux/macOS)
#          or \\.\pipe\kapsl (Windows)
client = KapslClient()

# Explicit Unix socket
client = KapslClient("/var/run/kapsl.sock")
client = KapslClient("unix:///var/run/kapsl.sock")

# TCP
client = KapslClient("tcp://127.0.0.1:9096")
client = KapslClient(host="127.0.0.1", port=9096)

# Windows named pipe
client = KapslClient(r"\\.\pipe\kapsl")
client = KapslClient("pipe://kapsl")
```

### Connection pool

`KapslClient` maintains a pool of reusable connections (default size: 8). Requests borrow a connection, use it, and return it. On a broken connection the client transparently retries once on a fresh connection.

```python
# Custom pool size — set to 0 to disable pooling
client = KapslClient(max_pool_size=16)
```

### When to use

- Default choice for all deployments
- Required when connecting to a remote runtime over TCP
- Multi-threaded applications (each thread borrows a separate connection)

---

## KapslShmClient

Shared-memory client. Exchanges tensor payloads through a named shared-memory region instead of a socket, eliminating data copies for large tensors on the same machine.

```python
from kapsl_sdk import KapslShmClient

client = KapslShmClient("kapsl-shm-default")
result = client.infer(
    model_id=0,
    shape=[1, 224, 224, 3],
    dtype="float32",
    data=image_bytes,
)
```

### When to use

- High-throughput local inference where tensor size is large (images, audio)
- Both the Python process and `kapsl-runtime` must be on the same machine
- The shared-memory region name must match the one configured in the runtime

---

## KapslHybridClient

Hybrid client that combines IPC socket signaling with shared-memory payloads. Coordination messages go over the socket; tensor data moves through shared memory.

```python
from kapsl_sdk import KapslHybridClient

client = KapslHybridClient(
    shm_name="kapsl-shm-default",
    socket_path="/tmp/kapsl.sock",
)
result = client.infer(
    model_id=0,
    shape=[1, 224, 224, 3],
    dtype="float32",
    data=image_bytes,
)
```

### When to use

- Maximum local throughput with lowest latency
- Requires shared-memory support configured in `kapsl-runtime`
- Same-machine deployments only

---

## Choosing a client

| Scenario | Recommended client |
|----------|--------------------|
| Remote runtime over network | `KapslClient` (TCP) |
| Local runtime, general use | `KapslClient` (socket) |
| Local runtime, large tensors | `KapslShmClient` |
| Local runtime, maximum speed | `KapslHybridClient` |
