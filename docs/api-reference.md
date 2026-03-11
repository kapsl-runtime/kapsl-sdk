# API Reference

## KapslClient

General-purpose inference client over Unix socket, TCP, or Windows named pipe.

### Constructor

```python
KapslClient(
    endpoint: str | None = None,
    *,
    protocol: str | None = None,
    host: str | None = None,
    port: int | None = None,
    socket_path: str | None = None,
    pipe_name: str | None = None,
    max_pool_size: int = 8,
    api_token: str | None = None,
)
```

**Endpoint resolution** (first match wins):

1. `endpoint` string — parsed as URI (`tcp://`, `unix://`, `pipe://`) or bare path/address
2. `protocol` + optional `host`/`port`/`socket_path`/`pipe_name`
3. `host` and/or `port` → TCP
4. `socket_path` → Unix socket
5. `pipe_name` → Windows named pipe
6. Default: `/tmp/kapsl.sock` (Unix) or `\\.\pipe\kapsl` (Windows)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `endpoint` | `None` | URI or bare path/address |
| `protocol` | `None` | `"socket"`, `"tcp"`, or `"pipe"` |
| `host` | `"127.0.0.1"` | TCP host (only when using protocol/host/port) |
| `port` | `9096` | TCP port |
| `socket_path` | `/tmp/kapsl.sock` | Unix socket path |
| `pipe_name` | `\\.\pipe\kapsl` | Windows named pipe name |
| `max_pool_size` | `8` | Connection pool capacity; `0` disables pooling |
| `api_token` | `None` | Bearer token sent with every request |

**Raises**: `ValueError` on invalid or conflicting endpoint options.

---

### infer()

```python
client.infer(
    model_id: int,
    shape: list[int],
    dtype: str,
    data: bytes,
    additional_inputs: dict[str, tuple[list[int], str, bytes]] | None = None,
    session_id: str | None = None,
) -> bytes
```

Sends a synchronous inference request and returns the raw output bytes.

| Parameter | Type | Description |
|-----------|------|-------------|
| `model_id` | `int` | Numeric ID of the loaded model |
| `shape` | `list[int]` | Input tensor shape |
| `dtype` | `str` | Input dtype (`float32`, `int64`, `uint8`, etc.) |
| `data` | `bytes` | Input tensor bytes |
| `additional_inputs` | `dict` | Extra named tensors: `{name: (shape, dtype, data)}` |
| `session_id` | `str` | Optional session ID for stateful inference |

**Returns**: `bytes` — raw output tensor data.

**Raises**: `ConnectionError` on transport failure, `RuntimeError` on server-side error.

---

### infer_tensor()

```python
client.infer_tensor(
    model_id: int,
    shape: list[int],
    dtype: str,
    data: bytes,
    additional_inputs: dict[str, tuple[list[int], str, bytes]] | None = None,
    session_id: str | None = None,
) -> tuple[bytes, list[int], str]
```

Like `infer()` but returns `(data, shape, dtype)` so callers can reconstruct the output tensor without hardcoding its dimensions.

**Returns**: `(bytes, list[int], str)` — output data, output shape, output dtype string.

---

### infer_stream()

```python
client.infer_stream(
    model_id: int,
    shape: list[int],
    dtype: str,
    data: bytes,
    additional_inputs: dict[str, tuple[list[int], str, bytes]] | None = None,
    session_id: str | None = None,
) -> Iterator[bytes]
```

Sends a streaming inference request. Returns an iterator that yields one `bytes` chunk per output token/frame as they arrive.

Uses a dedicated connection (not pooled). The connection is held until the iterator is exhausted or garbage-collected.

**Raises**: `RuntimeError` if the server returns an error mid-stream.

---

### protocol() / endpoint()

```python
client.protocol() -> str   # "socket", "tcp", or "pipe"
client.endpoint() -> str   # e.g. "tcp://127.0.0.1:9096"
```

---

## KapslShmClient

Shared-memory inference client.

```python
from kapsl_sdk import KapslShmClient

client = KapslShmClient(shm_name: str)
```

Exposes the same `infer`, `infer_tensor`, and `infer_stream` methods as `KapslClient`.

---

## KapslHybridClient

Hybrid IPC + shared-memory client.

```python
from kapsl_sdk import KapslHybridClient

client = KapslHybridClient(shm_name: str, socket_path: str)
```

Exposes the same `infer`, `infer_tensor`, and `infer_stream` methods as `KapslClient`.

---

## list_voices()

```python
from kapsl_sdk import list_voices

list_voices() -> list[str]
```

Returns the names of all bundled voice embeddings (without the `.bin` extension), sorted alphabetically.

---

## load_voice()

```python
from kapsl_sdk import load_voice

load_voice(name: str) -> bytes
```

Loads a bundled voice embedding by name (e.g., `"af_bella"`).

Returns raw `float32` bytes. Reshape as `(-1, 1, 256)` to index by token length.

**Raises**: `FileNotFoundError` if the name is not found. The error message lists available voices.

---

## Supported dtypes

| String | NumPy type | Bytes per element |
|--------|-----------|------------------|
| `float32` | `np.float32` | 4 |
| `float64` | `np.float64` | 8 |
| `float16` | `np.float16` | 2 |
| `int32` | `np.int32` | 4 |
| `int64` | `np.int64` | 8 |
| `uint8` | `np.uint8` | 1 |
