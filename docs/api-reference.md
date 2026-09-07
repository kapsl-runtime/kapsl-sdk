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
    timeout_ms: int | None = None,
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
| `timeout_ms` | `None` | Default deadline for the complete request/stream |

**Raises**: `ValueError` on invalid or conflicting endpoint options.

All inference methods also accept keyword [request options](python-sdk-0.2.md#request-options),
including deadlines, priority, CPU selection, and generation overrides.
Use `with KapslClient(...) as client` or `client.close()` to cancel active
operations and close pooled connections. `client.closed` reports its state.

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

**Raises**: `ConnectionError` on transport failure, `RuntimeError` on server-side
error, `TimeoutError` when the complete request deadline expires.

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

Uses a dedicated connection. The returned stream supports `close()`, `cancel()`,
`closed`, `with`, `async with`, and `async for`. Use a context manager when
stopping early. Cancellation interrupts a blocked read. A deadline covers the
whole stream, including time when the caller is not reading.

**Raises**: `RuntimeError` if the server returns an error mid-stream.

### infer_stream_tensors()

Accepts the same arguments as `infer_stream`, yielding immutable `Tensor`
values with `data: bytes`, `shape: tuple[int, ...]`, and `dtype: str`.

## KapslGrpcClient / AsyncKapslGrpcClient

Install `kapsl-sdk[grpc]`. Both constructors accept `target`, `api_token`,
`tls`, `root_certificates`, `private_key`, `certificate_chain`, `timeout_ms`,
and `max_message_bytes`. PEM configuration uses bytes.

Both clients expose the four inference methods above, plus `server_live`,
`server_ready`, `server_metadata`, `model_ready`, `model_metadata`, and
`list_models`. Model selection accepts a name or numeric ID; `input_name`
selects the primary tensor name. gRPC does not support `stop_token_ids` in 0.3.

The async client uses `async with`, awaited unary methods, and `async for`
streams. Errors preserve gRPC status codes. See the
[gRPC and migration guide](python-sdk-0.2.md) for complete examples.

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

client = KapslShmClient(shm_name="kapsl-shm-default")
```

`infer(shape, dtype, data, *, model_id=0)` returns the output bytes. Streaming,
session metadata, and additional named inputs use `KapslClient` instead.

---

## KapslHybridClient

Hybrid IPC + shared-memory client.

```python
from kapsl_sdk import KapslHybridClient

client = KapslHybridClient(
    shm_name="kapsl-shm-default",
    socket_path="/tmp/kapsl.sock",
)
```

`infer(shape, dtype, data, *, model_id=0)` returns the output bytes. Streaming,
session metadata, and additional named inputs use `KapslClient` instead.

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
| `string` | UTF-8 bytes | Variable |
