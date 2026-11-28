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

## KapslHttpClient

Client for the engine's HTTP API (`kapsl run --metrics-port`, default 9095).
Added in 0.3.0; it needs no optional dependencies.

```python
KapslHttpClient(
    base_url: str = "http://127.0.0.1:9095",
    *,
    api_token: str | None = None,
    timeout_ms: int | None = None,
    ssl_context: ssl.SSLContext | None = None,
)
```

`base_url` may end in `/api` or `/v1`. `api_token` is sent as
`Authorization: Bearer TOKEN`. `timeout_ms` bounds connecting and each socket
read, not a whole request; every method also takes `timeout_ms`. Connection
failures raise `urllib.error.URLError`. Error responses raise `KapslHttpError`
with `status`, `message`, `reason`, `error_type`, `retry_after_seconds`, and
the decoded `body`.

Every result is a frozen dataclass. Fields an older engine omits are `None` or
`False`; response fields the SDK does not model are kept in `extra`.

| Method | Route | Role | Result | Engine |
|--------|-------|------|--------|--------|
| `health()` | `GET /api/health` | reader | `HealthStatus` | 0.2.8 |
| `hardware()` | `GET /api/hardware` | reader | `dict` | 0.2.8 |
| `system_stats()` | `GET /api/system/stats` | reader | `SystemStats` | 0.2.8; `cuda` and `gpu_sampling*` 0.3.0 |
| `metrics()` | `GET /metrics` | admin | Prometheus text | 0.2.8 |
| `list_models()` | `GET /api/models` | reader | `list[ModelStatus]` | 0.2.8; idle fields 0.3.0 |
| `get_model(model_id)` | `GET /api/models/{id}` | reader | `ModelStatus` | 0.2.8 |
| `start_model(model_path, *, model_id, topology, tp_degree, idle_unload_seconds, wait)` | `POST /api/models/start` | admin | `ModelLoadResult` | 0.2.8; `wait` and `idle_unload_seconds` 0.3.0 |
| `stop_model(model_id)` | `POST /api/models/{id}/stop` | admin | message | 0.2.8 |
| `remove_model(model_id)` | `POST /api/models/{id}/remove` | admin | message | 0.2.8 |
| `get_scaling_policy(model_id)` | `GET /api/models/{id}/scaling` | reader | `ScalingPolicy` | 0.2.8 |
| `set_scaling_policy(model_id, policy)` | `POST /api/models/{id}/scaling` | admin | `None` | 0.2.8 |
| `login(token=None)` | `POST /api/auth/login` | public | `AuthLogin` | 0.2.8 |
| `auth_status()` | `GET /api/auth/access/status` | admin | `AuthStatus` | 0.2.8 |
| `role_summaries()` | `GET /api/auth/access/roles` | admin | `list[RoleSummary]` | 0.2.8 |
| `role_token_status()` | `GET /api/auth/roles` | admin | `RoleTokenStatus` | 0.3.0 (redacted) |
| `set_role_tokens(*, reader_token, writer_token, admin_token)` | `POST /api/auth/roles` | admin | `RoleTokenStatus` | 0.2.8 |
| `list_users()` | `GET /api/auth/access/users` | admin | `list[AuthUser]` | 0.2.8 |
| `create_user(username, role, *, display_name, status)` | `POST /api/auth/access/users` | admin | `AuthUser` | 0.2.8 |
| `update_user(user_id, *, display_name, role, status)` | `PATCH /api/auth/access/users/{id}` | admin | `AuthUser` | 0.2.8 |
| `list_api_keys(user_id=None)` | `GET /api/auth/access/keys` | admin | `list[ApiKey]` | 0.2.8 |
| `create_api_key(user_id, name, *, scopes, expires_in_days)` | `POST /api/auth/access/users/{id}/keys` | admin | `CreatedApiKey` | 0.2.8 |
| `revoke_api_key(key_id)` | `POST /api/auth/access/keys/{id}/revoke` | admin | `ApiKey` | 0.2.8 |
| `list_openai_models()` | `GET /v1/models` | reader | `list[OpenAIModel]` | 0.2.8 |
| `get_openai_model(model)` | `GET /v1/models/{model}` | reader | `OpenAIModel` | 0.2.8 |
| `chat_completion(model, messages, ...)` | `POST /v1/chat/completions` | reader | `ChatCompletion` | 0.2.8 |
| `chat_completion_stream(model, messages, ..., include_usage=False)` | `POST /v1/chat/completions` | reader | stream of `ChatCompletionChunk` | 0.2.8 |
| `embeddings(model, input, *, encoding_format, dimensions, user)` | `POST /v1/embeddings` | reader | `EmbeddingList` | 0.2.8 |

### Model lifecycle

`start_model` loads a package from a path on the engine host. Without `wait`
it returns `ModelLoadResult(status="started")` as soon as the load is queued.
With `wait=True` it returns when the load is `active`, or `denied` by the
engine (HTTP 409): `reason="memory_admission"` with `requested` bytes per
memory domain, or `reason="device_unavailable"` with `device_id`. Other
failures raise `KapslHttpError`. Use a `timeout_ms` that covers the load.
`idle_unload_seconds` sets this model's idle timeout; `0` keeps it loaded.

`ModelStatus.idle_unloaded` means the idle unloader stopped the model, and
`reloads_on_request` means the next request loads it again (`--idle-reload`).
Stopping an idle-unloaded model keeps it down. If a reload is denied memory,
HTTP requests fail with status 503 and `retry_after_seconds == 10`.

`set_scaling_policy` replaces every field; change one with
`dataclasses.replace(client.get_scaling_policy(0), max_replicas=2)`.
`ScalingPolicy.validate()` applies the engine's rules before sending.

### Chat completions

`messages` are `{"role": ..., "content": ...}` mappings or earlier
`ChatCompletionMessage` results. Generation keywords are `max_tokens`,
`max_completion_tokens`, `temperature`, `top_p`, `top_k`,
`repetition_penalty`, `seed`, `stop` and `user`. `session_id` is sent as the
`X-Kapsl-Session` header for KV-cache affinity. `extra_body` adds fields the
method does not name; managed vLLM models receive them unchanged.

`chat_completion_stream` returns a stream that supports `for`, `async for`,
`with`, `cancel()` and `close()`. Closing it disconnects and cancels
generation. A failure reported inside the stream raises `KapslHttpError`
with `status == 200`; a connection that ends before `[DONE]` raises
`ConnectionError`. `ChatCompletion.finish_reason` and
`ChatCompletionChunk.finish_reason` are reported unchanged: engine 0.3.0
returns `"stop"` for truncated replies except on managed vLLM models.

### Access management

`create_api_key` returns `CreatedApiKey(api_key, raw_key)`. The secret is only
returned here and is left out of `repr`. A key has its user's role; `scopes`
such as `api:read` narrow it. `set_role_tokens` replaces all three legacy
role tokens and requires `admin_token` whenever another token is set.

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
