# Python SDK 0.2.0

This version packages the current native transports and optional gRPC clients.
It removes old wire decoding, the SHM response queue and notification pipes,
unleased hybrid memory access, and the allocator's overwrite-on-exhaustion API.
Requests with incomplete metadata or trailing bytes are rejected.

## Upgrade together

| Component | Required version |
| --- | --- |
| Python `kapsl-sdk` | 0.2.0 |
| Rust `kapsl-transport`, `kapsl-ipc`, `kapsl-shm`, `kapsl-communication` | 0.4.0 |
| Rust `kapsl-grpc` | 0.3.0 |
| Native tensor inference envelope | `KIRQ` + little-endian uint16 version 1 |
| Direct and hybrid SHM wire/region version | 3 |

Upgrade the engine and Python processes together and restart them to recreate
their shared-memory regions. There is no downgrade negotiation or legacy
decoder. Python 0.1.x is not a supported native client for this transport
release. Rust clients implementing `TransportClient` must implement
`infer_request` and preserve its metadata.

Native `OP_INFER` and `OP_INFER_STREAM` bodies start with `KIRQ` and a uint16
version, followed by the complete bincode request. The shared
`write_request_value` and `encode_inference_request` helpers add this envelope;
`decode_inference_request` verifies it before decoding fields. The fixed frame
headers and the separate versioned OpenAI wire operations retain their current
contracts.

Python and Rust package versions are independent. To build this source before
the Python release reaches PyPI:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install maturin
maturin develop --release
pip install 'grpcio>=1.71.2,<2' 'protobuf>=5.29,<7'
```

Released wheels use the CPython 3.9 stable ABI. gRPC is optional; a native-only
installation imports and works without grpcio or protobuf.

## Request options

`KapslClient` and both gRPC clients accept these keyword options on `infer`,
`infer_tensor`, `infer_stream`, and `infer_stream_tensors`:

| Option | Meaning |
| --- | --- |
| `timeout_ms` | Positive deadline covering connection, inference, and the entire stream |
| `request_id` | Caller-supplied correlation ID |
| `priority` | `0` for latency critical; `1..255` for throughput |
| `force_cpu` | Request CPU execution, subject to backend support |
| `model_version` | Model version; gRPC resolves it before dispatch |
| `max_new_tokens`, `min_new_tokens` | Generation length bounds |
| `temperature`, `top_p`, `top_k`, `repetition_penalty`, `seed` | Generation overrides supported by the backend |
| `stop_token_ids` | Native transport only; gRPC 0.3 rejects this option |

Unknown options and invalid ranges raise an error before sending a request.
Credentials come from the constructor's `api_token`, not request options.
`additional_inputs={"mask": (shape, dtype, data)}` and `session_id` remain
available. The engine applies scheduling and memory governance.

The constructor's `timeout_ms` supplies a default. Passing `timeout_ms=None`
on an inference call disables that default for that call. Health/discovery
methods use the constructor default when their timeout argument is omitted or
`None`.

Native requests are never automatically replayed after an I/O error, because
the server may already have executed them. Handle retries at the application
level with awareness of model/session side effects.

## gRPC clients

```python
from kapsl_sdk import KapslGrpcClient

with KapslGrpcClient("127.0.0.1:9097", api_token="reader-token") as client:
    assert client.server_live()
    assert client.server_ready()
    print(client.server_metadata())
    print(client.list_models())
    print(client.model_metadata("my-model"))
    result, shape, dtype = client.infer_tensor(
        "my-model", [1], "string", b"Hello",
        timeout_ms=30_000, max_new_tokens=128, min_new_tokens=1,
    )
```

The configured port must be the engine's `--grpc-port`, distinct from its native
TCP listener. `model_id` accepts a model name or a numeric ID. The primary
input name defaults to `input`; use `input_name` to match model metadata.
Numeric tensor bytes must be little endian and match their shape. String
tensors contain one UTF-8 value with shape `[1]`; the client supplies the OIP
BYTES length prefix. The current service returns one output tensor.

Discovery methods expose generated protobuf messages, except `server_live`,
`server_ready`, and `model_ready` (booleans), and `list_models` (a list of model
messages). Advanced callers can use `client.inference_stub`,
`client.streaming_stub`, and `kapsl_sdk.grpc_protocol` directly.

The gRPC clients use `Authorization: Bearer ...` metadata. The engine's shared
authorization adapter accepts public API reader tokens and scopes session IDs.
Native TCP uses the separately configured `KAPSL_TCP_AUTH_TOKEN`.

For a TLS endpoint, set `tls=True`. `root_certificates` accepts CA PEM bytes;
mutual TLS additionally takes `private_key` and `certificate_chain` PEM bytes.
The current engine gRPC listener is plaintext, so TLS requires a terminating
proxy. `max_message_bytes` configures both client send and receive limits
(default 16 MiB); the server also enforces its own limit.

## Stream ownership and async use

`infer_stream` yields `bytes`. `infer_stream_tensors` yields immutable `Tensor`
objects with `data`, `shape`, and `dtype`. gRPC tensors also include output name,
model name/version, and request ID.

Use a context manager when stopping early. `close()`/`cancel()` interrupts a
blocked reader; closing a client cancels its active requests and streams.
Deadlines expire even while the caller is not reading. A server error ends
iteration after raising once.

```python
from kapsl_sdk import KapslClient

with KapslClient(api_token="native-token") as client:
    with client.infer_stream(0, [1], "string", b"Hello", timeout_ms=30_000) as stream:
        for chunk in stream:
            if should_stop(chunk):
                break
            consume(chunk)
```

Native streams also support `async for` and `async with`; reading runs on a
worker thread and task cancellation closes the underlying connection.
`AsyncKapslGrpcClient` uses `grpc.aio` for both unary calls and streams:

```python
from kapsl_sdk import AsyncKapslGrpcClient

async def generate():
    async with AsyncKapslGrpcClient("127.0.0.1:9097", api_token="reader-token") as client:
        assert await client.server_ready()
        async with client.infer_stream_tensors(
            "my-model", [1], "string", b"Hello", timeout_ms=30_000,
        ) as stream:
            async for tensor in stream:
                consume(tensor.data)
```

Async gRPC client/stream `close()` methods must be awaited; stream `cancel()` is
synchronous. Native failures raise `ConnectionError`, `RuntimeError`, or
`TimeoutError`; gRPC failures preserve `grpc.RpcError`/`grpc.aio.AioRpcError`
status codes.

SHM and hybrid clients expose unary `infer(shape, dtype, data, *, model_id=0)`
and return bytes. They use process-shared leases and direct-SHM mailboxes for
concurrent calls. Streaming, named inputs, sessions, and request options use
the native or gRPC clients.

## Validation and generation

Build and install the wheel before testing, so packaging omissions are caught:

```bash
pip install maturin pytest
maturin build --locked --out dist
cargo build -p kapsl-communication --example python_test_server --locked
python scripts/test-python-wheel.py
```

The fixture runs current TCP, local IPC, SHM, hybrid, and gRPC servers with an
echo scheduler. Tests cover metadata/auth, cancellation/deadlines, Unicode,
typed output, concurrent SHM calls, and rejection of a captured 0.1.23 request.
These tests do not exercise a model backend or GPU execution.

Generated clients are checked in. Regenerate in a separate environment with
`grpcio-tools==1.71.2`, then run `python scripts/generate-python-grpc.py`.
`--check` verifies consistency with the crate's canonical protos. A private
protobuf descriptor pool allows importing Triton and Kapsl in the same process.
