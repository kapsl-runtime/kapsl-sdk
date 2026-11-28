# kapsl-sdk

Python clients for Kapsl inference and orchestration. Supports CPython 3.10+
through Rust abi3 bindings, with optional gRPC dependencies.

```bash
pip install kapsl-sdk         # native socket/TCP, SHM, hybrid
pip install 'kapsl-sdk[grpc]' # adds gRPC and grpc.aio clients
```

## Native inference

```python
import struct
from kapsl_sdk import KapslClient

with KapslClient(api_token="native-token") as client:
    result = client.infer(
        0, [2], "float32", struct.pack("<2f", 1.0, 2.0), timeout_ms=5000,
    )
```

## Typed gRPC streaming

```python
from kapsl_sdk import KapslGrpcClient

with KapslGrpcClient("127.0.0.1:9097", api_token="reader-token") as client:
    print(client.list_models())
    with client.infer_stream_tensors(
        "my-model", [1], "string", b"Hello", timeout_ms=30_000,
        max_new_tokens=128,
    ) as stream:
        for tensor in stream:
            print(tensor.data.decode("utf-8"), end="", flush=True)
```

`AsyncKapslGrpcClient` supports awaited unary methods and async stream
iteration. Native streams also support `async for`. Streams expose explicit
cancellation and close operations; deadlines cover their complete lifetime.
Generated protobuf modules are bundled, so consumers do not need protoc.

## HTTP management and OpenAI-compatible API

```python
from kapsl_sdk import KapslHttpClient

with KapslHttpClient("http://127.0.0.1:9095", api_token="admin-token") as http:
    result = http.start_model("/models/assistant.aimod", wait=True)
    print(result.status, result.reason)          # active, or denied + reason
    print([m.idle_unloaded for m in http.list_models()])
    reply = http.chat_completion("assistant", [{"role": "user", "content": "Hi"}])
    print(reply.text, reply.finish_reason)
```

`KapslHttpClient` uses only the standard library. It covers health, hardware,
system and memory-authority statistics, `/metrics`, model lifecycle with
memory-admission results, idle-unload state, autoscaling policy, access
management, and the `/v1` models, chat completion (streaming and
non-streaming) and embeddings routes. Errors raise `KapslHttpError` with the
HTTP status, the engine's `reason`, and `retry_after_seconds`.

`KapslShmClient` and `KapslHybridClient` provide local unary tensor inference
with `infer(shape, dtype, data, *, model_id=0)`. All tensor clients return bytes;
`infer_tensor` preserves shape/dtype and `infer_stream_tensors` yields `Tensor`
objects on native/gRPC clients. SHM uses request-owned response mailboxes and
shared allocation leases to isolate concurrent requests.

## Version 0.3.0

Adds `KapslHttpClient` and its typed results. Native, SHM, hybrid and gRPC
protocols are unchanged from 0.2.0 and need engine 0.2.8 or later; the HTTP
routes are available from engine 0.2.8. Waited model starts with
memory-admission results, idle-unload fields, CUDA/GPU-sampling statistics,
and redacted role-token status need engine 0.3.0. See the repository README
for the full table and known engine 0.3.0 behaviour.

## Version 0.2.0 migration

Upgrade and restart the Python clients and engine together. This Python
version uses `kapsl-transport`, `kapsl-ipc`, `kapsl-shm`, and
`kapsl-communication` 0.4.0 with SHM protocol/region version 3, and
`kapsl-grpc` 0.3.0. Old native encodings and SHM layouts are rejected. There is
no fallback decoder, unleased memory access, or automatic inference replay.

Request options include timeouts, priority, force-CPU, request/model IDs, and
supported generation controls. The engine remains responsible for scheduling,
authorization, and memory governance. Native TCP uses its configured TCP
token; gRPC uses public API reader tokens. TLS/mTLS gRPC connections require
a TLS endpoint such as a terminating proxy.

See the [migration and API guide](https://github.com/kapsl-runtime/kapsl-sdk/blob/main/docs/python-sdk-0.2.md).
The package includes `list_voices()` and `load_voice()` for bundled embeddings.

## License

Proprietary — see [kapsl.ai](https://kapsl.ai) for licensing information.
