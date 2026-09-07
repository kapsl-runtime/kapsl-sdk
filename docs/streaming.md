# Streaming Inference

`infer_stream` sends a request and receives output as an iterator of chunks. This is designed for LLMs that generate tokens one at a time, giving the caller each token as soon as it is produced rather than waiting for the full response.

In Python 0.2.0, streams support explicit cancellation, full-stream deadlines,
context managers, and async iteration. `infer_stream_tensors` preserves output
shape/dtype in `Tensor` objects. The native and gRPC clients share these methods;
see the [gRPC and async examples](python-sdk-0.2.md#stream-ownership-and-async-use).

Use a context manager when stopping before EOF:

```python
with client.infer_stream(0, [1], "string", b"Hello", timeout_ms=30_000) as stream:
    for chunk in stream:
        if should_stop(chunk):
            break
        process(chunk)
```

`stream.cancel()` and `stream.close()` interrupt blocked reads. Client shutdown
also cancels its streams. A deadline applies even while the caller is idle;
native clients raise `TimeoutError`, and gRPC clients preserve the
`DEADLINE_EXCEEDED` status. Stream errors are raised once and terminate iteration.

## Basic usage — text in, text out

For LLMs that accept a raw text prompt and stream back text chunks (e.g. GGUF models):

```python
from kapsl_sdk import KapslClient

client = KapslClient()

prompt = "<|im_start|>user\nWhat is the capital of France?<|im_end|>\n<|im_start|>assistant\n"

stream = client.infer_stream(
    model_id=0,
    shape=[1, 1],
    dtype="string",
    data=prompt.encode("utf-8"),
)

# Print tokens as they arrive
for chunk in stream:
    print(chunk.decode("utf-8"), end="", flush=True)

print()  # newline after generation
```

To collect the full response instead of printing:

```python
full_response = b"".join(stream)
print(full_response.decode("utf-8"))
```

## Basic usage — token IDs in, token IDs out

For ONNX-based LLMs that operate on integer token arrays:

```python
import numpy as np
from kapsl_sdk import KapslClient

client = KapslClient()

prompt_tokens = tokenizer.encode("What is the capital of France?")
token_array = np.array([prompt_tokens], dtype=np.int64)

for chunk in client.infer_stream(
    model_id=0,
    shape=list(token_array.shape),
    dtype="int64",
    data=token_array.tobytes(),
):
    token_id = int(np.frombuffer(chunk, dtype=np.int64)[0])
    print(tokenizer.decode([token_id]), end="", flush=True)

print()
```

## Parameters

`infer_stream` accepts the same parameters as `infer`:

| Parameter | Type | Description |
|-----------|------|-------------|
| `model_id` | `int` | Numeric model ID |
| `shape` | `list[int]` | Input tensor dimensions |
| `dtype` | `str` | Input data type |
| `data` | `bytes` | Input tensor bytes |
| `additional_inputs` | `dict` | Extra named tensors (optional) |
| `session_id` | `str` | Session ID for stateful generation (optional) |

## Return value

A Python iterator. Each iteration yields `bytes` — one chunk of output data. The iterator stops when the runtime sends the end-of-stream signal.

> `infer_stream` opens a dedicated connection per call and does **not** use the connection pool. The connection is held open for the duration of the stream.

## Collecting all chunks

If you need the full output as one buffer:

```python
chunks = list(client.infer_stream(
    model_id=0,
    shape=[1, len(prompt_tokens)],
    dtype="int64",
    data=token_array.tobytes(),
))

all_data = b"".join(chunks)
```

## With authentication

```python
client = KapslClient("tcp://192.168.1.10:9096", api_token="my-token")

for chunk in client.infer_stream(model_id=0, shape=[1, 8], dtype="int64", data=input_bytes):
    ...
```

Authentication tokens are included in the stream request automatically.

## Error handling

If the runtime returns an error mid-stream, the iterator raises `RuntimeError`:

```python
try:
    for chunk in client.infer_stream(model_id=0, ...):
        process(chunk)
except RuntimeError as e:
    print(f"Streaming failed: {e}")
```

## Relationship to batch inference

`infer_stream` and `infer` target the same model but use different wire operations:

| | `infer` | `infer_stream` |
|-|---------|----------------|
| Wire op | `OP_INFER` | `OP_INFER_STREAM` |
| Response | Single tensor | Iterator of chunks |
| Connection | Pooled | Dedicated per call |
| Use case | Classification, ONNX models | LLMs, token generation |
