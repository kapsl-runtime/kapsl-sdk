# Streaming Inference

`infer_stream` sends a request and receives output as an iterator of chunks. This is designed for LLMs that generate tokens one at a time, giving the caller each token as soon as it is produced rather than waiting for the full response.

## Basic usage

```python
from kapsl_sdk import KapslClient

client = KapslClient()

# Encode prompt as token IDs
prompt_tokens = tokenizer.encode("What is the capital of France?")
import numpy as np
token_array = np.array([prompt_tokens], dtype=np.int64)

# Stream the response
for chunk in client.infer_stream(
    model_id=0,
    shape=list(token_array.shape),
    dtype="int64",
    data=token_array.tobytes(),
):
    # Each chunk is raw bytes for one output token
    token_id = int(np.frombuffer(chunk, dtype=np.int64)[0])
    print(tokenizer.decode([token_id]), end="", flush=True)

print()  # newline after generation
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
