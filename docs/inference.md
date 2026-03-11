# Inference

## Basic inference — `infer()`

`infer` sends one tensor to the runtime and returns the raw output bytes.

```python
import numpy as np
from kapsl_sdk import KapslClient

client = KapslClient()

# Build input tensor
x = np.random.rand(1, 3, 224, 224).astype(np.float32)

result_bytes = client.infer(
    model_id=0,
    shape=list(x.shape),   # [1, 3, 224, 224]
    dtype="float32",
    data=x.tobytes(),
)

# Decode — you must know the output shape for this model
logits = np.frombuffer(result_bytes, dtype=np.float32)
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `model_id` | `int` | Numeric ID of the loaded model |
| `shape` | `list[int]` | Input tensor dimensions |
| `dtype` | `str` | Data type: `float32`, `float64`, `float16`, `int32`, `int64`, `uint8` |
| `data` | `bytes` | Raw tensor bytes in row-major order |
| `additional_inputs` | `dict` | Extra named tensors (optional, see below) |
| `session_id` | `str` | Session identifier for stateful models (optional) |

### Return value

`bytes` — raw output tensor data. The shape and dtype depend on the model. If you need them, use `infer_tensor` instead.

---

## Typed inference — `infer_tensor()`

`infer_tensor` returns a `(data, shape, dtype)` tuple so you can interpret the output without knowing the model's output layout in advance. Use this for models with variable-length output (TTS, diffusion, video generation).

```python
data_bytes, out_shape, out_dtype = client.infer_tensor(
    model_id=0,
    shape=[1, 3, 224, 224],
    dtype="float32",
    data=x.tobytes(),
)

output = np.frombuffer(data_bytes, dtype=np.dtype(out_dtype)).reshape(out_shape)
```

### Return value

`(bytes, list[int], str)` — `(data, shape, dtype_string)`

---

## Multi-input models — `additional_inputs`

Some models require more than one named input tensor. Pass them via the `additional_inputs` dictionary.

Each entry maps an input name to a tuple of `(shape, dtype, data)`:

```python
additional_inputs = {
    "input_name": ([dim1, dim2, ...], "dtype", bytes_value),
}
```

### Example: image segmentation with a mask

```python
image = np.random.rand(1, 3, 512, 512).astype(np.float32)
mask = np.ones((1, 1, 512, 512), dtype=np.float32)

result_bytes = client.infer(
    model_id=0,
    shape=[1, 3, 512, 512],
    dtype="float32",
    data=image.tobytes(),
    additional_inputs={
        "mask": ([1, 1, 512, 512], "float32", mask.tobytes()),
    },
)
```

### Example: Kokoro TTS with style and speed

```python
import numpy as np
from kapsl_sdk import KapslClient, load_voice

client = KapslClient()

# Tokenise text → input_ids (int64)
tokens = tokenizer.encode("Hello, world!")
input_ids = np.array([tokens], dtype=np.int64)

# Load a bundled voice and select the style vector for this token length
voice_data = load_voice("af_bella")
voices = np.frombuffer(voice_data, dtype=np.float32).reshape(-1, 1, 256)
style = voices[len(tokens)].reshape(1, 1, 256)

speed = np.array([[1.0]], dtype=np.float32)

data_bytes, shape, dtype = client.infer_tensor(
    model_id=0,
    shape=list(input_ids.shape),
    dtype="int64",
    data=input_ids.tobytes(),
    additional_inputs={
        "style": (list(style.shape), "float32", style.tobytes()),
        "speed": (list(speed.shape), "float32", speed.tobytes()),
    },
)

# Reconstruct PCM audio (float32 at 24 kHz)
audio = np.frombuffer(data_bytes, dtype=np.float32).reshape(shape)
```

---

## Session IDs

Pass a `session_id` to associate a request with a logical session. This is used for stateful or multi-step inference (e.g., autoregressive generation, diffusion denoising steps).

```python
import uuid

session = str(uuid.uuid4())

for step in range(num_steps):
    output = client.infer(
        model_id=0,
        shape=[1, latent_dim],
        dtype="float32",
        data=latent.tobytes(),
        session_id=session,
    )
```

The runtime may use the session ID to route requests to the same replica or maintain step-level state, depending on the model and scheduler configuration.

---

## Supported dtypes

| dtype string | NumPy equivalent |
|-------------|-----------------|
| `float32` | `np.float32` |
| `float64` | `np.float64` |
| `float16` | `np.float16` |
| `int32` | `np.int32` |
| `int64` | `np.int64` |
| `uint8` | `np.uint8` |
