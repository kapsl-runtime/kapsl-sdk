# TTS & Voice Embeddings

`kapsl-sdk` ships with bundled voice embedding files for the Kokoro TTS model. These are `.bin` files included in the Python wheel — no separate download needed.

## List available voices

```python
from kapsl_sdk import list_voices

voices = list_voices()
print(voices)
# ['af_bella', 'af_heart', 'af_jessica', 'am_adam', 'bf_emma', ...]
```

## Load a voice embedding

```python
from kapsl_sdk import load_voice

raw = load_voice("af_bella")  # returns bytes
```

The returned bytes are raw `float32` values. Reshape as `(-1, 1, 256)` to index by token length:

```python
import numpy as np

voice_data = load_voice("af_bella")
voices = np.frombuffer(voice_data, dtype=np.float32).reshape(-1, 1, 256)
```

## Full Kokoro TTS example

Kokoro is an ONNX-based TTS model that requires three inputs:

| Name | Shape | dtype | Description |
|------|-------|-------|-------------|
| `input_ids` (primary) | `[1, T]` | `int64` | Token IDs for the text |
| `style` | `[1, 1, 256]` | `float32` | Style vector for this token length |
| `speed` | `[1, 1]` | `float32` | Speaking speed multiplier (1.0 = normal) |

Output: `float32` PCM audio at 24 kHz.

```python
import numpy as np
from kapsl_sdk import KapslClient, load_voice

client = KapslClient()

# 1. Tokenise the input text
text = "Hello, welcome to Kapsl."
tokens = tokenizer.encode(text)  # list of int token IDs
input_ids = np.array([tokens], dtype=np.int64)  # shape [1, T]

# 2. Load voice and select the style vector for this token length
voice_data = load_voice("af_bella")
voices = np.frombuffer(voice_data, dtype=np.float32).reshape(-1, 1, 256)

# Index by token length (clamped to the maximum available)
token_len = min(len(tokens), voices.shape[0] - 1)
style = voices[token_len].reshape(1, 1, 256)  # [1, 1, 256]

# 3. Set speaking speed
speed = np.array([[1.0]], dtype=np.float32)  # [1, 1]

# 4. Run inference
data_bytes, out_shape, out_dtype = client.infer_tensor(
    model_id=0,
    shape=list(input_ids.shape),
    dtype="int64",
    data=input_ids.tobytes(),
    additional_inputs={
        "style": (list(style.shape), "float32", style.tobytes()),
        "speed": (list(speed.shape), "float32", speed.tobytes()),
    },
)

# 5. Reconstruct audio
audio = np.frombuffer(data_bytes, dtype=np.float32).reshape(out_shape)

# 6. Write to a WAV file
import wave, struct
sample_rate = 24000
with wave.open("output.wav", "w") as wf:
    wf.setnchannels(1)
    wf.setsampwidth(2)   # 16-bit
    wf.setframerate(sample_rate)
    pcm_int16 = (audio.flatten() * 32767).astype(np.int16)
    wf.writeframes(pcm_int16.tobytes())
```

## Adjusting speed

Pass any positive `float32` value as the speed multiplier:

```python
# 0.8 = slightly slower, 1.2 = slightly faster
speed = np.array([[0.8]], dtype=np.float32)
```

## Using a different voice

```python
voice_data = load_voice("am_adam")   # male voice
voices = np.frombuffer(voice_data, dtype=np.float32).reshape(-1, 1, 256)
```

Use `list_voices()` to see all available names.

## Voice file format

Each `.bin` file contains a sequence of `float32` style vectors, one per input token length, each of dimension 256. Reshaping as `(-1, 1, 256)` gives an array where `voices[i]` is the style vector to use when the input has `i` tokens.
