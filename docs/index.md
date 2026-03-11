# kapsl-sdk

`kapsl-sdk` is the official Python client library for [kapsl-runtime](https://kapsl.ai) — a high-performance Rust-based inference runtime for local and embedded AI model execution.

Use `kapsl-sdk` to connect a Python application to a running `kapsl-runtime` instance and run inference over any packaged model.

## What it provides

- **`KapslClient`** — general-purpose client over Unix socket, TCP, or Windows named pipe
- **`KapslShmClient`** — zero-copy shared-memory client for maximum local throughput
- **`KapslHybridClient`** — combines IPC signaling with shared-memory payloads
- Bundled voice embeddings for TTS models (`load_voice`, `list_voices`)
- CPython 3.9 – 3.13 support via PyO3 / Maturin

## When to use it

Use `kapsl-sdk` when your Python service or tool needs to:

- Run inference on an AI model hosted by `kapsl-runtime`
- Stream token-by-token output from an LLM
- Send multi-input inference requests (e.g., Kokoro TTS, diffusion models)
- Call the runtime from a backend service with low latency requirements

`kapsl-sdk` is a **client library**, not the runtime server itself. You need a running `kapsl-runtime` process to use it.

## Navigation

| Page | Description |
|------|-------------|
| [Installation](./installation.md) | Install the package and requirements |
| [Quick Start](./quickstart.md) | Run your first inference in minutes |
| [Client Types](./client-types.md) | Choosing the right transport |
| [Authentication](./authentication.md) | API token setup |
| [Inference](./inference.md) | `infer()`, `infer_tensor()`, `additional_inputs` |
| [Streaming](./streaming.md) | Token-by-token streaming with `infer_stream()` |
| [TTS & Voices](./tts-voices.md) | Kokoro TTS and bundled voice embeddings |
| [API Reference](./api-reference.md) | Full method signatures and parameters |
