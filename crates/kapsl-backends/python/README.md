# Kapsl PyTorch sidecar worker

Reference Python worker for the Rust `PyTorchBackend` (`kapsl-backends`, feature
`pytorch`). It runs PyTorch-family models (HuggingFace `transformers`, or custom
code) in a separate process and serves text generation to the engine over a unix
domain socket.

## Why a sidecar

The PyTorch model runs **out of process**, so:
- it can be *any* PyTorch model (not just TorchScript-exportable ones);
- a native crash in PyTorch cannot take down the engine;
- it participates in the engine's `GlobalKvScheduler` at the **admission** level
  (wired on the Rust side), but has its **own CUDA context and KV cache** — it
  does *not* share the engine's in-process KV block pool.

## Install

```bash
pip install torch transformers          # or your project's pinned versions
export PYTHONPATH="$PWD/crates/kapsl-backends/python:$PYTHONPATH"
```

Build the engine with the backend enabled:

```bash
cargo build -p kapsl-backends --features pytorch   # (or the workspace bin)
```

A model is selected for this backend when its manifest `framework` is
`"pytorch"` (also inferred from `.pt` / `.pth` / `.safetensors`).

## How the engine launches it

The Rust side spawns, by default:

```
python3 -m kapsl_pytorch_worker --model <path> --socket <path> --device <id>
```

Override the whole command with the `KAPSL_PYTORCH_WORKER` env var
(space-separated), e.g. to use a venv interpreter or a vLLM-backed worker that
speaks the same protocol:

```bash
export KAPSL_PYTORCH_WORKER="/opt/venv/bin/python -m my_vllm_worker"
```

## Protocol (line-delimited JSON, one connection per request)

Readiness: the worker binds the socket **after** the model finishes loading, so
the engine's connect-poll only succeeds once it can serve.

Request (single line + `\n`):

```json
{"prompt":"...","max_new_tokens":256,"temperature":0.7,"top_p":0.9,"top_k":40,
 "repetition_penalty":1.1,"seed":null,"stop_token_ids":[],"session_id":null,
 "request_id":null}
```

Response (one JSON object per line, streamed):

```json
{"type":"chunk","text":"..."}
{"type":"done"}
{"type":"error","message":"..."}
```

## Scope

This reference handles one request per connection with `transformers.generate`
+ `TextIteratorStreamer`. Continuous batching / paged attention is out of scope
here — a vLLM- or TGI-backed worker can replace it behind the same protocol with
no Rust changes.
