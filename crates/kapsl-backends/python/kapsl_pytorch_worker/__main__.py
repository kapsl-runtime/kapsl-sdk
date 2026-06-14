"""Reference PyTorch sidecar worker for the Kapsl PyTorchBackend.

Loads a HuggingFace `transformers` causal-LM and serves text generation over a
unix domain socket using the line-delimited JSON protocol the Rust
`PyTorchBackend` expects:

    request  (one line):  {"prompt": "...", "max_new_tokens": 256, ...}
    response (per line):  {"type":"chunk","text":"..."}
                          {"type":"done"}
                          {"type":"error","message":"..."}

Readiness is signalled by binding the socket *after* the model finishes loading,
so the Rust side's connect-poll only succeeds once we can serve.

Run:
    python3 -m kapsl_pytorch_worker --model <path> --socket <path> --device <id>

This is a minimal, single-request-at-a-time reference. Batching / continuous
batching (e.g. a vLLM-backed worker) can replace it behind the same protocol.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import threading


def log(msg: str) -> None:
    print(f"[kapsl-pytorch-worker] {msg}", file=sys.stderr, flush=True)


def resolve_device(device_id: int) -> str:
    try:
        import torch
    except ImportError:
        return "cpu"
    if device_id >= 0 and torch.cuda.is_available():
        return f"cuda:{device_id}"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class Generator:
    def __init__(self, model_path: str, device: str) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        log(f"loading model from {model_path} on {device}")
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path)
        self.model.to(device)
        self.model.eval()
        log("model loaded")

    def generate(self, req: dict, emit) -> None:
        """Stream generated text chunks via emit(text)."""
        import torch
        from transformers import TextIteratorStreamer

        prompt = req.get("prompt", "")
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        streamer = TextIteratorStreamer(
            self.tokenizer, skip_prompt=True, skip_special_tokens=True
        )
        gen_kwargs = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=int(req.get("max_new_tokens") or 512),
            do_sample=(req.get("temperature") or 0.0) > 0.0,
        )
        if req.get("min_new_tokens") is not None:
            gen_kwargs["min_new_tokens"] = int(req["min_new_tokens"])
        if req.get("temperature") is not None:
            gen_kwargs["temperature"] = float(req["temperature"])
        if req.get("top_p") is not None:
            gen_kwargs["top_p"] = float(req["top_p"])
        if req.get("top_k") is not None:
            gen_kwargs["top_k"] = int(req["top_k"])
        if req.get("repetition_penalty") is not None:
            gen_kwargs["repetition_penalty"] = float(req["repetition_penalty"])
        if req.get("stop_token_ids"):
            gen_kwargs["eos_token_id"] = list(req["stop_token_ids"])
        if req.get("seed") is not None:
            torch.manual_seed(int(req["seed"]))

        thread = threading.Thread(target=self.model.generate, kwargs=gen_kwargs)
        thread.start()
        for text in streamer:
            if text:
                emit(text)
        thread.join()


def serve(generator: Generator, socket_path: str) -> None:
    if os.path.exists(socket_path):
        os.remove(socket_path)

    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    # Bind only after the model is ready — this is the readiness signal.
    server.bind(socket_path)
    server.listen(16)
    log(f"listening on {socket_path}")

    while True:
        conn, _ = server.accept()
        threading.Thread(
            target=handle_connection, args=(generator, conn), daemon=True
        ).start()


def handle_connection(generator: Generator, conn: socket.socket) -> None:
    with conn:
        reader = conn.makefile("r", encoding="utf-8")
        line = reader.readline()
        if not line:
            return

        def send(obj: dict) -> None:
            conn.sendall((json.dumps(obj) + "\n").encode("utf-8"))

        try:
            req = json.loads(line)
            generator.generate(req, lambda text: send({"type": "chunk", "text": text}))
            send({"type": "done"})
        except Exception as exc:  # noqa: BLE001 - report any failure to the client
            log(f"generation error: {exc}")
            try:
                send({"type": "error", "message": str(exc)})
            except OSError:
                pass


def main() -> None:
    parser = argparse.ArgumentParser(prog="kapsl_pytorch_worker")
    parser.add_argument("--model", required=True)
    parser.add_argument("--socket", required=True)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()

    device = resolve_device(args.device)
    generator = Generator(args.model, device)
    serve(generator, args.socket)


if __name__ == "__main__":
    main()
