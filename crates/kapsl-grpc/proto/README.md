The Open Inference Protocol schema is vendored from
[kserve/open-inference-protocol](https://github.com/kserve/open-inference-protocol/blob/d49cc23f89d709d87b210ef9449e273ae243984e/specification/protocol/open_inference_grpc.proto)
at commit `d49cc23f89d709d87b210ef9449e273ae243984e`, with only trailing whitespace removed.
Its Apache 2.0 copyright notice is retained.

`kapsl_inference.proto` adds model discovery and one-request, server-streaming
inference. It imports the standard tensor messages without modifying the
standard service. It does not implement Triton's bidirectional ModelStreamInfer.

These files are the language-neutral client contract. Client generation requires
no runtime, scheduler, or backend dependency. See ../../../docs/grpc.md for usage.
