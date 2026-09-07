# kapsl-grpc

Reusable gRPC clients, protobuf contracts, tensor conversion, and server for
Kapsl. Model backends, scheduling, memory governance, and credential policy stay
in the embedding runtime, supplied through `EngineFacade` and
`RequestAuthorizer`.

```toml
[dependencies]
kapsl-grpc = "0.3.0"
```

## Services

- `inference.GRPCInferenceService`: Open Inference Protocol / KServe V2 health,
  readiness, metadata, and unary `ModelInfer`.
- `kapsl.v1.KapslInference`: model discovery and `InferStream`, which accepts one
  inference request and streams typed response packets.

Rust clients are generated at build time and exposed as
`kapsl_grpc::inference::grpc_inference_service_client::GrpcInferenceServiceClient`
and `kapsl_grpc::kapsl::v1::kapsl_inference_client::KapslInferenceClient`.
The crate bundles its protobuf files and a build-time `protoc` dependency; no
separate system `protoc` installation is needed for Rust consumers. The same
schemas can generate Python, Go, Java, C#, and other gRPC clients.

## Runtime integration

Use `start_server` with an `EngineFacade`, a `RequestAuthorizer`, and a
`GrpcServerConfig`. Every RPC invokes authorization, including health and model
discovery. Request deadlines, client cancellation, and server shutdown propagate
cancellation to the engine. Responses are polled on demand. Access records use
the embedding process's `log` subscriber.

The initial protocol supports single-output models with FP16, FP32, FP64,
INT32, INT64, UINT8, or a single UTF-8 BYTES value. Triton's standard unary
clients can use the supported KServe RPCs. Use the generated Kapsl service for
server streaming; Triton's bidirectional streaming and management extensions
are outside this API.

The server listener is plaintext HTTP/2. External deployments should terminate
TLS at an HTTP/2 reverse proxy and provide their runtime authorization policy.
See the [transport guide](https://github.com/kapsl-runtime/kapsl-sdk/blob/kapsl-grpc-v0.3.0/docs/grpc.md)
for request parameters, encoding rules, limits, and Python client generation.

For a local demonstration without model files, run
`cargo run -p kapsl-grpc --example echo_server` from the SDK repository.
