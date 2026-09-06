# gRPC transport

`kapsl-grpc` owns the reusable protobuf contract, generated Rust clients, tensor
conversion, and gRPC server. It is also available through
`kapsl-communication` with `default-features = false, features = ["grpc"]`.
The generated clients use normal HTTP/2 gRPC and do not depend on a model
backend. The crate implements no scheduler, memory-admission, or credential-store
policy. An embedding runtime supplies `EngineFacade` and `RequestAuthorizer`.
The server writes access events through the process's existing `log` subscriber.

The gRPC module exposes its generated clients directly; it does not implement
the native framed `TransportClient` protocol or select a transport automatically.
The `kapsl-engine-api` dependency shares the SDK workspace contract during
development and resolves from the registry when this crate is published.

## Protocol surface

| Service | RPCs |
| --- | --- |
| `inference.GRPCInferenceService` | `ServerLive`, `ServerReady`, `ServerMetadata`, `ModelReady`, `ModelMetadata`, `ModelInfer` |
| `kapsl.v1.KapslInference` | `ListModels`, `InferStream` |

The first service follows the
[Open Inference Protocol / KServe V2 schema](https://github.com/kserve/open-inference-protocol/blob/d49cc23f89d709d87b210ef9449e273ae243984e/specification/protocol/open_inference_grpc.proto).
`InferStream` accepts one `ModelInferRequest` and emits one
`ModelInferResponse` per output packet. A successful stream ends with OK gRPC
trailers. A failed stream ends immediately with a non-OK status; no JSON
sentinel or success packet is emitted after an error.

This is a defined subset for Kapsl's engine contract. Supported datatypes are
`FP16`, `FP32`, `FP64`, `INT32`, `INT64`, `UINT8`, and `BYTES`.
Numeric raw contents use little-endian, contiguous tensor bytes. Typed contents
are accepted for every supported numeric type except FP16, which requires raw
contents. Invalid shapes, wrong byte counts, mixed raw/typed encodings, unknown
inputs, and unsupported datatypes are rejected before dispatch.

`BYTES` represents exactly one UTF-8 string (shape `[1]`, or another
one-element shape on input). Raw BYTES contents use a four-byte little-endian
length prefix. Responses normalize string shape to `[1]`, including generated
text chunks. Use `UINT8` tensors for arbitrary binary data. String arrays and
multiple model outputs are unsupported by this adapter because the engine
boundary returns one packet. Model metadata requires the backend to provide
names, shapes, and datatypes.

Clients may select a model by name or numeric model ID. Duplicate names require
a version or an unambiguous ID. Unknown versions are rejected. An empty version
selects the runtime's current version.

Triton's standard unary clients can call the supported KServe RPCs. Triton's
bidirectional `ModelStreamInfer`, shared-memory registration, repository
management, statistics, and other Triton extensions are not implemented.
Use the generated Kapsl client for server streaming.

## Request parameters and lifecycle

Supported request parameters are `session_id` (string), `timeout_ms` (positive
integer), `priority` (integer 0–255), `force_cpu` (boolean), `max_new_tokens`,
`min_new_tokens`, `top_k`, `seed` (nonnegative integers), and `temperature`,
`top_p`, `repetition_penalty` (double). Unknown parameters and all tensor-level
parameters are rejected. Credentials belong in `authorization` metadata.

Every RPC, including health and discovery, invokes the authorizer. It receives
the actual connection peer address. Session IDs are transformed by the
runtime's authorizer before they reach inference.

The earlier of `grpc-timeout` and `timeout_ms` applies to both inference startup
and the entire response stream. Deadline expiry, client disconnect/cancellation,
and server shutdown signal the engine cancellation token. Enforcement by a
specific backend depends on its cancellation support. The deadline watchdog
runs independently of client reads. Responses are polled on demand, without an
unbounded adapter queue.

The embedding runtime sets a positive message-size limit for both directions.
HTTP/2 allows at most 64 concurrent RPC streams per connection. No default total
generation duration is imposed when neither timeout is set.

## Python client

From the SDK root, create generated modules in an isolated directory:

```sh
python -m venv .venv-grpc
.venv-grpc/bin/python -m pip install grpcio grpcio-tools
mkdir -p /tmp/kapsl-grpc-python
.venv-grpc/bin/python -m grpc_tools.protoc \
  -I crates/kapsl-grpc/proto \
  --python_out=/tmp/kapsl-grpc-python \
  --grpc_python_out=/tmp/kapsl-grpc-python \
  crates/kapsl-grpc/proto/open_inference_grpc.proto \
  crates/kapsl-grpc/proto/kapsl_inference.proto
PYTHONPATH=/tmp/kapsl-grpc-python .venv-grpc/bin/python \
  crates/kapsl-grpc/examples/stream.py 127.0.0.1:9096 MODEL "Hello"
```

Set `KAPSL_API_TOKEN` when engine authentication is enabled. Pass
`--input-name` if the model declares a different primary input name. Use
`--tls` for a TLS endpoint. The example applies a 60-second deadline, iterates
typed responses, and cancels the RPC if iteration is interrupted.

The same proto files can generate Go, Java, C#, and other standard gRPC clients.
Go generation should supply a `Mopen_inference_grpc.proto=...` and
`Mkapsl_inference.proto=...` mapping for the desired module paths because the
upstream schema does not declare a Go package.

For a local transport demonstration with no model files:

```sh
cargo run -p kapsl-grpc --example echo_server
```

It prints a loopback address with an automatically assigned port. Use model
`text` with the Python example; each request returns the input twice. Model
`tensor` exposes a UINT8 echo for testing a standard KServe/Triton unary client.

## Rust clients

Use `kapsl_grpc::inference::grpc_inference_service_client::GrpcInferenceServiceClient`
for the standard API and
`kapsl_grpc::kapsl::v1::kapsl_inference_client::KapslInferenceClient` for streaming.
They expose normal Tonic request metadata and streaming APIs. Note that Tonic
currently reports its own client-side timeout before response headers as
`CANCELLED`; the server emits `DEADLINE_EXCEEDED` for server-enforced deadlines.
