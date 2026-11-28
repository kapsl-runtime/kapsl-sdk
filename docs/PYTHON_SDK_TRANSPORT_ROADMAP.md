# Python SDK transport status

Python 0.2.0 and 0.3.0 use the current native transport (request envelope
version 1, now in `kapsl-communication` 0.5.0) and SHM version 3, supported
by engine 0.2.8 and later. See the [migration guide](python-sdk-0.2.md) for
the supported client/server pair. Old request formats are rejected; no legacy
decoder or protocol negotiation is retained.

| Capability | Implementation |
| --- | --- |
| Native TCP / Unix socket / Windows pipe | `KapslClient` |
| Full-request deadlines and cancellation | Native and gRPC clients |
| Token authentication | Native TCP token metadata; shared API authorization for gRPC and HTTP |
| Typed server streaming and discovery | `KapslGrpcClient`, `AsyncKapslGrpcClient` |
| TLS and mutual TLS | Python gRPC channels to a TLS-terminating proxy; HTTPS for `KapslHttpClient` |
| Concurrent local tensor transfer | SHM mailboxes and process-shared allocation leases |
| Model lifecycle, memory admission, idle-unload state, scaling policy, statistics, access management | `KapslHttpClient` (0.3.0) |
| OpenAI-compatible models, chat completions and embeddings | `KapslHttpClient` (0.3.0) |

The protobuf API and reusable Rust gRPC server live in the SDK's `kapsl-grpc`
crate. The engine supplies model discovery, authorization, scheduling, and
memory governance adapters. Python exposes gRPC as separate client classes
because its discovery, generated messages, and channel lifecycle differ from
the native framed protocol. `KapslHttpClient` is pure Python and uses only the
standard library.

WebSocket and bidirectional generation controls remain outside this release.
The engine's gRPC listener currently requires a proxy for TLS termination.
Native TCP TLS, HTTP tensor and media inference (`/api/models/{id}/infer`),
the Responses API, hot swap, RAG, extension management, and multiple-output
tensor inference can be considered separately when integrations need them.
