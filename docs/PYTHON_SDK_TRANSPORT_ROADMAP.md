# Python SDK transport status

Python 0.2.0 uses the current native transport 0.4.0 and SHM version 3.
See the [migration guide](python-sdk-0.2.md) for the supported client/server pair.
Old request formats are rejected; no legacy decoder or protocol negotiation is
retained.

| Capability | Implementation |
| --- | --- |
| Native TCP / Unix socket / Windows pipe | `KapslClient` |
| Full-request deadlines and cancellation | Native and gRPC clients |
| Token authentication | Native TCP token metadata; shared API authorization for gRPC |
| Typed server streaming and discovery | `KapslGrpcClient`, `AsyncKapslGrpcClient` |
| TLS and mutual TLS | Python gRPC channels to a TLS-terminating proxy |
| Concurrent local tensor transfer | SHM mailboxes and process-shared allocation leases |

The protobuf API and reusable Rust gRPC server live in the SDK's `kapsl-grpc`
crate. The engine supplies model discovery, authorization, scheduling, and
memory governance adapters. Python exposes gRPC as separate client classes
because its discovery, generated messages, and channel lifecycle differ from
the native framed protocol.

WebSocket and bidirectional generation controls remain outside this release.
The engine's gRPC listener currently requires a proxy for TLS termination.
Native TCP TLS, governance management helpers, and multiple-output tensor
inference can be considered separately when integrations need them.
