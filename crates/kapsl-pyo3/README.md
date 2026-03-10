# kapsl-sdk

`kapsl-sdk` is the Python client package for [kapsl-runtime](https://kapsl.ai), the Kapsl inference runtime for local and embedded model execution.

It provides Python bindings for the runtime's transport layer so Python applications can talk to a running `kapsl-runtime` process over:

- TCP / IPC client connections with `KapslClient`
- shared-memory transport with `KapslShmClient`
- hybrid IPC + shared-memory transport with `KapslHybridClient`

The package is intended for low-latency inference integrations where Python is orchestrating requests while the runtime performs the heavy model execution work.

## Installation

```bash
pip install kapsl-sdk
```

## What It Includes

- Python bindings backed by the Rust runtime client implementation
- support for CPython 3.9+
- transport options for local development and high-throughput deployments
- a small API surface aimed at direct integration into backend services and tools

## Requirements

- A running `kapsl-runtime` instance
- The runtime endpoint details for your deployment
- Shared-memory transport setup if you want to use `KapslShmClient` or `KapslHybridClient`

## Usage

```python
from kapsl_sdk import KapslClient

client = KapslClient()
result = client.infer(
    model_id=0,
    shape=[1, 4],
    dtype="float32",
    data=b"...",
)
```

## Client Types

### `KapslClient`

General-purpose client for runtime connections over TCP, Unix sockets, or Windows named pipes depending on platform and endpoint configuration.

### `KapslShmClient`

Shared-memory client for lower-overhead local inference workflows where request and response payloads are exchanged through shared memory.

### `KapslHybridClient`

Hybrid client that combines shared memory for tensor payloads with IPC signaling for coordination.

## Notes

- `kapsl-sdk` is the package name on PyPI.
- The Python import name is `kapsl_sdk`.
- This package is an API client, not the runtime server itself.

## License

Proprietary — see [kapsl.ai](https://kapsl.ai) for licensing information.
