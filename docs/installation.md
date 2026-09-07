# Installation

## Requirements

- Python 3.9 or later (CPython)
- A running `kapsl-runtime` instance

## Install from PyPI

```bash
pip install kapsl-sdk
```

The package ships pre-compiled **abi3 wheels** for Linux (x86_64, aarch64), macOS (x86_64, Apple Silicon), and Windows (x86_64). A single wheel covers Python 3.9 and later — including Python 3.14+ — with no recompilation needed.

## Install from source

If you need to build from the monorepo:

```bash
cd kapsl-sdk
# Python extension module (requires Rust 1.92+ and an active virtualenv)
pip install maturin
maturin develop --release
```

## Verify the installation

```python
from kapsl_sdk import KapslClient, KapslShmClient, KapslHybridClient
from kapsl_sdk import list_voices, load_voice

print(list_voices())  # lists bundled voice embeddings
```

## Runtime dependency

For optional gRPC support, install `pip install 'kapsl-sdk[grpc]'`.
Version 0.2.0 requires the coordinated native transport 0.4.0 engine update;
see the [migration guide](python-sdk-0.2.md).

`kapsl-sdk` is a client library — it does not bundle the inference engine. You need a running `kapsl-runtime` process on the same machine or reachable over TCP. See the [kapsl-runtime deployment guide](https://kapsl.ai/docs/engine/deployment) for setup instructions.
