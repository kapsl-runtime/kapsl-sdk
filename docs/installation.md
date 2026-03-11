# Installation

## Requirements

- Python 3.9 or later (CPython)
- A running `kapsl-runtime` instance

## Install from PyPI

```bash
pip install kapsl-sdk
```

The package ships pre-compiled wheels for Linux (x86_64, aarch64), macOS (x86_64, Apple Silicon), and Windows (x86_64) for Python 3.9 – 3.13. No Rust toolchain needed.

## Install from source

If you need to build from the monorepo:

```bash
# Requires Rust 1.75+ and maturin
pip install maturin
cd kapsl-sdk/crates/kapsl-pyo3
maturin develop --release
```

## Verify the installation

```python
from kapsl_sdk import KapslClient, KapslShmClient, KapslHybridClient
from kapsl_sdk import list_voices, load_voice

print(list_voices())  # lists bundled voice embeddings
```

## Runtime dependency

`kapsl-sdk` is a client library — it does not bundle the inference engine. You need a running `kapsl-runtime` process on the same machine or reachable over TCP. See the [kapsl-runtime deployment guide](https://kapsl.ai/docs/engine/deployment) for setup instructions.
