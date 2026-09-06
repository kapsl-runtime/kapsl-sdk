from .kapsl_sdk import KapslHybridClient, KapslShmClient
from .client import KapslClient
from .grpc_client import KapslGrpcClient, AsyncKapslGrpcClient
from ._types import Tensor
from importlib.metadata import version as _package_version

import pathlib

__version__ = _package_version("kapsl-sdk")

__all__ = [
    "KapslClient", "KapslGrpcClient", "AsyncKapslGrpcClient",
    "KapslHybridClient", "KapslShmClient", "Tensor", "load_voice", "list_voices",
]

_VOICES_DIR = pathlib.Path(__file__).parent


def list_voices() -> list[str]:
    """Return the names of all bundled voice embeddings (without the .bin extension)."""
    return sorted(p.stem for p in _VOICES_DIR.glob("*.bin"))


def load_voice(name: str) -> bytes:
    """Load a bundled voice embedding by name (e.g. ``'af_bella'``).

    Returns the raw float32 bytes. Reshape as ``(-1, 1, 256)`` to index by
    token length::

        import numpy as np
        data = load_voice("af_bella")
        voices = np.frombuffer(data, dtype=np.float32).reshape(-1, 1, 256)
        style = voices[len(tokens)].reshape(1, 1, 256)
    """
    available = list_voices()
    if name not in available:
        raise FileNotFoundError(
            f"Voice '{name}' not found. Available: {available}"
        )
    return (_VOICES_DIR / f"{name}.bin").read_bytes()
