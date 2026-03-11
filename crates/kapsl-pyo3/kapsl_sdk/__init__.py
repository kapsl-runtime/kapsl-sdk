from .kapsl_sdk import KapslClient, KapslHybridClient, KapslShmClient

import pathlib

__all__ = ["KapslClient", "KapslHybridClient", "KapslShmClient", "load_voice", "list_voices"]

_VOICES_DIR = pathlib.Path(__file__).parent


def list_voices() -> list:
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
    path = _VOICES_DIR / f"{name}.bin"
    if not path.exists():
        available = list_voices()
        raise FileNotFoundError(
            f"Voice '{name}' not found. Available: {available}"
        )
    return path.read_bytes()
