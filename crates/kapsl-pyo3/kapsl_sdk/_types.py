"""Transport-independent Python request validation and tensor results."""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real
from typing import Any, Mapping


@dataclass(frozen=True)
class Tensor:
    data: bytes
    shape: tuple[int, ...]
    dtype: str
    name: str = ""
    model_name: str = ""
    model_version: str = ""
    request_id: str = ""


_INTEGER_OPTIONS = {
    "timeout_ms": (1, 2**64 - 1),
    "priority": (0, 255),
    "max_new_tokens": (0, 2**32 - 1),
    "min_new_tokens": (0, 2**32 - 1),
    "top_k": (0, 2**32 - 1),
    "seed": (0, 2**64 - 1),
}
_FLOAT_OPTIONS = {
    "temperature": (0, None, True),
    "top_p": (0, 1, True),
    "repetition_penalty": (0, None, False),
}
_STRING_OPTIONS = {"request_id", "model_version"}


def request_options(values: Mapping[str, Any], default_timeout_ms=None) -> dict[str, Any]:
    result = {}
    for key, value in values.items():
        if key not in {*_INTEGER_OPTIONS, *_FLOAT_OPTIONS, *_STRING_OPTIONS,
                       "force_cpu", "stop_token_ids"}:
            raise TypeError(f"Unknown request option: {key}")
        if value is None:
            continue
        if key in _INTEGER_OPTIONS:
            minimum, maximum = _INTEGER_OPTIONS[key]
            if type(value) is not int or not minimum <= value <= maximum:
                raise ValueError(f"{key} must be an integer between {minimum} and {maximum}")
        elif key in _FLOAT_OPTIONS:
            minimum, maximum, inclusive = _FLOAT_OPTIONS[key]
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
                raise ValueError(f"{key} must be a finite number")
            value = float(value)
            if value < minimum or (not inclusive and value == minimum) or (
                maximum is not None and value > maximum
            ) or value > 3.4028234663852886e38:
                raise ValueError(f"{key} is outside its supported range")
        elif key in _STRING_OPTIONS:
            if not isinstance(value, str):
                raise TypeError(f"{key} must be a string")
        elif key == "force_cpu":
            if type(value) is not bool:
                raise TypeError("force_cpu must be a bool")
        elif key == "stop_token_ids":
            value = list(value)
            if any(type(item) is not int or not 0 <= item < 2**32 for item in value):
                raise ValueError("stop_token_ids must contain unsigned 32-bit integers")
        result[key] = value
    if "timeout_ms" not in values and default_timeout_ms is not None:
        result["timeout_ms"] = default_timeout_ms
    if result.get("min_new_tokens", 0) > result.get("max_new_tokens", 2**32 - 1):
        raise ValueError("min_new_tokens must not exceed max_new_tokens")
    return result


def timeout_default(value):
    return request_options({"timeout_ms": value}).get("timeout_ms")


def tensor_result(value) -> Tensor:
    data, shape, dtype = value
    return Tensor(bytes(data), tuple(shape), dtype)
