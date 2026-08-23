"""Transport adapter for Kapsl's newline-delimited JSON KV control protocol."""

from __future__ import annotations

import json
import socket
import uuid
from collections.abc import Callable, Mapping
from typing import Any
from urllib.parse import unquote, urlsplit

from .contract import (
    ContractValidationError,
    make_envelope,
    validate_lease,
    validate_registration,
    validate_reserve_request,
    validate_response,
)


class KapslKvControlError(RuntimeError):
    """Control-plane connection, protocol, or coordinator error."""

    def __init__(self, message: str, *, kind: str | None = None):
        super().__init__(message)
        self.kind = kind


class KapslKvControlClient:
    """Synchronous client used at vLLM scheduler lifecycle boundaries.

    A fresh Unix connection is used for every operation. This avoids sharing a
    socket across vLLM's forked scheduler/worker processes and makes retries
    unambiguous through the envelope request ID.
    """

    def __init__(
        self,
        endpoint: str,
        participant_id: str,
        *,
        timeout_seconds: float = 2.0,
        max_frame_bytes: int = 1024 * 1024,
        request_id_factory: Callable[[], str] | None = None,
    ) -> None:
        self.socket_path = _unix_socket_path(endpoint)
        if not participant_id.strip():
            raise ValueError("participant_id must not be empty")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if max_frame_bytes <= 0:
            raise ValueError("max_frame_bytes must be positive")
        self.participant_id = participant_id
        self.timeout_seconds = timeout_seconds
        self.max_frame_bytes = max_frame_bytes
        self._request_id_factory = request_id_factory or (lambda: uuid.uuid4().hex)

    def register(self, registration: Mapping[str, Any]) -> None:
        validate_registration(registration)
        response = self._rpc("register", registration=dict(registration))
        self._expect(response, {"registered", "ack"})

    def reserve(self, request: Mapping[str, Any]) -> dict[str, Any]:
        validate_reserve_request(request)
        response = self._rpc(
            "reserve",
            participant_id=self.participant_id,
            request=dict(request),
        )
        self._expect(response, {"lease"})
        try:
            return validate_lease(_as_mapping(response.get("lease"), "lease"))
        except ContractValidationError as error:
            raise KapslKvControlError(f"invalid lease response: {error}") from error

    def commit(self, lease_id: str, computed_tokens: int) -> None:
        if not lease_id.strip():
            raise ValueError("lease_id must not be empty")
        if computed_tokens < 0:
            raise ValueError("computed_tokens cannot be negative")
        response = self._rpc(
            "commit",
            participant_id=self.participant_id,
            request={
                "lease_id": lease_id,
                "computed_tokens": int(computed_tokens),
            },
        )
        self._expect(response, {"ack"})

    def touch(self, lease_id: str) -> None:
        response = self._rpc(
            "touch",
            participant_id=self.participant_id,
            lease_id=_required(lease_id, "lease_id"),
        )
        self._expect(response, {"ack"})

    def heartbeat(self) -> None:
        response = self._rpc(
            "heartbeat",
            participant_id=self.participant_id,
        )
        self._expect(response, {"ack"})

    def release(self, lease_id: str) -> None:
        response = self._rpc(
            "release",
            participant_id=self.participant_id,
            lease_id=_required(lease_id, "lease_id"),
        )
        self._expect(response, {"ack"})

    def _rpc(self, operation: str, **payload: Any) -> dict[str, Any]:
        request_id = self._request_id_factory()
        envelope = make_envelope(request_id, operation, **payload)
        frame = json.dumps(envelope, separators=(",", ":"), sort_keys=True).encode("utf-8")
        if len(frame) + 1 > self.max_frame_bytes:
            raise KapslKvControlError("KV control request exceeds maximum frame size")

        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
                connection.settimeout(self.timeout_seconds)
                connection.connect(self.socket_path)
                connection.sendall(frame + b"\n")
                raw_response = _read_frame(connection, self.max_frame_bytes)
        except (OSError, TimeoutError) as error:
            raise KapslKvControlError(
                f"KV control request '{operation}' failed: {error}", kind="transport"
            ) from error

        try:
            decoded = json.loads(raw_response)
            response = dict(_as_mapping(decoded, "response"))
            validate_response(response, request_id)
        except (UnicodeDecodeError, json.JSONDecodeError, ContractValidationError) as error:
            raise KapslKvControlError(f"invalid KV control response: {error}") from error

        if response["result"] == "error":
            remote = _as_mapping(response.get("error"), "error")
            kind = str(remote.get("kind", "internal"))
            message = str(remote.get("message") or remote.get("operation") or kind)
            raise KapslKvControlError(message, kind=kind)
        return response

    @staticmethod
    def _expect(response: Mapping[str, Any], allowed: set[str]) -> None:
        result = response.get("result")
        if result not in allowed:
            expected = ", ".join(sorted(allowed))
            raise KapslKvControlError(
                f"unexpected KV control result '{result}', expected {expected}"
            )


def _unix_socket_path(endpoint: str) -> str:
    parsed = urlsplit(endpoint)
    if parsed.scheme != "unix" or parsed.netloc or parsed.query or parsed.fragment:
        raise ValueError("kapsl_control_endpoint must be an absolute unix:/// socket URL")
    path = unquote(parsed.path)
    if not path.startswith("/") or path == "/":
        raise ValueError("kapsl_control_endpoint must contain an absolute socket path")
    return path


def _read_frame(connection: socket.socket, max_frame_bytes: int) -> bytes:
    chunks: list[bytes] = []
    size = 0
    while True:
        chunk = connection.recv(min(65536, max_frame_bytes - size + 1))
        if not chunk:
            raise KapslKvControlError("KV control peer closed before a complete frame")
        newline = chunk.find(b"\n")
        if newline >= 0:
            chunks.append(chunk[:newline])
            break
        chunks.append(chunk)
        size += len(chunk)
        if size >= max_frame_bytes:
            raise KapslKvControlError("KV control response exceeds maximum frame size")
    frame = b"".join(chunks)
    if not frame:
        raise KapslKvControlError("KV control response is empty")
    if len(frame) > max_frame_bytes:
        raise KapslKvControlError("KV control response exceeds maximum frame size")
    return frame


def _as_mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ContractValidationError(f"{field} must be an object")
    return value


def _required(value: str, field: str) -> str:
    if not value.strip():
        raise ValueError(f"{field} must not be empty")
    return value
