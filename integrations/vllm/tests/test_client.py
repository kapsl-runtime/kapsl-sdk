from __future__ import annotations

import json
import socket
import tempfile
import threading
import unittest
from pathlib import Path
from typing import Any

from kapsl_vllm_connector.client import KapslKvControlClient, KapslKvControlError
from kapsl_vllm_connector.contract import ABI_VERSION, make_reserve_request, opaque_registration

CAPACITY_GROUPS = [
    {
        "group_id": "vllm.group.0",
        "pool_id": "vllm.pool.0",
        "allocation_granularity_tokens": 16,
        "bytes_per_allocation": 1_048_576,
        "memory_domains": [{"kind": "cuda", "device_id": 0}],
        "max_allocations": 1024,
    }
]


class FakeCoordinator:
    def __init__(self, socket_path: Path, expected_requests: int, *, reject=False):
        self.socket_path = socket_path
        self.expected_requests = expected_requests
        self.reject = reject
        self.requests: list[dict[str, Any]] = []
        self.error: BaseException | None = None
        self._server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._server.bind(str(socket_path))
        self._server.listen(expected_requests)
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def join(self) -> None:
        self._thread.join(timeout=5)
        self._server.close()
        if self._thread.is_alive():
            raise AssertionError("fake coordinator did not finish")
        if self.error is not None:
            raise self.error

    def _run(self) -> None:
        try:
            for _ in range(self.expected_requests):
                connection, _ = self._server.accept()
                with connection:
                    request = json.loads(_read_line(connection))
                    self.requests.append(request)
                    response = self._response(request)
                    connection.sendall(json.dumps(response).encode("utf-8") + b"\n")
        except BaseException as error:
            self.error = error

    def _response(self, request: dict[str, Any]) -> dict[str, Any]:
        base = {
            "abi_version": dict(ABI_VERSION),
            "request_id": request["request_id"],
        }
        if self.reject:
            return {
                **base,
                "result": "error",
                "error": {
                    "kind": "capacity_exhausted",
                    "message": "device KV budget exhausted",
                },
            }
        if request["operation"] == "register":
            return {**base, "result": "registered"}
        if request["operation"] == "reserve":
            reserve = request["request"]
            return {
                **base,
                "result": "lease",
                "lease": {
                    "lease_id": "lease-1",
                    "sequence": reserve["sequence"],
                    "groups": [
                        {
                            "group_id": reserve["groups"][0]["group_id"],
                            "token_capacity": reserve["groups"][0]["token_capacity"],
                            "blocks": [],
                        }
                    ],
                },
            }
        return {**base, "result": "ack"}


class ClientTests(unittest.TestCase):
    def test_lifecycle_uses_versioned_flat_envelopes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            socket_path = Path(directory) / "kv.sock"
            server = FakeCoordinator(socket_path, 6)
            server.start()
            request_ids = iter(f"rpc-{index}" for index in range(6))
            client = KapslKvControlClient(
                f"unix://{socket_path}",
                "vllm-0",
                request_id_factory=lambda: next(request_ids),
            )

            client.register(
                opaque_registration("vllm-0", "sha256:model", CAPACITY_GROUPS)
            )
            lease = client.reserve(
                make_reserve_request(
                    request_id="request-1",
                    sequence_id="request-1",
                    token_capacity=4096,
                    group_ids=["vllm.group.0"],
                )
            )
            client.commit(lease["lease_id"], 128)
            client.touch(lease["lease_id"])
            client.heartbeat()
            client.release(lease["lease_id"])
            server.join()

            self.assertEqual(
                [request["operation"] for request in server.requests],
                ["register", "reserve", "commit", "touch", "heartbeat", "release"],
            )
            self.assertEqual(server.requests[0]["abi_version"], ABI_VERSION)
            self.assertEqual(server.requests[1]["request"]["groups"][0]["token_capacity"], 4096)

    def test_coordinator_error_preserves_machine_readable_kind(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            socket_path = Path(directory) / "kv.sock"
            server = FakeCoordinator(socket_path, 1, reject=True)
            server.start()
            client = KapslKvControlClient(
                f"unix://{socket_path}",
                "vllm-0",
                request_id_factory=lambda: "rpc-rejected",
            )

            with self.assertRaises(KapslKvControlError) as caught:
                client.reserve(
                    make_reserve_request(
                        request_id="request-1",
                        sequence_id="request-1",
                        token_capacity=4096,
                        group_ids=["vllm.group.0"],
                    )
                )
            server.join()

            self.assertEqual(caught.exception.kind, "capacity_exhausted")
            self.assertIn("budget exhausted", str(caught.exception))


def _read_line(connection: socket.socket) -> bytes:
    chunks: list[bytes] = []
    while True:
        chunk = connection.recv(4096)
        if not chunk:
            raise RuntimeError("client closed before newline")
        if b"\n" in chunk:
            before, _ = chunk.split(b"\n", 1)
            chunks.append(before)
            return b"".join(chunks)
        chunks.append(chunk)


if __name__ == "__main__":
    unittest.main()
