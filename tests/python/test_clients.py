import asyncio
from concurrent.futures import ThreadPoolExecutor
from importlib.metadata import version
import json
import socket
from pathlib import Path
import struct
import threading
import time

import grpc
import pytest
import kapsl_sdk
from kapsl_sdk import KapslClient, KapslGrpcClient, AsyncKapslGrpcClient, Tensor

TOKEN = "sdk-test-token"
DATA = struct.pack("<2f", 1.25, -3.5)


def native(server, **kwargs):
    return KapslClient("tcp://" + server["tcp"], api_token=TOKEN, **kwargs)


def rpc(server, **kwargs):
    return KapslGrpcClient(server["grpc"], api_token=TOKEN, **kwargs)


@pytest.mark.parametrize("make_client", [native, rpc])
def test_unary_streaming_and_typed_results(server, make_client):
    with make_client(server) as client:
        assert client.infer(7, [2], "float32", DATA) == DATA
        assert client.infer_tensor(7, [2], "float32", DATA) == (DATA, [2], "float32")
        stream = client.infer_stream(7, [2], "float32", DATA)
        assert list(stream) == [DATA, DATA]
        assert stream.closed
        for _ in range(2):
            with pytest.raises(StopIteration):
                next(stream)
        packets = list(client.infer_stream_tensors(7, [2], "float32", DATA))
        assert len(packets) == 2
        assert all(isinstance(p, Tensor) and p.shape == (2,) and p.data == DATA for p in packets)
    assert client.closed


def test_native_socket_and_generation_options(server):
    with KapslClient(socket_path=server["socket"]) as client:
        output = json.loads(client.infer(
            11, [2], "float32", DATA,
            additional_inputs={"mask": ([2], "float32", DATA)}, session_id="native-session",
            request_id="native-options", timeout_ms=1000, priority=0, force_cpu=True,
            model_version="1", max_new_tokens=12, min_new_tokens=3,
            temperature=0.5, top_p=0.75, top_k=4, repetition_penalty=1.1,
            seed=42, stop_token_ids=[9, 10],
        ))
        assert output["metadata"]["min_new_tokens"] == 3
        assert output["metadata"]["max_new_tokens"] == 12
        assert output["metadata"]["stop_token_ids"] == [9, 10]
        assert output["metadata"]["seed"] == 42
        assert output["metadata"]["temperature"] == 0.5
        assert output["metadata"]["timeout_ms"] == 1000
        assert output["session_id"] == "native-session"
        assert output["additional_inputs"] == ["mask"]
        assert output["priority"] == "LatencyCritical"
        assert output["force_cpu"] is True


def test_native_tcp_preserves_authentication_metadata(server):
    with native(server) as client:
        output = json.loads(client.infer(
            11, [2], "float32", DATA, session_id="token-session", request_id="native-auth",
        ))
        assert output["metadata"]["auth_token"] == TOKEN
        assert list(client.infer_stream(7, [2], "float32", DATA)) == [DATA, DATA]
    with KapslClient("tcp://" + server["tcp"], api_token="wrong") as client:
        for model_id in (7, 999):
            with pytest.raises(RuntimeError, match="Unauthorized"):
                client.infer(model_id, [2], "float32", DATA)
        with pytest.raises(RuntimeError, match="Unauthorized"):
            next(client.infer_stream(7, [2], "float32", DATA))


def test_released_python_request_is_rejected_without_metadata_fallback(server):
    # Captured from the 0.1.23 wheel with a dummy audit token. That request
    # predates min_new_tokens and must never be decoded by dropping metadata.
    payload = bytes.fromhex(
        "010000000000000001000000000000000700000000000000666c6f617433320104"
        "000000000000000000803f000000000000000000010d0000000000000072656c65"
        "6173652d617564697401000000000001190000000000000072656c656173652d61"
        "756469742d6c6f63616c2d746f6b656e00000000000000"
    )
    host, port = server["tcp"].split(":")
    with socket.create_connection((host, int(port)), timeout=2) as connection:
        connection.sendall(struct.pack("<III", 7, 2, len(payload)) + payload)
        with connection.makefile("rb") as response:
            status, size = struct.unpack("<II", response.read(8))
            message = response.read(size).decode()
    assert status == 1
    assert "deserializ" in message.lower() or "invalid request" in message.lower()
    assert "Unauthorized" not in message


@pytest.mark.parametrize("make_client", [native, rpc])
def test_deadlines_release_unary_and_stream_startup(server, released, make_client):
    with make_client(server, timeout_ms=100) as client:
        for operation in ("infer", "infer_stream"):
            request_id = f"{make_client.__name__}-{operation}-deadline"
            with pytest.raises((TimeoutError, grpc.RpcError)):
                result = getattr(client, operation)(
                    9, [2], "float32", DATA, request_id=request_id,
                )
                if operation == "infer_stream":
                    next(result)
            released(request_id)


@pytest.mark.parametrize("make_client", [native, rpc])
def test_deadline_expires_while_caller_is_not_reading(server, released, make_client):
    with make_client(server, timeout_ms=150) as client:
        request_id = f"{make_client.__name__}-idle-deadline"
        stream = client.infer_stream(12, [2], "float32", DATA, request_id=request_id)
        assert next(stream) == DATA
        released(request_id)
        with pytest.raises((TimeoutError, grpc.RpcError)):
            next(stream)
        assert stream.closed


@pytest.mark.parametrize("make_client", [native, rpc])
def test_cancel_interrupts_a_blocked_reader(server, released, make_client):
    with make_client(server) as client:
        request_id = f"{make_client.__name__}-cancel-reader"
        stream = client.infer_stream(12, [2], "float32", DATA, request_id=request_id)
        assert next(stream) == DATA
        reading = threading.Event()

        def read():
            reading.set()
            try:
                next(stream)
            except (StopIteration, grpc.RpcError):
                return "closed"
            return "unexpected chunk"

        with ThreadPoolExecutor(max_workers=1) as pool:
            pending = pool.submit(read)
            assert reading.wait(1)
            time.sleep(0.05)
            stream.cancel()
            assert pending.result(timeout=2) == "closed"
        released(request_id)
        stream.close()
        assert stream.closed


@pytest.mark.parametrize("make_client", [native, rpc])
def test_client_close_releases_open_streams(server, released, make_client):
    client = make_client(server)
    request_id = f"{make_client.__name__}-client-close"
    stream = client.infer_stream(12, [2], "float32", DATA, request_id=request_id)
    assert next(stream) == DATA
    client.close()
    released(request_id)
    assert stream.closed


@pytest.mark.parametrize("make_client", [native, rpc])
def test_stream_failure_terminates_iteration(server, make_client):
    with make_client(server) as client:
        stream = client.infer_stream(10, [2], "float32", DATA)
        assert next(stream) == DATA
        with pytest.raises((RuntimeError, grpc.RpcError)):
            next(stream)
        with pytest.raises(StopIteration):
            next(stream)


@pytest.mark.parametrize("options", [
    {"timeout_ms": 0}, {"timeout_ms": True}, {"priority": 256},
    {"force_cpu": 1}, {"temperature": float("nan")}, {"top_p": 1.1},
    {"repetition_penalty": 0}, {"min_new_tokens": 3, "max_new_tokens": 2},
    {"unknown": 1}, {"auth_token": "in-body"}, {"stop_token_ids": [-1]},
])
@pytest.mark.parametrize("make_client", [native, rpc])
def test_invalid_request_options_are_rejected(server, make_client, options):
    with make_client(server) as client:
        with pytest.raises((ValueError, TypeError)):
            client.infer(7, [2], "float32", DATA, **options)


def test_grpc_discovery_and_request_options(server):
    with rpc(server) as client:
        assert client.server_live()
        assert client.server_ready()
        assert client.server_metadata().version == "python-test"
        assert {model.id for model in client.list_models()} == set(range(7, 13))
        assert client.model_ready("echo", model_version="1")
        assert client.model_metadata("echo").inputs[0].datatype == "FP32"
        output = json.loads(client.infer(
            "options", [2], "float32", DATA,
            additional_inputs={"mask": ([2], "float32", DATA)},
            session_id="grpc-session", request_id="grpc-options",
            timeout_ms=1000, model_version="1", priority=0, force_cpu=True,
            max_new_tokens=12, min_new_tokens=3, seed=42, temperature=0.5,
        ))
        assert output["metadata"]["min_new_tokens"] == 3
        assert output["metadata"]["request_id"] == "grpc-options"
        assert output["session_id"] == "test:grpc-session"
        assert output["force_cpu"] is True
        assert output["priority"] == "LatencyCritical"
    with KapslGrpcClient(server["grpc"], api_token="wrong") as client:
        with pytest.raises(grpc.RpcError) as error:
            client.server_live()
        assert error.value.code() == grpc.StatusCode.UNAUTHENTICATED


def test_grpc_unicode_and_typed_streaming(server):
    text = "Hello 🌏 — tiếng Việt".encode()
    with rpc(server) as client:
        assert client.infer("text", [1], "string", text) == text
        packets = list(client.infer_stream_tensors("text", [1], "string", text))
        assert all(p.data == text and p.dtype == "string" and p.shape == (1,) for p in packets)
        assert all(p.model_name == "text" and p.model_version == "1" for p in packets)


def test_grpc_size_limits_and_unsupported_options(server):
    with rpc(server, max_message_bytes=64) as client:
        with pytest.raises(ValueError, match="max_message_bytes"):
            client.infer(7, [100], "float32", bytes(400))
    with rpc(server) as client:
        with pytest.raises(ValueError, match="stop_token_ids"):
            client.infer(7, [2], "float32", DATA, stop_token_ids=[1])
        with pytest.raises(ValueError, match="byte count"):
            client.infer(7, [3], "float32", DATA)


@pytest.mark.parametrize("make_client", [native, rpc])
def test_per_request_none_disables_default_deadline(server, make_client):
    with make_client(server, timeout_ms=100) as client:
        stream = client.infer_stream(12, [2], "float32", DATA, timeout_ms=None)
        assert next(stream) == DATA
        time.sleep(0.2)
        assert not stream.closed
        stream.close()


def test_async_grpc_and_native_stream_iteration(server, released):
    async def run():
        async with AsyncKapslGrpcClient(server["grpc"], api_token=TOKEN) as client:
            assert await client.server_ready()
            assert await client.infer(7, [2], "float32", DATA) == DATA
            packets = [p async for p in client.infer_stream_tensors(7, [2], "float32", DATA)]
            assert [p.data for p in packets] == [DATA, DATA]
            stream = client.infer_stream(12, [2], "float32", DATA, request_id="async-grpc-cancel")
            assert await stream.__anext__() == DATA
            pending = asyncio.create_task(stream.__anext__())
            await asyncio.sleep(0.05)
            pending.cancel()
            with pytest.raises(asyncio.CancelledError):
                await pending
            assert stream.closed
            pending = asyncio.create_task(client.infer(
                9, [2], "float32", DATA, request_id="async-grpc-unary-cancel",
            ))
            await asyncio.sleep(0.05)
            pending.cancel()
            with pytest.raises(asyncio.CancelledError):
                await pending
        with native(server) as client:
            stream = client.infer_stream(7, [2], "float32", DATA)
            assert [p async for p in stream] == [DATA, DATA]
            stream = client.infer_stream(12, [2], "float32", DATA, request_id="async-native-cancel")
            assert await stream.__anext__() == DATA
            pending = asyncio.create_task(stream.__anext__())
            await asyncio.sleep(0.05)
            pending.cancel()
            with pytest.raises(asyncio.CancelledError):
                await pending
    asyncio.run(run())
    released("async-grpc-cancel")
    released("async-native-cancel")
    released("async-grpc-unary-cancel")


@pytest.mark.parametrize("transport", ["shm", "hybrid"])
def test_shm_clients_use_current_layout_and_isolate_concurrent_calls(server, transport):
    def make():
        if transport == "shm":
            return kapsl_sdk.KapslShmClient(server["shm"])
        return kapsl_sdk.KapslHybridClient(server["hybrid_shm"], server["socket"])
    clients = [make(), make()]

    def infer(index):
        data = struct.pack("<2f", index, index + 0.5)
        result = clients[index % 2].infer([2], "float32", data, model_id=7)
        assert isinstance(result, bytes)
        assert result == data
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(infer, range(40)))


def test_wheel_contains_helpers_and_generated_clients():
    assert kapsl_sdk.__version__ == version("kapsl-sdk")
    assert callable(kapsl_sdk.list_voices)
    from kapsl_sdk.grpc_protocol import inference
    assert inference.ModelInferRequest.__module__.startswith("kapsl_sdk.")
    assert Path(kapsl_sdk.__file__).with_name("py.typed").is_file()
