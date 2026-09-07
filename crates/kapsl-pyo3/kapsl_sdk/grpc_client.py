"""Optional gRPC clients for Kapsl's Open Inference Protocol services."""

from __future__ import annotations

import math
import struct
from ._streams import InferenceStream, AsyncInferenceStream
from ._types import Tensor, request_options, timeout_default

_DTYPES = {
    "float16": ("FP16", 2, None, None),
    "float32": ("FP32", 4, "f", "fp32_contents"),
    "float64": ("FP64", 8, "d", "fp64_contents"),
    "int32": ("INT32", 4, "i", "int_contents"),
    "int64": ("INT64", 8, "q", "int64_contents"),
    "uint8": ("UINT8", 1, "B", "uint_contents"),
    "string": ("BYTES", 1, None, "bytes_contents"),
}
_ALIASES = {
    "fp16": "float16", "fp32": "float32", "fp64": "float64",
    "i32": "int32", "i64": "int64", "u8": "uint8",
    "utf8": "string", "bytes": "string",
}
_WIRE_TYPES = {value[0]: key for key, value in _DTYPES.items()}
_UNBOUNDED = object()


def _load_protocol():
    try:
        import grpc
        from ._grpc import open_inference_grpc_pb2 as inference
        from ._grpc import open_inference_grpc_pb2_grpc as inference_grpc
        from ._grpc import kapsl_inference_pb2 as kapsl
        from ._grpc import kapsl_inference_pb2_grpc as kapsl_grpc
    except ModuleNotFoundError as exc:
        raise ImportError(
            "gRPC support requires: pip install 'kapsl-sdk[grpc]'"
        ) from exc
    return grpc, inference, inference_grpc, kapsl, kapsl_grpc


def _model_name(model_id):
    if isinstance(model_id, str) and model_id:
        return model_id
    if type(model_id) is int and 0 <= model_id < 2**32:
        return str(model_id)
    raise ValueError("model_id must be a model name or unsigned 32-bit ID")


def _shape(shape):
    shape = tuple(shape)
    if len(shape) > 32 or any(type(d) is not int or not 0 <= d < 2**63 for d in shape):
        raise ValueError("shape must contain at most 32 nonnegative int64 dimensions")
    count = math.prod(shape)
    if count >= 2**63:
        raise ValueError("shape is too large")
    return shape, count


def _input(inference, name, shape, dtype, data):
    if not isinstance(name, str) or not name:
        raise ValueError("Tensor input names must be nonempty strings")
    dtype = dtype.lower()
    dtype = _ALIASES.get(dtype, dtype)
    if dtype not in _DTYPES:
        raise ValueError(f"Unsupported datatype: {dtype}")
    wire_dtype, width, _, _ = _DTYPES[dtype]
    shape, count = _shape(shape)
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise TypeError("Tensor data must be bytes, bytearray, or memoryview")
    data = bytes(data)
    if dtype == "string":
        data.decode("utf-8")
        if count != 1 and shape != (1, len(data)):
            raise ValueError("String tensors contain one UTF-8 value; use shape [1]")
        shape = (1,)
        data = struct.pack("<I", len(data)) + data
    elif len(data) != count * width:
        raise ValueError("Tensor byte count does not match shape and datatype")
    return inference.ModelInferRequest.InferInputTensor(
        name=name, datatype=wire_dtype, shape=shape,
    ), data


def _decode(response):
    if len(response.outputs) != 1:
        raise ValueError("The Kapsl tensor client requires exactly one output")
    output = response.outputs[0]
    dtype = _WIRE_TYPES.get(output.datatype)
    if dtype is None:
        raise ValueError(f"Unsupported output datatype: {output.datatype}")
    shape, count = _shape(output.shape)
    _, width, fmt, field = _DTYPES[dtype]
    if response.raw_output_contents:
        if len(response.raw_output_contents) != 1:
            raise ValueError("Invalid raw output count")
        data = bytes(response.raw_output_contents[0])
        if dtype == "string":
            if len(data) < 4 or struct.unpack("<I", data[:4])[0] != len(data) - 4:
                raise ValueError("Invalid BYTES output framing")
            data = data[4:]
    elif dtype == "string":
        if len(output.contents.bytes_contents) != 1:
            raise ValueError("Expected one UTF-8 output value")
        data = bytes(output.contents.bytes_contents[0])
    elif fmt is not None:
        values = getattr(output.contents, field)
        data = struct.pack(f"<{len(values)}{fmt}", *values)
    else:
        raise ValueError("FP16 output requires raw contents")
    if dtype == "string":
        if count != 1:
            raise ValueError("Expected a single UTF-8 output value")
        data.decode("utf-8")
    elif len(data) != count * width:
        raise ValueError("Output byte count does not match its shape")
    return Tensor(
        data, shape, dtype, output.name,
        response.model_name, response.model_version, response.id,
    )


class KapslGrpcClient:
    """Synchronous gRPC discovery, inference, and server streaming.

    Install the optional extra with pip install 'kapsl-sdk[grpc]'. Tensor bytes
    use little-endian encoding. String tensors accept UTF-8 bytes.
    """

    _async = False

    def __init__(
        self, target="127.0.0.1:9096", *, api_token=None, tls=False,
        root_certificates=None, private_key=None, certificate_chain=None,
        timeout_ms=None, max_message_bytes=16 * 1024 * 1024,
    ):
        self._grpc, self._inference, inference_grpc, self._kapsl, kapsl_grpc = _load_protocol()
        self._timeout_ms = timeout_default(timeout_ms)
        if not isinstance(target, str) or not target.strip():
            raise ValueError("target must be a nonempty gRPC host:port")
        if type(max_message_bytes) is not int or not 0 < max_message_bytes <= 2**31 - 1:
            raise ValueError("max_message_bytes must be a positive int32")
        if api_token is not None and not isinstance(api_token, str):
            raise TypeError("api_token must be a string")
        if (private_key is None) != (certificate_chain is None):
            raise ValueError("Both private_key and certificate_chain are required for mTLS")
        if not tls and any(x is not None for x in (
            root_certificates, private_key, certificate_chain,
        )):
            raise ValueError("Certificate configuration requires tls=True")
        for value in (root_certificates, private_key, certificate_chain):
            if value is not None and not isinstance(value, bytes):
                raise TypeError("Certificates and private keys must be PEM-encoded bytes")
        token = api_token.strip() if api_token else ""
        if token.lower().startswith("bearer "):
            token = token[7:].strip()
        self._metadata = (("authorization", f"Bearer {token}"),) if token else ()
        options = (
            ("grpc.max_send_message_length", max_message_bytes),
            ("grpc.max_receive_message_length", max_message_bytes),
        )
        self._max_message_bytes = max_message_bytes
        self._target = target
        api = self._grpc.aio if self._async else self._grpc
        if tls:
            credentials = self._grpc.ssl_channel_credentials(
                root_certificates=root_certificates,
                private_key=private_key, certificate_chain=certificate_chain,
            )
            self._channel = api.secure_channel(target, credentials, options=options)
        else:
            self._channel = api.insecure_channel(target, options=options)
        self.inference_stub = inference_grpc.GRPCInferenceServiceStub(self._channel)
        self.streaming_stub = kapsl_grpc.KapslInferenceStub(self._channel)
        self._closed = False

    @property
    def closed(self):
        return self._closed

    def protocol(self):
        return "grpc"

    def endpoint(self):
        return self._target

    def _call_options(self, timeout_ms):
        if self._closed:
            raise RuntimeError("The client is closed")
        timeout_ms = None if timeout_ms is _UNBOUNDED else timeout_default(
            self._timeout_ms if timeout_ms is None else timeout_ms,
        )
        return {
            "metadata": self._metadata,
            "timeout": None if timeout_ms is None else timeout_ms / 1000,
        }

    def _unary(self, method, request, timeout_ms=None, transform=lambda x: x):
        call = method.future(request, **self._call_options(timeout_ms))
        try:
            return transform(call.result())
        except BaseException:
            call.cancel()
            raise

    def server_live(self, *, timeout_ms=None):
        return self._unary(
            self.inference_stub.ServerLive, self._inference.ServerLiveRequest(),
            timeout_ms, lambda result: result.live,
        )

    def server_ready(self, *, timeout_ms=None):
        return self._unary(
            self.inference_stub.ServerReady, self._inference.ServerReadyRequest(),
            timeout_ms, lambda result: result.ready,
        )

    def server_metadata(self, *, timeout_ms=None):
        return self._unary(
            self.inference_stub.ServerMetadata, self._inference.ServerMetadataRequest(),
            timeout_ms,
        )

    def model_ready(self, model_id, *, model_version="", timeout_ms=None):
        return self._unary(
            self.inference_stub.ModelReady,
            self._inference.ModelReadyRequest(name=_model_name(model_id), version=model_version),
            timeout_ms, lambda result: result.ready,
        )

    def model_metadata(self, model_id, *, model_version="", timeout_ms=None):
        return self._unary(
            self.inference_stub.ModelMetadata,
            self._inference.ModelMetadataRequest(name=_model_name(model_id), version=model_version),
            timeout_ms,
        )

    def list_models(self, *, timeout_ms=None):
        return self._unary(
            self.streaming_stub.ListModels, self._kapsl.ListModelsRequest(),
            timeout_ms, lambda result: list(result.models),
        )

    def _request(self, model_id, shape, dtype, data, additional_inputs,
                 session_id, options, input_name):
        values = request_options(options, self._timeout_ms)
        if "stop_token_ids" in values:
            raise ValueError("stop_token_ids is not supported by the gRPC 0.3 API")
        request = self._inference.ModelInferRequest(
            model_name=_model_name(model_id),
            model_version=values.pop("model_version", ""),
            id=values.pop("request_id", ""),
        )
        inputs = [(input_name, (shape, dtype, data))]
        if additional_inputs:
            if input_name in additional_inputs:
                raise ValueError("The primary input name is duplicated")
            inputs.extend(additional_inputs.items())
        for name, (tensor_shape, tensor_dtype, tensor_data) in inputs:
            tensor, raw = _input(self._inference, name, tensor_shape, tensor_dtype, tensor_data)
            request.inputs.append(tensor)
            request.raw_input_contents.append(raw)
        if session_id is not None:
            if not isinstance(session_id, str):
                raise TypeError("session_id must be a string")
            values["session_id"] = session_id
        for name, value in values.items():
            parameter = request.parameters[name]
            if isinstance(value, bool):
                parameter.bool_param = value
            elif isinstance(value, int):
                if value >= 2**63:
                    raise ValueError(f"{name} exceeds the gRPC API's int64 range")
                parameter.int64_param = value
            elif isinstance(value, float):
                parameter.double_param = value
            else:
                parameter.string_param = value
        if request.ByteSize() > self._max_message_bytes:
            raise ValueError("Request exceeds max_message_bytes")
        return request, values.get("timeout_ms", _UNBOUNDED)

    def infer(self, model_id, shape, dtype, data, additional_inputs=None,
              session_id=None, *, input_name="input", **options):
        request, timeout = self._request(
            model_id, shape, dtype, data, additional_inputs, session_id, options, input_name,
        )
        return self._unary(
            self.inference_stub.ModelInfer, request, timeout,
            lambda response: _decode(response).data,
        )

    def infer_tensor(self, model_id, shape, dtype, data, additional_inputs=None,
                     session_id=None, *, input_name="input", **options):
        request, timeout = self._request(
            model_id, shape, dtype, data, additional_inputs, session_id, options, input_name,
        )

        def decode(response):
            tensor = _decode(response)
            return tensor.data, list(tensor.shape), tensor.dtype

        return self._unary(self.inference_stub.ModelInfer, request, timeout, decode)

    def _stream(self, model_id, shape, dtype, data, additional_inputs,
                session_id, options, input_name, typed):
        request, timeout = self._request(
            model_id, shape, dtype, data, additional_inputs, session_id, options, input_name,
        )
        call = self.streaming_stub.InferStream(request, **self._call_options(timeout))
        if self._async:
            return AsyncInferenceStream(call, _decode, self._grpc.aio.EOF, typed=typed, owner=self)

        def read():
            try:
                return _decode(next(call))
            except StopIteration:
                return None

        return InferenceStream(
            read, call.cancel, typed=typed, owner=self, is_closed=lambda: not call.is_active(),
        )

    def infer_stream(self, model_id, shape, dtype, data, additional_inputs=None,
                     session_id=None, *, input_name="input", **options):
        return self._stream(
            model_id, shape, dtype, data, additional_inputs, session_id,
            options, input_name, False,
        )

    def infer_stream_tensors(self, model_id, shape, dtype, data, additional_inputs=None,
                             session_id=None, *, input_name="input", **options):
        return self._stream(
            model_id, shape, dtype, data, additional_inputs, session_id,
            options, input_name, True,
        )

    def close(self):
        self._closed = True
        self._channel.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class AsyncKapslGrpcClient(KapslGrpcClient):
    """grpc.aio client; await unary methods and async-iterate streams."""

    _async = True

    async def _unary(self, method, request, timeout_ms=None, transform=lambda x: x):
        call = method(request, **self._call_options(timeout_ms))
        try:
            return transform(await call)
        except BaseException:
            call.cancel()
            raise

    async def close(self):
        self._closed = True
        await self._channel.close()

    def __enter__(self):
        raise TypeError("Use 'async with' for AsyncKapslGrpcClient")

    def __exit__(self, *exc):
        raise TypeError("Use 'async with' for AsyncKapslGrpcClient")

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        await self.close()
