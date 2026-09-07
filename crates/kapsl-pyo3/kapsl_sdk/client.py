"""Native socket, TCP, and named-pipe inference clients."""

from __future__ import annotations

from .kapsl_sdk import KapslClient as _NativeClient
from ._streams import InferenceStream
from ._types import request_options, tensor_result, timeout_default


class KapslClient:
    """Native client with request options and explicitly cancellable streams.

    Request keyword options include timeout_ms, priority, force_cpu, request_id,
    model_version, max_new_tokens, min_new_tokens, temperature, top_p, top_k,
    repetition_penalty, seed, and stop_token_ids.
    """

    def __init__(
        self, endpoint=None, *, protocol=None, host=None, port=None,
        socket_path=None, pipe_name=None, max_pool_size=8, api_token=None,
        timeout_ms=None,
    ):
        self._timeout_ms = timeout_default(timeout_ms)
        self._native = _NativeClient(
            endpoint, protocol=protocol, host=host, port=port,
            socket_path=socket_path, pipe_name=pipe_name,
            max_pool_size=max_pool_size, api_token=api_token,
        )

    def protocol(self):
        return self._native.protocol()

    def endpoint(self):
        return self._native.endpoint()

    @property
    def closed(self):
        return self._native.closed

    def close(self):
        self._native.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def infer(
        self, model_id, shape, dtype, data, additional_inputs=None,
        session_id=None, **options,
    ):
        return bytes(self._native.infer(
            model_id, shape, dtype, data, additional_inputs, session_id,
            options=request_options(options, self._timeout_ms),
        ))

    def infer_tensor(
        self, model_id, shape, dtype, data, additional_inputs=None,
        session_id=None, **options,
    ):
        result = tensor_result(self._native.infer_tensor(
            model_id, shape, dtype, data, additional_inputs, session_id,
            options=request_options(options, self._timeout_ms),
        ))
        return result.data, list(result.shape), result.dtype

    def _stream(self, model_id, shape, dtype, data, additional_inputs,
                session_id, options, typed):
        native = self._native.infer_stream(
            model_id, shape, dtype, data, additional_inputs, session_id,
            options=request_options(options, self._timeout_ms),
        )

        def read():
            result = native.next_tensor()
            return None if result is None else tensor_result(result)

        return InferenceStream(
            read, native.cancel, typed=typed, owner=self,
            is_closed=lambda: native.closed,
        )

    def infer_stream(
        self, model_id, shape, dtype, data, additional_inputs=None,
        session_id=None, **options,
    ):
        return self._stream(
            model_id, shape, dtype, data, additional_inputs, session_id, options, False,
        )

    def infer_stream_tensors(
        self, model_id, shape, dtype, data, additional_inputs=None,
        session_id=None, **options,
    ):
        return self._stream(
            model_id, shape, dtype, data, additional_inputs, session_id, options, True,
        )
