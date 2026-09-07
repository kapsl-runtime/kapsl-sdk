"""Explicitly owned synchronous and asynchronous inference streams."""

from __future__ import annotations

import asyncio


class InferenceStream:
    def __init__(self, read, cancel, *, typed=False, owner=None, is_closed=None):
        self._read = read
        self._cancel = cancel
        self._typed = typed
        self._owner = owner
        self._is_closed = is_closed
        self._ended = False

    @property
    def closed(self):
        return self._ended or bool(self._is_closed and self._is_closed())

    def __iter__(self):
        return self

    def __next__(self):
        if self._ended:
            raise StopIteration
        try:
            packet = self._read()
            if packet is None:
                self.close()
                raise StopIteration
            return packet if self._typed else packet.data
        except BaseException:
            self.close()
            raise

    def cancel(self):
        active = not self._ended
        self._ended = True
        try:
            self._cancel()
        finally:
            self._owner = None
        return active

    def close(self):
        self.cancel()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __aiter__(self):
        return self

    def _next_event(self):
        try:
            return True, next(self)
        except StopIteration:
            return False, None

    async def __anext__(self):
        try:
            present, value = await asyncio.to_thread(self._next_event)
        except asyncio.CancelledError:
            self.cancel()
            raise
        if not present:
            raise StopAsyncIteration
        return value

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        self.close()

    async def aclose(self):
        self.close()

    def __del__(self):
        if hasattr(self, "_cancel"):
            self.close()


class AsyncInferenceStream:
    def __init__(self, call, decode, eof, *, typed=False, owner=None):
        self._call = call
        self._decode = decode
        self._eof = eof
        self._typed = typed
        self._owner = owner
        self._ended = False

    @property
    def closed(self):
        return self._ended or self._call.done()

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._ended:
            raise StopAsyncIteration
        try:
            response = await self._call.read()
            if response is self._eof:
                self.cancel()
                raise StopAsyncIteration
            packet = self._decode(response)
            return packet if self._typed else packet.data
        except BaseException:
            self.cancel()
            raise

    def cancel(self):
        active = not self._ended
        self._ended = True
        self._call.cancel()
        self._owner = None
        return active

    async def close(self):
        self.cancel()

    aclose = close

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        self.cancel()

    def __del__(self):
        if hasattr(self, "_call"):
            self.cancel()
