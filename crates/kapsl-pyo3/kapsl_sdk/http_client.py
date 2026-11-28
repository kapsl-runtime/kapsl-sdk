"""HTTP client for Kapsl's management API and OpenAI-compatible routes.

Uses only the standard library, so it works in a native-only installation.
"""

from __future__ import annotations

from collections.abc import Mapping
from importlib.metadata import PackageNotFoundError, version as _package_version
import json
import os
import socket
import threading
import urllib.error
import urllib.parse
import urllib.request

from ._streams import InferenceStream, read_sse_data
from ._types import (
    ChatCompletionMessage, ScalingPolicy, api_key, auth_login, chat_completion,
    chat_completion_chunk, created_api_key, embedding_list, model_load_result,
    model_status, request_options, role_token_status, scaling_policy, system_stats,
    timeout_default, _record, _records, AuthStatus, AuthUser, HealthStatus,
    OpenAIModel, RoleSummary,
)

_ROLES = {"reader", "writer", "admin"}
_USER_STATUSES = {"active", "suspended"}
_ENCODINGS = {"float", "base64"}

try:
    _USER_AGENT = f"kapsl-sdk-python/{_package_version('kapsl-sdk')}"
except PackageNotFoundError:  # pragma: no cover - source checkouts
    _USER_AGENT = "kapsl-sdk-python"


class KapslHttpError(RuntimeError):
    """An HTTP API request failed.

    ``status`` is the HTTP status (200 for an error reported inside a
    successful response, such as a failed stream). ``reason`` is the engine's
    machine-readable cause when it sends one, for example
    ``memory_admission``, ``idle_reload_memory_unavailable`` or
    ``model_not_loaded``. ``retry_after_seconds`` comes from the response's
    ``retry_after_seconds`` field or ``Retry-After`` header. ``error_type`` is
    the OpenAI error ``type`` on ``/v1`` routes.
    """

    def __init__(self, status, message, *, reason=None, error_type=None,
                 retry_after_seconds=None, body=None):
        super().__init__(f"HTTP {status}: {message}")
        self.status = status
        self.message = message
        self.reason = reason
        self.error_type = error_type
        self.retry_after_seconds = retry_after_seconds
        self.body = body

    @classmethod
    def from_response(cls, status, body, headers=None):
        message = f"HTTP {status}"
        reason = error_type = retry_after = None
        if isinstance(body, Mapping):
            error = body.get("error")
            if isinstance(error, Mapping):
                message = str(error.get("message") or message)
                error_type = error.get("type")
            elif error is not None:
                message = str(error)
                if body.get("detail"):
                    message = f"{message}: {body['detail']}"
            elif body.get("object") == "error" and body.get("message"):
                message = str(body["message"])
                error_type = body.get("type")
            reason = body.get("reason")
            retry_after = body.get("retry_after_seconds")
        elif isinstance(body, str) and body.strip():
            message = body.strip()
        if retry_after is None and headers is not None:
            value = headers.get("Retry-After")
            if value is not None and value.strip().isdigit():
                retry_after = int(value)
        return cls(
            status, message, reason=reason, error_type=error_type,
            retry_after_seconds=retry_after, body=body,
        )


def _base_url(value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("base_url must be a nonempty http:// or https:// URL")
    parsed = urllib.parse.urlsplit(value.strip())
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise ValueError("base_url must be an http:// or https:// URL")
    if parsed.query or parsed.fragment:
        raise ValueError("base_url must not contain a query or fragment")
    path = parsed.path.rstrip("/")
    # Accept the documented API roots as well as the server root.
    for suffix in ("/api", "/v1"):
        if path.endswith(suffix):
            path = path[: -len(suffix)]
            break
    return urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, path, "", ""))


def _model_id(value, name="model_id"):
    if type(value) is not int or not 0 <= value < 2**32:
        raise ValueError(f"{name} must be an unsigned 32-bit integer")
    return value


def _segment(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    # The engine's router does not percent-decode path parameters, so keep
    # every character RFC 3986 allows in a segment (such as ":" in
    # "name:version") and escape only what would change the route.
    return urllib.parse.quote(value, safe=":@!$&'()*+,;=")


def _text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _integer(value, name, minimum, maximum):
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be an integer between {minimum} and {maximum}")
    return value


def _choice(value, name, allowed):
    if value not in allowed:
        raise ValueError(f"{name} must be one of: {', '.join(sorted(allowed))}")
    return value


def _join(body, extra_body):
    if extra_body is None:
        return body
    if not isinstance(extra_body, Mapping):
        raise TypeError("extra_body must be a mapping")
    duplicated = sorted(set(body) & set(extra_body))
    if duplicated:
        raise ValueError(f"extra_body repeats explicit fields: {', '.join(duplicated)}")
    return {**body, **extra_body}


def _messages(messages):
    if isinstance(messages, (str, bytes, Mapping)):
        raise TypeError("messages must be a sequence of chat messages")
    result = []
    for message in messages:
        if isinstance(message, ChatCompletionMessage):
            message = {"role": message.role, "content": message.content}
        if not isinstance(message, Mapping) or not isinstance(message.get("role"), str) \
                or not message["role"].strip():
            raise ValueError("Each chat message must be a mapping with a nonempty 'role'")
        result.append(dict(message))
    if not result:
        raise ValueError("messages must contain at least one message")
    return result


def _stop(value):
    if value is None:
        return None
    if isinstance(value, str):
        return value
    values = list(value)
    if not all(isinstance(item, str) for item in values):
        raise TypeError("stop must be a string or a sequence of strings")
    return values


def _generation(max_tokens, max_completion_tokens, temperature, top_p, top_k,
                repetition_penalty, seed):
    values = request_options({
        "temperature": temperature, "top_p": top_p, "top_k": top_k,
        "repetition_penalty": repetition_penalty, "seed": seed,
    })
    for name, value in (("max_tokens", max_tokens),
                        ("max_completion_tokens", max_completion_tokens)):
        if value is not None:
            values[name] = _integer(value, name, 0, 2**32 - 1)
    return values


class KapslHttpClient:
    """Synchronous client for the engine's HTTP API (default port 9095).

    Covers health, hardware and memory statistics, model lifecycle (including
    memory-admission results and idle-unload state), autoscaling policy,
    Prometheus metrics, access management under ``/api/auth``, and the
    OpenAI-compatible ``/v1`` models, chat completions and embeddings routes.

    ``api_token`` is sent as ``Authorization: Bearer TOKEN``. Model listing,
    inference and statistics need a reader token; lifecycle, scaling updates,
    metrics and access management need an admin token. Without engine
    authentication only loopback clients are accepted.

    ``timeout_ms`` applies to connecting and to each read from the socket, not
    to a request's total duration; a waited model start or a long stream can
    take longer. ``None`` waits indefinitely. Every method takes its own
    ``timeout_ms`` to override the default. Connection failures raise
    ``urllib.error.URLError`` (an ``OSError``); error responses raise
    ``KapslHttpError``.
    """

    def __init__(self, base_url="http://127.0.0.1:9095", *, api_token=None,
                 timeout_ms=None, ssl_context=None):
        self._base_url = _base_url(base_url)
        self._timeout_ms = timeout_default(timeout_ms)
        if api_token is not None and not isinstance(api_token, str):
            raise TypeError("api_token must be a string")
        token = api_token.strip() if api_token else ""
        if token.lower().startswith("bearer "):
            token = token[7:].strip()
        self._token = token
        handlers = []
        if ssl_context is not None:
            handlers.append(urllib.request.HTTPSHandler(context=ssl_context))
        self._opener = urllib.request.build_opener(*handlers)
        self._closed = False

    @property
    def closed(self):
        return self._closed

    def protocol(self):
        return "http"

    def endpoint(self):
        return self._base_url

    def close(self):
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # -- transport -------------------------------------------------------

    def _open(self, method, path, body=None, *, query=None, timeout_ms=None,
              headers=None, token=None, accept="application/json"):
        if self._closed:
            raise RuntimeError("The client is closed")
        timeout_ms = timeout_default(self._timeout_ms if timeout_ms is None else timeout_ms)
        url = self._base_url + path
        if query:
            url += "?" + urllib.parse.urlencode(query)
        request_headers = {"Accept": accept, "User-Agent": _USER_AGENT}
        token = self._token if token is None else token
        if token:
            request_headers["Authorization"] = f"Bearer {token}"
        data = None
        if body is not None:
            data = json.dumps(body).encode("utf-8")
            request_headers["Content-Type"] = "application/json"
        elif method == "POST":
            data = b""
        if headers:
            request_headers.update(headers)
        request = urllib.request.Request(url, data=data, method=method, headers=request_headers)
        try:
            return self._opener.open(
                request, timeout=None if timeout_ms is None else timeout_ms / 1000,
            )
        except urllib.error.HTTPError as error:
            with error:
                raw = error.read()
            raise KapslHttpError.from_response(
                error.code, _decode_body(raw, error.headers), error.headers,
            ) from None

    def _call(self, method, path, body=None, **kwargs):
        with self._open(method, path, body, **kwargs) as response:
            return response.status, _decode_body(response.read(), response.headers)

    def _json(self, method, path, body=None, **kwargs):
        return self._call(method, path, body, **kwargs)[1]

    # -- runtime ---------------------------------------------------------

    def health(self, *, timeout_ms=None):
        """``GET /api/health`` (reader)."""
        return _record(HealthStatus, self._json("GET", "/api/health", timeout_ms=timeout_ms))

    def hardware(self, *, timeout_ms=None):
        """``GET /api/hardware`` (reader): the detected devices, as JSON."""
        return self._json("GET", "/api/hardware", timeout_ms=timeout_ms)

    def system_stats(self, *, timeout_ms=None):
        """``GET /api/system/stats`` (reader): process, GPU and memory-authority stats."""
        return system_stats(self._json("GET", "/api/system/stats", timeout_ms=timeout_ms))

    def metrics(self, *, timeout_ms=None):
        """``GET /metrics`` (admin): Prometheus text exposition."""
        with self._open("GET", "/metrics", timeout_ms=timeout_ms, accept="text/plain") as response:
            return response.read().decode("utf-8")

    # -- models ----------------------------------------------------------

    def list_models(self, *, timeout_ms=None):
        """``GET /api/models`` (reader): every registered model and replica."""
        body = self._json("GET", "/api/models", timeout_ms=timeout_ms)
        if not isinstance(body, list):
            raise ValueError("Expected a JSON array from /api/models")
        return [model_status(item) for item in body]

    def get_model(self, model_id, *, timeout_ms=None):
        """``GET /api/models/{id}`` (reader)."""
        status, body = self._call(
            "GET", f"/api/models/{_model_id(model_id)}", timeout_ms=timeout_ms,
        )
        # Engine 0.3.0 reports an unknown ID as HTTP 200 with an error body.
        if isinstance(body, Mapping) and "error" in body and "id" not in body:
            raise KapslHttpError.from_response(status, body)
        return model_status(body)

    def start_model(self, model_path, *, model_id=None, topology=None, tp_degree=None,
                    idle_unload_seconds=None, wait=False, timeout_ms=None):
        """``POST /api/models/start`` (admin): load a package on the engine host.

        ``model_path`` names a ``.aimod`` file on the engine's filesystem.
        Without ``wait`` the engine queues the load and returns ``started``.
        With ``wait=True`` (engine 0.3.0) the call returns once the load is
        ``active`` or ``denied``; a denial is returned, not raised, so its
        ``reason`` and ``requested`` bytes can be inspected. Other failures
        raise ``KapslHttpError``. ``idle_unload_seconds`` (engine 0.3.0)
        overrides the idle timeout for this model; ``0`` keeps it loaded.
        """
        body = {"model_path": _text(os.fspath(model_path), "model_path")}
        if model_id is not None:
            body["model_id"] = _model_id(model_id)
        if topology is not None:
            body["topology"] = _text(topology, "topology")
        if tp_degree is not None:
            body["tp_degree"] = _integer(tp_degree, "tp_degree", 1, 2**32 - 1)
        if idle_unload_seconds is not None:
            body["idle_unload_seconds"] = _integer(
                idle_unload_seconds, "idle_unload_seconds", 0, 2**64 - 1,
            )
        if type(wait) is not bool:
            raise TypeError("wait must be a bool")
        try:
            status, response = self._call(
                "POST", "/api/models/start", body,
                query={"wait": "true"} if wait else None, timeout_ms=timeout_ms,
            )
        except KapslHttpError as error:
            if error.status == 409 and isinstance(error.body, Mapping) \
                    and error.body.get("status") == "denied":
                return model_load_result(409, error.body)
            raise
        return model_load_result(status, response)

    def stop_model(self, model_id, *, timeout_ms=None):
        """``POST /api/models/{id}/stop`` (admin). Returns the engine's message.

        Stopping a model the idle unloader already stopped keeps it down: the
        next request no longer reloads it.
        """
        return self._message("POST", f"/api/models/{_model_id(model_id)}/stop", timeout_ms)

    def remove_model(self, model_id, *, timeout_ms=None):
        """``POST /api/models/{id}/remove`` (admin): stop and unregister a model."""
        return self._message("POST", f"/api/models/{_model_id(model_id)}/remove", timeout_ms)

    def _message(self, method, path, timeout_ms):
        body = self._json(method, path, timeout_ms=timeout_ms)
        return body.get("message", "") if isinstance(body, Mapping) else ""

    def get_scaling_policy(self, model_id, *, timeout_ms=None):
        """``GET /api/models/{id}/scaling`` (reader).

        The engine returns its default policy for a model without one.
        """
        return scaling_policy(self._json(
            "GET", f"/api/models/{_model_id(model_id)}/scaling", timeout_ms=timeout_ms,
        ))

    def set_scaling_policy(self, model_id, policy, *, timeout_ms=None):
        """``POST /api/models/{id}/scaling`` (admin): replace the policy.

        Every field is sent. Use ``dataclasses.replace`` on the current policy
        to change one value.
        """
        if not isinstance(policy, ScalingPolicy):
            raise TypeError("policy must be a ScalingPolicy")
        policy.validate()
        self._json(
            "POST", f"/api/models/{_model_id(model_id)}/scaling", policy.to_json(),
            timeout_ms=timeout_ms,
        )

    # -- authentication --------------------------------------------------

    def login(self, token=None, *, timeout_ms=None):
        """``POST /api/auth/login`` (public): the access a token grants.

        Checks ``token`` when given, otherwise the client's ``api_token``.
        """
        if token is not None:
            token = _text(token, "token").strip()
            if token.lower().startswith("bearer "):
                token = token[7:].strip()
        return auth_login(self._json(
            "POST", "/api/auth/login", {}, token=token, timeout_ms=timeout_ms,
        ))

    def auth_status(self, *, timeout_ms=None):
        """``GET /api/auth/access/status`` (admin)."""
        return _record(AuthStatus, self._json(
            "GET", "/api/auth/access/status", timeout_ms=timeout_ms,
        ))

    def role_summaries(self, *, timeout_ms=None):
        """``GET /api/auth/access/roles`` (admin)."""
        return list(_records(RoleSummary)(self._json(
            "GET", "/api/auth/access/roles", timeout_ms=timeout_ms,
        )))

    def role_token_status(self, *, timeout_ms=None):
        """``GET /api/auth/roles`` (admin): which role tokens are configured."""
        return role_token_status(self._json("GET", "/api/auth/roles", timeout_ms=timeout_ms))

    def set_role_tokens(self, *, reader_token=None, writer_token=None, admin_token=None,
                        timeout_ms=None):
        """``POST /api/auth/roles`` (admin): replace the legacy role tokens.

        All three are replaced: a role left as ``None`` loses its token, and
        passing none disables role-token authentication. An admin token is
        required whenever another token is set.
        """
        body = {}
        for name, value in (("reader_token", reader_token), ("writer_token", writer_token),
                            ("admin_token", admin_token)):
            if value is not None:
                body[name] = _text(value, name)
        if body and "admin_token" not in body:
            raise ValueError("admin_token is required when role-token authentication is enabled")
        return role_token_status(self._json(
            "POST", "/api/auth/roles", body, timeout_ms=timeout_ms,
        ))

    def list_users(self, *, timeout_ms=None):
        """``GET /api/auth/access/users`` (admin)."""
        return list(_records(AuthUser)(self._json(
            "GET", "/api/auth/access/users", timeout_ms=timeout_ms,
        )))

    def create_user(self, username, role, *, display_name=None, status=None, timeout_ms=None):
        """``POST /api/auth/access/users`` (admin). ``role`` is reader, writer or admin."""
        body = {"username": _text(username, "username"), "role": _choice(role, "role", _ROLES)}
        if display_name is not None:
            body["display_name"] = _text(display_name, "display_name")
        if status is not None:
            body["status"] = _choice(status, "status", _USER_STATUSES)
        return _record(AuthUser, self._json(
            "POST", "/api/auth/access/users", body, timeout_ms=timeout_ms,
        ))

    def update_user(self, user_id, *, display_name=None, role=None, status=None,
                    timeout_ms=None):
        """``PATCH /api/auth/access/users/{id}`` (admin).

        Only the given fields change. ``display_name=""`` clears the name.
        """
        body = {}
        if display_name is not None:
            if not isinstance(display_name, str):
                raise TypeError("display_name must be a string")
            body["display_name"] = display_name
        if role is not None:
            body["role"] = _choice(role, "role", _ROLES)
        if status is not None:
            body["status"] = _choice(status, "status", _USER_STATUSES)
        return _record(AuthUser, self._json(
            "PATCH", f"/api/auth/access/users/{_segment(user_id, 'user_id')}", body,
            timeout_ms=timeout_ms,
        ))

    def list_api_keys(self, user_id=None, *, timeout_ms=None):
        """``GET /api/auth/access/keys`` (admin), optionally for one user."""
        query = None if user_id is None else {"user_id": _text(user_id, "user_id")}
        body = self._json("GET", "/api/auth/access/keys", query=query, timeout_ms=timeout_ms)
        if not isinstance(body, list):
            raise ValueError("Expected a JSON array from /api/auth/access/keys")
        return [api_key(item) for item in body]

    def create_api_key(self, user_id, name, *, scopes=None, expires_in_days=None,
                       timeout_ms=None):
        """``POST /api/auth/access/users/{id}/keys`` (admin).

        The key has its user's role; ``scopes`` (such as ``api:read``) can
        narrow it further. The secret is only available from this result.
        """
        body = {"name": _text(name, "name")}
        if scopes is not None:
            if isinstance(scopes, str):
                raise TypeError("scopes must be a sequence of strings")
            body["scopes"] = [_text(scope, "scope") for scope in scopes]
        if expires_in_days is not None:
            body["expires_in_days"] = _integer(expires_in_days, "expires_in_days", 1, 2**32 - 1)
        return created_api_key(self._json(
            "POST", f"/api/auth/access/users/{_segment(user_id, 'user_id')}/keys", body,
            timeout_ms=timeout_ms,
        ))

    def revoke_api_key(self, key_id, *, timeout_ms=None):
        """``POST /api/auth/access/keys/{id}/revoke`` (admin)."""
        return api_key(self._json(
            "POST", f"/api/auth/access/keys/{_segment(key_id, 'key_id')}/revoke",
            timeout_ms=timeout_ms,
        ))

    # -- OpenAI-compatible routes ---------------------------------------

    def list_openai_models(self, *, timeout_ms=None):
        """``GET /v1/models`` (reader): servable primary models, by name.

        With idle reload on, models stopped for being idle are listed too.
        """
        body = self._json("GET", "/v1/models", timeout_ms=timeout_ms)
        if not isinstance(body, Mapping):
            raise ValueError("Expected a JSON object from /v1/models")
        return list(_records(OpenAIModel)(body.get("data", [])))

    def get_openai_model(self, model, *, timeout_ms=None):
        """``GET /v1/models/{model}`` (reader): by name, ``name:version`` or ID."""
        return _record(OpenAIModel, self._json(
            "GET", f"/v1/models/{_segment(model, 'model')}", timeout_ms=timeout_ms,
        ))

    def _chat_body(self, model, messages, stream, max_tokens, max_completion_tokens,
                   temperature, top_p, top_k, repetition_penalty, seed, stop, user):
        body = {"model": _text(model, "model"), "messages": _messages(messages), "stream": stream}
        body.update(_generation(
            max_tokens, max_completion_tokens, temperature, top_p, top_k,
            repetition_penalty, seed,
        ))
        stop = _stop(stop)
        if stop is not None:
            body["stop"] = stop
        if user is not None:
            body["user"] = _text(user, "user")
        return body

    @staticmethod
    def _session_headers(session_id):
        if session_id is None:
            return None
        return {"X-Kapsl-Session": _text(session_id, "session_id")}

    def chat_completion(self, model, messages, *, max_tokens=None,
                        max_completion_tokens=None, temperature=None, top_p=None,
                        top_k=None, repetition_penalty=None, seed=None, stop=None,
                        user=None, session_id=None, extra_body=None, timeout_ms=None):
        """``POST /v1/chat/completions`` (reader) without streaming.

        ``messages`` are ``{"role": ..., "content": ...}`` mappings or
        ``ChatCompletionMessage`` results. ``session_id`` is sent as
        ``X-Kapsl-Session`` for KV-cache affinity; otherwise ``user`` serves as
        the session key. ``extra_body`` adds request fields this method does
        not name; managed vLLM models receive them unchanged. With idle reload
        on, the first request after an idle unload waits for the model to load.
        """
        body = self._chat_body(
            model, messages, False, max_tokens, max_completion_tokens, temperature,
            top_p, top_k, repetition_penalty, seed, stop, user,
        )
        return chat_completion(self._json(
            "POST", "/v1/chat/completions", _join(body, extra_body),
            headers=self._session_headers(session_id), timeout_ms=timeout_ms,
        ))

    def chat_completion_stream(self, model, messages, *, max_tokens=None,
                               max_completion_tokens=None, temperature=None, top_p=None,
                               top_k=None, repetition_penalty=None, seed=None, stop=None,
                               user=None, session_id=None, include_usage=False,
                               extra_body=None, timeout_ms=None):
        """``POST /v1/chat/completions`` with ``stream=true``.

        Returns a stream of ``ChatCompletionChunk`` objects that supports
        ``for``, ``async for``, ``with``, ``cancel()`` and ``close()``; closing
        it disconnects, which cancels generation. An error the engine reports
        after the stream started raises ``KapslHttpError`` with status 200, and
        a connection that ends before ``[DONE]`` raises ``ConnectionError``.
        """
        body = self._chat_body(
            model, messages, True, max_tokens, max_completion_tokens, temperature,
            top_p, top_k, repetition_penalty, seed, stop, user,
        )
        if type(include_usage) is not bool:
            raise TypeError("include_usage must be a bool")
        if include_usage:
            body["stream_options"] = {"include_usage": True}
        response = self._open(
            "POST", "/v1/chat/completions", _join(body, extra_body),
            headers=self._session_headers(session_id), timeout_ms=timeout_ms,
            accept="text/event-stream",
        )
        status = response.status
        # Reads and the final close share a lock: http.client is not safe to
        # close while another thread reads. Cancelling first shuts the socket
        # down, which wakes a blocked read so the lock is released promptly.
        lock = threading.Lock()
        cancelled = threading.Event()

        def read():
            with lock:
                if cancelled.is_set():
                    return None
                try:
                    data = read_sse_data(response.readline)
                except (OSError, ValueError):
                    if cancelled.is_set():
                        return None
                    raise
            if data is None:
                if cancelled.is_set():
                    return None
                raise ConnectionError("The chat completion stream ended before [DONE]")
            if data.strip() == "[DONE]":
                return None
            event = json.loads(data)
            if isinstance(event, Mapping) and (
                "error" in event or event.get("object") == "error"
            ):
                raise KapslHttpError.from_response(status, event)
            return chat_completion_chunk(event)

        def cancel():
            cancelled.set()
            _shutdown(response)
            with lock:
                response.close()

        return InferenceStream(
            read, cancel, typed=True, owner=self, is_closed=lambda: response.closed,
        )

    def embeddings(self, model, input, *, encoding_format=None, dimensions=None,
                   user=None, session_id=None, extra_body=None, timeout_ms=None):
        """``POST /v1/embeddings`` (reader) for models whose package serves embeddings.

        ``input`` is a string, a list of strings, a list of token IDs, or a list
        of token-ID lists.
        """
        body = {"model": _text(model, "model"), "input": input}
        if encoding_format is not None:
            body["encoding_format"] = _choice(encoding_format, "encoding_format", _ENCODINGS)
        if dimensions is not None:
            body["dimensions"] = _integer(dimensions, "dimensions", 1, 2**63 - 1)
        if user is not None:
            body["user"] = _text(user, "user")
        return embedding_list(self._json(
            "POST", "/v1/embeddings", _join(body, extra_body),
            headers=self._session_headers(session_id), timeout_ms=timeout_ms,
        ))


def _shutdown(response):
    # Async iteration reads in a worker thread. Shutting the socket down makes
    # a read blocked there return at once instead of at the next event.
    sock = getattr(getattr(getattr(response, "fp", None), "raw", None), "_sock", None)
    if sock is not None:
        try:
            sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass


def _decode_body(raw, headers):
    if not raw:
        return None
    text = raw.decode("utf-8", errors="replace")
    content_type = (headers.get("Content-Type") or "") if headers is not None else ""
    if "json" in content_type or text.lstrip()[:1] in ("{", "["):
        try:
            return json.loads(text)
        except ValueError:
            return text
    return text
