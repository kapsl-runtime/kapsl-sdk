"""KapslHttpClient against a local fake of the engine 0.3.0 HTTP API.

Response bodies follow the engine's serializers in kapsl-cli/src/http.
"""

import asyncio
import dataclasses
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading
import time
import urllib.error
import urllib.parse

import pytest
import kapsl_sdk
from kapsl_sdk import (
    ChatCompletion, ChatCompletionChunk, ChatCompletionMessage, KapslHttpClient,
    KapslHttpError, ModelLoadResult, ModelStatus, RoleTokenStatus, ScalingPolicy,
    SystemStats,
)

ADMIN = "kvg1_test_admin"
READER = "kvg1_test_reader"

MODEL = {
    "id": 3, "name": "assistant-7b", "version": "1.0.0", "framework": "gguf",
    "device": "cuda", "optimization_level": "basic", "loaded_at": 1764300000,
    "model_path": "/models/assistant-7b.aimod", "status": "inactive",
    "replica_id": 0, "base_model_id": 3, "format": "gguf",
    "model_type": "causal-lm", "task": "generate",
    "active_inferences": 0, "total_inferences": 12, "queue_depth": [0, 1],
    "memory_usage": 0, "gpu_utilization": 0.0, "throughput": 0.0,
    "prompt_tokens_total": 340, "generated_tokens_total": 900,
    "generated_tokens_per_sec": 0.0, "total_tokens_per_sec": 0.0,
    "decode_steps_total": 450, "decode_tokens_evaluated_total": 900,
    "avg_tokens_evaluated_per_decode_step": 2.0,
    "kv_partial_reuse_hits_total": 4, "kv_partial_reuse_tokens_saved_total": 128,
    "onnx_session_pool_total": 0, "onnx_session_pool_idle": 0,
    "onnx_session_pool_waits_total": 0, "onnx_session_pool_wait_seconds_total": 0.0,
    "avg_latency_ms": 41.5, "p99_latency_ms": 90.0, "time_to_first_token_ms": 12.0,
    "healthy": True, "idle_unloaded": True, "reloads_on_request": True,
    "idle_unload_seconds": 900,
    "future_field": {"added": "later"},
}

SYSTEM_STATS = {
    "pid": 4242, "process_memory_bytes": 1073741824,
    "total_system_memory_bytes": 34359738368, "gpu_utilization": 0.42,
    "gpu_memory_bytes": 8589934592, "gpu_memory_total_bytes": 25769803776,
    "foreign_gpu_memory_bytes": None, "pressure_state": "normal",
    "collected_at_ms": 1764300000000,
    "cuda": {"driver_version": "580.65", "driver_cuda_version": "13.0",
             "runtime_version": None},
    "gpu_sampling": "stale", "gpu_sampling_age_secs": 7,
    "memory_authority": {
        "model_bytes": 9663676416, "domain_used_bytes": 10737418240,
        "domain_budget_bytes": 25769803776, "domain_available_bytes": 15032385536,
        "foreign_pressure_active": False,
        "models": [{
            "model_id": 3, "owner_kind": "model", "name": "assistant-7b",
            "replica_count": 2, "planned_bytes": 9663676416,
            "reserved_bytes": 9663676416, "committed_bytes": 9663676416,
            "observed_bytes": 9000000000, "used_bytes": 9663676416, "percentage": 100.0,
        }],
        "domains": [
            {"domain": "host", "budget_bytes": 30000000000, "planned_bytes": 0,
             "reserved_bytes": 0, "committed_bytes": 0, "observed_bytes": 1,
             "used_bytes": 1, "available_bytes": 29999999999,
             "authority_available_bytes": 29999999999},
            {"domain": "cuda:0", "budget_bytes": 25769803776, "planned_bytes": 9663676416,
             "reserved_bytes": 9663676416, "committed_bytes": 9663676416,
             "observed_bytes": 10737418240, "used_bytes": 10737418240,
             "available_bytes": 15032385536, "authority_available_bytes": 15100000000},
        ],
    },
}

KEY = {
    "id": "key_0123456789abcdef", "user_id": "usr_0123456789abcdef",
    "username": "svc", "user_role": "reader", "name": "service-key",
    "key_prefix": "kpsl_abcd", "scopes": ["api:read"], "created_at": 1764300000,
    "created_by": "usr_admin", "active": True,
}


def chat_body(finish_reason="stop", content="Hello there"):
    return {
        "id": "chatcmpl-test", "object": "chat.completion", "created": 1764300000,
        "model": "assistant-7b",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content},
                     "logprobs": None, "finish_reason": finish_reason}],
        "usage": {"prompt_tokens": 3, "completion_tokens": 3, "total_tokens": 6},
    }


def chunk(delta, finish_reason=None):
    return "data: " + json.dumps({
        "id": "chatcmpl-test", "object": "chat.completion.chunk", "created": 1764300000,
        "model": "assistant-7b",
        "choices": [{"index": 0, "delta": delta, "logprobs": None,
                     "finish_reason": finish_reason}],
    }) + "\n\n"


class FakeEngine:
    """Records requests and replies with canned engine responses."""

    def __init__(self):
        self.requests = []
        self.routes = {}
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), self._handler())
        self.server.daemon_threads = True
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def route(self, method, path, status=200, body=None, headers=None, stream=None):
        self.routes[(method, path)] = (status, body, headers or {}, stream)

    @property
    def last(self):
        return self.requests[-1]

    def _handler(self):
        engine = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def _serve(self):
                url = urllib.parse.urlsplit(self.path)
                length = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(length) if length else b""
                engine.requests.append({
                    "method": self.command, "path": url.path,
                    "query": dict(urllib.parse.parse_qsl(url.query)),
                    "headers": {key.lower(): value for key, value in self.headers.items()},
                    "body": json.loads(raw) if raw else None,
                })
                route = engine.routes.get((self.command, url.path))
                if route is None:
                    self.send_response(404)
                    self.end_headers()
                    return
                status, body, headers, stream = route
                self.send_response(status)
                for key, value in headers.items():
                    self.send_header(key, value)
                if stream is not None:
                    self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                    self.end_headers()
                    try:
                        for part in stream:
                            if callable(part):
                                part()
                                continue
                            self.wfile.write(part.encode())
                            self.wfile.flush()
                    except OSError:
                        pass
                    self.close_connection = True
                    return
                if isinstance(body, str):
                    payload, content_type = body.encode(), "text/plain; version=0.0.4"
                else:
                    payload, content_type = json.dumps(body).encode(), "application/json"
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            do_GET = do_POST = do_PATCH = _serve

        return Handler

    def close(self):
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def engine():
    fake = FakeEngine()
    yield fake
    fake.close()


@pytest.fixture
def client(engine):
    with KapslHttpClient(engine.url, api_token=ADMIN, timeout_ms=5000) as client:
        yield client


def test_package_exports_http_client_and_types():
    for name in ("KapslHttpClient", "KapslHttpError", "ModelStatus", "ModelLoadResult",
                 "ScalingPolicy", "SystemStats", "ChatCompletion", "ChatCompletionChunk"):
        assert name in kapsl_sdk.__all__ and hasattr(kapsl_sdk, name)


@pytest.mark.parametrize("base", ["", "/", "/api", "/api/", "/v1"])
def test_base_url_accepts_server_and_api_roots(engine, base):
    engine.route("GET", "/api/health", body={
        "status": "healthy", "total_models": 1, "healthy_models": 1, "unhealthy_models": 0,
    })
    with KapslHttpClient(engine.url + base) as client:
        assert client.endpoint() == engine.url and client.protocol() == "http"
        health = client.health()
    assert health.status == "healthy" and health.total_models == 1
    assert "authorization" not in engine.last["headers"]
    assert engine.last["headers"]["user-agent"].startswith("kapsl-sdk-python/")


@pytest.mark.parametrize("url", ["127.0.0.1:9095", "ftp://host", "http://host/?q=1", ""])
def test_base_url_validation(url):
    with pytest.raises(ValueError):
        KapslHttpClient(url)


def test_bearer_token_and_closed_client(engine):
    engine.route("GET", "/api/health", body={"status": "healthy"})
    client = KapslHttpClient(engine.url, api_token="Bearer " + READER)
    client.health()
    assert engine.last["headers"]["authorization"] == f"Bearer {READER}"
    client.close()
    assert client.closed
    with pytest.raises(RuntimeError, match="closed"):
        client.health()


def test_system_stats_memory_authority_and_cuda(client, engine):
    engine.route("GET", "/api/system/stats", body=SYSTEM_STATS)
    stats = client.system_stats()
    assert isinstance(stats, SystemStats)
    assert stats.pid == 4242 and stats.pressure_state == "normal"
    assert stats.foreign_gpu_memory_bytes is None
    assert stats.cuda.driver_cuda_version == "13.0" and stats.cuda.runtime_version is None
    assert stats.gpu_sampling == "stale" and stats.gpu_sampling_age_secs == 7
    authority = stats.memory_authority
    assert authority.domain_available_bytes == 15032385536
    assert authority.models[0].replica_count == 2
    cuda = authority.domain("cuda:0")
    assert cuda.available_bytes == 15032385536
    assert cuda.authority_available_bytes == 15100000000
    assert authority.domain("cuda:1") is None


def test_system_stats_from_an_older_engine(client, engine):
    older = {key: value for key, value in SYSTEM_STATS.items()
             if key not in {"cuda", "gpu_sampling", "gpu_sampling_age_secs"}}
    older["memory_authority"] = dict(older["memory_authority"], domains=[{
        key: value for key, value in SYSTEM_STATS["memory_authority"]["domains"][1].items()
        if key != "authority_available_bytes"
    }])
    engine.route("GET", "/api/system/stats", body=older)
    stats = client.system_stats()
    assert stats.cuda is None and stats.gpu_sampling is None
    assert stats.memory_authority.domains[0].authority_available_bytes is None


def test_hardware_and_metrics(client, engine):
    engine.route("GET", "/api/hardware", body={"cpu_cores": 16, "has_cuda": False})
    engine.route("GET", "/metrics", body="# TYPE kapsl_up gauge\nkapsl_up 1\n")
    assert client.hardware() == {"cpu_cores": 16, "has_cuda": False}
    assert "kapsl_up 1" in client.metrics()
    assert engine.last["headers"]["accept"] == "text/plain"


def test_model_status_reports_idle_state_and_keeps_new_fields(client, engine):
    replica = dict(MODEL, id=4, replica_id=1, status="active", idle_unloaded=False,
                   reloads_on_request=False, idle_unload_seconds=None)
    engine.route("GET", "/api/models", body=[MODEL, replica])
    primary, copy = client.list_models()
    assert isinstance(primary, ModelStatus)
    assert primary.status == "inactive" and primary.idle_unloaded
    assert primary.reloads_on_request and primary.idle_unload_seconds == 900
    assert primary.queue_depth == (0, 1) and primary.task == "generate"
    assert primary.extra == {"future_field": {"added": "later"}}
    assert primary.successful_inferences is None
    assert copy.is_replica and not primary.is_replica
    assert copy.idle_unload_seconds is None


def test_model_status_from_an_older_engine(client, engine):
    older = {key: value for key, value in MODEL.items()
             if key not in {"idle_unloaded", "reloads_on_request", "idle_unload_seconds"}}
    engine.route("GET", "/api/models/3", body=dict(
        older, successful_inferences=10, failed_inferences=2,
    ))
    model = client.get_model(3)
    assert not model.idle_unloaded and not model.reloads_on_request
    assert model.idle_unload_seconds is None
    assert (model.successful_inferences, model.failed_inferences) == (10, 2)


def test_get_model_reports_an_unknown_id_as_an_error(client, engine):
    # Engine 0.3.0 answers an unknown ID with HTTP 200 and an error body.
    engine.route("GET", "/api/models/99", body={"error": "Model 99 not found"})
    with pytest.raises(KapslHttpError, match="Model 99 not found") as raised:
        client.get_model(99)
    assert raised.value.status == 200
    with pytest.raises(ValueError):
        client.get_model(-1)


def test_start_model_queued(client, engine):
    engine.route("POST", "/api/models/start", status=202, body={
        "message": "Model load started", "model_id": 5,
    })
    result = client.start_model("/models/embed.aimod")
    assert isinstance(result, ModelLoadResult)
    assert result.status == "started" and result.model_id == 5 and result.http_status == 202
    assert not result.admitted and not result.denied
    assert engine.last["query"] == {}
    assert engine.last["body"] == {"model_path": "/models/embed.aimod"}


def test_start_model_waits_and_sends_placement_and_idle_timeout(client, engine):
    engine.route("POST", "/api/models/start", body={
        "message": "Model loaded", "model_id": 7, "status": "active",
    })
    result = client.start_model(
        "/models/llm.aimod", model_id=7, topology="data-parallel", tp_degree=2,
        idle_unload_seconds=0, wait=True, timeout_ms=600_000,
    )
    assert result.admitted and result.message == "Model loaded"
    assert engine.last["query"] == {"wait": "true"}
    assert engine.last["body"] == {
        "model_path": "/models/llm.aimod", "model_id": 7, "topology": "data-parallel",
        "tp_degree": 2, "idle_unload_seconds": 0,
    }


def test_start_model_returns_memory_admission_denial(client, engine):
    engine.route("POST", "/api/models/start", status=409, body={
        "error": "memory admission denied for cuda:0", "model_id": 7, "status": "denied",
        "reason": "memory_admission",
        "requested": [{"domain": "cuda:0", "bytes": 8589934592},
                      {"domain": "host", "bytes": 1048576}],
    })
    result = client.start_model("/models/llm.aimod", wait=True)
    assert result.denied and not result.admitted and result.http_status == 409
    assert result.reason == "memory_admission"
    assert result.message == "memory admission denied for cuda:0"
    assert [(item.domain, item.bytes) for item in result.requested] == [
        ("cuda:0", 8589934592), ("host", 1048576),
    ]


def test_start_model_returns_device_unavailable_denial(client, engine):
    engine.route("POST", "/api/models/start", status=409, body={
        "error": "requests CUDA device 1", "model_id": 7, "status": "denied",
        "reason": "device_unavailable", "device_id": 1,
    })
    result = client.start_model("/models/llm.aimod", wait=True)
    assert result.denied and result.reason == "device_unavailable" and result.device_id == 1


@pytest.mark.parametrize("status,body", [
    (500, {"error": "backend failed", "model_id": 7, "status": "failed"}),
    (400, {"error": "Model path does not exist: \"/missing.aimod\""}),
    (409, {"error": "conflict without a load status"}),
])
def test_start_model_raises_for_failures(client, engine, status, body):
    engine.route("POST", "/api/models/start", status=status, body=body)
    with pytest.raises(KapslHttpError) as raised:
        client.start_model("/missing.aimod", wait=True)
    assert raised.value.status == status and raised.value.body == body


def test_start_model_validation(client):
    for kwargs in ({"tp_degree": 0}, {"idle_unload_seconds": -1}, {"model_id": 2**32},
                   {"topology": ""}):
        with pytest.raises(ValueError):
            client.start_model("/m.aimod", **kwargs)
    with pytest.raises(TypeError):
        client.start_model("/m.aimod", wait="yes")
    with pytest.raises(ValueError):
        client.start_model("")


def test_stop_and_remove_return_engine_messages(client, engine):
    engine.route("POST", "/api/models/3/stop", body={
        "message": "Model 3 was already stopped for being idle; it will stay stopped until started",
    })
    engine.route("POST", "/api/models/3/remove", body={"message": "Model 3 removed successfully"})
    assert "stay stopped" in client.stop_model(3)
    assert engine.last["body"] is None and engine.last["headers"]["content-length"] == "0"
    assert client.remove_model(3) == "Model 3 removed successfully"
    engine.route("POST", "/api/models/8/stop", status=404, body={"error": "Model ID 8 not found"})
    with pytest.raises(KapslHttpError, match="Model ID 8 not found") as raised:
        client.stop_model(8)
    assert raised.value.status == 404


def test_scaling_policy_round_trip(client, engine):
    engine.route("GET", "/api/models/3/scaling", body={
        "min_replicas": 1, "max_replicas": 4, "target_queue_depth": 5,
        "scale_down_threshold": 2, "cooldown_seconds": 300,
    })
    engine.route("POST", "/api/models/3/scaling", body={"status": "ok"})
    policy = client.get_scaling_policy(3)
    assert policy == ScalingPolicy()
    updated = dataclasses.replace(policy, max_replicas=2, cooldown_seconds=60)
    assert client.set_scaling_policy(3, updated) is None
    assert engine.last["body"] == {
        "min_replicas": 1, "max_replicas": 2, "target_queue_depth": 5,
        "scale_down_threshold": 2, "cooldown_seconds": 60,
    }


@pytest.mark.parametrize("fields", [
    {"min_replicas": 0}, {"min_replicas": 3, "max_replicas": 2},
    {"target_queue_depth": 0}, {"cooldown_seconds": 0}, {"max_replicas": -1},
])
def test_scaling_policy_is_validated_before_sending(client, engine, fields):
    with pytest.raises(ValueError):
        client.set_scaling_policy(3, ScalingPolicy(**fields))
    assert engine.requests == []
    with pytest.raises(TypeError):
        client.set_scaling_policy(3, {"min_replicas": 1})


def test_login_checks_the_given_token_and_maps_rejections(client, engine):
    engine.route("POST", "/api/auth/login", body={
        "authenticated": True, "auth_enabled": True, "role_token_auth_enabled": False,
        "role": "reader", "scopes": ["api:read"], "mode": "api-key",
        "access": {"read": True, "write": False, "admin": False},
    })
    login = client.login("Bearer " + READER)
    assert login.role == "reader" and login.mode == "api-key" and login.scopes == ("api:read",)
    assert login.access.read and not login.access.admin
    assert engine.last["headers"]["authorization"] == f"Bearer {READER}"
    assert engine.last["body"] == {}
    client.login()
    assert engine.last["headers"]["authorization"] == f"Bearer {ADMIN}"
    engine.route("POST", "/api/auth/login", status=401, body={
        "error": "Unauthorized", "detail": "Invalid or missing API token.",
    })
    with pytest.raises(KapslHttpError, match="Unauthorized: Invalid or missing") as raised:
        client.login("kvg1_test_wrong")
    assert raised.value.status == 401


def test_access_status_and_role_summaries(client, engine):
    engine.route("GET", "/api/auth/access/status", body={
        "auth_enabled": True, "role_token_auth_enabled": True,
        "store_path": "/var/lib/kapsl/auth.json", "user_count": 2, "key_count": 3,
        "active_key_count": 2, "active_admin_key_count": 1,
    })
    engine.route("GET", "/api/auth/access/roles", body=[
        {"role": "admin", "description": "Full control", "user_count": 1,
         "active_key_count": 1},
    ])
    status = client.auth_status()
    assert status.auth_enabled and status.active_admin_key_count == 1
    assert client.role_summaries()[0].role == "admin"


def test_role_token_status_never_keeps_token_values(client, engine):
    engine.route("GET", "/api/auth/roles", body={
        "reader": {"configured": True}, "writer": {"configured": False},
        "admin": {"configured": True},
    })
    assert client.role_token_status() == RoleTokenStatus(True, False, True)
    # Engines before 0.3.0 returned the values themselves.
    engine.route("GET", "/api/auth/roles", body={
        "reader_token": "kvg1_test_r", "admin_token": "kvg1_test_a",
    })
    status = client.role_token_status()
    assert status == RoleTokenStatus(True, False, True)
    assert "kvg1_test" not in repr(status)


def test_set_role_tokens_replaces_all_roles(client, engine):
    engine.route("POST", "/api/auth/roles", body={
        "reader": {"configured": True}, "writer": {"configured": False},
        "admin": {"configured": True},
    })
    status = client.set_role_tokens(reader_token="kvg1_test_r", admin_token="kvg1_test_a")
    assert status.reader_configured and not status.writer_configured
    assert engine.last["body"] == {"reader_token": "kvg1_test_r", "admin_token": "kvg1_test_a"}
    count = len(engine.requests)
    with pytest.raises(ValueError, match="admin_token"):
        client.set_role_tokens(reader_token="kvg1_test_r")
    assert len(engine.requests) == count
    client.set_role_tokens()
    assert engine.last["body"] == {}


def test_user_management(client, engine):
    user = {"id": "usr_1", "username": "svc", "display_name": "Service", "role": "reader",
            "status": "active", "created_at": 1, "updated_at": 2, "key_count": 0,
            "active_key_count": 0}
    engine.route("GET", "/api/auth/access/users", body=[user])
    engine.route("POST", "/api/auth/access/users", status=201, body=user)
    engine.route("PATCH", "/api/auth/access/users/usr_1", body=dict(
        user, role="writer", status="suspended", display_name=None,
    ))
    assert client.list_users()[0].display_name == "Service"
    created = client.create_user("svc", "reader", display_name="Service")
    assert created.id == "usr_1"
    assert engine.last["body"] == {"username": "svc", "role": "reader", "display_name": "Service"}
    updated = client.update_user("usr_1", role="writer", status="suspended", display_name="")
    assert updated.role == "writer" and updated.status == "suspended"
    assert engine.last["method"] == "PATCH"
    assert engine.last["body"] == {"display_name": "", "role": "writer", "status": "suspended"}
    with pytest.raises(ValueError):
        client.create_user("svc", "owner")
    with pytest.raises(ValueError):
        client.update_user("usr_1", status="deleted")


def test_api_key_management_hides_the_secret(client, engine):
    engine.route("POST", "/api/auth/access/users/usr_0123456789abcdef/keys", status=201,
                 body={"api_key": KEY, "raw_key": "kvg1_test_secret"})
    engine.route("GET", "/api/auth/access/keys", body=[KEY])
    engine.route("POST", "/api/auth/access/keys/key_0123456789abcdef/revoke",
                 body=dict(KEY, active=False, revoked_at=1764300100))
    created = client.create_api_key(
        "usr_0123456789abcdef", "service-key", scopes=["api:read"], expires_in_days=30,
    )
    assert created.raw_key == "kvg1_test_secret" and "kvg1_test_secret" not in repr(created)
    assert created.api_key.scopes == ("api:read",)
    assert engine.last["body"] == {"name": "service-key", "scopes": ["api:read"],
                                   "expires_in_days": 30}
    keys = client.list_api_keys("usr_0123456789abcdef")
    assert engine.last["query"] == {"user_id": "usr_0123456789abcdef"}
    assert keys[0].active and keys[0].last_used_at is None
    client.list_api_keys()
    assert engine.last["query"] == {}
    revoked = client.revoke_api_key("key_0123456789abcdef")
    assert not revoked.active and revoked.revoked_at == 1764300100
    with pytest.raises(ValueError):
        client.create_api_key("usr_1", "k", expires_in_days=0)
    with pytest.raises(TypeError):
        client.create_api_key("usr_1", "k", scopes="api:read")


def test_path_segments_are_escaped(client, engine):
    engine.route("POST", "/api/auth/access/keys/a/b/revoke", body=KEY)
    with pytest.raises(KapslHttpError):
        client.revoke_api_key("a/b")
    assert engine.last["path"] == "/api/auth/access/keys/a%2Fb/revoke"


def test_forbidden_responses_raise(client, engine):
    engine.route("GET", "/metrics", status=403, body={"error": "Forbidden"})
    with pytest.raises(KapslHttpError, match="HTTP 403: Forbidden") as raised:
        client.metrics()
    assert raised.value.reason is None and raised.value.retry_after_seconds is None


def test_openai_models(client, engine):
    engine.route("GET", "/v1/models", body={"object": "list", "data": [
        {"id": "assistant-7b", "object": "model", "created": 1764300000, "owned_by": "kapsl"},
    ]})
    engine.route("GET", "/v1/models/assistant-7b:1.0.0", body={
        "id": "assistant-7b", "object": "model", "created": 1764300000, "owned_by": "kapsl",
    })
    assert [model.id for model in client.list_openai_models()] == ["assistant-7b"]
    # The engine does not percent-decode path parameters; ":" must stay literal.
    assert client.get_openai_model("assistant-7b:1.0.0").owned_by == "kapsl"
    assert engine.last["path"] == "/v1/models/assistant-7b:1.0.0"


def test_chat_completion_sends_parameters_and_session(client, engine):
    engine.route("POST", "/v1/chat/completions", body=chat_body())
    completion = client.chat_completion(
        "assistant-7b",
        [{"role": "system", "content": "Be terse."}, {"role": "user", "content": "Hi"}],
        max_tokens=16, temperature=0.2, top_p=0.9, top_k=40, repetition_penalty=1.1,
        seed=7, stop=["END"], user="end-user", session_id="conversation-1",
        extra_body={"min_tokens": 2},
    )
    assert isinstance(completion, ChatCompletion)
    assert completion.text == "Hello there" and completion.finish_reason == "stop"
    assert completion.usage.total_tokens == 6
    assert completion.choices[0].extra == {"logprobs": None}
    request = engine.last
    assert request["headers"]["x-kapsl-session"] == "conversation-1"
    assert request["body"] == {
        "model": "assistant-7b", "stream": False,
        "messages": [{"role": "system", "content": "Be terse."},
                     {"role": "user", "content": "Hi"}],
        "max_tokens": 16, "temperature": 0.2, "top_p": 0.9, "top_k": 40,
        "repetition_penalty": 1.1, "seed": 7, "stop": ["END"], "user": "end-user",
        "min_tokens": 2,
    }


def test_chat_completion_accepts_previous_replies(client, engine):
    engine.route("POST", "/v1/chat/completions", body=chat_body())
    reply = client.chat_completion("assistant-7b", [{"role": "user", "content": "Hi"}])
    client.chat_completion("assistant-7b", [
        {"role": "user", "content": "Hi"}, reply.choices[0].message,
        {"role": "user", "content": "More"},
    ])
    assert engine.last["body"]["messages"][1] == {"role": "assistant", "content": "Hello there"}
    assert isinstance(reply.choices[0].message, ChatCompletionMessage)


def test_chat_completion_reports_finish_reason_unchanged(client, engine):
    # Managed vLLM responses are relayed, so "length" can arrive; the
    # translated path of engine 0.3.0 always says "stop".
    engine.route("POST", "/v1/chat/completions", body=chat_body("length"))
    assert client.chat_completion("m", [{"role": "user", "content": "x"}]).finish_reason == "length"


@pytest.mark.parametrize("kwargs,error", [
    ({"messages": []}, ValueError), ({"messages": "hi"}, TypeError),
    ({"messages": [{"content": "no role"}]}, ValueError),
    ({"temperature": -1}, ValueError), ({"top_p": 2}, ValueError),
    ({"max_tokens": -1}, ValueError), ({"stop": [1]}, TypeError),
    ({"extra_body": {"model": "other"}}, ValueError), ({"extra_body": []}, TypeError),
    ({"session_id": ""}, ValueError),
])
def test_chat_completion_validation(client, engine, kwargs, error):
    arguments = {"model": "m", "messages": [{"role": "user", "content": "x"}], **kwargs}
    with pytest.raises(error):
        client.chat_completion(**arguments)
    assert engine.requests == []


def test_idle_reload_memory_denial_carries_a_retry_hint(client, engine):
    engine.route("POST", "/v1/chat/completions", status=503, headers={"Retry-After": "10"},
                 body={"error": {
                     "message": "The model 'assistant-7b' is not ready: model 3 was stopped "
                                "for being idle and cannot be reloaded right now",
                     "type": "server_error", "param": None, "code": None,
                 }})
    with pytest.raises(KapslHttpError) as raised:
        client.chat_completion("assistant-7b", [{"role": "user", "content": "x"}])
    error = raised.value
    assert error.status == 503 and error.retry_after_seconds == 10
    assert error.error_type == "server_error" and "stopped for being idle" in error.message


def test_native_route_errors_keep_reason_and_retry_hint():
    error = KapslHttpError.from_response(503, {
        "error": "model 3 was stopped for being idle and cannot be reloaded right now",
        "reason": "idle_reload_memory_unavailable", "retry_after_seconds": 10,
    })
    assert error.reason == "idle_reload_memory_unavailable"
    assert error.retry_after_seconds == 10 and str(error).startswith("HTTP 503: model 3")


def test_chat_completion_stream(client, engine):
    usage = "data: " + json.dumps({
        "id": "chatcmpl-test", "object": "chat.completion.chunk", "created": 1764300000,
        "model": "assistant-7b", "choices": [],
        "usage": {"prompt_tokens": 2, "completion_tokens": 2, "total_tokens": 4},
    }) + "\n\n"
    engine.route("POST", "/v1/chat/completions", stream=[
        ": keep-alive\n\n",
        chunk({"role": "assistant", "content": ""}), chunk({"content": "Hel"}),
        chunk({"content": "lo"}), chunk({}, "stop"), usage, "data: [DONE]\n\n",
    ])
    with client.chat_completion_stream(
        "assistant-7b", [{"role": "user", "content": "Hi"}], include_usage=True,
        session_id="conversation-1",
    ) as stream:
        chunks = list(stream)
    assert stream.closed
    assert all(isinstance(item, ChatCompletionChunk) for item in chunks)
    assert "".join(item.text for item in chunks) == "Hello"
    assert chunks[3].finish_reason == "stop"
    assert chunks[-1].choices == () and chunks[-1].usage.total_tokens == 4
    request = engine.last
    assert request["headers"]["accept"] == "text/event-stream"
    assert request["headers"]["x-kapsl-session"] == "conversation-1"
    assert request["body"]["stream"] is True
    assert request["body"]["stream_options"] == {"include_usage": True}


def test_chat_completion_stream_async_iteration(client, engine):
    engine.route("POST", "/v1/chat/completions", stream=[
        chunk({"content": "a"}), chunk({"content": "b"}), chunk({}, "stop"),
        "data: [DONE]\n\n",
    ])

    async def collect():
        async with client.chat_completion_stream("m", [{"role": "user", "content": "x"}]) as stream:
            return [item.text async for item in stream]

    assert asyncio.run(collect()) == ["a", "b", ""]


def test_chat_completion_stream_reports_in_band_errors(client, engine):
    error = 'data: {"error": {"message": "backend failed", "type": "server_error", ' \
            '"param": null, "code": null}}\n\n'
    engine.route("POST", "/v1/chat/completions", stream=[
        chunk({"content": "partial"}), error, "data: [DONE]\n\n",
    ])
    stream = client.chat_completion_stream("m", [{"role": "user", "content": "x"}])
    assert next(stream).text == "partial"
    with pytest.raises(KapslHttpError, match="backend failed") as raised:
        next(stream)
    assert raised.value.status == 200 and raised.value.error_type == "server_error"
    assert stream.closed


def test_chat_completion_stream_detects_truncation(client, engine):
    engine.route("POST", "/v1/chat/completions", stream=[chunk({"content": "a"}), "data: {"])
    stream = client.chat_completion_stream("m", [{"role": "user", "content": "x"}])
    assert next(stream).text == "a"
    with pytest.raises(ConnectionError, match="before \\[DONE\\]"):
        next(stream)


def test_chat_completion_stream_can_be_cancelled(client, engine):
    release = threading.Event()
    engine.route("POST", "/v1/chat/completions", stream=[
        chunk({"content": "first"}), lambda: release.wait(5), chunk({"content": "late"}),
        "data: [DONE]\n\n",
    ])
    stream = client.chat_completion_stream("m", [{"role": "user", "content": "x"}])
    assert next(stream).text == "first"
    started = time.monotonic()
    assert stream.cancel() is True
    assert time.monotonic() - started < 2
    assert stream.closed and stream.cancel() is False
    with pytest.raises(StopIteration):
        next(stream)
    release.set()


def test_chat_completion_stream_cancel_from_another_thread(client, engine):
    release = threading.Event()
    engine.route("POST", "/v1/chat/completions", stream=[
        chunk({"content": "first"}), lambda: release.wait(10),
        chunk({"content": "late"}), "data: [DONE]\n\n",
    ])
    stream = client.chat_completion_stream("m", [{"role": "user", "content": "x"}])
    assert next(stream).text == "first"
    outcome = []

    def reader():
        try:
            outcome.append(next(stream))
        except BaseException as error:  # noqa: BLE001 - recorded for the assertion
            outcome.append(error)

    thread = threading.Thread(target=reader)
    thread.start()
    time.sleep(0.2)
    started = time.monotonic()
    assert stream.cancel() is True
    # Cancelling does not wait for the blocked read.
    assert time.monotonic() - started < 1 and stream.closed
    # The read ends once the connection closes (on Linux and macOS, at once).
    release.set()
    thread.join(5)
    assert not thread.is_alive()
    assert len(outcome) == 1 and isinstance(outcome[0], StopIteration)


def test_chat_completion_stream_asyncio_cancellation(client, engine):
    release = threading.Event()
    engine.route("POST", "/v1/chat/completions", stream=[
        chunk({"content": "first"}), lambda: release.wait(10),
        chunk({"content": "late"}), "data: [DONE]\n\n",
    ])

    async def run():
        stream = client.chat_completion_stream("m", [{"role": "user", "content": "x"}])
        assert (await stream.__anext__()).text == "first"
        started = time.monotonic()
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(stream.__anext__(), 0.2)
        # The cancelled read did not block the event loop.
        elapsed = time.monotonic() - started
        assert stream.closed
        with pytest.raises(StopAsyncIteration):
            await stream.__anext__()
        release.set()
        return elapsed

    try:
        assert asyncio.run(run()) < 1.5
    finally:
        release.set()


def test_chat_completion_stream_raises_http_errors_before_streaming(client, engine):
    engine.route("POST", "/v1/chat/completions", status=404, body={"error": {
        "message": "The model 'gpt-4o' does not exist. Available models: assistant-7b",
        "type": "invalid_request_error", "param": None, "code": None,
    }})
    with pytest.raises(KapslHttpError) as raised:
        client.chat_completion_stream("gpt-4o", [{"role": "user", "content": "x"}])
    assert raised.value.status == 404 and raised.value.error_type == "invalid_request_error"


def test_embeddings(client, engine):
    engine.route("POST", "/v1/embeddings", body={
        "object": "list", "model": "embed",
        "data": [{"object": "embedding", "embedding": [0.5, -0.25], "index": 0},
                 {"object": "embedding", "embedding": [0.0, 1.0], "index": 1}],
        "usage": {"prompt_tokens": 4, "total_tokens": 4},
    })
    result = client.embeddings("embed", ["a", "b"], encoding_format="float",
                               session_id="s", user="u")
    assert [item.embedding for item in result.data] == [[0.5, -0.25], [0.0, 1.0]]
    assert result.usage.prompt_tokens == 4 and result.usage.completion_tokens == 0
    assert engine.last["body"] == {"model": "embed", "input": ["a", "b"],
                                   "encoding_format": "float", "user": "u"}
    with pytest.raises(ValueError):
        client.embeddings("embed", "a", encoding_format="int8")


def test_connection_failures_raise_os_errors(engine):
    url = engine.url
    engine.close()
    with KapslHttpClient(url, timeout_ms=2000) as client:
        with pytest.raises(urllib.error.URLError):
            client.health()
