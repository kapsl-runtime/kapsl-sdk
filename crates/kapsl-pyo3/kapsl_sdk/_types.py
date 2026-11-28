"""Transport-independent Python request validation and typed results."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
import math
from numbers import Real
from typing import Any, Mapping, Optional


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


# HTTP API results. Each record keeps the response fields this SDK version
# does not model in ``extra``, so newer engine fields stay reachable. Fields
# added in a later engine release default to ``None``/``False`` when an older
# engine omits them.

def _extra():
    return field(default_factory=dict, compare=False, repr=False)


def _record(cls, value, **convert):
    if not isinstance(value, Mapping):
        raise ValueError(f"Expected a JSON object for {cls.__name__}, got {type(value).__name__}")
    names = {item.name for item in fields(cls) if item.name != "extra"}
    values = {}
    for name in names & value.keys():
        item = value[name]
        if name in convert and item is not None:
            item = convert[name](item)
        values[name] = item
    extra = {key: item for key, item in value.items() if key not in names}
    try:
        return cls(**values, extra=extra)
    except TypeError as error:
        raise ValueError(f"Malformed {cls.__name__} response: {error}") from None


def _records(cls, **convert):
    def build(items):
        if not isinstance(items, list):
            raise ValueError(f"Expected a JSON array of {cls.__name__} objects")
        return tuple(_record(cls, item, **convert) for item in items)
    return build


@dataclass(frozen=True)
class HealthStatus:
    """``GET /api/health``: ``status`` is ``healthy`` or ``degraded``."""

    status: str
    total_models: int = 0
    healthy_models: int = 0
    unhealthy_models: int = 0
    extra: Mapping[str, Any] = _extra()


@dataclass(frozen=True)
class ModelStatus:
    """One entry of ``GET /api/models`` or ``GET /api/models/{id}``.

    ``status`` is ``active``, ``inactive``, ``starting``, ``stopping`` or
    ``loading``. ``idle_unloaded``, ``reloads_on_request`` and
    ``idle_unload_seconds`` need engine 0.3.0; ``successful_inferences`` and
    ``failed_inferences`` are only reported for a single model, and the latency
    fields only by the list.
    """

    id: int
    name: str
    version: str = ""
    framework: str = ""
    device: str = ""
    optimization_level: str = ""
    loaded_at: int = 0
    model_path: str = ""
    status: str = ""
    replica_id: int = 0
    base_model_id: int = 0
    format: Optional[str] = None
    model_type: Optional[str] = None
    task: Optional[str] = None
    preprocess: Optional[str] = None
    healthy: bool = True
    active_inferences: int = 0
    total_inferences: int = 0
    successful_inferences: Optional[int] = None
    failed_inferences: Optional[int] = None
    queue_depth: tuple[int, int] = (0, 0)
    memory_usage: int = 0
    gpu_utilization: float = 0.0
    throughput: float = 0.0
    prompt_tokens_total: int = 0
    generated_tokens_total: int = 0
    generated_tokens_per_sec: float = 0.0
    total_tokens_per_sec: float = 0.0
    decode_steps_total: int = 0
    decode_tokens_evaluated_total: int = 0
    avg_tokens_evaluated_per_decode_step: float = 0.0
    kv_partial_reuse_hits_total: int = 0
    kv_partial_reuse_tokens_saved_total: int = 0
    onnx_session_pool_total: int = 0
    onnx_session_pool_idle: int = 0
    onnx_session_pool_waits_total: int = 0
    onnx_session_pool_wait_seconds_total: float = 0.0
    avg_latency_ms: Optional[float] = None
    p99_latency_ms: Optional[float] = None
    time_to_first_token_ms: Optional[float] = None
    idle_unloaded: bool = False
    reloads_on_request: bool = False
    idle_unload_seconds: Optional[int] = None
    extra: Mapping[str, Any] = _extra()

    @property
    def is_replica(self) -> bool:
        return self.id != self.base_model_id


def model_status(value) -> ModelStatus:
    return _record(ModelStatus, value, queue_depth=tuple)


@dataclass(frozen=True)
class MemoryRequest:
    """Bytes a denied load asked for in one memory domain, such as ``cuda:0``."""

    domain: str
    bytes: int
    extra: Mapping[str, Any] = _extra()


@dataclass(frozen=True)
class ModelLoadResult:
    """Outcome of ``POST /api/models/start``.

    ``status`` is ``started`` when the load was only queued (``wait=False``,
    HTTP 202), ``active`` once a waited load is serving, or ``denied`` when the
    engine refused it (HTTP 409). Denials carry ``reason``: ``memory_admission``
    with the ``requested`` bytes per domain, or ``device_unavailable`` with the
    ``device_id`` this process cannot use. Waited results need engine 0.3.0.
    """

    model_id: int
    status: str
    message: str = ""
    reason: Optional[str] = None
    requested: tuple[MemoryRequest, ...] = ()
    device_id: Optional[int] = None
    http_status: int = 0
    extra: Mapping[str, Any] = _extra()

    @property
    def admitted(self) -> bool:
        return self.status == "active"

    @property
    def denied(self) -> bool:
        return self.status == "denied"


def model_load_result(http_status, value) -> ModelLoadResult:
    if not isinstance(value, Mapping):
        raise ValueError("Expected a JSON object for ModelLoadResult")
    value = dict(value)
    message = value.pop("message", None)
    error = value.pop("error", None)
    value["message"] = message if message is not None else (error or "")
    value.setdefault("status", "started" if http_status == 202 else "")
    value["http_status"] = http_status
    return _record(ModelLoadResult, value, requested=_records(MemoryRequest))


@dataclass(frozen=True)
class ScalingPolicy:
    """Replica autoscaling policy of ``/api/models/{id}/scaling``.

    The defaults match the engine's. ``validate`` applies the engine's rules
    before a policy is sent.
    """

    min_replicas: int = 1
    max_replicas: int = 4
    target_queue_depth: int = 5
    scale_down_threshold: int = 2
    cooldown_seconds: int = 300
    extra: Mapping[str, Any] = _extra()

    def validate(self) -> None:
        for name, maximum in (
            ("min_replicas", 2**32 - 1), ("max_replicas", 2**32 - 1),
            ("target_queue_depth", 2**64 - 1), ("scale_down_threshold", 2**64 - 1),
            ("cooldown_seconds", 2**64 - 1),
        ):
            value = getattr(self, name)
            if type(value) is not int or not 0 <= value <= maximum:
                raise ValueError(f"{name} must be a nonnegative integer")
        if self.min_replicas < 1:
            raise ValueError("min_replicas must be at least 1")
        if self.max_replicas < self.min_replicas:
            raise ValueError("max_replicas must be greater than or equal to min_replicas")
        if self.target_queue_depth == 0:
            raise ValueError("target_queue_depth must be greater than 0")
        if self.cooldown_seconds == 0:
            raise ValueError("cooldown_seconds must be greater than 0")

    def to_json(self) -> dict[str, int]:
        value = asdict(self)
        value.pop("extra")
        return value


def scaling_policy(value) -> ScalingPolicy:
    return _record(ScalingPolicy, value)


@dataclass(frozen=True)
class CudaVersions:
    driver_version: Optional[str] = None
    driver_cuda_version: Optional[str] = None
    runtime_version: Optional[str] = None
    extra: Mapping[str, Any] = _extra()


@dataclass(frozen=True)
class ModelMemoryUsage:
    """A workload's rows in the memory authority, rolled up by base model.

    ``owner_kind`` is ``model``, ``runtime`` or ``external_kv_participant``.
    """

    model_id: int
    owner_kind: str = "model"
    name: str = ""
    replica_count: int = 0
    planned_bytes: int = 0
    reserved_bytes: int = 0
    committed_bytes: int = 0
    observed_bytes: int = 0
    used_bytes: int = 0
    percentage: float = 0.0
    extra: Mapping[str, Any] = _extra()


@dataclass(frozen=True)
class MemoryDomainUsage:
    """One memory domain, such as ``host`` or ``cuda:0``.

    ``available_bytes`` is what a new allocation would be granted;
    ``authority_available_bytes`` (engine 0.3.0) is the authority's own room
    before the CUDA device ledger is applied.
    """

    domain: str
    budget_bytes: int = 0
    planned_bytes: int = 0
    reserved_bytes: int = 0
    committed_bytes: int = 0
    observed_bytes: int = 0
    used_bytes: int = 0
    available_bytes: int = 0
    authority_available_bytes: Optional[int] = None
    extra: Mapping[str, Any] = _extra()


@dataclass(frozen=True)
class MemoryAuthority:
    model_bytes: int = 0
    domain_used_bytes: int = 0
    domain_budget_bytes: int = 0
    domain_available_bytes: int = 0
    foreign_pressure_active: bool = False
    models: tuple[ModelMemoryUsage, ...] = ()
    domains: tuple[MemoryDomainUsage, ...] = ()
    extra: Mapping[str, Any] = _extra()

    def domain(self, name: str) -> Optional[MemoryDomainUsage]:
        return next((domain for domain in self.domains if domain.domain == name), None)


@dataclass(frozen=True)
class SystemStats:
    """``GET /api/system/stats``.

    ``pressure_state`` is ``normal``, ``conserve`` or ``emergency``.
    ``cuda``, ``gpu_sampling`` (``fresh``, ``stale`` or ``failed``) and
    ``gpu_sampling_age_secs`` need engine 0.3.0.
    """

    pid: int
    process_memory_bytes: int = 0
    total_system_memory_bytes: Optional[int] = None
    gpu_utilization: float = 0.0
    gpu_memory_bytes: Optional[int] = None
    gpu_memory_total_bytes: Optional[int] = None
    foreign_gpu_memory_bytes: Optional[int] = None
    pressure_state: str = ""
    collected_at_ms: int = 0
    cuda: Optional[CudaVersions] = None
    gpu_sampling: Optional[str] = None
    gpu_sampling_age_secs: Optional[int] = None
    memory_authority: Optional[MemoryAuthority] = None
    extra: Mapping[str, Any] = _extra()


def system_stats(value) -> SystemStats:
    return _record(
        SystemStats, value,
        cuda=lambda item: _record(CudaVersions, item),
        memory_authority=lambda item: _record(
            MemoryAuthority, item,
            models=_records(ModelMemoryUsage), domains=_records(MemoryDomainUsage),
        ),
    )


@dataclass(frozen=True)
class AuthAccess:
    read: bool = False
    write: bool = False
    admin: bool = False
    extra: Mapping[str, Any] = _extra()


@dataclass(frozen=True)
class AuthLogin:
    """``POST /api/auth/login``: the access a token grants.

    ``mode`` is ``api-key``, ``role-token`` or ``local-loopback``.
    """

    authenticated: bool
    role: str
    mode: str
    access: AuthAccess
    auth_enabled: bool = True
    role_token_auth_enabled: bool = False
    scopes: tuple[str, ...] = ()
    extra: Mapping[str, Any] = _extra()


def auth_login(value) -> AuthLogin:
    return _record(
        AuthLogin, value, access=lambda item: _record(AuthAccess, item), scopes=tuple,
    )


@dataclass(frozen=True)
class AuthStatus:
    auth_enabled: bool
    role_token_auth_enabled: bool = False
    store_path: str = ""
    user_count: int = 0
    key_count: int = 0
    active_key_count: int = 0
    active_admin_key_count: int = 0
    extra: Mapping[str, Any] = _extra()


@dataclass(frozen=True)
class RoleSummary:
    role: str
    description: str = ""
    user_count: int = 0
    active_key_count: int = 0
    extra: Mapping[str, Any] = _extra()


@dataclass(frozen=True)
class RoleTokenStatus:
    """Which legacy role tokens are configured; never the token values."""

    reader_configured: bool = False
    writer_configured: bool = False
    admin_configured: bool = False


def role_token_status(value) -> RoleTokenStatus:
    if not isinstance(value, Mapping):
        raise ValueError("Expected a JSON object for RoleTokenStatus")

    def configured(role):
        state = value.get(role)
        if isinstance(state, Mapping):
            return bool(state.get("configured"))
        # Engines before 0.3.0 returned the token values. Keep only whether
        # one is set, so the credential is never retained.
        return bool(value.get(f"{role}_token"))

    return RoleTokenStatus(configured("reader"), configured("writer"), configured("admin"))


@dataclass(frozen=True)
class AuthUser:
    id: str
    username: str
    role: str
    status: str = "active"
    created_at: int = 0
    updated_at: int = 0
    key_count: int = 0
    active_key_count: int = 0
    display_name: Optional[str] = None
    extra: Mapping[str, Any] = _extra()


@dataclass(frozen=True)
class ApiKey:
    """API key metadata. The secret is only returned once, by key creation."""

    id: str
    user_id: str
    name: str
    key_prefix: str
    active: bool
    username: str = ""
    user_role: str = ""
    scopes: tuple[str, ...] = ()
    created_at: int = 0
    created_by: Optional[str] = None
    last_used_at: Optional[int] = None
    expires_at: Optional[int] = None
    revoked_at: Optional[int] = None
    extra: Mapping[str, Any] = _extra()


def api_key(value) -> ApiKey:
    return _record(ApiKey, value, scopes=tuple)


@dataclass(frozen=True)
class CreatedApiKey:
    """A new API key; ``raw_key`` is the secret and is omitted from ``repr``."""

    api_key: ApiKey
    raw_key: str = field(repr=False)


def created_api_key(value) -> CreatedApiKey:
    if not isinstance(value, Mapping) or not isinstance(value.get("raw_key"), str):
        raise ValueError("Malformed CreatedApiKey response")
    return CreatedApiKey(api_key(value.get("api_key")), value["raw_key"])


@dataclass(frozen=True)
class OpenAIModel:
    id: str
    object: str = "model"
    created: int = 0
    owned_by: str = ""
    extra: Mapping[str, Any] = _extra()


@dataclass(frozen=True)
class TokenUsage:
    """OpenAI ``usage``. Kapsl estimates non-streaming chat usage from text."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    extra: Mapping[str, Any] = _extra()


@dataclass(frozen=True)
class ChatCompletionMessage:
    role: str = "assistant"
    content: Optional[str] = None
    extra: Mapping[str, Any] = _extra()


@dataclass(frozen=True)
class ChatCompletionChoice:
    index: int
    message: ChatCompletionMessage
    finish_reason: Optional[str] = None
    extra: Mapping[str, Any] = _extra()


@dataclass(frozen=True)
class ChatCompletion:
    """A non-streaming ``chat.completion``.

    ``finish_reason`` is reported as the engine sends it. Engine 0.3.0 returns
    ``stop`` even when ``max_tokens`` cut a reply short, except for managed
    vLLM models, whose responses are relayed from vLLM.
    """

    id: str
    model: str
    choices: tuple[ChatCompletionChoice, ...]
    created: int = 0
    object: str = "chat.completion"
    usage: Optional[TokenUsage] = None
    extra: Mapping[str, Any] = _extra()

    @property
    def text(self) -> str:
        return (self.choices[0].message.content or "") if self.choices else ""

    @property
    def finish_reason(self) -> Optional[str]:
        return self.choices[0].finish_reason if self.choices else None


def chat_completion(value) -> ChatCompletion:
    return _record(
        ChatCompletion, value,
        choices=_records(
            ChatCompletionChoice,
            message=lambda item: _record(ChatCompletionMessage, item),
        ),
        usage=lambda item: _record(TokenUsage, item),
    )


@dataclass(frozen=True)
class ChatCompletionDelta:
    role: Optional[str] = None
    content: Optional[str] = None
    extra: Mapping[str, Any] = _extra()


@dataclass(frozen=True)
class ChatCompletionChunkChoice:
    index: int
    delta: ChatCompletionDelta
    finish_reason: Optional[str] = None
    extra: Mapping[str, Any] = _extra()


@dataclass(frozen=True)
class ChatCompletionChunk:
    """One ``chat.completion.chunk`` server-sent event.

    With ``include_usage`` the final chunk has no choices and carries ``usage``.
    """

    id: str
    model: str
    choices: tuple[ChatCompletionChunkChoice, ...] = ()
    created: int = 0
    object: str = "chat.completion.chunk"
    usage: Optional[TokenUsage] = None
    extra: Mapping[str, Any] = _extra()

    @property
    def text(self) -> str:
        return "".join(choice.delta.content or "" for choice in self.choices)

    @property
    def finish_reason(self) -> Optional[str]:
        return next(
            (choice.finish_reason for choice in self.choices if choice.finish_reason), None,
        )


def chat_completion_chunk(value) -> ChatCompletionChunk:
    return _record(
        ChatCompletionChunk, value,
        choices=_records(
            ChatCompletionChunkChoice,
            delta=lambda item: _record(ChatCompletionDelta, item),
        ),
        usage=lambda item: _record(TokenUsage, item),
    )


@dataclass(frozen=True)
class Embedding:
    """``embedding`` is a list of floats, or a base64 string of little-endian
    float32 values when ``encoding_format="base64"`` was requested."""

    index: int
    embedding: Any
    object: str = "embedding"
    extra: Mapping[str, Any] = _extra()


@dataclass(frozen=True)
class EmbeddingList:
    model: str
    data: tuple[Embedding, ...]
    object: str = "list"
    usage: Optional[TokenUsage] = None
    extra: Mapping[str, Any] = _extra()


def embedding_list(value) -> EmbeddingList:
    return _record(
        EmbeddingList, value,
        data=_records(Embedding), usage=lambda item: _record(TokenUsage, item),
    )
