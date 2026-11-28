# Authentication

Configure the token for the selected listener. `KapslHttpClient`,
`KapslGrpcClient` and `AsyncKapslGrpcClient` use public API tokens (role tokens
or API keys) through the engine's shared authorization adapter. Native TCP
uses the separately configured `KAPSL_TCP_AUTH_TOKEN`. Local socket and
shared-memory access follow their transport's local access policy.

## Setting the token

Pass `api_token` to the client constructor:

```python
from kapsl_sdk import KapslClient

client = KapslClient(
    "tcp://192.168.1.10:9096",
    api_token="your-api-token",
)
```

The token is attached to every `infer`, `infer_tensor`, and `infer_stream` call without any extra work on your part.

## Public API token roles (gRPC)

Use a public API token with at least **reader** role for gRPC inference and
discovery. The Python gRPC client sends it as `authorization: Bearer TOKEN`
metadata on every RPC; the engine scopes session IDs to that credential.

| Role | Inference | Model management | Admin |
|------|-----------|-----------------|-------|
| `reader` | Yes | No | No |
| `writer` | Yes | Yes | No |
| `admin` | Yes | Yes | Yes |

## Getting a token

Tokens are managed on the runtime side. An admin can create API keys via the
web dashboard, the management API, or `KapslHttpClient`. A key has the role of
the user it belongs to; optional `scopes` such as `api:read` narrow it. The
first active key must belong to an admin user.

```python
from kapsl_sdk import KapslHttpClient

with KapslHttpClient("http://127.0.0.1:9095", api_token="admin-token") as admin:
    user = admin.create_user("my-service", "reader")
    created = admin.create_api_key(user.id, "my-service-key", expires_in_days=90)
    service_token = created.raw_key  # only returned once
    print(admin.login(service_token).access)
```

```bash
# The same request with curl (admin token required)
curl -X POST http://127.0.0.1:9095/api/auth/access/users/{user_id}/keys \
  -H "Authorization: Bearer <admin-token>" \
  -H "Content-Type: application/json" \
  -d '{"name": "my-service-key", "scopes": ["api:read"]}'
```

See the [kapsl-runtime Authentication guide](https://kapsl.ai/docs/engine/authentication) for full token management instructions.

## Local (unauthenticated) deployments

If the runtime is running without authentication (local-only mode), you do not need a token:

```python
# No api_token needed for an unauthenticated local runtime
client = KapslClient()
```

## Token in environment variables

A common pattern is to read the token from an environment variable rather than hardcoding it:

```python
import os
from kapsl_sdk import KapslClient

client = KapslClient(
    os.environ["KAPSL_RUNTIME_URL"],
    api_token=os.environ.get("KAPSL_API_TOKEN"),
)
```

## Error handling

A missing or invalid native TCP token raises `RuntimeError` with the message
`Unauthorized`. gRPC returns `grpc.StatusCode.UNAUTHENTICATED` through
`grpc.RpcError` (or `grpc.aio.AioRpcError` for asynchronous calls).
`KapslHttpClient` raises `KapslHttpError` with `status` 401 for a missing or
invalid token and 403 for a token without the required role, or for a
non-loopback client when engine authentication is disabled.

```python
try:
    result = client.infer(model_id=0, shape=[1, 4], dtype="float32", data=b"...")
except RuntimeError as e:
    if "Unauthorized" in str(e):
        print("Check your api_token")
    raise
```
