# Authentication

When `kapsl-runtime` is running with API authentication enabled, every request must include a valid token. `kapsl-sdk` handles this automatically once you configure the client.

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

## Token roles

`kapsl-runtime` has three roles. Use a token with at least **reader** role to call inference endpoints.

| Role | Inference | Model management | Admin |
|------|-----------|-----------------|-------|
| `reader` | Yes | No | No |
| `writer` | Yes | Yes | No |
| `admin` | Yes | Yes | Yes |

## Getting a token

Tokens are managed on the runtime side. An admin can create API keys via the web dashboard or the management API:

```bash
# Create an API key for a user (admin token required)
curl -X POST http://127.0.0.1:9095/api/auth/access/users/{user_id}/keys \
  -H "Authorization: Bearer <admin-token>" \
  -H "Content-Type: application/json" \
  -d '{"name": "my-service-key", "role": "reader"}'
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

A missing or invalid token raises `RuntimeError` with the message `Unauthorized`:

```python
try:
    result = client.infer(model_id=0, shape=[1, 4], dtype="float32", data=b"...")
except RuntimeError as e:
    if "Unauthorized" in str(e):
        print("Check your api_token")
    raise
```
