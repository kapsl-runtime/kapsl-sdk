"""Out-of-tree vLLM KV connector for Kapsl."""

from .client import KapslKvControlClient, KapslKvControlError
from .connector import ADAPTER_PROFILE_ID, ADAPTER_VERSION, KapslConnectorV1
from .contract import (
    ABI_VERSION,
    make_shared_pool_attachment,
    make_shared_pool_detach_request,
    opaque_registration,
    shared_pool_registration,
)

__all__ = [
    "ABI_VERSION",
    "ADAPTER_PROFILE_ID",
    "ADAPTER_VERSION",
    "KapslConnectorV1",
    "KapslKvControlClient",
    "KapslKvControlError",
    "make_shared_pool_attachment",
    "make_shared_pool_detach_request",
    "opaque_registration",
    "shared_pool_registration",
]
