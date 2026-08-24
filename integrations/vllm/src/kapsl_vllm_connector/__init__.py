"""Out-of-tree vLLM KV connector for Kapsl."""

from .client import KapslKvControlClient, KapslKvControlError
from .certification import (
    CertificationError,
    allowlist_entry,
    validate_certification_report,
)
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
    "CertificationError",
    "KapslConnectorV1",
    "KapslKvControlClient",
    "KapslKvControlError",
    "allowlist_entry",
    "make_shared_pool_attachment",
    "make_shared_pool_detach_request",
    "opaque_registration",
    "shared_pool_registration",
    "validate_certification_report",
]
