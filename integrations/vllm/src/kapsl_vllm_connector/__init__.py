"""Out-of-tree vLLM KV connector for Kapsl."""

from .client import KapslKvControlClient, KapslKvControlError
from .certification import (
    CertificationError,
    allowlist_entry,
    validate_certification_report,
)
from .connector import (
    ADAPTER_PROFILE_ID,
    ADAPTER_VERSION,
    ELASTIC_ADAPTER_PROFILE_ID,
    KapslConnectorV1,
)
from .contract import (
    ABI_VERSION,
    make_shared_pool_attachment,
    make_shared_pool_detach_request,
    opaque_registration,
    shared_pool_registration,
)
from .planning import (
    PLANNER_SCHEMA_VERSION,
    CacheGroupGeometry,
    ElementType,
    GeometryDescriptor,
    PlannerIdentity,
    PlanningError,
    PlanningResult,
    RankGeometry,
    RankSizing,
    SizingPolicy,
    build_plan,
    extract_rank_geometry,
    geometry_from_resolved_configs,
    planner_error_json_schema,
    planner_json_schema,
)

__all__ = [
    "ABI_VERSION",
    "ADAPTER_PROFILE_ID",
    "ADAPTER_VERSION",
    "ELASTIC_ADAPTER_PROFILE_ID",
    "CertificationError",
    "KapslConnectorV1",
    "KapslKvControlClient",
    "KapslKvControlError",
    "PLANNER_SCHEMA_VERSION",
    "CacheGroupGeometry",
    "ElementType",
    "GeometryDescriptor",
    "PlannerIdentity",
    "PlanningError",
    "PlanningResult",
    "RankGeometry",
    "RankSizing",
    "SizingPolicy",
    "allowlist_entry",
    "build_plan",
    "extract_rank_geometry",
    "geometry_from_resolved_configs",
    "make_shared_pool_attachment",
    "make_shared_pool_detach_request",
    "opaque_registration",
    "planner_error_json_schema",
    "planner_json_schema",
    "shared_pool_registration",
    "validate_certification_report",
]
