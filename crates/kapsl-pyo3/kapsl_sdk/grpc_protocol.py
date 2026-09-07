"""Bundled protobuf messages and stubs for direct typed integrations."""

from .grpc_client import _load_protocol

_, inference, inference_grpc, kapsl, kapsl_grpc = _load_protocol()

__all__ = ["inference", "inference_grpc", "kapsl", "kapsl_grpc"]
