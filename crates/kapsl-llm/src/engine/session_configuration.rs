//! Adapter-owned ONNX session configuration, within the integration library.

use kapsl_engine_api::EngineError;
use ort::session::builder::SessionBuilder;
use std::path::Path;

/// The concrete model or pipeline stage being loaded on one device.
#[derive(Clone, Copy, Debug)]
pub struct OnnxSessionContext<'a> {
    pub model_path: &'a Path,
    pub provider: &'a str,
    pub device_id: i32,
}

/// Configure an ONNX session before its model is loaded.
///
/// The adapter registers the requested execution providers and any per-model
/// options here, including TensorRT shape profiles. The hook runs separately
/// for every model file, pipeline stage, device, safe-load retry, and reload.
/// An error aborts that load; the SDK does not retry with its default provider
/// configuration. Implementations must not modify process-wide environment
/// variables to configure a session.
///
/// When a device allocation-scope provider is installed, this hook runs inside
/// its model scope. The SDK enforces environment allocator participation and
/// disables CPU execution-provider fallback after the hook returns. This Rust
/// interface stays inside the integration library; it is not a native host ABI.
pub trait OnnxSessionConfigurator: Send + Sync {
    fn configure(
        &self,
        builder: SessionBuilder,
        context: OnnxSessionContext<'_>,
    ) -> Result<SessionBuilder, EngineError>;
}
