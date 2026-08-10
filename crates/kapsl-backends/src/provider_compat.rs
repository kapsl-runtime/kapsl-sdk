use crate::env_util::read_env_flag;
use kapsl_core::{
    accelerator_provider_pack_installed, AcceleratorProviderPack, ALLOW_UNMANAGED_PROVIDERS_ENV,
};
use kapsl_hal::device::{Device, DeviceBackend, DeviceInfo};
use ort::execution_providers::ExecutionProvider as _;
use ort::execution_providers::{CUDAExecutionProvider, TensorRTExecutionProvider};
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::Value;
use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

const CUDA_SMOKE_TEST_ENV: &str = "KAPSL_ORT_PROVIDER_SMOKE_TEST";
const ALLOW_UNSUPPORTED_CUDA_ENV: &str = "KAPSL_ORT_ALLOW_UNSUPPORTED_CUDA";
const CUDA_MAX_CC_ENV: &str = "KAPSL_ORT_CUDA_MAX_COMPUTE_CAPABILITY";
const DEFAULT_CUDA_MAX_CC: (u32, u32) = (11, 9);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum OnnxAcceleratorProvider {
    Cuda,
    TensorRt,
}

impl OnnxAcceleratorProvider {
    fn as_str(self) -> &'static str {
        match self {
            Self::Cuda => "cuda",
            Self::TensorRt => "tensorrt",
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct ProviderAcceptance {
    pub bundle: &'static str,
    pub device_summary: String,
}

pub(crate) fn resolve_onnx_accelerator_provider(
    provider: OnnxAcceleratorProvider,
    device_info: &DeviceInfo,
    device_id: i32,
) -> Result<ProviderAcceptance, String> {
    let pack = match provider {
        OnnxAcceleratorProvider::Cuda => AcceleratorProviderPack::Cuda,
        OnnxAcceleratorProvider::TensorRt => AcceleratorProviderPack::TensorRt,
    };
    if !accelerator_provider_pack_installed(pack) {
        return Err(format!(
            "{} provider pack is not installed. Install the Kapsl {} accelerator pack, or set {}=1 for an explicitly managed system installation",
            provider.as_str(),
            pack.display_name(),
            ALLOW_UNMANAGED_PROVIDERS_ENV
        ));
    }

    let device = cuda_device(device_info, device_id).ok_or_else(|| {
        format!(
            "{} requires CUDA device {}, but no matching CUDA device was detected",
            provider.as_str(),
            device_id
        )
    })?;

    validate_cuda_static_compatibility(device)?;

    if smoke_test_enabled() {
        smoke_test_provider(provider, device_id)?;
    } else {
        log::warn!(
            "ORT {} provider smoke test disabled by {}",
            provider.as_str(),
            CUDA_SMOKE_TEST_ENV
        );
    }

    Ok(ProviderAcceptance {
        bundle: "ort 2.0.0-rc.11 download-binaries accelerator bundle",
        device_summary: cuda_device_summary(device),
    })
}

fn cuda_device(device_info: &DeviceInfo, device_id: i32) -> Option<&Device> {
    if device_id < 0 {
        return None;
    }
    let id = device_id as usize;
    device_info
        .devices
        .iter()
        .find(|device| matches!(device.backend, DeviceBackend::Cuda) && device.id == id)
}

fn validate_cuda_static_compatibility(device: &Device) -> Result<(), String> {
    validate_cuda_static_compatibility_with(
        device,
        configured_max_cuda_cc(),
        allow_unsupported_cuda(),
    )
}

fn validate_cuda_static_compatibility_with(
    device: &Device,
    max_cc: (u32, u32),
    allow_unsupported: bool,
) -> Result<(), String> {
    let Some(cc) = device
        .compute_capability
        .as_deref()
        .and_then(parse_dotted_version)
    else {
        return Ok(());
    };

    if compare_versions(cc, max_cc).is_gt() && !allow_unsupported {
        return Err(format!(
            "CUDA device '{}' reports compute capability {}.{}, but the bundled ONNX Runtime CUDA provider is only accepted through {}.{} by default. \
Set {}=1 to force CUDA, or install/select a provider bundle built for this GPU generation.",
            device.name,
            cc.0,
            cc.1,
            max_cc.0,
            max_cc.1,
            ALLOW_UNSUPPORTED_CUDA_ENV
        ));
    }

    Ok(())
}

fn configured_max_cuda_cc() -> (u32, u32) {
    std::env::var(CUDA_MAX_CC_ENV)
        .ok()
        .and_then(|value| parse_dotted_version(value.trim()))
        .unwrap_or(DEFAULT_CUDA_MAX_CC)
}

fn allow_unsupported_cuda() -> bool {
    read_env_flag(ALLOW_UNSUPPORTED_CUDA_ENV, false)
}

fn smoke_test_enabled() -> bool {
    read_env_flag(CUDA_SMOKE_TEST_ENV, true)
}

fn parse_dotted_version(value: &str) -> Option<(u32, u32)> {
    let trimmed = value.trim();
    if trimmed.is_empty() || trimmed.eq_ignore_ascii_case("n/a") {
        return None;
    }

    let mut parts = trimmed.split('.');
    let major = parts.next()?.parse::<u32>().ok()?;
    let minor = parts
        .next()
        .and_then(|part| part.parse::<u32>().ok())
        .unwrap_or(0);
    Some((major, minor))
}

fn compare_versions(left: (u32, u32), right: (u32, u32)) -> std::cmp::Ordering {
    left.0.cmp(&right.0).then_with(|| left.1.cmp(&right.1))
}

fn cuda_device_summary(device: &Device) -> String {
    format!(
        "{} (id={}, driver={}, cuda={}, cc={})",
        device.name,
        device.id,
        device.driver_version.as_deref().unwrap_or("unknown"),
        device.cuda_version.as_deref().unwrap_or("unknown"),
        device.compute_capability.as_deref().unwrap_or("unknown")
    )
}

type ProviderSmokeCache = Mutex<HashMap<(OnnxAcceleratorProvider, i32), Result<(), String>>>;

fn smoke_cache() -> &'static ProviderSmokeCache {
    static CACHE: OnceLock<ProviderSmokeCache> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

fn smoke_test_provider(provider: OnnxAcceleratorProvider, device_id: i32) -> Result<(), String> {
    let key = (provider, device_id);
    if let Ok(cache) = smoke_cache().lock() {
        if let Some(result) = cache.get(&key) {
            return result.clone();
        }
    }

    let result = run_smoke_test_provider(provider, device_id);

    if let Ok(mut cache) = smoke_cache().lock() {
        cache.insert(key, result.clone());
    }

    result
}

fn run_smoke_test_provider(
    provider: OnnxAcceleratorProvider,
    device_id: i32,
) -> Result<(), String> {
    let mut builder = Session::builder()
        .map_err(|e| format!("failed to create ORT smoke-test session builder: {e}"))?
        .with_optimization_level(GraphOptimizationLevel::Disable)
        .map_err(|e| format!("failed to configure ORT smoke-test optimization level: {e}"))?;

    builder = match provider {
        OnnxAcceleratorProvider::Cuda => {
            if !CUDAExecutionProvider::default()
                .is_available()
                .unwrap_or(false)
            {
                return Err("CUDA execution provider is not available in ONNX Runtime".to_string());
            }
            builder
                .with_execution_providers([CUDAExecutionProvider::default()
                    .with_device_id(device_id)
                    .build()])
                .map_err(|e| format!("failed to register CUDA execution provider: {e}"))?
        }
        OnnxAcceleratorProvider::TensorRt => {
            if !TensorRTExecutionProvider::default()
                .is_available()
                .unwrap_or(false)
            {
                return Err(
                    "TensorRT execution provider is not available in ONNX Runtime".to_string(),
                );
            }
            builder
                .with_execution_providers([
                    TensorRTExecutionProvider::default()
                        .with_device_id(device_id)
                        .build(),
                    CUDAExecutionProvider::default()
                        .with_device_id(device_id)
                        .build(),
                ])
                .map_err(|e| format!("failed to register TensorRT execution provider: {e}"))?
        }
    };

    let mut session = builder
        .commit_from_memory(&identity_smoke_model())
        .map_err(|e| {
            format!(
                "{} execution provider failed ORT smoke-test session creation: {e}",
                provider.as_str()
            )
        })?;
    let input = Value::from_array(([1_usize], vec![1.0_f32]))
        .map_err(|e| format!("failed to create ORT smoke-test input tensor: {e}"))?;

    session.run(ort::inputs![input]).map(|_| ()).map_err(|e| {
        format!(
            "{} execution provider failed ORT smoke-test inference: {e}",
            provider.as_str()
        )
    })
}

fn identity_smoke_model() -> Vec<u8> {
    let mut model = Vec::new();
    write_varint_field(&mut model, 1, 7);
    write_message_field(&mut model, 7, &graph_proto());
    write_message_field(&mut model, 8, &opset_import_proto(13));
    model
}

fn graph_proto() -> Vec<u8> {
    let mut graph = Vec::new();
    write_message_field(&mut graph, 1, &identity_node_proto());
    write_string_field(&mut graph, 2, "kapsl_provider_smoke");
    write_message_field(&mut graph, 11, &value_info_proto("input"));
    write_message_field(&mut graph, 12, &value_info_proto("output"));
    graph
}

fn identity_node_proto() -> Vec<u8> {
    let mut node = Vec::new();
    write_string_field(&mut node, 1, "input");
    write_string_field(&mut node, 2, "output");
    write_string_field(&mut node, 4, "Identity");
    node
}

fn value_info_proto(name: &str) -> Vec<u8> {
    let mut value = Vec::new();
    write_string_field(&mut value, 1, name);
    write_message_field(&mut value, 2, &type_proto());
    value
}

fn type_proto() -> Vec<u8> {
    let mut tensor = Vec::new();
    write_varint_field(&mut tensor, 1, 1);
    write_message_field(&mut tensor, 2, &tensor_shape_proto());

    let mut ty = Vec::new();
    write_message_field(&mut ty, 1, &tensor);
    ty
}

fn tensor_shape_proto() -> Vec<u8> {
    let mut dim = Vec::new();
    write_varint_field(&mut dim, 1, 1);

    let mut shape = Vec::new();
    write_message_field(&mut shape, 1, &dim);
    shape
}

fn opset_import_proto(version: u64) -> Vec<u8> {
    let mut opset = Vec::new();
    write_varint_field(&mut opset, 2, version);
    opset
}

fn write_string_field(buf: &mut Vec<u8>, field: u32, value: &str) {
    write_key(buf, field, 2);
    write_varint(buf, value.len() as u64);
    buf.extend_from_slice(value.as_bytes());
}

fn write_message_field(buf: &mut Vec<u8>, field: u32, value: &[u8]) {
    write_key(buf, field, 2);
    write_varint(buf, value.len() as u64);
    buf.extend_from_slice(value);
}

fn write_varint_field(buf: &mut Vec<u8>, field: u32, value: u64) {
    write_key(buf, field, 0);
    write_varint(buf, value);
}

fn write_key(buf: &mut Vec<u8>, field: u32, wire_type: u8) {
    write_varint(buf, ((field as u64) << 3) | u64::from(wire_type));
}

fn write_varint(buf: &mut Vec<u8>, mut value: u64) {
    while value >= 0x80 {
        buf.push((value as u8) | 0x80);
        value >>= 7;
    }
    buf.push(value as u8);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cuda_device_with_cc(cc: Option<&str>) -> Device {
        Device {
            id: 0,
            name: "NVIDIA Test GPU".to_string(),
            backend: DeviceBackend::Cuda,
            memory_mb: 16 * 1024,
            compute_units: 0,
            pci_bus_id: None,
            partition_id: None,
            driver_version: Some("580.126.09".to_string()),
            cuda_version: Some("13.0".to_string()),
            compute_capability: cc.map(|value| value.to_string()),
            utilization_gpu_pct: None,
            temperature_c: None,
            supports_fp16: true,
            supports_int8: true,
        }
    }

    #[test]
    fn accepts_cuda_device_with_known_compute_capability() {
        let device = cuda_device_with_cc(Some("8.9"));

        assert!(
            validate_cuda_static_compatibility_with(&device, DEFAULT_CUDA_MAX_CC, false).is_ok()
        );
    }

    #[test]
    fn rejects_cuda_device_newer_than_supported_bundle() {
        let device = cuda_device_with_cc(Some("12.0"));

        let err = validate_cuda_static_compatibility_with(&device, DEFAULT_CUDA_MAX_CC, false)
            .unwrap_err();

        assert!(err.contains("compute capability 12.0"));
        assert!(err.contains(ALLOW_UNSUPPORTED_CUDA_ENV));
    }

    #[test]
    fn allows_new_cuda_device_when_override_is_set() {
        let device = cuda_device_with_cc(Some("12.0"));

        let result = validate_cuda_static_compatibility_with(&device, DEFAULT_CUDA_MAX_CC, true);

        assert!(result.is_ok());
    }

    #[test]
    fn identity_smoke_model_is_nonempty_onnx_model() {
        let model = identity_smoke_model();

        assert!(model.len() > 64);
        assert!(model.windows("Identity".len()).any(|w| w == b"Identity"));
    }

    #[test]
    fn identity_smoke_model_loads_and_runs_on_cpu() {
        let mut session = Session::builder()
            .expect("session builder")
            .with_optimization_level(GraphOptimizationLevel::Disable)
            .expect("optimization level")
            .commit_from_memory(&identity_smoke_model())
            .expect("commit smoke model");
        let input = Value::from_array(([1_usize], vec![3.0_f32])).expect("input tensor");

        session.run(ort::inputs![input]).expect("smoke inference");
    }
}
