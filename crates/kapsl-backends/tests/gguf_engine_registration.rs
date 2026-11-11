//! The engine calls the `create_gguf_*` constructors directly, which take no
//! device list. Registering the inventory first is what lets GGUF placement map an
//! engine CUDA device to a llama.cpp one by PCI bus id (own test binary: the
//! registry is process-global).

use kapsl_backends::BackendFactory;
use kapsl_hal::device::{Device, DeviceBackend, DeviceInfo};
use kapsl_llm::gguf_placement::{
    engine_devices, select_cuda_llama_device, LlamaDeviceInfo, LlamaDeviceKind,
};

fn device(id: usize, backend: DeviceBackend, pci: Option<&str>) -> Device {
    Device {
        id,
        name: format!("fake-{id}"),
        backend,
        memory_mb: 24 * 1024,
        compute_units: 1,
        pci_bus_id: pci.map(str::to_string),
        partition_id: None,
        driver_version: None,
        cuda_version: None,
        compute_capability: None,
        utilization_gpu_pct: None,
        temperature_c: None,
        supports_fp16: true,
        supports_int8: true,
    }
}

fn llama_cuda(name: &str, pci: &str) -> LlamaDeviceInfo {
    LlamaDeviceInfo {
        name: name.into(),
        backend: "CUDA".into(),
        kind: LlamaDeviceKind::Gpu,
        pci_bus_id: Some(pci.into()),
        unified_memory: false,
    }
}

#[test]
fn registering_the_inventory_lets_two_gpu_hosts_select_by_pci() {
    let info = DeviceInfo {
        cpu_cores: 8,
        total_memory: 64 << 30,
        os_type: "linux".into(),
        os_release: "test".into(),
        has_cuda: true,
        has_metal: false,
        has_rocm: false,
        has_directml: false,
        devices: vec![
            device(0, DeviceBackend::Cpu, None),
            device(0, DeviceBackend::Cuda, Some("00000000:02:00.0")),
            device(1, DeviceBackend::Cuda, Some("00000000:03:00.0")),
        ],
    };
    let llama = [
        llama_cuda("CUDA0", "0000:02:00.0"),
        llama_cuda("CUDA1", "0000:03:00.0"),
    ];

    // Before registration: the reported failure.
    assert!(select_cuda_llama_device(&llama, &engine_devices(), 0).is_err());

    BackendFactory::register_gguf_engine_devices(&info);
    let registered = engine_devices();
    assert_eq!(registered.len(), 2, "the CPU device is not an accelerator");
    assert!(registered.iter().all(|d| d.pci_bus_id.is_some()));
    assert_eq!(select_cuda_llama_device(&llama, &registered, 0), Ok(0));
    assert_eq!(select_cuda_llama_device(&llama, &registered, 1), Ok(1));
}
