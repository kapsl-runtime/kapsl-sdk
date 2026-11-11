//! Placement rules over fake device lists (no llama.cpp, no hardware).

use super::*;

const PCI_A: &str = "0000:01:00.0";
const PCI_B: &str = "0000:41:00.0";
const NVML_A: &str = "00000000:01:00.0";
const NVML_B: &str = "00000000:41:00.0";

fn cpu() -> LlamaDeviceInfo {
    LlamaDeviceInfo {
        name: "CPU".into(),
        backend: "CPU".into(),
        kind: LlamaDeviceKind::Cpu,
        pci_bus_id: None,
        unified_memory: true,
    }
}

fn gpu(name: &str, backend: &str, pci: Option<&str>) -> LlamaDeviceInfo {
    LlamaDeviceInfo {
        name: name.into(),
        backend: backend.into(),
        kind: LlamaDeviceKind::Gpu,
        pci_bus_id: pci.map(str::to_string),
        unified_memory: false,
    }
}

fn cuda(name: &str, pci: &str) -> LlamaDeviceInfo {
    gpu(name, "CUDA", Some(pci))
}

fn engine_cuda(id: usize, pci: &str) -> EngineDeviceInfo {
    EngineDeviceInfo {
        id,
        backend: "cuda".into(),
        pci_bus_id: Some(pci.into()),
    }
}

fn bytes(weights: u64, kv: u64, compute: u64) -> DeviceBytes {
    DeviceBytes {
        weights,
        kv,
        compute,
    }
}

fn row(class: MemoryAllocationClass, domain: MemoryDomain, bytes: usize) -> PlacedRow {
    PlacedRow {
        class,
        domain,
        bytes,
    }
}

fn cuda_domain(device_id: usize) -> MemoryDomain {
    MemoryDomain::Cuda { device_id }
}

fn provider(name: &str, id: usize) -> MemoryDomain {
    MemoryDomain::Provider {
        provider: name.into(),
        device_id: Some(id),
    }
}

use MemoryAllocationClass::{KvCache, PersistentWeights, TransientWorkspace};

// ---- identity ----

#[test]
fn pci_bus_ids_normalize_across_cuda_and_nvml_forms() {
    assert_eq!(
        normalize_pci_bus_id("0000:01:00.0").as_deref(),
        Some("00000000:01:00.0")
    );
    assert_eq!(
        normalize_pci_bus_id("00000000:01:00.0").as_deref(),
        Some("00000000:01:00.0")
    );
    assert_eq!(
        normalize_pci_bus_id("0000:AF:00.0\0").as_deref(),
        Some("00000000:af:00.0")
    );
    assert_eq!(
        normalize_pci_bus_id("01:00.0").as_deref(),
        Some("00000000:01:00.0")
    );
    assert_eq!(
        normalize_pci_bus_id("0001:0a:00.1").as_deref(),
        Some("00000001:0a:00.1")
    );
    assert_eq!(normalize_pci_bus_id(""), None);
    assert_eq!(normalize_pci_bus_id("not-a-bus-id"), None);
    assert_eq!(normalize_pci_bus_id("0000:01:00"), None);
}

// ---- NVIDIA ----

#[test]
fn single_cuda_gpu_maps_by_pci_even_with_the_nvml_spelling() {
    let all = [cuda("CUDA0", PCI_A), cpu()];
    let engine = [engine_cuda(0, NVML_A)];
    assert_eq!(
        place_device(0, &all, &engine, Some(0)),
        Placement::Domain(cuda_domain(0))
    );
}

#[test]
fn two_identical_gpus_each_get_their_own_cuda_domain() {
    let all = [cuda("CUDA0", PCI_A), cuda("CUDA1", PCI_B), cpu()];
    let engine = [engine_cuda(0, NVML_A), engine_cuda(1, NVML_B)];
    let report = place_bytes(
        &all,
        &engine,
        Some(0),
        &[(0, bytes(100, 10, 5)), (1, bytes(100, 10, 5))],
        (0, 0, 0),
    );
    assert!(report.warnings.is_empty());
    for id in [0, 1] {
        assert!(report
            .rows
            .contains(&row(PersistentWeights, cuda_domain(id), 100)));
        assert!(report.rows.contains(&row(KvCache, cuda_domain(id), 10)));
        assert!(report
            .rows
            .contains(&row(TransientWorkspace, cuda_domain(id), 5)));
    }
    assert_eq!(report.rows.len(), 6, "one row per device and class");
}

#[test]
fn different_gpus_where_cuda_and_nvml_order_differ_follow_the_pci_id() {
    // FASTEST_FIRST puts the faster card (PCI B) first: CUDA0 is NVML index 1.
    let all = [cuda("CUDA0", PCI_B), cuda("CUDA1", PCI_A)];
    let engine = [engine_cuda(0, NVML_A), engine_cuda(1, NVML_B)];
    assert_eq!(
        place_device(0, &all, &engine, Some(0)),
        Placement::Domain(cuda_domain(1))
    );
    assert_eq!(
        place_device(1, &all, &engine, Some(0)),
        Placement::Domain(cuda_domain(0))
    );
    // Loading "on engine device 0" must pick llama's CUDA1, not CUDA0.
    assert_eq!(select_cuda_llama_device(&all, &engine, 0), Ok(1));
    assert_eq!(select_cuda_llama_device(&all, &engine, 1), Ok(0));
}

#[test]
fn cuda_visible_devices_by_index_remaps_the_ordinal_but_not_the_domain() {
    // CUDA_VISIBLE_DEVICES=1: the engine still sees both GPUs, llama.cpp sees the
    // second one as CUDA0.
    let all = [cuda("CUDA0", PCI_B)];
    let engine = [engine_cuda(0, NVML_A), engine_cuda(1, NVML_B)];
    assert_eq!(
        place_device(0, &all, &engine, Some(1)),
        Placement::Domain(cuda_domain(1))
    );
    assert_eq!(select_cuda_llama_device(&all, &engine, 1), Ok(0));
    let hidden = select_cuda_llama_device(&all, &engine, 0).unwrap_err();
    assert!(hidden.contains("not visible to llama.cpp"), "{hidden}");
}

#[test]
fn cuda_visible_devices_by_uuid_gives_the_same_llama_view_and_the_same_mapping() {
    // CUDA_VISIBLE_DEVICES=GPU-<uuid of the second card>: ggml never reports a
    // UUID, only the PCI bus id, so the llama list is identical to the by-index
    // case and the mapping must be too. Both orders of the visible pair work.
    let all = [cuda("CUDA0", PCI_B), cuda("CUDA1", PCI_A)];
    let engine = [engine_cuda(0, NVML_A), engine_cuda(1, NVML_B)];
    let ids: Vec<_> = (0..2)
        .map(|i| place_device(i, &all, &engine, None).domain())
        .collect();
    assert_eq!(ids, vec![cuda_domain(1), cuda_domain(0)]);
}

#[test]
fn engine_devices_that_share_a_pci_id_are_refused_not_guessed() {
    // The MIG case: the engine lists two instances with one bus id (NVIDIA's docs
    // do not say whether MIG instances report their parent's PCI id, so this
    // assumes it); llama.cpp can see only one of them. Whatever the cause, two
    // engine devices with the same PCI id must never be told apart by guessing.
    let all = [cuda("CUDA0", PCI_A)];
    let engine = [engine_cuda(0, NVML_A), engine_cuda(1, NVML_A)];
    let placement = place_device(0, &all, &engine, Some(0));
    match &placement {
        Placement::Unattributed(reason) => {
            assert!(reason.contains("cannot tell them apart"), "{reason}")
        }
        other => panic!("expected unattributed, got {other:?}"),
    }
    let report = place_bytes(&all, &engine, Some(0), &[(0, bytes(100, 10, 5))], (0, 0, 0));
    assert_eq!(report.warnings.len(), 1);
    assert!(report
        .rows
        .iter()
        .all(|r| r.domain == unattributed_domain()));
    assert!(
        select_cuda_llama_device(&all, &engine, 0).is_ok(),
        "selection by PCI is ambiguous only for placement, one llama device exists"
    );
    // A single MIG instance engine-side is unambiguous.
    let single = [engine_cuda(3, NVML_A)];
    assert_eq!(
        place_device(0, &all, &single, None),
        Placement::Domain(cuda_domain(3))
    );
}

#[test]
fn cuda_without_pci_ids_is_mapped_only_when_unambiguous() {
    let one = [gpu("CUDA0", "CUDA", None)];
    let no_engine: [EngineDeviceInfo; 0] = [];
    assert_eq!(
        place_device(0, &one, &no_engine, Some(2)),
        Placement::Domain(cuda_domain(2))
    );

    let two = [gpu("CUDA0", "CUDA", None), gpu("CUDA1", "CUDA", None)];
    assert!(matches!(
        place_device(0, &two, &no_engine, Some(0)),
        Placement::Unattributed(_)
    ));
    assert!(select_cuda_llama_device(&two, &no_engine, 0).is_err());
}

#[test]
fn selecting_an_engine_device_the_engine_does_not_have_is_an_error() {
    // One visible llama CUDA device, no PCI ids: only the engine's own device
    // maps to it. Asking for device 1 must not silently load on CUDA0.
    let one = [gpu("CUDA0", "CUDA", None)];
    let engine = [engine_cuda(0, NVML_A)];
    assert!(select_cuda_llama_device(&one, &[], 1).is_err());
    assert_eq!(select_cuda_llama_device(&one, &[], 0), Ok(0));
    let engine_no_pci = [EngineDeviceInfo {
        pci_bus_id: None,
        ..engine[0].clone()
    }];
    assert!(select_cuda_llama_device(&one, &engine_no_pci, 1).is_err());
    assert_eq!(select_cuda_llama_device(&one, &engine_no_pci, 0), Ok(0));
}

#[test]
fn cuda_device_missing_from_the_engine_list_is_unattributed() {
    let all = [cuda("CUDA0", PCI_B)];
    let engine = [engine_cuda(0, NVML_A)];
    match place_device(0, &all, &engine, Some(0)) {
        Placement::Unattributed(reason) => {
            assert!(reason.contains("matches no engine CUDA device"))
        }
        other => panic!("{other:?}"),
    }
}

// ---- other vendors and unified memory ----

#[test]
fn rocm_and_hip_devices_get_accounting_only_rocm_provider_domains() {
    let all = [
        gpu("ROCm0", "ROCm", Some(PCI_A)),
        gpu("ROCm1", "ROCm", Some(PCI_B)),
        cpu(),
    ];
    assert_eq!(
        place_device(0, &all, &[], None),
        Placement::Domain(provider("rocm", 0))
    );
    assert_eq!(
        place_device(1, &all, &[], None),
        Placement::Domain(provider("rocm", 1))
    );
    let hip = [gpu("HIP0", "HIP", None)];
    assert_eq!(
        place_device(0, &hip, &[], None),
        Placement::Domain(provider("rocm", 0))
    );
}

#[test]
fn vulkan_discrete_is_a_provider_domain_and_ordinals_count_per_backend() {
    let all = [
        cuda("CUDA0", PCI_A),
        gpu("Vulkan1", "Vulkan", Some(PCI_B)),
        gpu("Vulkan2", "Vulkan", Some("0000:81:00.0")),
    ];
    let engine = [engine_cuda(0, NVML_A)];
    assert_eq!(
        place_device(1, &all, &engine, None),
        Placement::Domain(provider("vulkan", 0))
    );
    assert_eq!(
        place_device(2, &all, &engine, None),
        Placement::Domain(provider("vulkan", 1))
    );
}

#[test]
fn vulkan_integrated_gpu_counts_in_host() {
    let igpu = LlamaDeviceInfo {
        kind: LlamaDeviceKind::IntegratedGpu,
        unified_memory: true,
        ..gpu("Vulkan0", "Vulkan", Some(PCI_A))
    };
    assert_eq!(
        place_device(0, &[igpu], &[], None),
        Placement::Domain(MemoryDomain::Host)
    );
}

#[test]
fn apple_metal_unified_memory_counts_in_host_and_merges_with_cpu_rows() {
    let metal = LlamaDeviceInfo {
        unified_memory: true,
        ..gpu("MTL0", "MTL", None)
    };
    let all = [metal, cpu()];
    let report = place_bytes(&all, &[], None, &[(0, bytes(0, 400, 23))], (468, 0, 58));
    assert!(report.warnings.is_empty());
    assert!(report.rows.iter().all(|r| r.domain == MemoryDomain::Host));
    assert!(report
        .rows
        .contains(&row(PersistentWeights, MemoryDomain::Host, 468)));
    assert!(report.rows.contains(&row(KvCache, MemoryDomain::Host, 400)));
    assert!(report
        .rows
        .contains(&row(TransientWorkspace, MemoryDomain::Host, 23 + 58)));
}

#[test]
fn nvidia_integrated_device_reporting_unified_memory_counts_in_host() {
    // Jetson-class: ggml reports type GPU but the driver says integrated.
    let jetson = LlamaDeviceInfo {
        unified_memory: true,
        ..cuda("CUDA0", PCI_A)
    };
    assert_eq!(
        place_device(0, &[jetson], &[engine_cuda(0, NVML_A)], Some(0)),
        Placement::Domain(MemoryDomain::Host)
    );
}

#[test]
fn cpu_only_counts_everything_in_host() {
    let report = place_bytes(&[cpu()], &[], None, &[(0, bytes(218, 100, 318))], (0, 0, 0));
    assert!(report.rows.iter().all(|r| r.domain == MemoryDomain::Host));
    assert_eq!(report.rows.len(), 3);
}

#[test]
fn unknown_devices_are_unattributed_with_a_warning_never_cuda0() {
    let other = LlamaDeviceInfo {
        kind: LlamaDeviceKind::Other,
        ..gpu("META0", "META", None)
    };
    let report = place_bytes(
        &[other],
        &[engine_cuda(0, NVML_A)],
        Some(0),
        &[(0, bytes(10, 20, 30))],
        (0, 0, 0),
    );
    assert_eq!(report.warnings.len(), 1);
    assert!(report.warnings[0].contains("META0"));
    assert_eq!(report.rows.len(), 3);
    assert!(report
        .rows
        .iter()
        .all(|r| r.domain == unattributed_domain()));
    assert_eq!(domain_label(&unattributed_domain()), "unattributed");
    assert_eq!(domain_label(&provider("rocm", 1)), "rocm:1");
    assert_eq!(domain_label(&cuda_domain(3)), "cuda:3");
}

// ---- partial offload and multi-device ----

#[test]
fn reported_kv_follows_the_layers_between_gpu_and_host() {
    // 8 of 32 layers offloaded: llama.cpp keeps the other layers' KV on the CPU
    // buffer type, which the fork reports as host-resident bytes.
    let all = [cuda("CUDA0", PCI_A), cpu()];
    let engine = [engine_cuda(0, NVML_A)];
    let report = place_bytes(
        &all,
        &engine,
        Some(0),
        &[(0, bytes(120, 64, 40))],
        (360, 192, 10),
    );
    assert!(report.rows.contains(&row(KvCache, cuda_domain(0), 64)));
    assert!(report.rows.contains(&row(KvCache, MemoryDomain::Host, 192)));
    assert!(report
        .rows
        .contains(&row(PersistentWeights, cuda_domain(0), 120)));
    assert!(report
        .rows
        .contains(&row(PersistentWeights, MemoryDomain::Host, 360)));
}

#[test]
fn layer_split_across_two_gpus_reports_one_row_per_device() {
    let all = [cuda("CUDA0", PCI_A), cuda("CUDA1", PCI_B)];
    let engine = [engine_cuda(0, NVML_A), engine_cuda(1, NVML_B)];
    let report = place_bytes(
        &all,
        &engine,
        None,
        &[(0, bytes(300, 30, 8)), (1, bytes(200, 20, 8))],
        (0, 0, 0),
    );
    assert!(report
        .rows
        .contains(&row(PersistentWeights, cuda_domain(0), 300)));
    assert!(report
        .rows
        .contains(&row(PersistentWeights, cuda_domain(1), 200)));
    assert!(report.rows.contains(&row(KvCache, cuda_domain(0), 30)));
    assert!(report.rows.contains(&row(KvCache, cuda_domain(1), 20)));
}

#[test]
fn plan_splits_weights_and_kv_by_offloaded_fraction_into_the_same_domains() {
    let all = [cuda("CUDA0", PCI_A), cpu()];
    let engine = [engine_cuda(0, NVML_A)];
    let input = PlanInput {
        weights: 3200,
        kv: 640,
        scratch: 400,
        n_layers: 32,
        n_gpu_layers: Some(8),
    };
    let plan = plan_rows(Some(0), &all, &engine, Some(0), input);
    assert!(plan
        .rows
        .contains(&row(PersistentWeights, cuda_domain(0), 800)));
    assert!(plan
        .rows
        .contains(&row(PersistentWeights, MemoryDomain::Host, 2400)));
    assert!(plan.rows.contains(&row(KvCache, cuda_domain(0), 160)));
    assert!(plan.rows.contains(&row(KvCache, MemoryDomain::Host, 480)));
    // Scratch: 100 on the device; the host's 300 carries 25% CPU headroom.
    assert!(plan
        .rows
        .contains(&row(TransientWorkspace, cuda_domain(0), 100)));
    assert!(plan
        .rows
        .contains(&row(TransientWorkspace, MemoryDomain::Host, 375)));

    // Full offload (default) puts everything on the device.
    let full = plan_rows(
        Some(0),
        &all,
        &engine,
        Some(0),
        PlanInput {
            n_gpu_layers: None,
            ..input
        },
    );
    assert!(full.rows.iter().all(|r| r.domain == cuda_domain(0)));
    // No GPU, or n_gpu_layers=0 resolved to no primary: everything in Host.
    let cpu_plan = plan_rows(None, &[cpu()], &[], None, input);
    assert!(cpu_plan.rows.iter().all(|r| r.domain == MemoryDomain::Host));
    assert_eq!(
        cpu_plan.rows.iter().map(|r| r.bytes).sum::<usize>(),
        3200 + 640 + 500 // scratch carries 25% CPU headroom
    );
}

#[test]
fn plan_on_a_unified_device_lands_in_host_like_the_report() {
    let metal = LlamaDeviceInfo {
        unified_memory: true,
        ..gpu("MTL0", "MTL", None)
    };
    let all = [metal, cpu()];
    let plan = plan_rows(
        Some(0),
        &all,
        &[],
        None,
        PlanInput {
            weights: 100,
            kv: 50,
            scratch: 10,
            n_layers: 24,
            n_gpu_layers: None,
        },
    );
    assert!(plan.rows.iter().all(|r| r.domain == MemoryDomain::Host));
    assert_eq!(plan.rows.iter().map(|r| r.bytes).sum::<usize>(), 160);
}

// ---- allocation ids ----

#[test]
fn a_class_split_across_domains_gets_distinct_ids_with_one_base_id() {
    let rows = vec![
        row(KvCache, MemoryDomain::Host, 192),
        row(KvCache, cuda_domain(0), 64),
        row(PersistentWeights, cuda_domain(0), 120),
    ];
    // Domains sort by name: "cuda:0" < "host", so the GPU row keeps the base id.
    assert_eq!(allocation_id_for("kv", &rows, &rows[1]), "kv");
    assert_eq!(allocation_id_for("kv", &rows, &rows[0]), "kv@host");
    // A class with a single domain always keeps its base id.
    assert_eq!(allocation_id_for("w", &rows, &rows[2]), "w");
}

#[test]
fn selection_matches_nvml_and_cuda_spellings_in_any_case() {
    // ggml/CUDA: 4-digit domain, upper case; NVML: 8-digit domain, lower case.
    let all = [cuda("CUDA0", "0000:0A:00.0"), cuda("CUDA1", "0000:0B:00.0")];
    let engine = [
        engine_cuda(0, "00000000:0a:00.0"),
        engine_cuda(1, "00000000:0b:00.0"),
    ];
    assert_eq!(select_cuda_llama_device(&all, &engine, 0), Ok(0));
    assert_eq!(select_cuda_llama_device(&all, &engine, 1), Ok(1));
    // Same inventory in the other order (fastest-first): PCI wins over position.
    let swapped = [all[1].clone(), all[0].clone()];
    assert_eq!(select_cuda_llama_device(&swapped, &engine, 0), Ok(1));
}

#[test]
fn selection_with_a_real_pci_mismatch_still_fails() {
    let all = [cuda("CUDA0", "0000:0a:00.0"), cuda("CUDA1", "0000:0b:00.0")];
    let engine = [engine_cuda(0, "00000000:0c:00.0")];
    let error = select_cuda_llama_device(&all, &engine, 0).unwrap_err();
    assert!(error.contains("not visible to llama.cpp"), "{error}");
}

#[test]
fn selection_without_a_registered_inventory_says_so() {
    // The reported failure: two llama CUDA devices and an empty engine list.
    let all = [cuda("CUDA0", "0000:02:00.0"), cuda("CUDA1", "0000:03:00.0")];
    let error = select_cuda_llama_device(&all, &[], 0).unwrap_err();
    assert!(error.contains("never registered"), "{error}");
    // Registered, with a device that has no PCI id, is a different message.
    let no_pci = [
        EngineDeviceInfo {
            pci_bus_id: None,
            ..engine_cuda(0, NVML_A)
        },
        engine_cuda(1, NVML_B),
    ];
    let error = select_cuda_llama_device(&all, &no_pci, 0).unwrap_err();
    assert!(error.contains("no usable PCI bus id"), "{error}");
}
