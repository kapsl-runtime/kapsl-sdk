//! CUDA ordinal <-> engine id mapping over fake inventories (no GPU needed).

use super::*;

const A: &str = "00000000:01:00.0"; // NVML spelling
const B: &str = "00000000:41:00.0";
const C: &str = "00000000:81:00.0";

fn dev(id: usize, pci: Option<&str>) -> EngineCudaDevice {
    EngineCudaDevice {
        id,
        pci_bus_id: pci.map(str::to_string),
    }
}

fn pcis(list: &[&str]) -> Vec<Option<String>> {
    // The CUDA runtime spells the same ids with a 4-digit domain.
    list.iter()
        .map(|p| Some(p.replacen("00000000:", "0000:", 1)))
        .collect()
}

fn two_gpus() -> Vec<EngineCudaDevice> {
    vec![dev(0, Some(A)), dev(1, Some(B))]
}

#[test]
fn single_gpu_maps_to_ordinal_zero_exactly_as_before() {
    let map = CudaOrdinalMap::build(&[dev(0, Some(A))], &pcis(&[A]));
    assert_eq!(map.ordinal_for(0), Some(0));
    assert!(map.is_identity());
}

#[test]
fn container_with_one_visible_gpu_and_no_pci_ids_is_the_identity() {
    // Cloud Run / Kubernetes style: one GPU, neither side reports a PCI id.
    let map = CudaOrdinalMap::build(&[dev(0, None)], &[None]);
    assert_eq!(map.ordinal_for(0), Some(0));
    assert!(map.is_identity());
    // One GPU whose id is not 0 in the inventory still opens ordinal 0.
    let map = CudaOrdinalMap::build(&[dev(3, Some(A))], &[None]);
    assert_eq!(map.ordinal_for(3), Some(0));
    // And with PCI ids known only on one side.
    let map = CudaOrdinalMap::build(&[dev(0, None)], &pcis(&[A]));
    assert_eq!(map.ordinal_for(0), Some(0));
}

#[test]
fn two_gpus_in_the_same_order_are_the_identity() {
    let map = CudaOrdinalMap::build(&two_gpus(), &pcis(&[A, B]));
    assert!(map.is_identity());
    assert_eq!((map.ordinal_for(0), map.ordinal_for(1)), (Some(0), Some(1)));
}

#[test]
fn fastest_first_order_swaps_the_ordinals() {
    // CUDA put the card at bus 0x41 first.
    let map = CudaOrdinalMap::build(&two_gpus(), &pcis(&[B, A]));
    assert_eq!(map.ordinal_for(0), Some(1));
    assert_eq!(map.ordinal_for(1), Some(0));
    assert_eq!(map.engine_id_for(0), Some(1));
    assert_eq!(map.engine_id_for(1), Some(0));
    assert!(!map.is_identity());
}

#[test]
fn cuda_visible_devices_1_hides_device_0_and_renumbers_device_1() {
    // CUDA_VISIBLE_DEVICES=1: the process enumerates only the second card.
    let map = CudaOrdinalMap::build(&two_gpus(), &pcis(&[B]));
    assert_eq!(map.status(0), Some(&CudaMatch::Hidden));
    assert_eq!(map.ordinal_for(1), Some(0));
    assert_eq!(map.engine_id_for(0), Some(1));
}

#[test]
fn cuda_visible_devices_1_0_reorders_both_devices() {
    let map = CudaOrdinalMap::build(&two_gpus(), &pcis(&[B, A]));
    assert_eq!((map.ordinal_for(0), map.ordinal_for(1)), (Some(1), Some(0)));
}

#[test]
fn a_uuid_form_mask_gives_the_same_mapping_because_identity_comes_from_pci() {
    // CUDA_VISIBLE_DEVICES=GPU-<uuid of the second card>,GPU-<uuid of the first>:
    // the driver exposes the same two devices in that order, and the map is
    // built from what the driver enumerates (PCI ids), not from the mask text.
    let by_uuid = CudaOrdinalMap::build(&two_gpus(), &pcis(&[B, A]));
    let by_index = CudaOrdinalMap::build(&two_gpus(), &pcis(&[B, A]));
    assert_eq!(by_uuid.describe(), by_index.describe());
    assert_eq!(by_uuid.ordinal_for(0), Some(1));
    // A single UUID selecting the first card keeps it at ordinal 0 and hides the other.
    let one = CudaOrdinalMap::build(&two_gpus(), &pcis(&[A]));
    assert_eq!(one.ordinal_for(0), Some(0));
    assert_eq!(one.status(1), Some(&CudaMatch::Hidden));
}

#[test]
fn a_device_with_no_pci_match_is_hidden_not_guessed() {
    // The inventory lists a card the CUDA runtime never enumerates.
    let engine = vec![dev(0, Some(A)), dev(1, Some(C))];
    let map = CudaOrdinalMap::build(&engine, &pcis(&[A, B]));
    assert_eq!(map.ordinal_for(0), Some(0));
    assert_eq!(map.status(1), Some(&CudaMatch::Hidden));
    assert_eq!(map.ordinal_for(1), None);
}

#[test]
fn a_device_with_no_pci_id_among_several_is_unmatched() {
    let engine = vec![dev(0, Some(A)), dev(1, None)];
    let map = CudaOrdinalMap::build(&engine, &pcis(&[A, B]));
    assert_eq!(map.ordinal_for(0), Some(0));
    assert!(matches!(map.status(1), Some(CudaMatch::Unmatched(_))));
    // No PCI ids anywhere on the CUDA side with several GPUs: nothing can be matched.
    let map = CudaOrdinalMap::build(&two_gpus(), &[None, None, None]);
    assert!(matches!(map.status(0), Some(CudaMatch::Unmatched(_))));
}

#[test]
fn legacy_hosts_with_no_pci_ids_and_equal_counts_keep_the_identity() {
    let engine = vec![dev(0, None), dev(1, None)];
    let map = CudaOrdinalMap::build(&engine, &[None, None]);
    assert_eq!((map.ordinal_for(0), map.ordinal_for(1)), (Some(0), Some(1)));
}

#[test]
fn engine_devices_sharing_a_pci_id_are_unmatched() {
    // MIG-style: two engine devices with one bus id.
    let engine = vec![dev(0, Some(A)), dev(1, Some(A))];
    let map = CudaOrdinalMap::build(&engine, &pcis(&[A]));
    for id in [0, 1] {
        match map.status(id) {
            Some(CudaMatch::Unmatched(reason)) => {
                assert!(reason.contains("cannot tell them apart"), "{reason}")
            }
            other => panic!("{other:?}"),
        }
    }
}

#[test]
fn describe_lists_every_device_and_the_runtime_enumeration() {
    let map = CudaOrdinalMap::build(&two_gpus(), &pcis(&[B]));
    let lines = map.describe();
    assert_eq!(lines.len(), 3);
    assert!(lines[0].contains("engine CUDA device 0") && lines[0].contains("not visible"));
    assert!(lines[1].contains("-> CUDA ordinal 0"));
    assert!(lines[2].starts_with("CUDA runtime enumerates 1 device(s)"));
}

#[test]
fn conversion_is_the_identity_until_a_map_is_installed_then_follows_it() {
    // The only test touching the process-wide map: install, check, reset.
    assert_eq!(to_cuda_ordinal(5), 5);
    let map = CudaOrdinalMap::build(&two_gpus(), &pcis(&[B, A]));
    install(map);
    assert_eq!(to_cuda_ordinal(0), 1);
    assert_eq!(to_cuda_ordinal(1), 0);
    assert_eq!(to_cuda_ordinal_i32(0), 1);
    assert_eq!(to_cuda_ordinal(9), 9, "unknown ids pass through");
    *GLOBAL.write().unwrap() = None;
    assert_eq!(to_cuda_ordinal(0), 0);
}

#[test]
fn pci_normalization_accepts_both_spellings() {
    assert_eq!(
        normalize_pci("0000:01:00.0"),
        normalize_pci("00000000:01:00.0")
    );
    assert_eq!(
        normalize_pci("0000:AF:00.0\0").as_deref(),
        Some("00000000:af:00.0")
    );
    assert_eq!(normalize_pci("garbage"), None);
}
