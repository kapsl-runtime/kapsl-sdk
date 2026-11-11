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
    assert_eq!(to_cuda_ordinal(5), Ok(5));
    let map = CudaOrdinalMap::build(&two_gpus(), &pcis(&[B, A]));
    install(map);
    assert_eq!(to_cuda_ordinal(0), Ok(1));
    assert_eq!(to_cuda_ordinal(1), Ok(0));
    assert_eq!(to_cuda_ordinal_i32(0), Ok(1));
    assert_eq!(
        to_cuda_ordinal(9),
        Err(CudaOrdinalError::Unknown { engine_id: 9 }),
        "an id the map does not know is an error, never passed through"
    );

    // CUDA_VISIBLE_DEVICES=1: engine device 0 is hidden and must not become ordinal 0.
    install(CudaOrdinalMap::build(&two_gpus(), &pcis(&[B])));
    assert_eq!(to_cuda_ordinal(1), Ok(0));
    assert_eq!(
        to_cuda_ordinal(0),
        Err(CudaOrdinalError::Hidden { engine_id: 0 })
    );
    assert_eq!(
        to_cuda_ordinal_i32(0),
        Err(CudaOrdinalError::Hidden { engine_id: 0 })
    );
    assert_eq!(to_cuda_ordinal_i32(1), Ok(0));
    assert_eq!(
        to_cuda_ordinal_i32(-1),
        Err(CudaOrdinalError::InvalidId { engine_id: -1 })
    );

    // CUDA_VISIBLE_DEVICES=0: engine device 1 is hidden and must not become ordinal 1.
    install(CudaOrdinalMap::build(&two_gpus(), &pcis(&[A])));
    assert_eq!(to_cuda_ordinal(0), Ok(0));
    let error = to_cuda_ordinal(1).unwrap_err();
    assert_eq!(error, CudaOrdinalError::Hidden { engine_id: 1 });
    assert!(error.to_string().contains("not visible"), "{error}");

    // An unmatched device is an error with its reason.
    install(CudaOrdinalMap::build(
        &[dev(0, Some(A)), dev(1, None)],
        &pcis(&[A, B]),
    ));
    assert!(matches!(
        to_cuda_ordinal(1),
        Err(CudaOrdinalError::Unmatched { engine_id: 1, .. })
    ));
    *GLOBAL.write().unwrap() = None;
    assert_eq!(to_cuda_ordinal(0), Ok(0));
    assert!(to_cuda_ordinal_i32(-1).is_err());
}

#[test]
fn a_process_that_sees_no_cuda_device_has_every_engine_device_hidden() {
    // CUDA_VISIBLE_DEVICES="" enumerates nothing: that is a real "all hidden",
    // never an identity mapping.
    let map = CudaOrdinalMap::build(&two_gpus(), &[]);
    for id in [0, 1] {
        assert_eq!(map.status(id), Some(&CudaMatch::Hidden));
        assert_eq!(
            map.ordinal_or_error(id),
            Err(CudaOrdinalError::Hidden { engine_id: id })
        );
    }
    // Same without PCI ids on the engine side: nothing is guessed either.
    let map = CudaOrdinalMap::build(&[dev(0, None), dev(1, None)], &[]);
    for id in [0, 1] {
        assert!(map.ordinal_for(id).is_none());
        assert!(map.ordinal_or_error(id).is_err());
    }
}

#[test]
fn ordinal_or_error_names_the_failure_for_every_status() {
    let map = CudaOrdinalMap::build(
        &[dev(0, Some(A)), dev(1, Some(B)), dev(2, None)],
        &pcis(&[B, C]),
    );
    assert_eq!(map.ordinal_or_error(1), Ok(0));
    assert_eq!(
        map.ordinal_or_error(0),
        Err(CudaOrdinalError::Hidden { engine_id: 0 })
    );
    assert!(matches!(
        map.ordinal_or_error(2),
        Err(CudaOrdinalError::Unmatched { engine_id: 2, .. })
    ));
    assert_eq!(
        map.ordinal_or_error(7),
        Err(CudaOrdinalError::Unknown { engine_id: 7 })
    );
}

#[test]
fn pci_normalization_compares_values_not_spellings() {
    let canonical = Some("00000002:0a:00.0".to_string());
    // NVML 8-digit domain, CUDA/ggml 4-digit domain, bare bus form with no domain
    // only for domain 0, any hex case, surrounding NULs and spaces.
    for raw in [
        "00000002:0a:00.0",
        "0002:0A:00.0",
        "2:a:0.0",
        "0002:0a:00.0\0\0",
        " 0002:0A:00.0 ",
    ] {
        assert_eq!(normalize_pci(raw), canonical, "{raw:?}");
    }
    assert_eq!(
        normalize_pci("0000:0A:00.0"),
        normalize_pci("00000000:0a:00.0")
    );
    assert_eq!(normalize_pci("0a:00.0"), Some("00000000:0a:00.0".into()));
    assert_eq!(
        normalize_pci("000000000:02:00.0"),
        normalize_pci("0000:02:00.0"),
        "extra leading zeros are the same domain"
    );
    // Real differences stay different.
    assert_ne!(normalize_pci("0000:02:00.0"), normalize_pci("0000:03:00.0"));
    assert_ne!(normalize_pci("0000:02:00.0"), normalize_pci("0001:02:00.0"));
    assert_ne!(normalize_pci("0000:02:00.0"), normalize_pci("0000:02:00.1"));
    assert_ne!(normalize_pci("0000:02:00.0"), normalize_pci("0000:02:01.0"));
    // Malformed or out-of-range ids never normalize (so never match).
    for bad in [
        "",
        "garbage",
        "0000:02:00",
        "0000:02:00.8",
        "0000:100:00.0",
        "0000:02:20.0",
        "0000:0g:00.0",
        "0000:02:00.0:1",
        "-1:02:00.0",
        "+1:02:00.0",
        "1ffffffff:02:00.0",
    ] {
        assert_eq!(normalize_pci(bad), None, "{bad:?}");
    }
}

#[test]
fn the_map_matches_mixed_spellings_and_still_refuses_a_real_mismatch() {
    // Engine side NVML style upper case; CUDA side 4-digit lower case.
    let engine = vec![
        dev(0, Some("00000000:0A:00.0")),
        dev(1, Some("00000000:0B:00.0")),
    ];
    let map = CudaOrdinalMap::build(
        &engine,
        &[
            Some("0000:0b:00.0".to_string()),
            Some("0000:0a:00.0".to_string()),
        ],
    );
    assert_eq!(map.ordinal_or_error(0), Ok(1));
    assert_eq!(map.ordinal_or_error(1), Ok(0));
    // A CUDA device that is on another bus is not the engine device.
    let map = CudaOrdinalMap::build(
        &engine,
        &[
            Some("0000:0c:00.0".to_string()),
            Some("0000:0a:00.0".to_string()),
        ],
    );
    assert_eq!(map.ordinal_or_error(0), Ok(1));
    assert_eq!(map.status(1), Some(&CudaMatch::Hidden));
}
