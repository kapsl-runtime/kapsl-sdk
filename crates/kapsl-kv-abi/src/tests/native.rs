use super::*;

fn geometry() -> KapslKvGeometryV1 {
    KapslKvGeometryV1 {
        struct_size: std::mem::size_of::<KapslKvGeometryV1>() as u32,
        device_id: 0,
        requested_blocks: 128,
        block_size_tokens: 16,
        num_layers: 4,
        num_kv_heads: 2,
        key_head_dim: 64,
        value_head_dim: 64,
        element_bytes: 2,
        max_sequences: 4,
        max_blocks_per_sequence: 8,
        model_fingerprint: 1,
        flags: 0,
        reserved: 0,
    }
}

unsafe extern "C" fn create(
    _data: *mut c_void,
    _geometry: *const KapslKvGeometryV1,
    _out: *mut KapslKvPoolV1,
) -> i32 {
    0
}
unsafe extern "C" fn reserve(
    _data: *mut c_void,
    _id: u64,
    _request: *const KapslKvReserveRequestV1,
    _out: *mut KapslKvReservationV1,
) -> i32 {
    0
}
unsafe extern "C" fn commit(_data: *mut c_void, _id: u64, _out: *mut *mut u32) -> i32 {
    0
}
unsafe extern "C" fn sequence(_data: *mut c_void, _id: u64, _sequence: u64) -> i32 {
    0
}
unsafe extern "C" fn bytes(_data: *mut c_void, _id: u64, _out: *mut u64) -> i32 {
    0
}
unsafe extern "C" fn destroy(_data: *mut c_void, _id: u64) -> i32 {
    0
}

fn table() -> KapslNativeKvHostV1 {
    KapslNativeKvHostV1 {
        struct_size: std::mem::size_of::<KapslNativeKvHostV1>() as u32,
        abi_version: 1,
        device_id: 0,
        model_id: 7,
        replica_id: 2,
        reserved: 0,
        user_data: std::ptr::dangling_mut::<u8>().cast(),
        create_pool: Some(create),
        reserve: Some(reserve),
        commit: Some(commit),
        release: Some(sequence),
        touch: Some(sequence),
        pool_bytes: Some(bytes),
        destroy_pool: Some(destroy),
    }
}

#[test]
fn geometry_rejects_overflow_unknown_flags_and_empty_dimensions() {
    let original = geometry();
    assert!(original.is_well_formed());
    assert_eq!(original.storage_bytes(), Some(128 * 16 * 2 * 128 * 2));
    let mut candidate = original;
    candidate.requested_blocks = u64::MAX;
    assert!(!candidate.is_well_formed());
    let mut candidate = original;
    candidate.max_blocks_per_sequence = u32::MAX;
    assert!(!candidate.is_well_formed());
    let mut candidate = original;
    candidate.num_layers = 0;
    assert!(!candidate.is_well_formed());
    let mut candidate = original;
    candidate.flags = 1;
    assert!(!candidate.is_well_formed());
    let mut candidate = original;
    candidate.struct_size -= 1;
    assert!(!candidate.is_well_formed());
}

#[test]
fn request_ownership_is_required_even_for_sequence_zero() {
    let mut request = KapslKvReserveRequestV1 {
        struct_size: std::mem::size_of::<KapslKvReserveRequestV1>() as u32,
        flags: KAPSL_KV_RESERVE_UPLOAD,
        model_id: 7,
        replica_id: 2,
        request_id: 1,
        sequence_id: 0,
        sequence_slot: 0,
        tokens_needed: 16,
    };
    assert!(request.is_well_formed());
    request.request_id = 0;
    assert!(!request.is_well_formed());
    request.request_id = 1;
    request.flags = 2;
    assert!(!request.is_well_formed());
    request.flags = 0;
    request.tokens_needed = 0;
    assert!(!request.is_well_formed());
}

#[test]
fn host_requires_version_size_context_and_all_memory_callbacks() {
    let mut candidate = table();
    assert!(candidate.is_well_formed());
    assert!(unsafe {
        KapslNativeKvHostV1::from_ptr((&candidate as *const KapslNativeKvHostV1).cast())
    }
    .is_some());
    candidate.abi_version = 2;
    assert!(!candidate.is_well_formed());
    candidate = table();
    candidate.user_data = std::ptr::null_mut();
    assert!(!candidate.is_well_formed());
    candidate = table();
    candidate.create_pool = None;
    assert!(!candidate.is_well_formed());
    candidate = table();
    candidate.reserve = None;
    assert!(!candidate.is_well_formed());
    candidate = table();
    candidate.commit = None;
    assert!(!candidate.is_well_formed());
    candidate = table();
    candidate.release = None;
    assert!(!candidate.is_well_formed());
    candidate = table();
    candidate.touch = None;
    assert!(!candidate.is_well_formed());
    candidate = table();
    candidate.pool_bytes = None;
    assert!(!candidate.is_well_formed());
    candidate = table();
    candidate.destroy_pool = None;
    assert!(!candidate.is_well_formed());
    let only_size = 4u32;
    assert!(unsafe { KapslNativeKvHostV1::from_ptr((&only_size as *const u32).cast()) }.is_none());
    assert!(unsafe { KapslNativeKvHostV1::from_ptr(std::ptr::null()) }.is_none());
}

#[test]
#[cfg(target_pointer_width = "64")]
fn c_layouts_are_frozen() {
    assert_eq!(std::mem::size_of::<KapslKvGeometryV1>(), 64);
    assert_eq!(std::mem::size_of::<KapslKvPoolV1>(), 56);
    assert_eq!(std::mem::size_of::<KapslKvReserveRequestV1>(), 40);
    assert_eq!(std::mem::size_of::<KapslKvReservationV1>(), 16);
    assert_eq!(std::mem::size_of::<KapslNativeKvHostV1>(), 88);
    assert_eq!(std::mem::offset_of!(KapslNativeKvHostV1, create_pool), 32);
}
