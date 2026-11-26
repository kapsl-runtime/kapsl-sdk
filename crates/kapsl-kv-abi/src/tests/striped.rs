//! Striped elastic pools (ABI 1.6) and version negotiation with 1.5 peers.

use super::*;

const GRANULE: u64 = 65_536;
const STRIPES: u32 = 4;
/// Bytes of one block in one stripe; 64 blocks fill one granule per stripe.
const STRIPE_BYTES_PER_BLOCK: u64 = 1024;
const BYTES_PER_BLOCK: u64 = STRIPE_BYTES_PER_BLOCK * STRIPES as u64;

fn segment(
    segment_id: &str,
    stripe: u32,
    offset_bytes: u64,
    length_bytes: u64,
    handle_index: u32,
) -> KvVmmSegmentDescriptor {
    KvVmmSegmentDescriptor {
        segment_id: segment_id.to_string(),
        stripe,
        offset_bytes,
        length_bytes,
        handle_index,
    }
}

/// One segment per stripe covering `[start, end)` of that stripe.
fn stripe_segments(
    prefix: &str,
    stripes: u32,
    start: u64,
    end: u64,
) -> Vec<KvVmmSegmentDescriptor> {
    (0..stripes)
        .map(|stripe| {
            segment(
                &format!("{prefix}-{stripe}"),
                stripe,
                start,
                end - start,
                stripe,
            )
        })
        .collect()
}

fn striped_registration() -> KvParticipantRegistration {
    let mut registration: KvParticipantRegistration = serde_json::from_str(include_str!(
        "../../tests/fixtures/shared_pool_registration.json"
    ))
    .expect("shared registration fixture");
    registration.capabilities = KvBackendCapabilities::cuda_vmm_shared_pool()
        .with_feature(KvFeature::ParticipantBlockSelection)
        .with_feature(KvFeature::StripedElasticPool);
    registration
        .topology
        .as_mut()
        .expect("fixture topology")
        .abi_version = KAPSL_KV_ABI_VERSION;
    registration.capacity_model.groups[0].max_allocations = Some(256);
    registration.capacity_model.groups[0].bytes_per_allocation = BYTES_PER_BLOCK;
    registration
}

/// Four stripes; 64 blocks minimum, 128 mapped and 256 virtual per stripe.
fn striped_receipt(registration: &KvParticipantRegistration) -> KvRegistrationReceipt {
    KvRegistrationReceipt {
        participant_id: registration.participant_id.clone(),
        participant_epoch: 3,
        shared_pools: vec![KvSharedPoolDescriptor {
            binding_id: "cuda:0".to_string(),
            capacity_pool_id: "vllm.pool.0".to_string(),
            generation: 1,
            group_ids: vec!["vllm.group.0".to_string()],
            memory_domain: KvMemoryDomain::Cuda { device_id: 0 },
            block_count: 256,
            bytes_per_block: BYTES_PER_BLOCK,
            allocation_mode: KvSharedPoolAllocationMode::ParticipantManaged,
            transport: KvTransport::CudaVmm,
            descriptor: "scm_rights:cuda-vmm-v1".to_string(),
            elastic: Some(KvElasticPoolDescriptor {
                minimum_block_count: 64,
                mapped_block_count: 128,
                maximum_block_count: 256,
                allocation_granularity_bytes: GRANULE,
                resize_alignment_blocks: 64,
                stripe_count: Some(STRIPES),
                segments: stripe_segments("initial", STRIPES, 0, 128 * STRIPE_BYTES_PER_BLOCK),
            }),
        }],
    }
}

fn elastic(receipt: &mut KvRegistrationReceipt) -> &mut KvElasticPoolDescriptor {
    receipt.shared_pools[0].elastic.as_mut().expect("elastic")
}

/// Growth from 128 to 192 blocks: one 64 KiB tail segment per stripe.
fn striped_growth() -> KvPoolResizeOperation {
    KvPoolResizeOperation {
        participant_epoch: 3,
        resize_generation: 7,
        binding_id: "cuda:0".to_string(),
        stage: KvPoolResizeStage::MapWorkers,
        from_block_count: 128,
        target_block_count: 192,
        bytes_per_block: BYTES_PER_BLOCK,
        allocation_granularity_bytes: GRANULE,
        stripe_count: Some(STRIPES),
        segments: stripe_segments("grow-7", STRIPES, 128 * 1024, 192 * 1024),
    }
}

#[track_caller]
fn assert_invalid_capabilities(result: Result<(), KvContractError>, needle: &str) {
    match result {
        Err(KvContractError::InvalidCapabilities { message }) => {
            assert!(message.contains(needle), "'{message}' lacks '{needle}'")
        }
        other => panic!("expected invalid capabilities ('{needle}'), got {other:?}"),
    }
}

#[track_caller]
fn assert_invalid_request(result: Result<(), KvContractError>, needle: &str) {
    match result {
        Err(KvContractError::InvalidRequest { message }) => {
            assert!(message.contains(needle), "'{message}' lacks '{needle}'")
        }
        other => panic!("expected invalid request ('{needle}'), got {other:?}"),
    }
}

#[test]
fn striped_receipt_round_trips_with_per_stripe_segments() {
    let registration = striped_registration();
    registration.validate().expect("valid striped registration");
    let receipt = striped_receipt(&registration);
    receipt
        .validate_for(&registration)
        .expect("valid striped receipt");

    let value = serde_json::to_value(&receipt).expect("serialize striped receipt");
    let wire = &value["shared_pools"][0]["elastic"];
    assert_eq!(wire["stripe_count"], 4);
    // Stripe 0 is the default and stays off the wire; offsets are per stripe.
    assert!(wire["segments"][0].get("stripe").is_none());
    assert_eq!(wire["segments"][3]["stripe"], 3);
    assert_eq!(wire["segments"][3]["offset_bytes"], 0);
    let decoded: KvRegistrationReceipt =
        serde_json::from_value(value).expect("deserialize striped receipt");
    assert_eq!(decoded, receipt);

    // Several segments may back one stripe as long as they stay dense.
    let mut split = receipt;
    let segments = &mut elastic(&mut split).segments;
    segments[2].length_bytes = 64 * 1024;
    segments.push(segment("initial-2b", 2, 64 * 1024, 64 * 1024, 9));
    split
        .validate_for(&registration)
        .expect("a stripe may hold several segments");
}

#[test]
fn documents_without_stripes_reserialize_byte_identically() {
    let receipt = r#"{"participant_id":"vllm-0","participant_epoch":3,"shared_pools":[{"binding_id":"cuda:0","capacity_pool_id":"vllm.pool.0","generation":1,"group_ids":["vllm.group.0"],"memory_domain":{"kind":"cuda","device_id":0},"block_count":64,"bytes_per_block":4096,"allocation_mode":"participant_managed","transport":{"kind":"cuda_vmm"},"descriptor":"scm_rights:cuda-vmm-v1","elastic":{"minimum_block_count":16,"mapped_block_count":32,"maximum_block_count":64,"allocation_granularity_bytes":65536,"resize_alignment_blocks":16,"segments":[{"segment_id":"initial-0","offset_bytes":0,"length_bytes":131072,"handle_index":0}]}}]}"#;
    let decoded: KvRegistrationReceipt = serde_json::from_str(receipt).expect("1.5 receipt");
    decoded.shared_pools[0]
        .validate()
        .expect("valid 1.5 binding");
    let elastic = decoded.shared_pools[0].elastic.as_ref().expect("elastic");
    assert_eq!((elastic.stripe_count, elastic.stripe_count()), (None, 1));
    assert_eq!(elastic.segments[0].stripe, 0);
    assert_eq!(serde_json::to_string(&decoded).unwrap(), receipt);

    let resize = r#"{"pending":true,"operations":[{"participant_epoch":3,"resize_generation":7,"binding_id":"cuda:0","stage":"map_workers","from_block_count":32,"target_block_count":48,"bytes_per_block":4096,"allocation_granularity_bytes":65536,"segments":[{"segment_id":"grow-7-0","offset_bytes":131072,"length_bytes":65536,"handle_index":0}]}]}"#;
    let decoded: KvResizePollResult = serde_json::from_str(resize).expect("1.5 resize poll");
    decoded.validate().expect("valid 1.5 resize");
    assert_eq!(decoded.operations[0].stripe_count(), 1);
    assert_eq!(serde_json::to_string(&decoded).unwrap(), resize);

    let attachment = r#"{"participant_epoch":7,"binding_id":"cuda:0","shard":{"tensor_parallel_rank":0,"tensor_parallel_world_size":1,"pipeline_parallel_rank":0,"pipeline_parallel_world_size":1},"profile":{"adapter_id":"kapsl-vllm-connector","adapter_version":"0.4.0","backend_version":"test-vllm","profile_id":"vllm-v1-packed-cuda-vmm"},"imported_bytes":262144,"mapped_bytes":131072,"views":[{"group_id":"vllm.group.0","layer":{"index":0,"name":"model.layers.0.attn"},"offset_bytes":0,"length_bytes":262144}]}"#;
    let decoded: KvSharedPoolAttachment = serde_json::from_str(attachment).expect("1.5 attachment");
    decoded.validate().expect("valid 1.5 attachment");
    assert_eq!(serde_json::to_string(&decoded).unwrap(), attachment);
}

#[test]
fn stripe_count_has_one_representation_and_an_upper_bound() {
    let registration = striped_registration();
    for invalid in [0, 1, KV_MAX_POOL_STRIPES + 1] {
        let mut receipt = striped_receipt(&registration);
        elastic(&mut receipt).stripe_count = Some(invalid);
        assert_invalid_capabilities(
            receipt.shared_pools[0].validate(),
            "stripe_count must lie between 2 and KV_MAX_POOL_STRIPES",
        );
    }

    // A single range is written without the field; an explicit 1 is refused.
    let mut value = serde_json::to_value(striped_receipt(&registration).shared_pools[0].clone())
        .expect("serialize binding");
    value["elastic"]["stripe_count"] = 1.into();
    value["elastic"]["segments"] =
        serde_json::to_value(vec![segment("initial-0", 0, 0, 128 * BYTES_PER_BLOCK, 0)]).unwrap();
    let explicit_one: KvSharedPoolDescriptor = serde_json::from_value(value.clone()).unwrap();
    assert_invalid_capabilities(explicit_one.validate(), "omit it for a single range");
    value["elastic"]
        .as_object_mut()
        .unwrap()
        .remove("stripe_count");
    let single: KvSharedPoolDescriptor = serde_json::from_value(value).unwrap();
    single
        .validate()
        .expect("the same binding without the field");

    // The bound itself is accepted.
    let stripes = KV_MAX_POOL_STRIPES;
    let widest = KvSharedPoolDescriptor {
        block_count: 2,
        bytes_per_block: GRANULE * u64::from(stripes),
        elastic: Some(KvElasticPoolDescriptor {
            minimum_block_count: 1,
            mapped_block_count: 1,
            maximum_block_count: 2,
            allocation_granularity_bytes: GRANULE,
            resize_alignment_blocks: 1,
            stripe_count: Some(stripes),
            segments: stripe_segments("widest", stripes, 0, GRANULE),
        }),
        ..striped_receipt(&registration).shared_pools[0].clone()
    };
    widest.validate().expect("the stripe bound is inclusive");
}

#[test]
fn striped_bytes_per_block_must_divide_by_the_stripe_count() {
    let registration = striped_registration();
    let mut receipt = striped_receipt(&registration);
    elastic(&mut receipt).stripe_count = Some(3);
    elastic(&mut receipt).segments.pop();
    assert_invalid_capabilities(
        receipt.shared_pools[0].validate(),
        "divide evenly across stripes",
    );
}

#[test]
fn striped_sizes_must_align_to_the_granule_in_each_stripe() {
    // In each case the whole binding is a whole number of granules, but each
    // stripe's share ends half way through a granule.
    let registration = striped_registration();

    let mut minimum = striped_receipt(&registration);
    elastic(&mut minimum).resize_alignment_blocks = 32;
    elastic(&mut minimum).minimum_block_count = 32;
    assert_invalid_capabilities(
        minimum.shared_pools[0].validate(),
        "(per stripe) must align to CUDA VMM granularity",
    );

    let mut mapped = striped_receipt(&registration);
    elastic(&mut mapped).resize_alignment_blocks = 32;
    elastic(&mut mapped).mapped_block_count = 96;
    assert_invalid_capabilities(
        mapped.shared_pools[0].validate(),
        "(per stripe) must align to CUDA VMM granularity",
    );

    let mut maximum = striped_receipt(&registration);
    maximum.shared_pools[0].block_count = 224;
    elastic(&mut maximum).maximum_block_count = 224;
    assert_invalid_capabilities(
        maximum.shared_pools[0].validate(),
        "(per stripe) must align to CUDA VMM granularity",
    );
}

#[test]
fn every_stripe_must_densely_cover_the_same_mapped_prefix() {
    let registration = striped_registration();
    let mapped = 128 * STRIPE_BYTES_PER_BLOCK;
    let cover = "exactly cover mapped physical bytes in every stripe";

    for missing in [1, 3] {
        let mut receipt = striped_receipt(&registration);
        elastic(&mut receipt)
            .segments
            .retain(|segment| segment.stripe != missing);
        assert_invalid_capabilities(receipt.validate_for(&registration), cover);
    }

    let mut short = striped_receipt(&registration);
    elastic(&mut short).segments[1].length_bytes = mapped - GRANULE;
    assert_invalid_capabilities(short.validate_for(&registration), cover);

    let mut long = striped_receipt(&registration);
    elastic(&mut long).segments[1].length_bytes = mapped + GRANULE;
    assert_invalid_capabilities(long.validate_for(&registration), cover);

    // A segment cannot continue into the next stripe: offsets laid out as one
    // concatenated range are refused.
    let mut concatenated = striped_receipt(&registration);
    for (index, segment) in elastic(&mut concatenated).segments.iter_mut().enumerate() {
        segment.stripe = 0;
        segment.offset_bytes = index as u64 * mapped;
    }
    assert_invalid_capabilities(concatenated.validate_for(&registration), cover);

    let mut out_of_range = striped_receipt(&registration);
    elastic(&mut out_of_range).segments[3].stripe = STRIPES;
    assert_invalid_capabilities(
        out_of_range.validate_for(&registration),
        "must name a stripe of their binding",
    );

    // A binding without stripes only has stripe 0.
    let mut unstriped = striped_receipt(&registration);
    elastic(&mut unstriped).stripe_count = None;
    elastic(&mut unstriped).segments = vec![segment("initial", 1, 0, 128 * BYTES_PER_BLOCK, 0)];
    assert_invalid_capabilities(
        unstriped.shared_pools[0].validate(),
        "must name a stripe of their binding",
    );
}

#[test]
fn segment_ids_and_handles_are_unique_across_stripes() {
    let registration = striped_registration();
    let unique = "unique IDs/handles";

    let mut duplicate_id = striped_receipt(&registration);
    elastic(&mut duplicate_id).segments[2].segment_id = "initial-0".to_string();
    assert_invalid_capabilities(duplicate_id.validate_for(&registration), unique);

    let mut duplicate_handle = striped_receipt(&registration);
    elastic(&mut duplicate_handle).segments[2].handle_index = 0;
    assert_invalid_capabilities(duplicate_handle.validate_for(&registration), unique);
}

#[test]
fn striped_resize_changes_every_stripe_by_the_same_tail() {
    let growth = striped_growth();
    growth.validate().expect("valid striped growth");
    assert_eq!(growth.stripe_count(), STRIPES);

    let activation = KvPoolResizeOperation {
        stage: KvPoolResizeStage::ActivateScheduler,
        segments: Vec::new(),
        ..growth.clone()
    };
    activation
        .validate()
        .expect("scheduler phases carry the stripe count but no segments");

    let shrink = KvPoolResizeOperation {
        stage: KvPoolResizeStage::UnmapWorkers,
        from_block_count: 192,
        target_block_count: 128,
        ..growth.clone()
    };
    shrink.validate().expect("valid striped shrink");

    let tail = "exactly cover the changed physical tail in every stripe";
    let mut missing_stripe = growth.clone();
    missing_stripe.segments.remove(2);
    assert_invalid_request(missing_stripe.validate(), tail);

    let mut uneven = growth.clone();
    uneven.segments[2].length_bytes += GRANULE;
    assert_invalid_request(uneven.validate(), tail);

    // Offsets are per stripe, not whole-binding offsets.
    let mut whole_binding_offsets = growth.clone();
    for segment in &mut whole_binding_offsets.segments {
        segment.offset_bytes = 128 * BYTES_PER_BLOCK;
    }
    assert_invalid_request(whole_binding_offsets.validate(), "densely cover");

    let mut duplicate_handle = growth.clone();
    duplicate_handle.segments[3].handle_index = 0;
    assert_invalid_request(duplicate_handle.validate(), "uniquely and densely cover");

    let mut out_of_range = growth.clone();
    out_of_range.segments[3].stripe = STRIPES;
    assert_invalid_request(out_of_range.validate(), "must name a stripe");

    let mut explicit_one = growth.clone();
    explicit_one.stripe_count = Some(1);
    assert_invalid_request(explicit_one.validate(), "omit it for a single range");

    let mut indivisible = growth.clone();
    indivisible.stripe_count = Some(3);
    assert_invalid_request(indivisible.validate(), "divide evenly across stripes");

    // 160 blocks: aligned for the whole binding, half a granule per stripe.
    let mut misaligned = growth.clone();
    misaligned.target_block_count = 160;
    assert_invalid_request(
        misaligned.validate(),
        "align to CUDA VMM allocation granularity",
    );

    let mut unstriped = growth;
    unstriped.stripe_count = None;
    unstriped.segments = vec![segment(
        "grow-7",
        1,
        128 * BYTES_PER_BLOCK,
        64 * BYTES_PER_BLOCK,
        0,
    )];
    assert_invalid_request(unstriped.validate(), "must name a stripe");
}

#[test]
fn striped_bindings_require_the_participant_feature() {
    let striped =
        KvBackendCapabilities::cuda_vmm_shared_pool().with_feature(KvFeature::StripedElasticPool);
    striped.validate().expect("striped VMM participant");

    let mut too_old = striped.clone();
    too_old.abi_version = KvAbiVersion::new(1, 5);
    assert_invalid_capabilities(too_old.validate(), "require KV ABI 1.6");

    let fixed =
        KvBackendCapabilities::cuda_ipc_shared_pool().with_feature(KvFeature::StripedElasticPool);
    assert_invalid_capabilities(fixed.validate(), "live pool resize");

    let registration = striped_registration();
    let receipt = striped_receipt(&registration);

    let mut not_asked = registration.clone();
    not_asked
        .capabilities
        .features
        .remove(&KvFeature::StripedElasticPool);
    not_asked.validate().expect("unstriped VMM participant");
    assert_invalid_capabilities(
        receipt.validate_for(&not_asked),
        "stripe layout does not match participant capabilities",
    );

    let mut unstriped = receipt;
    elastic(&mut unstriped).stripe_count = None;
    elastic(&mut unstriped).segments = vec![segment("initial", 0, 0, 128 * BYTES_PER_BLOCK, 0)];
    unstriped
        .validate_for(&not_asked)
        .expect("unstriped participant receives a single range");
    assert_invalid_capabilities(
        unstriped.validate_for(&registration),
        "stripe layout does not match participant capabilities",
    );
}

#[test]
fn striped_attachment_names_each_view_stripe() {
    let span = 256 * STRIPE_BYTES_PER_BLOCK;
    let view = |layer: u32, stripe: u32| KvAttachmentView {
        group_id: "vllm.group.0".to_string(),
        layer: KvLayerId::indexed(layer),
        stripe,
        offset_bytes: 0,
        length_bytes: span,
    };
    let attachment = KvSharedPoolAttachment {
        participant_epoch: 3,
        binding_id: "cuda:0".to_string(),
        shard: KvShard::default(),
        profile: KvAdapterProfile {
            adapter_id: "kapsl-sglang-adapter".to_string(),
            adapter_version: "0.1.0".to_string(),
            backend_version: "test-sglang".to_string(),
            profile_id: "sglang-striped-cuda-vmm".to_string(),
        },
        imported_bytes: 256 * BYTES_PER_BLOCK,
        mapped_bytes: Some(128 * BYTES_PER_BLOCK),
        // K and V of each layer live in separate stripes.
        views: vec![view(0, 0), view(0, 1), view(1, 2), view(1, 3)],
    };
    attachment.validate().expect("valid striped attachment");

    let value = serde_json::to_value(&attachment).expect("serialize attachment");
    assert!(value["views"][0].get("stripe").is_none());
    assert_eq!(value["views"][1]["stripe"], 1);
    let decoded: KvSharedPoolAttachment = serde_json::from_value(value).unwrap();
    assert_eq!(decoded, attachment);

    let mut duplicate = attachment;
    duplicate.views[3] = view(1, 2);
    assert_invalid_request(duplicate.validate(), "unique group/layer/stripe");
}

struct HeartbeatCoordinator;

impl KvCoordinator for HeartbeatCoordinator {
    fn register(
        &self,
        registration: &KvParticipantRegistration,
    ) -> Result<KvRegistrationReceipt, KvContractError> {
        Ok(KvRegistrationReceipt::opaque(
            registration.participant_id.clone(),
            1,
        ))
    }

    fn reserve(&self, _: &str, _: &KvReserveRequest) -> Result<KvLease, KvContractError> {
        Err(KvContractError::unsupported("reserve"))
    }

    fn commit(&self, _: &str, _: &KvCommitRequest) -> Result<(), KvContractError> {
        Err(KvContractError::unsupported("commit"))
    }

    fn touch(&self, _: &str, _: &str) -> Result<(), KvContractError> {
        Err(KvContractError::unsupported("touch"))
    }

    fn heartbeat(&self, _: &str) -> Result<(), KvContractError> {
        Ok(())
    }

    fn release(
        &self,
        _: &str,
        _: &str,
        _: Option<&KvReleaseCompletion>,
    ) -> Result<(), KvContractError> {
        Err(KvContractError::unsupported("release"))
    }
}

#[test]
fn hosts_answer_1_5_participants_in_1_5() {
    let v1_5 = KvAbiVersion::new(1, 5);
    let v1_7 = KvAbiVersion::new(1, 7);
    assert_eq!(KAPSL_KV_ABI_VERSION, KvAbiVersion::new(1, 6));
    assert_eq!(KAPSL_KV_ABI_VERSION.negotiate(v1_5), Some(v1_5));
    assert_eq!(
        KAPSL_KV_ABI_VERSION.negotiate(KAPSL_KV_ABI_VERSION),
        Some(KAPSL_KV_ABI_VERSION)
    );
    assert_eq!(KAPSL_KV_ABI_VERSION.negotiate(v1_7), None);
    assert_eq!(
        KAPSL_KV_ABI_VERSION.negotiate(KvAbiVersion::new(2, 0)),
        None
    );

    // The 1.5 connector documents still validate unchanged.
    let opaque: KvParticipantRegistration = serde_json::from_str(include_str!(
        "../../tests/fixtures/opaque_registration.json"
    ))
    .expect("opaque fixture");
    opaque.validate().expect("1.5 opaque registration");
    let shared: KvParticipantRegistration = serde_json::from_str(include_str!(
        "../../tests/fixtures/shared_pool_registration.json"
    ))
    .expect("shared fixture");
    shared.validate().expect("1.5 shared-pool registration");

    let dispatch = |abi_version: KvAbiVersion, request: KvControlRequest| {
        dispatch_control_request(
            &HeartbeatCoordinator,
            KvControlRequestEnvelope {
                abi_version,
                request_id: "rpc-1".to_string(),
                request,
            },
        )
    };
    let heartbeat = || KvControlRequest::Heartbeat {
        participant_id: "vllm-0".to_string(),
    };

    let registered = dispatch(
        v1_5,
        KvControlRequest::Register {
            registration: opaque,
        },
    );
    assert_eq!(registered.abi_version, v1_5);
    assert!(matches!(
        registered.response,
        KvControlResponse::Registered { .. }
    ));

    let ack = dispatch(v1_5, heartbeat());
    assert_eq!(
        (ack.abi_version, ack.response),
        (v1_5, KvControlResponse::Ack)
    );
    let current = dispatch(KAPSL_KV_ABI_VERSION, heartbeat());
    assert_eq!(current.abi_version, KAPSL_KV_ABI_VERSION);

    let invalid = dispatch(
        v1_5,
        KvControlRequest::Heartbeat {
            participant_id: " ".to_string(),
        },
    );
    assert_eq!(invalid.abi_version, v1_5);
    assert!(matches!(
        invalid.response,
        KvControlResponse::Error {
            error: KvContractError::InvalidRequest { .. }
        }
    ));

    let newer = dispatch(v1_7, heartbeat());
    assert_eq!(newer.abi_version, KAPSL_KV_ABI_VERSION);
    assert_eq!(
        newer.response,
        KvControlResponse::Error {
            error: KvContractError::VersionMismatch {
                host: KAPSL_KV_ABI_VERSION,
                participant: v1_7,
            }
        }
    );
}
