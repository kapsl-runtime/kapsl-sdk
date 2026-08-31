use super::*;

fn cuda_domains() -> Vec<KvMemoryDomain> {
    vec![KvMemoryDomain::Cuda { device_id: 0 }]
}

fn attention_group(id: &str, layers: &[u32], policy: KvCachePolicy) -> KvCacheGroup {
    KvCacheGroup {
        group_id: id.to_string(),
        layers: layers.iter().copied().map(KvLayerId::indexed).collect(),
        geometry: KvCacheGeometry::PagedAttention {
            block_size_tokens: 16,
            kv_heads: 8,
            key_head_dim: 128,
            value_head_dim: 128,
            element_type: KvElementType::F16,
            layout: KvTensorLayout::BlockKvHeadTokenDim,
        },
        policy,
    }
}

#[test]
fn version_compatibility_is_major_strict_and_minor_backward_compatible() {
    let host = KvAbiVersion::new(1, 2);
    assert!(host.accepts(KvAbiVersion::new(1, 0)));
    assert!(host.accepts(KvAbiVersion::new(1, 2)));
    assert!(!host.accepts(KvAbiVersion::new(1, 3)));
    assert!(!host.accepts(KvAbiVersion::new(2, 0)));
}

#[test]
fn unmanaged_and_shared_pool_capabilities_enforce_tier_invariants() {
    assert!(KvBackendCapabilities::unmanaged().validate().is_ok());
    assert!(KvBackendCapabilities::in_process_shared_pool()
        .validate()
        .is_ok());

    let mut invalid = KvBackendCapabilities::in_process_shared_pool();
    invalid.ownership = KvCacheOwnership::Backend;
    assert!(matches!(
        invalid.validate(),
        Err(KvContractError::InvalidCapabilities { .. })
    ));
}

#[test]
fn opaque_mode_is_connected_but_not_shared_pool() {
    let opaque = KvBackendCapabilities::opaque_connected();
    assert_eq!(opaque.tier, KvIntegrationTier::KvConnected);
    assert_eq!(opaque.metadata_mode, KvMetadataMode::Opaque);
    assert_eq!(opaque.ownership, KvCacheOwnership::Backend);
    assert!(opaque.validate().is_ok());
}

#[test]
fn opaque_capacity_rounds_tokens_to_accountable_backend_pages() {
    let group = KvCapacityGroup {
        group_id: "vllm.group.0".to_string(),
        pool_id: "vllm.pool.0".to_string(),
        allocation_granularity_tokens: 16,
        bytes_per_allocation: 4096,
        memory_domains: cuda_domains(),
        max_allocations: Some(10),
    };

    assert_eq!(group.bytes_for_tokens(0), Some(0));
    assert_eq!(group.bytes_for_tokens(1), Some(4096));
    assert_eq!(group.bytes_for_tokens(16), Some(4096));
    assert_eq!(group.bytes_for_tokens(17), Some(8192));
    assert_eq!(group.bytes_for_tokens(161), None);
    assert_eq!(
        group.bytes_for_reservation(&KvGroupReservation {
            group_id: "vllm.group.0".to_string(),
            token_capacity: 1,
            minimum_blocks: Some(3),
        }),
        Some(12_288)
    );
}

#[test]
fn aliased_cache_groups_are_charged_once_per_physical_pool() {
    let model = KvCapacityModel {
        groups: vec![
            KvCapacityGroup {
                group_id: "full".to_string(),
                pool_id: "device-pool".to_string(),
                allocation_granularity_tokens: 16,
                bytes_per_allocation: 4096,
                memory_domains: cuda_domains(),
                max_allocations: None,
            },
            KvCapacityGroup {
                group_id: "swa".to_string(),
                pool_id: "device-pool".to_string(),
                allocation_granularity_tokens: 16,
                bytes_per_allocation: 4096,
                memory_domains: cuda_domains(),
                max_allocations: None,
            },
        ],
    };
    let reservations = vec![
        KvGroupReservation {
            group_id: "full".to_string(),
            token_capacity: 17,
            minimum_blocks: None,
        },
        KvGroupReservation {
            group_id: "swa".to_string(),
            token_capacity: 1,
            minimum_blocks: None,
        },
    ];

    assert_eq!(model.bytes_for_reservations(&reservations), Some(8192));
}

#[test]
fn replicated_pool_is_charged_on_every_physical_domain() {
    let model = KvCapacityModel {
        groups: vec![KvCapacityGroup {
            group_id: "attention".to_string(),
            pool_id: "tp-pool".to_string(),
            allocation_granularity_tokens: 16,
            bytes_per_allocation: 4096,
            memory_domains: vec![
                KvMemoryDomain::Cuda { device_id: 0 },
                KvMemoryDomain::Cuda { device_id: 1 },
            ],
            max_allocations: None,
        }],
    };
    let reservations = vec![KvGroupReservation {
        group_id: "attention".to_string(),
        token_capacity: 17,
        minimum_blocks: None,
    }];

    let bytes = model
        .bytes_by_domain_for_reservations(&reservations)
        .expect("valid placement accounting");
    assert_eq!(bytes[&KvMemoryDomain::Cuda { device_id: 0 }], 8192);
    assert_eq!(bytes[&KvMemoryDomain::Cuda { device_id: 1 }], 8192);
    assert_eq!(model.bytes_for_reservations(&reservations), Some(16_384));
}

#[test]
fn hybrid_multi_group_topology_round_trips() {
    let topology = KvTopology {
        abi_version: KAPSL_KV_ABI_VERSION,
        model_fingerprint: "sha256:example".to_string(),
        shard: KvShard::default(),
        cache_groups: vec![
            attention_group("full", &[0, 2], KvCachePolicy::FullAttention),
            attention_group(
                "swa",
                &[1, 3],
                KvCachePolicy::SlidingWindow {
                    window_tokens: 4096,
                },
            ),
            KvCacheGroup {
                group_id: "ssm".to_string(),
                layers: vec![KvLayerId::indexed(4)],
                geometry: KvCacheGeometry::RecurrentState {
                    state_bytes_per_sequence: 64 * 1024,
                    element_type: KvElementType::F16,
                    layout: KvTensorLayout::BackendNative {
                        layout_id: "mamba-state-v1".to_string(),
                    },
                },
                policy: KvCachePolicy::Recurrent,
            },
        ],
    };
    assert!(topology.validate().is_ok());

    let encoded = serde_json::to_string(&topology).expect("serialize topology");
    let decoded: KvTopology = serde_json::from_str(&encoded).expect("deserialize topology");
    assert_eq!(decoded, topology);
}

#[test]
fn topology_allows_one_layer_to_own_independent_cache_groups() {
    let topology = KvTopology {
        abi_version: KAPSL_KV_ABI_VERSION,
        model_fingerprint: "model".to_string(),
        shard: KvShard::default(),
        cache_groups: vec![
            attention_group("a", &[0, 1], KvCachePolicy::FullAttention),
            attention_group("b", &[1, 2], KvCachePolicy::FullAttention),
        ],
    };
    assert!(topology.validate().is_ok());
}

#[test]
fn topology_rejects_duplicate_layers_within_one_group() {
    let topology = KvTopology {
        abi_version: KAPSL_KV_ABI_VERSION,
        model_fingerprint: "model".to_string(),
        shard: KvShard::default(),
        cache_groups: vec![attention_group(
            "attention",
            &[0, 0],
            KvCachePolicy::FullAttention,
        )],
    };
    assert!(matches!(
        topology.validate(),
        Err(KvContractError::InvalidTopology { .. })
    ));
}

#[test]
fn structured_registration_requires_multi_group_capability() {
    let topology = KvTopology {
        abi_version: KAPSL_KV_ABI_VERSION,
        model_fingerprint: "model".to_string(),
        shard: KvShard::default(),
        cache_groups: vec![
            attention_group("full", &[0], KvCachePolicy::FullAttention),
            attention_group(
                "swa",
                &[1],
                KvCachePolicy::SlidingWindow { window_tokens: 128 },
            ),
        ],
    };
    let mut registration = KvParticipantRegistration {
        participant_id: "worker-0".to_string(),
        backend: "test".to_string(),
        model_fingerprint: "model".to_string(),
        capabilities: KvBackendCapabilities::in_process_shared_pool(),
        capacity_model: KvCapacityModel {
            groups: vec![
                KvCapacityGroup {
                    group_id: "full".to_string(),
                    pool_id: "test.pool".to_string(),
                    allocation_granularity_tokens: 16,
                    bytes_per_allocation: 4096,
                    memory_domains: cuda_domains(),
                    max_allocations: Some(1024),
                },
                KvCapacityGroup {
                    group_id: "swa".to_string(),
                    pool_id: "test.pool".to_string(),
                    allocation_granularity_tokens: 16,
                    bytes_per_allocation: 4096,
                    memory_domains: cuda_domains(),
                    max_allocations: Some(1024),
                },
            ],
        },
        adapter_profile: None,
        topology: Some(topology),
        provisioning_grant: None,
    };
    assert!(registration.validate().is_err());
    registration
        .capabilities
        .features
        .insert(KvFeature::MultipleCacheGroups);
    assert!(registration.validate().is_ok());
}

#[test]
fn shared_pool_receipt_covers_every_runtime_owned_binding() {
    let registration = KvParticipantRegistration {
        participant_id: "vllm-worker-0".to_string(),
        backend: "vllm".to_string(),
        model_fingerprint: "model".to_string(),
        capabilities: KvBackendCapabilities::cuda_ipc_shared_pool(),
        capacity_model: KvCapacityModel {
            groups: vec![KvCapacityGroup {
                group_id: "attention".to_string(),
                pool_id: "kv-pool".to_string(),
                allocation_granularity_tokens: 16,
                bytes_per_allocation: 4096,
                memory_domains: cuda_domains(),
                max_allocations: Some(64),
            }],
        },
        adapter_profile: Some(KvAdapterProfile {
            adapter_id: "kapsl-test-adapter".to_string(),
            adapter_version: "1.0.0".to_string(),
            backend_version: "test-backend-1".to_string(),
            profile_id: "test-cuda-ipc-v1".to_string(),
        }),
        topology: Some(KvTopology {
            abi_version: KAPSL_KV_ABI_VERSION,
            model_fingerprint: "model".to_string(),
            shard: KvShard::default(),
            cache_groups: vec![attention_group(
                "attention",
                &[0, 1],
                KvCachePolicy::FullAttention,
            )],
        }),
        provisioning_grant: None,
    };
    registration.validate().expect("valid shared registration");
    let mut missing_profile = registration.clone();
    missing_profile.adapter_profile = None;
    assert!(matches!(
        missing_profile.validate(),
        Err(KvContractError::InvalidCapabilities { .. })
    ));

    let receipt = KvRegistrationReceipt {
        participant_id: registration.participant_id.clone(),
        participant_epoch: 7,
        shared_pools: vec![KvSharedPoolDescriptor {
            binding_id: "runtime-binding-0".to_string(),
            capacity_pool_id: "kv-pool".to_string(),
            generation: 11,
            group_ids: vec!["attention".to_string()],
            memory_domain: KvMemoryDomain::Cuda { device_id: 0 },
            block_count: 64,
            bytes_per_block: 4096,
            allocation_mode: KvSharedPoolAllocationMode::RuntimeLeased,
            transport: KvTransport::CudaIpc,
            descriptor: "base64-cuda-ipc-handle".to_string(),
            elastic: None,
        }],
    };
    receipt
        .validate_for(&registration)
        .expect("receipt covers the registration");

    let mut oversized = receipt.clone();
    oversized.shared_pools[0].block_count = 65;
    assert!(matches!(
        oversized.validate_for(&registration),
        Err(KvContractError::InvalidCapabilities { .. })
    ));

    let mut participant_managed_registration = registration.clone();
    participant_managed_registration
        .capabilities
        .features
        .insert(KvFeature::ParticipantBlockSelection);
    assert!(matches!(
        receipt.validate_for(&participant_managed_registration),
        Err(KvContractError::InvalidCapabilities { .. })
    ));
    let mut participant_managed_receipt = receipt.clone();
    participant_managed_receipt.shared_pools[0].allocation_mode =
        KvSharedPoolAllocationMode::ParticipantManaged;
    participant_managed_receipt
        .validate_for(&participant_managed_registration)
        .expect("participant-managed mode matches its advertised feature");

    let mut missing = receipt;
    missing.shared_pools.clear();
    assert!(matches!(
        missing.validate_for(&registration),
        Err(KvContractError::InvalidCapabilities { .. })
    ));
}

#[test]
fn provisioning_grant_requires_exact_shared_pool_capability_pair() {
    let mut registration: KvParticipantRegistration = serde_json::from_str(include_str!(
        "../../tests/fixtures/shared_pool_registration.json"
    ))
    .expect("valid shared-pool fixture");
    let grant = KvProvisioningGrant {
        token: format!("kvg1_{}", "ab".repeat(32)),
        geometry_digest: format!("sha256:{}", "cd".repeat(32)),
        authority_generation: 7,
        expires_at_unix_ms: 1_800_000_000_000,
    };
    registration.provisioning_grant = Some(grant.clone());
    registration
        .capabilities
        .features
        .insert(KvFeature::ProvisioningGrant);
    registration.validate().expect("valid provisioning grant");

    let mut missing_capability = registration.clone();
    missing_capability
        .capabilities
        .features
        .remove(&KvFeature::ProvisioningGrant);
    assert!(matches!(
        missing_capability.validate(),
        Err(KvContractError::InvalidCapabilities { .. })
    ));

    let mut malformed = registration.clone();
    malformed
        .provisioning_grant
        .as_mut()
        .expect("grant")
        .geometry_digest = "SHA256:not-canonical".to_string();
    assert!(matches!(
        malformed.validate(),
        Err(KvContractError::InvalidRequest { .. })
    ));

    let mut opaque = registration;
    opaque.capabilities =
        KvBackendCapabilities::opaque_connected().with_feature(KvFeature::ProvisioningGrant);
    opaque.adapter_profile = None;
    opaque.topology = None;
    opaque.provisioning_grant = Some(grant);
    assert!(matches!(
        opaque.validate(),
        Err(KvContractError::InvalidCapabilities { .. })
    ));
}

#[test]
fn cuda_vmm_receipt_separates_virtual_and_mapped_capacity() {
    let mut registration: KvParticipantRegistration = serde_json::from_str(include_str!(
        "../../tests/fixtures/shared_pool_registration.json"
    ))
    .expect("shared registration fixture");
    registration.capabilities = KvBackendCapabilities::cuda_vmm_shared_pool()
        .with_feature(KvFeature::ParticipantBlockSelection);
    registration.capacity_model.groups[0].max_allocations = Some(64);
    registration.capacity_model.groups[0].bytes_per_allocation = 4096;

    let receipt = KvRegistrationReceipt {
        participant_id: registration.participant_id.clone(),
        participant_epoch: 3,
        shared_pools: vec![KvSharedPoolDescriptor {
            binding_id: "cuda:0".to_string(),
            capacity_pool_id: "vllm.pool.0".to_string(),
            generation: 1,
            group_ids: vec!["vllm.group.0".to_string()],
            memory_domain: KvMemoryDomain::Cuda { device_id: 0 },
            block_count: 64,
            bytes_per_block: 4096,
            allocation_mode: KvSharedPoolAllocationMode::ParticipantManaged,
            transport: KvTransport::CudaVmm,
            descriptor: "scm_rights:cuda-vmm-v1".to_string(),
            elastic: Some(KvElasticPoolDescriptor {
                minimum_block_count: 16,
                mapped_block_count: 32,
                maximum_block_count: 64,
                allocation_granularity_bytes: 65_536,
                resize_alignment_blocks: 16,
                segments: vec![KvVmmSegmentDescriptor {
                    segment_id: "initial-0".to_string(),
                    offset_bytes: 0,
                    length_bytes: 131_072,
                    handle_index: 0,
                }],
            }),
        }],
    };
    receipt
        .validate_for(&registration)
        .expect("valid elastic receipt");

    let mut inexact = receipt.clone();
    inexact.shared_pools[0]
        .elastic
        .as_mut()
        .expect("elastic")
        .mapped_block_count = 31;
    assert!(matches!(
        inexact.validate_for(&registration),
        Err(KvContractError::InvalidCapabilities { .. })
    ));

    let mut below_minimum = receipt.clone();
    below_minimum.shared_pools[0]
        .elastic
        .as_mut()
        .expect("elastic")
        .minimum_block_count = 48;
    assert!(matches!(
        below_minimum.validate_for(&registration),
        Err(KvContractError::InvalidCapabilities { .. })
    ));

    let mut missing_feature = registration;
    missing_feature
        .capabilities
        .features
        .remove(&KvFeature::LivePoolResize);
    assert!(matches!(
        receipt.validate_for(&missing_feature),
        Err(KvContractError::InvalidCapabilities { .. })
    ));
}

#[test]
fn resize_operation_enforces_worker_then_scheduler_ordering() {
    let operation = KvPoolResizeOperation {
        participant_epoch: 3,
        resize_generation: 7,
        binding_id: "cuda:0".to_string(),
        stage: KvPoolResizeStage::MapWorkers,
        from_block_count: 32,
        target_block_count: 48,
        bytes_per_block: 4096,
        allocation_granularity_bytes: 65_536,
        segments: vec![KvVmmSegmentDescriptor {
            segment_id: "grow-7-0".to_string(),
            offset_bytes: 131_072,
            length_bytes: 65_536,
            handle_index: 0,
        }],
    };
    operation.validate().expect("valid worker map operation");

    let mut wrong_stage = operation.clone();
    wrong_stage.stage = KvPoolResizeStage::RetireScheduler;
    assert!(matches!(
        wrong_stage.validate(),
        Err(KvContractError::InvalidRequest { .. })
    ));

    let scheduler = KvPoolResizeOperation {
        stage: KvPoolResizeStage::ActivateScheduler,
        segments: Vec::new(),
        ..operation
    };
    scheduler
        .validate()
        .expect("scheduler activation follows worker mapping");
}

#[test]
fn release_completion_rejects_opaque_or_empty_fences() {
    assert!(KvReleaseCompletion::BackendSynchronized.validate().is_ok());
    assert!(KvReleaseCompletion::TransportFence {
        transport: KvTransport::CudaIpc,
        descriptor: "event-handle".to_string(),
    }
    .validate()
    .is_ok());
    assert!(KvReleaseCompletion::TransportFence {
        transport: KvTransport::BackendOpaque,
        descriptor: String::new(),
    }
    .validate()
    .is_err());
}

#[test]
fn attachment_evidence_is_bounded_and_pointer_free_on_the_wire() {
    let attachment = KvSharedPoolAttachment {
        participant_epoch: 7,
        binding_id: "binding-0".to_string(),
        shard: KvShard::default(),
        profile: KvAdapterProfile {
            adapter_id: "kapsl-vllm-connector".to_string(),
            adapter_version: "0.4.0".to_string(),
            backend_version: "test-vllm".to_string(),
            profile_id: "vllm-v1-packed-cuda-ipc".to_string(),
        },
        imported_bytes: 4096,
        mapped_bytes: None,
        views: vec![KvAttachmentView {
            group_id: "vllm.group.0".to_string(),
            layer: KvLayerId {
                index: 0,
                name: Some("model.layers.0.attn".to_string()),
            },
            offset_bytes: 128,
            length_bytes: 1024,
        }],
    };
    attachment.validate().expect("valid attachment evidence");
    let envelope = KvControlRequestEnvelope {
        abi_version: KAPSL_KV_ABI_VERSION,
        request_id: "rpc-attach".to_string(),
        request: KvControlRequest::Attach {
            participant_id: "vllm-0".to_string(),
            attachment,
        },
    };
    let value = serde_json::to_value(envelope).expect("serialize attachment");
    assert_eq!(value["operation"], "attach");
    assert_eq!(value["attachment"]["views"][0]["offset_bytes"], 128);
    assert!(!value.to_string().contains("pointer"));
}

#[test]
fn attachment_view_cannot_extend_past_the_imported_binding() {
    let attachment = KvSharedPoolAttachment {
        participant_epoch: 1,
        binding_id: "binding-0".to_string(),
        shard: KvShard::default(),
        profile: KvAdapterProfile {
            adapter_id: "adapter".to_string(),
            adapter_version: "1".to_string(),
            backend_version: "1".to_string(),
            profile_id: "profile".to_string(),
        },
        imported_bytes: 64,
        mapped_bytes: None,
        views: vec![KvAttachmentView {
            group_id: "group-0".to_string(),
            layer: KvLayerId::indexed(0),
            offset_bytes: 32,
            length_bytes: 64,
        }],
    };
    assert!(matches!(
        attachment.validate(),
        Err(KvContractError::InvalidRequest { .. })
    ));
}

#[test]
fn control_envelope_has_stable_flat_json_shape() {
    let envelope = KvControlRequestEnvelope {
        abi_version: KAPSL_KV_ABI_VERSION,
        request_id: "rpc-7".to_string(),
        request: KvControlRequest::Touch {
            participant_id: "vllm-0".to_string(),
            lease_id: "lease-9".to_string(),
        },
    };

    let value = serde_json::to_value(&envelope).expect("serialize control request");
    assert_eq!(value["abi_version"]["major"], 1);
    assert_eq!(value["request_id"], "rpc-7");
    assert_eq!(value["operation"], "touch");
    assert_eq!(value["participant_id"], "vllm-0");
    assert_eq!(value["lease_id"], "lease-9");
    let decoded: KvControlRequestEnvelope =
        serde_json::from_value(value).expect("deserialize control request");
    assert_eq!(decoded, envelope);
}

#[test]
fn opaque_registration_matches_out_of_tree_connector_fixture() {
    let registration = KvParticipantRegistration {
        participant_id: "vllm-0".to_string(),
        backend: "vllm".to_string(),
        model_fingerprint: "sha256:model".to_string(),
        capabilities: KvBackendCapabilities::opaque_connected(),
        capacity_model: KvCapacityModel {
            groups: vec![KvCapacityGroup {
                group_id: "vllm.group.0".to_string(),
                pool_id: "vllm.pool.0".to_string(),
                allocation_granularity_tokens: 16,
                bytes_per_allocation: 1_048_576,
                memory_domains: cuda_domains(),
                max_allocations: Some(1024),
            }],
        },
        adapter_profile: None,
        topology: None,
        provisioning_grant: None,
    };
    registration.validate().expect("valid registration");

    let fixture: serde_json::Value = serde_json::from_str(include_str!(
        "../../tests/fixtures/opaque_registration.json"
    ))
    .expect("valid connector fixture");
    assert_eq!(serde_json::to_value(registration).unwrap(), fixture);
}

#[test]
fn shared_pool_registration_accepts_out_of_tree_connector_fixture() {
    let fixture: KvParticipantRegistration = serde_json::from_str(include_str!(
        "../../tests/fixtures/shared_pool_registration.json"
    ))
    .expect("connector fixture must use the Rust wire shape");

    fixture.validate().expect("valid shared-pool registration");
    let topology = fixture.topology.expect("shared-pool topology");
    let KvCacheGeometry::PagedAttention { element_type, .. } = &topology.cache_groups[0].geometry
    else {
        panic!("fixture must contain paged-attention geometry");
    };
    assert_eq!(element_type, &KvElementType::F16);
}
