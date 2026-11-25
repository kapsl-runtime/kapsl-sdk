use super::*;

const OPAQUE_FIXTURE: &str = include_str!("../../tests/fixtures/participant_config_opaque.json");
const SHARED_POOL_FIXTURE: &str =
    include_str!("../../tests/fixtures/participant_config_shared_pool.json");

fn opaque_config() -> KvParticipantConfig {
    KvParticipantConfig {
        control_endpoint: "unix:///run/kapsl/kv-control.sock".to_string(),
        participant_id: "kapsl-model-1-0".to_string(),
        model_fingerprint: "sha256:model".to_string(),
        mode: KvParticipantMode::Opaque,
        memory_domains: vec![KvMemoryDomain::Cuda { device_id: 0 }],
        rank_device_map: BTreeMap::new(),
        lease_ttl_ms: 30_000,
        timeout_ms: None,
        live_resize: false,
        vmm_conformance: false,
        provisioning_grant: None,
    }
}

fn shared_pool_config() -> KvParticipantConfig {
    KvParticipantConfig {
        mode: KvParticipantMode::SharedPool,
        memory_domains: vec![
            KvMemoryDomain::Cuda { device_id: 2 },
            KvMemoryDomain::Cuda { device_id: 3 },
        ],
        rank_device_map: BTreeMap::from([("0".to_string(), 2), ("1".to_string(), 3)]),
        timeout_ms: Some(2_000),
        live_resize: true,
        provisioning_grant: Some(KvProvisioningGrant {
            token: "kvg1_test".to_string(),
            geometry_digest:
                "sha256:0221ae33b2c0585d8b3d97c875bbd9e8ceb6a8b7df5f4accd4ff435b6708b697"
                    .to_string(),
            authority_generation: 7,
            expires_at_unix_ms: 1_900_000_000_000,
        }),
        ..opaque_config()
    }
}

#[test]
fn participant_config_serializes_to_the_adapter_fixtures() {
    for (config, fixture) in [
        (opaque_config(), OPAQUE_FIXTURE),
        (shared_pool_config(), SHARED_POOL_FIXTURE),
    ] {
        config.validate().expect("valid participant config");
        let fixture: serde_json::Value = serde_json::from_str(fixture).expect("fixture is JSON");
        assert_eq!(serde_json::to_value(&config).unwrap(), fixture);
        let decoded: KvParticipantConfig =
            serde_json::from_value(fixture).expect("fixture uses the Rust wire shape");
        assert_eq!(decoded, config);
    }
}

#[test]
fn participant_config_tolerates_unknown_fields_and_defaults_optional_ones() {
    let config: KvParticipantConfig = serde_json::from_str(
        r#"{
            "kapsl_control_endpoint": "unix:///run/kapsl/kv-control.sock",
            "kapsl_participant_id": "kapsl-model-1-0",
            "kapsl_model_fingerprint": "sha256:model",
            "kapsl_kv_mode": "opaque",
            "kapsl_memory_domains": [{"kind": "host"}],
            "kapsl_lease_ttl_ms": 1000,
            "kapsl_added_by_a_newer_runtime": true
        }"#,
    )
    .expect("unknown fields are ignored");
    config.validate().expect("valid participant config");
    assert!(!config.live_resize && !config.vmm_conformance);
    assert!(config.rank_device_map.is_empty() && config.timeout_ms.is_none());

    assert!(serde_json::from_str::<KvParticipantConfig>(
        &OPAQUE_FIXTURE.replace("\"opaque\"", "\"something_else\"")
    )
    .is_err());
}

#[test]
fn participant_config_rejects_incomplete_or_inconsistent_documents() {
    let invalid: Vec<(&str, Box<dyn Fn(&mut KvParticipantConfig)>)> = vec![
        (
            "empty endpoint",
            Box::new(|config| config.control_endpoint = " ".to_string()),
        ),
        (
            "empty participant",
            Box::new(|config| config.participant_id.clear()),
        ),
        (
            "empty fingerprint",
            Box::new(|config| config.model_fingerprint.clear()),
        ),
        (
            "no domains",
            Box::new(|config| config.memory_domains.clear()),
        ),
        (
            "duplicate domains",
            Box::new(|config| config.memory_domains.push(config.memory_domains[0].clone())),
        ),
        (
            "unnamed provider domain",
            Box::new(|config| {
                config.memory_domains = vec![KvMemoryDomain::Provider {
                    provider: String::new(),
                    device_id: None,
                }]
            }),
        ),
        (
            "non-decimal rank",
            Box::new(|config| {
                config.rank_device_map.insert("rank0".to_string(), 0);
            }),
        ),
        ("zero TTL", Box::new(|config| config.lease_ttl_ms = 0)),
        (
            "zero timeout",
            Box::new(|config| config.timeout_ms = Some(0)),
        ),
        (
            "conformance without live resize",
            Box::new(|config| {
                config.live_resize = false;
                config.vmm_conformance = true;
            }),
        ),
        (
            "malformed grant",
            Box::new(|config| {
                config.provisioning_grant.as_mut().unwrap().token.clear();
            }),
        ),
    ];
    for (name, mutate) in invalid {
        let mut config = shared_pool_config();
        mutate(&mut config);
        assert!(config.validate().is_err(), "{name} must be rejected");
    }

    for (name, mutate) in [
        (
            "opaque live resize",
            Box::new(|config: &mut KvParticipantConfig| config.live_resize = true)
                as Box<dyn Fn(&mut KvParticipantConfig)>,
        ),
        (
            "opaque grant",
            Box::new(|config: &mut KvParticipantConfig| {
                config.provisioning_grant = shared_pool_config().provisioning_grant
            }),
        ),
    ] {
        let mut config = opaque_config();
        mutate(&mut config);
        assert!(config.validate().is_err(), "{name} must be rejected");
    }
}
