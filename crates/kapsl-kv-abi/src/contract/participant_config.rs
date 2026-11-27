//! Configuration a managing runtime hands to a participant before it registers.

use super::*;

/// Environment variable carrying one [`KvParticipantConfig`] JSON document.
///
/// Adapters for backends without a per-connector configuration channel read
/// the document from this variable. A backend that has such a channel (vLLM's
/// `kv_connector_extra_config`) receives the same document through it.
pub const KV_PARTICIPANT_CONFIG_ENV: &str = "KAPSL_KV_PARTICIPANT_CONFIG";

/// Integration mode the runtime expects the participant to register with.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum KvParticipantMode {
    /// `kv_connected`, opaque metadata, backend-owned KV.
    Opaque,
    /// `shared_pool`, structured metadata, runtime-owned KV.
    SharedPool,
}

/// Identity and control-plane settings a runtime assigns to one participant.
///
/// The runtime chooses every value before the backend process exists; the
/// participant repeats them in its registration. The serialized field names
/// are the contract and are shared by every adapter, whichever channel
/// delivers the document. Adapters ignore fields they do not know, so a
/// runtime may add fields for a deeper tier without breaking an opaque one.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct KvParticipantConfig {
    /// Control endpoint of the runtime, for example
    /// `unix:///run/kapsl/kv-control.sock`.
    #[serde(rename = "kapsl_control_endpoint")]
    pub control_endpoint: String,
    /// Stable base identity. A participant registers as this value or as
    /// `<base>:<suffix>`, so the runtime can recognize what it launched.
    #[serde(rename = "kapsl_participant_id")]
    pub participant_id: String,
    #[serde(rename = "kapsl_model_fingerprint")]
    pub model_fingerprint: String,
    #[serde(rename = "kapsl_kv_mode")]
    pub mode: KvParticipantMode,
    /// Physical domains of the participant's KV pool, in the runtime's device
    /// namespace (which may differ from the device indices the child sees).
    #[serde(rename = "kapsl_memory_domains")]
    pub memory_domains: Vec<KvMemoryDomain>,
    /// Runtime device ID of each parallel rank, keyed by decimal rank.
    #[serde(
        rename = "kapsl_rank_device_map",
        default,
        skip_serializing_if = "BTreeMap::is_empty"
    )]
    pub rank_device_map: BTreeMap<String, u32>,
    /// Lease TTL the participant requests; never above the runtime maximum.
    #[serde(rename = "kapsl_lease_ttl_ms")]
    pub lease_ttl_ms: u64,
    /// Control round-trip timeout. Adapters apply their own default if absent.
    #[serde(
        rename = "kapsl_timeout_ms",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub timeout_ms: Option<u64>,
    #[serde(rename = "kapsl_live_resize", default)]
    pub live_resize: bool,
    #[serde(rename = "kapsl_vmm_conformance", default)]
    pub vmm_conformance: bool,
    /// Pre-start reservation proof the participant repeats at registration.
    #[serde(
        rename = "kapsl_provisioning_grant",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub provisioning_grant: Option<KvProvisioningGrant>,
}

impl KvParticipantConfig {
    pub fn validate(&self) -> Result<(), KvContractError> {
        if self.control_endpoint.trim().is_empty()
            || self.participant_id.trim().is_empty()
            || self.model_fingerprint.trim().is_empty()
        {
            return Err(KvContractError::invalid_request(
                "participant config requires a control endpoint, participant ID, and model fingerprint",
            ));
        }
        if self.memory_domains.is_empty() {
            return Err(KvContractError::invalid_request(
                "participant config requires at least one memory domain",
            ));
        }
        let mut domains = BTreeSet::new();
        for domain in &self.memory_domains {
            domain.validate()?;
            if !domains.insert(domain) {
                return Err(KvContractError::invalid_request(
                    "participant config memory domains must be unique",
                ));
            }
        }
        if self
            .rank_device_map
            .keys()
            .any(|rank| rank.is_empty() || !rank.bytes().all(|byte| byte.is_ascii_digit()))
        {
            return Err(KvContractError::invalid_request(
                "participant config rank_device_map keys must be decimal ranks",
            ));
        }
        if self.lease_ttl_ms == 0 || self.timeout_ms == Some(0) {
            return Err(KvContractError::invalid_request(
                "participant config lease TTL and timeout must be positive",
            ));
        }
        if self.vmm_conformance && !self.live_resize {
            return Err(KvContractError::invalid_request(
                "participant config vmm_conformance requires live_resize",
            ));
        }
        if self.mode == KvParticipantMode::Opaque
            && (self.live_resize || self.provisioning_grant.is_some())
        {
            return Err(KvContractError::invalid_request(
                "live resize and provisioning grants require the shared_pool mode",
            ));
        }
        if let Some(grant) = &self.provisioning_grant {
            grant.validate()?;
        }
        Ok(())
    }
}

#[cfg(test)]
#[path = "../tests/participant_config.rs"]
mod tests;
