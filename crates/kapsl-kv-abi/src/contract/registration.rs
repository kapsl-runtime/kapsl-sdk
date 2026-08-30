//! KV participant registration and capability consistency.

use super::*;

/// Registration document exchanged when a backend joins a Kapsl runtime.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct KvParticipantRegistration {
    pub participant_id: String,
    /// Stable backend family identifier such as `llama.cpp`, `vllm`, or `tgi`.
    pub backend: String,
    /// Model identity is required even in opaque mode so leases and prefix
    /// domains can never alias across models with similar geometry.
    pub model_fingerprint: String,
    pub capabilities: KvBackendCapabilities,
    pub capacity_model: KvCapacityModel,
    /// Exact build/profile identity checked before an external shared pool is
    /// provisioned, then repeated in each worker attachment.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub adapter_profile: Option<KvAdapterProfile>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub topology: Option<KvTopology>,
    /// Exact pre-start reservation proof for runtime-owned external pools.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provisioning_grant: Option<KvProvisioningGrant>,
}

impl KvParticipantRegistration {
    pub fn validate(&self) -> Result<(), KvContractError> {
        if self.participant_id.trim().is_empty()
            || self.backend.trim().is_empty()
            || self.model_fingerprint.trim().is_empty()
        {
            return Err(KvContractError::InvalidRequest {
                message: "participant_id, backend, and model_fingerprint must not be empty"
                    .to_string(),
            });
        }
        self.capabilities.validate()?;
        self.capacity_model.validate()?;
        if self.capabilities.tier == KvIntegrationTier::UnmanagedEndpoint {
            return Err(KvContractError::invalid_capabilities(
                "unmanaged endpoints do not register as KV participants",
            ));
        }
        if let Some(profile) = &self.adapter_profile {
            profile.validate()?;
        }
        if let Some(grant) = &self.provisioning_grant {
            grant.validate()?;
        }
        let advertises_grant = self
            .capabilities
            .features
            .contains(&KvFeature::ProvisioningGrant);
        if advertises_grant != self.provisioning_grant.is_some() {
            return Err(KvContractError::invalid_capabilities(
                "provisioning_grant capability and registration proof must be present together",
            ));
        }
        if self.provisioning_grant.is_some()
            && (self.capabilities.tier != KvIntegrationTier::SharedPool
                || self.capabilities.ownership != KvCacheOwnership::KapslRuntime
                || !self
                    .capabilities
                    .features
                    .contains(&KvFeature::ExternalPoolAttachment))
        {
            return Err(KvContractError::invalid_capabilities(
                "provisioning grants require an externally attached runtime-owned shared pool",
            ));
        }
        if self.capabilities.tier != KvIntegrationTier::SharedPool && self.adapter_profile.is_some()
        {
            return Err(KvContractError::invalid_capabilities(
                "adapter_profile is valid only for shared-pool participants",
            ));
        }
        if self.capabilities.tier == KvIntegrationTier::SharedPool {
            let has_external_transport = self
                .capabilities
                .transports
                .iter()
                .any(|transport| !matches!(transport, KvTransport::InProcess));
            if has_external_transport
                && !self
                    .capabilities
                    .features
                    .contains(&KvFeature::ExternalPoolAttachment)
            {
                return Err(KvContractError::invalid_capabilities(
                    "out-of-process shared pools require the external_pool_attachment feature",
                ));
            }
            if has_external_transport && self.adapter_profile.is_none() {
                return Err(KvContractError::invalid_capabilities(
                    "out-of-process shared pools require an adapter_profile before provisioning",
                ));
            }
            let mut pools = BTreeMap::<&str, (u64, u64)>::new();
            for group in &self.capacity_model.groups {
                let maximum = group.max_allocations.ok_or_else(|| {
                    KvContractError::invalid_capabilities(format!(
                        "shared-pool capacity group '{}' requires max_allocations",
                        group.group_id
                    ))
                })?;
                let shape = (group.bytes_per_allocation, maximum);
                if pools
                    .insert(group.pool_id.as_str(), shape)
                    .is_some_and(|existing| existing != shape)
                {
                    return Err(KvContractError::invalid_capabilities(format!(
                        "shared-pool groups aliasing '{}' must use the same physical block size and count",
                        group.pool_id
                    )));
                }
            }
        }
        if let Some(topology) = &self.topology {
            topology.validate()?;
            if topology.abi_version != self.capabilities.abi_version {
                return Err(KvContractError::VersionMismatch {
                    host: self.capabilities.abi_version,
                    participant: topology.abi_version,
                });
            }
            if topology.model_fingerprint != self.model_fingerprint {
                return Err(KvContractError::invalid_topology(
                    "registration and topology model fingerprints must match",
                ));
            }
            let topology_groups = topology
                .cache_groups
                .iter()
                .map(|group| group.group_id.as_str())
                .collect::<BTreeSet<_>>();
            let capacity_groups = self
                .capacity_model
                .groups
                .iter()
                .map(|group| group.group_id.as_str())
                .collect::<BTreeSet<_>>();
            if topology_groups != capacity_groups {
                return Err(KvContractError::invalid_topology(
                    "topology and capacity model must describe the same cache groups",
                ));
            }
            if self.capabilities.metadata_mode != KvMetadataMode::Structured {
                return Err(KvContractError::invalid_capabilities(
                    "a topology may only be advertised in structured metadata mode",
                ));
            }
            if topology.cache_groups.len() > 1
                && !self
                    .capabilities
                    .features
                    .contains(&KvFeature::MultipleCacheGroups)
            {
                return Err(KvContractError::invalid_capabilities(
                    "multiple topology groups require the multiple_cache_groups feature",
                ));
            }
        } else if self.capabilities.metadata_mode == KvMetadataMode::Structured {
            return Err(KvContractError::invalid_capabilities(
                "structured metadata mode requires a topology",
            ));
        }
        Ok(())
    }
}
