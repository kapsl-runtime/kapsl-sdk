//! Runtime-owned shared-pool provisioning and lifecycle contracts.

use super::*;
use crate::topology::validate_shard;

/// One isolated runtime-owned pool mapping offered to a shared-pool
/// participant. `binding_id` identifies one physical replica (for example one
/// tensor-parallel CUDA device), while `capacity_pool_id` refers back to the
/// participant's logical capacity model.
///
/// The transport descriptor is opaque to the control plane. For CUDA IPC it
/// is the base64-encoded `CUipcMemHandle`; it must refer only to this
/// participant's allocation, never to a process-wide allocator backing.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum KvSharedPoolAllocationMode {
    /// Kapsl selects physical indices and returns generation-checked block
    /// handles in each lease.
    #[default]
    RuntimeLeased,
    /// The participant selects physical indices using its native allocator;
    /// Kapsl leases aggregate capacity and returns no physical block handles.
    ParticipantManaged,
}

/// One runtime-owned CUDA VMM allocation mapped into a stable participant
/// virtual-address range. `handle_index` identifies the POSIX file descriptor
/// transferred alongside the JSON response with `SCM_RIGHTS`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct KvVmmSegmentDescriptor {
    pub segment_id: String,
    pub offset_bytes: u64,
    pub length_bytes: u64,
    pub handle_index: u32,
}

impl KvVmmSegmentDescriptor {
    pub fn validate(&self, allocation_granularity_bytes: u64) -> Result<(), KvContractError> {
        if self.segment_id.trim().is_empty()
            || self.length_bytes == 0
            || allocation_granularity_bytes == 0
            || !self
                .offset_bytes
                .is_multiple_of(allocation_granularity_bytes)
            || !self
                .length_bytes
                .is_multiple_of(allocation_granularity_bytes)
            || self.offset_bytes.checked_add(self.length_bytes).is_none()
        {
            return Err(KvContractError::invalid_capabilities(
                "CUDA VMM segments require an ID and bounded granularity-aligned byte range",
            ));
        }
        Ok(())
    }
}

/// Elastic physical state behind one stable virtual shared-pool binding.
/// `KvSharedPoolDescriptor::block_count` is the maximum virtual block count;
/// `mapped_block_count` is the physical capacity available at registration.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct KvElasticPoolDescriptor {
    /// Smallest physical prefix that may remain mapped. This is the certified
    /// capacity for one maximum-length request, including fixed/null blocks.
    pub minimum_block_count: u64,
    pub mapped_block_count: u64,
    pub maximum_block_count: u64,
    pub allocation_granularity_bytes: u64,
    pub resize_alignment_blocks: u64,
    pub segments: Vec<KvVmmSegmentDescriptor>,
}

impl KvElasticPoolDescriptor {
    fn validate(
        &self,
        virtual_block_count: u64,
        bytes_per_block: u64,
    ) -> Result<(), KvContractError> {
        if self.minimum_block_count == 0
            || self.mapped_block_count == 0
            || self.maximum_block_count != virtual_block_count
            || self.minimum_block_count > self.mapped_block_count
            || self.mapped_block_count > self.maximum_block_count
            || self.allocation_granularity_bytes == 0
            || self.resize_alignment_blocks == 0
            || self.segments.is_empty()
            || !self
                .minimum_block_count
                .is_multiple_of(self.resize_alignment_blocks)
            || !self
                .mapped_block_count
                .is_multiple_of(self.resize_alignment_blocks)
        {
            return Err(KvContractError::invalid_capabilities(
                "elastic pool geometry requires an aligned minimum and mapped capacity within the maximum virtual capacity",
            ));
        }
        let minimum_bytes = self
            .minimum_block_count
            .checked_mul(bytes_per_block)
            .ok_or_else(|| {
                KvContractError::invalid_capabilities("elastic pool byte size overflow")
            })?;
        let mapped_bytes = self
            .mapped_block_count
            .checked_mul(bytes_per_block)
            .ok_or_else(|| {
                KvContractError::invalid_capabilities("elastic pool byte size overflow")
            })?;
        let maximum_bytes = self
            .maximum_block_count
            .checked_mul(bytes_per_block)
            .ok_or_else(|| {
                KvContractError::invalid_capabilities("elastic pool byte size overflow")
            })?;
        if !minimum_bytes.is_multiple_of(self.allocation_granularity_bytes)
            || !mapped_bytes.is_multiple_of(self.allocation_granularity_bytes)
            || !maximum_bytes.is_multiple_of(self.allocation_granularity_bytes)
        {
            return Err(KvContractError::invalid_capabilities(
                "elastic pool mapped and maximum sizes must align to CUDA VMM granularity",
            ));
        }

        let mut segment_ids = BTreeSet::new();
        let mut handle_indices = BTreeSet::new();
        let mut ordered = self.segments.iter().collect::<Vec<_>>();
        ordered.sort_by_key(|segment| segment.offset_bytes);
        let mut expected_offset = 0u64;
        for segment in ordered {
            segment.validate(self.allocation_granularity_bytes)?;
            if !segment_ids.insert(segment.segment_id.as_str())
                || !handle_indices.insert(segment.handle_index)
                || segment.offset_bytes != expected_offset
            {
                return Err(KvContractError::invalid_capabilities(
                    "elastic pool segments must have unique IDs/handles and densely cover the mapped prefix",
                ));
            }
            expected_offset = expected_offset
                .checked_add(segment.length_bytes)
                .ok_or_else(|| {
                    KvContractError::invalid_capabilities("elastic segment range overflow")
                })?;
        }
        if expected_offset != mapped_bytes {
            return Err(KvContractError::invalid_capabilities(
                "elastic pool segments must exactly cover mapped physical bytes",
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct KvSharedPoolDescriptor {
    pub binding_id: String,
    pub capacity_pool_id: String,
    pub generation: u64,
    pub group_ids: Vec<String>,
    pub memory_domain: KvMemoryDomain,
    pub block_count: u64,
    pub bytes_per_block: u64,
    #[serde(default)]
    pub allocation_mode: KvSharedPoolAllocationMode,
    pub transport: KvTransport,
    pub descriptor: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub elastic: Option<KvElasticPoolDescriptor>,
}

/// Exact backend and adapter combination claiming a shared-pool attachment.
/// A deployment may allowlist profiles that passed its conformance matrix.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct KvAdapterProfile {
    pub adapter_id: String,
    pub adapter_version: String,
    pub backend_version: String,
    pub profile_id: String,
}

/// Opaque proof that MemoryAuthority precharged an exact external KV backing.
///
/// The host keeps the authoritative participant/device/byte scope. The wire
/// value deliberately carries only replay-fencing metadata and the certified
/// geometry digest; a participant cannot use it to choose a different grant.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct KvProvisioningGrant {
    pub token: String,
    pub geometry_digest: String,
    pub authority_generation: u64,
    pub expires_at_unix_ms: u64,
}

impl KvProvisioningGrant {
    pub fn validate(&self) -> Result<(), KvContractError> {
        let digest = self
            .geometry_digest
            .strip_prefix("sha256:")
            .filter(|digest| {
                digest.len() == 64
                    && digest
                        .bytes()
                        .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
            });
        if self.token.trim().is_empty()
            || self.token.len() > 256
            || digest.is_none()
            || self.authority_generation == 0
            || self.expires_at_unix_ms == 0
        {
            return Err(KvContractError::invalid_request(
                "provisioning grant requires a bounded token, canonical sha256 geometry digest, generation, and expiry",
            ));
        }
        Ok(())
    }
}

impl KvAdapterProfile {
    pub fn validate(&self) -> Result<(), KvContractError> {
        if self.adapter_id.trim().is_empty()
            || self.adapter_version.trim().is_empty()
            || self.backend_version.trim().is_empty()
            || self.profile_id.trim().is_empty()
        {
            return Err(KvContractError::invalid_request(
                "adapter profile identity fields must not be empty",
            ));
        }
        Ok(())
    }
}

/// One backend tensor view expressed as offsets inside an imported binding.
/// Raw process-local pointers are deliberately excluded from the wire format.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct KvAttachmentView {
    pub group_id: String,
    pub layer: KvLayerId,
    pub offset_bytes: u64,
    pub length_bytes: u64,
}

impl KvAttachmentView {
    fn validate(&self) -> Result<(), KvContractError> {
        if self.group_id.trim().is_empty()
            || self
                .layer
                .name
                .as_ref()
                .is_some_and(|name| name.trim().is_empty())
            || self.length_bytes == 0
            || self.offset_bytes.checked_add(self.length_bytes).is_none()
        {
            return Err(KvContractError::invalid_request(
                "attachment views require a group, valid layer, and bounded non-zero byte range",
            ));
        }
        Ok(())
    }
}

/// Evidence reported after one worker has imported a provisioned binding and
/// constructed its backend-native KV tensor views.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct KvSharedPoolAttachment {
    pub participant_epoch: u64,
    pub binding_id: String,
    pub shard: KvShard,
    pub profile: KvAdapterProfile,
    /// Stable virtual span backing all tensor views.
    pub imported_bytes: u64,
    /// Physical prefix currently mapped into the virtual span. Absent for
    /// fixed CUDA IPC pools and required for CUDA VMM pools.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mapped_bytes: Option<u64>,
    pub views: Vec<KvAttachmentView>,
}

impl KvSharedPoolAttachment {
    pub fn validate(&self) -> Result<(), KvContractError> {
        if self.participant_epoch == 0
            || self.binding_id.trim().is_empty()
            || self.imported_bytes == 0
            || self
                .mapped_bytes
                .is_some_and(|mapped| mapped == 0 || mapped > self.imported_bytes)
            || self.views.is_empty()
        {
            return Err(KvContractError::invalid_request(
                "shared-pool attachment requires an epoch, binding, imported size, and tensor views",
            ));
        }
        validate_shard(self.shard)?;
        self.profile.validate()?;
        let mut layers = BTreeSet::new();
        for view in &self.views {
            view.validate()?;
            if view.offset_bytes + view.length_bytes > self.imported_bytes {
                return Err(KvContractError::invalid_request(
                    "attachment tensor view exceeds the imported binding",
                ));
            }
            if !layers.insert((view.group_id.as_str(), view.layer.index)) {
                return Err(KvContractError::invalid_request(
                    "attachment tensor views must identify unique group/layer pairs",
                ));
            }
        }
        Ok(())
    }
}

/// Clean worker detach after its backend has stopped using the imported pool.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct KvSharedPoolDetachRequest {
    pub participant_epoch: u64,
    pub binding_ids: Vec<String>,
    pub shard: KvShard,
    pub completion: KvReleaseCompletion,
}

impl KvSharedPoolDetachRequest {
    pub fn validate(&self) -> Result<(), KvContractError> {
        if self.participant_epoch == 0 || self.binding_ids.is_empty() {
            return Err(KvContractError::invalid_request(
                "shared-pool detach requires an epoch and at least one binding",
            ));
        }
        validate_shard(self.shard)?;
        self.completion.validate()?;
        let mut binding_ids = BTreeSet::new();
        if self.binding_ids.iter().any(|binding_id| {
            binding_id.trim().is_empty() || !binding_ids.insert(binding_id.as_str())
        }) {
            return Err(KvContractError::invalid_request(
                "detach binding IDs must be non-empty and unique",
            ));
        }
        Ok(())
    }
}

impl KvSharedPoolDescriptor {
    pub fn validate(&self) -> Result<(), KvContractError> {
        let group_ids = self
            .group_ids
            .iter()
            .map(String::as_str)
            .collect::<BTreeSet<_>>();
        if self.binding_id.trim().is_empty()
            || self.capacity_pool_id.trim().is_empty()
            || self.generation == 0
            || self.group_ids.is_empty()
            || group_ids.len() != self.group_ids.len()
            || group_ids.iter().any(|group_id| group_id.trim().is_empty())
            || self.block_count == 0
            || self.bytes_per_block == 0
            || self.descriptor.trim().is_empty()
            || !self.transport.is_direct()
        {
            return Err(KvContractError::invalid_capabilities(
                "shared-pool bindings require unique IDs, non-zero geometry, and a direct transport descriptor",
            ));
        }
        match (&self.transport, &self.elastic) {
            (KvTransport::CudaVmm, Some(elastic)) => {
                elastic.validate(self.block_count, self.bytes_per_block)?;
            }
            (KvTransport::CudaVmm, None) => {
                return Err(KvContractError::invalid_capabilities(
                    "CUDA VMM shared-pool bindings require elastic geometry",
                ));
            }
            (_, Some(_)) => {
                return Err(KvContractError::invalid_capabilities(
                    "elastic geometry is valid only for CUDA VMM shared-pool bindings",
                ));
            }
            (_, None) => {}
        }
        self.memory_domain.validate()
    }
}

/// Coordinator-issued registration result. The participant epoch changes
/// whenever a registration is replaced, preventing an adapter from silently
/// reusing pool handles from an older runtime-owned allocation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct KvRegistrationReceipt {
    pub participant_id: String,
    pub participant_epoch: u64,
    #[serde(default)]
    pub shared_pools: Vec<KvSharedPoolDescriptor>,
}

impl KvRegistrationReceipt {
    pub fn opaque(participant_id: impl Into<String>, participant_epoch: u64) -> Self {
        Self {
            participant_id: participant_id.into(),
            participant_epoch,
            shared_pools: Vec::new(),
        }
    }

    /// Validate a runtime receipt against the participant request that caused
    /// it. Every logical pool/group/domain tuple must have exactly one physical
    /// binding, and a runtime may provision no more than the advertised cap.
    pub fn validate_for(
        &self,
        registration: &KvParticipantRegistration,
    ) -> Result<(), KvContractError> {
        registration.validate()?;
        if self.participant_id != registration.participant_id || self.participant_epoch == 0 {
            return Err(KvContractError::invalid_capabilities(
                "registration receipt participant and epoch must match the live registration",
            ));
        }

        if registration.capabilities.tier != KvIntegrationTier::SharedPool {
            if !self.shared_pools.is_empty() {
                return Err(KvContractError::invalid_capabilities(
                    "only shared_pool participants may receive physical pool bindings",
                ));
            }
            return Ok(());
        }
        if self.shared_pools.is_empty() {
            return Err(KvContractError::invalid_capabilities(
                "shared_pool registration requires at least one physical pool binding",
            ));
        }

        let groups = registration
            .capacity_model
            .groups
            .iter()
            .map(|group| (group.group_id.as_str(), group))
            .collect::<BTreeMap<_, _>>();
        let mut expected = BTreeSet::new();
        for group in &registration.capacity_model.groups {
            for domain in &group.memory_domains {
                expected.insert((group.pool_id.as_str(), group.group_id.as_str(), domain));
            }
        }

        let mut binding_ids = BTreeSet::new();
        let mut pool_block_counts = BTreeMap::<&str, u64>::new();
        let participant_managed = registration
            .capabilities
            .features
            .contains(&KvFeature::ParticipantBlockSelection);
        let live_resize = registration
            .capabilities
            .features
            .contains(&KvFeature::LivePoolResize);
        for binding in &self.shared_pools {
            binding.validate()?;
            if !binding_ids.insert(binding.binding_id.as_str()) {
                return Err(KvContractError::invalid_capabilities(
                    "shared-pool binding IDs must be unique",
                ));
            }
            if !registration
                .capabilities
                .transports
                .contains(&binding.transport)
            {
                return Err(KvContractError::invalid_capabilities(format!(
                    "shared-pool binding '{}' uses an unadvertised transport",
                    binding.binding_id
                )));
            }
            if (binding.allocation_mode == KvSharedPoolAllocationMode::ParticipantManaged)
                != participant_managed
            {
                return Err(KvContractError::invalid_capabilities(format!(
                    "shared-pool binding '{}' allocation mode does not match participant block-selection capabilities",
                    binding.binding_id
                )));
            }
            if binding.elastic.is_some() != live_resize
                || matches!(binding.transport, KvTransport::CudaVmm) != live_resize
            {
                return Err(KvContractError::invalid_capabilities(format!(
                    "shared-pool binding '{}' live-resize transport does not match participant capabilities",
                    binding.binding_id
                )));
            }
            if pool_block_counts
                .insert(binding.capacity_pool_id.as_str(), binding.block_count)
                .is_some_and(|existing| existing != binding.block_count)
            {
                return Err(KvContractError::invalid_capabilities(format!(
                    "replicas of shared pool '{}' must expose the same block count",
                    binding.capacity_pool_id
                )));
            }
            for group_id in &binding.group_ids {
                let group = groups.get(group_id.as_str()).ok_or_else(|| {
                    KvContractError::invalid_capabilities(format!(
                        "shared-pool binding '{}' references unknown group '{}'",
                        binding.binding_id, group_id
                    ))
                })?;
                if group.pool_id != binding.capacity_pool_id
                    || group.bytes_per_allocation != binding.bytes_per_block
                    || group
                        .max_allocations
                        .is_none_or(|maximum| binding.block_count > maximum)
                    || !group.memory_domains.contains(&binding.memory_domain)
                    || !expected.remove(&(
                        group.pool_id.as_str(),
                        group.group_id.as_str(),
                        &binding.memory_domain,
                    ))
                {
                    return Err(KvContractError::invalid_capabilities(format!(
                        "shared-pool binding '{}' does not match group '{}' capacity and placement",
                        binding.binding_id, group_id
                    )));
                }
            }
        }
        if !expected.is_empty() {
            return Err(KvContractError::invalid_capabilities(
                "registration receipt does not bind every shared-pool group and memory domain",
            ));
        }
        Ok(())
    }
}
