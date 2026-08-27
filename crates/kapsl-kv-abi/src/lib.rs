//! Backend-neutral control and data-plane contract for KV-cache participants.
//!
//! The contract deliberately separates inference compatibility from KV
//! integration. An OpenAI-compatible endpoint can be routed by Kapsl without
//! implementing this crate, but it remains an unmanaged endpoint. Official
//! Kapsl backends advertise at least [`KvIntegrationTier::KvConnected`].
//!
//! Opaque mode is a first-class connected mode: the backend may keep its
//! physical block layout private while Kapsl still controls capacity leases,
//! admission, lifecycle, and global budgets. [`KvIntegrationTier::SharedPool`]
//! is the deepest mode and lets backend attention consume Kapsl-owned blocks.

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use thiserror::Error;

/// Version implemented by this crate.
pub const KAPSL_KV_ABI_VERSION: KvAbiVersion = KvAbiVersion::new(1, 4);

/// Semantic version of the KV participant contract.
///
/// A host accepts a participant when the major versions match and the host's
/// minor version is at least the participant's minor version.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct KvAbiVersion {
    pub major: u16,
    pub minor: u16,
}

impl KvAbiVersion {
    pub const fn new(major: u16, minor: u16) -> Self {
        Self { major, minor }
    }

    /// Return whether `self`, as the host, can accept `participant`.
    pub const fn accepts(self, participant: Self) -> bool {
        self.major == participant.major && self.minor >= participant.minor
    }
}

/// Depth of a backend's integration with Kapsl's KV memory authority.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum KvIntegrationTier {
    /// Inference is routable, but Kapsl cannot coordinate the backend's KV.
    UnmanagedEndpoint,
    /// Kapsl controls KV policy/lifecycle while the backend may own the layout.
    KvConnected,
    /// Backend attention directly consumes blocks owned by the Kapsl runtime.
    SharedPool,
}

impl KvIntegrationTier {
    /// Whether this tier satisfies a deployment's minimum integration depth.
    pub const fn satisfies(self, minimum: Self) -> bool {
        self as u8 >= minimum as u8
    }
}

/// How much physical cache metadata the backend exposes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum KvMetadataMode {
    /// No KV participant contract exists.
    Unavailable,
    /// Capacity and lifecycle are visible; physical layout and handles are not.
    Opaque,
    /// Cache groups, geometry, placement, and block handles are described.
    Structured,
}

/// Authority that owns live KV capacity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum KvCacheOwnership {
    None,
    Backend,
    KapslRuntime,
}

/// Optional operations implemented by a KV participant.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum KvFeature {
    CapacityLeasing,
    PrefixLookup,
    Eviction,
    Restore,
    CrossDeviceMigration,
    MultipleCacheGroups,
    AsyncTransfer,
    LayerwiseTransfer,
    DirectAttentionAccess,
    /// The participant suballocates logical blocks inside a runtime-owned
    /// physical pool. Kapsl still owns the allocation and aggregate admission,
    /// but lease responses do not prescribe backend block-table indices.
    ParticipantBlockSelection,
    /// An out-of-process shared-pool participant reports imported tensor views
    /// and must be activated before Kapsl grants request leases.
    ExternalPoolAttachment,
    /// The runtime provisioned this participant from an exact, single-use
    /// MemoryAuthority grant created before the backend process started.
    ProvisioningGrant,
}

/// Data-plane mechanism used to identify or transfer KV blocks.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum KvTransport {
    /// Backend-private handles. Kapsl never dereferences them.
    BackendOpaque,
    /// Direct callbacks and Kapsl-owned pointers in the same process.
    InProcess,
    /// CUDA inter-process memory/event handles.
    CudaIpc,
    /// NIXL-backed transfer.
    Nixl,
    /// Host or pinned-host staging copies.
    HostStaging,
    /// A backend-specific transport negotiated by name.
    Custom { name: String },
}

/// Capability advertisement returned by every inference engine.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct KvBackendCapabilities {
    pub abi_version: KvAbiVersion,
    pub tier: KvIntegrationTier,
    pub metadata_mode: KvMetadataMode,
    pub ownership: KvCacheOwnership,
    #[serde(default)]
    pub features: BTreeSet<KvFeature>,
    #[serde(default)]
    pub transports: BTreeSet<KvTransport>,
}

impl KvBackendCapabilities {
    /// Compatibility-only backend with no Kapsl KV control.
    pub fn unmanaged() -> Self {
        Self {
            abi_version: KAPSL_KV_ABI_VERSION,
            tier: KvIntegrationTier::UnmanagedEndpoint,
            metadata_mode: KvMetadataMode::Unavailable,
            ownership: KvCacheOwnership::None,
            features: BTreeSet::new(),
            transports: BTreeSet::new(),
        }
    }

    /// Connected backend whose physical cache representation remains private.
    pub fn opaque_connected() -> Self {
        Self {
            abi_version: KAPSL_KV_ABI_VERSION,
            tier: KvIntegrationTier::KvConnected,
            metadata_mode: KvMetadataMode::Opaque,
            ownership: KvCacheOwnership::Backend,
            features: BTreeSet::from([KvFeature::CapacityLeasing]),
            transports: BTreeSet::from([KvTransport::BackendOpaque]),
        }
    }

    /// Backend that directly consumes a runtime-owned KV pool in-process.
    pub fn in_process_shared_pool() -> Self {
        Self {
            abi_version: KAPSL_KV_ABI_VERSION,
            tier: KvIntegrationTier::SharedPool,
            metadata_mode: KvMetadataMode::Structured,
            ownership: KvCacheOwnership::KapslRuntime,
            features: BTreeSet::from([
                KvFeature::CapacityLeasing,
                KvFeature::DirectAttentionAccess,
            ]),
            transports: BTreeSet::from([KvTransport::InProcess]),
        }
    }

    /// Out-of-process backend that imports isolated runtime-owned CUDA pools.
    pub fn cuda_ipc_shared_pool() -> Self {
        Self {
            abi_version: KAPSL_KV_ABI_VERSION,
            tier: KvIntegrationTier::SharedPool,
            metadata_mode: KvMetadataMode::Structured,
            ownership: KvCacheOwnership::KapslRuntime,
            features: BTreeSet::from([
                KvFeature::CapacityLeasing,
                KvFeature::DirectAttentionAccess,
                KvFeature::ExternalPoolAttachment,
            ]),
            transports: BTreeSet::from([KvTransport::CudaIpc]),
        }
    }

    pub fn with_feature(mut self, feature: KvFeature) -> Self {
        self.features.insert(feature);
        self
    }

    pub fn with_transport(mut self, transport: KvTransport) -> Self {
        self.transports.insert(transport);
        self
    }

    pub fn validate(&self) -> Result<(), KvContractError> {
        if !KAPSL_KV_ABI_VERSION.accepts(self.abi_version) {
            return Err(KvContractError::VersionMismatch {
                host: KAPSL_KV_ABI_VERSION,
                participant: self.abi_version,
            });
        }

        if self
            .features
            .contains(&KvFeature::ParticipantBlockSelection)
            && self.tier != KvIntegrationTier::SharedPool
        {
            return Err(KvContractError::invalid_capabilities(
                "participant block selection is valid only for shared-pool backends",
            ));
        }

        if self.features.contains(&KvFeature::ExternalPoolAttachment)
            && self.tier != KvIntegrationTier::SharedPool
        {
            return Err(KvContractError::invalid_capabilities(
                "external pool attachment is valid only for shared-pool backends",
            ));
        }

        match self.tier {
            KvIntegrationTier::UnmanagedEndpoint => {
                if self.metadata_mode != KvMetadataMode::Unavailable
                    || self.ownership != KvCacheOwnership::None
                    || !self.features.is_empty()
                    || !self.transports.is_empty()
                {
                    return Err(KvContractError::invalid_capabilities(
                        "unmanaged endpoints cannot advertise KV ownership, metadata, features, or transports",
                    ));
                }
            }
            KvIntegrationTier::KvConnected => {
                if self.metadata_mode == KvMetadataMode::Unavailable
                    || self.ownership == KvCacheOwnership::None
                    || !self.features.contains(&KvFeature::CapacityLeasing)
                    || self.transports.is_empty()
                {
                    return Err(KvContractError::invalid_capabilities(
                        "KV-connected backends require metadata, an owner, capacity leasing, and a transport",
                    ));
                }
            }
            KvIntegrationTier::SharedPool => {
                if self.metadata_mode != KvMetadataMode::Structured
                    || self.ownership != KvCacheOwnership::KapslRuntime
                    || !self.features.contains(&KvFeature::CapacityLeasing)
                    || !self.features.contains(&KvFeature::DirectAttentionAccess)
                    || !self.transports.iter().any(KvTransport::is_direct)
                {
                    return Err(KvContractError::invalid_capabilities(
                        "shared-pool backends require structured metadata, runtime ownership, capacity leasing, direct attention access, and a direct transport",
                    ));
                }
            }
        }

        Ok(())
    }
}

impl Default for KvBackendCapabilities {
    fn default() -> Self {
        Self::unmanaged()
    }
}

impl KvTransport {
    fn is_direct(&self) -> bool {
        matches!(
            self,
            Self::InProcess | Self::CudaIpc | Self::Nixl | Self::Custom { .. }
        )
    }
}

/// Element type stored by a cache group.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum KvElementType {
    F16,
    Bf16,
    F32,
    I8,
    Fp8E4m3,
    Custom { name: String },
}

/// Tensor layout of a structured cache group.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum KvTensorLayout {
    /// Separate K/V planes laid out as layer, sequence block, head, dimension.
    LayerSequenceHeadDim,
    /// Paged-attention block, K/V plane, head, token, dimension.
    BlockKvHeadTokenDim,
    /// A stable backend-defined layout identifier.
    BackendNative { layout_id: String },
}

/// Geometry of one independently managed cache group.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum KvCacheGeometry {
    PagedAttention {
        block_size_tokens: u32,
        kv_heads: u32,
        key_head_dim: u32,
        value_head_dim: u32,
        element_type: KvElementType,
        layout: KvTensorLayout,
    },
    RecurrentState {
        state_bytes_per_sequence: u64,
        element_type: KvElementType,
        layout: KvTensorLayout,
    },
    /// Geometry hints for a KV-connected backend using opaque metadata.
    Opaque {
        layout_id: String,
        block_size_tokens: Option<u32>,
        bytes_per_block: Option<u64>,
    },
}

/// Attention/state policy associated with a cache group.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum KvCachePolicy {
    FullAttention,
    SlidingWindow { window_tokens: u32 },
    Recurrent,
    BackendDefined { policy_id: String },
}

/// Stable logical identity for a layer in a cache group.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct KvLayerId {
    pub index: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

impl KvLayerId {
    pub fn indexed(index: u32) -> Self {
        Self { index, name: None }
    }
}

/// Tensor/pipeline shard that owns the described cache groups.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct KvShard {
    pub tensor_parallel_rank: u32,
    pub tensor_parallel_world_size: u32,
    pub pipeline_parallel_rank: u32,
    pub pipeline_parallel_world_size: u32,
}

impl Default for KvShard {
    fn default() -> Self {
        Self {
            tensor_parallel_rank: 0,
            tensor_parallel_world_size: 1,
            pipeline_parallel_rank: 0,
            pipeline_parallel_world_size: 1,
        }
    }
}

/// One cache group. Groups may use different geometries (for example full
/// attention, sliding-window attention, and recurrent state in one model).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct KvCacheGroup {
    pub group_id: String,
    pub layers: Vec<KvLayerId>,
    pub geometry: KvCacheGeometry,
    pub policy: KvCachePolicy,
}

/// Logical cache topology for one model replica/shard.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct KvTopology {
    pub abi_version: KvAbiVersion,
    pub model_fingerprint: String,
    #[serde(default)]
    pub shard: KvShard,
    pub cache_groups: Vec<KvCacheGroup>,
}

impl KvTopology {
    pub fn validate(&self) -> Result<(), KvContractError> {
        if !KAPSL_KV_ABI_VERSION.accepts(self.abi_version) {
            return Err(KvContractError::VersionMismatch {
                host: KAPSL_KV_ABI_VERSION,
                participant: self.abi_version,
            });
        }
        if self.model_fingerprint.trim().is_empty() {
            return Err(KvContractError::invalid_topology(
                "model_fingerprint must not be empty",
            ));
        }
        validate_shard(self.shard)?;
        if self.cache_groups.is_empty() {
            return Err(KvContractError::invalid_topology(
                "at least one cache group is required",
            ));
        }

        let mut group_ids = BTreeSet::new();
        for group in &self.cache_groups {
            if group.group_id.trim().is_empty() || !group_ids.insert(group.group_id.as_str()) {
                return Err(KvContractError::invalid_topology(
                    "cache group IDs must be non-empty and unique",
                ));
            }
            if group.layers.is_empty() {
                return Err(KvContractError::invalid_topology(format!(
                    "cache group '{}' has no layers",
                    group.group_id
                )));
            }
            let mut layer_indices = BTreeSet::new();
            for layer in &group.layers {
                if layer
                    .name
                    .as_ref()
                    .is_some_and(|name| name.trim().is_empty())
                    || !layer_indices.insert(layer.index)
                {
                    return Err(KvContractError::invalid_topology(
                        "layer indices must be unique and layer names cannot be empty",
                    ));
                }
            }
            validate_geometry(&group.geometry)?;
            validate_policy(&group.policy)?;
            validate_geometry_policy(&group.geometry, &group.policy)?;
        }
        Ok(())
    }
}

fn validate_shard(shard: KvShard) -> Result<(), KvContractError> {
    if shard.tensor_parallel_world_size == 0
        || shard.pipeline_parallel_world_size == 0
        || shard.tensor_parallel_rank >= shard.tensor_parallel_world_size
        || shard.pipeline_parallel_rank >= shard.pipeline_parallel_world_size
    {
        return Err(KvContractError::invalid_topology(
            "parallel world sizes must be non-zero and ranks must be in range",
        ));
    }
    Ok(())
}

fn validate_geometry(geometry: &KvCacheGeometry) -> Result<(), KvContractError> {
    match geometry {
        KvCacheGeometry::PagedAttention {
            block_size_tokens,
            kv_heads,
            key_head_dim,
            value_head_dim,
            element_type,
            layout,
        } => {
            if *block_size_tokens == 0
                || *kv_heads == 0
                || *key_head_dim == 0
                || *value_head_dim == 0
            {
                return Err(KvContractError::invalid_topology(
                    "paged-attention dimensions must be non-zero",
                ));
            }
            validate_element_type(element_type)?;
            validate_layout(layout)?;
        }
        KvCacheGeometry::RecurrentState {
            state_bytes_per_sequence,
            element_type,
            layout,
        } => {
            if *state_bytes_per_sequence == 0 {
                return Err(KvContractError::invalid_topology(
                    "recurrent state size must be non-zero",
                ));
            }
            validate_element_type(element_type)?;
            validate_layout(layout)?;
        }
        KvCacheGeometry::Opaque {
            layout_id,
            block_size_tokens,
            bytes_per_block,
        } => {
            if layout_id.trim().is_empty()
                || block_size_tokens.is_some_and(|value| value == 0)
                || bytes_per_block.is_some_and(|value| value == 0)
            {
                return Err(KvContractError::invalid_topology(
                    "opaque geometry requires a layout ID and non-zero optional hints",
                ));
            }
        }
    }
    Ok(())
}

fn validate_element_type(element_type: &KvElementType) -> Result<(), KvContractError> {
    if matches!(element_type, KvElementType::Custom { name } if name.trim().is_empty()) {
        return Err(KvContractError::invalid_topology(
            "custom element type name must not be empty",
        ));
    }
    Ok(())
}

fn validate_layout(layout: &KvTensorLayout) -> Result<(), KvContractError> {
    if matches!(layout, KvTensorLayout::BackendNative { layout_id } if layout_id.trim().is_empty())
    {
        return Err(KvContractError::invalid_topology(
            "backend-native layout ID must not be empty",
        ));
    }
    Ok(())
}

fn validate_policy(policy: &KvCachePolicy) -> Result<(), KvContractError> {
    match policy {
        KvCachePolicy::SlidingWindow { window_tokens } if *window_tokens == 0 => Err(
            KvContractError::invalid_topology("sliding window must be non-zero"),
        ),
        KvCachePolicy::BackendDefined { policy_id } if policy_id.trim().is_empty() => Err(
            KvContractError::invalid_topology("backend-defined policy ID must not be empty"),
        ),
        _ => Ok(()),
    }
}

fn validate_geometry_policy(
    geometry: &KvCacheGeometry,
    policy: &KvCachePolicy,
) -> Result<(), KvContractError> {
    match (geometry, policy) {
        (KvCacheGeometry::PagedAttention { .. }, KvCachePolicy::Recurrent)
        | (KvCacheGeometry::RecurrentState { .. }, KvCachePolicy::FullAttention)
        | (KvCacheGeometry::RecurrentState { .. }, KvCachePolicy::SlidingWindow { .. }) => Err(
            KvContractError::invalid_topology("cache geometry and policy are incompatible"),
        ),
        _ => Ok(()),
    }
}

/// Identity of one logical sequence across scheduler and worker processes.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct KvSequenceKey {
    pub request_id: String,
    pub sequence_id: String,
}

/// Content-addressed prefix identity. The hash algorithm is named so it can be
/// upgraded without silently aliasing entries.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct KvPrefixKey {
    pub namespace: String,
    pub model_fingerprint: String,
    pub hash_algorithm: String,
    pub digest: String,
    pub token_count: u32,
}

/// Capacity requested from one topology group.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct KvGroupReservation {
    pub group_id: String,
    pub token_capacity: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub minimum_blocks: Option<u32>,
}

/// Request to admit a sequence and reserve its KV capacity.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct KvReserveRequest {
    pub sequence: KvSequenceKey,
    pub groups: Vec<KvGroupReservation>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prefix: Option<KvPrefixKey>,
    #[serde(default)]
    pub priority: i32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ttl_ms: Option<u64>,
}

impl KvReserveRequest {
    pub fn validate(&self) -> Result<(), KvContractError> {
        validate_sequence(&self.sequence)?;
        if self.groups.is_empty() {
            return Err(KvContractError::invalid_request(
                "at least one cache-group reservation is required",
            ));
        }
        if self.ttl_ms.is_some_and(|ttl| ttl == 0) {
            return Err(KvContractError::invalid_request(
                "ttl_ms must be non-zero when present",
            ));
        }
        let mut groups = BTreeSet::new();
        for group in &self.groups {
            if group.group_id.trim().is_empty()
                || !groups.insert(group.group_id.as_str())
                || group.token_capacity == 0
                || group.minimum_blocks.is_some_and(|blocks| blocks == 0)
            {
                return Err(KvContractError::invalid_request(
                    "reservation group IDs must be unique and capacities must be non-zero",
                ));
            }
        }
        if let Some(prefix) = &self.prefix {
            validate_prefix(prefix)?;
        }
        Ok(())
    }
}

/// Transport-neutral reference to one physical or logical block.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum KvBlockHandle {
    RuntimePool {
        pool_id: String,
        block_index: u64,
        generation: u64,
    },
    BackendOpaque {
        namespace: String,
        handle: String,
    },
    Transport {
        transport: KvTransport,
        descriptor: String,
    },
}

/// Blocks assigned to one cache group by a lease.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct KvGroupLease {
    pub group_id: String,
    pub token_capacity: u32,
    #[serde(default)]
    pub blocks: Vec<KvBlockHandle>,
}

/// Revocable KV capacity lease issued for one sequence.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct KvLease {
    pub lease_id: String,
    pub sequence: KvSequenceKey,
    pub groups: Vec<KvGroupLease>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expires_at_unix_ms: Option<u64>,
}

impl KvLease {
    pub fn validate(&self) -> Result<(), KvContractError> {
        if self.lease_id.trim().is_empty() || self.groups.is_empty() {
            return Err(KvContractError::invalid_request(
                "a lease requires an ID and at least one cache group",
            ));
        }
        validate_sequence(&self.sequence)?;
        let mut groups = BTreeSet::new();
        for group in &self.groups {
            if group.group_id.trim().is_empty()
                || !groups.insert(group.group_id.as_str())
                || group.token_capacity == 0
            {
                return Err(KvContractError::invalid_request(
                    "lease group IDs must be unique and capacities must be non-zero",
                ));
            }
            for block in &group.blocks {
                validate_block_handle(block)?;
            }
        }
        Ok(())
    }
}

fn validate_block_handle(handle: &KvBlockHandle) -> Result<(), KvContractError> {
    let valid = match handle {
        KvBlockHandle::RuntimePool {
            pool_id,
            generation,
            ..
        } => !pool_id.trim().is_empty() && *generation != 0,
        KvBlockHandle::BackendOpaque { namespace, handle } => {
            !namespace.trim().is_empty() && !handle.trim().is_empty()
        }
        KvBlockHandle::Transport {
            transport,
            descriptor,
        } => {
            !matches!(transport, KvTransport::BackendOpaque)
                && !descriptor.trim().is_empty()
                && !matches!(transport, KvTransport::Custom { name } if name.trim().is_empty())
        }
    };
    if !valid {
        return Err(KvContractError::invalid_request(
            "KV block handles require a valid namespace/pool/transport descriptor",
        ));
    }
    Ok(())
}

fn validate_sequence(sequence: &KvSequenceKey) -> Result<(), KvContractError> {
    if sequence.request_id.trim().is_empty() || sequence.sequence_id.trim().is_empty() {
        return Err(KvContractError::invalid_request(
            "request_id and sequence_id must not be empty",
        ));
    }
    Ok(())
}

fn validate_prefix(prefix: &KvPrefixKey) -> Result<(), KvContractError> {
    if prefix.namespace.trim().is_empty()
        || prefix.model_fingerprint.trim().is_empty()
        || prefix.hash_algorithm.trim().is_empty()
        || prefix.digest.trim().is_empty()
        || prefix.token_count == 0
    {
        return Err(KvContractError::invalid_request(
            "prefix identity fields and token_count must be non-empty",
        ));
    }
    Ok(())
}

/// Commit newly computed tokens and optionally publish a reusable prefix.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct KvCommitRequest {
    pub lease_id: String,
    pub computed_tokens: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prefix: Option<KvPrefixKey>,
}

impl KvCommitRequest {
    pub fn validate(&self) -> Result<(), KvContractError> {
        if self.lease_id.trim().is_empty() {
            return Err(KvContractError::invalid_request(
                "commit lease_id must not be empty",
            ));
        }
        if let Some(prefix) = &self.prefix {
            validate_prefix(prefix)?;
            if prefix.token_count > self.computed_tokens {
                return Err(KvContractError::invalid_request(
                    "committed prefix token_count cannot exceed computed_tokens",
                ));
            }
        }
        Ok(())
    }
}

/// Result of a prefix lookup before admission.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct KvPrefixMatch {
    pub matched_tokens: u32,
    #[serde(default)]
    pub group_ids: Vec<String>,
}

/// Storage target for an eviction operation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum KvStorageTier {
    Host,
    Device { device_id: u32 },
    Remote { location: String },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct KvEvictRequest {
    pub lease_id: String,
    pub target: KvStorageTier,
    #[serde(default)]
    pub group_ids: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct KvRestoreRequest {
    pub lease_id: String,
    #[serde(default)]
    pub group_ids: Vec<String>,
}

/// Completion state for transfers that may outlive one scheduler step.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "status")]
pub enum KvOperationStatus {
    Complete,
    Pending { ticket: String },
}

/// Physical memory domain occupied by a participant-owned KV pool.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum KvMemoryDomain {
    Host,
    HostPinned {
        provider: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        device_id: Option<u32>,
    },
    HostMapped {
        provider: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        device_id: Option<u32>,
    },
    Cuda {
        device_id: u32,
    },
    Provider {
        provider: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        device_id: Option<u32>,
    },
}

impl KvMemoryDomain {
    fn validate(&self) -> Result<(), KvContractError> {
        match self {
            Self::Host | Self::Cuda { .. } => Ok(()),
            Self::HostPinned { provider, .. }
            | Self::HostMapped { provider, .. }
            | Self::Provider { provider, .. }
                if provider.trim().is_empty() =>
            {
                Err(KvContractError::invalid_capabilities(
                    "KV memory-domain provider names must not be empty",
                ))
            }
            Self::HostPinned { .. } | Self::HostMapped { .. } | Self::Provider { .. } => Ok(()),
        }
    }
}

/// Logical-to-physical accounting for one cache group. Opaque participants
/// must expose this much information even though their block layout is private.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct KvCapacityGroup {
    pub group_id: String,
    /// Groups with the same pool ID alias one physical allocation pool. Their
    /// reservation cost is the maximum within the pool, not the sum.
    pub pool_id: String,
    /// Tokens covered by one backend allocation unit (normally one page).
    pub allocation_granularity_tokens: u32,
    /// Device bytes consumed by that allocation across every layer in the group.
    pub bytes_per_allocation: u64,
    /// Every physical domain containing this pool. The reservation is charged
    /// once per domain, which models tensor-parallel workers with one allocator
    /// pool on each device without exposing their backend-private block IDs.
    pub memory_domains: Vec<KvMemoryDomain>,
    /// Current ceiling advertised by a backend-owned allocator, or the maximum
    /// block count requested from a runtime-owned shared-pool provisioner.
    /// Required for `shared_pool` registrations.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_allocations: Option<u64>,
}

impl KvCapacityGroup {
    pub fn bytes_for_tokens(&self, tokens: u64) -> Option<u64> {
        if self.allocation_granularity_tokens == 0 || self.bytes_per_allocation == 0 {
            return None;
        }
        let granularity = u64::from(self.allocation_granularity_tokens);
        let allocations = tokens.div_ceil(granularity);
        if self
            .max_allocations
            .is_some_and(|maximum| allocations > maximum)
        {
            return None;
        }
        allocations.checked_mul(self.bytes_per_allocation)
    }

    pub fn bytes_for_reservation(&self, reservation: &KvGroupReservation) -> Option<u64> {
        if reservation.group_id != self.group_id
            || reservation.token_capacity == 0
            || self.allocation_granularity_tokens == 0
            || self.bytes_per_allocation == 0
        {
            return None;
        }
        let token_allocations = u64::from(reservation.token_capacity)
            .div_ceil(u64::from(self.allocation_granularity_tokens));
        let allocations = token_allocations.max(u64::from(reservation.minimum_blocks.unwrap_or(0)));
        if self
            .max_allocations
            .is_some_and(|maximum| allocations > maximum)
        {
            return None;
        }
        allocations.checked_mul(self.bytes_per_allocation)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct KvCapacityModel {
    pub groups: Vec<KvCapacityGroup>,
}

impl KvCapacityModel {
    pub fn validate(&self) -> Result<(), KvContractError> {
        if self.groups.is_empty() {
            return Err(KvContractError::invalid_capabilities(
                "at least one KV capacity group is required",
            ));
        }
        let mut group_ids = BTreeSet::new();
        let mut pool_domains = BTreeMap::<&str, BTreeSet<&KvMemoryDomain>>::new();
        for group in &self.groups {
            let domains = group.memory_domains.iter().collect::<BTreeSet<_>>();
            if group.group_id.trim().is_empty()
                || group.pool_id.trim().is_empty()
                || !group_ids.insert(group.group_id.as_str())
                || group.allocation_granularity_tokens == 0
                || group.bytes_per_allocation == 0
                || group.max_allocations.is_some_and(|value| value == 0)
                || domains.is_empty()
                || domains.len() != group.memory_domains.len()
            {
                return Err(KvContractError::invalid_capabilities(
                    "capacity group IDs and memory domains must be unique and accounting values must be non-zero",
                ));
            }
            for domain in &group.memory_domains {
                domain.validate()?;
            }
            if let Some(existing) = pool_domains.get(group.pool_id.as_str()) {
                if existing != &domains {
                    return Err(KvContractError::invalid_capabilities(format!(
                        "capacity groups sharing pool '{}' must name the same memory domains",
                        group.pool_id
                    )));
                }
            } else {
                pool_domains.insert(group.pool_id.as_str(), domains);
            }
        }
        Ok(())
    }

    /// Compute device bytes while honoring cache groups that alias one backend
    /// allocation pool (as vLLM's hybrid memory allocator does).
    pub fn bytes_for_reservations(&self, reservations: &[KvGroupReservation]) -> Option<u64> {
        self.bytes_by_domain_for_reservations(reservations)?
            .values()
            .try_fold(0u64, |total, bytes| total.checked_add(*bytes))
    }

    /// Compute physical bytes per authority domain while honoring cache groups
    /// that alias the same backend allocation pool.
    pub fn bytes_by_domain_for_reservations(
        &self,
        reservations: &[KvGroupReservation],
    ) -> Option<BTreeMap<KvMemoryDomain, u64>> {
        self.validate().ok()?;
        let groups = self
            .groups
            .iter()
            .map(|group| (group.group_id.as_str(), group))
            .collect::<BTreeMap<_, _>>();
        let mut pool_bytes = BTreeMap::<(KvMemoryDomain, &str), u64>::new();
        for reservation in reservations {
            let group = groups.get(reservation.group_id.as_str())?;
            let bytes = group.bytes_for_reservation(reservation)?;
            for domain in &group.memory_domains {
                pool_bytes
                    .entry((domain.clone(), group.pool_id.as_str()))
                    .and_modify(|current| *current = (*current).max(bytes))
                    .or_insert(bytes);
            }
        }
        let mut domain_bytes = BTreeMap::<KvMemoryDomain, u64>::new();
        for ((domain, _), bytes) in pool_bytes {
            let total = domain_bytes.entry(domain).or_default();
            *total = total.checked_add(bytes)?;
        }
        Some(domain_bytes)
    }
}

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
    pub imported_bytes: u64,
    pub views: Vec<KvAttachmentView>,
}

impl KvSharedPoolAttachment {
    pub fn validate(&self) -> Result<(), KvContractError> {
        if self.participant_epoch == 0
            || self.binding_id.trim().is_empty()
            || self.imported_bytes == 0
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

/// Proof that a backend has stopped accessing a shared-pool lease before its
/// physical blocks are made available to another sequence. Opaque leases do
/// not require a completion value. A transport fence is negotiated for future
/// asynchronous release paths; coordinators may reject fence kinds they cannot
/// wait on safely.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum KvReleaseCompletion {
    BackendSynchronized,
    TransportFence {
        transport: KvTransport,
        descriptor: String,
    },
}

impl KvReleaseCompletion {
    pub fn validate(&self) -> Result<(), KvContractError> {
        match self {
            Self::BackendSynchronized => Ok(()),
            Self::TransportFence {
                transport,
                descriptor,
            } if transport.is_direct()
                && !descriptor.trim().is_empty()
                && !matches!(transport, KvTransport::InProcess) =>
            {
                Ok(())
            }
            Self::TransportFence { .. } => Err(KvContractError::invalid_request(
                "release fences require a non-empty out-of-process direct transport descriptor",
            )),
        }
    }
}

/// One newline-delimited JSON request sent from a backend adapter to Kapsl's
/// KV coordinator. The envelope keeps version negotiation and correlation
/// independent of the transport (Unix socket, TCP, or an in-process codec).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct KvControlRequestEnvelope {
    pub abi_version: KvAbiVersion,
    pub request_id: String,
    #[serde(flatten)]
    pub request: KvControlRequest,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "operation")]
pub enum KvControlRequest {
    Register {
        registration: KvParticipantRegistration,
    },
    Attach {
        participant_id: String,
        attachment: KvSharedPoolAttachment,
    },
    Activate {
        participant_id: String,
        participant_epoch: u64,
    },
    Reserve {
        participant_id: String,
        request: KvReserveRequest,
    },
    Commit {
        participant_id: String,
        request: KvCommitRequest,
    },
    Touch {
        participant_id: String,
        lease_id: String,
    },
    Heartbeat {
        participant_id: String,
    },
    Release {
        participant_id: String,
        lease_id: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        completion: Option<KvReleaseCompletion>,
    },
    Detach {
        participant_id: String,
        request: KvSharedPoolDetachRequest,
    },
}

impl KvControlRequest {
    pub fn validate(&self) -> Result<(), KvContractError> {
        match self {
            Self::Register { registration } => registration.validate(),
            Self::Attach {
                participant_id,
                attachment,
            } => {
                validate_participant_id(participant_id)?;
                attachment.validate()
            }
            Self::Activate {
                participant_id,
                participant_epoch,
            } => {
                validate_participant_id(participant_id)?;
                if *participant_epoch == 0 {
                    return Err(KvContractError::invalid_request(
                        "participant_epoch must be non-zero",
                    ));
                }
                Ok(())
            }
            Self::Reserve {
                participant_id,
                request,
            } => {
                validate_participant_id(participant_id)?;
                request.validate()
            }
            Self::Commit {
                participant_id,
                request,
            } => {
                validate_participant_id(participant_id)?;
                request.validate()
            }
            Self::Touch {
                participant_id,
                lease_id,
            } => {
                validate_participant_id(participant_id)?;
                if lease_id.trim().is_empty() {
                    return Err(KvContractError::invalid_request(
                        "lease_id must not be empty",
                    ));
                }
                Ok(())
            }
            Self::Release {
                participant_id,
                lease_id,
                completion,
            } => {
                validate_participant_id(participant_id)?;
                if lease_id.trim().is_empty() {
                    return Err(KvContractError::invalid_request(
                        "lease_id must not be empty",
                    ));
                }
                if let Some(completion) = completion {
                    completion.validate()?;
                }
                Ok(())
            }
            Self::Heartbeat { participant_id } => validate_participant_id(participant_id),
            Self::Detach {
                participant_id,
                request,
            } => {
                validate_participant_id(participant_id)?;
                request.validate()
            }
        }
    }
}

fn validate_participant_id(participant_id: &str) -> Result<(), KvContractError> {
    if participant_id.trim().is_empty() {
        Err(KvContractError::invalid_request(
            "participant_id must not be empty",
        ))
    } else {
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct KvControlResponseEnvelope {
    pub abi_version: KvAbiVersion,
    pub request_id: String,
    #[serde(flatten)]
    pub response: KvControlResponse,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "result")]
pub enum KvControlResponse {
    Registered { receipt: KvRegistrationReceipt },
    Lease { lease: KvLease },
    Ack,
    Error { error: KvContractError },
}

/// Validate and dispatch one transport-decoded control request. Transport
/// servers should always serialize the returned envelope, including contract
/// errors, so clients retain request correlation and machine-readable causes.
pub fn dispatch_control_request(
    coordinator: &(impl KvCoordinator + ?Sized),
    envelope: KvControlRequestEnvelope,
) -> KvControlResponseEnvelope {
    let request_id = envelope.request_id;
    let response = if request_id.trim().is_empty() {
        Err(KvContractError::invalid_request(
            "control request_id must not be empty",
        ))
    } else if !KAPSL_KV_ABI_VERSION.accepts(envelope.abi_version) {
        Err(KvContractError::VersionMismatch {
            host: KAPSL_KV_ABI_VERSION,
            participant: envelope.abi_version,
        })
    } else if let Err(error) = envelope.request.validate() {
        Err(error)
    } else {
        match envelope.request {
            KvControlRequest::Register { registration } => coordinator
                .register(&registration)
                .map(|receipt| KvControlResponse::Registered { receipt }),
            KvControlRequest::Attach {
                participant_id,
                attachment,
            } => coordinator
                .attach(&participant_id, &attachment)
                .map(|()| KvControlResponse::Ack),
            KvControlRequest::Activate {
                participant_id,
                participant_epoch,
            } => coordinator
                .activate(&participant_id, participant_epoch)
                .map(|()| KvControlResponse::Ack),
            KvControlRequest::Reserve {
                participant_id,
                request,
            } => coordinator
                .reserve(&participant_id, &request)
                .map(|lease| KvControlResponse::Lease { lease }),
            KvControlRequest::Commit {
                participant_id,
                request,
            } => coordinator
                .commit(&participant_id, &request)
                .map(|()| KvControlResponse::Ack),
            KvControlRequest::Touch {
                participant_id,
                lease_id,
            } => coordinator
                .touch(&participant_id, &lease_id)
                .map(|()| KvControlResponse::Ack),
            KvControlRequest::Heartbeat { participant_id } => coordinator
                .heartbeat(&participant_id)
                .map(|()| KvControlResponse::Ack),
            KvControlRequest::Release {
                participant_id,
                lease_id,
                completion,
            } => coordinator
                .release(&participant_id, &lease_id, completion.as_ref())
                .map(|()| KvControlResponse::Ack),
            KvControlRequest::Detach {
                participant_id,
                request,
            } => coordinator
                .detach(&participant_id, &request)
                .map(|()| KvControlResponse::Ack),
        }
    }
    .unwrap_or_else(|error| KvControlResponse::Error { error });

    KvControlResponseEnvelope {
        abi_version: KAPSL_KV_ABI_VERSION,
        request_id,
        response,
    }
}

/// Backend-facing half of the contract, implemented by Kapsl's device-wide KV
/// authority. Out-of-process adapters call the same operations through the
/// control envelopes above.
pub trait KvCoordinator: Send + Sync {
    fn register(
        &self,
        registration: &KvParticipantRegistration,
    ) -> Result<KvRegistrationReceipt, KvContractError>;

    fn attach(
        &self,
        _participant_id: &str,
        _attachment: &KvSharedPoolAttachment,
    ) -> Result<(), KvContractError> {
        Err(KvContractError::unsupported("attach_shared_pool"))
    }

    fn activate(
        &self,
        _participant_id: &str,
        _participant_epoch: u64,
    ) -> Result<(), KvContractError> {
        Err(KvContractError::unsupported("activate_shared_pool"))
    }

    fn reserve(
        &self,
        participant_id: &str,
        request: &KvReserveRequest,
    ) -> Result<KvLease, KvContractError>;

    fn commit(
        &self,
        participant_id: &str,
        request: &KvCommitRequest,
    ) -> Result<(), KvContractError>;

    fn touch(&self, participant_id: &str, lease_id: &str) -> Result<(), KvContractError>;

    /// Renew every live lease owned by one participant in a single control
    /// operation. High-concurrency adapters should prefer this over touching
    /// each lease separately.
    fn heartbeat(&self, participant_id: &str) -> Result<(), KvContractError>;

    fn release(
        &self,
        participant_id: &str,
        lease_id: &str,
        completion: Option<&KvReleaseCompletion>,
    ) -> Result<(), KvContractError>;

    fn detach(
        &self,
        _participant_id: &str,
        _request: &KvSharedPoolDetachRequest,
    ) -> Result<(), KvContractError> {
        Err(KvContractError::unsupported("detach_shared_pool"))
    }
}

/// Runtime-facing contract implemented by a deep KV backend adapter.
///
/// Transport adapters can proxy this trait over IPC/RPC; no raw pointer is
/// part of the serialized contract. In-process shared-pool implementations map
/// [`KvBlockHandle::RuntimePool`] to validated local allocations separately.
pub trait KvParticipant: Send + Sync {
    fn registration(&self) -> KvParticipantRegistration;

    fn reserve(&self, request: &KvReserveRequest) -> Result<KvLease, KvContractError>;

    fn commit(&self, request: &KvCommitRequest) -> Result<(), KvContractError>;

    fn touch(&self, lease_id: &str) -> Result<(), KvContractError>;

    fn release(&self, lease_id: &str) -> Result<(), KvContractError>;

    fn lookup_prefix(
        &self,
        _prefix: &KvPrefixKey,
    ) -> Result<Option<KvPrefixMatch>, KvContractError> {
        Err(KvContractError::unsupported("lookup_prefix"))
    }

    fn evict(&self, _request: &KvEvictRequest) -> Result<KvOperationStatus, KvContractError> {
        Err(KvContractError::unsupported("evict"))
    }

    fn restore(&self, _request: &KvRestoreRequest) -> Result<KvOperationStatus, KvContractError> {
        Err(KvContractError::unsupported("restore"))
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Error, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum KvContractError {
    #[error("KV ABI mismatch: host {host:?}, participant {participant:?}")]
    VersionMismatch {
        host: KvAbiVersion,
        participant: KvAbiVersion,
    },
    #[error("invalid KV capabilities: {message}")]
    InvalidCapabilities { message: String },
    #[error("invalid KV topology: {message}")]
    InvalidTopology { message: String },
    #[error("invalid KV request: {message}")]
    InvalidRequest { message: String },
    #[error("KV capacity exhausted: {message}")]
    CapacityExhausted { message: String },
    #[error("KV object not found: {message}")]
    NotFound { message: String },
    #[error("KV operation '{operation}' is unsupported")]
    Unsupported { operation: String },
    #[error("KV transport error: {message}")]
    Transport { message: String },
    #[error("KV participant error: {message}")]
    Internal { message: String },
}

impl KvContractError {
    pub fn invalid_capabilities(message: impl Into<String>) -> Self {
        Self::InvalidCapabilities {
            message: message.into(),
        }
    }

    pub fn invalid_topology(message: impl Into<String>) -> Self {
        Self::InvalidTopology {
            message: message.into(),
        }
    }

    pub fn invalid_request(message: impl Into<String>) -> Self {
        Self::InvalidRequest {
            message: message.into(),
        }
    }

    pub fn unsupported(operation: impl Into<String>) -> Self {
        Self::Unsupported {
            operation: operation.into(),
        }
    }
}

#[cfg(test)]
mod tests {
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
            "../tests/fixtures/shared_pool_registration.json"
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

        let fixture: serde_json::Value =
            serde_json::from_str(include_str!("../tests/fixtures/opaque_registration.json"))
                .expect("valid connector fixture");
        assert_eq!(serde_json::to_value(registration).unwrap(), fixture);
    }

    #[test]
    fn shared_pool_registration_accepts_out_of_tree_connector_fixture() {
        let fixture: KvParticipantRegistration = serde_json::from_str(include_str!(
            "../tests/fixtures/shared_pool_registration.json"
        ))
        .expect("connector fixture must use the Rust wire shape");

        fixture.validate().expect("valid shared-pool registration");
        let topology = fixture.topology.expect("shared-pool topology");
        let KvCacheGeometry::PagedAttention { element_type, .. } =
            &topology.cache_groups[0].geometry
        else {
            panic!("fixture must contain paged-attention geometry");
        };
        assert_eq!(element_type, &KvElementType::F16);
    }
}
