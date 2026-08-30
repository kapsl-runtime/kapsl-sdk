//! ABI versioning and backend capability negotiation.

use super::*;

/// Version implemented by this crate.
pub const KAPSL_KV_ABI_VERSION: KvAbiVersion = KvAbiVersion::new(1, 5);

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
    /// A runtime-owned shared pool exposes a stable virtual address while its
    /// mapped physical tail can grow and shrink at cache-block boundaries.
    LivePoolResize,
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
    /// CUDA virtual-memory allocations exported as OS handles. The JSON
    /// contract carries only segment metadata; handles travel out-of-band.
    CudaVmm,
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

    /// Out-of-process shared pool with a stable CUDA virtual address and
    /// runtime-coordinated physical tail resizing.
    pub fn cuda_vmm_shared_pool() -> Self {
        let mut capabilities = Self::cuda_ipc_shared_pool();
        capabilities.features.insert(KvFeature::LivePoolResize);
        capabilities.transports.remove(&KvTransport::CudaIpc);
        capabilities.transports.insert(KvTransport::CudaVmm);
        capabilities
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

        if self.features.contains(&KvFeature::LivePoolResize)
            && (self.tier != KvIntegrationTier::SharedPool
                || self.ownership != KvCacheOwnership::KapslRuntime
                || !self.features.contains(&KvFeature::ExternalPoolAttachment)
                || !self.transports.contains(&KvTransport::CudaVmm))
        {
            return Err(KvContractError::invalid_capabilities(
                "live pool resize requires an externally attached runtime-owned CUDA VMM shared pool",
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
    pub(crate) fn is_direct(&self) -> bool {
        matches!(
            self,
            Self::InProcess | Self::CudaIpc | Self::CudaVmm | Self::Nixl | Self::Custom { .. }
        )
    }
}
