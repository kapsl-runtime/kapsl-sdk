//! Sequence admission, capacity leases, and cache lifecycle requests.

use super::*;

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
