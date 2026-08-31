//! Shared-pool release completion and live-resize coordination.

use super::*;
use crate::topology::validate_shard;

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

/// Participant process applying one phase of a live shared-pool resize.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "role")]
pub enum KvResizeActor {
    Scheduler,
    Worker { shard: KvShard },
}

impl KvResizeActor {
    fn validate(&self) -> Result<(), KvContractError> {
        match self {
            Self::Scheduler => Ok(()),
            Self::Worker { shard } => validate_shard(*shard),
        }
    }
}

/// Ordered resize phase. Growth maps workers before exposing scheduler blocks;
/// shrinkage retires scheduler blocks before workers unmap physical pages.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum KvPoolResizeStage {
    MapWorkers,
    ActivateScheduler,
    RetireScheduler,
    UnmapWorkers,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct KvPoolResizeOperation {
    pub participant_epoch: u64,
    pub resize_generation: u64,
    pub binding_id: String,
    pub stage: KvPoolResizeStage,
    pub from_block_count: u64,
    pub target_block_count: u64,
    pub bytes_per_block: u64,
    pub allocation_granularity_bytes: u64,
    #[serde(default)]
    pub segments: Vec<KvVmmSegmentDescriptor>,
}

impl KvPoolResizeOperation {
    pub fn validate(&self) -> Result<(), KvContractError> {
        if self.participant_epoch == 0
            || self.resize_generation == 0
            || self.binding_id.trim().is_empty()
            || self.from_block_count == 0
            || self.target_block_count == 0
            || self.from_block_count == self.target_block_count
            || self.bytes_per_block == 0
            || self.allocation_granularity_bytes == 0
        {
            return Err(KvContractError::invalid_request(
                "resize operations require an epoch, generation, binding, and distinct non-zero block counts",
            ));
        }
        let growing = self.target_block_count > self.from_block_count;
        let stage_matches_direction = matches!(
            (growing, self.stage),
            (true, KvPoolResizeStage::MapWorkers)
                | (true, KvPoolResizeStage::ActivateScheduler)
                | (false, KvPoolResizeStage::RetireScheduler)
                | (false, KvPoolResizeStage::UnmapWorkers)
        );
        if !stage_matches_direction {
            return Err(KvContractError::invalid_request(
                "resize stage ordering does not match its growth or shrink direction",
            ));
        }
        let from_bytes = self
            .from_block_count
            .checked_mul(self.bytes_per_block)
            .ok_or_else(|| KvContractError::invalid_request("resize byte size overflow"))?;
        let target_bytes = self
            .target_block_count
            .checked_mul(self.bytes_per_block)
            .ok_or_else(|| KvContractError::invalid_request("resize byte size overflow"))?;
        if from_bytes % self.allocation_granularity_bytes != 0
            || target_bytes % self.allocation_granularity_bytes != 0
        {
            return Err(KvContractError::invalid_request(
                "resize endpoints must align to CUDA VMM allocation granularity",
            ));
        }
        let physical_stage = matches!(
            self.stage,
            KvPoolResizeStage::MapWorkers | KvPoolResizeStage::UnmapWorkers
        );
        if physical_stage == self.segments.is_empty() {
            return Err(KvContractError::invalid_request(
                "worker resize phases require segment ranges and scheduler phases must not carry them",
            ));
        }
        let expected_start = from_bytes.min(target_bytes);
        let expected_end = from_bytes.max(target_bytes);
        let mut ordered = self.segments.iter().collect::<Vec<_>>();
        ordered.sort_by_key(|segment| segment.offset_bytes);
        let mut expected_offset = expected_start;
        let mut segment_ids = BTreeSet::new();
        let mut handle_indices = BTreeSet::new();
        for segment in ordered {
            segment.validate(self.allocation_granularity_bytes)?;
            if segment.offset_bytes != expected_offset
                || !segment_ids.insert(segment.segment_id.as_str())
                || !handle_indices.insert(segment.handle_index)
            {
                return Err(KvContractError::invalid_request(
                    "resize segments must uniquely and densely cover the changed physical tail",
                ));
            }
            expected_offset = expected_offset
                .checked_add(segment.length_bytes)
                .ok_or_else(|| KvContractError::invalid_request("resize segment overflow"))?;
        }
        if physical_stage && expected_offset != expected_end {
            return Err(KvContractError::invalid_request(
                "resize segments must exactly cover the changed physical tail",
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct KvResizePollRequest {
    pub participant_epoch: u64,
    pub actor: KvResizeActor,
    /// Last resize generation fully applied by this process. Zero means none.
    pub applied_generation: u64,
}

impl KvResizePollRequest {
    pub fn validate(&self) -> Result<(), KvContractError> {
        if self.participant_epoch == 0 {
            return Err(KvContractError::invalid_request(
                "resize poll requires a participant epoch",
            ));
        }
        self.actor.validate()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct KvResizeAckRequest {
    pub participant_epoch: u64,
    pub actor: KvResizeActor,
    pub binding_id: String,
    pub resize_generation: u64,
    pub stage: KvPoolResizeStage,
    pub applied_block_count: u64,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct KvResizePollResult {
    /// True while the coordinator is waiting for another actor or stage even
    /// when this poll has no immediately applicable operation.
    pub pending: bool,
    #[serde(default)]
    pub operations: Vec<KvPoolResizeOperation>,
}

impl KvResizePollResult {
    pub fn validate(&self) -> Result<(), KvContractError> {
        if !self.pending && !self.operations.is_empty() {
            return Err(KvContractError::invalid_request(
                "resize operations require a pending resize transaction",
            ));
        }
        for operation in &self.operations {
            operation.validate()?;
        }
        Ok(())
    }
}

impl KvResizeAckRequest {
    pub fn validate(&self) -> Result<(), KvContractError> {
        if self.participant_epoch == 0
            || self.binding_id.trim().is_empty()
            || self.resize_generation == 0
            || self.applied_block_count == 0
        {
            return Err(KvContractError::invalid_request(
                "resize acknowledgement requires an epoch, binding, generation, and applied block count",
            ));
        }
        self.actor.validate()
    }
}
