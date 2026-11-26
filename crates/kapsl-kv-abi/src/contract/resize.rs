//! Shared-pool release completion and live-resize coordination.

use super::*;
use crate::shared_pool::{check_segment_coverage, checked_stripe_count, SegmentCoverageError};
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
    /// Bytes of one block, summed over all stripes of a striped binding.
    pub bytes_per_block: u64,
    pub allocation_granularity_bytes: u64,
    /// The binding's stripe count, absent for a single range. A striped
    /// resize changes every stripe by the same tail, and worker phases carry
    /// segments covering that tail in each stripe.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stripe_count: Option<u32>,
    #[serde(default)]
    pub segments: Vec<KvVmmSegmentDescriptor>,
}

impl KvPoolResizeOperation {
    /// Number of virtual ranges the resized binding backs: 1 unless striped.
    pub fn stripe_count(&self) -> u32 {
        self.stripe_count.unwrap_or(1)
    }

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
        let stripes = checked_stripe_count(self.stripe_count, self.bytes_per_block)
            .map_err(KvContractError::invalid_request)?;
        // As for the binding: whole-binding sizes must not overflow, while
        // alignment and segment offsets are per stripe.
        let stripe_bytes = |block_count: u64| {
            block_count
                .checked_mul(self.bytes_per_block)
                .map(|bytes| bytes / u64::from(stripes))
                .ok_or_else(|| KvContractError::invalid_request("resize byte size overflow"))
        };
        let from_bytes = stripe_bytes(self.from_block_count)?;
        let target_bytes = stripe_bytes(self.target_block_count)?;
        if !from_bytes.is_multiple_of(self.allocation_granularity_bytes)
            || !target_bytes.is_multiple_of(self.allocation_granularity_bytes)
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
        if !physical_stage {
            return Ok(());
        }
        check_segment_coverage(
            &self.segments,
            stripes,
            self.allocation_granularity_bytes,
            from_bytes.min(target_bytes),
            from_bytes.max(target_bytes),
        )
        .map_err(|error| match error {
            SegmentCoverageError::Segment(error) => error,
            SegmentCoverageError::UnknownStripe => KvContractError::invalid_request(
                "resize segments must name a stripe of their binding",
            ),
            SegmentCoverageError::NotDense => KvContractError::invalid_request(
                "resize segments must uniquely and densely cover the changed physical tail",
            ),
            SegmentCoverageError::Overflow => {
                KvContractError::invalid_request("resize segment overflow")
            }
            SegmentCoverageError::Incomplete => KvContractError::invalid_request(
                "resize segments must exactly cover the changed physical tail in every stripe",
            ),
        })
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
