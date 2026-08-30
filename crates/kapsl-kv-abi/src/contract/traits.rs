//! Coordinator and participant behavioral contracts.

use super::*;

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

    fn poll_resize(
        &self,
        _participant_id: &str,
        _request: &KvResizePollRequest,
    ) -> Result<KvResizePollResult, KvContractError> {
        Err(KvContractError::unsupported("poll_shared_pool_resize"))
    }

    fn ack_resize(
        &self,
        _participant_id: &str,
        _request: &KvResizeAckRequest,
    ) -> Result<(), KvContractError> {
        Err(KvContractError::unsupported("ack_shared_pool_resize"))
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
