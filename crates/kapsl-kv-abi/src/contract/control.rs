//! Transport-neutral KV coordinator request and response protocol.

use super::*;

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
    ResizePoll {
        participant_id: String,
        request: KvResizePollRequest,
    },
    ResizeAck {
        participant_id: String,
        request: KvResizeAckRequest,
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
            Self::ResizePoll {
                participant_id,
                request,
            } => {
                validate_participant_id(participant_id)?;
                request.validate()
            }
            Self::ResizeAck {
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
    Registered {
        receipt: KvRegistrationReceipt,
    },
    Lease {
        lease: KvLease,
    },
    Resize {
        pending: bool,
        #[serde(default)]
        operations: Vec<KvPoolResizeOperation>,
    },
    Ack,
    Error {
        error: KvContractError,
    },
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
            KvControlRequest::ResizePoll {
                participant_id,
                request,
            } => coordinator
                .poll_resize(&participant_id, &request)
                .and_then(|result| {
                    result.validate()?;
                    Ok(KvControlResponse::Resize {
                        pending: result.pending,
                        operations: result.operations,
                    })
                }),
            KvControlRequest::ResizeAck {
                participant_id,
                request,
            } => coordinator
                .ack_resize(&participant_id, &request)
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
