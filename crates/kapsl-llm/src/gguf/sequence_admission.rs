//! Adapter-owned request bindings at the point the GGUF scheduler assigns a slot.
//!
//! This Rust hook lives entirely inside an integration library. The integration
//! captures the engine-issued request identity and uses its native KV ABI host
//! when a slot is assigned. Neither callback order nor a recycled slot ID is an
//! engine request identity.

use kapsl_engine_api::{
    CancellationToken, EngineError, RequestMemoryAdmission, RequestMemoryAdmissionGuard,
};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};

/// A one-shot binding from one inference request to its assigned sequence slot.
///
/// The callback runs on the scheduler thread before any KV operation for the
/// slot. It may reject admission. Its returned guard remains alive until the
/// sequence's KV use has ended, including prefill copies, cancellation and errors.
/// It must not wait for a different request to finish: multiple slots remain
/// independently active during continuous batching.
#[derive(Clone)]
pub struct GgufSequenceAdmission {
    inner: Arc<SequenceAdmissionInner>,
}

type BindSequence = dyn Fn(u32) -> Result<Box<dyn Send>, EngineError> + Send + Sync;

struct SequenceAdmissionInner {
    acquired: AtomicBool,
    acquire: Box<BindSequence>,
}

impl GgufSequenceAdmission {
    pub fn new<F, T>(acquire: F) -> Self
    where
        F: Fn(u32) -> Result<T, EngineError> + Send + Sync + 'static,
        T: Send + 'static,
    {
        Self {
            inner: Arc::new(SequenceAdmissionInner {
                acquired: AtomicBool::new(false),
                acquire: Box::new(move |slot| {
                    acquire(slot).map(|guard| Box::new(guard) as Box<dyn Send>)
                }),
            }),
        }
    }

    fn acquire(&self, slot: u32) -> Result<Box<dyn Send>, EngineError> {
        if self.inner.acquired.swap(true, Ordering::AcqRel) {
            return Err(EngineError::backend(
                "GGUF sequence admission was acquired more than once",
            ));
        }
        (self.inner.acquire)(slot)
    }
}

pub(crate) struct RequestAdmission {
    pub memory: Option<RequestMemoryAdmission>,
    pub sequence: Option<GgufSequenceAdmission>,
    pub cancellation: Option<CancellationToken>,
}

impl RequestAdmission {
    pub fn is_cancelled(&self) -> bool {
        self.cancellation
            .as_ref()
            .is_some_and(CancellationToken::is_cancelled)
    }

    pub fn acquire(&self, slot: u32) -> Result<RequestGuard, EngineError> {
        if self.is_cancelled() {
            return Err(EngineError::cancelled(
                "GGUF request cancelled before sequence admission",
            ));
        }
        let memory = self
            .memory
            .as_ref()
            .map(RequestMemoryAdmission::acquire)
            .transpose()?;
        let sequence = self
            .sequence
            .as_ref()
            .map(|hook| hook.acquire(slot))
            .transpose()?;
        let guard = RequestGuard {
            _sequence: sequence,
            _memory: memory,
            cancellation: self.cancellation.clone(),
        };
        if guard.is_cancelled() {
            return Err(EngineError::cancelled(
                "GGUF request cancelled during sequence admission",
            ));
        }
        Ok(guard)
    }
}

// Drop the sequence binding before returning the memory admission lease. Both
// survive every KV callback and are dropped before the terminal response.
pub(crate) struct RequestGuard {
    _sequence: Option<Box<dyn Send>>,
    _memory: Option<RequestMemoryAdmissionGuard>,
    cancellation: Option<CancellationToken>,
}

impl RequestGuard {
    pub fn is_cancelled(&self) -> bool {
        self.cancellation
            .as_ref()
            .is_some_and(CancellationToken::is_cancelled)
    }
}

#[cfg(test)]
#[path = "sequence_admission_tests.rs"]
mod tests;
