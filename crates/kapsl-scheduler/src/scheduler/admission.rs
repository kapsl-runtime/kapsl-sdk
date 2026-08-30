use super::QueueOverflowPolicy;
use crate::priority::Priority;
use futures::Stream;
use kapsl_engine_api::{
    BinaryTensorPacket, CancellationToken, EngineError, EngineStream, OpenAiWireStream,
};
use parking_lot::Mutex;
use std::pin::Pin;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::task::{Context, Poll};
use tokio::sync::{Notify, OwnedSemaphorePermit, Semaphore, TryAcquireError};

pub(super) struct StreamAdmissionGuard {
    counter: Arc<AtomicUsize>,
    _permit: Option<PriorityAdmissionPermit>,
}

pub(super) struct ActiveCountGuard {
    pub(super) counter: Arc<AtomicUsize>,
}

impl Drop for ActiveCountGuard {
    fn drop(&mut self) {
        self.counter.fetch_sub(1, Ordering::Release);
    }
}

impl StreamAdmissionGuard {
    pub(super) fn new(counter: Arc<AtomicUsize>, permit: Option<PriorityAdmissionPermit>) -> Self {
        counter.fetch_add(1, Ordering::Relaxed);
        Self {
            counter,
            _permit: permit,
        }
    }
}

#[derive(Default)]
struct PriorityAdmissionState {
    latency_waiters: usize,
    throughput_waiters: usize,
}

/// Bounded direct-dispatch admission with strict priority between waiting
/// classes. The semaphore bounds active work; waiter counters plus `Notify`
/// avoid an unbounded handoff queue while ensuring a queued throughput request
/// cannot take a newly released slot while any latency-critical request waits.
pub(super) struct PriorityAdmission {
    slots: Arc<Semaphore>,
    state: Mutex<PriorityAdmissionState>,
    changed: Notify,
}

impl PriorityAdmission {
    pub(super) fn new(capacity: usize) -> Arc<Self> {
        Arc::new(Self {
            slots: Arc::new(Semaphore::new(capacity)),
            state: Mutex::new(PriorityAdmissionState::default()),
            changed: Notify::new(),
        })
    }

    pub(super) async fn acquire(
        self: &Arc<Self>,
        priority: Priority,
        overflow_policy: QueueOverflowPolicy,
        cancellation: Option<&CancellationToken>,
    ) -> Result<PriorityAdmissionPermit, EngineError> {
        if cancellation.is_some_and(CancellationToken::is_cancelled) {
            return Err(EngineError::cancelled(
                "Request cancelled while awaiting admission",
            ));
        }
        match overflow_policy {
            QueueOverflowPolicy::Block => self.acquire_blocking(priority, cancellation).await,
            QueueOverflowPolicy::DropNewest => self.try_acquire().map_err(|error| match error {
                TryAcquireError::Closed => {
                    EngineError::overloaded("GPU stream slots closed".to_string())
                }
                TryAcquireError::NoPermits => EngineError::overloaded(format!(
                    "GPU stream slots full (policy={})",
                    overflow_policy.as_str()
                )),
            }),
            QueueOverflowPolicy::DropOldest => self.try_acquire().map_err(|error| match error {
                TryAcquireError::Closed => {
                    EngineError::overloaded("GPU stream slots closed".to_string())
                }
                TryAcquireError::NoPermits => EngineError::overloaded(
                    "GPU stream slots full (policy=drop_oldest; active streams cannot be evicted)"
                        .to_string(),
                ),
            }),
        }
    }

    async fn acquire_blocking(
        self: &Arc<Self>,
        priority: Priority,
        cancellation: Option<&CancellationToken>,
    ) -> Result<PriorityAdmissionPermit, EngineError> {
        let mut waiter = PriorityAdmissionWaiter::register(Arc::clone(self), priority);
        loop {
            if cancellation.is_some_and(CancellationToken::is_cancelled) {
                waiter.finish();
                return Err(EngineError::cancelled(
                    "Request cancelled while awaiting admission",
                ));
            }
            // Register the notification before checking state so a release or
            // high-priority cancellation cannot be lost between the check and
            // the await.
            let notified = self.changed.notified();
            tokio::pin!(notified);
            notified.as_mut().enable();

            let attempt = {
                let state = self.state.lock();
                let may_acquire =
                    priority == Priority::LatencyCritical || state.latency_waiters == 0;
                may_acquire.then(|| self.slots.clone().try_acquire_owned())
            };

            match attempt {
                Some(Ok(permit)) => {
                    waiter.finish();
                    return Ok(PriorityAdmissionPermit::new(Arc::clone(self), permit));
                }
                Some(Err(TryAcquireError::Closed)) => {
                    waiter.finish();
                    return Err(EngineError::overloaded(
                        "GPU stream slots closed".to_string(),
                    ));
                }
                Some(Err(TryAcquireError::NoPermits)) | None => {
                    if let Some(cancellation) = cancellation {
                        tokio::select! {
                            _ = &mut notified => {}
                            _ = cancellation.cancelled() => {
                                waiter.finish();
                                return Err(EngineError::cancelled(
                                    "Request cancelled while awaiting admission",
                                ));
                            }
                        }
                    } else {
                        notified.await;
                    }
                }
            }
        }
    }

    fn try_acquire(self: &Arc<Self>) -> Result<PriorityAdmissionPermit, TryAcquireError> {
        self.slots
            .clone()
            .try_acquire_owned()
            .map(|permit| PriorityAdmissionPermit::new(Arc::clone(self), permit))
    }

    fn unregister_waiter(&self, priority: Priority) {
        {
            let mut state = self.state.lock();
            let waiters = match priority {
                Priority::LatencyCritical => &mut state.latency_waiters,
                Priority::Throughput => &mut state.throughput_waiters,
            };
            debug_assert!(*waiters > 0, "priority admission waiter underflow");
            *waiters = waiters.saturating_sub(1);
        }
        // In particular, wake throughput waiters when the last queued latency
        // request is cancelled or admitted.
        self.changed.notify_waiters();
    }

    pub(super) fn close(&self) {
        self.slots.close();
        self.changed.notify_waiters();
    }

    #[cfg(test)]
    pub(super) fn waiter_counts(&self) -> (usize, usize) {
        let state = self.state.lock();
        (state.latency_waiters, state.throughput_waiters)
    }

    #[cfg(test)]
    pub(super) fn available_permits(&self) -> usize {
        self.slots.available_permits()
    }
}

struct PriorityAdmissionWaiter {
    admission: Arc<PriorityAdmission>,
    priority: Priority,
    registered: bool,
}

impl PriorityAdmissionWaiter {
    fn register(admission: Arc<PriorityAdmission>, priority: Priority) -> Self {
        {
            let mut state = admission.state.lock();
            match priority {
                Priority::LatencyCritical => state.latency_waiters += 1,
                Priority::Throughput => state.throughput_waiters += 1,
            }
        }
        Self {
            admission,
            priority,
            registered: true,
        }
    }

    fn finish(&mut self) {
        if std::mem::replace(&mut self.registered, false) {
            self.admission.unregister_waiter(self.priority);
        }
    }
}

impl Drop for PriorityAdmissionWaiter {
    fn drop(&mut self) {
        self.finish();
    }
}

pub(super) struct PriorityAdmissionPermit {
    admission: Arc<PriorityAdmission>,
    permit: Option<OwnedSemaphorePermit>,
}

impl PriorityAdmissionPermit {
    fn new(admission: Arc<PriorityAdmission>, permit: OwnedSemaphorePermit) -> Self {
        Self {
            admission,
            permit: Some(permit),
        }
    }
}

impl Drop for PriorityAdmissionPermit {
    fn drop(&mut self) {
        // Return capacity before waking waiters. The state lock in acquisition
        // then makes the high-priority check and permit claim one atomic choice.
        drop(self.permit.take());
        self.admission.changed.notify_waiters();
    }
}

impl Drop for StreamAdmissionGuard {
    fn drop(&mut self) {
        self.counter.fetch_sub(1, Ordering::Relaxed);
    }
}

pub(super) struct TrackedEngineStream {
    pub(super) inner: EngineStream,
    pub(super) _guard: StreamAdmissionGuard,
}

impl Unpin for TrackedEngineStream {}

impl Stream for TrackedEngineStream {
    type Item = Result<BinaryTensorPacket, EngineError>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        self.inner.as_mut().poll_next(cx)
    }
}

pub(super) struct TrackedOpenAiWireStream {
    pub(super) inner: OpenAiWireStream,
    pub(super) _guard: StreamAdmissionGuard,
}

impl Unpin for TrackedOpenAiWireStream {}

impl Stream for TrackedOpenAiWireStream {
    type Item = Result<Vec<u8>, EngineError>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        self.inner.as_mut().poll_next(cx)
    }
}
