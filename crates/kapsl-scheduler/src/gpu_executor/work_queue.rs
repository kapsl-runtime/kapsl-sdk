use crate::priority::Priority;
use crate::request::Request;
use crate::scheduler::SharedSchedulerObserver;
use kapsl_engine_api::EngineError;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tokio::sync::Notify;
use tokio::time::timeout;

struct WorkQueueInner {
    queue: Mutex<VecDeque<QueuedRequest>>,
    capacity: usize,
    queue_len: AtomicUsize,
    closed: AtomicBool,
    not_empty: Notify,
    not_full: Notify,
    observer: SharedSchedulerObserver,
    priority: Priority,
    operation: &'static str,
}

struct QueuedRequest {
    request: Request,
    queued_at: Instant,
}

#[derive(Clone)]
pub(crate) struct WorkQueue {
    inner: Arc<WorkQueueInner>,
}

impl WorkQueue {
    fn reject(request: Request, message: &'static str) {
        let _ = request
            .response_tx
            .send(Err(EngineError::overloaded(message.to_string())));
    }

    #[cfg(test)]
    pub(crate) fn new(capacity: usize) -> Self {
        Self::new_observed(
            capacity,
            Priority::Throughput,
            "translated",
            Arc::new(parking_lot::RwLock::new(None)),
        )
    }

    pub(crate) fn new_observed(
        capacity: usize,
        priority: Priority,
        operation: &'static str,
        observer: SharedSchedulerObserver,
    ) -> Self {
        let capacity = capacity.max(1);
        Self {
            inner: Arc::new(WorkQueueInner {
                queue: Mutex::new(VecDeque::with_capacity(capacity)),
                capacity,
                queue_len: AtomicUsize::new(0),
                closed: AtomicBool::new(false),
                not_empty: Notify::new(),
                not_full: Notify::new(),
                observer,
                priority,
                operation,
            }),
        }
    }

    pub(crate) fn close(&self) {
        if self.inner.closed.swap(true, Ordering::AcqRel) {
            return;
        }
        let mut queue = self.inner.queue.lock().unwrap();
        let dropped = queue.drain(..).collect::<Vec<_>>();
        self.inner.queue_len.store(0, Ordering::Release);
        drop(queue);
        for queued in dropped {
            Self::reject(queued.request, "Scheduler queue closed");
        }
        self.inner.not_empty.notify_waiters();
        self.inner.not_full.notify_waiters();
    }

    pub(super) fn is_closed(&self) -> bool {
        self.inner.closed.load(Ordering::Acquire)
    }

    pub(crate) fn len(&self) -> usize {
        self.inner.queue_len.load(Ordering::Relaxed)
    }

    pub(crate) fn capacity(&self) -> usize {
        self.inner.capacity
    }

    // Returning ownership lets callers retry or report the exact request
    // without a heap allocation on the scheduler hot path.
    #[allow(clippy::result_large_err)]
    pub(crate) fn try_push_drop_newest(&self, request: Request) -> Result<(), Request> {
        if self.is_closed() {
            return Err(request);
        }
        let mut queue = self.inner.queue.lock().unwrap();
        if self.is_closed() {
            return Err(request);
        }
        if queue.len() >= self.inner.capacity {
            return Err(request);
        }
        queue.push_back(QueuedRequest {
            request,
            queued_at: Instant::now(),
        });
        self.inner.queue_len.fetch_add(1, Ordering::Relaxed);
        drop(queue);
        self.inner.not_empty.notify_one();
        Ok(())
    }

    pub(crate) fn push_drop_oldest(&self, request: Request) -> Option<Request> {
        if self.is_closed() {
            return Some(request);
        }
        let mut queue = self.inner.queue.lock().unwrap();
        if self.is_closed() {
            return Some(request);
        }
        let is_full = queue.len() >= self.inner.capacity;
        let dropped = if is_full { queue.pop_front() } else { None };
        queue.push_back(QueuedRequest {
            request,
            queued_at: Instant::now(),
        });
        if !is_full {
            self.inner.queue_len.fetch_add(1, Ordering::Relaxed);
        }
        drop(queue);
        self.inner.not_empty.notify_one();
        dropped.map(|queued| queued.request)
    }

    pub(crate) async fn push_block(&self, request: Request) {
        let mut pending = Some(QueuedRequest {
            request,
            queued_at: Instant::now(),
        });
        loop {
            if self.is_closed() {
                let queued = pending.take().expect("pending request must exist");
                Self::reject(queued.request, "Scheduler queue closed");
                return;
            }

            // Register before checking capacity so a dequeue between the check
            // and the await cannot be lost.
            let notified = self.inner.not_full.notified();
            tokio::pin!(notified);
            notified.as_mut().enable();

            let queued = {
                let mut queue = self.inner.queue.lock().unwrap();
                if self.is_closed() {
                    drop(queue);
                    let queued = pending.take().expect("pending request must exist");
                    Self::reject(queued.request, "Scheduler queue closed");
                    return;
                } else if queue.len() < self.inner.capacity {
                    queue.push_back(pending.take().expect("pending request must exist"));
                    self.inner.queue_len.fetch_add(1, Ordering::Relaxed);
                    true
                } else {
                    false
                }
            };

            if queued {
                self.inner.not_empty.notify_one();
                return;
            }

            let cancellation = pending
                .as_ref()
                .and_then(|queued| queued.request.input.cancellation.clone());
            if let Some(cancellation) = cancellation {
                tokio::select! {
                    _ = &mut notified => {}
                    _ = cancellation.cancelled() => {
                        let queued = pending.take().expect("pending request must exist");
                        let _ = queued.request.response_tx.send(Err(EngineError::cancelled(
                            "Request cancelled while awaiting queue capacity",
                        )));
                        return;
                    }
                }
            } else {
                notified.await;
            }
        }
    }

    pub(crate) fn pop_nowait(&self) -> Option<Request> {
        let mut queue = self.inner.queue.lock().unwrap();
        let popped = queue.pop_front();
        if popped.is_some() {
            self.inner.queue_len.fetch_sub(1, Ordering::Relaxed);
        }
        drop(queue);
        if popped.is_some() {
            self.inner.not_full.notify_one();
        }
        popped.map(|queued| {
            if let Some(observer) = self.inner.observer.read().as_ref() {
                observer.observe_queue_wait(
                    self.inner.priority,
                    self.inner.operation,
                    queued.queued_at.elapsed(),
                );
            }
            queued.request
        })
    }

    pub(crate) async fn pop_timeout(&self, timeout_duration: Duration) -> Option<Request> {
        let deadline = Instant::now() + timeout_duration;
        loop {
            // Register before checking the queue so an enqueue cannot race
            // with creation of the notification future.
            let notified = self.inner.not_empty.notified();
            tokio::pin!(notified);
            notified.as_mut().enable();

            if let Some(request) = self.pop_nowait() {
                return Some(request);
            }
            if self.is_closed() {
                return None;
            }

            let now = Instant::now();
            if now >= deadline {
                return None;
            }
            let remaining = deadline.saturating_duration_since(now);

            if timeout(remaining, &mut notified).await.is_err() {
                return None;
            }
        }
    }

    pub(crate) async fn wait_for_item(&self) {
        loop {
            // Register before checking state to avoid losing a concurrent
            // enqueue or close notification.
            let notified = self.inner.not_empty.notified();
            tokio::pin!(notified);
            notified.as_mut().enable();

            if !self.inner.queue.lock().unwrap().is_empty() {
                return;
            }
            if self.is_closed() {
                return;
            }
            notified.await;
        }
    }
}
