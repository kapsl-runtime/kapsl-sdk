use crate::request::Request;
use kapsl_engine_api::{EngineError, EngineHandle, InferenceRequest, RequestMetadata};
use log::info;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tokio::sync::Notify;
use tokio::time::timeout;

struct InFlightGuard {
    counter: Arc<AtomicUsize>,
    n: usize,
}

impl Drop for InFlightGuard {
    fn drop(&mut self) {
        self.counter.fetch_sub(self.n, Ordering::Relaxed);
    }
}

struct WorkQueueInner {
    queue: Mutex<VecDeque<Request>>,
    capacity: usize,
    queue_len: AtomicUsize,
    not_empty: Notify,
    not_full: Notify,
}

#[derive(Clone)]
pub(crate) struct WorkQueue {
    inner: Arc<WorkQueueInner>,
}

impl WorkQueue {
    pub(crate) fn new(capacity: usize) -> Self {
        let capacity = capacity.max(1);
        Self {
            inner: Arc::new(WorkQueueInner {
                queue: Mutex::new(VecDeque::with_capacity(capacity)),
                capacity,
                queue_len: AtomicUsize::new(0),
                not_empty: Notify::new(),
                not_full: Notify::new(),
            }),
        }
    }

    pub(crate) fn len(&self) -> usize {
        self.inner.queue_len.load(Ordering::Relaxed)
    }

    pub(crate) fn capacity(&self) -> usize {
        self.inner.capacity
    }

    pub(crate) fn try_push_drop_newest(&self, request: Request) -> Result<(), Request> {
        let mut queue = self.inner.queue.lock().unwrap();
        if queue.len() >= self.inner.capacity {
            return Err(request);
        }
        queue.push_back(request);
        self.inner.queue_len.fetch_add(1, Ordering::Relaxed);
        drop(queue);
        self.inner.not_empty.notify_one();
        Ok(())
    }

    pub(crate) fn push_drop_oldest(&self, request: Request) -> Option<Request> {
        let mut queue = self.inner.queue.lock().unwrap();
        let is_full = queue.len() >= self.inner.capacity;
        let dropped = if is_full { queue.pop_front() } else { None };
        queue.push_back(request);
        if !is_full {
            self.inner.queue_len.fetch_add(1, Ordering::Relaxed);
        }
        drop(queue);
        self.inner.not_empty.notify_one();
        dropped
    }

    pub(crate) async fn push_block(&self, request: Request) {
        let mut pending = Some(request);
        loop {
            let queued = {
                let mut queue = self.inner.queue.lock().unwrap();
                if queue.len() < self.inner.capacity {
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

            self.inner.not_full.notified().await;
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
        popped
    }

    pub(crate) async fn pop_timeout(&self, timeout_duration: Duration) -> Option<Request> {
        let deadline = Instant::now() + timeout_duration;
        loop {
            if let Some(request) = self.pop_nowait() {
                return Some(request);
            }

            let now = Instant::now();
            if now >= deadline {
                return None;
            }
            let remaining = deadline.saturating_duration_since(now);

            if timeout(remaining, self.inner.not_empty.notified())
                .await
                .is_err()
            {
                return None;
            }
        }
    }

    pub(crate) async fn wait_for_item(&self) {
        loop {
            if !self.inner.queue.lock().unwrap().is_empty() {
                return;
            }
            self.inner.not_empty.notified().await;
        }
    }
}

/// Default admission headroom: pause admission to a self-batching backend once
/// its free KV cache falls below this percentage of total capacity. Overridable
/// via `KAPSL_ADMISSION_MIN_FREE_PCT`. 0 disables occupancy-driven admission.
const DEFAULT_ADMISSION_MIN_FREE_PCT: usize = 5;

/// Back-off between occupancy re-checks while admission is paused.
const ADMISSION_POLL: Duration = Duration::from_millis(2);

/// GPU Executor that processes requests from priority queues
pub struct GpuExecutor {
    high_priority_queue: WorkQueue,
    low_priority_queue: WorkQueue,
    engine: EngineHandle,
    max_micro_batch: usize,
    queue_delay: Duration,
    in_flight: Arc<AtomicUsize>,
    /// Minimum free-KV headroom (percent of total) below which a self-batching
    /// backend stops accepting new requests. 0 disables the gate.
    admission_min_free_pct: usize,
}

impl GpuExecutor {
    pub(crate) fn new(
        high_priority_queue: WorkQueue,
        low_priority_queue: WorkQueue,
        engine: EngineHandle,
        max_micro_batch: usize,
        queue_delay_ms: u64,
        in_flight: Arc<AtomicUsize>,
    ) -> Self {
        let admission_min_free_pct = std::env::var("KAPSL_ADMISSION_MIN_FREE_PCT")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(DEFAULT_ADMISSION_MIN_FREE_PCT)
            .min(100);
        Self {
            high_priority_queue,
            low_priority_queue,
            engine,
            max_micro_batch: max_micro_batch.max(1),
            queue_delay: Duration::from_millis(queue_delay_ms),
            in_flight,
            admission_min_free_pct,
        }
    }

    /// Override the free-KV admission headroom (percent). Used by tests and
    /// tuning; production reads the default / `KAPSL_ADMISSION_MIN_FREE_PCT`.
    #[cfg(test)]
    pub(crate) fn with_admission_min_free_pct(mut self, pct: usize) -> Self {
        self.admission_min_free_pct = pct.min(100);
        self
    }

    /// Whether a self-batching backend has exhausted its admission headroom,
    /// based on its published KV-cache occupancy. Returns `false` when the gate
    /// is disabled or the backend exposes no KV signal yet (nothing to gate on),
    /// so backends without KV telemetry are never throttled.
    fn backend_saturated(&self) -> bool {
        if self.admission_min_free_pct == 0 {
            return false;
        }
        let metrics = self.engine.metrics();
        if metrics.kv_cache_blocks_total == 0 {
            return false;
        }
        metrics.kv_cache_blocks_free.saturating_mul(100)
            < metrics
                .kv_cache_blocks_total
                .saturating_mul(self.admission_min_free_pct)
    }

    /// Stamp the resolved queue priority onto a request's metadata so a
    /// self-batching backend can honor it in its own internal queue. The
    /// scheduler's queue choice is the authoritative priority (it already folds
    /// in SLA/size promotion), so this overwrites any raw hint the caller set.
    /// Uses the engine-api convention: 0 = latency-critical, 1 = throughput.
    fn stamp_priority(req: &mut Request, latency_critical: bool) {
        let priority = if latency_critical { 0 } else { 1 };
        req.input
            .metadata
            .get_or_insert_with(RequestMetadata::default)
            .priority = Some(priority);
    }

    fn dispatch_single(engine: EngineHandle, req: Request, in_flight: Arc<AtomicUsize>) {
        if req
            .input
            .cancellation
            .as_ref()
            .is_some_and(|token| token.is_cancelled())
        {
            let _ = req
                .response_tx
                .send(Err(EngineError::cancelled("Request cancelled")));
            return;
        }

        in_flight.fetch_add(1, Ordering::Relaxed);
        tokio::task::spawn_blocking(move || {
            let _guard = InFlightGuard {
                counter: in_flight,
                n: 1,
            };
            let result = engine.infer(&req.input);
            let _ = req.response_tx.send(result);
        });
    }

    fn dispatch_batch(engine: EngineHandle, requests: Vec<Request>, in_flight: Arc<AtomicUsize>) {
        if requests.is_empty() {
            return;
        }

        let mut active = Vec::with_capacity(requests.len());
        for request in requests {
            if request
                .input
                .cancellation
                .as_ref()
                .is_some_and(|token| token.is_cancelled())
            {
                let _ = request
                    .response_tx
                    .send(Err(EngineError::cancelled("Request cancelled")));
                continue;
            }
            active.push(request);
        }

        if active.is_empty() {
            return;
        }

        let n = active.len();
        in_flight.fetch_add(n, Ordering::Relaxed);
        tokio::task::spawn_blocking(move || {
            let _guard = InFlightGuard {
                counter: in_flight,
                n,
            };
            if active.len() == 1 {
                let req = active
                    .into_iter()
                    .next()
                    .expect("single request batch must have one item");
                let result = engine.infer(&req.input);
                let _ = req.response_tx.send(result);
                return;
            }

            let infer_requests: Vec<InferenceRequest> =
                active.iter().map(|request| request.input.clone()).collect();

            match engine.infer_batch(&infer_requests) {
                Ok(outputs) => {
                    if outputs.len() != active.len() {
                        let reason = format!(
                            "Batched inference result length mismatch: expected {}, got {}",
                            active.len(),
                            outputs.len()
                        );
                        for request in active {
                            let _ = request.response_tx.send(Err(EngineError::InferenceError {
                                reason: reason.clone(),
                                source: None,
                            }));
                        }
                        return;
                    }

                    for (request, output) in active.into_iter().zip(outputs.into_iter()) {
                        let _ = request.response_tx.send(Ok(output));
                    }
                }
                Err(error) => {
                    let reason = format!("Batched inference failed: {}", error);
                    for request in active {
                        let _ = request.response_tx.send(Err(EngineError::InferenceError {
                            reason: reason.clone(),
                            source: None,
                        }));
                    }
                }
            }
        });
    }

    /// Coalesce pending requests into a batch of at most `cap`, starting from an
    /// already-dequeued `first` taken from `queue`.
    ///
    /// Greedily takes anything already queued without waiting (zero added
    /// latency when a burst has already arrived). Only if room remains AND there
    /// is concurrent work in flight does it wait up to `queue_delay` for
    /// stragglers — so an isolated request (`in_flight == 0`) is dispatched
    /// immediately with no latency penalty. `interrupt`, when set, aborts the
    /// straggler wait as soon as that higher-priority queue has items.
    async fn accumulate_batch(
        &self,
        queue: &WorkQueue,
        first: Request,
        cap: usize,
        interrupt: Option<&WorkQueue>,
    ) -> Vec<Request> {
        let mut batch = Vec::with_capacity(cap);
        batch.push(first);

        // Greedy, non-blocking: grab co-arrived requests with no delay.
        while batch.len() < cap {
            match queue.pop_nowait() {
                Some(req) => batch.push(req),
                None => break,
            }
        }

        // Adaptive window: only pay latency to gather stragglers when the model
        // is already under concurrent load. Skipped entirely for isolated
        // requests so single-stream latency does not regress.
        if batch.len() < cap
            && !self.queue_delay.is_zero()
            && self.in_flight.load(Ordering::Relaxed) > 0
        {
            let deadline = Instant::now() + self.queue_delay;
            while batch.len() < cap {
                if interrupt.is_some_and(|q| q.len() > 0) {
                    break;
                }
                let now = Instant::now();
                if now >= deadline {
                    break;
                }
                let remaining = deadline.saturating_duration_since(now);
                match queue.pop_timeout(remaining).await {
                    Some(req) => batch.push(req),
                    None => break,
                }
            }
        }

        batch
    }

    pub async fn run(self) {
        info!("GPU Executor started");
        // Backends that advertise max_batch() > 1 (e.g. an ONNX model with a
        // dynamic batch dim) implement a real stacked infer_batch. For those we
        // coalesce requests from BOTH priority queues, since small models are
        // classified latency-critical yet benefit most from batching. Backends
        // that don't (max_batch() == 1, the default) keep the original
        // single-dispatch / throughput-only micro-batch behavior untouched.
        //
        // Self-batching backends (e.g. an autoregressive LLM that continuously
        // batches active sequences at the decode step) must never be coalesced
        // here: request-level `infer_batch` would run the whole batch to
        // completion in one call and fight the backend's own batcher. They are
        // always dispatched individually and left to multiplex internally.
        let self_batches = self.engine.self_batches();
        let batch_cap = self.engine.max_batch().min(self.max_micro_batch);
        let batch_capable = batch_cap > 1 && !self_batches;

        loop {
            // Occupancy-driven admission (self-batching backends only): when the
            // backend signals it is saturated, stop pulling from the priority
            // queues so requests stay priority-ordered there instead of piling
            // into the backend's internal FIFO. Always admit when nothing is in
            // flight, so a full/stale occupancy snapshot can never stall forward
            // progress (an isolated request that alone exceeds the headroom must
            // still run).
            if self_batches
                && self.in_flight.load(Ordering::Relaxed) > 0
                && self.backend_saturated()
            {
                tokio::time::sleep(ADMISSION_POLL).await;
                continue;
            }

            if batch_capable {
                // Latency-critical first, then throughput; coalesce either queue.
                if let Some(req) = self.high_priority_queue.pop_nowait() {
                    let batch = self
                        .accumulate_batch(&self.high_priority_queue, req, batch_cap, None)
                        .await;
                    Self::dispatch_batch(self.engine.clone(), batch, self.in_flight.clone());
                    continue;
                }
                if let Some(req) = self.low_priority_queue.pop_nowait() {
                    let batch = self
                        .accumulate_batch(
                            &self.low_priority_queue,
                            req,
                            batch_cap,
                            Some(&self.high_priority_queue),
                        )
                        .await;
                    Self::dispatch_batch(self.engine.clone(), batch, self.in_flight.clone());
                    continue;
                }
            } else {
                if let Some(mut req) = self.high_priority_queue.pop_nowait() {
                    let engine = self.engine.clone();
                    // Propagate the resolved priority so a self-batching backend
                    // can jump this request ahead in its internal queue.
                    if self_batches {
                        Self::stamp_priority(&mut req, true);
                    }
                    Self::dispatch_single(engine, req, self.in_flight.clone());
                    continue;
                }

                if let Some(mut req) = self.low_priority_queue.pop_nowait() {
                    let engine = self.engine.clone();
                    // Self-batching backends multiplex internally, so hand each
                    // request over immediately rather than holding it back to
                    // build a serial `infer_batch` group.
                    if self_batches || self.max_micro_batch <= 1 || self.queue_delay.is_zero() {
                        if self_batches {
                            Self::stamp_priority(&mut req, false);
                        }
                        Self::dispatch_single(engine, req, self.in_flight.clone());
                        continue;
                    }

                    let mut batch = Vec::with_capacity(self.max_micro_batch);
                    batch.push(req);
                    let deadline = Instant::now() + self.queue_delay;

                    while batch.len() < self.max_micro_batch {
                        if self.high_priority_queue.len() > 0 {
                            break;
                        }

                        let now = Instant::now();
                        if now >= deadline {
                            break;
                        }

                        let remaining = deadline.saturating_duration_since(now);
                        match self.low_priority_queue.pop_timeout(remaining).await {
                            Some(next_req) => batch.push(next_req),
                            None => break,
                        }
                    }

                    Self::dispatch_batch(engine, batch, self.in_flight.clone());
                    continue;
                }
            }

            tokio::select! {
                _ = self.high_priority_queue.wait_for_item() => {}
                _ = self.low_priority_queue.wait_for_item() => {}
            }
        }
    }
}
