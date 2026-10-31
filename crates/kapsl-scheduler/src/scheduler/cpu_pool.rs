use super::admission::{PriorityAdmission, PriorityAdmissionPermit};
use super::QueueOverflowPolicy;
use crate::priority::Priority;
use kapsl_engine_api::{CancellationToken, EngineError};
use std::sync::Arc;

/// Rayon threads for `force_cpu` requests, admitted one request per thread.
///
/// Admission applies the caller's overflow policy and serves latency-critical
/// waiters first, so CPU work gets the same backpressure as the GPU queues
/// instead of piling up in Rayon's unbounded injector.
///
/// Clones share both the threads and the admission, so one pool handed to
/// every [`Scheduler`](super::Scheduler) bounds CPU work across all of them
/// instead of each replica starting threads of its own.
#[derive(Clone)]
pub struct CpuPool {
    pool: Arc<rayon::ThreadPool>,
    admission: Arc<PriorityAdmission>,
}

impl CpuPool {
    /// Builds a pool with `threads` workers. Zero keeps Rayon's automatic
    /// thread-count selection.
    pub fn new(threads: usize) -> Result<Self, rayon::ThreadPoolBuildError> {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()?;
        let admission = PriorityAdmission::new(pool.current_num_threads(), "CPU");
        Ok(Self {
            pool: Arc::new(pool),
            admission,
        })
    }

    pub fn threads(&self) -> usize {
        self.pool.current_num_threads()
    }

    /// Requests holding a thread, across every scheduler sharing this pool.
    pub fn active(&self) -> usize {
        self.threads()
            .saturating_sub(self.admission.available_permits())
    }

    /// Admits one request under `overflow_policy`: `Block` waits for a free
    /// thread, the drop policies fail at once when every thread is taken.
    pub(super) async fn acquire(
        &self,
        priority: Priority,
        overflow_policy: QueueOverflowPolicy,
        cancellation: Option<&CancellationToken>,
    ) -> Result<PriorityAdmissionPermit, EngineError> {
        self.admission
            .acquire(priority, overflow_policy, cancellation)
            .await
    }

    /// Runs `job` on the pool, holding `slot` until the job finishes.
    pub(super) fn spawn(&self, slot: PriorityAdmissionPermit, job: impl FnOnce() + Send + 'static) {
        self.pool.spawn(move || {
            let _slot = slot;
            job();
        });
    }
}
