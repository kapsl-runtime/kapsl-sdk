use super::admission::ActiveCountGuard;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

/// Rayon threads for `force_cpu` requests, plus a count of the jobs on them.
///
/// Clones share both the threads and the count, so one pool handed to every
/// [`Scheduler`](super::Scheduler) bounds CPU work across all of them instead
/// of each replica starting threads of its own.
#[derive(Clone)]
pub struct CpuPool {
    pool: Arc<rayon::ThreadPool>,
    active: Arc<AtomicUsize>,
}

impl CpuPool {
    /// Builds a pool with `threads` workers. Zero keeps Rayon's automatic
    /// thread-count selection.
    pub fn new(threads: usize) -> Result<Self, rayon::ThreadPoolBuildError> {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()?;
        Ok(Self {
            pool: Arc::new(pool),
            active: Arc::new(AtomicUsize::new(0)),
        })
    }

    pub fn threads(&self) -> usize {
        self.pool.current_num_threads()
    }

    /// Jobs running or queued on this pool, across every scheduler sharing it.
    pub fn active(&self) -> usize {
        self.active.load(Ordering::Relaxed)
    }

    /// Claims a slot only while a thread is free, atomically, so concurrent
    /// non-blocking callers cannot all observe the same free thread.
    pub(super) fn try_reserve(&self) -> bool {
        let capacity = self.threads();
        self.active
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |active| {
                (active < capacity).then_some(active + 1)
            })
            .is_ok()
    }

    /// Runs `job` on the pool. `reserved` is true when [`Self::try_reserve`]
    /// has already counted it.
    pub(super) fn spawn(&self, reserved: bool, job: impl FnOnce() + Send + 'static) {
        if !reserved {
            self.active.fetch_add(1, Ordering::AcqRel);
        }
        let counter = self.active.clone();
        self.pool.spawn(move || {
            let _active = ActiveCountGuard { counter };
            job();
        });
    }

    #[cfg(test)]
    pub(super) fn release_reserved(&self) {
        self.active.fetch_sub(1, Ordering::Release);
    }
}
