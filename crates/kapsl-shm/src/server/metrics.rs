use crate::allocator::ShmAllocatorSnapshot;
use prometheus::{IntCounter, IntGauge, Opts, Registry};
use std::sync::Arc;

#[derive(Clone)]
pub(super) struct ShmPoolMetrics {
    pool_in_use: IntGauge,
    pool_exhausted_total: IntCounter,
    pool_oldest_lease_ms: IntGauge,
}

impl ShmPoolMetrics {
    pub(super) fn register(registry: &Arc<Registry>) -> Result<Self, prometheus::Error> {
        let pool_in_use = IntGauge::with_opts(Opts::new(
            "kapsl_shm_pool_in_use",
            "Number of currently leased SHM tensor slots",
        ))?;
        let pool_exhausted_total = IntCounter::with_opts(Opts::new(
            "kapsl_shm_pool_exhausted_total",
            "Total number of SHM tensor pool allocation failures",
        ))?;
        let pool_oldest_lease_ms = IntGauge::with_opts(Opts::new(
            "kapsl_shm_pool_oldest_lease_ms",
            "Age in milliseconds of the oldest active SHM slot lease",
        ))?;

        registry.register(Box::new(pool_in_use.clone()))?;
        registry.register(Box::new(pool_exhausted_total.clone()))?;
        registry.register(Box::new(pool_oldest_lease_ms.clone()))?;

        Ok(Self {
            pool_in_use,
            pool_exhausted_total,
            pool_oldest_lease_ms,
        })
    }

    pub(super) fn update_from_snapshot(&self, snapshot: ShmAllocatorSnapshot) {
        self.pool_in_use
            .set(snapshot.in_use_slots.min(i64::MAX as usize) as i64);
        self.pool_oldest_lease_ms
            .set(snapshot.oldest_lease_ms.min(i64::MAX as u64) as i64);
    }

    pub(super) fn on_exhausted(&self) {
        self.pool_exhausted_total.inc();
    }
}
