//! Replica selection, load balancing, and failover.

use crate::metrics::MetricsAccumulator;
use crate::priority::Priority;
use kapsl_engine_api::{
    BinaryTensorPacket, EngineError, EngineMetrics, InferenceRequest, OpenAiWireRequest,
    OpenAiWireResponse, OpenAiWireStreamResponse,
};
use parking_lot::RwLock;
use std::cmp::Reverse;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

/// Strategy for selecting which replica to route requests to
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PoolStrategy {
    /// Distribute requests evenly in round-robin fashion
    RoundRobin,
    /// Route to replica with lowest queue depth
    LeastLoaded,
    /// Sticky session routing based on session_id
    Sticky,
}

/// Statistics for a single replica
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReplicaStats {
    pub replica_id: u32,
    pub requests_total: u64,
    pub queue_depth: (usize, usize),
    pub healthy: bool,
}

/// Pool of replicas for load balancing
pub struct ReplicaPool<T> {
    replicas: RwLock<Vec<PooledReplica<T>>>,
    strategy: PoolStrategy,
    round_robin_counter: AtomicUsize,
}

struct PooledReplica<T> {
    replica_id: u32,
    scheduler: Arc<T>,
    requests_total: AtomicUsize,
}

struct DispatchPlan<T> {
    selected_replica_id: u32,
    selected_scheduler: Arc<T>,
    fallback_schedulers: Vec<(u32, Arc<T>)>,
}

impl<T> ReplicaPool<T>
where
    T: ReplicaScheduler + Send + Sync + 'static,
{
    fn paged_kv_routing_key(metrics: &EngineMetrics) -> Option<(u8, usize, usize)> {
        let total_blocks = metrics.kv_cache_blocks_total;
        if total_blocks == 0 {
            return None;
        }

        let free_blocks = metrics.kv_cache_blocks_free.min(total_blocks);
        let used_blocks = total_blocks.saturating_sub(free_blocks);
        let utilization_permille = used_blocks.saturating_mul(1000) / total_blocks.max(1);
        let pressure_tier = match utilization_permille {
            0..=699 => 0,
            700..=849 => 1,
            850..=949 => 2,
            _ => 3,
        };
        let free_bytes = metrics
            .kv_cache_bytes_capacity
            .saturating_sub(metrics.kv_cache_bytes_used);

        Some((pressure_tier, free_blocks, free_bytes))
    }

    fn memory_routing_enabled(replicas: &[PooledReplica<T>]) -> bool {
        replicas
            .iter()
            .filter(|replica| replica.scheduler.is_healthy())
            .any(|replica| Self::paged_kv_routing_key(&replica.scheduler.get_metrics()).is_some())
    }

    fn routing_key(
        &self,
        replica: &PooledReplica<T>,
        memory_aware: bool,
    ) -> (u8, usize, Reverse<usize>, Reverse<usize>, usize) {
        let (high, low) = replica.scheduler.get_queue_depth();
        let total_depth = high.saturating_add(low);

        if memory_aware {
            if let Some((pressure_tier, free_blocks, free_bytes)) =
                Self::paged_kv_routing_key(&replica.scheduler.get_metrics())
            {
                return (
                    pressure_tier,
                    total_depth,
                    Reverse(free_blocks),
                    Reverse(free_bytes),
                    replica.requests_total.load(Ordering::Relaxed),
                );
            }

            return (
                u8::MAX,
                total_depth,
                Reverse(0usize),
                Reverse(0usize),
                replica.requests_total.load(Ordering::Relaxed),
            );
        }

        (
            0,
            total_depth,
            Reverse(0usize),
            Reverse(0usize),
            replica.requests_total.load(Ordering::Relaxed),
        )
    }

    fn select_index(&self, replicas: &[PooledReplica<T>], session_id: Option<&str>) -> usize {
        match self.strategy {
            PoolStrategy::RoundRobin => self.select_round_robin(replicas),
            PoolStrategy::LeastLoaded => self.select_least_loaded(replicas),
            PoolStrategy::Sticky => self.select_sticky_session(replicas, session_id),
        }
    }

    fn dispatch_plan(&self, session_id: Option<&str>) -> Result<DispatchPlan<T>, EngineError> {
        let replicas = self.replicas.read();
        if replicas.is_empty() {
            return Err(EngineError::overloaded(
                "No replicas available in pool".to_string(),
            ));
        }

        let memory_aware = Self::memory_routing_enabled(&replicas);
        let selected_index = self.select_index(&replicas, session_id);
        let selected = &replicas[selected_index];
        selected.requests_total.fetch_add(1, Ordering::Relaxed);

        let mut fallback_schedulers = replicas
            .iter()
            .enumerate()
            .filter(|(index, replica)| *index != selected_index && replica.scheduler.is_healthy())
            .map(|(_, replica)| {
                (
                    self.routing_key(replica, memory_aware),
                    replica.replica_id,
                    replica.scheduler.clone(),
                )
            })
            .collect::<Vec<_>>();
        fallback_schedulers.sort_by(|left, right| left.0.cmp(&right.0));

        Ok(DispatchPlan {
            selected_replica_id: selected.replica_id,
            selected_scheduler: selected.scheduler.clone(),
            fallback_schedulers: fallback_schedulers
                .into_iter()
                .map(|(_, replica_id, scheduler)| (replica_id, scheduler))
                .collect(),
        })
    }

    /// Create a new replica pool with the specified strategy
    pub fn new(strategy: PoolStrategy) -> Self {
        Self {
            replicas: RwLock::new(Vec::new()),
            strategy,
            round_robin_counter: AtomicUsize::new(0),
        }
    }

    /// Add a replica to the pool
    pub fn add_replica(&self, replica_id: u32, scheduler: Arc<T>) {
        let mut replicas = self.replicas.write();
        replicas.push(PooledReplica {
            replica_id,
            scheduler,
            requests_total: AtomicUsize::new(0),
        });
    }

    /// Remove a replica from the pool by replica_id
    pub fn remove_replica(&self, replica_id: u32) -> bool {
        let mut replicas = self.replicas.write();
        if let Some(pos) = replicas.iter().position(|r| r.replica_id == replica_id) {
            replicas.remove(pos);
            true
        } else {
            false
        }
    }

    /// Get the number of replicas in the pool
    pub fn size(&self) -> usize {
        self.replicas.read().len()
    }

    /// Get the number of healthy replicas in the pool
    pub fn get_healthy_replica_count(&self) -> usize {
        self.replicas
            .read()
            .iter()
            .filter(|replica| replica.scheduler.is_healthy())
            .count()
    }

    /// Get statistics for all replicas
    pub fn stats(&self) -> Vec<ReplicaStats> {
        self.replicas
            .read()
            .iter()
            .map(|replica| ReplicaStats {
                replica_id: replica.replica_id,
                requests_total: replica.requests_total.load(Ordering::Relaxed) as u64,
                queue_depth: replica.scheduler.get_queue_depth(),
                healthy: replica.scheduler.is_healthy(),
            })
            .collect()
    }

    /// Execute a request on an appropriate replica
    pub async fn execute(
        &self,
        request: InferenceRequest,
        priority: Priority,
        force_cpu: bool,
    ) -> Result<BinaryTensorPacket, EngineError> {
        // Selection clones the scheduler handles before awaiting so scaling
        // can add or remove replicas while inference is in flight.
        let DispatchPlan {
            selected_replica_id,
            selected_scheduler,
            fallback_schedulers,
        } = self.dispatch_plan(request.session_id.as_deref())?;

        let result = selected_scheduler
            .infer(&request, priority, force_cpu)
            .await;

        if result.is_err() && !fallback_schedulers.is_empty() {
            log::warn!(
                "Request failed on replica {}, attempting failover",
                selected_replica_id
            );

            for (replica_id, scheduler) in fallback_schedulers {
                if !scheduler.is_healthy() {
                    continue;
                }
                log::info!("Failing over to replica {}", replica_id);
                if let Ok(response) = scheduler.infer(&request, priority, force_cpu).await {
                    if let Some(replica) = self
                        .replicas
                        .read()
                        .iter()
                        .find(|replica| replica.replica_id == replica_id)
                    {
                        replica.requests_total.fetch_add(1, Ordering::Relaxed);
                    }
                    return Ok(response);
                }
            }
        }

        result
    }

    /// Select one locally healthy replica before dispatching a generation.
    ///
    /// Failover is deliberately limited to this pre-dispatch decision. Once a
    /// wire request has entered an engine, a timeout/reset/backend error is
    /// ambiguous: replaying it could run the same generation twice.
    fn select_openai_scheduler(&self, request: &OpenAiWireRequest) -> Result<Arc<T>, EngineError> {
        let replicas = self.replicas.read();
        let memory_aware = Self::memory_routing_enabled(&replicas);
        if replicas.is_empty() {
            return Err(EngineError::overloaded(
                "No replicas available in pool".to_string(),
            ));
        }

        let selected_idx = self.select_index(&replicas, request.session_id.as_deref());
        let selected = &replicas[selected_idx];
        let chosen = if selected.scheduler.is_healthy() {
            selected
        } else {
            replicas
                .iter()
                .enumerate()
                .filter(|(index, replica)| *index != selected_idx && replica.scheduler.is_healthy())
                .min_by_key(|(_, replica)| self.routing_key(replica, memory_aware))
                .map(|(_, replica)| replica)
                .ok_or_else(|| {
                    EngineError::overloaded("No healthy replicas available in pool".to_string())
                })?
        };
        if chosen.replica_id != selected.replica_id {
            log::info!(
                "Selecting healthy OpenAI wire replica {} instead of unroutable replica {}",
                chosen.replica_id,
                selected.replica_id
            );
        }
        chosen.requests_total.fetch_add(1, Ordering::Relaxed);
        Ok(chosen.scheduler.clone())
    }

    fn select_round_robin(&self, replicas: &[PooledReplica<T>]) -> usize {
        let counter = self.round_robin_counter.fetch_add(1, Ordering::Relaxed);
        counter % replicas.len()
    }

    fn select_least_loaded(&self, replicas: &[PooledReplica<T>]) -> usize {
        let memory_aware = Self::memory_routing_enabled(replicas);
        let mut best_idx = 0;
        type RoutingKey = (u8, usize, Reverse<usize>, Reverse<usize>, usize);
        let mut best_key: Option<RoutingKey> = None;

        for (idx, replica) in replicas.iter().enumerate() {
            if !replica.scheduler.is_healthy() {
                continue;
            }

            let key = self.routing_key(replica, memory_aware);
            if best_key.as_ref().map(|best| key < *best).unwrap_or(true) {
                best_key = Some(key);
                best_idx = idx;
            }
        }

        best_idx
    }

    fn select_sticky_session(
        &self,
        replicas: &[PooledReplica<T>],
        session_id: Option<&str>,
    ) -> usize {
        // If the request has a session ID, use hash-based routing.
        if let Some(session_id) = session_id {
            use std::collections::hash_map::DefaultHasher;
            use std::hash::{Hash, Hasher};

            let mut hasher = DefaultHasher::new();
            session_id.hash(&mut hasher);
            (hasher.finish() as usize) % replicas.len()
        } else {
            // No session_id, fall back to round-robin
            self.select_round_robin(replicas)
        }
    }
}

#[async_trait::async_trait]
impl<T> ReplicaScheduler for ReplicaPool<T>
where
    T: ReplicaScheduler + Send + Sync + 'static,
{
    fn get_queue_depth(&self) -> (usize, usize) {
        let mut total_high = 0usize;
        let mut total_low = 0usize;
        for replica in self.replicas.read().iter() {
            let (high, low) = replica.scheduler.get_queue_depth();
            total_high = total_high.saturating_add(high);
            total_low = total_low.saturating_add(low);
        }
        (total_high, total_low)
    }

    fn is_healthy(&self) -> bool {
        self.replicas
            .read()
            .iter()
            .any(|replica| replica.scheduler.is_healthy())
    }

    fn get_metrics(&self) -> kapsl_engine_api::EngineMetrics {
        let replicas = self.replicas.read();
        let mut aggregate = MetricsAccumulator::default();
        let mut queue_depth = 0usize;
        for replica in replicas.iter() {
            aggregate.add(&replica.scheduler.get_metrics());
            let (cpu, gpu) = replica.scheduler.get_queue_depth();
            queue_depth = queue_depth.saturating_add(cpu).saturating_add(gpu);
        }
        let mut metrics = aggregate.finish();
        metrics.queue_depth = queue_depth;
        metrics
    }

    fn model_info(&self) -> Option<kapsl_engine_api::EngineModelInfo> {
        self.replicas
            .read()
            .iter()
            .find_map(|replica| replica.scheduler.model_info())
    }

    async fn infer(
        &self,
        request: &InferenceRequest,
        priority: Priority,
        force_cpu: bool,
    ) -> Result<BinaryTensorPacket, EngineError> {
        self.execute(request.clone(), priority, force_cpu).await
    }

    async fn infer_openai_wire(
        &self,
        request: OpenAiWireRequest,
        priority: Priority,
        force_cpu: bool,
    ) -> Result<OpenAiWireResponse, EngineError> {
        self.select_openai_scheduler(&request)?
            .infer_openai_wire(request, priority, force_cpu)
            .await
    }

    async fn infer_openai_wire_stream(
        &self,
        request: OpenAiWireRequest,
        priority: Priority,
        force_cpu: bool,
    ) -> Result<OpenAiWireStreamResponse, EngineError> {
        self.select_openai_scheduler(&request)?
            .infer_openai_wire_stream(request, priority, force_cpu)
            .await
    }

    async fn infer_stream(
        &self,
        request: InferenceRequest,
        priority: Priority,
        force_cpu: bool,
    ) -> Result<
        std::pin::Pin<
            Box<dyn futures::Stream<Item = Result<BinaryTensorPacket, EngineError>> + Send>,
        >,
        EngineError,
    > {
        let DispatchPlan {
            selected_replica_id,
            selected_scheduler,
            fallback_schedulers,
        } = self.dispatch_plan(request.session_id.as_deref())?;

        if selected_scheduler.is_healthy() {
            match selected_scheduler
                .infer_stream(request.clone(), priority, force_cpu)
                .await
            {
                Ok(stream) => return Ok(stream),
                Err(e) => {
                    log::warn!(
                        "Streaming request failed on replica {}: {}, attempting failover",
                        selected_replica_id,
                        e
                    );
                }
            }
        }

        // Failover to other healthy replicas.
        for (replica_id, scheduler) in fallback_schedulers {
            if !scheduler.is_healthy() {
                continue;
            }
            log::info!("Failing over streaming request to replica {}", replica_id);
            match scheduler
                .infer_stream(request.clone(), priority, force_cpu)
                .await
            {
                Ok(stream) => {
                    if let Some(replica) = self
                        .replicas
                        .read()
                        .iter()
                        .find(|replica| replica.replica_id == replica_id)
                    {
                        replica.requests_total.fetch_add(1, Ordering::Relaxed);
                    }
                    return Ok(stream);
                }
                Err(_) => continue,
            }
        }

        // If we got here, all attempts failed
        Err(EngineError::overloaded(
            "All replicas failed or overloaded".to_string(),
        ))
    }
}

/// Trait that replica schedulers must implement
#[async_trait::async_trait]
pub trait ReplicaScheduler: Send + Sync {
    fn get_queue_depth(&self) -> (usize, usize);
    fn is_healthy(&self) -> bool;
    fn get_metrics(&self) -> kapsl_engine_api::EngineMetrics;
    fn model_info(&self) -> Option<kapsl_engine_api::EngineModelInfo> {
        None
    }
    async fn infer(
        &self,
        request: &InferenceRequest,
        priority: Priority,
        force_cpu: bool,
    ) -> Result<BinaryTensorPacket, EngineError>;

    async fn infer_openai_wire(
        &self,
        _request: OpenAiWireRequest,
        _priority: Priority,
        _force_cpu: bool,
    ) -> Result<OpenAiWireResponse, EngineError> {
        Err(EngineError::backend(
            "scheduler does not support protocol-native OpenAI requests",
        ))
    }

    async fn infer_openai_wire_stream(
        &self,
        _request: OpenAiWireRequest,
        _priority: Priority,
        _force_cpu: bool,
    ) -> Result<OpenAiWireStreamResponse, EngineError> {
        Err(EngineError::backend(
            "scheduler does not support protocol-native OpenAI streams",
        ))
    }

    async fn infer_stream(
        &self,
        request: InferenceRequest,
        priority: Priority,
        force_cpu: bool,
    ) -> Result<
        std::pin::Pin<
            Box<dyn futures::Stream<Item = Result<BinaryTensorPacket, EngineError>> + Send>,
        >,
        EngineError,
    >;
}

#[cfg(test)]
mod tests;
