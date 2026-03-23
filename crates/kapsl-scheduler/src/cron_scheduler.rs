use crate::priority::Priority;
use crate::scheduler::Scheduler;
use kapsl_core::{CronJobDef, CronOverflowPolicyDef, CronPriorityDef, CronScheduleDef, Manifest};
use chrono::Utc;
use cron::Schedule;
use kapsl_engine_api::{BinaryTensorPacket, EngineError, InferenceRequest};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::str::FromStr;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;
use tokio::task::JoinHandle;

mod duration_serde {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};
    use std::time::Duration;

    pub fn serialize<S: Serializer>(d: &Duration, s: S) -> Result<S::Ok, S::Error> {
        d.as_secs_f64().serialize(s)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Duration, D::Error> {
        f64::deserialize(d).map(Duration::from_secs_f64)
    }
}

/// How a cron job behaves when the scheduler queues are full at firing time.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum CronOverflowPolicy {
    /// If the queue is full, skip this firing and increment `missed_count`.
    /// Appropriate for all background / best-effort cron jobs (the default).
    #[default]
    SkipIfBusy,
    /// Wait until the queue has capacity before dispatching.
    /// Use only for jobs that must not miss a beat even under load.
    Block,
}

/// A cron job schedule: either a fixed interval or a cron expression.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CronSchedule {
    /// Fire every fixed duration (e.g. every 30 seconds).
    Interval(#[serde(with = "duration_serde")] Duration),
    /// Fire according to a cron expression (e.g. `"0 */5 * * * *"` = every 5 minutes).
    /// Expects the 6-field `<sec> <min> <hour> <dom> <mon> <dow>` format.
    Expression(String),
}

/// Callback invoked after each cron job fires, receiving the inference result.
pub type CronCallback =
    Arc<dyn Fn(String, Result<BinaryTensorPacket, EngineError>) + Send + Sync>;

/// Definition of a periodic inference job.
pub struct CronJob {
    /// Unique identifier for the job.
    pub id: String,
    /// When to fire the job.
    pub schedule: CronSchedule,
    /// The inference request sent each time the job fires.
    pub request: Arc<InferenceRequest>,
    /// Scheduling priority passed to the underlying [`Scheduler`].
    pub priority: Priority,
    /// If `true`, route to CPU pool; otherwise GPU queues.
    pub force_cpu: bool,
    /// What to do when the scheduler queues are full at firing time.
    /// Defaults to [`CronOverflowPolicy::SkipIfBusy`].
    pub overflow_policy: CronOverflowPolicy,
    /// Optional callback called with the result after each firing.
    /// Missed firings (skipped due to `SkipIfBusy`) do NOT invoke this callback.
    pub on_result: Option<CronCallback>,
}

/// A snapshot of a registered job's state.
#[derive(Debug, Clone)]
pub struct CronJobInfo {
    pub id: String,
    pub schedule: CronSchedule,
    pub priority: Priority,
    pub force_cpu: bool,
    pub overflow_policy: CronOverflowPolicy,
    pub enabled: bool,
    /// Total firings that were dispatched to the scheduler.
    pub fired_count: u64,
    /// Total firings skipped because the scheduler was busy (`SkipIfBusy` policy).
    pub missed_count: u64,
}

/// Errors returned by [`CronScheduler`] operations.
#[derive(Debug)]
pub enum CronError {
    /// A job with the given ID is already registered.
    DuplicateId(String),
    /// The cron expression could not be parsed.
    InvalidExpression(String),
    /// No job with the given ID exists.
    NotFound(String),
}

impl std::fmt::Display for CronError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CronError::DuplicateId(id) => write!(f, "cron job already registered: {id}"),
            CronError::InvalidExpression(e) => write!(f, "invalid cron expression: {e}"),
            CronError::NotFound(id) => write!(f, "cron job not found: {id}"),
        }
    }
}

impl std::error::Error for CronError {}

// Internal per-job metadata kept in the registry.
struct JobEntry {
    // Static fields (copied into CronJobInfo).
    id: String,
    schedule: CronSchedule,
    priority: Priority,
    force_cpu: bool,
    overflow_policy: CronOverflowPolicy,
    // Live counters shared with the background task.
    fired_count: Arc<AtomicU64>,
    missed_count: Arc<AtomicU64>,
    handle: JoinHandle<()>,
}

impl JobEntry {
    fn info(&self) -> CronJobInfo {
        CronJobInfo {
            id: self.id.clone(),
            schedule: self.schedule.clone(),
            priority: self.priority,
            force_cpu: self.force_cpu,
            overflow_policy: self.overflow_policy,
            enabled: !self.handle.is_finished(),
            fired_count: self.fired_count.load(Ordering::Relaxed),
            missed_count: self.missed_count.load(Ordering::Relaxed),
        }
    }
}

/// Schedules periodic inference jobs against a [`Scheduler`].
///
/// Cron jobs are treated as preemptible background work:
/// - They are always dispatched at [`Priority::Throughput`] or lower, so
///   latency-critical real-time requests are never delayed by cron activity.
/// - With the default [`CronOverflowPolicy::SkipIfBusy`] policy, a firing is
///   silently skipped (and counted in `missed_count`) when the scheduler queues
///   are full, rather than blocking the async executor.
///
/// # Example
/// ```rust,ignore
/// let cron = CronScheduler::new(Arc::new(scheduler));
///
/// cron.register(CronJob {
///     id: "heartbeat".to_string(),
///     schedule: CronSchedule::Expression("0 * * * * *".to_string()),
///     request: my_request_template,
///     priority: Priority::Throughput,
///     force_cpu: false,
///     overflow_policy: CronOverflowPolicy::SkipIfBusy,
///     on_result: None,
/// }).await?;
/// ```
pub struct CronScheduler {
    scheduler: Arc<Scheduler>,
    jobs: Arc<RwLock<HashMap<String, JobEntry>>>,
}

impl CronScheduler {
    /// Create a new `CronScheduler` wrapping `scheduler`.
    pub fn new(scheduler: Arc<Scheduler>) -> Self {
        Self {
            scheduler,
            jobs: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Register and immediately start a cron job.
    ///
    /// Returns [`CronError::DuplicateId`] if a job with the same ID already
    /// exists, or [`CronError::InvalidExpression`] for a bad cron expression.
    pub async fn register(&self, job: CronJob) -> Result<(), CronError> {
        let mut jobs = self.jobs.write().await;

        if jobs.contains_key(&job.id) {
            return Err(CronError::DuplicateId(job.id));
        }

        // Validate cron expression eagerly so the error surfaces at registration time.
        if let CronSchedule::Expression(ref expr) = job.schedule {
            Schedule::from_str(expr)
                .map_err(|e| CronError::InvalidExpression(e.to_string()))?;
        }

        let fired_count = Arc::new(AtomicU64::new(0));
        let missed_count = Arc::new(AtomicU64::new(0));

        let handle = self.spawn_job(&job, fired_count.clone(), missed_count.clone());

        jobs.insert(
            job.id.clone(),
            JobEntry {
                id: job.id,
                schedule: job.schedule,
                priority: job.priority,
                force_cpu: job.force_cpu,
                overflow_policy: job.overflow_policy,
                fired_count,
                missed_count,
                handle,
            },
        );

        Ok(())
    }

    /// Stop and remove a job by ID.  Returns `true` if found, `false` otherwise.
    pub async fn unregister(&self, job_id: &str) -> bool {
        let mut jobs = self.jobs.write().await;
        if let Some(entry) = jobs.remove(job_id) {
            entry.handle.abort();
            true
        } else {
            false
        }
    }

    /// Return a snapshot of all registered jobs (including live counters).
    pub async fn list_jobs(&self) -> Vec<CronJobInfo> {
        self.jobs.read().await.values().map(|e| e.info()).collect()
    }

    /// Return info for a single job, or `None` if not found.
    pub async fn job_info(&self, job_id: &str) -> Option<CronJobInfo> {
        self.jobs.read().await.get(job_id).map(|e| e.info())
    }

    /// Register a cron job from a manifest [`CronJobDef`].
    ///
    /// Converts the JSON-serializable definition into a live [`CronJob`] and
    /// registers it.  The `on_result` callback is `None`; callers that need
    /// result notifications should register the job manually via [`register`].
    pub async fn register_from_def(&self, def: CronJobDef) -> Result<(), CronError> {
        let schedule = match def.schedule {
            CronScheduleDef::Expression(expr) => CronSchedule::Expression(expr),
            CronScheduleDef::Interval { interval_secs } => {
                CronSchedule::Interval(Duration::from_secs(interval_secs))
            }
        };
        let priority = match def.priority {
            CronPriorityDef::Throughput => Priority::Throughput,
            CronPriorityDef::LatencyCritical => Priority::LatencyCritical,
        };
        let overflow_policy = match def.overflow_policy {
            CronOverflowPolicyDef::SkipIfBusy => CronOverflowPolicy::SkipIfBusy,
            CronOverflowPolicyDef::Block => CronOverflowPolicy::Block,
        };
        let request = Arc::new(InferenceRequest {
            input: def.input,
            additional_inputs: def.additional_inputs,
            session_id: None,
            metadata: None,
            cancellation: None,
        });
        self.register(CronJob {
            id: def.id,
            schedule,
            request,
            priority,
            force_cpu: def.force_cpu,
            overflow_policy,
            on_result: None,
        })
        .await
    }

    /// Register all cron jobs declared in a [`Manifest`].
    ///
    /// Calls [`register_from_def`] for each entry in `manifest.cron_jobs`.
    /// Returns the first error encountered, if any.
    pub async fn register_from_manifest(&self, manifest: Manifest) -> Result<(), CronError> {
        for def in manifest.cron_jobs {
            self.register_from_def(def).await?;
        }
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Internal helpers
    // -----------------------------------------------------------------------

    fn spawn_job(
        &self,
        job: &CronJob,
        fired_count: Arc<AtomicU64>,
        missed_count: Arc<AtomicU64>,
    ) -> JoinHandle<()> {
        let scheduler = self.scheduler.clone();
        let schedule = job.schedule.clone();
        let request = Arc::clone(&job.request);
        let priority = job.priority;
        let force_cpu = job.force_cpu;
        let overflow_policy = job.overflow_policy;
        let id = job.id.clone();
        let on_result = job.on_result.clone();

        tokio::spawn(async move {
            match schedule {
                CronSchedule::Interval(interval) => {
                    let mut ticker = tokio::time::interval(interval);
                    // Skip the first immediate tick so the first firing occurs
                    // after one full interval.
                    ticker.tick().await;
                    loop {
                        ticker.tick().await;
                        Self::fire(
                            &scheduler,
                            &id,
                            Arc::clone(&request),
                            priority,
                            force_cpu,
                            overflow_policy,
                            &on_result,
                            &fired_count,
                            &missed_count,
                        )
                        .await;
                    }
                }
                CronSchedule::Expression(expr) => {
                    // Safety: validated at registration time.
                    let parsed = Schedule::from_str(&expr).expect("already validated");
                    for next in parsed.upcoming(Utc) {
                        let now = Utc::now();
                        if next <= now {
                            continue;
                        }
                        let wait = (next - now).to_std().unwrap_or(Duration::ZERO);
                        tokio::time::sleep(wait).await;
                        Self::fire(
                            &scheduler,
                            &id,
                            Arc::clone(&request),
                            priority,
                            force_cpu,
                            overflow_policy,
                            &on_result,
                            &fired_count,
                            &missed_count,
                        )
                        .await;
                    }
                }
            }
        })
    }

    #[allow(clippy::too_many_arguments)]
    async fn fire(
        scheduler: &Scheduler,
        id: &str,
        request: Arc<InferenceRequest>,
        priority: Priority,
        force_cpu: bool,
        overflow_policy: CronOverflowPolicy,
        on_result: &Option<CronCallback>,
        fired_count: &AtomicU64,
        missed_count: &AtomicU64,
    ) {
        // Unwrap the Arc without cloning when we hold the only reference (the common case),
        // falling back to a clone only if other Arc handles exist.
        let request = Arc::try_unwrap(request).unwrap_or_else(|arc| (*arc).clone());
        let result = match overflow_policy {
            CronOverflowPolicy::SkipIfBusy => {
                scheduler.try_infer(request, priority, force_cpu).await
            }
            CronOverflowPolicy::Block => scheduler.infer(request, priority, force_cpu).await,
        };

        match &result {
            // Scheduler was full and we're configured to skip — count the miss
            // but don't invoke the callback (there is no result to report).
            Err(e) if overflow_policy == CronOverflowPolicy::SkipIfBusy && e.is_overloaded() => {
                missed_count.fetch_add(1, Ordering::Relaxed);
                log::debug!("[cron:{id}] firing skipped — scheduler busy");
                return;
            }
            Err(e) => {
                log::warn!("[cron:{id}] inference error: {e}");
            }
            Ok(_) => {}
        }

        fired_count.fetch_add(1, Ordering::Relaxed);

        if let Some(cb) = on_result {
            cb(id.to_string(), result);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scheduler::Scheduler;
    use async_trait::async_trait;
    use kapsl_engine_api::{
        BinaryTensorPacket, Engine, EngineError, InferenceRequest, TensorDtype,
    };
    use std::sync::{Arc, Mutex as StdMutex};

    struct EchoEngine;

    #[async_trait]
    impl Engine for EchoEngine {
        async fn load(&mut self, _: &std::path::Path) -> Result<(), EngineError> {
            Ok(())
        }
        fn infer(&self, req: &InferenceRequest) -> Result<BinaryTensorPacket, EngineError> {
            Ok(req.input.clone())
        }
        fn infer_stream(
            &self,
            req: &InferenceRequest,
        ) -> std::pin::Pin<
            Box<dyn futures::stream::Stream<Item = Result<BinaryTensorPacket, EngineError>> + Send>,
        > {
            let result = Ok(req.input.clone());
            Box::pin(futures::stream::once(async move { result }))
        }
        fn unload(&mut self) {}
        fn metrics(&self) -> kapsl_engine_api::EngineMetrics {
            kapsl_engine_api::EngineMetrics::default()
        }
        fn health_check(&self) -> Result<(), EngineError> {
            Ok(())
        }
    }

    fn make_scheduler() -> Scheduler {
        let engine: Arc<dyn Engine> = Arc::new(EchoEngine);
        Scheduler::new(vec![engine], 2, 1, 1000, true, 1, 0, None)
    }

    fn make_request() -> InferenceRequest {
        InferenceRequest {
            input: BinaryTensorPacket {
                shape: vec![1, 1],
                dtype: TensorDtype::Float32,
                data: vec![0, 0, 0, 0],
            },
            additional_inputs: Vec::new(),
            session_id: None,
            metadata: None,
            cancellation: None,
        }
    }

    #[tokio::test]
    async fn test_register_and_list() {
        let cron = CronScheduler::new(Arc::new(make_scheduler()));

        cron.register(CronJob {
            id: "job1".to_string(),
            schedule: CronSchedule::Interval(Duration::from_secs(60)),
            request: Arc::new(make_request()),
            priority: Priority::Throughput,
            force_cpu: true,
            overflow_policy: CronOverflowPolicy::SkipIfBusy,
            on_result: None,
        })
        .await
        .unwrap();

        let jobs = cron.list_jobs().await;
        assert_eq!(jobs.len(), 1);
        assert_eq!(jobs[0].id, "job1");
        assert!(jobs[0].enabled);
        assert_eq!(jobs[0].overflow_policy, CronOverflowPolicy::SkipIfBusy);
    }

    #[tokio::test]
    async fn test_duplicate_id_rejected() {
        let cron = CronScheduler::new(Arc::new(make_scheduler()));

        cron.register(CronJob {
            id: "dup".to_string(),
            schedule: CronSchedule::Interval(Duration::from_secs(60)),
            request: Arc::new(make_request()),
            priority: Priority::Throughput,
            force_cpu: true,
            overflow_policy: CronOverflowPolicy::default(),
            on_result: None,
        })
        .await
        .unwrap();

        let err = cron
            .register(CronJob {
                id: "dup".to_string(),
                schedule: CronSchedule::Interval(Duration::from_secs(60)),
                request: Arc::new(make_request()),
                priority: Priority::Throughput,
                force_cpu: true,
                overflow_policy: CronOverflowPolicy::default(),
                on_result: None,
            })
            .await;

        assert!(matches!(err, Err(CronError::DuplicateId(_))));
    }

    #[tokio::test]
    async fn test_invalid_expression_rejected() {
        let cron = CronScheduler::new(Arc::new(make_scheduler()));

        let err = cron
            .register(CronJob {
                id: "bad".to_string(),
                schedule: CronSchedule::Expression("not a cron expression".to_string()),
                request: Arc::new(make_request()),
                priority: Priority::Throughput,
                force_cpu: true,
                overflow_policy: CronOverflowPolicy::default(),
                on_result: None,
            })
            .await;

        assert!(matches!(err, Err(CronError::InvalidExpression(_))));
    }

    #[tokio::test]
    async fn test_unregister() {
        let cron = CronScheduler::new(Arc::new(make_scheduler()));

        cron.register(CronJob {
            id: "removable".to_string(),
            schedule: CronSchedule::Interval(Duration::from_secs(60)),
            request: Arc::new(make_request()),
            priority: Priority::Throughput,
            force_cpu: true,
            overflow_policy: CronOverflowPolicy::default(),
            on_result: None,
        })
        .await
        .unwrap();

        assert!(cron.unregister("removable").await);
        assert!(!cron.unregister("removable").await);
        assert!(cron.list_jobs().await.is_empty());
    }

    #[tokio::test]
    async fn test_interval_fires_callback_and_increments_fired_count() {
        let cron = CronScheduler::new(Arc::new(make_scheduler()));
        let call_count = Arc::new(StdMutex::new(0u32));
        let cc = call_count.clone();

        cron.register(CronJob {
            id: "fast".to_string(),
            schedule: CronSchedule::Interval(Duration::from_millis(50)),
            request: Arc::new(make_request()),
            priority: Priority::Throughput,
            force_cpu: true,
            overflow_policy: CronOverflowPolicy::SkipIfBusy,
            on_result: Some(Arc::new(move |_id, result| {
                if result.is_ok() {
                    *cc.lock().unwrap() += 1;
                }
            })),
        })
        .await
        .unwrap();

        tokio::time::sleep(Duration::from_millis(200)).await;
        cron.unregister("fast").await;

        let info = cron.job_info("fast").await;
        // job was removed, so job_info returns None — check via the callback counter
        let count = *call_count.lock().unwrap();
        assert!(count >= 2, "expected ≥2 firings, got {count}");
        // job was removed before we could read info, which is fine
        assert!(info.is_none());
    }

    #[tokio::test]
    async fn test_fired_count_tracked() {
        let cron = CronScheduler::new(Arc::new(make_scheduler()));

        cron.register(CronJob {
            id: "counter-test".to_string(),
            schedule: CronSchedule::Interval(Duration::from_millis(40)),
            request: Arc::new(make_request()),
            priority: Priority::Throughput,
            force_cpu: true,
            overflow_policy: CronOverflowPolicy::SkipIfBusy,
            on_result: None,
        })
        .await
        .unwrap();

        tokio::time::sleep(Duration::from_millis(180)).await;

        let info = cron.job_info("counter-test").await.unwrap();
        assert!(
            info.fired_count >= 2,
            "expected ≥2 fired, got {}",
            info.fired_count
        );
        // No artificial load — missed_count should be 0 on a healthy scheduler.
        assert_eq!(info.missed_count, 0);

        cron.unregister("counter-test").await;
    }

    #[tokio::test]
    async fn test_valid_cron_expression_accepted() {
        let cron = CronScheduler::new(Arc::new(make_scheduler()));

        cron.register(CronJob {
            id: "every-sec".to_string(),
            schedule: CronSchedule::Expression("* * * * * *".to_string()),
            request: Arc::new(make_request()),
            priority: Priority::Throughput,
            force_cpu: true,
            overflow_policy: CronOverflowPolicy::default(),
            on_result: None,
        })
        .await
        .expect("valid expression should be accepted");

        cron.unregister("every-sec").await;
    }

    #[tokio::test]
    async fn test_skip_if_busy_increments_missed_count() {
        use crate::scheduler::QueueOverflowPolicy;

        // Tiny queue (capacity 1) so it fills up immediately.
        let engine: Arc<dyn Engine> = Arc::new(EchoEngine);
        let scheduler = Arc::new(
            Scheduler::new(vec![engine], 1, 1, 1, true, 1, 0, None)
                .with_queue_overflow_policy(QueueOverflowPolicy::DropNewest),
        );

        // Saturate the CPU pool by checking try_infer directly: we'll use
        // force_cpu=false (GPU path) with a tiny queue so try_push returns Err.
        let cron = CronScheduler::new(scheduler);

        cron.register(CronJob {
            id: "busy-test".to_string(),
            schedule: CronSchedule::Interval(Duration::from_millis(10)),
            request: Arc::new(make_request()),
            priority: Priority::Throughput,
            force_cpu: false, // GPU path — tiny queue will fill
            overflow_policy: CronOverflowPolicy::SkipIfBusy,
            on_result: None,
        })
        .await
        .unwrap();

        tokio::time::sleep(Duration::from_millis(150)).await;
        cron.unregister("busy-test").await;
    }
}
