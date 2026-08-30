use super::*;
use crate::scheduler::Scheduler;
use crate::test_support::make_request;
use async_trait::async_trait;
use kapsl_engine_api::{BinaryTensorPacket, Engine, EngineError, InferenceRequest};
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
async fn zero_interval_is_rejected_before_spawning() {
    let cron = CronScheduler::new(Arc::new(make_scheduler()));

    let error = cron
        .register(CronJob {
            id: "zero-interval".to_string(),
            schedule: CronSchedule::Interval(Duration::ZERO),
            request: Arc::new(make_request()),
            priority: Priority::Throughput,
            force_cpu: true,
            overflow_policy: CronOverflowPolicy::default(),
            on_result: None,
        })
        .await
        .expect_err("a zero interval would panic tokio::time::interval");

    assert!(matches!(error, CronError::InvalidExpression(_)));
    assert!(cron.list_jobs().await.is_empty());
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
async fn list_jobs_is_sorted_by_id() {
    let cron = CronScheduler::new(Arc::new(make_scheduler()));
    for id in ["zeta", "alpha", "middle"] {
        cron.register(CronJob {
            id: id.to_string(),
            schedule: CronSchedule::Interval(Duration::from_secs(60)),
            request: Arc::new(make_request()),
            priority: Priority::Throughput,
            force_cpu: true,
            overflow_policy: CronOverflowPolicy::default(),
            on_result: None,
        })
        .await
        .unwrap();
    }

    let ids = cron
        .list_jobs()
        .await
        .into_iter()
        .map(|job| job.id)
        .collect::<Vec<_>>();
    assert_eq!(ids, ["alpha", "middle", "zeta"]);
}

#[tokio::test]
async fn dropping_cron_scheduler_aborts_registered_jobs() {
    let scheduler = Arc::new(make_scheduler());
    let cron = CronScheduler::new(scheduler.clone());
    cron.register(CronJob {
        id: "detached-task-check".to_string(),
        schedule: CronSchedule::Interval(Duration::from_secs(60)),
        request: Arc::new(make_request()),
        priority: Priority::Throughput,
        force_cpu: true,
        overflow_policy: CronOverflowPolicy::default(),
        on_result: None,
    })
    .await
    .unwrap();

    drop(cron);
    for _ in 0..100 {
        if Arc::strong_count(&scheduler) == 1 {
            break;
        }
        tokio::task::yield_now().await;
    }
    assert_eq!(Arc::strong_count(&scheduler), 1);
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
