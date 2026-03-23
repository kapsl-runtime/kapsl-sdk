use async_trait::async_trait;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use kapsl_engine_api::{
    BinaryTensorPacket, Engine, EngineError, InferenceRequest, TensorDtype,
};
use kapsl_scheduler::{
    cron_scheduler::{CronJob, CronOverflowPolicy, CronSchedule, CronScheduler},
    priority::Priority,
    scheduler::Scheduler,
};
use std::sync::Arc;
use std::time::Duration;
use tokio::runtime::Runtime;

// ---------------------------------------------------------------------------
// Shared test helpers
// ---------------------------------------------------------------------------

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

fn make_scheduler() -> Arc<Scheduler> {
    let engine: Arc<dyn Engine> = Arc::new(EchoEngine);
    Arc::new(Scheduler::new(vec![engine], 2, 1, 1000, true, 1, 0, None))
}

fn make_request() -> Arc<InferenceRequest> {
    Arc::new(InferenceRequest {
        input: BinaryTensorPacket {
            shape: vec![1, 1],
            dtype: TensorDtype::Float32,
            data: vec![0, 0, 0, 0],
        },
        additional_inputs: Vec::new(),
        session_id: None,
        metadata: None,
        cancellation: None,
    })
}

fn make_job(id: &str) -> CronJob {
    CronJob {
        id: id.to_string(),
        schedule: CronSchedule::Interval(Duration::from_secs(3600)),
        request: make_request(),
        priority: Priority::Throughput,
        force_cpu: true,
        overflow_policy: CronOverflowPolicy::SkipIfBusy,
        on_result: None,
    }
}

// ---------------------------------------------------------------------------
// Benchmarks
// ---------------------------------------------------------------------------

/// Measures the round-trip cost of registering a single job then unregistering it.
fn bench_register_unregister(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    c.bench_function("register_unregister", |b| {
        b.iter(|| {
            rt.block_on(async {
                let cron = CronScheduler::new(make_scheduler());
                cron.register(make_job("bench")).await.unwrap();
                cron.unregister("bench").await;
            });
        });
    });
}

/// Measures `list_jobs` under increasing registry sizes (read-path RwLock).
fn bench_list_jobs(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("list_jobs");

    for n in [1usize, 10, 50, 100] {
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, &n| {
            let cron = rt.block_on(async {
                let cron = CronScheduler::new(make_scheduler());
                for i in 0..n {
                    cron.register(make_job(&format!("job-{i}"))).await.unwrap();
                }
                cron
            });

            b.iter(|| rt.block_on(async { cron.list_jobs().await }));
        });
    }

    group.finish();
}

/// Measures `job_info` lookup (single-key read, common monitoring hot path).
fn bench_job_info(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    let cron = rt.block_on(async {
        let cron = CronScheduler::new(make_scheduler());
        for i in 0..50 {
            cron.register(make_job(&format!("job-{i}"))).await.unwrap();
        }
        cron
    });

    c.bench_function("job_info_50_jobs", |b| {
        b.iter(|| rt.block_on(async { cron.job_info("job-25").await }));
    });
}

/// Measures the end-to-end firing latency for a high-frequency interval job
/// (wall-clock: 10 ms interval, measure how many firings occur in 500 ms).
fn bench_firing_throughput(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    c.bench_function("firing_throughput_10ms", |b| {
        b.iter(|| {
            rt.block_on(async {
                let cron = CronScheduler::new(make_scheduler());
                cron.register(make_job("fast")).await.unwrap();
                tokio::time::sleep(Duration::from_millis(500)).await;
                let info = cron.job_info("fast").await.unwrap();
                cron.unregister("fast").await;
                info.fired_count
            })
        });
    });
}

criterion_group!(
    benches,
    bench_register_unregister,
    bench_list_jobs,
    bench_job_info,
    bench_firing_throughput,
);
criterion_main!(benches);
