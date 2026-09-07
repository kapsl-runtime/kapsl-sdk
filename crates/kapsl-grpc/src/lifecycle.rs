//! Keep cancellation, deadlines, and access logging alive with the response body.

use std::{
    future::Future,
    pin::Pin,
    sync::{Arc, Mutex},
    task::{Context, Poll},
    time::Duration,
};

use crate::EngineStream;
use kapsl_engine_api::CancellationToken;
use tokio::{task::JoinHandle, time::Instant};
use tonic::{Code, Status};

type StreamSlot = Arc<Mutex<Option<EngineStream>>>;

pub(crate) struct SharedStream(StreamSlot);

impl futures::Stream for SharedStream {
    type Item = Result<kapsl_engine_api::BinaryTensorPacket, kapsl_engine_api::EngineError>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        match self.0.lock().expect("stream slot poisoned").as_mut() {
            Some(stream) => stream.as_mut().poll_next(cx),
            None => Poll::Ready(None),
        }
    }
}

pub(crate) struct Call {
    method: &'static str,
    started: Instant,
    deadline: Option<Instant>,
    pub(crate) cancellation: CancellationToken,
    shutdown: tokio_util::sync::CancellationToken,
    watchdog: Option<JoinHandle<()>>,
    code: Code,
    stream: StreamSlot,
}

impl Call {
    pub(crate) fn new(
        method: &'static str,
        shutdown: tokio_util::sync::CancellationToken,
        started: Instant,
    ) -> Self {
        Self {
            method,
            started,
            deadline: None,
            cancellation: CancellationToken::new(),
            shutdown,
            watchdog: None,
            code: Code::Cancelled,
            stream: Arc::new(Mutex::new(None)),
        }
    }

    pub(crate) fn set_timeout(&mut self, timeout: Duration) -> Result<(), Status> {
        let end = self
            .started
            .checked_add(timeout)
            .ok_or_else(|| Status::invalid_argument("Timeout exceeds supported range"))?;
        self.deadline = Some(self.deadline.map_or(end, |existing| existing.min(end)));
        Ok(())
    }

    pub(crate) fn activate(&mut self) {
        let deadline = self.deadline;
        let shutdown = self.shutdown.clone();
        let cancellation = self.cancellation.clone();
        let stream = self.stream.clone();
        // A timer independent of response polling also cancels generation when
        // a slow client has exhausted the HTTP/2 flow-control window.
        self.watchdog = Some(tokio::spawn(async move {
            tokio::select! {
                _ = shutdown.cancelled() => {},
                _ = async {
                    match deadline {
                        Some(end) => tokio::time::sleep_until(end).await,
                        None => std::future::pending().await,
                    }
                } => {},
            }
            cancellation.cancel();
            // Release scheduler/admission guards even when HTTP/2 flow control
            // has stopped polling the response body.
            stream.lock().expect("stream slot poisoned").take();
        }));
    }

    pub(crate) fn attach_stream(&self, stream: EngineStream) -> SharedStream {
        let mut slot = self.stream.lock().expect("stream slot poisoned");
        if !self.cancellation.is_cancelled() {
            *slot = Some(stream);
        }
        SharedStream(self.stream.clone())
    }

    pub(crate) async fn wait<T>(
        &self,
        future: impl Future<Output = Result<T, Status>>,
    ) -> Result<T, Status> {
        tokio::select! {
            biased;
            _ = async {
                match self.deadline {
                    Some(end) => tokio::time::sleep_until(end).await,
                    None => std::future::pending().await,
                }
            } => Err(Status::deadline_exceeded("Inference deadline exceeded")),
            _ = self.cancellation.cancelled() => Err(self.cancellation_status()),
            result = future => result,
        }
    }

    fn cancellation_status(&self) -> Status {
        if self.deadline.is_some_and(|end| Instant::now() >= end) {
            Status::deadline_exceeded("Inference deadline exceeded")
        } else if self.shutdown.is_cancelled() {
            Status::unavailable("Server is shutting down")
        } else {
            Status::cancelled("Inference cancelled")
        }
    }

    pub(crate) fn result<T>(&mut self, result: Result<T, Status>) -> Result<T, Status> {
        self.code = result
            .as_ref()
            .map_or_else(|error| error.code(), |_| Code::Ok);
        result
    }
}

impl Drop for Call {
    fn drop(&mut self) {
        self.cancellation.cancel();
        self.stream.lock().expect("stream slot poisoned").take();
        if let Some(task) = self.watchdog.take() {
            task.abort();
        }
        // Only controlled method/status values are logged. No metadata,
        // request/session IDs, model names, tensor contents, or backend errors.
        log::info!(target: "kapsl::access",
            "protocol=grpc method={} grpc_status={} elapsed_us={}",
            self.method, self.code as i32, self.started.elapsed().as_micros());
    }
}

pub(crate) fn parse_timeout(value: &str) -> Result<Duration, Status> {
    let invalid = || Status::invalid_argument("Invalid grpc-timeout metadata");
    if !(2..=9).contains(&value.len()) {
        return Err(invalid());
    }
    let (digits, unit) = value.split_at(value.len() - 1);
    if !digits.bytes().all(|byte| byte.is_ascii_digit()) {
        return Err(invalid());
    }
    let amount: u64 = digits.parse().map_err(|_| invalid())?;
    match unit {
        "H" => Ok(Duration::from_secs(amount * 3600)),
        "M" => Ok(Duration::from_secs(amount * 60)),
        "S" => Ok(Duration::from_secs(amount)),
        "m" => Ok(Duration::from_millis(amount)),
        "u" => Ok(Duration::from_micros(amount)),
        "n" => Ok(Duration::from_nanos(amount)),
        _ => Err(invalid()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, Ordering};

    struct Lease(Arc<AtomicBool>);
    impl Drop for Lease {
        fn drop(&mut self) {
            self.0.store(true, Ordering::SeqCst);
        }
    }

    #[tokio::test]
    async fn deadline_releases_stream_lease_when_consumer_stops_polling() {
        let released = Arc::new(AtomicBool::new(false));
        let lease = Lease(released.clone());
        let stream = Box::pin(futures::stream::poll_fn(move |_| {
            let _hold = &lease;
            Poll::Pending
        }));
        let mut call = Call::new(
            "InferStream",
            tokio_util::sync::CancellationToken::new(),
            Instant::now(),
        );
        call.set_timeout(Duration::from_millis(20)).unwrap();
        call.activate();
        let _unpolled_response = call.attach_stream(stream);
        tokio::time::timeout(Duration::from_secs(1), call.cancellation.cancelled())
            .await
            .unwrap();
        assert!(released.load(Ordering::SeqCst));
    }
}
