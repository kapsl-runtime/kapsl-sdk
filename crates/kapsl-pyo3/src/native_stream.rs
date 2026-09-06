use std::{
    future::Future,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc, Mutex,
    },
    time::Duration,
};

use kapsl_communication::transport::protocol::{
    asynchronous, StreamResponse, DEFAULT_MAX_FRAME_PAYLOAD_BYTES,
};
use kapsl_engine_api::{BinaryTensorPacket, CancellationToken};
use pyo3::prelude::*;
use tokio::{
    io::{AsyncRead, AsyncWrite},
    runtime::Runtime,
    task::JoinHandle,
    time::Instant,
};

use crate::client::ClientError;

pub(crate) trait ConnectionIo: AsyncRead + AsyncWrite + Send + Unpin {}
impl<T: AsyncRead + AsyncWrite + Send + Unpin> ConnectionIo for T {}
pub(crate) type ClientConnection = Box<dyn ConnectionIo>;
type TensorResult = (Vec<u8>, Vec<i64>, String);

pub(crate) fn deadline(timeout_ms: Option<u64>) -> Result<Option<Instant>, ClientError> {
    timeout_ms
        .map(|ms| {
            Instant::now()
                .checked_add(Duration::from_millis(ms))
                .ok_or_else(|| ClientError::InvalidEndpoint("timeout_ms is too large".into()))
        })
        .transpose()
}

pub(crate) async fn with_deadline<T>(
    deadline: Option<Instant>,
    operation: impl Future<Output = Result<T, ClientError>>,
) -> Result<T, ClientError> {
    match deadline {
        Some(deadline) => {
            let result = tokio::time::timeout_at(deadline, operation).await;
            // Tokio polls the operation before its timer. A server closing at
            // the same deadline can otherwise surface as an EOF or even a
            // successful response after the client's budget has elapsed.
            if Instant::now() >= deadline {
                return Err(ClientError::Timeout);
            }
            result.map_err(|_| ClientError::Timeout)?
        }
        None => operation.await,
    }
}

struct StreamState {
    connection: Mutex<Option<ClientConnection>>,
    cancelled: CancellationToken,
    timed_out: AtomicBool,
    reading: AtomicBool,
}

impl StreamState {
    fn close(&self) {
        self.cancelled.cancel();
        self.connection.lock().unwrap().take();
    }

    fn terminal(&self) -> Result<Option<BinaryTensorPacket>, ClientError> {
        if self.timed_out.swap(false, Ordering::AcqRel) {
            Err(ClientError::Timeout)
        } else {
            Ok(None)
        }
    }
}

#[pyclass]
pub(crate) struct StreamIterator {
    runtime: Arc<Runtime>,
    state: Arc<StreamState>,
    deadline: Option<Instant>,
    watchdog: JoinHandle<()>,
}

impl StreamIterator {
    pub(crate) fn new(
        runtime: Arc<Runtime>,
        connection: ClientConnection,
        deadline: Option<Instant>,
        client_closed: CancellationToken,
    ) -> Self {
        let state = Arc::new(StreamState {
            connection: Mutex::new(Some(connection)),
            cancelled: CancellationToken::new(),
            timed_out: AtomicBool::new(false),
            reading: AtomicBool::new(false),
        });
        let weak = Arc::downgrade(&state);
        let cancelled = state.cancelled.clone();
        let watchdog = runtime.spawn(async move {
            let expired = tokio::select! {
                biased;
                _ = cancelled.cancelled() => return,
                _ = client_closed.cancelled() => false,
                _ = async {
                    match deadline {
                        Some(at) => tokio::time::sleep_until(at).await,
                        None => std::future::pending().await,
                    }
                } => true,
            };
            if let Some(state) = weak.upgrade() {
                state.timed_out.store(expired, Ordering::Release);
                state.close();
            }
        });
        Self {
            runtime,
            state,
            deadline,
            watchdog,
        }
    }

    fn next_packet(&self) -> Result<Option<BinaryTensorPacket>, ClientError> {
        if self.state.cancelled.is_cancelled() {
            return self.state.terminal();
        }
        if self.state.reading.swap(true, Ordering::AcqRel) {
            return Err(ClientError::Server(
                "The stream already has an active reader".into(),
            ));
        }
        struct Reading<'a>(&'a AtomicBool);
        impl Drop for Reading<'_> {
            fn drop(&mut self) {
                self.0.store(false, Ordering::Release);
            }
        }
        let _reading = Reading(&self.state.reading);
        let connection = self.state.connection.lock().unwrap().take();
        let Some(mut connection) = connection else {
            return self.state.terminal();
        };
        let result = self.runtime.block_on(async {
            tokio::select! {
                biased;
                _ = self.state.cancelled.cancelled() => Ok(None),
                result = with_deadline(self.deadline, async {
                    asynchronous::read_stream_packet(
                        connection.as_mut(), DEFAULT_MAX_FRAME_PAYLOAD_BYTES,
                    ).await.map_err(ClientError::from)
                }) => result.map(Some),
            }
        });
        match result {
            Ok(Some(StreamResponse::Chunk(packet))) => {
                let mut slot = self.state.connection.lock().unwrap();
                if self.state.cancelled.is_cancelled() {
                    return self.state.terminal();
                }
                *slot = Some(connection);
                Ok(Some(packet))
            }
            Ok(_) => {
                self.state.close();
                self.state.terminal()
            }
            Err(error) => {
                self.state.close();
                self.state.timed_out.store(false, Ordering::Release);
                Err(error)
            }
        }
    }
}

impl Drop for StreamIterator {
    fn drop(&mut self) {
        self.state.close();
        self.watchdog.abort();
    }
}

#[pymethods]
impl StreamIterator {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(&self, py: Python<'_>) -> PyResult<Option<Vec<u8>>> {
        py.detach(|| self.next_packet())
            .map(|packet| packet.map(|packet| packet.data))
            .map_err(PyErr::from)
    }

    fn next_tensor(&self, py: Python<'_>) -> PyResult<Option<TensorResult>> {
        py.detach(|| self.next_packet())
            .map(|packet| {
                packet.map(|packet| (packet.data, packet.shape, packet.dtype.as_str().to_owned()))
            })
            .map_err(PyErr::from)
    }

    fn cancel(&self) -> bool {
        let active = !self.state.cancelled.is_cancelled();
        self.state.close();
        self.watchdog.abort();
        active
    }

    fn close(&self) {
        self.cancel();
    }

    #[getter]
    fn closed(&self) -> bool {
        self.state.cancelled.is_cancelled()
    }
}
