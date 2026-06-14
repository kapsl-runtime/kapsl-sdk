//! PyTorch sidecar backend.
//!
//! Runs PyTorch-family models (HuggingFace `transformers`, custom code) in an
//! **out-of-process Python worker**, with the Rust side acting as a thin
//! `Engine` that spawns the worker and forwards text-generation requests over a
//! line-delimited JSON protocol on a unix domain socket.
//!
//! # Why a sidecar
//!
//! libtorch/`tch-rs` can only run TorchScript/traced models and would not give
//! us the PyTorch *ecosystem*. A Python sidecar runs any PyTorch model, reuses
//! the same process-isolation shape the runtime already uses for isolated
//! workers, and keeps native crashes out of the engine process. The trade-off:
//! the worker has its own CUDA context and KV cache, so it does **not** share
//! the in-process KV block pool — it integrates with the cross-model
//! `GlobalKvScheduler` at the admission level only (wired by the runtime).
//!
//! # Protocol (one connection per request)
//!
//! Request (single line, then `\n`):
//! ```json
//! {"prompt":"...","max_new_tokens":256,"temperature":0.7,"top_p":0.9,
//!  "top_k":40,"repetition_penalty":1.1,"seed":null,"stop_token_ids":[],
//!  "session_id":null,"request_id":null}
//! ```
//! Response (one JSON object per line, streamed):
//! ```json
//! {"type":"chunk","text":"..."}
//! {"type":"done"}
//! {"type":"error","message":"..."}
//! ```
//!
//! # Worker command
//!
//! Defaults to `python3 -m kapsl_pytorch_worker`; override the whole argv with
//! the `KAPSL_PYTORCH_WORKER` env var (space-separated). The runtime appends
//! `--model <path> --socket <path> --device <id>`.

use std::path::Path;
use std::process::{Child, Command};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use kapsl_engine_api::{
    BinaryTensorPacket, Engine, EngineError, EngineMetrics, EngineModelInfo, EngineStream,
    InferenceRequest, TensorDtype,
};

/// How long to wait for the worker to bind its socket after spawn (model load
/// can be slow for large PyTorch models).
const READY_TIMEOUT: Duration = Duration::from_secs(120);
const DEFAULT_MAX_NEW_TOKENS: u32 = 512;

/// Monotonic counter so concurrent backends get distinct socket paths.
static SOCKET_COUNTER: AtomicU64 = AtomicU64::new(0);

/// A spawned worker process and the socket it serves on. Killed on drop.
struct Sidecar {
    child: Child,
    socket_path: String,
}

impl Drop for Sidecar {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
        let _ = std::fs::remove_file(&self.socket_path);
    }
}

/// Engine backend that proxies inference to an out-of-process PyTorch worker.
pub struct PyTorchBackend {
    device_id: i32,
    sidecar: Mutex<Option<Sidecar>>,
}

impl PyTorchBackend {
    pub fn new(device_id: i32) -> Result<Self, EngineError> {
        Ok(Self {
            device_id,
            sidecar: Mutex::new(None),
        })
    }

    fn socket_path_for(device_id: i32) -> String {
        let n = SOCKET_COUNTER.fetch_add(1, Ordering::Relaxed);
        format!(
            "/tmp/kapsl-pytorch-{}-{}-{}.sock",
            std::process::id(),
            device_id,
            n
        )
    }

    /// Worker argv. `KAPSL_PYTORCH_WORKER` (space-separated) overrides the
    /// default `python3 -m kapsl_pytorch_worker`.
    fn worker_argv() -> Vec<String> {
        if let Ok(cmd) = std::env::var("KAPSL_PYTORCH_WORKER") {
            let parts: Vec<String> = cmd.split_whitespace().map(|s| s.to_string()).collect();
            if !parts.is_empty() {
                return parts;
            }
        }
        vec![
            "python3".to_string(),
            "-m".to_string(),
            "kapsl_pytorch_worker".to_string(),
        ]
    }

    fn current_socket(&self) -> Result<String, EngineError> {
        self.sidecar
            .lock()
            .unwrap()
            .as_ref()
            .map(|s| s.socket_path.clone())
            .ok_or(EngineError::ModelNotLoaded)
    }

    /// Serialize a generation request to the worker protocol.
    fn build_request_json(req: &InferenceRequest) -> Result<String, EngineError> {
        let prompt = String::from_utf8_lossy(&req.input.data).to_string();
        let md = req.metadata.as_ref();
        let value = serde_json::json!({
            "prompt": prompt,
            "session_id": req.session_id,
            "request_id": md.and_then(|m| m.request_id.clone()),
            "max_new_tokens": md.and_then(|m| m.max_new_tokens).unwrap_or(DEFAULT_MAX_NEW_TOKENS),
            "min_new_tokens": md.and_then(|m| m.min_new_tokens),
            "temperature": md.and_then(|m| m.temperature),
            "top_p": md.and_then(|m| m.top_p),
            "top_k": md.and_then(|m| m.top_k),
            "repetition_penalty": md.and_then(|m| m.repetition_penalty),
            "seed": md.and_then(|m| m.seed),
            "stop_token_ids": md.and_then(|m| m.stop_token_ids.clone()),
        });
        serde_json::to_string(&value)
            .map_err(|e| EngineError::backend(format!("PyTorch request serialize failed: {e}")))
    }
}

/// Build a UTF-8 text packet matching the LLM text-streaming contract
/// (`shape = [1, len]`, `dtype = Utf8`).
fn text_packet(text: &str) -> BinaryTensorPacket {
    let data = text.as_bytes().to_vec();
    BinaryTensorPacket {
        shape: vec![1, data.len() as i64],
        dtype: TensorDtype::Utf8,
        data,
    }
}

#[cfg(unix)]
fn socket_ready(path: &str) -> bool {
    Path::new(path).exists() && std::os::unix::net::UnixStream::connect(path).is_ok()
}

/// Connect to the worker, send `request_json`, and drive the streamed response,
/// invoking `emit` for each item. `emit` returns `false` to stop early (e.g. the
/// receiver was dropped). Runs synchronously; call from a dedicated thread.
#[cfg(unix)]
fn run_sidecar_request(
    socket_path: &str,
    request_json: &str,
    cancel: Option<kapsl_engine_api::CancellationToken>,
    mut emit: impl FnMut(Result<BinaryTensorPacket, EngineError>) -> bool,
) {
    use std::io::{BufRead, BufReader, Write};
    use std::os::unix::net::UnixStream;

    let conn = match UnixStream::connect(socket_path) {
        Ok(c) => c,
        Err(e) => {
            emit(Err(EngineError::backend(format!(
                "PyTorch IPC connect failed: {e}"
            ))));
            return;
        }
    };
    let mut writer = match conn.try_clone() {
        Ok(w) => w,
        Err(e) => {
            emit(Err(EngineError::backend(format!(
                "PyTorch IPC clone failed: {e}"
            ))));
            return;
        }
    };
    if let Err(e) = writer
        .write_all(request_json.as_bytes())
        .and_then(|_| writer.write_all(b"\n"))
        .and_then(|_| writer.flush())
    {
        emit(Err(EngineError::backend(format!(
            "PyTorch IPC write failed: {e}"
        ))));
        return;
    }

    let reader = BufReader::new(conn);
    for line in reader.lines() {
        if cancel.as_ref().map(|c| c.is_cancelled()).unwrap_or(false) {
            emit(Err(EngineError::Cancelled {
                message: "Request cancelled".to_string(),
            }));
            return;
        }
        let line = match line {
            Ok(l) => l,
            Err(e) => {
                emit(Err(EngineError::backend(format!(
                    "PyTorch IPC read failed: {e}"
                ))));
                return;
            }
        };
        if line.trim().is_empty() {
            continue;
        }
        let value: serde_json::Value = match serde_json::from_str(&line) {
            Ok(v) => v,
            Err(e) => {
                emit(Err(EngineError::backend(format!(
                    "PyTorch IPC malformed response: {e}"
                ))));
                return;
            }
        };
        match value.get("type").and_then(|t| t.as_str()) {
            Some("chunk") => {
                let text = value.get("text").and_then(|t| t.as_str()).unwrap_or("");
                if !emit(Ok(text_packet(text))) {
                    return;
                }
            }
            Some("done") => return,
            Some("error") => {
                let msg = value
                    .get("message")
                    .and_then(|m| m.as_str())
                    .unwrap_or("unknown error");
                emit(Err(EngineError::backend(format!(
                    "PyTorch worker error: {msg}"
                ))));
                return;
            }
            _ => { /* ignore unknown control frames for forward-compat */ }
        }
    }
}

#[async_trait]
impl Engine for PyTorchBackend {
    async fn load(&mut self, model_path: &Path) -> Result<(), EngineError> {
        #[cfg(not(unix))]
        {
            let _ = model_path;
            return Err(EngineError::backend(
                "PyTorch sidecar backend is only supported on unix platforms",
            ));
        }
        #[cfg(unix)]
        {
            let socket_path = Self::socket_path_for(self.device_id);
            if Path::new(&socket_path).exists() {
                let _ = std::fs::remove_file(&socket_path);
            }

            let argv = Self::worker_argv();
            let mut command = Command::new(&argv[0]);
            command
                .args(&argv[1..])
                .arg("--model")
                .arg(model_path)
                .arg("--socket")
                .arg(&socket_path)
                .arg("--device")
                .arg(self.device_id.to_string());
            let child = command.spawn().map_err(|e| {
                EngineError::backend(format!(
                    "failed to spawn PyTorch worker '{}': {e}",
                    argv[0]
                ))
            })?;

            // Owns the child: if we return Err before storing it, Drop kills it.
            let mut sidecar = Sidecar {
                child,
                socket_path: socket_path.clone(),
            };

            let deadline = Instant::now() + READY_TIMEOUT;
            loop {
                if let Some(status) = sidecar.child.try_wait().ok().flatten() {
                    return Err(EngineError::backend(format!(
                        "PyTorch worker exited before ready: {status}"
                    )));
                }
                if socket_ready(&socket_path) {
                    break;
                }
                if Instant::now() >= deadline {
                    return Err(EngineError::backend(
                        "timed out waiting for PyTorch worker socket".to_string(),
                    ));
                }
                tokio::time::sleep(Duration::from_millis(200)).await;
            }

            *self.sidecar.lock().unwrap() = Some(sidecar);
            log::info!(
                "PyTorchBackend: worker ready (device {}, socket {})",
                self.device_id,
                socket_path
            );
            Ok(())
        }
    }

    fn infer(&self, req: &InferenceRequest) -> Result<BinaryTensorPacket, EngineError> {
        #[cfg(not(unix))]
        {
            let _ = req;
            return Err(EngineError::backend(
                "PyTorch sidecar backend is only supported on unix platforms",
            ));
        }
        #[cfg(unix)]
        {
            let socket_path = self.current_socket()?;
            let request_json = Self::build_request_json(req)?;
            let cancel = req.cancellation.clone();

            let mut text = String::new();
            let mut error: Option<EngineError> = None;
            run_sidecar_request(&socket_path, &request_json, cancel, |item| match item {
                Ok(packet) => {
                    text.push_str(&String::from_utf8_lossy(&packet.data));
                    true
                }
                Err(e) => {
                    error = Some(e);
                    false
                }
            });
            if let Some(e) = error {
                return Err(e);
            }
            Ok(text_packet(&text))
        }
    }

    fn infer_stream(&self, req: &InferenceRequest) -> EngineStream {
        #[cfg(not(unix))]
        {
            let _ = req;
            return Box::pin(futures::stream::once(async {
                Err(EngineError::backend(
                    "PyTorch sidecar backend is only supported on unix platforms",
                ))
            }));
        }
        #[cfg(unix)]
        {
            let socket_path = match self.current_socket() {
                Ok(s) => s,
                Err(e) => return Box::pin(futures::stream::once(async move { Err(e) })),
            };
            let request_json = match Self::build_request_json(req) {
                Ok(s) => s,
                Err(e) => return Box::pin(futures::stream::once(async move { Err(e) })),
            };
            let cancel = req.cancellation.clone();
            let (tx, rx) =
                tokio::sync::mpsc::channel::<Result<BinaryTensorPacket, EngineError>>(64);

            std::thread::spawn(move || {
                run_sidecar_request(&socket_path, &request_json, cancel, |item| {
                    tx.blocking_send(item).is_ok()
                });
            });

            Box::pin(futures::stream::unfold(rx, |mut rx| async move {
                rx.recv().await.map(|item| (item, rx))
            }))
        }
    }

    fn unload(&mut self) {
        // Dropping the Sidecar kills the worker and removes the socket file.
        let _ = self.sidecar.lock().unwrap().take();
        log::info!("PyTorchBackend: unloaded (device {})", self.device_id);
    }

    fn metrics(&self) -> EngineMetrics {
        EngineMetrics::default()
    }

    fn model_info(&self) -> Option<EngineModelInfo> {
        if self.sidecar.lock().unwrap().is_none() {
            return None;
        }
        Some(EngineModelInfo {
            input_names: vec!["prompt".into()],
            output_names: vec!["text".into()],
            input_shapes: vec![vec![-1]],
            output_shapes: vec![vec![-1]],
            input_dtypes: vec!["string".into()],
            output_dtypes: vec!["string".into()],
            framework: Some("pytorch".into()),
            model_version: None,
            peak_concurrency: None,
        })
    }

    fn health_check(&self) -> Result<(), EngineError> {
        let mut guard = self.sidecar.lock().unwrap();
        match guard.as_mut() {
            Some(sidecar) => match sidecar.child.try_wait().ok().flatten() {
                Some(status) => Err(EngineError::backend(format!(
                    "PyTorch worker exited: {status}"
                ))),
                None => Ok(()),
            },
            None => Err(EngineError::ModelNotLoaded),
        }
    }
}
