use std::collections::HashMap;
use std::io::{self, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};

use bytes::Bytes;
use kapsl_rag_sdk::protocol::{ConnectorRequest, ConnectorRequestKind, ConnectorResponse};
use kapsl_rag_sdk::read_frame;
use wasmtime::{Engine, Linker, Module, Store};
use wasmtime_wasi::p2::pipe::{MemoryInputPipe, MemoryOutputPipe};
use wasmtime_wasi::preview1::{add_to_linker_sync, WasiP1Ctx};
use wasmtime_wasi::{DirPerms, FilePerms, WasiCtxBuilder};

#[derive(thiserror::Error, Debug)]
pub enum RuntimeError {
    #[error("io error: {0}")]
    Io(String),
    #[error("serialization error: {0}")]
    Serialization(String),
    #[error("connector exited")]
    ConnectorExited,
    #[error("wasm error: {0}")]
    Wasm(String),
}

impl From<io::Error> for RuntimeError {
    fn from(err: io::Error) -> Self {
        RuntimeError::Io(err.to_string())
    }
}

impl From<serde_json::Error> for RuntimeError {
    fn from(err: serde_json::Error) -> Self {
        RuntimeError::Serialization(err.to_string())
    }
}

pub trait ConnectorRuntime {
    fn send(&mut self, request: ConnectorRequest) -> Result<ConnectorResponse, RuntimeError>;
    fn close(&mut self) -> Result<(), RuntimeError>;
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct WasiPermissions {
    pub preopen_dirs: Vec<PreopenDir>,
    pub env: HashMap<String, String>,
}

impl WasiPermissions {
    pub fn allow_dir(
        mut self,
        host_path: impl Into<PathBuf>,
        guest_path: impl Into<String>,
        read_only: bool,
    ) -> Self {
        self.preopen_dirs.push(PreopenDir {
            host_path: host_path.into(),
            guest_path: guest_path.into(),
            read_only,
        });
        self
    }

    pub fn with_env(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.env.insert(key.into(), value.into());
        self
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PreopenDir {
    pub host_path: PathBuf,
    pub guest_path: String,
    pub read_only: bool,
}

pub struct ConnectorClient<R: ConnectorRuntime> {
    runtime: R,
    next_id: u64,
}

impl<R: ConnectorRuntime> ConnectorClient<R> {
    pub fn new(runtime: R) -> Self {
        Self {
            runtime,
            next_id: 1,
        }
    }

    pub fn request(
        &mut self,
        kind: ConnectorRequestKind,
    ) -> Result<ConnectorResponse, RuntimeError> {
        let id = format!("req-{}", self.next_id);
        self.next_id = self
            .next_id
            .checked_add(1)
            .ok_or_else(|| RuntimeError::Serialization("request id space exhausted".to_string()))?;
        let request = ConnectorRequest::new(id.clone(), kind);
        let response = self.runtime.send(request)?;
        if response.id != id {
            return Err(RuntimeError::Serialization(format!(
                "response id {:?} does not match request id {id:?}",
                response.id
            )));
        }
        Ok(response)
    }

    pub fn shutdown(&mut self) -> Result<(), RuntimeError> {
        self.runtime.close()
    }
}

pub struct SidecarConnectorRuntime {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
}

impl SidecarConnectorRuntime {
    pub fn spawn(path: &Path) -> Result<Self, RuntimeError> {
        let mut child = Command::new(path)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()?;

        let stdin = child.stdin.take().ok_or(RuntimeError::ConnectorExited)?;
        let stdout = child.stdout.take().ok_or(RuntimeError::ConnectorExited)?;
        Ok(Self {
            child,
            stdin,
            stdout: BufReader::new(stdout),
        })
    }
}

impl ConnectorRuntime for SidecarConnectorRuntime {
    fn send(&mut self, request: ConnectorRequest) -> Result<ConnectorResponse, RuntimeError> {
        let json = serde_json::to_string(&request)?;
        self.stdin.write_all(json.as_bytes())?;
        self.stdin.write_all(b"\n")?;
        self.stdin.flush()?;

        let Some(frame) = read_frame(&mut self.stdout)
            .map_err(|error| RuntimeError::Serialization(error.to_string()))?
        else {
            return Err(RuntimeError::ConnectorExited);
        };
        let response = serde_json::from_slice(&frame)?;
        Ok(response)
    }

    fn close(&mut self) -> Result<(), RuntimeError> {
        terminate_child(&mut self.child)?;
        Ok(())
    }
}

impl Drop for SidecarConnectorRuntime {
    fn drop(&mut self) {
        let _ = terminate_child(&mut self.child);
    }
}

fn terminate_child(child: &mut Child) -> io::Result<()> {
    if child.try_wait()?.is_none() {
        child.kill()?;
        let _ = child.wait()?;
    }
    Ok(())
}

pub struct WasmConnectorRuntime {
    engine: Engine,
    module: Module,
    permissions: WasiPermissions,
}

impl WasmConnectorRuntime {
    pub fn spawn(path: &Path) -> Result<Self, RuntimeError> {
        Self::spawn_with_permissions(path, WasiPermissions::default())
    }

    pub fn spawn_with_permissions(
        path: &Path,
        permissions: WasiPermissions,
    ) -> Result<Self, RuntimeError> {
        let engine = Engine::default();
        let module =
            Module::from_file(&engine, path).map_err(|e| RuntimeError::Wasm(e.to_string()))?;
        Ok(Self {
            engine,
            module,
            permissions,
        })
    }

    fn run_once(&self, input: &str) -> Result<String, RuntimeError> {
        // Run the WASM connector to completion for this request. This keeps the
        // implementation simple and sandboxed, at the cost of per-request startup.
        let mut linker = Linker::<WasiP1Ctx>::new(&self.engine);
        add_to_linker_sync(&mut linker, |ctx| ctx)
            .map_err(|e| RuntimeError::Wasm(e.to_string()))?;

        let stdin = MemoryInputPipe::new(Bytes::from(input.as_bytes().to_vec()));
        let stdout = MemoryOutputPipe::new(4 * 1024 * 1024);
        let stderr = MemoryOutputPipe::new(256 * 1024);

        let mut builder = WasiCtxBuilder::new();
        let _ = builder.stdin(stdin);
        let _ = builder.stdout(stdout.clone());
        let _ = builder.stderr(stderr.clone());

        for (key, value) in &self.permissions.env {
            crate::validation::env_key_value(key, value).map_err(RuntimeError::Wasm)?;
            let _ = builder.env(key, value);
        }

        for dir in &self.permissions.preopen_dirs {
            crate::validation::guest_path(&dir.guest_path).map_err(RuntimeError::Wasm)?;
            crate::validation::host_path(&dir.host_path).map_err(RuntimeError::Wasm)?;
            let (dir_perms, file_perms) = perms_for(dir.read_only);
            builder
                .preopened_dir(&dir.host_path, &dir.guest_path, dir_perms, file_perms)
                .map_err(|e| RuntimeError::Wasm(e.to_string()))?;
        }

        let wasi = builder.build_p1();
        let mut store = Store::new(&self.engine, wasi);
        let instance = linker
            .instantiate(&mut store, &self.module)
            .map_err(|e| RuntimeError::Wasm(e.to_string()))?;

        let start = instance
            .get_typed_func::<(), ()>(&mut store, "_start")
            .map_err(|e| RuntimeError::Wasm(e.to_string()))?;
        start
            .call(&mut store, ())
            .map_err(|e| RuntimeError::Wasm(e.to_string()))?;

        let output = stdout.contents();
        let output =
            String::from_utf8(output.to_vec()).map_err(|e| RuntimeError::Wasm(e.to_string()))?;
        Ok(output)
    }
}

impl ConnectorRuntime for WasmConnectorRuntime {
    fn send(&mut self, request: ConnectorRequest) -> Result<ConnectorResponse, RuntimeError> {
        let payload = serde_json::to_string(&request)?;
        let output = self.run_once(&format!("{payload}\n"))?;
        let mut last_line = None;
        for line in output.lines() {
            let trimmed = line.trim();
            if !trimmed.is_empty() {
                last_line = Some(trimmed.to_string());
            }
        }
        let last_line = last_line.ok_or(RuntimeError::ConnectorExited)?;
        let response = serde_json::from_str(&last_line)?;
        Ok(response)
    }

    fn close(&mut self) -> Result<(), RuntimeError> {
        Ok(())
    }
}

fn perms_for(read_only: bool) -> (DirPerms, FilePerms) {
    let dir_perms = if read_only {
        DirPerms::READ
    } else {
        DirPerms::READ | DirPerms::MUTATE
    };
    let file_perms = if read_only {
        FilePerms::READ
    } else {
        FilePerms::READ | FilePerms::WRITE
    };
    (dir_perms, file_perms)
}

#[cfg(test)]
mod tests {
    use super::*;
    use kapsl_rag_sdk::protocol::{ConnectorResponseKind, ConnectorResult};

    struct EchoRuntime {
        response_id: Option<String>,
    }

    impl EchoRuntime {
        fn new() -> Self {
            Self { response_id: None }
        }
    }

    impl ConnectorRuntime for EchoRuntime {
        fn send(&mut self, request: ConnectorRequest) -> Result<ConnectorResponse, RuntimeError> {
            Ok(ConnectorResponse::ok(
                self.response_id
                    .clone()
                    .unwrap_or_else(|| request.id.clone()),
                ConnectorResult::Health("ok".to_string()),
            ))
        }

        fn close(&mut self) -> Result<(), RuntimeError> {
            Ok(())
        }
    }

    #[test]
    fn client_assigns_monotonic_request_ids() {
        let mut client = ConnectorClient::new(EchoRuntime::new());

        let first = client.request(ConnectorRequestKind::Health).unwrap();
        let second = client.request(ConnectorRequestKind::Health).unwrap();

        assert_eq!(first.id, "req-1");
        assert_eq!(second.id, "req-2");
        assert!(matches!(
            second.kind,
            ConnectorResponseKind::Ok(ConnectorResult::Health(_))
        ));
    }

    #[test]
    fn client_rejects_uncorrelated_responses() {
        let runtime = EchoRuntime {
            response_id: Some("wrong-request".to_string()),
        };
        let mut client = ConnectorClient::new(runtime);

        assert!(matches!(
            client.request(ConnectorRequestKind::Health),
            Err(RuntimeError::Serialization(_))
        ));
    }
}
