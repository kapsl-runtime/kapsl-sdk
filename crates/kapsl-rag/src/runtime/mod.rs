use std::collections::HashMap;
use std::io::{self, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};

use bytes::Bytes;
use kapsl_rag_sdk::protocol::{ConnectorRequest, ConnectorRequestKind, ConnectorResponse};
use kapsl_rag_sdk::read_frame;
use wasmtime::{Config, Engine, Linker, Module, Store, StoreLimits, StoreLimitsBuilder, Trap};
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
    #[error(
        "wasm connector exceeded its CPU budget of {fuel} fuel units and was stopped \
         (raise it with KAPSL_WASM_FUEL if the connector legitimately needs more)"
    )]
    WasmFuelExhausted { fuel: u64 },
    #[error(
        "wasm connector exceeded its memory limit of {limit_bytes} bytes and was stopped \
         (raise it with KAPSL_WASM_MAX_MEMORY_MB if the connector legitimately needs more)"
    )]
    WasmMemoryLimit { limit_bytes: usize },
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

/// Default CPU budget for one WASM connector request, in Wasmtime fuel units
/// (roughly one unit per executed WASM instruction, so a few seconds of work).
pub const DEFAULT_WASM_FUEL: u64 = 10_000_000_000;
/// Default linear-memory ceiling for one WASM connector instance.
pub const DEFAULT_WASM_MAX_MEMORY_BYTES: usize = 256 * 1024 * 1024;
/// Environment variable overriding [`DEFAULT_WASM_FUEL`].
pub const WASM_FUEL_ENV: &str = "KAPSL_WASM_FUEL";
/// Environment variable overriding [`DEFAULT_WASM_MAX_MEMORY_BYTES`], in MiB.
pub const WASM_MAX_MEMORY_MB_ENV: &str = "KAPSL_WASM_MAX_MEMORY_MB";

/// Resource limits applied to every WASM connector request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WasmLimits {
    /// Fuel available to one request. Execution traps when it runs out.
    pub fuel: u64,
    /// Maximum bytes of linear memory (initial size and growth).
    pub max_memory_bytes: usize,
}

impl Default for WasmLimits {
    fn default() -> Self {
        Self {
            fuel: DEFAULT_WASM_FUEL,
            max_memory_bytes: DEFAULT_WASM_MAX_MEMORY_BYTES,
        }
    }
}

impl WasmLimits {
    /// Defaults, overridden by `KAPSL_WASM_FUEL` and `KAPSL_WASM_MAX_MEMORY_MB`
    /// when those hold a positive integer. Invalid values are ignored.
    pub fn from_env() -> Self {
        Self::from_lookup(|key| std::env::var(key).ok())
    }

    fn from_lookup(lookup: impl Fn(&str) -> Option<String>) -> Self {
        let mut limits = Self::default();
        if let Some(fuel) = lookup(WASM_FUEL_ENV)
            .and_then(|value| value.trim().parse::<u64>().ok())
            .filter(|value| *value > 0)
        {
            limits.fuel = fuel;
        }
        if let Some(bytes) = lookup(WASM_MAX_MEMORY_MB_ENV)
            .and_then(|value| value.trim().parse::<usize>().ok())
            .filter(|value| *value > 0)
            .and_then(|mb| mb.checked_mul(1024 * 1024))
        {
            limits.max_memory_bytes = bytes;
        }
        limits
    }

    pub fn with_fuel(mut self, fuel: u64) -> Self {
        self.fuel = fuel;
        self
    }

    pub fn with_max_memory_bytes(mut self, max_memory_bytes: usize) -> Self {
        self.max_memory_bytes = max_memory_bytes;
        self
    }
}

/// Per-request store state: the WASI context plus the resource limiter.
struct ConnectorStoreState {
    wasi: WasiP1Ctx,
    limits: StoreLimits,
}

pub struct WasmConnectorRuntime {
    engine: Engine,
    module: Module,
    permissions: WasiPermissions,
    limits: WasmLimits,
}

impl WasmConnectorRuntime {
    pub fn spawn(path: &Path) -> Result<Self, RuntimeError> {
        Self::spawn_with_permissions(path, WasiPermissions::default())
    }

    pub fn spawn_with_permissions(
        path: &Path,
        permissions: WasiPermissions,
    ) -> Result<Self, RuntimeError> {
        Self::spawn_with_limits(path, permissions, WasmLimits::from_env())
    }

    pub fn spawn_with_limits(
        path: &Path,
        permissions: WasiPermissions,
        limits: WasmLimits,
    ) -> Result<Self, RuntimeError> {
        let mut config = Config::new();
        config.consume_fuel(true);
        let engine = Engine::new(&config).map_err(|e| RuntimeError::Wasm(e.to_string()))?;
        let module =
            Module::from_file(&engine, path).map_err(|e| RuntimeError::Wasm(e.to_string()))?;
        Ok(Self {
            engine,
            module,
            permissions,
            limits,
        })
    }

    /// Maps a Wasmtime error to a limit-specific error when a limit caused it.
    fn classify_error(&self, error: wasmtime::Error) -> RuntimeError {
        if matches!(error.downcast_ref::<Trap>(), Some(Trap::OutOfFuel)) {
            return RuntimeError::WasmFuelExhausted {
                fuel: self.limits.fuel,
            };
        }
        // Wasmtime reports a refused memory allocation (initial size too large, or
        // `memory.grow` with `trap_on_grow_failure`) with these messages.
        let rendered = format!("{error:#}");
        if rendered.contains("exceeds memory limits")
            || rendered.contains("forcing trap when growing memory")
            || rendered.contains("forcing a memory growth failure to be a trap")
        {
            return RuntimeError::WasmMemoryLimit {
                limit_bytes: self.limits.max_memory_bytes,
            };
        }
        RuntimeError::Wasm(rendered)
    }

    fn run_once(&self, input: &str) -> Result<String, RuntimeError> {
        // Run the WASM connector to completion for this request. This keeps the
        // implementation simple and sandboxed, at the cost of per-request startup.
        let mut linker = Linker::<ConnectorStoreState>::new(&self.engine);
        add_to_linker_sync(&mut linker, |state: &mut ConnectorStoreState| {
            &mut state.wasi
        })
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
        let limits = StoreLimitsBuilder::new()
            .memory_size(self.limits.max_memory_bytes)
            .trap_on_grow_failure(true)
            .build();
        let mut store = Store::new(&self.engine, ConnectorStoreState { wasi, limits });
        store.limiter(|state| &mut state.limits);
        store
            .set_fuel(self.limits.fuel)
            .map_err(|e| RuntimeError::Wasm(e.to_string()))?;
        let instance = linker
            .instantiate(&mut store, &self.module)
            .map_err(|e| self.classify_error(e))?;

        let start = instance
            .get_typed_func::<(), ()>(&mut store, "_start")
            .map_err(|e| RuntimeError::Wasm(e.to_string()))?;
        start
            .call(&mut store, ())
            .map_err(|e| self.classify_error(e))?;

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
    // Hand-assembled WASI command modules (no `wat` dependency).
    const WASM_HEADER: [u8; 8] = [0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00];
    const TYPE_FN_VOID: [u8; 6] = [0x01, 0x04, 0x01, 0x60, 0x00, 0x00];
    const FUNC_SECTION: [u8; 4] = [0x03, 0x02, 0x01, 0x00];
    const EXPORT_START: [u8; 12] = [
        0x07, 0x0a, 0x01, 0x06, b'_', b's', b't', b'a', b'r', b't', 0x00, 0x00,
    ];
    const MEMORY_ONE_PAGE: [u8; 5] = [0x05, 0x03, 0x01, 0x00, 0x01];

    fn module_bytes(with_memory: bool, body: &[u8]) -> Vec<u8> {
        let mut out = WASM_HEADER.to_vec();
        out.extend_from_slice(&TYPE_FN_VOID);
        out.extend_from_slice(&FUNC_SECTION);
        if with_memory {
            out.extend_from_slice(&MEMORY_ONE_PAGE);
        }
        out.extend_from_slice(&EXPORT_START);
        // code section: one function body (locals = 0, instructions, end)
        let mut func = vec![0x00];
        func.extend_from_slice(body);
        func.push(0x0b);
        out.push(0x0a);
        out.push((func.len() + 2) as u8);
        out.push(0x01);
        out.push(func.len() as u8);
        out.extend_from_slice(&func);
        out
    }

    fn write_module(dir: &tempfile::TempDir, name: &str, bytes: &[u8]) -> PathBuf {
        let path = dir.path().join(name);
        std::fs::write(&path, bytes).unwrap();
        path
    }

    fn runtime_for(bytes: &[u8], limits: WasmLimits) -> WasmConnectorRuntime {
        let dir = tempfile::tempdir().unwrap();
        let path = write_module(&dir, "connector.wasm", bytes);
        WasmConnectorRuntime::spawn_with_limits(&path, WasiPermissions::default(), limits).unwrap()
    }

    #[test]
    fn well_behaved_module_runs_within_default_limits() {
        // `_start` returns immediately.
        let runtime = runtime_for(&module_bytes(false, &[]), WasmLimits::default());
        assert_eq!(runtime.run_once("{}\n").unwrap(), "");
    }

    #[test]
    fn runaway_loop_is_stopped_with_a_fuel_error() {
        // loop; br 0; end  (never terminates)
        let runtime = runtime_for(
            &module_bytes(false, &[0x03, 0x40, 0x0c, 0x00, 0x0b]),
            WasmLimits::default().with_fuel(1_000_000),
        );
        let error = runtime.run_once("{}\n").unwrap_err();
        assert!(
            matches!(error, RuntimeError::WasmFuelExhausted { fuel: 1_000_000 }),
            "unexpected error: {error:?}"
        );
        assert!(error.to_string().contains("KAPSL_WASM_FUEL"));
    }

    #[test]
    fn runaway_memory_growth_is_stopped_with_a_memory_error() {
        // memory.grow(1024 pages = 64 MiB) against a 1 MiB limit.
        let runtime = runtime_for(
            &module_bytes(true, &[0x41, 0x80, 0x08, 0x40, 0x00, 0x1a]),
            WasmLimits::default().with_max_memory_bytes(1024 * 1024),
        );
        let error = runtime.run_once("{}\n").unwrap_err();
        assert!(
            matches!(
                error,
                RuntimeError::WasmMemoryLimit {
                    limit_bytes: 1_048_576
                }
            ),
            "unexpected error: {error:?}"
        );
    }

    #[test]
    fn module_whose_initial_memory_exceeds_the_limit_is_refused() {
        // Declares 1 page (64 KiB) of memory against a 16 KiB limit.
        let runtime = runtime_for(
            &module_bytes(true, &[]),
            WasmLimits::default().with_max_memory_bytes(16 * 1024),
        );
        let error = runtime.run_once("{}\n").unwrap_err();
        assert!(
            matches!(error, RuntimeError::WasmMemoryLimit { .. }),
            "unexpected error: {error:?}"
        );
    }

    #[test]
    fn memory_growth_within_the_limit_succeeds() {
        let runtime = runtime_for(
            &module_bytes(true, &[0x41, 0x80, 0x08, 0x40, 0x00, 0x1a]),
            WasmLimits::default().with_max_memory_bytes(128 * 1024 * 1024),
        );
        assert!(runtime.run_once("{}\n").is_ok());
    }

    #[test]
    fn limits_default_and_env_override() {
        let defaults = WasmLimits::default();
        assert_eq!(defaults.fuel, DEFAULT_WASM_FUEL);
        assert_eq!(defaults.max_memory_bytes, DEFAULT_WASM_MAX_MEMORY_BYTES);

        let lookup = |key: &str| match key {
            WASM_FUEL_ENV => Some("12345".to_string()),
            WASM_MAX_MEMORY_MB_ENV => Some("8".to_string()),
            _ => None,
        };
        let limits = WasmLimits::from_lookup(lookup);
        assert_eq!(limits.fuel, 12345);
        assert_eq!(limits.max_memory_bytes, 8 * 1024 * 1024);

        // Zero and garbage fall back to the defaults.
        let bad = |key: &str| match key {
            WASM_FUEL_ENV => Some("0".to_string()),
            WASM_MAX_MEMORY_MB_ENV => Some("lots".to_string()),
            _ => None,
        };
        assert_eq!(WasmLimits::from_lookup(bad), defaults);
    }
}
