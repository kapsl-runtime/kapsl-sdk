//! Test fixture for installed Python wheels. All listeners are local and
//! inference uses an in-process echo scheduler, without model backends.
use kapsl_communication::{
    CommunicationServer, SchedulerLookup, SchedulerSnapshot, TransportServer,
};
use kapsl_engine_api::{
    BinaryTensorPacket, EngineError, EngineMetrics, EngineModelInfo, EngineStream,
    InferenceRequest, TensorDtype,
};
use kapsl_grpc::{tonic::Status, EngineFacade, GrpcServerConfig, Model, RequestAuthorizer};
use kapsl_scheduler::{Priority, ReplicaScheduler};
use std::{
    collections::HashMap,
    fs::{File, OpenOptions},
    io::Write,
    net::IpAddr,
    path::PathBuf,
    sync::{Arc, Mutex},
    time::Duration,
};

const TOKEN: &str = "sdk-test-token";
type Events = Arc<Mutex<File>>;

struct Lease {
    events: Events,
    id: String,
}

impl Lease {
    fn new(events: Events, request: &InferenceRequest) -> Self {
        let id = request
            .metadata
            .as_ref()
            .and_then(|m| m.request_id.clone())
            .unwrap_or_else(|| "anonymous".into());
        writeln!(events.lock().unwrap(), "started:{id}").unwrap();
        Self { events, id }
    }
}

impl Drop for Lease {
    fn drop(&mut self) {
        writeln!(self.events.lock().unwrap(), "released:{}", self.id).unwrap();
    }
}

struct Echo {
    id: u32,
    events: Events,
}

impl Echo {
    fn output(
        &self,
        request: &InferenceRequest,
        priority: Priority,
        force_cpu: bool,
    ) -> BinaryTensorPacket {
        if self.id != 11 {
            return request.input.clone();
        }
        let data = serde_json::to_vec(&serde_json::json!({
            "metadata": request.metadata,
            "session_id": request.session_id,
            "additional_inputs": request.additional_inputs.iter().map(|x| &x.name).collect::<Vec<_>>(),
            "priority": format!("{priority:?}"),
            "force_cpu": force_cpu,
        })).unwrap();
        BinaryTensorPacket {
            shape: vec![1, data.len() as i64],
            dtype: TensorDtype::Utf8,
            data,
        }
    }
}

#[async_trait::async_trait]
impl ReplicaScheduler for Echo {
    fn get_queue_depth(&self) -> (usize, usize) {
        (0, 0)
    }
    fn is_healthy(&self) -> bool {
        true
    }
    fn get_metrics(&self) -> EngineMetrics {
        EngineMetrics::default()
    }
    async fn infer(
        &self,
        request: &InferenceRequest,
        priority: Priority,
        force_cpu: bool,
    ) -> Result<BinaryTensorPacket, EngineError> {
        let _lease = Lease::new(self.events.clone(), request);
        if self.id == 9 {
            tokio::time::sleep(Duration::from_secs(30)).await;
        }
        Ok(self.output(request, priority, force_cpu))
    }

    async fn infer_stream(
        &self,
        request: InferenceRequest,
        priority: Priority,
        force_cpu: bool,
    ) -> Result<EngineStream, EngineError> {
        let lease = Lease::new(self.events.clone(), &request);
        if self.id == 9 {
            tokio::time::sleep(Duration::from_secs(30)).await;
        }
        let id = self.id;
        let packet = self.output(&request, priority, force_cpu);
        Ok(Box::pin(futures::stream::unfold(
            (0, packet, lease),
            move |(index, packet, lease)| async move {
                if index == 2 {
                    return None;
                }
                if index == 1 && id == 12 {
                    std::future::pending::<()>().await;
                }
                let result = if index == 1 && id == 10 {
                    Err(EngineError::backend("fixture stream failure"))
                } else {
                    Ok(packet.clone())
                };
                Some((result, (index + 1, packet, lease)))
            },
        )))
    }
}

struct Facade(HashMap<u32, Arc<dyn ReplicaScheduler + Send + Sync>>);

#[async_trait::async_trait]
impl EngineFacade for Facade {
    fn models(&self) -> Vec<Model> {
        [
            (7, "echo"),
            (8, "text"),
            (9, "slow"),
            (10, "error"),
            (11, "options"),
            (12, "idle"),
        ]
        .into_iter()
        .map(|(id, name)| {
            let input_dtype = if id == 8 { "string" } else { "float32" };
            let output_dtype = if id == 8 || id == 11 {
                "string"
            } else {
                "float32"
            };
            let inputs = if id == 11 {
                vec!["input".into(), "mask".into()]
            } else {
                vec!["input".into()]
            };
            Model {
                id,
                name: name.into(),
                version: "1".into(),
                ready: true,
                info: Some(EngineModelInfo {
                    input_dtypes: vec![input_dtype.into(); inputs.len()],
                    input_shapes: vec![vec![-1]; inputs.len()],
                    input_names: inputs,
                    output_names: vec!["output".into()],
                    output_shapes: vec![vec![-1]],
                    output_dtypes: vec![output_dtype.into()],
                    framework: Some("test".into()),
                    model_version: Some("1".into()),
                    peak_concurrency: None,
                }),
            }
        })
        .collect()
    }

    async fn infer(
        &self,
        id: u32,
        request: InferenceRequest,
    ) -> Result<BinaryTensorPacket, EngineError> {
        let priority = match request.metadata.as_ref().and_then(|m| m.priority) {
            Some(0) => Priority::LatencyCritical,
            _ => Priority::Throughput,
        };
        let force_cpu = request
            .metadata
            .as_ref()
            .and_then(|m| m.force_cpu)
            .unwrap_or(false);
        self.0[&id].infer(&request, priority, force_cpu).await
    }

    async fn infer_stream(
        &self,
        id: u32,
        request: InferenceRequest,
    ) -> Result<kapsl_grpc::EngineStream, EngineError> {
        let priority = match request.metadata.as_ref().and_then(|m| m.priority) {
            Some(0) => Priority::LatencyCritical,
            _ => Priority::Throughput,
        };
        let force_cpu = request
            .metadata
            .as_ref()
            .and_then(|m| m.force_cpu)
            .unwrap_or(false);
        self.0[&id].infer_stream(request, priority, force_cpu).await
    }
}

struct Authorizer;
impl RequestAuthorizer for Authorizer {
    fn authorize_reader(&self, token: Option<&str>, peer: Option<IpAddr>) -> Result<(), Status> {
        if peer.is_some_and(|p| p.is_loopback()) && token == Some("Bearer sdk-test-token") {
            Ok(())
        } else {
            Err(Status::unauthenticated("Invalid token"))
        }
    }
    fn scope_session_id(&self, session: Option<&str>, _: Option<&str>) -> Option<String> {
        session.map(|s| format!("test:{s}"))
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let root = PathBuf::from(
        std::env::args()
            .nth(1)
            .expect("Pass a temporary state directory"),
    );
    std::fs::create_dir_all(&root)?;
    let events_path = root.join("events");
    let events = Arc::new(Mutex::new(
        OpenOptions::new()
            .create(true)
            .append(true)
            .open(&events_path)?,
    ));
    let schedulers: HashMap<u32, Arc<dyn ReplicaScheduler + Send + Sync>> = (7..=12)
        .map(|id| {
            (
                id,
                Arc::new(Echo {
                    id,
                    events: events.clone(),
                }) as Arc<dyn ReplicaScheduler + Send + Sync>,
            )
        })
        .collect();
    let shared = Arc::new(schedulers.clone());
    let lookup: SchedulerLookup = Arc::new(move |id| shared.get(&id).cloned());
    let snapshot_map = schedulers.clone();
    let snapshot: SchedulerSnapshot = Arc::new(move || snapshot_map.clone());
    let port = {
        let listener = std::net::TcpListener::bind("127.0.0.1:0")?;
        listener.local_addr()?.port()
    };
    let pid = std::process::id();
    #[cfg(unix)]
    let socket = format!("/tmp/kapsl-py-{pid}.sock");
    #[cfg(windows)]
    let socket = format!(r"\\.\pipe\kapsl-py-{pid}");
    let shm = format!("/kpy-s-{pid}");
    let hybrid_shm = format!("/kpy-h-{pid}");
    let servers = [
        CommunicationServer::tcp("127.0.0.1", port, lookup.clone(), Some(TOKEN)),
        CommunicationServer::hybrid(&socket, &hybrid_shm, 8 * 1024 * 1024, lookup.clone())?,
        CommunicationServer::shared_memory(&shm, 8 * 1024 * 1024, lookup, snapshot, None),
    ];
    let mut tasks = Vec::new();
    for server in servers {
        tasks.push(tokio::spawn(async move { server.run().await }));
    }
    let mut grpc = kapsl_grpc::start_server(
        GrpcServerConfig {
            bind_addr: "127.0.0.1:0".parse()?,
            max_message_bytes: 16 * 1024 * 1024,
            server_version: "python-test".into(),
        },
        Arc::new(Facade(schedulers)),
        Arc::new(Authorizer),
    )
    .await?;
    for _ in 0..100 {
        if tokio::net::TcpStream::connect(("127.0.0.1", port))
            .await
            .is_ok()
            && kapsl_communication::shm::ShmManager::connect(&shm).is_ok()
        {
            break;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    println!(
        "{}",
        serde_json::json!({
            "tcp": format!("127.0.0.1:{port}"), "socket": socket, "shm": shm,
            "hybrid_shm": hybrid_shm, "grpc": grpc.bound_addr().to_string(),
            "events": events_path,
        })
    );
    std::io::stdout().flush()?;
    tokio::signal::ctrl_c().await?;
    grpc.shutdown().await;
    for task in tasks {
        task.abort();
        let _ = task.await;
    }
    #[cfg(unix)]
    let _ = std::fs::remove_file(socket);
    Ok(())
}
