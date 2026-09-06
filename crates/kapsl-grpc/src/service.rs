use std::{io, net::SocketAddr, pin::Pin, sync::Arc, time::Duration};

use futures::{Stream, StreamExt};
use kapsl_engine_api::InferenceRequest;
use prost::Message;
use tokio::{net::TcpListener, task::JoinHandle};
use tokio_stream::wrappers::TcpListenerStream;
use tokio_util::sync::CancellationToken;
use tonic::{Request, Response, Status};

use crate::{
    codec, engine_status,
    inference::*,
    kapsl::v1::{
        kapsl_inference_server::{KapslInference, KapslInferenceServer},
        ListModelsRequest, ListModelsResponse,
    },
    lifecycle::{parse_timeout, Call},
    Dependencies, EngineFacade, Model, RequestAuthorizer,
};
use grpc_inference_service_server::{GrpcInferenceService, GrpcInferenceServiceServer};

pub struct GrpcServerConfig {
    pub bind_addr: SocketAddr,
    pub max_message_bytes: usize,
    pub server_version: String,
}

pub struct GrpcServerHandle {
    bound_addr: SocketAddr,
    shutdown: CancellationToken,
    task: JoinHandle<Result<(), tonic::transport::Error>>,
}

impl GrpcServerHandle {
    pub fn bound_addr(&self) -> SocketAddr {
        self.bound_addr
    }

    pub async fn wait(&mut self) -> io::Result<()> {
        (&mut self.task)
            .await
            .map_err(io::Error::other)?
            .map_err(io::Error::other)
    }

    pub async fn shutdown(&mut self) {
        self.shutdown.cancel();
        if !self.task.is_finished()
            && tokio::time::timeout(Duration::from_secs(5), &mut self.task)
                .await
                .is_err()
        {
            self.task.abort();
            let _ = (&mut self.task).await;
        }
    }
}

impl Drop for GrpcServerHandle {
    fn drop(&mut self) {
        self.shutdown.cancel();
        self.task.abort();
    }
}

pub async fn start_server(
    config: GrpcServerConfig,
    engine: Arc<dyn EngineFacade>,
    authorizer: Arc<dyn RequestAuthorizer>,
) -> io::Result<GrpcServerHandle> {
    if config.max_message_bytes == 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "gRPC message limit must be positive",
        ));
    }
    let listener = TcpListener::bind(config.bind_addr).await?;
    let bound_addr = listener.local_addr()?;
    let shutdown = CancellationToken::new();
    let service = Service {
        dependencies: Dependencies { engine, authorizer },
        shutdown: shutdown.clone(),
        server_version: config.server_version,
        max_message_bytes: config.max_message_bytes,
    };
    let inference = GrpcInferenceServiceServer::new(service.clone())
        .max_decoding_message_size(config.max_message_bytes)
        .max_encoding_message_size(config.max_message_bytes);
    let streaming = KapslInferenceServer::new(service)
        .max_decoding_message_size(config.max_message_bytes)
        .max_encoding_message_size(config.max_message_bytes);
    let signal = shutdown.clone();
    let task = tokio::spawn(async move {
        tonic::transport::Server::builder()
            .layer(tower::ServiceBuilder::new().map_request(record_received_at))
            .max_concurrent_streams(64)
            .add_service(inference)
            .add_service(streaming)
            .serve_with_incoming_shutdown(
                TcpListenerStream::new(listener),
                signal.cancelled_owned(),
            )
            .await
    });
    log::info!("gRPC server listening on {bound_addr}");
    Ok(GrpcServerHandle {
        bound_addr,
        shutdown,
        task,
    })
}

#[derive(Clone)]
struct Service {
    dependencies: Dependencies,
    shutdown: CancellationToken,
    server_version: String,
    max_message_bytes: usize,
}

struct PreparedInference {
    call: Call,
    model: Model,
    request: InferenceRequest,
    request_id: String,
    output_name: String,
}

#[derive(Clone, Copy)]
struct ReceivedAt(tokio::time::Instant);

fn record_received_at(
    mut request: http::Request<tonic::body::Body>,
) -> http::Request<tonic::body::Body> {
    // Capture time before protobuf decoding. This layer is called before
    // tonic starts its handler timeout, so our full-RPC deadline takes
    // precedence and consistently returns DEADLINE_EXCEEDED.
    request
        .extensions_mut()
        .insert(ReceivedAt(tokio::time::Instant::now()));
    request
}

impl Service {
    fn begin<T>(&self, request: &Request<T>, method: &'static str) -> Result<Call, Status> {
        let started = request
            .extensions()
            .get::<ReceivedAt>()
            .map_or_else(tokio::time::Instant::now, |time| time.0);
        let mut call = Call::new(method, self.shutdown.clone(), started);
        let result = (|| {
            let authorization = single_metadata(request, "authorization")?;
            self.dependencies
                .authorizer
                .authorize_reader(authorization, request.remote_addr().map(|addr| addr.ip()))?;
            if self.shutdown.is_cancelled() {
                return Err(Status::unavailable("Server is shutting down"));
            }
            if let Some(timeout) = single_metadata(request, "grpc-timeout")? {
                call.set_timeout(parse_timeout(timeout)?)?;
            }
            Ok(())
        })();
        if let Err(error) = result {
            return call.result(Err(error));
        }
        Ok(call)
    }

    fn control<T, R: Message>(
        &self,
        request: Request<T>,
        method: &'static str,
        f: impl FnOnce(T) -> Result<R, Status>,
    ) -> Result<Response<R>, Status> {
        let mut call = self.begin(&request, method)?;
        call.result(
            f(request.into_inner())
                .and_then(|message| checked_message(message, self.max_message_bytes))
                .map(Response::new),
        )
    }

    fn model(&self, name: &str, version: &str) -> Result<Model, Status> {
        let models = self.dependencies.engine.models();
        // IDs provide an unambiguous escape hatch when display names overlap.
        let by_id = name
            .parse::<u32>()
            .ok()
            .and_then(|id| models.iter().find(|model| model.id == id));
        let model = if let Some(model) = by_id {
            model
        } else {
            let mut matches = models.iter().filter(|model| {
                model.name == name && (version.is_empty() || model.version == version)
            });
            let model = matches
                .next()
                .ok_or_else(|| Status::not_found("Model not found"))?;
            if matches.next().is_some() {
                return Err(Status::failed_precondition(
                    "Model name is ambiguous; use its numeric ID",
                ));
            }
            model
        };
        if !version.is_empty() && version != model.version {
            return Err(Status::not_found("Model version not found"));
        }
        Ok(model.clone())
    }

    fn prepare(
        &self,
        request: Request<ModelInferRequest>,
        method: &'static str,
    ) -> Result<PreparedInference, Status> {
        let mut call = self.begin(&request, method)?;
        let result = (|| {
            let authorization = single_metadata(&request, "authorization")?.map(str::to_owned);
            let wire = request.into_inner();
            let model = self.model(&wire.model_name, &wire.model_version)?;
            if !model.ready {
                return Err(Status::unavailable("Model is not ready"));
            }
            let output_name = codec::output_name(&model, &wire)?;
            let request_id = wire.id.clone();
            let mut request = codec::decode(&model, wire)?;
            request.session_id = self
                .dependencies
                .authorizer
                .scope_session_id(request.session_id.as_deref(), authorization.as_deref());
            if let Some(timeout) = request
                .metadata
                .as_ref()
                .and_then(|metadata| metadata.timeout_ms)
            {
                call.set_timeout(Duration::from_millis(timeout))?;
            }
            request.cancellation = Some(call.cancellation.clone());
            Ok((model, request, request_id, output_name))
        })();
        match result {
            Ok((model, request, request_id, output_name)) => {
                call.activate();
                Ok(PreparedInference {
                    call,
                    model,
                    request,
                    request_id,
                    output_name,
                })
            }
            Err(error) => call.result(Err(error)),
        }
    }
}

fn single_metadata<'a, T>(
    request: &'a Request<T>,
    key: &'static str,
) -> Result<Option<&'a str>, Status> {
    let mut values = request.metadata().get_all(key).iter();
    let first = values.next();
    if values.next().is_some() {
        return Err(Status::invalid_argument("Duplicate RPC metadata"));
    }
    first
        .map(|value| {
            value
                .to_str()
                .map_err(|_| Status::invalid_argument("Invalid RPC metadata"))
        })
        .transpose()
}

fn checked_message<T: Message>(message: T, limit: usize) -> Result<T, Status> {
    if message.encoded_len() > limit {
        Err(Status::resource_exhausted(
            "Response exceeds the gRPC message limit",
        ))
    } else {
        Ok(message)
    }
}

#[tonic::async_trait]
impl GrpcInferenceService for Service {
    async fn server_live(
        &self,
        request: Request<ServerLiveRequest>,
    ) -> Result<Response<ServerLiveResponse>, Status> {
        self.control(request, "ServerLive", |_| {
            Ok(ServerLiveResponse { live: true })
        })
    }
    async fn server_ready(
        &self,
        request: Request<ServerReadyRequest>,
    ) -> Result<Response<ServerReadyResponse>, Status> {
        // An empty runtime is ready to accept model-management traffic.
        self.control(request, "ServerReady", |_| {
            Ok(ServerReadyResponse { ready: true })
        })
    }
    async fn server_metadata(
        &self,
        request: Request<ServerMetadataRequest>,
    ) -> Result<Response<ServerMetadataResponse>, Status> {
        self.control(request, "ServerMetadata", |_| {
            Ok(ServerMetadataResponse {
                name: "kapsl-engine".into(),
                version: self.server_version.clone(),
                extensions: vec!["kapsl_server_streaming".into()],
            })
        })
    }
    async fn model_ready(
        &self,
        request: Request<ModelReadyRequest>,
    ) -> Result<Response<ModelReadyResponse>, Status> {
        self.control(request, "ModelReady", |request| {
            Ok(ModelReadyResponse {
                ready: self.model(&request.name, &request.version)?.ready,
            })
        })
    }
    async fn model_metadata(
        &self,
        request: Request<ModelMetadataRequest>,
    ) -> Result<Response<ModelMetadataResponse>, Status> {
        self.control(request, "ModelMetadata", |request| {
            codec::metadata(&self.model(&request.name, &request.version)?)
        })
    }
    async fn model_infer(
        &self,
        request: Request<ModelInferRequest>,
    ) -> Result<Response<ModelInferResponse>, Status> {
        let PreparedInference {
            mut call,
            model,
            request,
            request_id,
            output_name,
        } = self.prepare(request, "ModelInfer")?;
        let result = call
            .wait(async {
                let packet = self
                    .dependencies
                    .engine
                    .infer(model.id, request)
                    .await
                    .map_err(engine_status)?;
                codec::encode(&model, request_id, output_name, packet)
                    .and_then(|message| checked_message(message, self.max_message_bytes))
                    .map(Response::new)
            })
            .await;
        call.result(result)
    }
}

#[tonic::async_trait]
impl KapslInference for Service {
    type InferStreamStream = Pin<Box<dyn Stream<Item = Result<ModelInferResponse, Status>> + Send>>;

    async fn list_models(
        &self,
        request: Request<ListModelsRequest>,
    ) -> Result<Response<ListModelsResponse>, Status> {
        self.control(request, "ListModels", |_| {
            let mut models = self.dependencies.engine.models();
            models.sort_by_key(|model| model.id);
            Ok(ListModelsResponse {
                models: models
                    .into_iter()
                    .map(|model| crate::kapsl::v1::Model {
                        id: model.id,
                        name: model.name,
                        version: model.version,
                        ready: model.ready,
                    })
                    .collect(),
            })
        })
    }

    async fn infer_stream(
        &self,
        request: Request<ModelInferRequest>,
    ) -> Result<Response<Self::InferStreamStream>, Status> {
        let PreparedInference {
            mut call,
            model,
            request,
            request_id,
            output_name,
        } = self.prepare(request, "InferStream")?;
        let result = call
            .wait(async {
                self.dependencies
                    .engine
                    .infer_stream(model.id, request)
                    .await
                    .map_err(engine_status)
            })
            .await;
        let stream = match result {
            Ok(stream) => stream,
            Err(error) => return call.result(Err(error)),
        };
        let stream = call.attach_stream(stream);
        // Direct polling preserves transport backpressure. No producer task
        // drains generation into an unbounded queue.
        let max_message_bytes = self.max_message_bytes;
        let output = futures::stream::unfold(
            Some((call, stream, model, request_id, output_name)),
            move |state| async move {
                let (mut call, mut stream, model, id, name) = state?;
                let next = call.wait(async { Ok(stream.next().await) }).await;
                let item = match next {
                    Ok(Some(Ok(packet))) => codec::encode(&model, id.clone(), name.clone(), packet),
                    Ok(Some(Err(error))) => Err(engine_status(error)),
                    Err(error) => Err(error),
                    Ok(None) => {
                        let _ = call.result(Ok(()));
                        return None;
                    }
                }
                .and_then(|message| checked_message(message, max_message_bytes));
                if item.is_err() {
                    let result = call.result(item);
                    // Drop the engine stream and guard immediately after the
                    // first failure; do not continue or report a success tail.
                    Some((result, None))
                } else {
                    Some((item, Some((call, stream, model, id, name))))
                }
            },
        );
        Ok(Response::new(Box::pin(output)))
    }
}
