use std::{
    net::IpAddr,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc, Mutex,
    },
    time::Duration,
};

use futures::{stream, StreamExt};
use kapsl_engine_api::{
    BinaryTensorPacket, CancellationToken, EngineError, EngineModelInfo, InferenceRequest,
};
use kapsl_grpc::{
    inference::{
        grpc_inference_service_client::GrpcInferenceServiceClient,
        infer_parameter::ParameterChoice, model_infer_request::InferInputTensor, InferParameter,
        InferTensorContents, ModelInferRequest, ModelMetadataRequest, ModelReadyRequest,
        ServerLiveRequest, ServerMetadataRequest, ServerReadyRequest,
    },
    kapsl::v1::{kapsl_inference_client::KapslInferenceClient, ListModelsRequest},
    start_server, EngineFacade, EngineStream, GrpcServerConfig, GrpcServerHandle, Model,
    RequestAuthorizer,
};
use tonic::{transport::Channel, Code, Request, Status};

#[derive(Clone, Copy)]
enum Mode {
    Finite,
    Hold,
    Fail,
    Starting,
}

struct Backend {
    mode: Mode,
    observed: Mutex<Vec<InferenceRequest>>,
}

#[async_trait::async_trait]
impl EngineFacade for Backend {
    fn models(&self) -> Vec<Model> {
        vec![Model {
            id: 7,
            name: "echo".into(),
            version: "1".into(),
            ready: true,
            info: Some(EngineModelInfo {
                input_names: vec!["input".into()],
                output_names: vec!["output".into()],
                input_shapes: vec![vec![-1]],
                output_shapes: vec![vec![-1]],
                input_dtypes: vec!["uint8".into()],
                output_dtypes: vec!["uint8".into()],
                framework: Some("test".into()),
                model_version: Some("1".into()),
                peak_concurrency: None,
            }),
        }]
    }

    async fn infer(
        &self,
        _: u32,
        request: InferenceRequest,
    ) -> Result<BinaryTensorPacket, EngineError> {
        self.observed.lock().unwrap().push(request.clone());
        if matches!(self.mode, Mode::Starting) {
            return std::future::pending().await;
        }
        Ok(request.input)
    }

    async fn infer_stream(
        &self,
        _: u32,
        request: InferenceRequest,
    ) -> Result<EngineStream, EngineError> {
        self.observed.lock().unwrap().push(request.clone());
        let packet = request.input;
        match self.mode {
            Mode::Finite => Ok(Box::pin(stream::iter(vec![Ok(packet.clone()), Ok(packet)]))),
            Mode::Hold => Ok(Box::pin(
                stream::once(async { Ok(packet) }).chain(stream::pending()),
            )),
            Mode::Fail => Ok(Box::pin(stream::iter(vec![
                Ok(packet.clone()),
                Err(EngineError::backend("private backend details")),
                Ok(packet),
            ]))),
            Mode::Starting => std::future::pending().await,
        }
    }
}

struct Authorizer(AtomicBool);

impl RequestAuthorizer for Authorizer {
    fn authorize_reader(
        &self,
        authorization: Option<&str>,
        remote: Option<IpAddr>,
    ) -> Result<(), Status> {
        assert!(remote.is_some_and(|ip| ip.is_loopback()));
        if authorization != Some("Bearer test-key") || !self.0.load(Ordering::SeqCst) {
            Err(Status::unauthenticated("Invalid API token"))
        } else {
            Ok(())
        }
    }

    fn scope_session_id(&self, session: Option<&str>, _: Option<&str>) -> Option<String> {
        session.map(|session| format!("scoped:{session}"))
    }
}

struct Fixture {
    server: GrpcServerHandle,
    backend: Arc<Backend>,
    auth: Arc<Authorizer>,
    channel: Channel,
}

impl Fixture {
    async fn new(mode: Mode, limit: usize) -> Self {
        let backend = Arc::new(Backend {
            mode,
            observed: Mutex::new(Vec::new()),
        });
        let auth = Arc::new(Authorizer(AtomicBool::new(true)));
        let server = start_server(
            GrpcServerConfig {
                bind_addr: "127.0.0.1:0".parse().unwrap(),
                max_message_bytes: limit,
                server_version: "test".into(),
            },
            backend.clone(),
            auth.clone(),
        )
        .await
        .unwrap();
        let channel = Channel::from_shared(format!("http://{}", server.bound_addr()))
            .unwrap()
            .connect()
            .await
            .unwrap();
        Self {
            server,
            backend,
            auth,
            channel,
        }
    }
    fn unary(&self) -> GrpcInferenceServiceClient<Channel> {
        GrpcInferenceServiceClient::new(self.channel.clone())
    }
    fn streaming(&self) -> KapslInferenceClient<Channel> {
        KapslInferenceClient::new(self.channel.clone())
    }
    fn cancellation(&self) -> CancellationToken {
        self.backend
            .observed
            .lock()
            .unwrap()
            .last()
            .unwrap()
            .cancellation
            .clone()
            .unwrap()
    }
}

fn authorized<T>(value: T) -> Request<T> {
    let mut request = Request::new(value);
    request
        .metadata_mut()
        .insert("authorization", "Bearer test-key".parse().unwrap());
    request
}

fn input() -> ModelInferRequest {
    ModelInferRequest {
        model_name: "echo".into(),
        id: "request-1".into(),
        inputs: vec![InferInputTensor {
            name: "input".into(),
            datatype: "UINT8".into(),
            shape: vec![4],
            contents: Some(InferTensorContents {
                uint_contents: vec![0, 127, 128, 255],
                ..Default::default()
            }),
            ..Default::default()
        }],
        ..Default::default()
    }
}

async fn cancelled(token: CancellationToken) {
    tokio::time::timeout(Duration::from_secs(3), token.cancelled())
        .await
        .expect("backend cancelled");
}

#[tokio::test]
async fn standard_health_metadata_and_discovery_are_authenticated() {
    let f = Fixture::new(Mode::Finite, 1024 * 1024).await;
    assert_eq!(
        f.unary()
            .server_live(ServerLiveRequest {})
            .await
            .unwrap_err()
            .code(),
        Code::Unauthenticated
    );
    assert_eq!(
        f.unary().model_infer(input()).await.unwrap_err().code(),
        Code::Unauthenticated
    );
    assert_eq!(
        f.streaming()
            .infer_stream(input())
            .await
            .unwrap_err()
            .code(),
        Code::Unauthenticated
    );
    assert_eq!(
        f.streaming()
            .list_models(ListModelsRequest {})
            .await
            .unwrap_err()
            .code(),
        Code::Unauthenticated
    );
    assert!(
        f.unary()
            .server_live(authorized(ServerLiveRequest {}))
            .await
            .unwrap()
            .into_inner()
            .live
    );
    assert!(
        f.unary()
            .server_ready(authorized(ServerReadyRequest {}))
            .await
            .unwrap()
            .into_inner()
            .ready
    );
    let metadata = f
        .unary()
        .server_metadata(authorized(ServerMetadataRequest {}))
        .await
        .unwrap()
        .into_inner();
    assert_eq!(metadata.extensions, ["kapsl_server_streaming"]);
    let model = f
        .unary()
        .model_metadata(authorized(ModelMetadataRequest {
            name: "7".into(),
            version: "1".into(),
        }))
        .await
        .unwrap()
        .into_inner();
    assert_eq!(model.inputs[0].datatype, "UINT8");
    assert_eq!(model.inputs[0].shape, [-1]);
    let models = f
        .streaming()
        .list_models(authorized(ListModelsRequest {}))
        .await
        .unwrap()
        .into_inner();
    assert_eq!(models.models[0].id, 7);
    f.auth.0.store(false, Ordering::SeqCst);
    // Reuse the established HTTP/2 connection: credentials are checked per RPC.
    assert_eq!(
        f.unary()
            .server_live(authorized(ServerLiveRequest {}))
            .await
            .unwrap_err()
            .code(),
        Code::Unauthenticated
    );
    assert!(f.backend.observed.lock().unwrap().is_empty());
}

#[tokio::test]
async fn binary_unary_and_streaming_round_trip_without_text_conversion() {
    let f = Fixture::new(Mode::Finite, 1024 * 1024).await;
    let output = f
        .unary()
        .model_infer(authorized(input()))
        .await
        .unwrap()
        .into_inner();
    assert_eq!(output.raw_output_contents[0], [0, 127, 128, 255]);
    assert_eq!(output.id, "request-1");
    assert_eq!(output.model_version, "1");
    let mut response = f
        .streaming()
        .infer_stream(authorized(input()))
        .await
        .unwrap()
        .into_inner();
    for _ in 0..2 {
        let output = response.message().await.unwrap().unwrap();
        assert_eq!(output.raw_output_contents[0], [0, 127, 128, 255]);
        assert_eq!(output.outputs[0].shape, [4]);
        assert_eq!(output.outputs[0].datatype, "UINT8");
    }
    assert!(response.message().await.unwrap().is_none());
    cancelled(f.cancellation()).await;
}

#[tokio::test]
async fn dropping_an_idle_stream_cancels_backend_without_another_output() {
    let f = Fixture::new(Mode::Hold, 1024 * 1024).await;
    let mut response = f
        .streaming()
        .infer_stream(authorized(input()))
        .await
        .unwrap()
        .into_inner();
    response.message().await.unwrap().unwrap();
    let token = f.cancellation();
    assert!(!token.is_cancelled());
    drop(response);
    cancelled(token).await;
}

#[tokio::test]
async fn grpc_deadline_covers_stream_start_and_the_full_body() {
    for mode in [Mode::Starting, Mode::Hold] {
        let f = Fixture::new(mode, 1024 * 1024).await;
        let mut request = authorized(input());
        request.set_timeout(Duration::from_millis(150));
        let result = f.streaming().infer_stream(request).await;
        let error = match result {
            Err(error) => error,
            Ok(response) => {
                let mut response = response.into_inner();
                response.message().await.unwrap().unwrap();
                response.message().await.unwrap_err()
            }
        };
        // Tonic's client maps its own pre-header timeout to CANCELLED. After
        // response headers, the server owns the full-body deadline.
        if matches!(mode, Mode::Starting) {
            assert!(matches!(
                error.code(),
                Code::Cancelled | Code::DeadlineExceeded
            ));
        } else {
            assert_eq!(error.code(), Code::DeadlineExceeded);
        }
        cancelled(f.cancellation()).await;
    }
}

#[tokio::test]
async fn unary_deadline_cancels_in_flight_inference() {
    let f = Fixture::new(Mode::Starting, 1024 * 1024).await;
    let mut request = authorized(input());
    request.set_timeout(Duration::from_millis(150));
    let error = f.unary().model_infer(request).await.unwrap_err();
    assert!(matches!(
        error.code(),
        Code::Cancelled | Code::DeadlineExceeded
    ));
    cancelled(f.cancellation()).await;
}

#[tokio::test]
async fn timeout_parameter_covers_stream_body_and_session_is_scoped() {
    let f = Fixture::new(Mode::Hold, 1024 * 1024).await;
    let mut request = input();
    request.parameters.insert(
        "timeout_ms".into(),
        InferParameter {
            parameter_choice: Some(ParameterChoice::Int64Param(150)),
        },
    );
    request.parameters.insert(
        "session_id".into(),
        InferParameter {
            parameter_choice: Some(ParameterChoice::StringParam("session".into())),
        },
    );
    let mut stream = f
        .streaming()
        .infer_stream(authorized(request))
        .await
        .unwrap()
        .into_inner();
    stream.message().await.unwrap().unwrap();
    assert_eq!(
        f.backend.observed.lock().unwrap()[0].session_id.as_deref(),
        Some("scoped:session")
    );
    assert_eq!(
        stream.message().await.unwrap_err().code(),
        Code::DeadlineExceeded
    );
    cancelled(f.cancellation()).await;
}

#[tokio::test]
async fn stream_failure_terminates_with_status_and_redacts_backend_details() {
    let f = Fixture::new(Mode::Fail, 1024 * 1024).await;
    let mut response = f
        .streaming()
        .infer_stream(authorized(input()))
        .await
        .unwrap()
        .into_inner();
    response.message().await.unwrap().unwrap();
    let error = response.message().await.unwrap_err();
    assert_eq!(error.code(), Code::Internal);
    assert!(!error.message().contains("private"));
    assert!(response.message().await.unwrap().is_none());
    cancelled(f.cancellation()).await;
}

#[tokio::test]
async fn malformed_tensors_and_unsupported_parameters_do_not_reach_backend() {
    let f = Fixture::new(Mode::Finite, 1024 * 1024).await;
    let mut cases = Vec::new();
    let mut request = input();
    request.inputs[0].shape = vec![-1];
    cases.push(request);
    let mut request = input();
    request.inputs[0].shape = vec![5];
    cases.push(request);
    let mut request = input();
    request.inputs[0].datatype = "BOOL".into();
    cases.push(request);
    let mut request = input();
    request.raw_input_contents = vec![vec![0; 4]];
    cases.push(request);
    let mut request = input();
    request.inputs[0].name = "wrong".into();
    cases.push(request);
    let mut request = input();
    request.parameters.insert(
        "auth_token".into(),
        InferParameter {
            parameter_choice: Some(ParameterChoice::StringParam("ignored".into())),
        },
    );
    cases.push(request);
    for request in cases {
        assert_eq!(
            f.unary()
                .model_infer(authorized(request))
                .await
                .unwrap_err()
                .code(),
            Code::InvalidArgument
        );
    }
    assert!(f.backend.observed.lock().unwrap().is_empty());
}

#[tokio::test]
async fn invalid_model_versions_and_unknown_models_return_not_found() {
    let f = Fixture::new(Mode::Finite, 1024 * 1024).await;
    for (name, version) in [("missing", ""), ("echo", "2"), ("7", "2")] {
        let error = f
            .unary()
            .model_ready(authorized(ModelReadyRequest {
                name: name.into(),
                version: version.into(),
            }))
            .await
            .unwrap_err();
        assert_eq!(error.code(), Code::NotFound);
    }
}

#[tokio::test]
async fn message_size_limit_rejects_request_before_dispatch() {
    let f = Fixture::new(Mode::Finite, 1024).await;
    let mut request = input();
    request.inputs[0].shape = vec![4096];
    request.inputs[0].contents = None;
    request.raw_input_contents = vec![vec![7; 4096]];
    let error = f
        .unary()
        .model_infer(authorized(request))
        .await
        .unwrap_err();
    assert_eq!(error.code(), Code::OutOfRange);
    assert!(f.backend.observed.lock().unwrap().is_empty());
}

#[tokio::test]
async fn shutdown_cancels_active_generation_and_drains_server() {
    let mut f = Fixture::new(Mode::Hold, 1024 * 1024).await;
    let mut response = f
        .streaming()
        .infer_stream(authorized(input()))
        .await
        .unwrap()
        .into_inner();
    response.message().await.unwrap().unwrap();
    let token = f.cancellation();
    f.server.shutdown().await;
    cancelled(token).await;
    assert_eq!(
        response.message().await.unwrap_err().code(),
        Code::Unavailable
    );
}

#[tokio::test]
async fn server_deadline_returns_deadline_exceeded_on_the_wire() {
    use http_body_util::{BodyExt, Full};
    use hyper_util::{client::legacy::Client, rt::TokioExecutor};
    use prost::Message;
    // Raw HTTP/2 avoids Tonic's independent client deadline, so this checks
    // the status emitted by the server itself, including during stream start.
    for path in [
        "inference.GRPCInferenceService/ModelInfer",
        "kapsl.v1.KapslInference/InferStream",
    ] {
        let f = Fixture::new(Mode::Starting, 1024 * 1024).await;
        let payload = input().encode_to_vec();
        let mut body = vec![0];
        body.extend((payload.len() as u32).to_be_bytes());
        body.extend(payload);
        let client = Client::builder(TokioExecutor::new())
            .http2_only(true)
            .build_http();
        let request = http::Request::post(format!("http://{}/{path}", f.server.bound_addr()))
            .header("content-type", "application/grpc")
            .header("te", "trailers")
            .header("authorization", "Bearer test-key")
            .header("grpc-timeout", "100m")
            .body(Full::new(prost::bytes::Bytes::from(body)))
            .unwrap();
        let response = client.request(request).await.unwrap();
        let header_status = response.headers().get("grpc-status").cloned();
        let body = response.into_body().collect().await.unwrap();
        let status = header_status.as_ref().or_else(|| {
            body.trailers()
                .and_then(|trailers| trailers.get("grpc-status"))
        });
        assert_eq!(status.unwrap(), "4");
        cancelled(f.cancellation()).await;
    }
}
