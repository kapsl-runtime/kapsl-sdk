//! Local protocol demonstration; no models, scheduler, or accelerator required.
use std::{net::IpAddr, sync::Arc};

use kapsl_engine_api::{BinaryTensorPacket, EngineError, EngineModelInfo, InferenceRequest};
use kapsl_grpc::{
    tonic::Status, EngineFacade, EngineStream, GrpcServerConfig, Model, RequestAuthorizer,
};

struct Echo;

#[async_trait::async_trait]
impl EngineFacade for Echo {
    fn models(&self) -> Vec<Model> {
        [("text", "string"), ("tensor", "uint8")]
            .into_iter()
            .enumerate()
            .map(|(id, (name, dtype))| Model {
                id: id as u32,
                name: name.into(),
                version: "1".into(),
                ready: true,
                info: Some(EngineModelInfo {
                    input_names: vec!["input".into()],
                    output_names: vec!["output".into()],
                    input_shapes: vec![vec![-1]],
                    output_shapes: vec![vec![-1]],
                    input_dtypes: vec![dtype.into()],
                    output_dtypes: vec![dtype.into()],
                    framework: Some("echo".into()),
                    model_version: Some("1".into()),
                    peak_concurrency: None,
                }),
            })
            .collect()
    }
    async fn infer(
        &self,
        _: u32,
        request: InferenceRequest,
    ) -> Result<BinaryTensorPacket, EngineError> {
        Ok(request.input)
    }
    async fn infer_stream(
        &self,
        _: u32,
        request: InferenceRequest,
    ) -> Result<EngineStream, EngineError> {
        Ok(Box::pin(futures::stream::iter([
            Ok(request.input.clone()),
            Ok(request.input),
        ])))
    }
}

struct LocalOnly;
impl RequestAuthorizer for LocalOnly {
    fn authorize_reader(&self, _: Option<&str>, ip: Option<IpAddr>) -> Result<(), Status> {
        if ip.is_some_and(|ip| ip.is_loopback()) {
            Ok(())
        } else {
            Err(Status::permission_denied("Local example only"))
        }
    }
    fn scope_session_id(&self, _: Option<&str>, _: Option<&str>) -> Option<String> {
        None
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut server = kapsl_grpc::start_server(
        GrpcServerConfig {
            bind_addr: "127.0.0.1:0".parse()?,
            max_message_bytes: 16 * 1024 * 1024,
            server_version: env!("CARGO_PKG_VERSION").into(),
        },
        Arc::new(Echo),
        Arc::new(LocalOnly),
    )
    .await?;
    println!("{}", server.bound_addr());
    server.wait().await?;
    Ok(())
}
