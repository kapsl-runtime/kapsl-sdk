//! Request-level tests for the preprocessing wrapper.

use super::{PreprocessBackend, Preprocessor};
use async_trait::async_trait;
use kapsl_engine_api::{
    BinaryTensorPacket, Engine, EngineError, EngineMetrics, EngineStream, InferenceRequest,
    NamedTensor, TensorDtype,
};
use std::path::Path;
use std::sync::{Arc, Mutex};

fn u8_packet(value: u8) -> BinaryTensorPacket {
    BinaryTensorPacket::new(vec![1], TensorDtype::Uint8, vec![value]).unwrap()
}

fn i64_packet(value: i64) -> BinaryTensorPacket {
    BinaryTensorPacket::new(vec![1], TensorDtype::Int64, value.to_le_bytes().to_vec()).unwrap()
}

struct DerivingPreprocessor;

impl Preprocessor for DerivingPreprocessor {
    fn apply(&self, _input: &BinaryTensorPacket) -> Result<BinaryTensorPacket, EngineError> {
        BinaryTensorPacket::new(vec![1, 2, 7], TensorDtype::Float32, vec![0; 2 * 7 * 4])
    }

    fn derived_inputs(
        &self,
        _output: &BinaryTensorPacket,
    ) -> Result<Vec<NamedTensor>, EngineError> {
        Ok(vec![NamedTensor {
            name: "length".to_string(),
            tensor: i64_packet(7),
        }])
    }

    fn name(&self) -> &'static str {
        "test"
    }
}

struct CapturingEngine {
    seen: Arc<Mutex<Option<InferenceRequest>>>,
}

#[async_trait]
impl Engine for CapturingEngine {
    async fn load(&mut self, _model_path: &Path) -> Result<(), EngineError> {
        Ok(())
    }

    fn infer(&self, request: &InferenceRequest) -> Result<BinaryTensorPacket, EngineError> {
        *self.seen.lock().unwrap() = Some(request.clone());
        Ok(request.input.clone())
    }

    fn infer_stream(&self, request: &InferenceRequest) -> EngineStream {
        let result = self.infer(request);
        Box::pin(futures::stream::once(async move { result }))
    }

    fn unload(&mut self) {}

    fn metrics(&self) -> EngineMetrics {
        EngineMetrics::new()
    }

    fn health_check(&self) -> Result<(), EngineError> {
        Ok(())
    }
}

#[test]
fn wrapper_injects_derived_input_and_replaces_stale_client_value() {
    let seen = Arc::new(Mutex::new(None));
    let backend = PreprocessBackend::new(
        Box::new(CapturingEngine {
            seen: Arc::clone(&seen),
        }),
        Box::new(DerivingPreprocessor),
    );
    let mut request = InferenceRequest::new(u8_packet(42));
    request.add_input("other", u8_packet(9));
    request.add_input("length", i64_packet(999));

    backend.infer(&request).unwrap();
    let transformed = seen.lock().unwrap().clone().unwrap();

    assert_eq!(transformed.input.shape, vec![1, 2, 7]);
    assert_eq!(transformed.additional_inputs.len(), 2);
    assert_eq!(
        transformed
            .additional_inputs
            .iter()
            .find(|input| input.name == "other")
            .unwrap()
            .tensor
            .data,
        vec![9]
    );
    let length = &transformed
        .additional_inputs
        .iter()
        .find(|input| input.name == "length")
        .unwrap()
        .tensor;
    assert_eq!(
        i64::from_le_bytes(length.data.as_slice().try_into().unwrap()),
        7
    );
}
