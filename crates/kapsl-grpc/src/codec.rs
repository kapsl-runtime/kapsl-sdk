//! Lossless conversion for the tensor types supported by kapsl-engine-api.

use std::collections::{HashMap, HashSet};

use kapsl_engine_api::{
    BinaryTensorPacket, InferenceRequest, NamedTensor, RequestMetadata, TensorDtype,
};
use tonic::Status;

use crate::inference::{
    infer_parameter::ParameterChoice, model_infer_response::InferOutputTensor,
    model_metadata_response::TensorMetadata, InferParameter, InferTensorContents,
    ModelInferRequest, ModelInferResponse, ModelMetadataResponse,
};
use crate::Model;

pub(crate) fn datatype(dtype: TensorDtype) -> &'static str {
    match dtype {
        TensorDtype::Float32 => "FP32",
        TensorDtype::Float64 => "FP64",
        TensorDtype::Float16 => "FP16",
        TensorDtype::Int32 => "INT32",
        TensorDtype::Int64 => "INT64",
        TensorDtype::Uint8 => "UINT8",
        TensorDtype::Utf8 => "BYTES",
    }
}

fn parse_dtype(dtype: &str) -> Result<TensorDtype, Status> {
    match dtype {
        "FP32" => Ok(TensorDtype::Float32),
        "FP64" => Ok(TensorDtype::Float64),
        "FP16" => Ok(TensorDtype::Float16),
        "INT32" => Ok(TensorDtype::Int32),
        "INT64" => Ok(TensorDtype::Int64),
        "UINT8" => Ok(TensorDtype::Uint8),
        "BYTES" => Ok(TensorDtype::Utf8),
        _ => Err(Status::invalid_argument("Unsupported tensor datatype")),
    }
}

pub(crate) fn metadata(model: &Model) -> Result<ModelMetadataResponse, Status> {
    let info = model.info.as_ref().ok_or_else(|| {
        Status::failed_precondition("Backend has not provided model tensor metadata")
    })?;
    let tensors = |names: &[String], shapes: &[Vec<i64>], dtypes: &[String]| {
        names
            .iter()
            .enumerate()
            .map(|(index, name)| {
                let dtype: TensorDtype = dtypes
                    .get(index)
                    .ok_or_else(|| {
                        Status::failed_precondition("Backend tensor datatype is unavailable")
                    })?
                    .parse()
                    .map_err(|_| {
                        Status::failed_precondition("Backend tensor datatype is unsupported")
                    })?;
                Ok(TensorMetadata {
                    name: name.clone(),
                    datatype: datatype(dtype).into(),
                    // Engine strings contain one UTF-8 value, not a byte tensor.
                    shape: if dtype == TensorDtype::Utf8 {
                        vec![1]
                    } else {
                        shapes.get(index).cloned().ok_or_else(|| {
                            Status::failed_precondition("Backend tensor shape is unavailable")
                        })?
                    },
                })
            })
            .collect::<Result<Vec<_>, Status>>()
    };
    Ok(ModelMetadataResponse {
        name: model.name.clone(),
        versions: vec![model.version.clone()],
        platform: info.framework.clone().unwrap_or_else(|| "kapsl".into()),
        inputs: tensors(&info.input_names, &info.input_shapes, &info.input_dtypes)?,
        outputs: tensors(&info.output_names, &info.output_shapes, &info.output_dtypes)?,
        properties: HashMap::new(),
    })
}

/// The SDK inference boundary currently returns one packet per call/chunk.
pub(crate) fn output_name(model: &Model, request: &ModelInferRequest) -> Result<String, Status> {
    let names = model
        .info
        .as_ref()
        .map(|info| info.output_names.as_slice())
        .unwrap_or_default();
    if names.len() > 1 {
        return Err(Status::unimplemented(
            "gRPC inference currently supports single-output models",
        ));
    }
    let name = names.first().map(String::as_str).unwrap_or("output");
    if request.outputs.len() > 1
        || request
            .outputs
            .first()
            .is_some_and(|output| output.name != name)
    {
        return Err(Status::invalid_argument(
            "Unknown or unsupported requested output",
        ));
    }
    if request
        .outputs
        .iter()
        .any(|output| !output.parameters.is_empty())
    {
        return Err(Status::invalid_argument(
            "Output tensor parameters are unsupported",
        ));
    }
    Ok(name.into())
}

pub(crate) fn decode(
    model: &Model,
    mut wire: ModelInferRequest,
) -> Result<InferenceRequest, Status> {
    if wire.inputs.is_empty() {
        return Err(Status::invalid_argument(
            "At least one input tensor is required",
        ));
    }
    let raw = !wire.raw_input_contents.is_empty();
    if raw && wire.raw_input_contents.len() != wire.inputs.len() {
        return Err(Status::invalid_argument(
            "raw_input_contents must match inputs",
        ));
    }
    let mut names = HashSet::new();
    let mut inputs = Vec::with_capacity(wire.inputs.len());
    let mut raw_contents = wire.raw_input_contents.into_iter();
    for input in wire.inputs {
        if input.name.is_empty() || !names.insert(input.name.clone()) {
            return Err(Status::invalid_argument(
                "Input tensor names must be nonempty and unique",
            ));
        }
        if !input.parameters.is_empty() {
            return Err(Status::invalid_argument(
                "Input tensor parameters are unsupported",
            ));
        }
        let dtype = parse_dtype(&input.datatype)?;
        let count = element_count(&input.shape)?;
        let data = if raw {
            if input.contents.is_some() {
                return Err(Status::invalid_argument(
                    "Cannot mix raw and typed tensor contents",
                ));
            }
            let data = raw_contents.next().expect("raw input count was validated");
            if dtype == TensorDtype::Utf8 {
                if count != 1 || data.len() < 4 {
                    return Err(Status::invalid_argument(
                        "BYTES requires exactly one length-prefixed UTF-8 value",
                    ));
                }
                let length = u32::from_le_bytes(data[..4].try_into().unwrap()) as usize;
                if length != data.len() - 4 {
                    return Err(Status::invalid_argument("Invalid BYTES length prefix"));
                }
                data[4..].to_vec()
            } else {
                data
            }
        } else {
            typed_contents(dtype, count, input.contents.unwrap_or_default())?
        };
        validate_data(dtype, count, &data)?;
        // Engine UTF-8 packets count bytes in their shape. Translate the
        // protocol's scalar string into that internal representation.
        let shape = if dtype == TensorDtype::Utf8 {
            vec![
                1,
                i64::try_from(data.len())
                    .map_err(|_| Status::invalid_argument("String is too large"))?,
            ]
        } else {
            input.shape
        };
        inputs.push(NamedTensor {
            name: input.name,
            tensor: BinaryTensorPacket::new(shape, dtype, data)
                .map_err(|error| Status::invalid_argument(error.to_string()))?,
        });
    }

    // The SDK represents the primary input positionally. Resolve it by the
    // backend's declared name so reordered KServe inputs retain their meaning.
    if let Some(info) = &model.info {
        if !info.input_names.is_empty() {
            if info.input_names.len() != inputs.len()
                || info.input_names.iter().any(|name| !names.contains(name))
            {
                return Err(Status::invalid_argument(
                    "Input names do not match model metadata",
                ));
            }
            let first = inputs
                .iter()
                .position(|input| input.name == info.input_names[0])
                .unwrap();
            inputs.swap(0, first);
        } else if inputs.len() > 1 {
            return Err(Status::failed_precondition(
                "Multiple inputs require backend tensor metadata",
            ));
        }
    } else if inputs.len() > 1 {
        return Err(Status::failed_precondition(
            "Multiple inputs require backend tensor metadata",
        ));
    }
    let primary = inputs.remove(0);
    let session_id = match wire.parameters.remove("session_id") {
        Some(InferParameter {
            parameter_choice: Some(ParameterChoice::StringParam(value)),
        }) => Some(value),
        Some(_) => return Err(Status::invalid_argument("session_id must be a string")),
        None => None,
    };
    let metadata = request_metadata(wire.id, wire.model_version, wire.parameters)?;
    Ok(InferenceRequest {
        input: primary.tensor,
        additional_inputs: inputs,
        session_id,
        metadata: Some(metadata),
        cancellation: None,
    })
}

fn element_count(shape: &[i64]) -> Result<usize, Status> {
    if shape.len() > 32 {
        return Err(Status::invalid_argument("Tensor rank exceeds 32"));
    }
    shape.iter().try_fold(1usize, |count, dimension| {
        let dimension = usize::try_from(*dimension)
            .map_err(|_| Status::invalid_argument("Tensor dimensions must be nonnegative"))?;
        count
            .checked_mul(dimension)
            .ok_or_else(|| Status::invalid_argument("Tensor shape overflows"))
    })
}

fn validate_data(dtype: TensorDtype, count: usize, data: &[u8]) -> Result<(), Status> {
    if dtype == TensorDtype::Utf8 {
        if count != 1 || std::str::from_utf8(data).is_err() {
            return Err(Status::invalid_argument(
                "BYTES requires exactly one UTF-8 value",
            ));
        }
    } else if count.checked_mul(dtype.size_bytes()) != Some(data.len()) {
        return Err(Status::invalid_argument(
            "Tensor contents do not match shape and datatype",
        ));
    }
    Ok(())
}

fn typed_contents(
    dtype: TensorDtype,
    count: usize,
    mut contents: InferTensorContents,
) -> Result<Vec<u8>, Status> {
    let data: Vec<u8> = match dtype {
        TensorDtype::Float32 => std::mem::take(&mut contents.fp32_contents)
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect(),
        TensorDtype::Float64 => std::mem::take(&mut contents.fp64_contents)
            .into_iter()
            .flat_map(f64::to_le_bytes)
            .collect(),
        TensorDtype::Int32 => std::mem::take(&mut contents.int_contents)
            .into_iter()
            .flat_map(i32::to_le_bytes)
            .collect(),
        TensorDtype::Int64 => std::mem::take(&mut contents.int64_contents)
            .into_iter()
            .flat_map(i64::to_le_bytes)
            .collect(),
        TensorDtype::Uint8 => std::mem::take(&mut contents.uint_contents)
            .into_iter()
            .map(|value| {
                u8::try_from(value)
                    .map_err(|_| Status::invalid_argument("UINT8 value is out of range"))
            })
            .collect::<Result<_, _>>()?,
        TensorDtype::Float16 => return Err(Status::invalid_argument("FP16 requires raw contents")),
        TensorDtype::Utf8 => {
            if count != 1 || contents.bytes_contents.len() != 1 {
                return Err(Status::invalid_argument(
                    "BYTES requires exactly one UTF-8 value",
                ));
            }
            contents.bytes_contents.pop().unwrap()
        }
    };
    if contents != InferTensorContents::default() {
        return Err(Status::invalid_argument(
            "Contents field does not match tensor datatype",
        ));
    }
    Ok(data)
}

fn request_metadata(
    id: String,
    version: String,
    parameters: HashMap<String, InferParameter>,
) -> Result<RequestMetadata, Status> {
    let mut result = RequestMetadata {
        request_id: (!id.is_empty()).then_some(id),
        model_version: (!version.is_empty()).then_some(version),
        ..Default::default()
    };
    for (name, value) in parameters {
        let parameter = value
            .parameter_choice
            .ok_or_else(|| Status::invalid_argument("Empty inference parameter"))?;
        let unsigned = || {
            match parameter {
                ParameterChoice::Int64Param(value) => u64::try_from(value).ok(),
                ParameterChoice::Uint64Param(value) => Some(value),
                _ => None,
            }
            .ok_or_else(|| Status::invalid_argument("Expected a nonnegative integer parameter"))
        };
        let u32_value = || {
            u32::try_from(unsigned()?)
                .map_err(|_| Status::invalid_argument("Integer parameter is out of range"))
        };
        let float = || match parameter {
            ParameterChoice::DoubleParam(value)
                if value.is_finite() && (value as f32).is_finite() =>
            {
                Ok(value as f32)
            }
            _ => Err(Status::invalid_argument(
                "Expected a finite double parameter",
            )),
        };
        match name.as_str() {
            "timeout_ms" => {
                let value = unsigned()?;
                if value == 0 {
                    return Err(Status::invalid_argument("timeout_ms must be positive"));
                }
                result.timeout_ms = Some(value);
            }
            "priority" => {
                result.priority = Some(
                    u8::try_from(unsigned()?)
                        .map_err(|_| Status::invalid_argument("priority is out of range"))?,
                )
            }
            "force_cpu" => {
                result.force_cpu = Some(match parameter {
                    ParameterChoice::BoolParam(value) => value,
                    _ => return Err(Status::invalid_argument("force_cpu must be boolean")),
                })
            }
            "max_new_tokens" => result.max_new_tokens = Some(u32_value()?),
            "min_new_tokens" => result.min_new_tokens = Some(u32_value()?),
            "top_k" => result.top_k = Some(u32_value()?),
            "seed" => result.seed = Some(unsigned()?),
            "temperature" => {
                let value = float()?;
                if value < 0.0 {
                    return Err(Status::invalid_argument("temperature must be nonnegative"));
                }
                result.temperature = Some(value);
            }
            "top_p" => {
                let value = float()?;
                if !(0.0..=1.0).contains(&value) {
                    return Err(Status::invalid_argument("top_p must be between 0 and 1"));
                }
                result.top_p = Some(value);
            }
            "repetition_penalty" => {
                let value = float()?;
                if value <= 0.0 {
                    return Err(Status::invalid_argument(
                        "repetition_penalty must be positive",
                    ));
                }
                result.repetition_penalty = Some(value);
            }
            _ => return Err(Status::invalid_argument("Unsupported inference parameter")),
        }
    }
    Ok(result)
}

pub(crate) fn encode(
    model: &Model,
    id: String,
    name: String,
    packet: BinaryTensorPacket,
) -> Result<ModelInferResponse, Status> {
    let (shape, data) = if packet.dtype == TensorDtype::Utf8 {
        if std::str::from_utf8(&packet.data).is_err() {
            return Err(Status::internal("Backend returned invalid UTF-8"));
        }
        let length = u32::try_from(packet.data.len())
            .map_err(|_| Status::resource_exhausted("Output string is too large"))?;
        let mut data = length.to_le_bytes().to_vec();
        data.extend(packet.data);
        (vec![1], data)
    } else {
        let count = element_count(&packet.shape)
            .map_err(|_| Status::internal("Backend returned an invalid tensor shape"))?;
        validate_data(packet.dtype, count, &packet.data)
            .map_err(|_| Status::internal("Backend returned invalid tensor contents"))?;
        (packet.shape, packet.data)
    };
    Ok(ModelInferResponse {
        model_name: model.name.clone(),
        model_version: model.version.clone(),
        id,
        outputs: vec![InferOutputTensor {
            name,
            datatype: datatype(packet.dtype).into(),
            shape,
            parameters: HashMap::new(),
            contents: None,
        }],
        raw_output_contents: vec![data],
        parameters: HashMap::new(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::model_infer_request::InferInputTensor;

    fn model() -> Model {
        Model {
            id: 1,
            name: "test".into(),
            version: "1".into(),
            ready: true,
            info: None,
        }
    }

    #[test]
    fn raw_numeric_types_round_trip_bit_exactly() {
        for (dtype, width) in [
            ("FP16", 2),
            ("FP32", 4),
            ("FP64", 8),
            ("INT32", 4),
            ("INT64", 8),
            ("UINT8", 1),
        ] {
            let data: Vec<u8> = (0..width * 2).map(|index| (index * 19) as u8).collect();
            let request = ModelInferRequest {
                inputs: vec![InferInputTensor {
                    name: "input".into(),
                    datatype: dtype.into(),
                    shape: vec![2],
                    ..Default::default()
                }],
                raw_input_contents: vec![data.clone()],
                ..Default::default()
            };
            let decoded = decode(&model(), request).unwrap();
            let response = encode(&model(), "id".into(), "output".into(), decoded.input).unwrap();
            assert_eq!(response.raw_output_contents, [data]);
            assert_eq!(response.outputs[0].datatype, dtype);
        }
    }

    #[test]
    fn typed_floats_and_int64_use_little_endian() {
        let data = typed_contents(
            TensorDtype::Int64,
            2,
            InferTensorContents {
                int64_contents: vec![i64::MIN, i64::MAX],
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(&data[..8], &i64::MIN.to_le_bytes());
        assert_eq!(&data[8..], &i64::MAX.to_le_bytes());
        let data = typed_contents(
            TensorDtype::Float64,
            1,
            InferTensorContents {
                fp64_contents: vec![1.25],
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(data, 1.25f64.to_le_bytes());
    }

    #[test]
    fn strings_preserve_utf8_and_use_standard_bytes_framing() {
        let text = "Hello 🧠";
        let wire = ModelInferRequest {
            inputs: vec![InferInputTensor {
                name: "input".into(),
                datatype: "BYTES".into(),
                shape: vec![1],
                contents: Some(InferTensorContents {
                    bytes_contents: vec![text.as_bytes().to_vec()],
                    ..Default::default()
                }),
                ..Default::default()
            }],
            ..Default::default()
        };
        let packet = decode(&model(), wire).unwrap().input;
        packet.validate().unwrap();
        assert_eq!(packet.shape, [1, text.len() as i64]);
        assert_eq!(packet.data, text.as_bytes());
        let result = encode(&model(), String::new(), "output".into(), packet).unwrap();
        let raw = &result.raw_output_contents[0];
        assert_eq!(&raw[..4], &(text.len() as u32).to_le_bytes());
        assert_eq!(&raw[4..], text.as_bytes());
        assert_eq!(result.outputs[0].shape, [1]);
        let mut request = ModelInferRequest {
            inputs: vec![InferInputTensor {
                name: "input".into(),
                datatype: "BYTES".into(),
                shape: vec![1],
                ..Default::default()
            }],
            raw_input_contents: vec![raw.clone()],
            ..Default::default()
        };
        assert_eq!(
            decode(&model(), request.clone()).unwrap().input.data,
            text.as_bytes()
        );
        request.raw_input_contents[0][0] = 0;
        assert!(decode(&model(), request).is_err());
    }

    #[test]
    fn invalid_encodings_are_rejected() {
        assert!(element_count(&[-1]).is_err());
        assert!(element_count(&[i64::MAX, i64::MAX]).is_err());
        assert!(typed_contents(
            TensorDtype::Uint8,
            1,
            InferTensorContents {
                uint_contents: vec![256],
                ..Default::default()
            }
        )
        .is_err());
        assert!(typed_contents(
            TensorDtype::Float32,
            1,
            InferTensorContents {
                fp32_contents: vec![1.0],
                int_contents: vec![1],
                ..Default::default()
            }
        )
        .is_err());
        assert!(validate_data(TensorDtype::Utf8, 1, &[255]).is_err());
        assert!(validate_data(TensorDtype::Utf8, 2, b"ab").is_err());
    }

    #[test]
    fn primary_input_is_selected_by_declared_name() {
        let mut model = model();
        model.info = Some(kapsl_engine_api::EngineModelInfo {
            input_names: vec!["primary".into(), "secondary".into()],
            output_names: vec!["output".into()],
            input_shapes: vec![],
            output_shapes: vec![],
            input_dtypes: vec![],
            output_dtypes: vec![],
            framework: None,
            model_version: None,
            peak_concurrency: None,
        });
        let request = ModelInferRequest {
            inputs: ["secondary", "primary"]
                .into_iter()
                .map(|name| InferInputTensor {
                    name: name.into(),
                    datatype: "UINT8".into(),
                    shape: vec![1],
                    ..Default::default()
                })
                .collect(),
            raw_input_contents: vec![vec![2], vec![1]],
            ..Default::default()
        };
        let result = decode(&model, request).unwrap();
        assert_eq!(result.input.data, [1]);
        assert_eq!(result.additional_inputs[0].name, "secondary");
        assert_eq!(result.additional_inputs[0].tensor.data, [2]);
    }
}
