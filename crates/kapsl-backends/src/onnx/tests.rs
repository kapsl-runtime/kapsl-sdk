//! Tests for the shared ONNX backend.

#[cfg(test)]
mod tests {
    use super::super::*;
    use half::f16;
    use kapsl_engine_api::{BinaryTensorPacket, InferenceRequest, NamedTensor, TensorDtype};

    fn f32_bytes(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|&v| v.to_ne_bytes().to_vec())
            .collect()
    }

    fn i32_bytes(values: &[i32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|&v| v.to_ne_bytes().to_vec())
            .collect()
    }

    fn f64_bytes(values: &[f64]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|&v| v.to_ne_bytes().to_vec())
            .collect()
    }

    fn f16_bytes(values: &[f16]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|v| v.to_bits().to_ne_bytes().to_vec())
            .collect()
    }

    fn i64_bytes(values: &[i64]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|&v| v.to_ne_bytes().to_vec())
            .collect()
    }

    #[test]
    fn test_copy_primitive_slice_as_ne_bytes_matches_float_encoding() {
        let values = [0.0f32, 1.0, -2.5, 3.25];
        assert_eq!(
            copy_primitive_slice_as_ne_bytes(&values),
            f32_bytes(&values)
        );

        let values = [f16::from_f32(0.0), f16::from_f32(-2.5)];
        assert_eq!(
            copy_primitive_slice_as_ne_bytes(&values),
            f16_bytes(&values)
        );
    }

    #[test]
    fn test_copy_primitive_slice_as_ne_bytes_matches_integer_encoding() {
        let i32_values = [0i32, -1, 123_456];
        assert_eq!(
            copy_primitive_slice_as_ne_bytes(&i32_values),
            i32_bytes(&i32_values)
        );

        let i64_values = [0i64, -1, 123_456_789];
        assert_eq!(
            copy_primitive_slice_as_ne_bytes(&i64_values),
            i64_bytes(&i64_values)
        );
    }

    #[test]
    fn test_validate_float32_success() {
        let values = vec![0.0f32, 1.0f32, -2.5f32, 3.25f32];
        let packet = BinaryTensorPacket {
            shape: vec![2, 2],
            dtype: TensorDtype::Float32,
            data: f32_bytes(&values),
        };

        match validate_and_prepare_input(&packet) {
            Ok((shape, PreparedInput::F32(vec))) => {
                assert_eq!(shape, vec![2, 2]);
                assert_eq!(vec.len(), 4);
                for (a, b) in vec.iter().zip(values.iter()) {
                    assert_eq!(a, b);
                }
            }
            other => panic!("Expected prepared f32 input, got: {:?}", other),
        }
    }

    #[test]
    fn test_parse_float32_aligned_input_borrows() {
        let values = [0.0f32, 1.0f32, -2.5f32, 3.25f32];
        let bytes = unsafe {
            std::slice::from_raw_parts(
                values.as_ptr().cast::<u8>(),
                values.len() * std::mem::size_of::<f32>(),
            )
        };

        match parse_ne_f32(bytes, values.len()) {
            std::borrow::Cow::Borrowed(parsed) => assert_eq!(parsed, values.as_slice()),
            other => panic!("Expected prepared f32 input, got: {:?}", other),
        }
    }

    #[test]
    fn test_parse_float32_unaligned_input_falls_back_to_owned() {
        let values = vec![1.0f32, 2.0f32];
        let mut data = vec![0u8];
        data.extend(f32_bytes(&values));

        match parse_ne_f32(&data[1..], values.len()) {
            std::borrow::Cow::Owned(parsed) => assert_eq!(parsed, values),
            std::borrow::Cow::Borrowed(_) => panic!("unaligned input should not be borrowed"),
        }
    }

    #[test]
    fn test_validate_float32_scalar_empty_shape() {
        let values = vec![42.0f32];
        let packet = BinaryTensorPacket {
            shape: vec![],
            dtype: TensorDtype::Float32,
            data: f32_bytes(&values),
        };

        match validate_and_prepare_input(&packet) {
            Ok((shape, PreparedInput::F32(vec))) => {
                assert!(shape.is_empty());
                assert_eq!(vec, values);
            }
            other => panic!("Expected prepared f32 scalar input, got: {:?}", other),
        }
    }

    #[test]
    fn test_validate_float32_bad_length() {
        // Provide only 3 floats for a 2x2 shape (needs 4)
        let values = vec![0.0f32, 1.0f32, -2.5f32];
        let packet = BinaryTensorPacket {
            shape: vec![2, 2],
            dtype: TensorDtype::Float32,
            data: f32_bytes(&values),
        };

        let res = validate_and_prepare_input(&packet);
        assert!(res.is_err());
        if let Err(EngineError::InvalidInput { message, .. }) = res {
            assert!(message.contains("Data length mismatch"));
        } else {
            panic!("Expected InvalidInput error for bad length");
        }
    }

    #[test]
    fn test_top_k_last_logits_packet_uses_last_row() {
        let scores = vec![
            100.0f32, 90.0, 80.0, 70.0, // earlier row should be ignored
            0.1, 4.0, -2.0, 3.5,
        ];

        let packet =
            top_k_last_logits_packet(&[1, 2, 4], scores.into_iter(), 2).expect("top-k packet");

        assert_eq!(packet.shape, vec![2, 2]);
        assert_eq!(packet.dtype, TensorDtype::Float32);
        let values: Vec<f32> = packet
            .data
            .chunks_exact(4)
            .map(|chunk| f32::from_ne_bytes(chunk.try_into().expect("f32 bytes")))
            .collect();
        assert_eq!(values, vec![1.0, 4.0, 3.0, 3.5]);
    }

    #[test]
    fn test_top_k_last_logits_packet_clamps_to_vocab() {
        let scores = vec![0.5f32, 2.0, 1.0];

        let packet =
            top_k_last_logits_packet(&[1, 3], scores.into_iter(), 99).expect("top-k packet");

        assert_eq!(packet.shape, vec![3, 2]);
        let values: Vec<f32> = packet
            .data
            .chunks_exact(4)
            .map(|chunk| f32::from_ne_bytes(chunk.try_into().expect("f32 bytes")))
            .collect();
        assert_eq!(values, vec![1.0, 2.0, 2.0, 1.0, 0.0, 0.5]);
    }

    #[test]
    fn test_validate_int32_bad_length() {
        let values = vec![1i32];
        let packet = BinaryTensorPacket {
            shape: vec![2],
            dtype: TensorDtype::Int32,
            data: i32_bytes(&values),
        };

        let res = validate_and_prepare_input(&packet);
        assert!(res.is_err());
        if let Err(EngineError::InvalidInput { message, .. }) = res {
            assert!(message.contains("Data length mismatch"));
        } else {
            panic!("Expected InvalidInput error for bad int32 length");
        }
    }

    #[test]
    fn test_validate_unsupported_dtype() {
        let packet = BinaryTensorPacket {
            shape: vec![2],
            dtype: TensorDtype::Utf8,
            data: vec![b'a', b'b'],
        };

        let res = validate_and_prepare_input(&packet);
        assert!(res.is_err());
        if let Err(EngineError::InvalidInput { message, .. }) = res {
            assert!(message.contains("Unsupported dtype"));
        } else {
            panic!("Expected InvalidInput for unsupported dtype");
        }
    }

    #[test]
    fn test_validate_invalid_shape_dimension_zero() {
        let packet = BinaryTensorPacket {
            shape: vec![0],
            dtype: TensorDtype::Float32,
            data: vec![],
        };

        let res = validate_and_prepare_input(&packet);
        assert!(res.is_err());
        if let Err(EngineError::InvalidInput { message, .. }) = res {
            assert!(message.contains("Invalid shape dimension"));
        } else {
            panic!("Expected InvalidInput error for invalid shape");
        }
    }

    #[test]
    fn test_validate_invalid_shape_dimension_negative() {
        let packet = BinaryTensorPacket {
            shape: vec![-1],
            dtype: TensorDtype::Float32,
            data: vec![],
        };

        let res = validate_and_prepare_input(&packet);
        assert!(res.is_err());
        if let Err(EngineError::InvalidInput { message, .. }) = res {
            assert!(message.contains("Invalid shape dimension"));
        } else {
            panic!("Expected InvalidInput error for invalid shape");
        }
    }

    #[test]
    fn test_validate_shape_multiplication_overflow() {
        let packet = BinaryTensorPacket {
            shape: vec![i64::MAX, i64::MAX],
            dtype: TensorDtype::Float32,
            data: vec![],
        };

        let res = validate_and_prepare_input(&packet);
        assert!(res.is_err());
        if let Err(EngineError::InvalidInput { message, .. }) = res {
            assert!(message.contains("Shape multiplication overflow"));
        } else {
            panic!("Expected InvalidInput error for overflow");
        }
    }

    #[test]
    fn test_validate_int32_success() {
        let values = vec![1i32, 2i32, 3i32];
        let packet = BinaryTensorPacket {
            shape: vec![3],
            dtype: TensorDtype::Int32,
            data: i32_bytes(&values),
        };

        match validate_and_prepare_input(&packet) {
            Ok((_shape, PreparedInput::I32(vec))) => {
                assert_eq!(vec, values);
            }
            other => panic!("Expected prepared i32 input, got: {:?}", other),
        }
    }

    #[test]
    fn test_validate_float16_success() {
        let values = vec![f16::from_f32(1.5), f16::from_f32(-2.25)];
        let packet = BinaryTensorPacket {
            shape: vec![2],
            dtype: TensorDtype::Float16,
            data: f16_bytes(&values),
        };

        match validate_and_prepare_input(&packet) {
            Ok((_shape, PreparedInput::F16(vec))) => {
                assert_eq!(vec, values);
            }
            other => panic!("Expected prepared f16 input, got: {:?}", other),
        }
    }

    #[test]
    fn test_validate_float64_success() {
        let values = vec![1.0f64, -3.5f64];
        let packet = BinaryTensorPacket {
            shape: vec![2],
            dtype: TensorDtype::Float64,
            data: f64_bytes(&values),
        };

        match validate_and_prepare_input(&packet) {
            Ok((_shape, PreparedInput::F64(vec))) => {
                assert_eq!(vec, values);
            }
            other => panic!("Expected prepared f64 input, got: {:?}", other),
        }
    }

    #[test]
    fn test_validate_int64_success() {
        let values = vec![10i64, -20i64];
        let packet = BinaryTensorPacket {
            shape: vec![2],
            dtype: TensorDtype::Int64,
            data: i64_bytes(&values),
        };

        match validate_and_prepare_input(&packet) {
            Ok((_shape, PreparedInput::I64(vec))) => {
                assert_eq!(vec, values);
            }
            other => panic!("Expected prepared i64 input, got: {:?}", other),
        }
    }

    #[test]
    fn test_validate_uint8_success() {
        let values = vec![1u8, 200u8, 17u8];
        let packet = BinaryTensorPacket {
            shape: vec![3],
            dtype: TensorDtype::Uint8,
            data: values.clone(),
        };

        match validate_and_prepare_input(&packet) {
            Ok((_shape, PreparedInput::U8(vec))) => {
                assert_eq!(vec, values);
            }
            other => panic!("Expected prepared u8 input, got: {:?}", other),
        }
    }

    #[test]
    fn test_infer_returns_model_not_loaded() {
        let backend = OnnxBackend::new_cpu();
        let request = InferenceRequest {
            input: BinaryTensorPacket {
                shape: vec![1],
                dtype: TensorDtype::Float32,
                data: f32_bytes(&[0.0f32]),
            },
            additional_inputs: Vec::new(),
            session_id: None,
            metadata: None,
            cancellation: None,
        };

        let res = backend.infer(&request);
        assert!(matches!(res, Err(EngineError::ModelNotLoaded)));
    }

    #[test]
    fn test_builder_device_id_validation() {
        let res = OnnxBackend::builder().with_device_id(-1);
        assert!(res.is_err());
        assert_eq!(res.unwrap_err(), "Device ID must be non-negative");

        let res = OnnxBackend::builder().with_device_id(0);
        assert!(res.is_ok());
    }

    #[test]
    fn test_builder_construction_defaults() {
        let backend = OnnxBackend::builder().build();
        // Check defaults: CPU, Level3, Device 0
        match backend.provider {
            ExecutionProvider::CPU => (),
            _ => panic!("Expected CPU provider by default"),
        }
        assert_eq!(backend.device_id, 0);
        // Optimization level 3 maps to 3
        assert_eq!(backend.optimization_level, 3);
        assert_eq!(backend.session_pool_size(), 1);
    }

    #[test]
    fn test_peak_concurrency_hint_controls_session_pool_size() {
        let backend = OnnxBackend::builder().with_peak_concurrency_hint(8).build();

        assert_eq!(backend.peak_concurrency_hint, Some(8));
        assert_eq!(backend.session_pool_size(), 8);
    }

    #[test]
    fn test_builder_all_optimization_level_mapping() {
        let backend = OnnxBackend::builder()
            .with_optimization_level(GraphOptimizationLevel::All)
            .build();
        assert_eq!(backend.optimization_level, 4);
        assert_eq!(backend.get_opt_level(), GraphOptimizationLevel::All);
    }

    #[test]
    fn test_new_cuda_builder_settings() {
        let backend = OnnxBackend::new_cuda_with_optimization(GraphOptimizationLevel::Level1, 2)
            .expect("builder should accept device id");
        match backend.provider {
            ExecutionProvider::CUDA => (),
            _ => panic!("Expected CUDA provider"),
        }
        assert_eq!(backend.device_id, 2);
        assert_eq!(backend.optimization_level, 1);
    }

    #[test]
    fn test_new_cuda_negative_device_id_rejected() {
        let res = OnnxBackend::new_cuda(-1);
        if let Err(msg) = res {
            assert_eq!(msg, "Device ID must be non-negative");
        } else {
            panic!("Expected error for negative device id");
        }
    }

    #[test]
    fn test_duplicate_additional_input_name_rejected() {
        let additional_inputs = vec![
            NamedTensor {
                name: "dup".to_string(),
                tensor: BinaryTensorPacket {
                    shape: vec![1],
                    dtype: TensorDtype::Uint8,
                    data: vec![1],
                },
            },
            NamedTensor {
                name: "dup".to_string(),
                tensor: BinaryTensorPacket {
                    shape: vec![1],
                    dtype: TensorDtype::Uint8,
                    data: vec![2],
                },
            },
        ];

        let result = ensure_unique_additional_input_names(&additional_inputs);
        assert!(matches!(result, Err(EngineError::InvalidInput { .. })));
    }

    fn f32_packet(shape: Vec<i64>, values: &[f32]) -> BinaryTensorPacket {
        BinaryTensorPacket {
            shape,
            dtype: TensorDtype::Float32,
            data: f32_bytes(values),
        }
    }

    #[test]
    fn test_stack_group_inputs_concatenates_along_batch() {
        let a = f32_packet(vec![1, 3], &[1.0, 2.0, 3.0]);
        let b = f32_packet(vec![1, 3], &[4.0, 5.0, 6.0]);
        let inputs = [&a, &b];

        let (stacked, row_counts) = stack_group_inputs(&inputs).expect("stack ok");

        assert_eq!(stacked.shape, vec![2, 3]);
        assert_eq!(stacked.dtype, TensorDtype::Float32);
        assert_eq!(stacked.data, f32_bytes(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]));
        assert_eq!(row_counts, vec![1, 1]);
    }

    #[test]
    fn test_stack_group_inputs_preserves_multi_row_batches() {
        // dim 0 may differ per request; row_counts must record each so the
        // output split gives every request the right number of rows back.
        let a = f32_packet(vec![2, 2], &[1.0, 2.0, 3.0, 4.0]);
        let b = f32_packet(vec![1, 2], &[5.0, 6.0]);
        let inputs = [&a, &b];

        let (stacked, row_counts) = stack_group_inputs(&inputs).expect("stack ok");

        assert_eq!(stacked.shape, vec![3, 2]);
        assert_eq!(row_counts, vec![2, 1]);
    }

    #[test]
    fn test_stack_group_inputs_rejects_mismatched_trailing_dims() {
        let a = f32_packet(vec![1, 3], &[1.0, 2.0, 3.0]);
        let b = f32_packet(vec![1, 4], &[1.0, 2.0, 3.0, 4.0]);
        let inputs = [&a, &b];

        assert!(stack_group_inputs(&inputs).is_err());
    }

    #[test]
    fn test_stack_group_inputs_rejects_bad_byte_length() {
        let good = f32_packet(vec![1, 3], &[1.0, 2.0, 3.0]);
        // Shape claims 3 elements but only 2 provided.
        let bad = BinaryTensorPacket {
            shape: vec![1, 3],
            dtype: TensorDtype::Float32,
            data: f32_bytes(&[1.0, 2.0]),
        };
        let inputs = [&good, &bad];

        assert!(stack_group_inputs(&inputs).is_err());
    }

    #[test]
    fn test_split_batched_output_round_trips_stack() {
        // Model with a [batch, 3] -> [batch, 2] shape: stack two requests,
        // then split a synthetic [2, 2] output back into two [1, 2] packets.
        let a = f32_packet(vec![1, 3], &[1.0, 2.0, 3.0]);
        let b = f32_packet(vec![1, 3], &[4.0, 5.0, 6.0]);
        let (_stacked, row_counts) = stack_group_inputs(&[&a, &b]).expect("stack ok");

        let batched_output = f32_packet(vec![2, 2], &[10.0, 11.0, 20.0, 21.0]);
        let split = split_batched_output(&batched_output, &row_counts).expect("split ok");

        assert_eq!(split.len(), 2);
        assert_eq!(split[0].shape, vec![1, 2]);
        assert_eq!(split[0].data, f32_bytes(&[10.0, 11.0]));
        assert_eq!(split[1].shape, vec![1, 2]);
        assert_eq!(split[1].data, f32_bytes(&[20.0, 21.0]));
    }

    #[test]
    fn test_split_batched_output_respects_multi_row_counts() {
        let batched_output = f32_packet(vec![3, 2], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let split = split_batched_output(&batched_output, &[2, 1]).expect("split ok");

        assert_eq!(split.len(), 2);
        assert_eq!(split[0].shape, vec![2, 2]);
        assert_eq!(split[0].data, f32_bytes(&[1.0, 2.0, 3.0, 4.0]));
        assert_eq!(split[1].shape, vec![1, 2]);
        assert_eq!(split[1].data, f32_bytes(&[5.0, 6.0]));
    }

    #[test]
    fn test_split_batched_output_rejects_batch_dim_mismatch() {
        // Model ignored the batch axis and returned a single row: must error so
        // the caller falls back to per-request inference.
        let batched_output = f32_packet(vec![1, 2], &[1.0, 2.0]);
        assert!(split_batched_output(&batched_output, &[1, 1]).is_err());
    }

    #[test]
    fn test_request_wants_top_k() {
        use kapsl_engine_api::RequestMetadata;

        let plain = InferenceRequest::new(f32_packet(vec![1, 3], &[1.0, 2.0, 3.0]));
        assert!(!request_wants_top_k(&plain));

        let meta = RequestMetadata {
            top_k: Some(5),
            ..RequestMetadata::default()
        };
        let with_top_k = plain.clone().with_metadata(meta);
        assert!(request_wants_top_k(&with_top_k));
    }
}
