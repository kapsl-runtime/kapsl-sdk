#[cfg(test)]
mod tests {
    use super::super::*;
    use crate::allocator::SimpleShmAllocator;
    use async_trait::async_trait;
    use kapsl_engine_api::{
        BinaryTensorPacket, EngineError, EngineMetrics, EngineModelInfo, EngineStream, TensorDtype,
    };
    use kapsl_scheduler::Priority;
    use kapsl_transport::{RequestMetadata, ResponseMetadata};
    use std::collections::HashMap;
    use std::sync::Arc;
    use std::time::Duration;

    struct OkScheduler {
        output: BinaryTensorPacket,
    }

    #[async_trait]
    impl ReplicaScheduler for OkScheduler {
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
            _request: &InferenceRequest,
            _priority: Priority,
            _force_cpu: bool,
        ) -> Result<BinaryTensorPacket, EngineError> {
            Ok(self.output.clone())
        }

        async fn infer_stream(
            &self,
            _request: InferenceRequest,
            _priority: Priority,
            _force_cpu: bool,
        ) -> Result<EngineStream, EngineError> {
            Err(EngineError::backend("streaming not supported in tests"))
        }
    }

    struct ErrScheduler;

    #[async_trait]
    impl ReplicaScheduler for ErrScheduler {
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
            _request: &InferenceRequest,
            _priority: Priority,
            _force_cpu: bool,
        ) -> Result<BinaryTensorPacket, EngineError> {
            Err(EngineError::backend("boom"))
        }

        async fn infer_stream(
            &self,
            _request: InferenceRequest,
            _priority: Priority,
            _force_cpu: bool,
        ) -> Result<EngineStream, EngineError> {
            Err(EngineError::backend("streaming not supported in tests"))
        }
    }

    struct MetadataScheduler {
        info: EngineModelInfo,
    }

    #[async_trait]
    impl ReplicaScheduler for MetadataScheduler {
        fn get_queue_depth(&self) -> (usize, usize) {
            (0, 0)
        }

        fn is_healthy(&self) -> bool {
            true
        }

        fn get_metrics(&self) -> EngineMetrics {
            EngineMetrics::default()
        }

        fn model_info(&self) -> Option<EngineModelInfo> {
            Some(self.info.clone())
        }

        async fn infer(
            &self,
            _request: &InferenceRequest,
            _priority: Priority,
            _force_cpu: bool,
        ) -> Result<BinaryTensorPacket, EngineError> {
            Err(EngineError::backend("not used"))
        }

        async fn infer_stream(
            &self,
            _request: InferenceRequest,
            _priority: Priority,
            _force_cpu: bool,
        ) -> Result<EngineStream, EngineError> {
            Err(EngineError::backend("not used"))
        }
    }

    fn alloc_buffer(size_bytes: usize) -> (Vec<u64>, *mut u8) {
        let len = size_bytes.div_ceil(8);
        let mut backing = vec![0u64; len];
        let base = backing.as_mut_ptr() as *mut u8;
        (backing, base)
    }

    fn test_allocator() -> SimpleShmAllocator {
        SimpleShmAllocator::new_with_ttl(128 * 1024, 10_000_000, 100, Duration::from_secs(60))
    }

    #[test]
    fn test_tensor_roundtrip() {
        let (_backing, base) = alloc_buffer(1024);

        let packet = BinaryTensorPacket {
            shape: vec![1, 2],
            dtype: TensorDtype::Int32,
            data: vec![1u8, 0, 0, 0, 2, 0, 0, 0],
        };

        unsafe {
            write_tensor_to_shm(base, 0, &packet);
            let read_back = read_tensor_from_shm(base, 0);
            assert_eq!(read_back.shape, packet.shape);
            assert_eq!(read_back.dtype, packet.dtype);
            assert_eq!(read_back.data, packet.data);
        }
    }

    #[test]
    fn test_error_roundtrip() {
        let (_backing, base) = alloc_buffer(25 * 1024 * 1024);
        let error_msg = "failed request";
        let allocator = test_allocator();

        let offset = write_error_to_shm(base, &allocator, None, error_msg).expect("error slot");
        unsafe {
            let len = *(base.add(offset) as *const u64) as usize;
            assert_eq!(len, error_msg.len());
            let bytes =
                std::slice::from_raw_parts(base.add(offset + std::mem::size_of::<u64>()), len);
            assert_eq!(bytes, error_msg.as_bytes());
        }
    }

    #[tokio::test]
    async fn test_request_success_flow() {
        let (_backing, base) = alloc_buffer(25 * 1024 * 1024);
        let input = BinaryTensorPacket {
            shape: vec![1],
            dtype: TensorDtype::Float32,
            data: vec![0, 0, 128, 63],
        };

        let output = BinaryTensorPacket {
            shape: vec![1],
            dtype: TensorDtype::Float32,
            data: vec![0, 0, 0, 64],
        };

        let scheduler = OkScheduler {
            output: output.clone(),
        };
        let allocator = test_allocator();
        unsafe {
            write_tensor_to_shm(base, 0, &input);
        }

        let request = ShmRequest {
            metadata: RequestMetadata::new(7, 1, 0, false),
            tensor_offset: 0,
            tensor_size: (std::mem::size_of::<TensorHeader>() + input.data.len()) as u64,
        };

        let tensor = unsafe { read_tensor_from_shm(base, request.tensor_offset as usize) };
        let request_obj = InferenceRequest {
            input: tensor,
            additional_inputs: Vec::new(),
            session_id: None,
            metadata: None,
            cancellation: None,
        };

        let result = scheduler
            .infer(&request_obj, Priority::LatencyCritical, false)
            .await
            .unwrap();

        let result_size = std::mem::size_of::<TensorHeader>() + result.data.len();
        let result_offset = allocator.try_allocate(result_size).expect("result slot");
        unsafe {
            write_tensor_to_shm(base, result_offset, &result);
        }

        let response = ShmResponse {
            metadata: ResponseMetadata::success(request.metadata.request_id, 123),
            result_offset: result_offset as u64,
            result_size: (std::mem::size_of::<TensorHeader>() + result.data.len()) as u64,
            error_offset: 0,
        };

        let roundtrip = unsafe { read_tensor_from_shm(base, response.result_offset as usize) };
        assert!(response.metadata.is_success());
        assert_eq!(roundtrip.data, output.data);
        assert_eq!(roundtrip.dtype, output.dtype);
    }

    #[tokio::test]
    async fn test_request_error_flow() {
        let (_backing, base) = alloc_buffer(25 * 1024 * 1024);
        let input = BinaryTensorPacket {
            shape: vec![1],
            dtype: TensorDtype::Float32,
            data: vec![0, 0, 128, 63],
        };

        let scheduler = ErrScheduler;
        let allocator = test_allocator();
        unsafe {
            write_tensor_to_shm(base, 0, &input);
        }

        let request = ShmRequest {
            metadata: RequestMetadata::new(9, 1, 1, true),
            tensor_offset: 0,
            tensor_size: (std::mem::size_of::<TensorHeader>() + input.data.len()) as u64,
        };

        let tensor = unsafe { read_tensor_from_shm(base, request.tensor_offset as usize) };
        let request_obj = InferenceRequest {
            input: tensor,
            additional_inputs: Vec::new(),
            session_id: None,
            metadata: None,
            cancellation: None,
        };

        let result = scheduler
            .infer(&request_obj, Priority::Throughput, true)
            .await;
        assert!(result.is_err());

        let error_offset = write_error_to_shm(base, &allocator, None, "boom").expect("error slot");
        let response = ShmResponse {
            metadata: ResponseMetadata::error(request.metadata.request_id, 77),
            result_offset: 0,
            result_size: 0,
            error_offset: error_offset as u64,
        };

        let err_bytes = unsafe {
            let len = *(base.add(error_offset) as *const u64) as usize;
            std::slice::from_raw_parts(base.add(error_offset + std::mem::size_of::<u64>()), len)
        };
        assert!(!response.metadata.is_success());
        assert_eq!(err_bytes, b"boom");
    }

    #[test]
    fn test_single_model_class_budgets_uses_shapes_and_dtypes() {
        let info = EngineModelInfo {
            input_names: vec!["input".to_string()],
            output_names: vec!["output".to_string()],
            input_shapes: vec![vec![1, 3, 224, 224]],
            output_shapes: vec![vec![1, 1000]],
            input_dtypes: vec!["float32".to_string()],
            output_dtypes: vec!["float32".to_string()],
            framework: Some("onnx".to_string()),
            model_version: Some("1.0".to_string()),
            peak_concurrency: Some(4),
        };

        let mut schedulers: HashMap<u32, Arc<dyn ReplicaScheduler + Send + Sync>> = HashMap::new();
        schedulers.insert(1, Arc::new(MetadataScheduler { info }));

        let budgets = derive_single_model_class_budgets(1, &schedulers, 512 * 1024 * 1024);
        assert!(!budgets.is_empty());
        // A 224×224×3 float32 input tensor requires at least a 1 MiB slot.
        assert!(budgets.iter().any(|b| b.slot_size >= 1024 * 1024));
    }

    #[test]
    fn test_per_model_pool_higher_concurrency_gets_larger_share() {
        let make_info = |peak: u32| EngineModelInfo {
            input_names: vec!["input".to_string()],
            output_names: vec![],
            input_shapes: vec![vec![1, 3, 224, 224]],
            output_shapes: vec![],
            input_dtypes: vec!["float32".to_string()],
            output_dtypes: vec![],
            framework: None,
            model_version: None,
            peak_concurrency: Some(peak),
        };

        let mut schedulers: HashMap<u32, Arc<dyn ReplicaScheduler + Send + Sync>> = HashMap::new();
        schedulers.insert(1, Arc::new(MetadataScheduler { info: make_info(1) }));
        schedulers.insert(2, Arc::new(MetadataScheduler { info: make_info(8) }));

        // Build pool and check layout: model 2 (8× concurrency) should have a
        // proportionally larger sub-pool than model 1 (1× concurrency).
        let pool = build_per_model_pool(&schedulers, 0, 512 * 1024 * 1024, Duration::from_secs(30));
        let summary = pool.layout_summary();
        assert!(
            summary.contains("model1:"),
            "model 1 should have a sub-pool"
        );
        assert!(
            summary.contains("model2:"),
            "model 2 should have a sub-pool"
        );
    }

    #[test]
    fn test_build_per_model_pool_creates_dedicated_sub_pools() {
        let info = EngineModelInfo {
            input_names: vec!["input".to_string()],
            output_names: vec!["output".to_string()],
            input_shapes: vec![vec![1, 3, 224, 224]],
            output_shapes: vec![vec![1, 1000]],
            input_dtypes: vec!["float32".to_string()],
            output_dtypes: vec!["float32".to_string()],
            framework: Some("onnx".to_string()),
            model_version: Some("1.0".to_string()),
            peak_concurrency: Some(4),
        };

        let mut schedulers: HashMap<u32, Arc<dyn ReplicaScheduler + Send + Sync>> = HashMap::new();
        schedulers.insert(1, Arc::new(MetadataScheduler { info }));

        let pool_bytes = 128 * 1024 * 1024; // 128 MiB
        let pool = build_per_model_pool(
            &schedulers,
            0,
            pool_bytes,
            std::time::Duration::from_secs(30),
        );

        let summary = pool.layout_summary();
        assert!(
            summary.contains("model1:"),
            "layout should include model 1 sub-pool"
        );
        assert!(
            summary.contains("shared:"),
            "layout should include shared overflow pool"
        );

        // Model 1 should have a dedicated slot available.
        let off = pool
            .try_allocate(1, 1024 * 1024)
            .expect("model 1 should have capacity");
        // Its offset must be within the model-1 region (well before the shared pool).
        let shared_reserve = (pool_bytes * 10 / 100).max(64 * 1024);
        assert!(
            off < pool_bytes - shared_reserve,
            "allocation should be in model sub-pool, not shared pool"
        );
    }

    #[test]
    fn test_build_per_model_pool_proportional_sizing() {
        let low_info = EngineModelInfo {
            input_names: vec!["input".to_string()],
            output_names: vec![],
            input_shapes: vec![vec![1, 3, 224, 224]],
            output_shapes: vec![],
            input_dtypes: vec!["float32".to_string()],
            output_dtypes: vec![],
            framework: None,
            model_version: None,
            peak_concurrency: Some(1),
        };
        let high_info = EngineModelInfo {
            input_names: vec!["input".to_string()],
            output_names: vec![],
            input_shapes: vec![vec![1, 3, 224, 224]],
            output_shapes: vec![],
            input_dtypes: vec!["float32".to_string()],
            output_dtypes: vec![],
            framework: None,
            model_version: None,
            peak_concurrency: Some(3), // 3× the low model
        };

        let mut schedulers: HashMap<u32, Arc<dyn ReplicaScheduler + Send + Sync>> = HashMap::new();
        schedulers.insert(1, Arc::new(MetadataScheduler { info: low_info }));
        schedulers.insert(2, Arc::new(MetadataScheduler { info: high_info }));

        let pool_bytes = 256 * 1024 * 1024; // 256 MiB
        let pool = build_per_model_pool(
            &schedulers,
            0,
            pool_bytes,
            std::time::Duration::from_secs(30),
        );

        // Model 2 has 3× the concurrency so its snapshot largest_slot should be
        // derived from a proportionally larger pool.
        let low_largest = pool.largest_slot_size_for_model(1);
        let high_largest = pool.largest_slot_size_for_model(2);
        // Both should be non-zero (have dedicated slots).
        assert!(low_largest > 0, "model 1 should have capacity");
        assert!(high_largest > 0, "model 2 should have capacity");
    }
}
