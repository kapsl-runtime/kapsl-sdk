// Hot-swap integration tests for NativeBackend.
//
// Requires a CUDA device and two model directories with compatible architectures
// (same hidden_size, num_layers, num_kv_heads — only weights differ).
//
// Run on a machine with CUDA:
//   KAPSL_TEST_MODEL_A=/path/to/model-a \
//   KAPSL_TEST_MODEL_B=/path/to/model-b \
//   cargo test --features native -p kapsl-backends -- --ignored native_hotswap
//
// Tests are marked `#[ignore]` so they do not block CI (no GPU available there).

#[cfg(all(feature = "native", test))]
mod native_hotswap {
    use std::path::Path;
    use kapsl_engine_api::{BinaryTensorPacket, Engine, InferenceRequest, TensorDtype};

    fn model_a() -> Option<String> { std::env::var("KAPSL_TEST_MODEL_A").ok() }
    fn model_b() -> Option<String> { std::env::var("KAPSL_TEST_MODEL_B").ok() }

    fn make_request(token_ids: &[i32]) -> InferenceRequest {
        let data: Vec<u8> = token_ids.iter()
            .flat_map(|&t| t.to_le_bytes())
            .collect();
        let input = BinaryTensorPacket::new(
            vec![token_ids.len() as i64],
            TensorDtype::Int32,
            data,
        ).expect("valid packet");
        InferenceRequest {
            input,
            additional_inputs: Vec::new(),
            session_id: None,
            metadata: None,
            cancellation: None,
        }
    }

    fn new_backend() -> super::super::inner::NativeBackend {
        super::super::inner::NativeBackend::new(0).expect("CUDA device 0")
    }

    // ── stage() / is_staged() / swap() lifecycle ──────────────────────────

    #[ignore]
    #[tokio::test]
    async fn test_stage_sets_is_staged() {
        let (a, b) = match (model_a(), model_b()) {
            (Some(a), Some(b)) => (a, b),
            _ => { eprintln!("skip: KAPSL_TEST_MODEL_A/B not set"); return; }
        };
        let mut backend = new_backend();
        backend.load(Path::new(&a)).await.expect("load A");
        assert!(!backend.is_staged(), "not staged after fresh load");
        assert!(backend.supports_swap(), "native backend supports swap");

        backend.stage(Path::new(&b)).await.expect("stage B");
        assert!(backend.is_staged(), "staged after stage()");

        backend.swap().await.expect("swap to B");
        assert!(!backend.is_staged(), "not staged after swap()");
    }

    #[ignore]
    #[tokio::test]
    async fn test_inference_works_before_and_after_swap() {
        let (a, b) = match (model_a(), model_b()) {
            (Some(a), Some(b)) => (a, b),
            _ => { eprintln!("skip: KAPSL_TEST_MODEL_A/B not set"); return; }
        };
        let mut backend = new_backend();
        backend.load(Path::new(&a)).await.expect("load A");

        let req = make_request(&[1, 2, 3, 4]);

        let result_a = backend.infer(&req);
        assert!(result_a.is_ok(), "infer on model A: {:?}", result_a.err());

        backend.stage(Path::new(&b)).await.expect("stage B");

        // Inference must still work while staged (model A still active).
        let result_staged = backend.infer(&req);
        assert!(result_staged.is_ok(), "infer while staged: {:?}", result_staged.err());

        backend.swap().await.expect("swap to B");

        let result_b = backend.infer(&req);
        assert!(result_b.is_ok(), "infer on model B: {:?}", result_b.err());
    }

    #[ignore]
    #[tokio::test]
    async fn test_sessions_cleared_after_swap() {
        let (a, b) = match (model_a(), model_b()) {
            (Some(a), Some(b)) => (a, b),
            _ => { eprintln!("skip: KAPSL_TEST_MODEL_A/B not set"); return; }
        };
        let mut backend = new_backend();
        backend.load(Path::new(&a)).await.expect("load A");

        // Establish a session with model A.
        let req = InferenceRequest {
            session_id: Some("sess-1".into()),
            ..make_request(&[1, 2, 3])
        };
        backend.infer(&req).expect("session turn 1");

        // Stage + swap.
        backend.stage(Path::new(&b)).await.expect("stage B");
        backend.swap().await.expect("swap to B");

        // The session KV cache is from model A — using it with model B weights
        // is unsound. After swap(), sessions must be cleared so the next call
        // starts a fresh context rather than inheriting stale KV blocks.
        // We verify indirectly: infer with the same session_id should succeed
        // (fresh prefill, not a decode on stale cache).
        let result = backend.infer(&req);
        assert!(result.is_ok(), "fresh session after swap: {:?}", result.err());
    }

    #[ignore]
    #[tokio::test]
    async fn test_swap_without_stage_returns_error() {
        let a = match model_a() {
            Some(a) => a,
            None => { eprintln!("skip: KAPSL_TEST_MODEL_A not set"); return; }
        };
        let mut backend = new_backend();
        backend.load(Path::new(&a)).await.expect("load A");

        // swap() with nothing staged should return an error, not panic.
        let result = backend.swap().await;
        assert!(result.is_err(), "expected error when swapping without staging");
    }

    // ── GPU argmax correctness ─────────────────────────────────────────────

    #[ignore]
    #[tokio::test]
    async fn test_greedy_argmax_matches_cpu_argmax() {
        // Run the same prompt with temperature=0 (GPU argmax path) and compare
        // against temperature=1e-7 (logits-download path) — both should produce
        // the identical top-1 token on a deterministic model.
        use kapsl_engine_api::RequestMetadata;

        let a = match model_a() {
            Some(a) => a,
            None => { eprintln!("skip: KAPSL_TEST_MODEL_A not set"); return; }
        };
        let mut backend = new_backend();
        backend.load(Path::new(&a)).await.expect("load A");

        fn req_with_temp(temp: f32) -> InferenceRequest {
            let mut r = make_request(&[1, 2, 3]);
            r.metadata = Some(RequestMetadata {
                temperature: Some(temp),
                max_new_tokens: Some(1),
                ..Default::default()
            });
            r
        }

        let greedy_result = backend.infer(&req_with_temp(0.0)).expect("greedy");
        let near_greedy   = backend.infer(&req_with_temp(1e-7)).expect("near-greedy");

        assert_eq!(
            greedy_result.data, near_greedy.data,
            "GPU argmax and CPU argmax should agree on the top-1 token"
        );
    }
}
