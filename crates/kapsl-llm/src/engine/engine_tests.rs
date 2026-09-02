#[cfg(test)]
mod tests {
    use super::super::{
        build_kv_array_f16, build_kv_array_f32_from_f16, device_kv_sequence_len, empty_kv_shape,
        empty_kv_shape_with_seq_len, infer_kv_layout, normalize_metadata_safe_load_setting,
        parse_safe_load_env_setting, parse_safe_load_setting, resolve_prepend_bos_token_id,
        tokenizer_add_bos_token, tokenizer_declared_bos_token_id, KvCacheMode, KvLayout, LLMEngine,
        LLMMetrics, SafeLoadSetting, SamplingParams, SchedulerConfig, SequenceGroup,
    };
    use crate::allocation_scope::{
        DeviceAllocationClass, DeviceAllocationScope, DeviceAllocationScopeGuard,
        DeviceAllocationScopeKind, DeviceAllocationScopeProvider,
    };
    use half::f16;
    use ort::tensor::TensorElementType;
    use serde_json::json;
    use std::collections::HashMap;
    use std::fs;
    use std::sync::atomic::AtomicU64;
    use std::sync::{Arc, Mutex};
    use std::time::Duration;
    use tokenizers::decoders::byte_fallback::ByteFallback;
    use tokenizers::decoders::metaspace::Metaspace;
    use tokenizers::models::bpe::BPE;
    use tokenizers::Tokenizer;
    use tokio::sync::mpsc;

    #[derive(Default)]
    struct RecordingAllocationScopeProvider {
        scopes: Mutex<Vec<DeviceAllocationScope>>,
    }

    impl DeviceAllocationScopeProvider for RecordingAllocationScopeProvider {
        fn enter(
            &self,
            scope: &DeviceAllocationScope,
        ) -> Result<Box<dyn DeviceAllocationScopeGuard>, String> {
            self.scopes.lock().unwrap().push(scope.clone());
            Ok(Box::new(()))
        }
    }

    #[test]
    fn parse_safe_load_setting_handles_bool_and_strings() {
        assert_eq!(
            parse_safe_load_setting(&json!(true)),
            Some(SafeLoadSetting::ForceOn)
        );
        assert_eq!(
            parse_safe_load_setting(&json!(false)),
            Some(SafeLoadSetting::ForceOff)
        );
        assert_eq!(
            parse_safe_load_setting(&json!("auto")),
            Some(SafeLoadSetting::Auto)
        );
        assert_eq!(
            parse_safe_load_setting(&json!("on")),
            Some(SafeLoadSetting::ForceOn)
        );
        assert_eq!(
            parse_safe_load_setting(&json!("off")),
            Some(SafeLoadSetting::ForceOff)
        );
        assert_eq!(parse_safe_load_setting(&json!("maybe")), None);
    }

    #[test]
    fn parse_safe_load_env_setting_handles_auto_and_bool_literals() {
        assert_eq!(
            parse_safe_load_env_setting("1"),
            Some(SafeLoadSetting::ForceOn)
        );
        assert_eq!(
            parse_safe_load_env_setting("false"),
            Some(SafeLoadSetting::ForceOff)
        );
        assert_eq!(
            parse_safe_load_env_setting("auto"),
            Some(SafeLoadSetting::Auto)
        );
        assert_eq!(parse_safe_load_env_setting("maybe"), None);
    }

    #[test]
    fn metadata_safe_load_true_is_advisory_auto() {
        assert_eq!(
            normalize_metadata_safe_load_setting(SafeLoadSetting::ForceOn),
            SafeLoadSetting::Auto
        );
        assert_eq!(
            normalize_metadata_safe_load_setting(SafeLoadSetting::ForceOff),
            SafeLoadSetting::ForceOff
        );
        assert_eq!(
            normalize_metadata_safe_load_setting(SafeLoadSetting::Auto),
            SafeLoadSetting::Auto
        );
    }

    #[test]
    fn bos_insertion_follows_tokenizer_policy_across_model_families() {
        // Qwen and GPT-style tokenizers commonly expose a bos_token_id for
        // generation semantics while explicitly disabling automatic insertion.
        assert_eq!(
            resolve_prepend_bos_token_id(Some(151_643), Some(false)),
            None
        );
        assert_eq!(
            resolve_prepend_bos_token_id(Some(50_256), Some(false)),
            None
        );

        // Gemma/Llama-style tokenizers that opt in retain their configured BOS.
        assert_eq!(resolve_prepend_bos_token_id(Some(2), Some(true)), Some(2));
        assert_eq!(
            resolve_prepend_bos_token_id(Some(128_000), Some(true)),
            Some(128_000)
        );

        // Older DeepSeek and other packages without tokenizer_config.json keep
        // the legacy config.json behavior instead of changing silently.
        assert_eq!(
            resolve_prepend_bos_token_id(Some(100_000), None),
            Some(100_000)
        );
        assert_eq!(resolve_prepend_bos_token_id(None, Some(true)), None);
    }

    #[test]
    fn tokenizer_bos_policy_is_read_from_the_model_assets() {
        let root =
            std::env::temp_dir().join(format!("kapsl_llm_bos_policy_{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&root).expect("create model root");
        let model_path = root.join("model.onnx");
        fs::write(&model_path, "").expect("model file");
        fs::write(
            root.join("tokenizer_config.json"),
            json!({ "add_bos_token": false }).to_string(),
        )
        .expect("tokenizer config");

        assert_eq!(tokenizer_add_bos_token(&model_path), Some(false));
    }

    #[test]
    fn legacy_tokenizer_post_processor_supplies_deepseek_bos() {
        let root =
            std::env::temp_dir().join(format!("kapsl_llm_deepseek_bos_{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&root).expect("create model root");
        let model_path = root.join("model.onnx");
        fs::write(&model_path, "").expect("model file");
        fs::write(
            root.join("tokenizer.json"),
            json!({
                "post_processor": {
                    "single": [{
                        "SpecialToken": { "id": "<｜begin▁of▁sentence｜>" }
                    }]
                }
            })
            .to_string(),
        )
        .expect("tokenizer json");
        let tokenizer = bpe_tokenizer(HashMap::from([("<｜begin▁of▁sentence｜>".to_string(), 0)]));

        assert_eq!(
            tokenizer_declared_bos_token_id(&model_path, &tokenizer),
            Some(0)
        );
    }

    #[test]
    fn infer_kv_layout_prefers_head_dim_axis() {
        let mut shapes = HashMap::new();
        shapes.insert("past_key_values.0.key".to_string(), vec![1, 4, 8, 16]);
        assert!(matches!(
            infer_kv_layout(&shapes, 4, 8),
            KvLayout::HeadDimFirst
        ));

        shapes.insert("past_key_values.0.key".to_string(), vec![1, 4, 16, 8]);
        assert!(matches!(infer_kv_layout(&shapes, 4, 8), KvLayout::SeqFirst));

        let shapes = HashMap::new();
        assert!(matches!(infer_kv_layout(&shapes, 4, 8), KvLayout::SeqFirst));
    }

    #[test]
    fn build_kv_array_seq_first_layout() {
        let data = vec![
            f16::from_f32(1.0),
            f16::from_f32(2.0),
            f16::from_f32(3.0),
            f16::from_f32(4.0),
            f16::from_f32(5.0),
            f16::from_f32(6.0),
        ];
        let arr = build_kv_array_f16(&data, 1, 2, 2, KvLayout::SeqFirst, "key").expect("kv array");
        assert_eq!(arr.shape(), &[1, 1, 2, 2]);
        let got: Vec<f32> = arr.iter().map(|v| v.to_f32()).collect();
        assert_eq!(got, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn build_kv_array_head_dim_first_layout() {
        let data = vec![
            f16::from_f32(1.0),
            f16::from_f32(2.0),
            f16::from_f32(3.0),
            f16::from_f32(4.0),
            f16::from_f32(5.0),
            f16::from_f32(6.0),
        ];
        let arr = build_kv_array_f32_from_f16(&data, 1, 2, 2, KvLayout::HeadDimFirst, "key")
            .expect("kv array");
        assert_eq!(arr.shape(), &[1, 1, 2, 2]);
        let got: Vec<f32> = arr.iter().cloned().collect();
        assert_eq!(got, vec![1.0, 3.0, 2.0, 4.0]);
    }

    #[test]
    fn build_kv_array_rejects_invalid_stride() {
        let data = vec![f16::from_f32(1.0)];
        assert!(build_kv_array_f16(&data, 0, 1, 1, KvLayout::SeqFirst, "key").is_err());
    }

    #[test]
    fn empty_kv_shape_prefers_layout_for_rank4() {
        let shape = empty_kv_shape(Some(&vec![1, 2, 3, 4]), KvLayout::SeqFirst, 8, 16);
        assert_eq!(shape, vec![1, 8, 1, 16]);

        let shape = empty_kv_shape(Some(&vec![2, -1, 5]), KvLayout::SeqFirst, 4, 8);
        assert_eq!(shape, vec![1, 1, 5]);

        let shape = empty_kv_shape(None, KvLayout::HeadDimFirst, 3, 7);
        assert_eq!(shape, vec![1, 3, 7, 1]);
    }

    #[test]
    fn cuda_empty_kv_shape_uses_a_real_zero_length_sequence() {
        let seq_first =
            empty_kv_shape_with_seq_len(Some(&vec![1, 8, -1, 16]), KvLayout::SeqFirst, 8, 16, 0);
        assert_eq!(seq_first, vec![1, 8, 0, 16]);

        let head_dim_first = empty_kv_shape_with_seq_len(
            Some(&vec![1, 8, 16, -1]),
            KvLayout::HeadDimFirst,
            8,
            16,
            0,
        );
        assert_eq!(head_dim_first, vec![1, 8, 16, 0]);
    }

    #[test]
    fn device_kv_sequence_axis_follows_export_layout() {
        assert_eq!(
            device_kv_sequence_len(&[1, 8, 37, 16], KvLayout::SeqFirst),
            Some(37)
        );
        assert_eq!(
            device_kv_sequence_len(&[1, 8, 16, 41], KvLayout::HeadDimFirst),
            Some(41)
        );
        assert_eq!(
            device_kv_sequence_len(&[2, 8, 37, 16], KvLayout::SeqFirst),
            None
        );
        assert_eq!(
            device_kv_sequence_len(&[1, 8, 16], KvLayout::SeqFirst),
            None
        );
    }

    #[test]
    fn device_kv_contract_requires_paired_past_and_present_tensors() {
        let (_request_tx, request_rx) = mpsc::channel(1);
        let mut engine = LLMEngine::new(
            SchedulerConfig {
                max_num_batched_tokens: 16,
                max_num_seqs: 1,
                max_paddings: 0,
            },
            16,
            1,
            request_rx,
            Arc::new(Mutex::new(LLMMetrics::default())),
            Some("cuda".to_string()),
            Some(0),
            None,
            true,
        );
        engine.num_layers = 1;
        engine.model_input_names = Arc::new(
            [
                "input_ids".to_string(),
                "past_key_values.0.key".to_string(),
                "past_key_values.0.value".to_string(),
            ]
            .into_iter()
            .collect(),
        );
        engine.model_input_shapes = Arc::new(HashMap::from([
            ("past_key_values.0.key".to_string(), vec![1, 8, -1, 16]),
            ("past_key_values.0.value".to_string(), vec![1, 8, -1, 16]),
        ]));
        engine.model_input_types = Arc::new(HashMap::from([
            (
                "past_key_values.0.key".to_string(),
                TensorElementType::Float16,
            ),
            (
                "past_key_values.0.value".to_string(),
                TensorElementType::Float16,
            ),
        ]));
        engine.model_output_names = Arc::new(
            [
                "logits".to_string(),
                "present.0.key".to_string(),
                "present.0.value".to_string(),
            ]
            .into_iter()
            .collect(),
        );

        assert!(engine.device_kv_contract_supported());

        Arc::make_mut(&mut engine.model_output_names).remove("present.0.value");
        assert!(!engine.device_kv_contract_supported());

        Arc::make_mut(&mut engine.model_output_names).insert("present.0.value".to_string());
        Arc::make_mut(&mut engine.model_input_shapes).remove("past_key_values.0.value");
        assert!(!engine.device_kv_contract_supported());
    }

    #[test]
    fn external_allocation_scopes_preserve_owner_request_sets_and_unique_ids() {
        let (_request_tx, request_rx) = mpsc::channel(1);
        let mut engine = LLMEngine::new(
            SchedulerConfig {
                max_num_batched_tokens: 16,
                max_num_seqs: 2,
                max_paddings: 0,
            },
            16,
            1,
            request_rx,
            Arc::new(Mutex::new(LLMMetrics::default())),
            Some("cuda".to_string()),
            Some(0),
            None,
            true,
        );
        let provider = Arc::new(RecordingAllocationScopeProvider::default());
        let scope_ids = Arc::new(AtomicU64::new(1));
        engine.set_memory_owner(7, 3);
        engine.set_allocation_scope_provider(provider.clone());
        engine.set_allocation_scope_id_source(scope_ids.clone());

        engine.use_env_allocators = false;
        assert!(engine
            .enter_allocation_scope(0, DeviceAllocationClass::PersistentWeights, &[])
            .is_err());
        engine.use_env_allocators = true;

        let (response_tx, _response_rx) = mpsc::channel(1);
        let unscoped_group = Arc::new(Mutex::new(SequenceGroup::new(
            "request".to_string(),
            None,
            "prompt".to_string(),
            Vec::new(),
            sampling_params(0.0),
            None,
            response_tx,
        )));
        assert!(engine.allocation_request_ids(&[unscoped_group]).is_err());

        drop(
            engine
                .enter_allocation_scope(0, DeviceAllocationClass::PersistentWeights, &[])
                .unwrap(),
        );
        drop(
            engine
                .enter_allocation_scope(0, DeviceAllocationClass::TransientWorkspace, &[41])
                .unwrap(),
        );
        drop(
            engine
                .enter_allocation_scope(0, DeviceAllocationClass::TransientWorkspace, &[41, 42])
                .unwrap(),
        );

        let (_request_tx, request_rx) = mpsc::channel(1);
        let mut reloaded_engine = LLMEngine::new(
            SchedulerConfig {
                max_num_batched_tokens: 16,
                max_num_seqs: 1,
                max_paddings: 0,
            },
            16,
            1,
            request_rx,
            Arc::new(Mutex::new(LLMMetrics::default())),
            Some("cuda".to_string()),
            Some(0),
            None,
            true,
        );
        reloaded_engine.set_memory_owner(7, 3);
        reloaded_engine.set_allocation_scope_provider(provider.clone());
        reloaded_engine.set_allocation_scope_id_source(scope_ids);
        drop(
            reloaded_engine
                .enter_allocation_scope(0, DeviceAllocationClass::TransientWorkspace, &[43])
                .unwrap(),
        );

        let scopes = provider.scopes.lock().unwrap();
        assert_eq!(scopes.len(), 4);
        assert_eq!(scopes[0].kind, DeviceAllocationScopeKind::Model);
        assert_eq!(scopes[1].kind, DeviceAllocationScopeKind::Request);
        assert_eq!(scopes[2].kind, DeviceAllocationScopeKind::RequestBatch);
        assert_eq!(scopes[2].request_ids, vec![41, 42]);
        assert!(scopes.iter().all(|scope| {
            scope.model_id == 7
                && scope.replica_id == 3
                && scope.device_id == 0
                && scope.is_well_formed()
        }));
        assert_eq!(scopes[3].kind, DeviceAllocationScopeKind::Request);
        assert_eq!(
            scopes
                .iter()
                .map(|scope| scope.scope_id)
                .collect::<Vec<_>>(),
            vec![1, 2, 3, 4]
        );
    }

    #[test]
    fn device_kv_paged_capacity_uses_the_configured_block_budget() {
        let (_request_tx, request_rx) = mpsc::channel(1);
        let mut engine = LLMEngine::new(
            SchedulerConfig {
                max_num_batched_tokens: 16,
                max_num_seqs: 1,
                max_paddings: 0,
            },
            16,
            1,
            request_rx,
            Arc::new(Mutex::new(LLMMetrics::default())),
            Some("cuda".to_string()),
            Some(0),
            None,
            true,
        );
        engine.num_layers = 2;
        engine.num_heads = 3;
        engine.head_dim = 5;
        engine.kv_cache_config.mode = KvCacheMode::Paged;
        engine.kv_cache_config.block_size = 4;
        engine.kv_cache_config.total_blocks = 10;

        // 2 layers * key/value * 3 heads * 5 elements * 2 bytes = 120
        // bytes/token; the configured pool represents 40 tokens.
        assert_eq!(engine.device_kv_usage(), (0, 4_800, 0));

        engine.device_kv_enabled = true;
        engine.max_seq_len = 12;
        assert_eq!(
            engine.projected_kv_reservation_bytes(10, 5, false),
            Some(1_440)
        );
        assert_eq!(
            engine.projected_kv_reservation_bytes(10, 5, true),
            Some(1_800)
        );
    }

    fn sampling_params(temperature: f32) -> SamplingParams {
        SamplingParams {
            max_tokens: 8,
            min_tokens: 2,
            temperature,
            top_p: 1.0,
            top_k: 1,
            stop_token_ids: vec![0],
            repetition_penalty: 1.0,
            seed: Some(42),
        }
    }

    #[test]
    fn greedy_sampling_suppresses_stop_token_below_minimum() {
        let logits = [10.0, 9.0, 8.0];
        let params = sampling_params(0.0);
        let mut rng = 42;

        assert_eq!(
            LLMEngine::sample_next_token(&logits, &params, 0, &mut rng),
            1
        );
        assert_eq!(
            LLMEngine::sample_next_token(&logits, &params, 2, &mut rng),
            0
        );
    }

    #[test]
    fn probabilistic_sampling_suppresses_stop_token_below_minimum() {
        let logits = [10.0, 9.0, 8.0];
        let params = sampling_params(1.0);
        let mut rng = 42;

        assert_eq!(
            LLMEngine::sample_next_token(&logits, &params, 0, &mut rng),
            1
        );
        assert_eq!(
            LLMEngine::sample_next_token(&logits, &params, 2, &mut rng),
            0
        );
    }

    #[tokio::test]
    async fn run_loop_stops_after_request_channel_closes() {
        let (request_tx, request_rx) = mpsc::channel(1);
        let mut engine = LLMEngine::new(
            SchedulerConfig {
                max_num_batched_tokens: 16,
                max_num_seqs: 1,
                max_paddings: 0,
            },
            16,
            1,
            request_rx,
            Arc::new(Mutex::new(LLMMetrics::default())),
            None,
            None,
            None,
            false,
        );

        let mut task = tokio::spawn(async move { engine.run_loop().await });
        drop(request_tx);

        match tokio::time::timeout(Duration::from_secs(1), &mut task).await {
            Ok(result) => result.expect("engine loop task panicked"),
            Err(_) => {
                task.abort();
                panic!("engine loop did not stop after its final request sender was dropped");
            }
        }
    }

    #[test]
    fn probabilistic_sampling_filters_unbounded_top_k_in_one_pass() {
        let logits = [10.0, 9.0, 8.0];
        let mut params = sampling_params(1.0);
        params.top_k = 0;
        params.stop_token_ids = vec![0, 1];
        let mut rng = 42;

        assert_eq!(
            LLMEngine::sample_next_token(&logits, &params, 0, &mut rng),
            2
        );
    }

    fn bpe_tokenizer(vocab: HashMap<String, u32>) -> Tokenizer {
        let model = BPE::builder()
            .vocab_and_merges(vocab, Vec::new())
            .byte_fallback(true)
            .build()
            .expect("BPE model");
        Tokenizer::new(model)
    }

    fn decode_token(
        tokenizer: &Tokenizer,
        seq: &mut super::super::Sequence,
        token_id: u32,
        finished: bool,
    ) -> String {
        seq.append_token_id(token_id, 0.0);
        if finished {
            seq.status = super::super::SequenceStatus::Finished(super::super::FinishReason::Length);
        }
        LLMEngine::decode_next_token(tokenizer, seq, token_id)
    }

    #[test]
    fn incremental_decode_preserves_partial_utf8_boundaries() {
        let mut tokenizer = bpe_tokenizer(HashMap::from([
            ("<0x20>".to_string(), 0),
            ("<0xC3>".to_string(), 1),
            ("<0xA9>".to_string(), 2),
        ]));
        tokenizer.with_decoder(Some(ByteFallback::default()));
        let mut seq = super::super::Sequence::new(1, String::new(), Vec::new());

        let chunks = [
            decode_token(&tokenizer, &mut seq, 0, false),
            decode_token(&tokenizer, &mut seq, 1, false),
            decode_token(&tokenizer, &mut seq, 2, true),
        ];

        assert_eq!(chunks, [" ", "", "é"]);
        assert_eq!(chunks.concat(), tokenizer.decode(&[0, 1, 2], true).unwrap());
    }

    #[test]
    fn incremental_decode_preserves_metaspace_between_tokens() {
        let mut tokenizer = bpe_tokenizer(HashMap::from([("▁This".to_string(), 0)]));
        tokenizer.with_decoder(Some(Metaspace::default()));
        let mut seq = super::super::Sequence::new(1, String::new(), Vec::new());
        let mut output = String::new();

        for index in 0..128 {
            output.push_str(&decode_token(&tokenizer, &mut seq, 0, index == 127));
        }

        assert_eq!(output, tokenizer.decode(&vec![0; 128], true).unwrap());
        assert!(seq.decode_stream.buffered_token_count() <= 2);
    }

    #[test]
    fn incremental_decode_flushes_partial_character_when_finished() {
        let mut tokenizer = bpe_tokenizer(HashMap::from([("<0xC3>".to_string(), 0)]));
        tokenizer.with_decoder(Some(ByteFallback::default()));
        let mut seq = super::super::Sequence::new(1, String::new(), Vec::new());

        let output = decode_token(&tokenizer, &mut seq, 0, true);

        assert_eq!(output, "�");
        assert_eq!(output, tokenizer.decode(&[0], true).unwrap());
    }

    #[test]
    fn reset_decode_stream_does_not_repeat_session_history() {
        let mut tokenizer = bpe_tokenizer(HashMap::from([("▁This".to_string(), 0)]));
        tokenizer.with_decoder(Some(Metaspace::default()));
        let mut seq = super::super::Sequence::new(1, String::new(), Vec::new());

        assert_eq!(decode_token(&tokenizer, &mut seq, 0, true), "This");
        seq.status = super::super::SequenceStatus::Waiting;
        seq.generated_this_turn = 0;
        seq.reset_decode_stream();

        assert_eq!(decode_token(&tokenizer, &mut seq, 0, true), "This");
    }
}
