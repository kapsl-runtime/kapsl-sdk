#[cfg(test)]
mod tests {
    use super::super::{
        build_kv_array_f16, build_kv_array_f32_from_f16, empty_kv_shape, infer_kv_layout,
        normalize_metadata_safe_load_setting, parse_safe_load_env_setting, parse_safe_load_setting,
        KvLayout, LLMEngine, LLMMetrics, SafeLoadSetting, SamplingParams, SchedulerConfig,
    };
    use half::f16;
    use serde_json::json;
    use std::collections::HashMap;
    use std::sync::{Arc, Mutex};
    use std::time::Duration;
    use tokio::sync::mpsc;

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
}
