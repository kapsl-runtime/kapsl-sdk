#[cfg(test)]
mod tests {
    use super::super::{
        default_sampling_params, extract_bos_token, extract_tag, load_model_runtime_config,
        LLMBackend, ModelRuntimeConfig,
    };
    use crate::prompt_adapter::ChatPromptTemplate;
    use crate::sequence::{FinishReason, SequenceGroupOutput};
    use futures::StreamExt;
    use kapsl_engine_api::{
        BinaryTensorPacket, Engine, InferenceRequest, MemoryAllocationClass, MemoryDomain,
        TensorDtype,
    };
    use serde_json::json;
    use std::fs;
    use std::path::PathBuf;
    use std::sync::atomic::Ordering;
    use tokio::sync::mpsc;

    fn make_temp_dir(label: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "kapsl_llm_llm_backend_{}_{}",
            label,
            uuid::Uuid::new_v4()
        ));
        fs::create_dir_all(&dir).expect("create temp dir");
        dir
    }

    #[test]
    fn extract_bos_token_from_tokenizer_json() {
        let tokenizer = json!({
            "post_processor": {
                "single": [
                    { "SpecialToken": { "id": "<s>" } }
                ]
            }
        });
        assert_eq!(extract_bos_token(&tokenizer), Some("<s>".to_string()));

        let tokenizer = json!({
            "post_processor": {
                "special_tokens": {
                    "<bos>": {}
                }
            }
        });
        assert_eq!(extract_bos_token(&tokenizer), Some("<bos>".to_string()));
    }

    #[test]
    fn extract_tag_finds_nearest_angle_brackets() {
        let template = "<|user|> User: {{prompt}} <|assistant|>";
        assert_eq!(extract_tag(template, "User"), Some("<|user|>".to_string()));
    }

    #[test]
    fn load_model_runtime_config_reads_generation_and_template_defaults() {
        let root = make_temp_dir("cfg");
        let model_path = root.join("model.onnx");
        fs::write(&model_path, "").expect("model file");

        let generation = json!({
            "temperature": 0.5,
            "max_new_tokens": 128,
            "top_p": 0.8,
            "top_k": 20,
            "repetition_penalty": 1.2,
            "eos_token_id": 2,
            "bos_token_id": 1
        });
        fs::write(root.join("generation_config.json"), generation.to_string())
            .expect("generation config");

        let cfg = json!({ "model_type": "qwen2" });
        fs::write(root.join("config.json"), cfg.to_string()).expect("config.json");

        let tokenizer = json!({
            "post_processor": {
                "special_tokens": {
                    "<bos>": {}
                }
            }
        });
        fs::write(root.join("tokenizer.json"), tokenizer.to_string()).expect("tokenizer.json");

        let template = "<|user|> User: {{prompt}} <|assistant|>";
        fs::write(root.join("chat_template.jinja"), template).expect("chat template");

        let runtime = load_model_runtime_config(&model_path);
        assert_eq!(
            runtime
                .prompt_template
                .as_ref()
                .map(|template| template.render("Hi")),
            Some("<bos><|user|>Hi<|assistant|>".to_string())
        );

        let sampling = runtime.sampling;
        assert_eq!(sampling.max_tokens, 128);
        assert!((sampling.temperature - 0.5).abs() < f32::EPSILON);
        assert!((sampling.top_p - 0.8).abs() < f32::EPSILON);
        assert_eq!(sampling.top_k, 20);
        assert!((sampling.repetition_penalty - 1.2).abs() < f32::EPSILON);
        assert_eq!(sampling.stop_token_ids, vec![2, 1]);
    }

    #[test]
    fn load_model_runtime_config_accepts_array_eos_token_id() {
        let root = make_temp_dir("cfg_eos_array");
        let model_path = root.join("model.onnx");
        fs::write(&model_path, "").expect("model file");

        let generation = json!({
            "eos_token_id": [1, 106],
            "bos_token_id": 2
        });
        fs::write(root.join("generation_config.json"), generation.to_string())
            .expect("generation config");

        let runtime = load_model_runtime_config(&model_path);
        assert_eq!(runtime.sampling.stop_token_ids, vec![1, 106, 2]);
    }

    #[test]
    fn actual_memory_tracks_device_kv_fallback_domain() {
        let backend = LLMBackend::with_device("cuda".to_string(), 2).with_env_allocators(true);
        backend.loaded_model_bytes.store(1024, Ordering::Release);

        let kv_allocation = |report: kapsl_engine_api::MemoryReport| {
            report
                .allocations
                .into_iter()
                .find(|allocation| allocation.class == MemoryAllocationClass::KvCache)
                .expect("KV allocation")
        };
        assert_eq!(
            kv_allocation(backend.actual_memory()).domain,
            MemoryDomain::Host
        );

        backend.metrics.lock().unwrap().kv_cache_device_resident = true;
        assert_eq!(
            kv_allocation(backend.actual_memory()).domain,
            MemoryDomain::Cuda { device_id: 2 }
        );
    }

    #[test]
    fn load_model_runtime_config_suppresses_added_special_tokens() {
        let root = make_temp_dir("cfg_added_special_tokens");
        let model_path = root.join("model.onnx");
        fs::write(&model_path, "").expect("model file");
        fs::write(
            root.join("generation_config.json"),
            json!({ "eos_token_id": 10 }).to_string(),
        )
        .expect("generation config");
        fs::write(
            root.join("tokenizer.json"),
            json!({
                "added_tokens": [
                    { "id": 10, "content": "<eos>", "special": true },
                    { "id": 11, "content": "<role>", "special": true },
                    { "id": 12, "content": "ordinary", "special": false }
                ]
            })
            .to_string(),
        )
        .expect("tokenizer");

        let runtime = load_model_runtime_config(&model_path);
        assert_eq!(runtime.sampling.stop_token_ids, vec![10, 11]);
    }

    #[test]
    fn load_model_runtime_config_reads_manifest_chat_template_for_onnx() {
        let root = make_temp_dir("manifest_template");
        let model_path = root.join("model.onnx");
        fs::write(&model_path, "").expect("model file");

        let manifest = json!({
            "metadata": {
                "llm": {
                    "chat_template": "gemma"
                }
            }
        });
        fs::write(root.join("metadata.json"), manifest.to_string()).expect("metadata");

        let runtime = load_model_runtime_config(&model_path);
        assert_eq!(
            runtime
                .prompt_template
                .as_ref()
                .map(|template| template.render("Hello")),
            Some("<start_of_turn>user\nHello<end_of_turn>\n<start_of_turn>model\n".to_string())
        );
    }

    #[test]
    fn load_model_runtime_config_uses_embedded_family_specific_chat_template() {
        let root = make_temp_dir("embedded_qwen_template");
        let model_path = root.join("model.onnx");
        fs::write(&model_path, "").expect("model file");
        fs::write(
            root.join("config.json"),
            json!({
                "_name_or_path": "Qwen/Qwen2.5-1.5B-Instruct",
                "model_type": "qwen2"
            })
            .to_string(),
        )
        .expect("config");
        fs::write(
            root.join("tokenizer_config.json"),
            json!({
                "add_bos_token": false,
                "chat_template": "<|im_start|>system\\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\\n<|im_start|>user\\n{{ message.content }}<|im_end|>"
            })
            .to_string(),
        )
        .expect("tokenizer config");

        let runtime = load_model_runtime_config(&model_path);
        assert_eq!(runtime.prompt_template, Some(ChatPromptTemplate::Qwen));
        assert_eq!(
            runtime
                .prompt_template
                .as_ref()
                .map(|template| template.render("what is your name?")),
            Some(
                "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n\
                 <|im_start|>user\nwhat is your name?<|im_end|>\n<|im_start|>assistant\n"
                    .to_string()
            )
        );

        fs::write(
            root.join("tokenizer_config.json"),
            json!({
                "chat_template": {
                    "default": "<|im_start|>{{ message.role }}\\n{{ message.content }}<|im_end|>"
                }
            })
            .to_string(),
        )
        .expect("generic tokenizer config");
        let runtime = load_model_runtime_config(&model_path);
        assert_eq!(runtime.prompt_template, Some(ChatPromptTemplate::ChatMl));
    }

    #[test]
    fn load_model_runtime_config_uses_manifest_identity_for_legacy_deepseek_package() {
        let root = make_temp_dir("legacy_deepseek_template");
        let model_path = root.join("model.onnx");
        fs::write(&model_path, "").expect("model file");
        fs::write(
            root.join("metadata.json"),
            json!({
                "project_name": "deepseek",
                "framework": "llm",
                "metadata": { "llm": {} }
            })
            .to_string(),
        )
        .expect("metadata");

        let runtime = load_model_runtime_config(&model_path);
        assert_eq!(
            runtime
                .prompt_template
                .as_ref()
                .map(|template| template.render("hello")),
            Some("<｜User｜>hello<｜Assistant｜><think>\n".to_string())
        );
    }

    #[tokio::test]
    async fn infer_stream_handles_cumulative_and_incremental_outputs() {
        let backend = LLMBackend::new();
        let (tx, mut rx) = mpsc::channel(1);
        *backend.request_tx.write().unwrap() = Some(tx);
        *backend.model_config.lock().unwrap() = ModelRuntimeConfig {
            prompt_template: Some(ChatPromptTemplate::Gemma),
            sampling: default_sampling_params(),
        };

        let request = InferenceRequest {
            input: BinaryTensorPacket {
                shape: vec![1, 2],
                dtype: TensorDtype::Utf8,
                data: b"Hi".to_vec(),
            },
            additional_inputs: Vec::new(),
            session_id: None,
            metadata: None,
            cancellation: None,
        };

        let stream: std::pin::Pin<
            Box<
                dyn futures::Stream<
                        Item = Result<BinaryTensorPacket, kapsl_engine_api::EngineError>,
                    > + Send,
            >,
        > = backend.infer_stream(&request);

        let handle = tokio::spawn(async move {
            let mut stream = stream;
            let mut chunks = Vec::new();
            while let Some(packet_res) = stream.next().await {
                let packet: BinaryTensorPacket = match packet_res {
                    Ok(packet) => packet,
                    Err(err) => panic!("stream err: {}", err),
                };
                chunks.push(String::from_utf8(packet.data).expect("utf8"));
            }
            chunks
        });

        let seq_group = rx.recv().await.expect("seq_group");
        *backend.request_tx.write().unwrap() = None;
        assert!(
            rx.is_closed(),
            "an active response stream must release its request sender after enqueue"
        );
        let queued_prompt = seq_group
            .sequences
            .get(&0)
            .expect("sequence")
            .lock()
            .unwrap()
            .prompt
            .clone();
        assert_eq!(
            queued_prompt,
            "<start_of_turn>user\nHi<end_of_turn>\n<start_of_turn>model\n"
        );

        seq_group
            .response_tx
            .send(SequenceGroupOutput {
                request_id: seq_group.request_id.clone(),
                text: "Hel".to_string(),
                finish_reason: None,
            })
            .await
            .expect("send first chunk");

        seq_group
            .response_tx
            .send(SequenceGroupOutput {
                request_id: seq_group.request_id.clone(),
                text: "Hello".to_string(),
                finish_reason: Some(FinishReason::Stop),
            })
            .await
            .expect("send second chunk");

        let chunks = handle.await.expect("join stream task");
        assert_eq!(chunks, vec!["Hel".to_string(), "lo".to_string()]);
    }

    #[test]
    fn unpolled_inference_stream_does_not_keep_request_channel_open() {
        let backend = LLMBackend::new();
        let (tx, rx) = mpsc::channel(1);
        *backend.request_tx.write().unwrap() = Some(tx);

        let request = InferenceRequest {
            input: BinaryTensorPacket {
                shape: vec![1, 2],
                dtype: TensorDtype::Utf8,
                data: b"Hi".to_vec(),
            },
            additional_inputs: Vec::new(),
            session_id: None,
            metadata: None,
            cancellation: None,
        };

        let stream = backend.infer_stream(&request);
        *backend.request_tx.write().unwrap() = None;

        assert!(
            rx.is_closed(),
            "an unpolled response stream must not keep the engine request channel open"
        );
        drop(stream);
    }
}
