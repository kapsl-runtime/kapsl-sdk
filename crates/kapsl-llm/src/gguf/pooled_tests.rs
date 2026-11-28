//! Unit tests for the pooled GGUF backend's model-independent logic: request
//! decoding, batch planning, rerank token layout, and output post-processing.

use super::*;
use kapsl_engine_api::NamedTensor;
use std::path::PathBuf;

fn text(value: &str) -> BinaryTensorPacket {
    BinaryTensorPacket::new(
        vec![value.len().max(1) as i64],
        TensorDtype::Utf8,
        if value.is_empty() {
            vec![b' ']
        } else {
            value.as_bytes().to_vec()
        },
    )
    .unwrap()
}

fn named(name: &str, tensor: BinaryTensorPacket) -> NamedTensor {
    NamedTensor {
        name: name.to_string(),
        tensor,
    }
}

fn request(input: BinaryTensorPacket, additional: Vec<NamedTensor>) -> InferenceRequest {
    let mut request = InferenceRequest::new(input);
    request.additional_inputs = additional;
    request
}

fn token(id: i32) -> LlamaToken {
    LlamaToken(id)
}

fn values(packet: &BinaryTensorPacket) -> Vec<f32> {
    packet
        .data
        .as_chunks::<4>()
        .0
        .iter()
        .map(|bytes| f32::from_le_bytes(*bytes))
        .collect()
}

#[test]
fn embed_request_takes_primary_and_indexed_text_inputs_in_order() {
    let items = request_items(
        GgufPooledTask::Embed,
        &request(
            text("first"),
            vec![
                named("text.1", text("second")),
                named("text", text("third")),
            ],
        ),
    )
    .unwrap();
    assert_eq!(
        items,
        vec![
            PooledItem::Text("first".into()),
            PooledItem::Text("second".into()),
            PooledItem::Text("third".into()),
        ]
    );
}

#[test]
fn embed_request_accepts_token_ids() {
    let ids: Vec<u8> = [101i64, 2023, 102]
        .iter()
        .flat_map(|id| id.to_le_bytes())
        .collect();
    let packet = BinaryTensorPacket::new(vec![1, 3], TensorDtype::Int64, ids).unwrap();
    let items = request_items(GgufPooledTask::Embed, &request(packet, Vec::new())).unwrap();
    assert_eq!(items, vec![PooledItem::Tokens(vec![101, 2023, 102])]);
}

#[test]
fn embed_request_rejects_foreign_inputs_and_bad_shapes() {
    let error = request_items(
        GgufPooledTask::Classify,
        &request(text("a"), vec![named("attention_mask", text("b"))]),
    )
    .unwrap_err();
    assert!(error.to_string().contains("attention_mask"), "{error}");

    let ids: Vec<u8> = [1i32, 2, 3, 4]
        .iter()
        .flat_map(|id| id.to_le_bytes())
        .collect();
    let matrix = BinaryTensorPacket::new(vec![2, 2], TensorDtype::Int32, ids).unwrap();
    let error = request_items(GgufPooledTask::Embed, &request(matrix, Vec::new())).unwrap_err();
    assert!(error.to_string().contains("[len]"), "{error}");

    let floats = BinaryTensorPacket::new(vec![1], TensorDtype::Float32, vec![0; 4]).unwrap();
    assert!(request_items(GgufPooledTask::Embed, &request(floats, Vec::new())).is_err());

    let invalid_utf8 = BinaryTensorPacket::new(vec![1], TensorDtype::Uint8, vec![0xff]).unwrap();
    let error =
        request_items(GgufPooledTask::Embed, &request(invalid_utf8, Vec::new())).unwrap_err();
    assert!(error.to_string().contains("UTF-8"), "{error}");
}

#[test]
fn input_names_match_base_or_numeric_index_only() {
    assert!(input_matches("text", "text"));
    assert!(input_matches("text.0", "text"));
    assert!(input_matches("document.12", "document"));
    assert!(!input_matches("text.", "text"));
    assert!(!input_matches("text.a", "text"));
    assert!(!input_matches("texts", "text"));
    assert!(!input_matches("document", "text"));
}

#[test]
fn rerank_request_pairs_the_query_with_every_document() {
    let items = request_items(
        GgufPooledTask::Rerank,
        &request(
            text("what is a panda?"),
            vec![
                named("document.0", text("hi")),
                named("document.1", text("The giant panda is a bear.")),
            ],
        ),
    )
    .unwrap();
    assert_eq!(
        items,
        vec![
            PooledItem::Pair {
                query: "what is a panda?".into(),
                document: "hi".into()
            },
            PooledItem::Pair {
                query: "what is a panda?".into(),
                document: "The giant panda is a bear.".into()
            },
        ]
    );
}

#[test]
fn rerank_request_needs_documents_and_text() {
    let error = request_items(GgufPooledTask::Rerank, &request(text("q"), Vec::new())).unwrap_err();
    assert!(error.to_string().contains("document"), "{error}");

    let ids = BinaryTensorPacket::new(vec![1], TensorDtype::Int32, vec![0; 4]).unwrap();
    assert!(request_items(
        GgufPooledTask::Rerank,
        &request(ids, vec![named("document", text("d"))])
    )
    .is_err());
}

#[test]
fn request_input_count_is_bounded() {
    let additional = (0..MAX_INPUTS_PER_REQUEST)
        .map(|index| named(&format!("text.{index}"), text("x")))
        .collect();
    let error = request_items(GgufPooledTask::Embed, &request(text("x"), additional)).unwrap_err();
    assert!(error.to_string().contains("at most"), "{error}");
}

#[test]
fn batches_respect_token_and_sequence_limits_in_order() {
    assert_eq!(plan_batches(&[], 8, 4), Vec::<Range<usize>>::new());
    assert_eq!(plan_batches(&[3, 3, 3], 8, 4), vec![0..2, 2..3]);
    assert_eq!(plan_batches(&[8, 1, 1], 8, 4), vec![0..1, 1..3]);
    assert_eq!(plan_batches(&[1, 1, 1, 1, 1], 8, 2), vec![0..2, 2..4, 4..5]);
    // A full-length sequence always gets a batch of its own.
    assert_eq!(plan_batches(&[2, 8, 2], 8, 4), vec![0..1, 1..2, 2..3]);
}

#[test]
fn rerank_layout_follows_the_vocabulary_flags() {
    let query = [token(10), token(11)];
    let document = [token(20)];
    let bert = RerankSpecials {
        bos: Some(token(101)),
        eos: Some(token(102)),
        sep: None,
    };
    assert_eq!(
        assemble_rerank_tokens(&query, &document, bert),
        vec![
            token(101),
            token(10),
            token(11),
            token(102),
            token(20),
            token(102)
        ]
    );
    let roberta = RerankSpecials {
        bos: Some(token(0)),
        eos: Some(token(2)),
        sep: Some(token(2)),
    };
    assert_eq!(
        assemble_rerank_tokens(&query, &document, roberta),
        vec![
            token(0),
            token(10),
            token(11),
            token(2),
            token(2),
            token(20),
            token(2)
        ]
    );
    assert_eq!(
        assemble_rerank_tokens(&query, &document, RerankSpecials::default()),
        vec![token(10), token(11), token(20)]
    );
}

#[test]
fn rerank_template_substitutes_each_placeholder_once() {
    assert_eq!(
        fill_rerank_template("<q>{query}</q><d>{document}</d>", "Q", "D"),
        "<q>Q</q><d>D</d>"
    );
    // Placeholder text inside the query is user data, not a placeholder.
    assert_eq!(
        fill_rerank_template("{query}|{document}", "{document}", "real"),
        "{document}|real"
    );
    assert_eq!(
        fill_rerank_template("{document} then {query}", "q", "d"),
        "d then q"
    );
    assert_eq!(
        fill_rerank_template("no placeholders", "q", "d"),
        "no placeholders"
    );
}

#[test]
fn embeddings_are_normalized_unless_disabled() {
    let config = GgufPooledConfig::new(GgufPooledTask::Embed);
    let packet = finish_outputs(&config, vec![vec![3.0, 4.0], vec![0.0, 2.0]], 2).unwrap();
    assert_eq!(packet.shape, vec![2, 2]);
    assert_eq!(values(&packet), vec![0.6, 0.8, 0.0, 1.0]);

    let raw = GgufPooledConfig {
        normalize: false,
        ..GgufPooledConfig::new(GgufPooledTask::Embed)
    };
    let packet = finish_outputs(&raw, vec![vec![3.0, 4.0]], 2).unwrap();
    assert_eq!(values(&packet), vec![3.0, 4.0]);
}

#[test]
fn classifier_rows_become_probabilities() {
    let config = GgufPooledConfig::new(GgufPooledTask::Classify);
    let packet = finish_outputs(&config, vec![vec![0.0, 0.0], vec![10.0, -10.0]], 2).unwrap();
    assert_eq!(packet.shape, vec![2, 2]);
    let probabilities = values(&packet);
    assert!((probabilities[0] - 0.5).abs() < 1e-6);
    assert!((probabilities[1] - 0.5).abs() < 1e-6);
    assert!(probabilities[2] > 0.999);
    assert!((probabilities[2] + probabilities[3] - 1.0).abs() < 1e-6);
}

#[test]
fn rerank_scores_are_the_first_head_output() {
    let config = GgufPooledConfig::new(GgufPooledTask::Rerank);
    let packet = finish_outputs(&config, vec![vec![-2.5], vec![7.25]], 1).unwrap();
    assert_eq!(packet.shape, vec![2]);
    assert_eq!(values(&packet), vec![-2.5, 7.25]);
}

#[test]
fn malformed_backend_rows_are_rejected() {
    let config = GgufPooledConfig::new(GgufPooledTask::Embed);
    assert!(finish_outputs(&config, vec![vec![1.0]], 2).is_err());
    assert!(finish_outputs(&config, vec![vec![f32::NAN, 1.0]], 2).is_err());
    assert!(finish_outputs(&config, vec![vec![]], 0).is_err());
}

#[test]
fn config_reads_task_and_metadata_from_the_manifest() {
    let config = GgufPooledConfig::from_manifest_json(&serde_json::json!({
        "task": "embed",
        "metadata": {"embed": {"normalize": false, "pooling": "LAST"}}
    }))
    .unwrap();
    assert_eq!(config.task, GgufPooledTask::Embed);
    assert!(!config.normalize);
    assert_eq!(config.pooling, Some(GgufPooling::Last));

    let config = GgufPooledConfig::from_manifest_json(&serde_json::json!({
        "model_type": "seq-classifier",
        "metadata": {"classify": {"apply_softmax": false}}
    }))
    .unwrap();
    assert_eq!(config.task, GgufPooledTask::Classify);
    assert!(!config.apply_softmax);

    let config =
        GgufPooledConfig::from_manifest_json(&serde_json::json!({"task": "rerank"})).unwrap();
    assert_eq!(config, GgufPooledConfig::new(GgufPooledTask::Rerank));
}

#[test]
fn config_rejects_generation_and_malformed_metadata() {
    assert!(
        GgufPooledConfig::from_manifest_json(&serde_json::json!({"task": "generate"})).is_err()
    );
    assert!(GgufPooledConfig::from_manifest_json(&serde_json::json!({})).is_err());
    assert!(GgufPooledConfig::from_manifest_json(&serde_json::json!({
        "task": "embed", "metadata": {"embed": {"pooling": "max"}}
    }))
    .is_err());
    assert!(GgufPooledConfig::from_manifest_json(&serde_json::json!({
        "task": "embed", "metadata": {"embed": {"normalize": "yes"}}
    }))
    .is_err());
}

#[test]
fn config_from_a_parsed_manifest_uses_the_effective_task() {
    let manifest = kapsl_core::Manifest {
        project_name: "m".into(),
        framework: "gguf".into(),
        version: "1.0.0".into(),
        created_at: "0".into(),
        model_file: "m.gguf".into(),
        format: Some("gguf".into()),
        model_type: Some("embedding".into()),
        task: None,
        metadata: None,
        hardware_requirements: Default::default(),
        cron_jobs: Vec::new(),
    };
    let config = GgufPooledConfig::from_manifest(&manifest).unwrap();
    assert_eq!(config.task, GgufPooledTask::Embed);
}

#[test]
fn pooling_must_fit_the_task() {
    for pooling in [
        LlamaPoolingType::Mean,
        LlamaPoolingType::Cls,
        LlamaPoolingType::Last,
    ] {
        assert!(check_pooling(GgufPooledTask::Embed, pooling).is_ok());
        assert!(check_pooling(GgufPooledTask::Rerank, pooling).is_err());
    }
    let error = check_pooling(GgufPooledTask::Embed, LlamaPoolingType::None).unwrap_err();
    assert!(error.contains("metadata.embed.pooling"), "{error}");
    assert!(check_pooling(GgufPooledTask::Embed, LlamaPoolingType::Rank).is_err());
    assert!(check_pooling(GgufPooledTask::Rerank, LlamaPoolingType::Rank).is_ok());
    assert!(check_pooling(GgufPooledTask::Classify, LlamaPoolingType::Rank).is_ok());
}

#[test]
fn compute_estimate_has_a_floor_and_grows_with_context() {
    assert_eq!(
        estimate_pooled_compute_bytes(384, 1536, 12, 0, 8, 8),
        MIN_POOLED_SCRATCH_BYTES
    );
    let short = estimate_pooled_compute_bytes(1024, 4096, 16, 0, 512, 512);
    let long = estimate_pooled_compute_bytes(1024, 4096, 16, 0, 2048, 2048);
    assert!(long > short);
    // 16 heads x 2048 x 2048 f32 scores alone are 256 MiB.
    assert!(long > 256 * 1024 * 1024);
}

enum Meta {
    U32(u32),
    Bool(bool),
    Str(&'static str),
}

/// Writes `dir/name.gguf`: a GGUF v3 file with `entries` as metadata and no
/// tensors.
fn metadata_only_gguf(dir: &Path, name: &str, entries: &[(&str, Meta)]) -> PathBuf {
    fn put_str(out: &mut Vec<u8>, value: &str) {
        out.extend_from_slice(&(value.len() as u64).to_le_bytes());
        out.extend_from_slice(value.as_bytes());
    }
    let mut out = b"GGUF".to_vec();
    out.extend_from_slice(&3u32.to_le_bytes());
    out.extend_from_slice(&0u64.to_le_bytes());
    out.extend_from_slice(&(entries.len() as u64).to_le_bytes());
    for (key, value) in entries {
        put_str(&mut out, key);
        match value {
            Meta::U32(v) => {
                out.extend_from_slice(&4u32.to_le_bytes());
                out.extend_from_slice(&v.to_le_bytes());
            }
            Meta::Bool(v) => {
                out.extend_from_slice(&7u32.to_le_bytes());
                out.push(u8::from(*v));
            }
            Meta::Str(v) => {
                out.extend_from_slice(&8u32.to_le_bytes());
                put_str(&mut out, v);
            }
        }
    }
    let path = dir.join(format!("{name}.gguf"));
    std::fs::write(&path, out).unwrap();
    path
}

/// Plans against the headers of Qwen3-Embedding-0.6B (a causal decoder
/// embedder) and bge-small-en-v1.5 (a BERT encoder). The Qwen3 bounds are what
/// llama.cpp allocated for it at the default 2048-token context: 234,881,024
/// KV bytes and 1,267,695,648 compute bytes.
#[test]
fn plan_covers_decoder_heads_and_declared_head_widths() {
    if std::env::var_os("KAPSL_GGUF_CTX_PER_SEQ").is_some() {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let qwen3 = metadata_only_gguf(
        dir.path(),
        "qwen3",
        &[
            ("general.architecture", Meta::Str("qwen3")),
            ("qwen3.block_count", Meta::U32(28)),
            ("qwen3.context_length", Meta::U32(32768)),
            ("qwen3.embedding_length", Meta::U32(1024)),
            ("qwen3.feed_forward_length", Meta::U32(3072)),
            ("qwen3.attention.head_count", Meta::U32(16)),
            ("qwen3.attention.head_count_kv", Meta::U32(8)),
            ("qwen3.attention.key_length", Meta::U32(128)),
            ("qwen3.attention.value_length", Meta::U32(128)),
            ("qwen3.vocab_size", Meta::U32(151_669)),
        ],
    );
    let (kv, scratch) = planned_pooled_bytes(&qwen3, 639_150_592);
    assert_eq!(kv, 234_881_024);
    assert!(scratch >= 1_267_695_648, "{scratch}");

    let bert = metadata_only_gguf(
        dir.path(),
        "bert",
        &[
            ("general.architecture", Meta::Str("bert")),
            ("bert.block_count", Meta::U32(12)),
            ("bert.context_length", Meta::U32(512)),
            ("bert.embedding_length", Meta::U32(384)),
            ("bert.feed_forward_length", Meta::U32(1536)),
            ("bert.attention.head_count", Meta::U32(12)),
            ("bert.attention.causal", Meta::Bool(false)),
            ("bert.vocab_size", Meta::U32(30_522)),
        ],
    );
    let (kv, scratch) = planned_pooled_bytes(&bert, 36_806_944);
    assert_eq!(kv, 12 * 12 * (32 + 32) * 2 * 512);
    // An encoder has no language-model head, so its estimate stays at the floor.
    assert_eq!(scratch, MIN_POOLED_SCRATCH_BYTES);
}

#[test]
fn task_names_round_trip() {
    for task in [
        GgufPooledTask::Embed,
        GgufPooledTask::Rerank,
        GgufPooledTask::Classify,
    ] {
        assert_eq!(GgufPooledTask::from_task(task.as_str()), Some(task));
    }
    assert_eq!(
        GgufPooledTask::from_task(" Embed "),
        Some(GgufPooledTask::Embed)
    );
    assert_eq!(GgufPooledTask::from_task("generate"), None);
}
