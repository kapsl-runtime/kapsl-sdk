//! Opt-in checks of the pooled GGUF backend against real models.
//!
//!   KAPSL_TEST_GGUF_EMBED_MODEL=/path/bge-small-en-v1.5-q8_0.gguf \
//!   KAPSL_TEST_GGUF_RERANK_MODEL=/path/bge-reranker-v2-m3-q8_0.gguf \
//!     cargo test -p kapsl-llm --features gguf --test gguf_pooled_models -- --ignored --nocapture
//!
//! Each test skips (passes) when its model variable is unset. The embedding
//! test checks the property most sensitive to batching bugs: a text embedded
//! alone and the same text packed into a multi-sequence batch must give the
//! same vector.
//!
//! Set `KAPSL_TEST_GGUF_EMBED_REFERENCE` to the JSON that llama.cpp's own
//! `llama-embedding --embd-normalize 2 --embd-output-format json` prints for
//! the four test texts (newline-separated, in order) to also require parity
//! with llama.cpp's reference embedding tool.
#![cfg(feature = "gguf")]

use kapsl_engine_api::{
    BinaryTensorPacket, Engine, InferenceRequest, MemoryAllocationClass, NamedTensor, TensorDtype,
};
use kapsl_llm::{GgufPooledBackend, GgufPooledConfig, GgufPooledTask};
use std::path::PathBuf;

fn model(var: &str) -> Option<PathBuf> {
    let path = std::env::var_os(var).map(PathBuf::from);
    if path.is_none() {
        eprintln!("skipping: {var} is not set");
    }
    path
}

fn text(value: &str) -> BinaryTensorPacket {
    BinaryTensorPacket::new(
        vec![value.len() as i64],
        TensorDtype::Utf8,
        value.as_bytes().to_vec(),
    )
    .unwrap()
}

fn request(first: &str, rest: &[&str], name: &str) -> InferenceRequest {
    let mut request = InferenceRequest::new(text(first));
    request.additional_inputs = rest
        .iter()
        .enumerate()
        .map(|(index, value)| NamedTensor {
            name: format!("{name}.{index}"),
            tensor: text(value),
        })
        .collect();
    request
}

fn f32_values(packet: &BinaryTensorPacket) -> Vec<f32> {
    packet
        .data
        .as_chunks::<4>()
        .0
        .iter()
        .map(|bytes| f32::from_le_bytes(*bytes))
        .collect()
}

fn rows(packet: &BinaryTensorPacket) -> Vec<Vec<f32>> {
    let values = f32_values(packet);
    let width = *packet.shape.last().unwrap() as usize;
    values.chunks(width).map(<[f32]>::to_vec).collect()
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

#[tokio::test]
#[ignore = "needs KAPSL_TEST_GGUF_EMBED_MODEL"]
async fn embeddings_are_batch_invariant_and_semantic() {
    let Some(path) = model("KAPSL_TEST_GGUF_EMBED_MODEL") else {
        return;
    };
    let mut backend = GgufPooledBackend::new(0, GgufPooledConfig::new(GgufPooledTask::Embed));
    let planned = backend.planned_memory(&path).expect("planned memory");
    backend.load(&path).await.expect("load");
    let info = backend.model_info().expect("model info");
    assert_eq!(info.output_names, vec!["embeddings"]);

    let texts = [
        "The cat sat on the mat.",
        "A kitten is resting on a rug.",
        "Quarterly revenue grew by twelve percent.",
        "short",
    ];
    let batched = rows(
        &backend
            .infer(&request(texts[0], &texts[1..], "text"))
            .expect("batched embed"),
    );
    assert_eq!(batched.len(), texts.len());
    for (index, value) in texts.iter().enumerate() {
        let alone = rows(&backend.infer(&request(value, &[], "text")).expect("embed"));
        let norm = dot(&alone[0], &alone[0]).sqrt();
        assert!((norm - 1.0).abs() < 1e-4, "text {index} norm {norm}");
        let max_diff = alone[0]
            .iter()
            .zip(&batched[index])
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        println!("text {index}: max |alone - batched| = {max_diff:e}");
        assert!(
            max_diff < 1e-3,
            "text {index} differs by {max_diff} when batched"
        );
    }
    let similar = dot(&batched[0], &batched[1]);
    let unrelated = dot(&batched[0], &batched[2]);
    println!("cos(similar)={similar:.4} cos(unrelated)={unrelated:.4}");
    assert!(similar > unrelated);

    if let Some(reference) = std::env::var_os("KAPSL_TEST_GGUF_EMBED_REFERENCE") {
        let reference: serde_json::Value =
            serde_json::from_slice(&std::fs::read(reference).expect("read reference JSON"))
                .expect("parse reference JSON");
        let data = reference["data"].as_array().expect("reference data array");
        assert_eq!(
            data.len(),
            texts.len(),
            "reference must embed the test texts"
        );
        for (index, item) in data.iter().enumerate() {
            let expected: Vec<f32> = item["embedding"]
                .as_array()
                .expect("reference embedding")
                .iter()
                .map(|value| value.as_f64().expect("number") as f32)
                .collect();
            assert_eq!(expected.len(), batched[index].len(), "text {index} width");
            let cosine = dot(&expected, &batched[index])
                / (dot(&expected, &expected).sqrt() * dot(&batched[index], &batched[index]).sqrt());
            let max_diff = expected
                .iter()
                .zip(&batched[index])
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            println!("text {index}: cos(reference)={cosine:.7} max |diff|={max_diff:e}");
            assert!(cosine > 0.9999, "text {index} cosine {cosine} vs llama.cpp");
            assert!(
                max_diff < 1e-4,
                "text {index} differs from llama.cpp by {max_diff}"
            );
        }
    }

    let actual = backend.actual_memory();
    for (label, report) in [("planned", &planned), ("actual", &actual)] {
        for allocation in &report.allocations {
            println!(
                "{label} {:?} {:?} {} bytes",
                allocation.class, allocation.domain, allocation.bytes
            );
        }
    }
    assert!(actual
        .allocations
        .iter()
        .any(|a| a.class == MemoryAllocationClass::PersistentWeights && a.bytes > 0));
    backend.unload();
    assert!(backend.health_check().is_err());
}

#[tokio::test]
#[ignore = "needs KAPSL_TEST_GGUF_RERANK_MODEL"]
async fn reranker_prefers_the_relevant_document() {
    let Some(path) = model("KAPSL_TEST_GGUF_RERANK_MODEL") else {
        return;
    };
    let mut backend = GgufPooledBackend::new(0, GgufPooledConfig::new(GgufPooledTask::Rerank));
    backend.load(&path).await.expect("load");
    let documents = [
        "hi",
        "The giant panda is a bear species endemic to China.",
        "it is a bear",
    ];
    let scores = backend
        .infer(&request("what is panda?", &documents, "document"))
        .expect("rerank");
    assert_eq!(scores.shape, vec![documents.len() as i64]);
    let scores = f32_values(&scores);
    println!("rerank scores: {scores:?}");
    assert!(
        scores[1] > scores[0],
        "relevant document must outscore `hi`"
    );
}

#[tokio::test]
#[ignore = "needs KAPSL_TEST_GGUF_EMBED_MODEL"]
async fn embedding_model_cannot_be_loaded_as_a_reranker() {
    let Some(path) = model("KAPSL_TEST_GGUF_EMBED_MODEL") else {
        return;
    };
    let mut backend = GgufPooledBackend::new(0, GgufPooledConfig::new(GgufPooledTask::Rerank));
    let error = backend
        .load(&path)
        .await
        .expect_err("no classification head");
    assert!(error.to_string().contains("classification head"), "{error}");
}
