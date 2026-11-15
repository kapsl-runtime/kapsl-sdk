//! Real-GGUF checks for the chat template and stop-token handling.
//!
//! Opt-in (`--ignored`): they load a real Qwen2.5-0.5B-Instruct GGUF on the CPU.
//!
//!   KAPSL_GGUF_N_GPU_LAYERS=0 \
//!   KAPSL_TEST_GGUF=/path/qwen2.5-0.5b-instruct-q4_k_m.gguf \
//!   [KAPSL_TEST_GGUF_EOS_SWAPPED=/path/same-file-with-eos_token_id=151643] \
//!   [KAPSL_TEST_GGUF_BAD_TEMPLATE=/path/same-file-with-unrecognised-chat-template] \
//!   cargo test -p kapsl-llm --features gguf --lib chat_stop -- --ignored --nocapture
//!
//! The two optional files are byte-for-byte copies of the first with one
//! metadata edit each (see the Stream D report for the exact commands).

use super::*;
use kapsl_engine_api::{
    BinaryTensorPacket, Engine, InferenceRequest, RequestMetadata, TensorDtype,
};
use std::path::PathBuf;

const QWEN_ENDOFTEXT: i32 = 151_643;
const QWEN_IM_END: i32 = 151_645;

fn model_path(var: &str) -> PathBuf {
    PathBuf::from(std::env::var(var).unwrap_or_else(|_| panic!("set {var} to a .gguf path")))
}

fn load_vocab_model(path: &Path) -> LlamaModel {
    let backend = global_gguf_backend().expect("llama backend");
    let params = LlamaModelParams::default().with_n_gpu_layers(0);
    LlamaModel::load_from_file(&backend, path, &params).expect("load gguf")
}

fn request(prompt: &str, max_new_tokens: u32) -> InferenceRequest {
    let mut request = InferenceRequest::new(BinaryTensorPacket {
        shape: vec![1, prompt.len() as i64],
        dtype: TensorDtype::Utf8,
        data: prompt.as_bytes().to_vec(),
    });
    request.metadata = Some(RequestMetadata {
        max_new_tokens: Some(max_new_tokens),
        ..RequestMetadata::default()
    });
    request
}

async fn reply(path: &Path, prompt: &str, max_new_tokens: u32) -> Result<String, EngineError> {
    let mut backend = GgufBackend::new();
    backend.load(path).await?;
    let packet = backend.infer(&request(prompt, max_new_tokens));
    // Release the weights before the process exits: ggml's Metal teardown aborts
    // when a model is still resident at exit.
    backend.unload();
    Ok(String::from_utf8_lossy(&packet?.data).into_owned())
}

/// Both Qwen turn terminators must end a generation whichever of them the
/// metadata names as EOS. Matching the single EOS id left the other one
/// through as literal text.
fn assert_qwen_terminators_retire(path: &Path) {
    let model = load_vocab_model(path);
    let end_tokens = GgufEndTokens::from_model(&model);
    for id in [QWEN_ENDOFTEXT, QWEN_IM_END] {
        assert!(
            end_tokens.contains(LlamaToken::new(id)),
            "token {id} must end a generation (eos={:?}, end tokens={:?})",
            model.token_eos(),
            end_tokens.ids
        );
    }
    // Ordinary text tokens never end a generation.
    let letter = model
        .str_to_token("hello", AddBos::Never)
        .expect("tokenize")[0];
    assert!(!end_tokens.contains(letter));
    // EOS itself is always included, and is the bias target for min_tokens.
    assert!(end_tokens.contains(model.token_eos()));
}

#[test]
#[ignore = "needs KAPSL_TEST_GGUF (real GGUF, CPU)"]
fn chat_stop_end_tokens_cover_both_qwen_terminators() {
    assert_qwen_terminators_retire(&model_path("KAPSL_TEST_GGUF"));
}

#[test]
#[ignore = "needs KAPSL_TEST_GGUF_EOS_SWAPPED (real GGUF, CPU)"]
fn chat_stop_end_tokens_survive_a_rewritten_eos_id() {
    let path = model_path("KAPSL_TEST_GGUF_EOS_SWAPPED");
    let model = load_vocab_model(&path);
    assert_eq!(
        model.token_eos(),
        LlamaToken::new(QWEN_ENDOFTEXT),
        "fixture must name <|endoftext|> as EOS"
    );
    assert_qwen_terminators_retire(&path);
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "needs KAPSL_TEST_GGUF (real GGUF, CPU)"]
async fn chat_stop_short_answer_ends_cleanly_before_the_cap() {
    let text = reply(&model_path("KAPSL_TEST_GGUF"), "What is 5 times 3?", 128)
        .await
        .expect("infer");
    assert!(text.contains("15"), "unexpected answer: {text:?}");
    assert!(!text.contains("<|"), "turn marker leaked: {text:?}");
    // 128 tokens would be a run-on; the answer is a sentence.
    assert!(text.len() < 200, "generation ran on: {text:?}");
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "needs KAPSL_TEST_GGUF_EOS_SWAPPED (real GGUF, CPU)"]
async fn chat_stop_short_answer_ends_cleanly_with_a_rewritten_eos_id() {
    let text = reply(
        &model_path("KAPSL_TEST_GGUF_EOS_SWAPPED"),
        "What is 5 times 3?",
        128,
    )
    .await
    .expect("infer");
    assert!(text.contains("15"), "unexpected answer: {text:?}");
    assert!(!text.contains("<|"), "turn marker leaked: {text:?}");
    assert!(text.len() < 200, "generation ran on: {text:?}");
}

/// A GGUF whose embedded template llama.cpp cannot render used to fail the
/// whole request (`chat template apply failed: ffi error -1`).
#[tokio::test(flavor = "multi_thread")]
#[ignore = "needs KAPSL_TEST_GGUF_BAD_TEMPLATE (real GGUF, CPU)"]
async fn chat_stop_unrenderable_embedded_template_falls_back_to_the_family_template() {
    let text = reply(
        &model_path("KAPSL_TEST_GGUF_BAD_TEMPLATE"),
        "What is 5 times 3?",
        128,
    )
    .await
    .expect("an unrenderable template must not fail the request");
    assert!(text.contains("15"), "unexpected answer: {text:?}");
    assert!(!text.contains("<|"), "turn marker leaked: {text:?}");
}
