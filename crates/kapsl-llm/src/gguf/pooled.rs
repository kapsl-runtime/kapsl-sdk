//! Pooled GGUF inference through llama.cpp: embeddings, cross-encoder
//! reranking, and sequence classification.
//!
//! [`GgufBackend`](super::GgufBackend) runs autoregressive generation with a
//! continuous batcher and a KV cache that lives across decode steps. Pooled
//! tasks are different: a request is one forward pass whose result is read
//! from llama.cpp's pooled-output buffer, and no state outlives the batch.
//! [`GgufPooledBackend`] keeps one llama.cpp context with embeddings enabled
//! on a worker thread, clears its memory before every batch, and packs the
//! inputs of one request into as few forward passes as the context allows.
//!
//! Request contract, identical in-process and across the native pack ABI:
//!
//! | task     | primary input              | additional inputs           | output                   |
//! |----------|----------------------------|-----------------------------|--------------------------|
//! | embed    | text or token ids          | `text` / `text.<n>`         | `float32 [n, n_embd]`    |
//! | classify | text or token ids          | `text` / `text.<n>`         | `float32 [n, n_cls_out]` |
//! | rerank   | query text                 | `document` / `document.<n>` | `float32 [n_documents]`  |
//!
//! Text is one UTF-8 string per tensor (`uint8` or `string` dtype, shape
//! `[byte_len]`). Token ids are `int32`/`int64` `[len]` or `[1, len]` in the
//! model's own vocabulary. Inputs are processed in the order given.
//!
//! Requests are not coalesced across callers: a batch failure would otherwise
//! fail every coalesced request, so one oversized input could fail another
//! user's call. Batching happens within a request instead.

use super::{
    actual_gguf_placement, allocations_from_placement, gguf_allocation_id, gguf_kv_allocation_id,
    gguf_scratch_allocation_id, global_gguf_backend, n_ctx_per_seq, planned_gguf_placement,
    shared_gguf_weights, ContextMemory, GgufWeights, GGUF_INSTANCE_COUNTER,
};
use async_trait::async_trait;
use kapsl_engine_api::{
    BatchingPolicy, BinaryTensorPacket, CancellationToken, Engine, EngineError, EngineMetrics,
    EngineModelInfo, EngineStream, ExternalDeviceMemory, ExternalDeviceMemoryReport,
    InferenceRequest, MemoryDomain, MemoryReport, TensorDtype,
};
use llama_cpp_2::context::params::{LlamaContextParams, LlamaPoolingType};
use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::LlamaModel;
use llama_cpp_2::token::LlamaToken;
use llama_cpp_sys_2::LLAMA_FLASH_ATTN_TYPE_AUTO;
use std::num::NonZeroU32;
use std::ops::Range;
use std::path::Path;
use std::sync::mpsc as std_mpsc;
use std::sync::{Arc, Mutex};
use std::time::Instant;

/// Maximum sequences packed into one llama.cpp batch.
///
/// The default of one sequence per llama.cpp call makes every result
/// bit-identical to the same input sent alone. With several sequences per
/// batch, llama.cpp's CPU kernels choose flash-attention tiling and matmul
/// tiles by batch shape, and quantized activations amplify the resulting
/// rounding changes: a packed input can drift by about 3e-3 (cosine about
/// 0.9998). llama-embedding does the same. Packing is also slower for encoder
/// models on CPU, because attention runs over every masked cross-sequence pair
/// in the batch. Measured with bge-small Q8_0 and 64 30-token passages, the
/// compute time was 575 ms with one sequence per call and 860 ms with 16.
const POOLED_MAX_SEQS_ENV: &str = "KAPSL_GGUF_POOLED_MAX_SEQS";
const POOLED_MAX_SEQS_DEFAULT: usize = 1;
/// llama.cpp's `LLAMA_MAX_SEQ`.
const LLAMA_MAX_SEQ: usize = 256;
/// Upper bound on inputs in one request, so a single call cannot queue an
/// unbounded amount of work behind the worker.
const MAX_INPUTS_PER_REQUEST: usize = 2048;
const MIN_POOLED_SCRATCH_BYTES: usize = 64 * 1024 * 1024;
/// GGUF tensor names of a classification head (`LLM_TENSOR_CLS`,
/// `LLM_TENSOR_CLS_OUT`). Without one, llama.cpp rank pooling silently
/// returns the first hidden-state value as the "score".
const CLASSIFIER_HEAD_TENSORS: [&str; 2] = ["cls.weight", "cls.output.weight"];

// ─── Configuration ────────────────────────────────────────────────────────────

/// The pooled task a [`GgufPooledBackend`] serves.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GgufPooledTask {
    /// One embedding vector per input text.
    Embed,
    /// One relevance score per (query, document) pair.
    Rerank,
    /// One class distribution per input text.
    Classify,
}

impl GgufPooledTask {
    /// Parse a manifest `task` value (`embed`, `rerank`, `classify`).
    pub fn from_task(task: &str) -> Option<Self> {
        match task.trim().to_ascii_lowercase().as_str() {
            "embed" => Some(Self::Embed),
            "rerank" => Some(Self::Rerank),
            "classify" => Some(Self::Classify),
            _ => None,
        }
    }

    /// The manifest `task` value.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Embed => "embed",
            Self::Rerank => "rerank",
            Self::Classify => "classify",
        }
    }
}

/// Embedding pooling chosen over the model's own default.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GgufPooling {
    /// Average of all token states.
    Mean,
    /// The first (CLS) token state.
    Cls,
    /// The last token state (decoder embedders such as Qwen3-Embedding).
    Last,
}

impl GgufPooling {
    /// Parse `metadata.embed.pooling` (`mean`, `cls`, `last`).
    pub fn parse(value: &str) -> Result<Self, String> {
        match value.trim().to_ascii_lowercase().as_str() {
            "mean" => Ok(Self::Mean),
            "cls" => Ok(Self::Cls),
            "last" => Ok(Self::Last),
            other => Err(format!(
                "metadata.embed.pooling `{other}` is not supported; expected mean, cls, or last"
            )),
        }
    }

    fn llama(self) -> LlamaPoolingType {
        match self {
            Self::Mean => LlamaPoolingType::Mean,
            Self::Cls => LlamaPoolingType::Cls,
            Self::Last => LlamaPoolingType::Last,
        }
    }
}

/// Task and post-processing options for a [`GgufPooledBackend`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GgufPooledConfig {
    pub task: GgufPooledTask,
    /// L2-normalize embeddings so cosine similarity equals the dot product
    /// (`metadata.embed.normalize`, default true). Embed only.
    pub normalize: bool,
    /// Apply a softmax to the classifier outputs
    /// (`metadata.classify.apply_softmax`, default true). Classify only.
    pub apply_softmax: bool,
    /// Override the model's pooling (`metadata.embed.pooling`). Embed only.
    pub pooling: Option<GgufPooling>,
}

impl GgufPooledConfig {
    /// Defaults for `task`: normalized embeddings, softmax probabilities, and
    /// the model's own pooling.
    pub fn new(task: GgufPooledTask) -> Self {
        Self {
            task,
            normalize: true,
            apply_softmax: true,
            pooling: None,
        }
    }

    /// Read the config from a model manifest serialized as JSON: `task` (or
    /// `model_type` when `task` is absent), `metadata.embed.normalize`,
    /// `metadata.embed.pooling` and `metadata.classify.apply_softmax`.
    pub fn from_manifest_json(manifest: &serde_json::Value) -> Result<Self, String> {
        let declared_task = manifest
            .get("task")
            .and_then(serde_json::Value::as_str)
            .map(|task| task.trim().to_ascii_lowercase())
            .filter(|task| !task.is_empty())
            .or_else(|| {
                let model_type = manifest.get("model_type")?.as_str()?;
                match model_type.trim().to_ascii_lowercase().as_str() {
                    "embedding" => Some("embed".to_string()),
                    "seq-classifier" => Some("classify".to_string()),
                    _ => None,
                }
            });
        let task = declared_task
            .as_deref()
            .and_then(GgufPooledTask::from_task)
            .ok_or_else(|| {
                format!(
                    "the model manifest does not declare a pooled GGUF task \
                     (embed, rerank, classify); task={declared_task:?}"
                )
            })?;
        let mut config = Self::new(task);
        let section = |name: &str| manifest.get("metadata").and_then(|m| m.get(name));
        if let Some(value) = section("embed").and_then(|s| s.get("normalize")) {
            config.normalize = value
                .as_bool()
                .ok_or("metadata.embed.normalize must be a boolean")?;
        }
        if let Some(value) = section("embed").and_then(|s| s.get("pooling")) {
            let value = value
                .as_str()
                .ok_or("metadata.embed.pooling must be a string")?;
            config.pooling = Some(GgufPooling::parse(value)?);
        }
        if let Some(value) = section("classify").and_then(|s| s.get("apply_softmax")) {
            config.apply_softmax = value
                .as_bool()
                .ok_or("metadata.classify.apply_softmax must be a boolean")?;
        }
        Ok(config)
    }

    /// [`Self::from_manifest_json`] for a parsed manifest. Legacy manifests
    /// without explicit axes use the manifest's effective task.
    pub fn from_manifest(manifest: &kapsl_core::Manifest) -> Result<Self, String> {
        let mut value = serde_json::to_value(manifest)
            .map_err(|error| format!("encode model manifest: {error}"))?;
        if value.get("task").is_none_or(serde_json::Value::is_null) {
            value["task"] =
                serde_json::Value::String(kapsl_core::engine_kind::effective_task(manifest));
        }
        Self::from_manifest_json(&value)
    }
}

fn pooled_max_sequences() -> usize {
    std::env::var(POOLED_MAX_SEQS_ENV)
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|&value| value > 0)
        .unwrap_or(POOLED_MAX_SEQS_DEFAULT)
        .min(LLAMA_MAX_SEQ)
}

// ─── Request decoding ─────────────────────────────────────────────────────────

/// One unit of work: one output row.
#[derive(Debug, Clone, PartialEq, Eq)]
enum PooledItem {
    Text(String),
    Tokens(Vec<i32>),
    Pair { query: String, document: String },
}

/// `name` is `base` or `base.<index>`.
fn input_matches(name: &str, base: &str) -> bool {
    name == base
        || name
            .strip_prefix(base)
            .and_then(|rest| rest.strip_prefix('.'))
            .is_some_and(|index| !index.is_empty() && index.bytes().all(|b| b.is_ascii_digit()))
}

fn packet_text(packet: &BinaryTensorPacket, label: &str) -> Result<String, EngineError> {
    if !matches!(packet.dtype, TensorDtype::Utf8 | TensorDtype::Uint8) {
        return Err(EngineError::invalid_input(format!(
            "input `{label}` must be UTF-8 text (string or uint8), got {}",
            packet.dtype
        )));
    }
    String::from_utf8(packet.data.clone()).map_err(|error| {
        EngineError::invalid_input(format!("input `{label}` is not valid UTF-8: {error}"))
    })
}

fn packet_token_ids(packet: &BinaryTensorPacket, label: &str) -> Result<Vec<i32>, EngineError> {
    match packet.shape.as_slice() {
        [_] | [1, _] => {}
        shape => {
            return Err(EngineError::invalid_input(format!(
                "token-id input `{label}` must have shape [len] or [1, len], got {shape:?}"
            )))
        }
    }
    match packet.dtype {
        TensorDtype::Int32 => Ok(packet
            .data
            .as_chunks::<4>()
            .0
            .iter()
            .map(|bytes| i32::from_le_bytes(*bytes))
            .collect()),
        TensorDtype::Int64 => packet
            .data
            .as_chunks::<8>()
            .0
            .iter()
            .map(|bytes| {
                let id = i64::from_le_bytes(*bytes);
                i32::try_from(id).map_err(|_| {
                    EngineError::invalid_input(format!(
                        "token id {id} in input `{label}` does not fit the model vocabulary"
                    ))
                })
            })
            .collect(),
        other => Err(EngineError::invalid_input(format!(
            "token-id input `{label}` must be int32 or int64, got {other}"
        ))),
    }
}

fn packet_item(packet: &BinaryTensorPacket, label: &str) -> Result<PooledItem, EngineError> {
    packet.validate()?;
    match packet.dtype {
        TensorDtype::Int32 | TensorDtype::Int64 => {
            packet_token_ids(packet, label).map(PooledItem::Tokens)
        }
        _ => packet_text(packet, label).map(PooledItem::Text),
    }
}

/// Decode a request into work items in input order, per the module contract.
fn request_items(
    task: GgufPooledTask,
    request: &InferenceRequest,
) -> Result<Vec<PooledItem>, EngineError> {
    let items = match task {
        GgufPooledTask::Embed | GgufPooledTask::Classify => {
            let mut items = Vec::with_capacity(request.additional_inputs.len() + 1);
            items.push(packet_item(&request.input, "text")?);
            for named in &request.additional_inputs {
                if !input_matches(&named.name, "text") {
                    return Err(EngineError::invalid_input(format!(
                        "unexpected input `{}`; GGUF {} takes `text` inputs",
                        named.name,
                        task.as_str()
                    )));
                }
                items.push(packet_item(&named.tensor, &named.name)?);
            }
            items
        }
        GgufPooledTask::Rerank => {
            request.input.validate()?;
            let query = packet_text(&request.input, "query")?;
            let mut items = Vec::with_capacity(request.additional_inputs.len());
            for named in &request.additional_inputs {
                if !input_matches(&named.name, "document") {
                    return Err(EngineError::invalid_input(format!(
                        "unexpected input `{}`; GGUF rerank takes a query and `document` inputs",
                        named.name
                    )));
                }
                named.tensor.validate()?;
                items.push(PooledItem::Pair {
                    query: query.clone(),
                    document: packet_text(&named.tensor, &named.name)?,
                });
            }
            if items.is_empty() {
                return Err(EngineError::invalid_input(
                    "GGUF rerank needs at least one `document` input",
                ));
            }
            items
        }
    };
    if items.len() > MAX_INPUTS_PER_REQUEST {
        return Err(EngineError::invalid_input(format!(
            "request has {} inputs; GGUF {} accepts at most {MAX_INPUTS_PER_REQUEST} per request",
            items.len(),
            task.as_str()
        )));
    }
    Ok(items)
}

// ─── Tokenization ─────────────────────────────────────────────────────────────

/// Special tokens around a rerank pair, each present only when the vocabulary
/// asks for it (`add_bos` / `add_eos` / `add_sep`).
#[derive(Debug, Clone, Copy, Default)]
struct RerankSpecials {
    bos: Option<LlamaToken>,
    eos: Option<LlamaToken>,
    sep: Option<LlamaToken>,
}

/// `[BOS] query [EOS] [SEP] document [EOS]`, llama-server's layout for
/// rerankers without a `rerank` chat template.
fn assemble_rerank_tokens(
    query: &[LlamaToken],
    document: &[LlamaToken],
    specials: RerankSpecials,
) -> Vec<LlamaToken> {
    let mut tokens = Vec::with_capacity(query.len() + document.len() + 4);
    tokens.extend(specials.bos);
    tokens.extend_from_slice(query);
    tokens.extend(specials.eos);
    tokens.extend(specials.sep);
    tokens.extend_from_slice(document);
    tokens.extend(specials.eos);
    tokens
}

/// Substitute `{query}` and `{document}` in a rerank template in one pass,
/// so placeholder text inside the query is not substituted again.
fn fill_rerank_template(template: &str, query: &str, document: &str) -> String {
    const QUERY: &str = "{query}";
    const DOCUMENT: &str = "{document}";
    let mut filled = String::with_capacity(template.len() + query.len() + document.len());
    let mut rest = template;
    loop {
        let next_query = rest.find(QUERY);
        let next_document = rest.find(DOCUMENT);
        let (index, placeholder, value) = match (next_query, next_document) {
            (Some(q), Some(d)) if q < d => (q, QUERY, query),
            (Some(q), None) => (q, QUERY, query),
            (_, Some(d)) => (d, DOCUMENT, document),
            (None, None) => break,
        };
        filled.push_str(&rest[..index]);
        filled.push_str(value);
        rest = &rest[index + placeholder.len()..];
    }
    filled.push_str(rest);
    filled
}

fn rerank_tokens(
    model: &LlamaModel,
    query: &str,
    document: &str,
) -> Result<Vec<LlamaToken>, String> {
    // Models converted with a rerank prompt (e.g. Qwen3-Reranker) ship it as
    // the `rerank` chat template; llama-server fills it the same way.
    if let Ok(template) = model.chat_template(Some("rerank")) {
        let template = template
            .to_str()
            .map_err(|error| format!("rerank template is not UTF-8: {error}"))?;
        let prompt = fill_rerank_template(template, query, document);
        return model
            .tokenize(&prompt, false, true)
            .map_err(|error| error.to_string());
    }
    let query_tokens = model
        .tokenize(query, false, false)
        .map_err(|error| error.to_string())?;
    let document_tokens = model
        .tokenize(document, false, false)
        .map_err(|error| error.to_string())?;
    let eos = model.token_eos();
    let sep = model.token_sep();
    let eos_or_sep = if eos.0 >= 0 { eos } else { sep };
    let specials = RerankSpecials {
        bos: model.add_bos_token().then(|| model.token_bos()),
        eos: model.add_eos_token().then_some(eos_or_sep),
        sep: model.add_sep_token().then_some(sep),
    };
    Ok(assemble_rerank_tokens(
        &query_tokens,
        &document_tokens,
        specials,
    ))
}

fn tokenize_item(
    model: &LlamaModel,
    item: &PooledItem,
    n_vocab: i32,
) -> Result<Vec<LlamaToken>, String> {
    match item {
        // Same as llama-server's embedding path: the vocabulary's special
        // tokens (CLS/SEP, BOS/EOS) are added, and special-token text parsed.
        PooledItem::Text(text) => model
            .tokenize(text, true, true)
            .map_err(|error| error.to_string()),
        PooledItem::Tokens(ids) => ids
            .iter()
            .map(|&id| {
                if (0..n_vocab).contains(&id) {
                    Ok(LlamaToken(id))
                } else {
                    Err(format!(
                        "token id {id} is outside the model vocabulary (0..{n_vocab})"
                    ))
                }
            })
            .collect(),
        PooledItem::Pair { query, document } => rerank_tokens(model, query, document),
    }
}

// ─── Batching and post-processing ─────────────────────────────────────────────

/// Group consecutive sequences into llama.cpp batches of at most `n_batch`
/// tokens and `max_seqs` sequences. Every length must already be in
/// `1..=n_batch`.
fn plan_batches(lengths: &[usize], n_batch: usize, max_seqs: usize) -> Vec<Range<usize>> {
    let mut groups = Vec::new();
    let mut start = 0;
    let mut tokens = 0;
    for (index, &length) in lengths.iter().enumerate() {
        if index > start && (tokens + length > n_batch || index - start >= max_seqs) {
            groups.push(start..index);
            start = index;
            tokens = 0;
        }
        tokens += length;
    }
    if start < lengths.len() {
        groups.push(start..lengths.len());
    }
    groups
}

fn l2_normalize(values: &mut [f32]) {
    let norm = values.iter().map(|v| v * v).sum::<f32>().sqrt();
    if norm > 0.0 {
        values.iter_mut().for_each(|v| *v /= norm);
    }
}

fn softmax(values: &mut [f32]) {
    let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0;
    for value in values.iter_mut() {
        *value = (*value - max).exp();
        sum += *value;
    }
    if sum > 0.0 {
        values.iter_mut().for_each(|v| *v /= sum);
    }
}

fn f32_packet(shape: Vec<i64>, values: &[f32]) -> Result<BinaryTensorPacket, EngineError> {
    let data = values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect();
    BinaryTensorPacket::new(shape, TensorDtype::Float32, data)
}

/// Turn raw pooled rows into the task's output tensor (see module docs).
fn finish_outputs(
    config: &GgufPooledConfig,
    mut rows: Vec<Vec<f32>>,
    width: usize,
) -> Result<BinaryTensorPacket, EngineError> {
    for (index, row) in rows.iter().enumerate() {
        if row.len() != width || width == 0 {
            return Err(EngineError::backend(format!(
                "llama.cpp returned {} values for input {index}, expected {width}",
                row.len()
            )));
        }
        if row.iter().any(|value| !value.is_finite()) {
            return Err(EngineError::backend(format!(
                "llama.cpp returned a non-finite value for input {index}"
            )));
        }
    }
    let n = rows.len() as i64;
    match config.task {
        GgufPooledTask::Embed => {
            if config.normalize {
                rows.iter_mut().for_each(|row| l2_normalize(row));
            }
            f32_packet(vec![n, width as i64], &rows.concat())
        }
        GgufPooledTask::Classify => {
            if config.apply_softmax {
                rows.iter_mut().for_each(|row| softmax(row));
            }
            f32_packet(vec![n, width as i64], &rows.concat())
        }
        // llama-server's relevance score is the head's first output.
        GgufPooledTask::Rerank => {
            let scores: Vec<f32> = rows.iter().map(|row| row[0]).collect();
            f32_packet(vec![n], &scores)
        }
    }
}

/// Validate the pooling the context resolved against the task.
fn check_pooling(task: GgufPooledTask, pooling: LlamaPoolingType) -> Result<(), String> {
    match (task, pooling) {
        (
            GgufPooledTask::Embed,
            LlamaPoolingType::Mean | LlamaPoolingType::Cls | LlamaPoolingType::Last,
        ) => Ok(()),
        (GgufPooledTask::Embed, LlamaPoolingType::Rank) => Err(
            "the model's default pooling is rank (a reranker or classifier head); package it \
             as seq-classifier with task rerank or classify, or set metadata.embed.pooling"
                .to_string(),
        ),
        (GgufPooledTask::Embed, other) => Err(format!(
            "the model declares no embedding pooling ({other:?}); set metadata.embed.pooling \
             to mean, cls, or last"
        )),
        (_, LlamaPoolingType::Rank) => Ok(()),
        (task, other) => Err(format!(
            "GGUF {} needs rank pooling, but llama.cpp resolved {other:?}",
            task.as_str()
        )),
    }
}

/// Planned compute-buffer bytes for a pooled context, before llama.cpp has
/// measured it. An upper bound in the spirit of `estimate_compute_buffer_bytes`:
///
/// * `4 * (n_embd + n_ff) * n_ubatch`: one f32 hidden and FFN row per token.
/// * `4 * n_head * n_ubatch * n_ctx`: the f32 attention scores of one
///   micro-batch when flash attention is not used (encoders attend over the
///   whole sequence at once).
/// * `4 * n_vocab_out * n_ubatch`: the f32 language-model head output of a
///   decoder. llama.cpp marks every token as an output when it computes
///   embeddings, so the head runs over the whole micro-batch: 1.24 GB of the
///   1.27 GB measured for Qwen3-Embedding-0.6B (151,669-token vocabulary,
///   `n_ubatch` 2048). Zero for encoders, which have no head.
/// * `256 * n_ctx`: masks and position bookkeeping.
/// * `17/16` headroom, with a 64 MiB floor. The measured buffer replaces the
///   estimate after load.
fn estimate_pooled_compute_bytes(
    n_embd: usize,
    n_ff: usize,
    n_head: usize,
    n_vocab_out: usize,
    n_ubatch: usize,
    n_ctx: usize,
) -> usize {
    let activations = 4usize
        .saturating_mul(n_embd.saturating_add(n_ff))
        .saturating_mul(n_ubatch);
    let attention = 4usize
        .saturating_mul(n_head.max(1))
        .saturating_mul(n_ubatch)
        .saturating_mul(n_ctx);
    let head = 4usize.saturating_mul(n_vocab_out).saturating_mul(n_ubatch);
    let bookkeeping = 256usize.saturating_mul(n_ctx);
    let raw = activations
        .saturating_add(attention)
        .saturating_add(head)
        .saturating_add(bookkeeping);
    raw.saturating_add(raw / 16).max(MIN_POOLED_SCRATCH_BYTES)
}

/// `(kv, scratch)` planned bytes for `model_path`, without loading it.
fn planned_pooled_bytes(model_path: &Path, file_bytes: usize) -> (usize, usize) {
    let fallback = (0, (file_bytes / 8).max(MIN_POOLED_SCRATCH_BYTES));
    let Ok(file) = kapsl_loader::gguf_loader::GgufFile::open(model_path) else {
        return fallback;
    };
    let (Ok(config), Ok((head_k, head_v))) = (file.extract_config(), file.attention_head_widths())
    else {
        return fallback;
    };
    let ctx = (n_ctx_per_seq() as usize)
        .min(config.max_position_embeddings)
        .max(1);
    // Upper bound: decoder embedders keep one context of f16 K and V while a
    // batch runs; encoder models allocate none.
    let kv = config
        .num_hidden_layers
        .max(1)
        .saturating_mul(config.num_kv_heads().max(1))
        .saturating_mul(head_k.max(1).saturating_add(head_v.max(1)))
        .saturating_mul(std::mem::size_of::<u16>())
        .saturating_mul(ctx);
    let has_lm_head = file.causal_attention() || file.has_tensor("output.weight");
    let scratch = estimate_pooled_compute_bytes(
        config.hidden_size,
        config.intermediate_size,
        config.num_attention_heads,
        if has_lm_head { config.vocab_size } else { 0 },
        ctx,
        ctx,
    );
    (kv, scratch)
}

fn model_has_classifier_head(model_path: &Path) -> Result<bool, EngineError> {
    let file = kapsl_loader::gguf_loader::GgufFile::open(model_path)
        .map_err(|error| EngineError::backend(format!("read GGUF tensors: {error}")))?;
    Ok(CLASSIFIER_HEAD_TENSORS
        .iter()
        .any(|name| file.has_tensor(name)))
}

// ─── Worker ───────────────────────────────────────────────────────────────────

struct PooledJob {
    items: Vec<PooledItem>,
    cancellation: Option<CancellationToken>,
    reply: std_mpsc::SyncSender<Result<PooledJobOutput, EngineError>>,
}

struct PooledJobOutput {
    rows: Vec<Vec<f32>>,
    tokens: usize,
}

#[derive(Debug, Clone, Copy)]
struct PooledContextPlan {
    task: GgufPooledTask,
    pooling: LlamaPoolingType,
    /// Tokens per batch and micro-batch, and the context size: one batch at a
    /// time lives in a unified KV cache that is cleared before the next.
    n_batch: u32,
    max_seqs: u32,
}

fn run_pooled_worker(
    weights: Arc<GgufWeights>,
    plan: PooledContextPlan,
    ready: std_mpsc::SyncSender<Result<ContextMemory, String>>,
    jobs: std_mpsc::Receiver<PooledJob>,
) {
    let Some(n_ctx) = NonZeroU32::new(plan.n_batch) else {
        let _ = ready.send(Err("invalid pooled GGUF context size 0".to_string()));
        return;
    };
    // Non-causal encoders attend over a whole sequence in one micro-batch, so
    // the micro-batch must hold the longest accepted sequence.
    let params = LlamaContextParams::default()
        .with_n_ctx(Some(n_ctx))
        .with_n_batch(plan.n_batch)
        .with_n_ubatch(plan.n_batch)
        .with_n_seq_max(plan.max_seqs)
        .with_kv_unified(true)
        .with_embeddings(true)
        .with_pooling_type(plan.pooling)
        .with_offload_kqv(true)
        .with_flash_attention_policy(LLAMA_FLASH_ATTN_TYPE_AUTO);
    let mut ctx = match weights.model.new_context(&weights.backend, params) {
        Ok(ctx) => ctx,
        Err(error) => {
            let _ = ready.send(Err(format!("GGUF pooled context creation failed: {error}")));
            return;
        }
    };
    if let Err(message) = check_pooling(plan.task, ctx.pooling_type()) {
        let _ = ready.send(Err(message));
        return;
    }
    ctx.reserve_compute_buffers();
    let memory = ContextMemory {
        compute: ctx.device_compute_memory(),
        kv: ctx.device_context_memory(),
        host: ctx.host_memory(),
    };
    log::info!(
        "[gguf-pooled] {} context ready (n_batch={}, max_seqs={}, pooling={:?}); compute per device: {:?}; KV per device: {:?}",
        plan.task.as_str(),
        plan.n_batch,
        plan.max_seqs,
        ctx.pooling_type(),
        memory.compute,
        memory.kv
    );
    if ready.send(Ok(memory)).is_err() {
        return;
    }

    let mut batch = LlamaBatch::new(plan.n_batch as usize, plan.max_seqs as i32);
    let use_encode = weights.model.has_encoder() && !weights.model.has_decoder();
    while let Ok(job) = jobs.recv() {
        let result = run_pooled_job(&weights.model, &mut ctx, &mut batch, use_encode, plan, &job);
        let _ = job.reply.send(result);
    }
}

fn run_pooled_job(
    model: &LlamaModel,
    ctx: &mut LlamaContext<'_>,
    batch: &mut LlamaBatch<'_>,
    use_encode: bool,
    plan: PooledContextPlan,
    job: &PooledJob,
) -> Result<PooledJobOutput, EngineError> {
    let n_batch = plan.n_batch as usize;
    let n_vocab = model.n_vocab();
    let mut sequences = Vec::with_capacity(job.items.len());
    for (index, item) in job.items.iter().enumerate() {
        let tokens = tokenize_item(model, item, n_vocab)
            .map_err(|error| EngineError::invalid_input(format!("input {index}: {error}")))?;
        if tokens.is_empty() {
            return Err(EngineError::invalid_input(format!(
                "input {index} produced no tokens"
            )));
        }
        if tokens.len() > n_batch {
            return Err(EngineError::invalid_input(format!(
                "input {index} has {} tokens; this model accepts at most {n_batch} tokens per \
                 input (the smaller of KAPSL_GGUF_CTX_PER_SEQ and its training context)",
                tokens.len()
            )));
        }
        sequences.push(tokens);
    }

    let lengths: Vec<usize> = sequences.iter().map(Vec::len).collect();
    let mut rows = vec![Vec::new(); sequences.len()];
    for group in plan_batches(&lengths, n_batch, plan.max_seqs as usize) {
        if job
            .cancellation
            .as_ref()
            .is_some_and(CancellationToken::is_cancelled)
        {
            return Err(EngineError::cancelled("pooled GGUF request cancelled"));
        }
        ctx.clear_kv_cache();
        batch.clear();
        for (seq, index) in group.clone().enumerate() {
            batch
                .add_sequence(&sequences[index], seq as i32, true)
                .map_err(|error| EngineError::backend(format!("build pooled batch: {error}")))?;
        }
        if use_encode {
            ctx.encode(batch)
                .map_err(|error| EngineError::backend(format!("llama.cpp encode: {error}")))?;
        } else {
            ctx.decode(batch)
                .map_err(|error| EngineError::backend(format!("llama.cpp decode: {error}")))?;
        }
        for (seq, index) in group.enumerate() {
            let output = match plan.task {
                GgufPooledTask::Embed => ctx.embeddings_seq_ith(seq as i32),
                GgufPooledTask::Rerank | GgufPooledTask::Classify => ctx.rank_seq_ith(seq as i32),
            }
            .map_err(|error| {
                EngineError::backend(format!("read pooled output for input {index}: {error}"))
            })?;
            rows[index] = output.to_vec();
        }
    }
    Ok(PooledJobOutput {
        rows,
        tokens: lengths.iter().sum(),
    })
}

// ─── Backend ──────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct PooledShape {
    n_embd: usize,
    n_cls_out: usize,
    n_batch: usize,
    labels: Vec<String>,
}

struct PooledInner {
    weights: Arc<GgufWeights>,
    jobs: std_mpsc::Sender<PooledJob>,
    worker: Option<std::thread::JoinHandle<()>>,
    memory: ContextMemory,
    shape: PooledShape,
    scratch_allocation_id: String,
    kv_allocation_id: String,
}

/// Embeddings, reranking or sequence classification for one GGUF model
/// through llama.cpp pooling. See the module docs for the request contract.
pub struct GgufPooledBackend {
    device_id: usize,
    config: GgufPooledConfig,
    /// Process-unique replica id, shared with `GgufBackend`'s counter so the
    /// per-replica compute and KV allocation ids never collide.
    instance_id: u64,
    inner: Option<PooledInner>,
    metrics: Arc<Mutex<EngineMetrics>>,
}

impl GgufPooledBackend {
    /// A backend for `config.task` whose weights live on engine device
    /// `device_id` (CPU builds ignore the device).
    pub fn new(device_id: usize, config: GgufPooledConfig) -> Self {
        Self {
            device_id,
            config,
            instance_id: GGUF_INSTANCE_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
            inner: None,
            metrics: Arc::new(Mutex::new(EngineMetrics::new())),
        }
    }

    /// The task and post-processing options this backend serves.
    pub fn config(&self) -> &GgufPooledConfig {
        &self.config
    }

    /// Classifier labels of the loaded model, in output order, when the model
    /// provides them (`classify` only).
    pub fn labels(&self) -> Option<&[String]> {
        self.inner
            .as_ref()
            .map(|inner| inner.shape.labels.as_slice())
            .filter(|labels| !labels.is_empty())
    }

    fn output_width(&self, shape: &PooledShape) -> usize {
        match self.config.task {
            GgufPooledTask::Embed => shape.n_embd,
            GgufPooledTask::Rerank | GgufPooledTask::Classify => shape.n_cls_out,
        }
    }

    fn run(
        &self,
        items: Vec<PooledItem>,
        cancellation: Option<CancellationToken>,
    ) -> Result<BinaryTensorPacket, EngineError> {
        let inner = self.inner.as_ref().ok_or(EngineError::ModelNotLoaded)?;
        let count = items.len();
        let started = Instant::now();
        let (reply, response) = std_mpsc::sync_channel(1);
        inner
            .jobs
            .send(PooledJob {
                items,
                cancellation,
                reply,
            })
            .map_err(|_| EngineError::backend("pooled GGUF worker has stopped"))?;
        let output = response
            .recv()
            .map_err(|_| EngineError::backend("pooled GGUF worker exited before replying"))??;
        let packet = finish_outputs(&self.config, output.rows, self.output_width(&inner.shape))?;
        if let Ok(mut metrics) = self.metrics.lock() {
            let elapsed = started.elapsed().as_secs_f64();
            metrics.inference_time = elapsed * 1000.0;
            metrics.batch_size = count;
            metrics.prompt_tokens_total = metrics
                .prompt_tokens_total
                .saturating_add(output.tokens as u64);
            if elapsed > 0.0 {
                metrics.throughput = count as f64 / elapsed;
            }
        }
        Ok(packet)
    }
}

#[async_trait]
impl Engine for GgufPooledBackend {
    fn planned_memory(&self, model_path: &Path) -> Result<MemoryReport, EngineError> {
        let weights = std::fs::metadata(model_path)
            .map_err(|error| EngineError::backend(format!("stat GGUF model: {error}")))?
            .len() as usize;
        let (kv, scratch) = planned_pooled_bytes(model_path, weights);
        let placement = planned_gguf_placement(self.device_id, model_path, weights, kv, scratch)?;
        Ok(MemoryReport {
            allocations: allocations_from_placement(
                &placement,
                &gguf_allocation_id(model_path, self.device_id),
                &gguf_kv_allocation_id(model_path, self.instance_id),
                &gguf_scratch_allocation_id(model_path, self.instance_id),
            ),
        })
    }

    fn planned_external_device_memory(
        &self,
        model_path: &Path,
    ) -> Result<ExternalDeviceMemoryReport, EngineError> {
        let report = self.planned_memory(model_path)?;
        Ok(ExternalDeviceMemoryReport {
            allocations: report
                .allocations
                .into_iter()
                .filter_map(|allocation| match allocation.domain {
                    MemoryDomain::Cuda { device_id } => Some(ExternalDeviceMemory {
                        allocation_id: allocation.allocation_id,
                        device_id,
                        bytes: allocation.bytes,
                    }),
                    _ => None,
                })
                .collect(),
        })
    }

    async fn load(&mut self, model_path: &Path) -> Result<(), EngineError> {
        if self.inner.is_some() {
            self.unload();
        }
        let task = self.config.task;
        if task != GgufPooledTask::Embed && !model_has_classifier_head(model_path)? {
            return Err(EngineError::backend(format!(
                "{} has no classification head (no {} tensor), so it cannot serve GGUF {}; \
                 package an embedding model with task embed instead",
                model_path.display(),
                CLASSIFIER_HEAD_TENSORS.join(" or "),
                task.as_str()
            )));
        }
        // llama.cpp must be initialized before devices or weights exist.
        global_gguf_backend()?;
        let weights = shared_gguf_weights(model_path, self.device_id).await?;
        let n_batch = (n_ctx_per_seq() as usize)
            .min(weights.n_ctx_train.max(1) as usize)
            .max(1);
        let n_cls_out = weights.model.n_cls_out() as usize;
        let shape = PooledShape {
            n_embd: weights.model.n_embd().max(0) as usize,
            n_cls_out,
            n_batch,
            labels: (0..n_cls_out as u32)
                .filter_map(|index| weights.model.cls_label(index))
                .collect(),
        };
        let pooling = match task {
            GgufPooledTask::Embed => self
                .config
                .pooling
                .map(GgufPooling::llama)
                .unwrap_or(LlamaPoolingType::Unspecified),
            GgufPooledTask::Rerank | GgufPooledTask::Classify => LlamaPoolingType::Rank,
        };
        let plan = PooledContextPlan {
            task,
            pooling,
            n_batch: n_batch as u32,
            max_seqs: pooled_max_sequences() as u32,
        };

        let (ready_tx, ready_rx) = std_mpsc::sync_channel(1);
        let (jobs, job_rx) = std_mpsc::channel();
        let worker_weights = Arc::clone(&weights);
        let worker = std::thread::Builder::new()
            .name(format!("kapsl-gguf-{}", task.as_str()))
            .spawn(move || run_pooled_worker(worker_weights, plan, ready_tx, job_rx))
            .map_err(|error| EngineError::backend(format!("start pooled GGUF worker: {error}")))?;
        let ready = tokio::task::spawn_blocking(move || ready_rx.recv())
            .await
            .map_err(|error| EngineError::backend(format!("pooled GGUF startup join: {error}")))?;
        let memory = match ready {
            Ok(Ok(memory)) => memory,
            Ok(Err(message)) => {
                let _ = worker.join();
                return Err(EngineError::backend(message));
            }
            Err(_) => {
                let _ = worker.join();
                return Err(EngineError::backend(
                    "pooled GGUF worker exited during startup",
                ));
            }
        };
        log::info!(
            "[gguf-pooled] loaded {} for {} (n_embd={}, n_cls_out={}, max tokens per input={})",
            model_path.display(),
            task.as_str(),
            shape.n_embd,
            shape.n_cls_out,
            shape.n_batch
        );
        self.inner = Some(PooledInner {
            weights,
            jobs,
            worker: Some(worker),
            memory,
            shape,
            scratch_allocation_id: gguf_scratch_allocation_id(model_path, self.instance_id),
            kv_allocation_id: gguf_kv_allocation_id(model_path, self.instance_id),
        });
        if let Ok(mut metrics) = self.metrics.lock() {
            *metrics = EngineMetrics::new();
        }
        Ok(())
    }

    fn actual_memory(&self) -> MemoryReport {
        let Some(inner) = self.inner.as_ref() else {
            return MemoryReport::default();
        };
        let placement = actual_gguf_placement(self.device_id, &inner.weights.model, &inner.memory);
        MemoryReport {
            allocations: allocations_from_placement(
                &placement,
                &inner.weights.allocation_id,
                &inner.kv_allocation_id,
                &inner.scratch_allocation_id,
            ),
        }
    }

    fn infer(&self, request: &InferenceRequest) -> Result<BinaryTensorPacket, EngineError> {
        let items = request_items(self.config.task, request)?;
        self.run(items, request.cancellation.clone())
    }

    fn infer_stream(&self, request: &InferenceRequest) -> EngineStream {
        let result = self.infer(request);
        Box::pin(futures::stream::once(async move { result }))
    }

    fn batching_policy(&self) -> BatchingPolicy {
        BatchingPolicy::none()
    }

    fn unload(&mut self) {
        if let Some(mut inner) = self.inner.take() {
            // Closing the job channel ends the worker loop after any job in
            // progress; the context is freed on the worker thread.
            drop(inner.jobs);
            if let Some(worker) = inner.worker.take() {
                if worker.join().is_err() {
                    log::warn!("[gguf-pooled] worker thread panicked during unload");
                }
            }
            drop(inner.weights);
        }
        if let Ok(mut metrics) = self.metrics.lock() {
            *metrics = EngineMetrics::new();
        }
    }

    fn metrics(&self) -> EngineMetrics {
        self.metrics
            .lock()
            .map(|metrics| metrics.clone())
            .unwrap_or_default()
    }

    fn model_info(&self) -> Option<EngineModelInfo> {
        let inner = self.inner.as_ref()?;
        let text = || vec![TensorDtype::Utf8.as_str().to_string()];
        let (input_names, input_dtypes, input_shapes, output_name, output_shape) =
            match self.config.task {
                GgufPooledTask::Embed => (
                    vec!["text".to_string()],
                    text(),
                    vec![vec![-1]],
                    "embeddings",
                    vec![-1, inner.shape.n_embd as i64],
                ),
                GgufPooledTask::Classify => (
                    vec!["text".to_string()],
                    text(),
                    vec![vec![-1]],
                    if self.config.apply_softmax {
                        "probabilities"
                    } else {
                        "logits"
                    },
                    vec![-1, inner.shape.n_cls_out as i64],
                ),
                GgufPooledTask::Rerank => (
                    vec!["query".to_string(), "document".to_string()],
                    vec![
                        TensorDtype::Utf8.as_str().to_string(),
                        TensorDtype::Utf8.as_str().to_string(),
                    ],
                    vec![vec![-1], vec![-1]],
                    "scores",
                    vec![-1],
                ),
            };
        Some(EngineModelInfo {
            input_names,
            output_names: vec![output_name.to_string()],
            input_shapes,
            output_shapes: vec![output_shape],
            input_dtypes,
            output_dtypes: vec![TensorDtype::Float32.as_str().to_string()],
            framework: Some("gguf".to_string()),
            model_version: None,
            peak_concurrency: Some(1),
        })
    }

    fn health_check(&self) -> Result<(), EngineError> {
        match self.inner.as_ref() {
            Some(inner) if inner.worker.as_ref().is_some_and(|w| !w.is_finished()) => Ok(()),
            Some(_) => Err(EngineError::backend("pooled GGUF worker has stopped")),
            None => Err(EngineError::ModelNotLoaded),
        }
    }
}

#[cfg(test)]
#[path = "pooled_tests.rs"]
mod pooled_tests;
