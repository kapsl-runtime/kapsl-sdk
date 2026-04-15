//! Safetensors weight loader.
//!
//! Maps the file into memory and pulls tensors by name into `TensorData`
//! structures. The file stays mapped for the lifetime of the returned
//! `ModelWeights`; no heap copies are made for the raw bytes.

use memmap2::Mmap;
use std::collections::HashMap;
use std::fs::File;
use std::path::Path;
use thiserror::Error;

use crate::config::ModelConfig;
use crate::weights::{DType, LayerWeights, ModelWeights, TensorData, WeightError};

#[derive(Debug, Error)]
pub enum LoadError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Config error: {0}")]
    Config(#[from] crate::config::ConfigError),
    #[error("Safetensors parse error: {0}")]
    Parse(String),
    #[error("Weight error: {0}")]
    Weight(#[from] WeightError),
    #[error("No safetensors files found in {0}")]
    NoSafetensors(String),
}

// ── Safetensors on-disk structures ──────────────────────────────────────────

#[derive(Debug, serde::Deserialize)]
struct TensorHeader {
    dtype: String,
    shape: Vec<usize>,
    data_offsets: [usize; 2],
}

/// Parse the safetensors header JSON and return a map of name → header.
fn parse_header(mmap: &Mmap) -> Result<HashMap<String, TensorHeader>, LoadError> {
    if mmap.len() < 8 {
        return Err(LoadError::Parse("file too small".into()));
    }
    let header_len = u64::from_le_bytes(mmap[..8].try_into().unwrap()) as usize;
    if 8 + header_len > mmap.len() {
        return Err(LoadError::Parse("header length exceeds file size".into()));
    }
    let json_bytes = &mmap[8..8 + header_len];
    let mut map: HashMap<String, serde_json::Value> = serde_json::from_slice(json_bytes)
        .map_err(|e| LoadError::Parse(format!("JSON parse: {e}")))?;

    // Remove the special __metadata__ key if present.
    map.remove("__metadata__");

    let mut headers = HashMap::with_capacity(map.len());
    for (name, val) in map {
        let h: TensorHeader = serde_json::from_value(val)
            .map_err(|e| LoadError::Parse(format!("tensor header '{name}': {e}")))?;
        headers.insert(name, h);
    }
    Ok(headers)
}

/// Extract a named tensor's bytes and metadata from the mmap.
fn extract_tensor(
    name: &str,
    headers: &HashMap<String, TensorHeader>,
    mmap: &Mmap,
    data_base: usize,
) -> Result<TensorData, WeightError> {
    let h = headers
        .get(name)
        .ok_or_else(|| WeightError::Missing(name.to_string()))?;

    let dtype = DType::from_str(&h.dtype)
        .ok_or_else(|| WeightError::UnsupportedDtype(h.dtype.clone(), name.to_string()))?;

    let [start, end] = h.data_offsets;
    let bytes = mmap[data_base + start..data_base + end].to_vec();

    Ok(TensorData::new(bytes, dtype, h.shape.clone()))
}

// ── Public API ───────────────────────────────────────────────────────────────

/// Load all safetensors shards from a model directory into CPU memory.
///
/// Shard files must be named `model.safetensors` (single shard) or
/// `model-00001-of-NNNNN.safetensors` (multi-shard). Config is loaded from
/// `config.json` in the same directory.
pub fn load_safetensors(model_dir: &Path) -> Result<ModelWeights, LoadError> {
    let config = ModelConfig::from_model_dir(model_dir)?;

    // Collect shard files.
    let mut shards: Vec<std::path::PathBuf> = Vec::new();
    let single = model_dir.join("model.safetensors");
    if single.exists() {
        shards.push(single);
    } else {
        // Multi-shard: model-00001-of-NNNNN.safetensors
        let mut entries: Vec<_> = std::fs::read_dir(model_dir)?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| {
                p.extension().and_then(|e| e.to_str()) == Some("safetensors")
                    && p.file_name()
                        .and_then(|n| n.to_str())
                        .map(|n| n.starts_with("model-"))
                        .unwrap_or(false)
            })
            .collect();
        entries.sort();
        shards.extend(entries);
    }

    if shards.is_empty() {
        return Err(LoadError::NoSafetensors(model_dir.display().to_string()));
    }

    // Load and merge all shards into a single flat map.
    let mut all: HashMap<String, TensorData> = HashMap::new();
    for shard_path in &shards {
        log::info!("Loading shard: {}", shard_path.display());
        let file = File::open(shard_path)?;
        // SAFETY: the mmap is read-only and the file is not modified during loading.
        let mmap = unsafe { Mmap::map(&file)? };
        let header_len =
            u64::from_le_bytes(mmap[..8].try_into().unwrap()) as usize;
        let data_base = 8 + header_len;
        let headers = parse_header(&mmap)?;

        for (name, h) in &headers {
            if all.contains_key(name) {
                continue; // dedup across shards
            }
            let dtype = match DType::from_str(&h.dtype) {
                Some(d) => d,
                None => {
                    log::warn!("Skipping tensor '{}' with unsupported dtype '{}'", name, h.dtype);
                    continue;
                }
            };
            let [start, end] = h.data_offsets;
            let bytes = mmap[data_base + start..data_base + end].to_vec();
            all.insert(name.clone(), TensorData::new(bytes, dtype, h.shape.clone()));
        }
    }

    assemble_weights(config, all)
}

// ── Weight assembly ──────────────────────────────────────────────────────────

/// Pull named tensors out of the flat map and build `ModelWeights`.
fn assemble_weights(
    config: ModelConfig,
    mut map: HashMap<String, TensorData>,
) -> Result<ModelWeights, LoadError> {
    let take = |map: &mut HashMap<String, TensorData>, name: &str| -> Result<TensorData, WeightError> {
        map.remove(name)
            .ok_or_else(|| WeightError::Missing(name.to_string()))
    };

    let embed_tokens = take(&mut map, "model.embed_tokens.weight")?;
    let norm = take(&mut map, "model.norm.weight")?;

    // lm_head may be tied to embed_tokens (not present as a separate key).
    let lm_head = map
        .remove("lm_head.weight")
        .unwrap_or_else(|| embed_tokens.clone());

    let num_layers = config.num_hidden_layers;
    let mut layers = Vec::with_capacity(num_layers);

    for i in 0..num_layers {
        let p = |n: &str| format!("model.layers.{i}.{n}");
        let layer = LayerWeights {
            input_layernorm: take(&mut map, &p("input_layernorm.weight"))?,
            q_proj: take(&mut map, &p("self_attn.q_proj.weight"))?,
            k_proj: take(&mut map, &p("self_attn.k_proj.weight"))?,
            v_proj: take(&mut map, &p("self_attn.v_proj.weight"))?,
            o_proj: take(&mut map, &p("self_attn.o_proj.weight"))?,
            post_attention_layernorm: take(
                &mut map,
                &p("post_attention_layernorm.weight"),
            )?,
            gate_proj: take(&mut map, &p("mlp.gate_proj.weight"))?,
            up_proj: take(&mut map, &p("mlp.up_proj.weight"))?,
            down_proj: take(&mut map, &p("mlp.down_proj.weight"))?,
        };
        layers.push(layer);
    }

    log::info!(
        "Loaded {} layers, {} tensors remaining (unused)",
        layers.len(),
        map.len()
    );

    Ok(ModelWeights {
        config,
        embed_tokens,
        layers,
        norm,
        lm_head,
    })
}
