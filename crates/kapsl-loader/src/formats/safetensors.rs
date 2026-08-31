//! Safetensors weight loader.
//!
//! Each shard is mapped temporarily, validated with the upstream
//! `safetensors` parser, and copied into owned `TensorData` buffers. Returning
//! owned buffers keeps the model independent of the shard files after loading.

use memmap2::Mmap;
use safetensors::{Dtype as SafetensorsDtype, SafeTensors};
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

// ── Safetensors dtype mapping ────────────────────────────────────────────────

fn supported_dtype(dtype: SafetensorsDtype) -> Option<DType> {
    match dtype {
        SafetensorsDtype::F32 => Some(DType::F32),
        SafetensorsDtype::F16 => Some(DType::F16),
        SafetensorsDtype::BF16 => Some(DType::BF16),
        SafetensorsDtype::I8 => Some(DType::I8),
        SafetensorsDtype::U8 => Some(DType::U8),
        _ => None,
    }
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
        let tensors =
            SafeTensors::deserialize(&mmap).map_err(|error| LoadError::Parse(error.to_string()))?;

        for (name, view) in tensors.iter() {
            if all.contains_key(name) {
                continue; // dedup across shards
            }
            let dtype = match supported_dtype(view.dtype()) {
                Some(d) => d,
                None => {
                    log::warn!(
                        "Skipping tensor '{}' with unsupported dtype '{:?}'",
                        name,
                        view.dtype()
                    );
                    continue;
                }
            };
            all.insert(
                name.to_string(),
                TensorData::new(view.data().to_vec(), dtype, view.shape().to_vec()),
            );
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
    let take =
        |map: &mut HashMap<String, TensorData>, name: &str| -> Result<TensorData, WeightError> {
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
            post_attention_layernorm: take(&mut map, &p("post_attention_layernorm.weight"))?,
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

#[cfg(test)]
mod tests {
    use super::*;
    use safetensors::tensor::TensorView;
    use std::sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    };

    static NEXT_FIXTURE_ID: AtomicUsize = AtomicUsize::new(0);

    struct ModelFixture {
        path: std::path::PathBuf,
    }

    impl ModelFixture {
        fn new() -> Self {
            let id = NEXT_FIXTURE_ID.fetch_add(1, Ordering::Relaxed);
            let path =
                std::env::temp_dir().join(format!("kapsl-loader-test-{}-{id}", std::process::id()));
            std::fs::create_dir_all(&path).expect("create model fixture directory");
            std::fs::write(
                path.join("config.json"),
                r#"{
                    "hidden_size": 2,
                    "intermediate_size": 4,
                    "num_hidden_layers": 0,
                    "num_attention_heads": 1,
                    "vocab_size": 2
                }"#,
            )
            .expect("write model config");
            Self { path }
        }
    }

    impl Drop for ModelFixture {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.path);
        }
    }

    #[test]
    fn rejects_a_truncated_safetensors_file_without_panicking() {
        let fixture = ModelFixture::new();
        std::fs::write(fixture.path.join("model.safetensors"), [0_u8; 3])
            .expect("write truncated shard");

        assert!(matches!(
            load_safetensors(&fixture.path),
            Err(LoadError::Parse(_))
        ));
    }

    #[test]
    fn loads_valid_weights_and_ties_a_missing_lm_head() {
        let fixture = ModelFixture::new();
        let embeddings = [0_u8, 0, 0, 0, 0, 0, 0, 0];
        let norm = [0_u8, 0, 0, 0];
        let tensors = [
            (
                "model.embed_tokens.weight",
                TensorView::new(SafetensorsDtype::F16, vec![2, 2], &embeddings)
                    .expect("build embeddings view"),
            ),
            (
                "model.norm.weight",
                TensorView::new(SafetensorsDtype::F16, vec![2], &norm).expect("build norm view"),
            ),
        ];
        let serialized = safetensors::serialize(tensors, &None).expect("serialize model weights");
        std::fs::write(fixture.path.join("model.safetensors"), serialized)
            .expect("write model shard");

        let weights = load_safetensors(&fixture.path).expect("load model weights");

        assert_eq!(weights.num_layers(), 0);
        assert_eq!(weights.embed_tokens.shape, vec![2, 2]);
        assert!(Arc::ptr_eq(
            &weights.embed_tokens.bytes,
            &weights.lm_head.bytes
        ));
    }
}
