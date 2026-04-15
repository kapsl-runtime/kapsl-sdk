//! TurboQuant KV-cache quantization.
//!
//! Wraps [`turboquant_rs::QuantizedKVCache`] with a configuration type and
//! error handling that matches the rest of `kapsl-quantization`.
//!
//! # Example
//!
//! ```rust
//! use kapsl_quantization::kv_cache::{KvCacheConfig, KvCacheQuantizer};
//!
//! let config = KvCacheConfig::new(3, 128, 32).unwrap();
//! let mut cache = KvCacheQuantizer::new(config, 42);
//!
//! let key = vec![0.1f32; 128];
//! let value = vec![0.2f32; 128];
//! cache.push(0, &key, &value).unwrap();
//!
//! let query = vec![0.1f32; 128];
//! let scores = cache.attention_scores(0, &query).unwrap();
//! ```

use anyhow::{anyhow, Result};
use half::f16;
use turboquant_rs::{QuantizedKVCache, TurboQuantConfig};

// ---------------------------------------------------------------------------
// KvCacheConfig
// ---------------------------------------------------------------------------

/// Configuration for the TurboQuant KV-cache.
#[derive(Debug, Clone, Copy)]
pub struct KvCacheConfig {
    /// Bits per value: 2, 3, or 4.
    pub bits: u8,
    /// Head dimension. Must be a power of two (e.g. 64, 128, 256).
    pub head_dim: usize,
    /// Number of transformer layers.
    pub num_layers: usize,
}

impl KvCacheConfig {
    /// Create a new config, returning an error if `bits` or `head_dim` are invalid.
    pub fn new(bits: u8, head_dim: usize, num_layers: usize) -> Result<Self> {
        if !matches!(bits, 2 | 3 | 4) {
            return Err(anyhow!("bits must be 2, 3, or 4; got {bits}"));
        }
        if head_dim == 0 || !head_dim.is_power_of_two() {
            return Err(anyhow!(
                "head_dim must be a non-zero power of two; got {head_dim}"
            ));
        }
        if num_layers == 0 {
            return Err(anyhow!("num_layers must be > 0"));
        }
        Ok(Self { bits, head_dim, num_layers })
    }
}

// ---------------------------------------------------------------------------
// KvCacheQuantizer
// ---------------------------------------------------------------------------

/// A quantized KV-cache backed by TurboQuant (PolarQuant + QJL).
///
/// Keys are stored as QJL-quantized blocks enabling fast approximate inner
/// products. Values are dequantized on demand for weighted accumulation.
pub struct KvCacheQuantizer {
    inner: QuantizedKVCache,
    config: KvCacheConfig,
}

impl KvCacheQuantizer {
    /// Create a new quantizer.
    ///
    /// `rotation_seed` controls the PolarQuant sign pattern; `qjl_seed`
    /// controls the Rademacher projection matrix. Using fixed seeds gives
    /// reproducible results.
    pub fn new(config: KvCacheConfig, qjl_seed: u64) -> Self {
        let tq_config = TurboQuantConfig::new(config.bits, config.head_dim)
            .expect("KvCacheConfig already validated bits and head_dim");
        Self {
            inner: QuantizedKVCache::new(tq_config, config.num_layers, qjl_seed),
            config,
        }
    }

    /// Create a new quantizer with an explicit rotation seed.
    pub fn with_rotation_seed(config: KvCacheConfig, rotation_seed: u64, qjl_seed: u64) -> Self {
        let tq_config = TurboQuantConfig::new(config.bits, config.head_dim)
            .expect("KvCacheConfig already validated bits and head_dim")
            .with_seed(rotation_seed);
        Self {
            inner: QuantizedKVCache::new(tq_config, config.num_layers, qjl_seed),
            config,
        }
    }

    /// Append a single key-value pair to `layer`.
    pub fn push(&mut self, layer: usize, key: &[f32], value: &[f32]) -> Result<()> {
        self.inner
            .push(layer, key, value)
            .map_err(|e| anyhow!("kv_cache push failed: {e}"))
    }

    /// Append a batch of key-value pairs to `layer`.
    pub fn push_batch(
        &mut self,
        layer: usize,
        keys: &[&[f32]],
        values: &[&[f32]],
    ) -> Result<()> {
        self.inner
            .push_batch(layer, keys, values)
            .map_err(|e| anyhow!("kv_cache push_batch failed: {e}"))
    }

    /// Compute approximate attention scores for `query` against all stored
    /// keys in `layer`. Returns one score per cached position.
    pub fn attention_scores(&self, layer: usize, query: &[f32]) -> Result<Vec<f32>> {
        self.inner
            .attention_scores(layer, query)
            .map_err(|e| anyhow!("kv_cache attention_scores failed: {e}"))
    }

    /// Number of cached positions in `layer`.
    pub fn entry_count(&self, layer: usize) -> usize {
        self.inner.entry_count(layer)
    }

    /// Bytes used by quantized data across all layers.
    pub fn memory_usage(&self) -> usize {
        self.inner.memory_usage()
    }

    /// Equivalent FP16 memory if the cache were uncompressed.
    pub fn fp16_equivalent_memory(&self) -> usize {
        self.inner.fp16_equivalent_memory()
    }

    /// Compression ratio vs uncompressed FP16.
    pub fn compression_ratio(&self) -> f64 {
        let quantized = self.memory_usage();
        if quantized == 0 {
            return 1.0;
        }
        self.fp16_equivalent_memory() as f64 / quantized as f64
    }

    /// Remove all cached entries across every layer by recreating the inner cache.
    pub fn clear(&mut self) {
        let tq_config = TurboQuantConfig::new(self.config.bits, self.config.head_dim)
            .expect("config already validated");
        let qjl_seed = self.inner.qjl_seed();
        self.inner = QuantizedKVCache::new(tq_config, self.config.num_layers, qjl_seed);
    }

    /// Return the configuration this quantizer was created with.
    pub fn config(&self) -> KvCacheConfig {
        self.config
    }

    /// Push a packed f16 KV layer (head-first layout: `[num_heads, length, head_dim]`)
    /// into `layer`, splitting into per-position vectors for TurboQuant.
    ///
    /// Called from the engine's `set_packed_layer` path when TQ compression is enabled.
    pub fn push_packed_layer(
        &mut self,
        layer: usize,
        length: usize,
        num_heads: usize,
        head_dim: usize,
        key_f16: &[f16],
        value_f16: &[f16],
    ) -> Result<()> {
        // Convert f16 head-first layout → per-position f32 vectors of dim num_heads*head_dim
        let total_dim = num_heads * head_dim;
        let mut keys: Vec<Vec<f32>> = Vec::with_capacity(length);
        let mut values: Vec<Vec<f32>> = Vec::with_capacity(length);

        for pos in 0..length {
            let mut k = vec![0.0f32; total_dim];
            let mut v = vec![0.0f32; total_dim];
            for h in 0..num_heads {
                let src = h * length * head_dim + pos * head_dim;
                let dst = h * head_dim;
                for d in 0..head_dim {
                    k[dst + d] = key_f16[src + d].to_f32();
                    v[dst + d] = value_f16[src + d].to_f32();
                }
            }
            keys.push(k);
            values.push(v);
        }

        let key_refs: Vec<&[f32]> = keys.iter().map(|k| k.as_slice()).collect();
        let val_refs: Vec<&[f32]> = values.iter().map(|v| v.as_slice()).collect();
        self.push_batch(layer, &key_refs, &val_refs)
    }

    /// Decompress all stored KV entries for `layer` back to packed f16 head-first layout
    /// `[num_heads, entry_count, head_dim]`. Returns `(key_f16, value_f16, length)`.
    ///
    /// Called from the engine's `get_packed_layer` path when TQ compression is enabled.
    pub fn dequantize_packed_layer(
        &self,
        layer: usize,
        num_heads: usize,
        head_dim: usize,
    ) -> Result<(Vec<f16>, Vec<f16>, usize)> {
        let keys_f32 = self
            .inner
            .dequantize_all_keys(layer)
            .map_err(|e| anyhow!("dequantize keys failed: {e}"))?;
        let values_f32 = self
            .inner
            .dequantize_all_values(layer)
            .map_err(|e| anyhow!("dequantize values failed: {e}"))?;

        let length = keys_f32.len();
        let total = num_heads * length * head_dim;
        let mut key_out = vec![f16::ZERO; total];
        let mut val_out = vec![f16::ZERO; total];

        // Convert per-position f32 vectors → head-first f16 layout
        for (pos, (k, v)) in keys_f32.iter().zip(values_f32.iter()).enumerate() {
            for h in 0..num_heads {
                let src = h * head_dim;
                let dst = h * length * head_dim + pos * head_dim;
                for d in 0..head_dim {
                    key_out[dst + d] = f16::from_f32(k[src + d]);
                    val_out[dst + d] = f16::from_f32(v[src + d]);
                }
            }
        }

        Ok((key_out, val_out, length))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const BITS: u8 = 3;
    const DIM: usize = 128;
    const LAYERS: usize = 2;

    fn make_cache() -> KvCacheQuantizer {
        let config = KvCacheConfig::new(BITS, DIM, LAYERS).unwrap();
        KvCacheQuantizer::new(config, 42)
    }

    #[test]
    fn config_rejects_invalid_bits() {
        assert!(KvCacheConfig::new(1, DIM, LAYERS).is_err());
        assert!(KvCacheConfig::new(5, DIM, LAYERS).is_err());
    }

    #[test]
    fn config_rejects_non_power_of_two_dim() {
        assert!(KvCacheConfig::new(BITS, 100, LAYERS).is_err());
        assert!(KvCacheConfig::new(BITS, 0, LAYERS).is_err());
    }

    #[test]
    fn config_rejects_zero_layers() {
        assert!(KvCacheConfig::new(BITS, DIM, 0).is_err());
    }

    #[test]
    fn push_and_entry_count() {
        let mut cache = make_cache();
        let key = vec![0.1f32; DIM];
        let value = vec![0.2f32; DIM];
        cache.push(0, &key, &value).unwrap();
        assert_eq!(cache.entry_count(0), 1);
        assert_eq!(cache.entry_count(1), 0);
    }

    #[test]
    fn push_batch_and_entry_count() {
        let mut cache = make_cache();
        let keys: Vec<Vec<f32>> = (0..8).map(|_| vec![0.1f32; DIM]).collect();
        let values: Vec<Vec<f32>> = (0..8).map(|_| vec![0.2f32; DIM]).collect();
        let key_refs: Vec<&[f32]> = keys.iter().map(|k| k.as_slice()).collect();
        let val_refs: Vec<&[f32]> = values.iter().map(|v| v.as_slice()).collect();
        cache.push_batch(0, &key_refs, &val_refs).unwrap();
        assert_eq!(cache.entry_count(0), 8);
    }

    #[test]
    fn attention_scores_length_matches_entries() {
        let mut cache = make_cache();
        for i in 0..16 {
            let key: Vec<f32> = (0..DIM).map(|j| (i * j) as f32 * 0.01).collect();
            let value = vec![0.5f32; DIM];
            cache.push(0, &key, &value).unwrap();
        }
        let query = vec![1.0f32; DIM];
        let scores = cache.attention_scores(0, &query).unwrap();
        assert_eq!(scores.len(), 16);
    }

    #[test]
    fn clear_resets_all_layers() {
        let mut cache = make_cache();
        let key = vec![0.0f32; DIM];
        let value = vec![0.0f32; DIM];
        cache.push(0, &key, &value).unwrap();
        cache.push(1, &key, &value).unwrap();
        cache.clear();
        assert_eq!(cache.entry_count(0), 0);
        assert_eq!(cache.entry_count(1), 0);
    }

    #[test]
    fn memory_usage_grows_with_entries() {
        let mut cache = make_cache();
        let before = cache.memory_usage();
        let key = vec![0.1f32; DIM];
        let value = vec![0.2f32; DIM];
        cache.push(0, &key, &value).unwrap();
        assert!(cache.memory_usage() > before);
    }

    #[test]
    fn compression_ratio_is_above_one() {
        let mut cache = make_cache();
        let key = vec![0.1f32; DIM];
        let value = vec![0.2f32; DIM];
        cache.push(0, &key, &value).unwrap();
        assert!(cache.compression_ratio() > 1.0);
    }
}
