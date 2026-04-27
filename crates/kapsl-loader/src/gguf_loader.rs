//! GGUF file parser and weight dequantizer.
//!
//! Reads GGUF v1/v2/v3 and dequantizes all tensor data to f16, producing a
//! `ModelWeights` compatible with the native CUDA backend.
//!
//! Supported quantization types: F32, F16, BF16, Q4_0, Q4_1, Q8_0,
//! Q4_K, Q5_K, Q6_K.

use half::{bf16, f16};
use memmap2::Mmap;
use std::collections::HashMap;
use std::fs::File;
use std::path::Path;
use thiserror::Error;

use crate::config::ModelConfig;
use crate::weights::{DType, LayerWeights, ModelWeights, TensorData};

// ─── Error ────────────────────────────────────────────────────────────────────

#[derive(Debug, Error)]
pub enum GgufError {
    #[error("I/O: {0}")]
    Io(#[from] std::io::Error),
    #[error("Invalid GGUF magic — not a GGUF file")]
    BadMagic,
    #[error("Unsupported GGUF version {0}")]
    UnsupportedVersion(u32),
    #[error("Parse error at offset {offset}: {msg}")]
    Parse { offset: usize, msg: String },
    #[error("Missing required metadata key: {0}")]
    MissingKey(String),
    #[error("Unsupported quantization type id {0}")]
    UnsupportedQuant(u32),
    #[error("Missing tensor: {0}")]
    MissingTensor(String),
}

// ─── GGUF value type enum ─────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(u32)]
enum GgufValueType {
    U8 = 0, I8 = 1, U16 = 2, I16 = 3, U32 = 4, I32 = 5,
    F32 = 6, Bool = 7, Str = 8, Array = 9, U64 = 10, I64 = 11, F64 = 12,
}

impl GgufValueType {
    fn from_u32(v: u32) -> Option<Self> {
        match v {
            0 => Some(Self::U8),   1 => Some(Self::I8),  2 => Some(Self::U16),
            3 => Some(Self::I16),  4 => Some(Self::U32), 5 => Some(Self::I32),
            6 => Some(Self::F32),  7 => Some(Self::Bool),8 => Some(Self::Str),
            9 => Some(Self::Array),10 => Some(Self::U64),11 => Some(Self::I64),
            12 => Some(Self::F64), _ => None,
        }
    }
}

#[derive(Debug, Clone)]
enum GgufValue {
    U8(u8), I8(i8), U16(u16), I16(i16), U32(u32), I32(i32),
    F32(f32), Bool(bool), Str(String), Array(Vec<GgufValue>),
    U64(u64), I64(i64), F64(f64),
}

impl GgufValue {
    fn as_u64(&self) -> Option<u64> {
        match self {
            Self::U64(v) => Some(*v), Self::U32(v) => Some(*v as u64),
            Self::I32(v) if *v >= 0 => Some(*v as u64), _ => None,
        }
    }
    fn as_f32(&self) -> Option<f32> {
        match self {
            Self::F32(v) => Some(*v), Self::F64(v) => Some(*v as f32), _ => None,
        }
    }
    fn as_str(&self) -> Option<&str> {
        match self { Self::Str(s) => Some(s), _ => None }
    }
    fn array_len(&self) -> Option<usize> {
        match self { Self::Array(v) => Some(v.len()), _ => None }
    }
}

// ─── GGML quantization type ───────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(u32)]
enum GgmlType {
    F32  = 0,  F16  = 1,
    Q4_0 = 2,  Q4_1 = 3,
    Q5_0 = 6,  Q5_1 = 7,
    Q8_0 = 8,  Q8_1 = 9,
    Q2K  = 10, Q3K  = 11, Q4K  = 12, Q5K  = 13, Q6K  = 14, Q8K = 15,
    BF16 = 30,
}

impl GgmlType {
    fn from_u32(v: u32) -> Option<Self> {
        match v {
            0  => Some(Self::F32),  1  => Some(Self::F16),
            2  => Some(Self::Q4_0), 3  => Some(Self::Q4_1),
            6  => Some(Self::Q5_0), 7  => Some(Self::Q5_1),
            8  => Some(Self::Q8_0), 9  => Some(Self::Q8_1),
            10 => Some(Self::Q2K),  11 => Some(Self::Q3K),
            12 => Some(Self::Q4K),  13 => Some(Self::Q5K),
            14 => Some(Self::Q6K),  15 => Some(Self::Q8K),
            30 => Some(Self::BF16), _ => None,
        }
    }

    /// (elements_per_block, bytes_per_block)
    fn block_params(self) -> (usize, usize) {
        match self {
            Self::F32  => (1, 4),  Self::F16  => (1, 2),  Self::BF16 => (1, 2),
            Self::Q4_0 => (32, 18),Self::Q4_1 => (32, 20),
            Self::Q5_0 => (32, 22),Self::Q5_1 => (32, 24),
            Self::Q8_0 => (32, 34),Self::Q8_1 => (32, 36),
            Self::Q2K  => (256, 84), Self::Q3K => (256, 110),
            Self::Q4K  => (256, 144),Self::Q5K  => (256, 176),
            Self::Q6K  => (256, 210),Self::Q8K  => (256, 292),
        }
    }
}

// ─── Binary reader ────────────────────────────────────────────────────────────

struct Reader<'a> { data: &'a [u8], pos: usize }

impl<'a> Reader<'a> {
    fn new(data: &'a [u8]) -> Self { Self { data, pos: 0 } }

    fn read_bytes(&mut self, n: usize) -> Result<&'a [u8], GgufError> {
        if self.pos + n > self.data.len() {
            return Err(GgufError::Parse {
                offset: self.pos,
                msg: format!("need {n} bytes, {} remain", self.data.len() - self.pos),
            });
        }
        let s = &self.data[self.pos..self.pos + n];
        self.pos += n;
        Ok(s)
    }

    fn u8(&mut self)  -> Result<u8,  GgufError> { Ok(self.read_bytes(1)?[0]) }
    fn i8(&mut self)  -> Result<i8,  GgufError> { Ok(self.u8()? as i8) }
    fn u16(&mut self) -> Result<u16, GgufError> { Ok(u16::from_le_bytes(self.read_bytes(2)?.try_into().unwrap())) }
    fn i16(&mut self) -> Result<i16, GgufError> { Ok(self.u16()? as i16) }
    fn u32(&mut self) -> Result<u32, GgufError> { Ok(u32::from_le_bytes(self.read_bytes(4)?.try_into().unwrap())) }
    fn i32(&mut self) -> Result<i32, GgufError> { Ok(self.u32()? as i32) }
    fn u64(&mut self) -> Result<u64, GgufError> { Ok(u64::from_le_bytes(self.read_bytes(8)?.try_into().unwrap())) }
    fn i64(&mut self) -> Result<i64, GgufError> { Ok(self.u64()? as i64) }
    fn f32(&mut self) -> Result<f32, GgufError> { Ok(f32::from_le_bytes(self.read_bytes(4)?.try_into().unwrap())) }
    fn f64(&mut self) -> Result<f64, GgufError> { Ok(f64::from_le_bytes(self.read_bytes(8)?.try_into().unwrap())) }

    fn string(&mut self) -> Result<String, GgufError> {
        let len = self.u64()? as usize;
        let bytes = self.read_bytes(len)?;
        String::from_utf8(bytes.to_vec())
            .map_err(|e| GgufError::Parse { offset: self.pos, msg: format!("non-UTF8 string: {e}") })
    }

    fn value(&mut self, vtype: GgufValueType) -> Result<GgufValue, GgufError> {
        match vtype {
            GgufValueType::U8   => Ok(GgufValue::U8(self.u8()?)),
            GgufValueType::I8   => Ok(GgufValue::I8(self.i8()?)),
            GgufValueType::U16  => Ok(GgufValue::U16(self.u16()?)),
            GgufValueType::I16  => Ok(GgufValue::I16(self.i16()?)),
            GgufValueType::U32  => Ok(GgufValue::U32(self.u32()?)),
            GgufValueType::I32  => Ok(GgufValue::I32(self.i32()?)),
            GgufValueType::F32  => Ok(GgufValue::F32(self.f32()?)),
            GgufValueType::Bool => Ok(GgufValue::Bool(self.u8()? != 0)),
            GgufValueType::Str  => Ok(GgufValue::Str(self.string()?)),
            GgufValueType::U64  => Ok(GgufValue::U64(self.u64()?)),
            GgufValueType::I64  => Ok(GgufValue::I64(self.i64()?)),
            GgufValueType::F64  => Ok(GgufValue::F64(self.f64()?)),
            GgufValueType::Array => {
                let elem_type_raw = self.u32()?;
                let elem_type = GgufValueType::from_u32(elem_type_raw)
                    .ok_or_else(|| GgufError::Parse { offset: self.pos,
                        msg: format!("unknown array element type {elem_type_raw}") })?;
                let count = self.u64()? as usize;
                let mut items = Vec::with_capacity(count.min(1 << 20));
                for _ in 0..count {
                    items.push(self.value(elem_type)?);
                }
                Ok(GgufValue::Array(items))
            }
        }
    }
}

// ─── Tensor info ──────────────────────────────────────────────────────────────

struct TensorInfo {
    /// Shape in PyTorch convention [out, in, ...] — reversed from GGUF storage
    shape: Vec<usize>,
    dtype: GgmlType,
    /// Byte offset from start of the data section
    offset: u64,
    numel: usize,
}

// ─── GgufFile ─────────────────────────────────────────────────────────────────

pub struct GgufFile {
    mmap: Mmap,
    metadata: HashMap<String, GgufValue>,
    tensors: HashMap<String, TensorInfo>,
    data_start: usize,
}

impl GgufFile {
    pub fn open(path: &Path) -> Result<Self, GgufError> {
        let file = File::open(path)?;
        // SAFETY: we hold a reference to `file` for the duration of the mmap.
        let mmap = unsafe { Mmap::map(&file)? };

        let mut r = Reader::new(&mmap);

        if r.read_bytes(4)? != b"GGUF" {
            return Err(GgufError::BadMagic);
        }
        let version = r.u32()?;
        if !(1..=3).contains(&version) {
            return Err(GgufError::UnsupportedVersion(version));
        }
        let tensor_count = r.u64()? as usize;
        let kv_count = r.u64()? as usize;

        let mut metadata = HashMap::with_capacity(kv_count);
        for _ in 0..kv_count {
            let key = r.string()?;
            let vtype_raw = r.u32()?;
            let vtype = GgufValueType::from_u32(vtype_raw)
                .ok_or_else(|| GgufError::Parse { offset: r.pos,
                    msg: format!("unknown value type {vtype_raw}") })?;
            let val = r.value(vtype)?;
            metadata.insert(key, val);
        }

        let mut tensors = HashMap::with_capacity(tensor_count);
        for _ in 0..tensor_count {
            let name = r.string()?;
            let n_dims = r.u32()? as usize;
            let mut dims_gguf = Vec::with_capacity(n_dims);
            for _ in 0..n_dims { dims_gguf.push(r.u64()? as usize); }

            let dtype_raw = r.u32()?;
            let dtype = GgmlType::from_u32(dtype_raw)
                .ok_or(GgufError::UnsupportedQuant(dtype_raw))?;
            let offset = r.u64()?;

            // GGUF stores dims innermost-first; reverse for PyTorch convention.
            let shape: Vec<usize> = dims_gguf.into_iter().rev().collect();
            let numel: usize = shape.iter().product();

            tensors.insert(name, TensorInfo { shape, dtype, offset, numel });
        }

        // Data section is 32-byte aligned after all tensor infos.
        let data_start = (r.pos + 31) & !31;

        Ok(Self { mmap, metadata, tensors, data_start })
    }

    // ── Metadata helpers ─────────────────────────────────────────────────────

    fn meta_u64(&self, key: &str) -> Result<u64, GgufError> {
        self.metadata.get(key)
            .ok_or_else(|| GgufError::MissingKey(key.into()))?
            .as_u64()
            .ok_or_else(|| GgufError::Parse { offset: 0,
                msg: format!("key '{key}' is not a uint") })
    }
    fn meta_u32(&self, key: &str) -> Result<u32, GgufError> {
        self.meta_u64(key).map(|v| v as u32)
    }
    fn meta_f32(&self, key: &str) -> Option<f32> {
        self.metadata.get(key)?.as_f32()
    }
    fn meta_str(&self, key: &str) -> Option<&str> {
        self.metadata.get(key)?.as_str()
    }

    // ── Config extraction ────────────────────────────────────────────────────

    pub fn extract_config(&self) -> Result<ModelConfig, GgufError> {
        let arch = self.meta_str("general.architecture").unwrap_or("llama");

        let hidden_size        = self.meta_u32(&format!("{arch}.embedding_length"))? as usize;
        let intermediate_size  = self.meta_u32(&format!("{arch}.feed_forward_length"))? as usize;
        let num_hidden_layers  = self.meta_u32(&format!("{arch}.block_count"))? as usize;
        let num_attention_heads= self.meta_u32(&format!("{arch}.attention.head_count"))? as usize;
        let num_kv_heads = self.meta_u32(&format!("{arch}.attention.head_count_kv"))
            .ok().map(|v| v as usize);

        let vocab_size = if let Ok(v) = self.meta_u32(&format!("{arch}.vocab_size")) {
            v as usize
        } else if let Some(n) = self.metadata.get("tokenizer.ggml.tokens").and_then(|v| v.array_len()) {
            n
        } else {
            return Err(GgufError::MissingKey(format!("{arch}.vocab_size")));
        };

        let rms_norm_eps = self.meta_f32(&format!("{arch}.attention.layer_norm_rms_epsilon"))
            .unwrap_or(1e-5) as f64;
        let rope_theta = self.meta_f32(&format!("{arch}.rope.freq_base"))
            .unwrap_or(10_000.0) as f64;
        let max_position_embeddings = self.meta_u32(&format!("{arch}.context_length"))
            .unwrap_or(4096) as usize;

        Ok(ModelConfig {
            hidden_size,
            intermediate_size,
            num_hidden_layers,
            num_attention_heads,
            num_key_value_heads: num_kv_heads,
            vocab_size,
            rms_norm_eps,
            rope_theta,
            max_position_embeddings,
            architectures: vec![arch.to_string()],
        })
    }

    // ── Tensor dequantization ────────────────────────────────────────────────

    fn dequantize(&self, info: &TensorInfo) -> Result<Vec<u8>, GgufError> {
        let start = self.data_start + info.offset as usize;
        let raw = &self.mmap[start..];
        let n = info.numel;

        let f16s: Vec<f16> = match info.dtype {
            GgmlType::F32  => deq_f32(raw, n),
            GgmlType::F16  => deq_f16(raw, n),
            GgmlType::BF16 => deq_bf16(raw, n),
            GgmlType::Q4_0 => deq_q4_0(raw, n),
            GgmlType::Q4_1 => deq_q4_1(raw, n),
            GgmlType::Q8_0 => deq_q8_0(raw, n),
            GgmlType::Q4K  => deq_q4_k(raw, n),
            GgmlType::Q5K  => deq_q5_k(raw, n),
            GgmlType::Q6K  => deq_q6_k(raw, n),
            t => return Err(GgufError::UnsupportedQuant(t as u32)),
        };

        let mut bytes = vec![0u8; f16s.len() * 2];
        for (i, v) in f16s.iter().enumerate() {
            let le = v.to_bits().to_le_bytes();
            bytes[i * 2]     = le[0];
            bytes[i * 2 + 1] = le[1];
        }
        Ok(bytes)
    }

    fn tensor(&self, name: &str) -> Result<TensorData, GgufError> {
        let info = self.tensors.get(name)
            .ok_or_else(|| GgufError::MissingTensor(name.into()))?;
        let bytes = self.dequantize(info)?;
        Ok(TensorData::new(bytes, DType::F16, info.shape.clone()))
    }

    // ── Weight assembly ──────────────────────────────────────────────────────

    pub fn load_weights(&self, config: &ModelConfig) -> Result<ModelWeights, GgufError> {
        let n = config.num_hidden_layers;

        let embed_tokens = self.tensor("token_embd.weight")?;
        let norm         = self.tensor("output_norm.weight")?;
        // Some models tie lm_head with embed_tokens (output.weight absent).
        let lm_head = self.tensor("output.weight").unwrap_or_else(|_| embed_tokens.clone());

        let mut layers = Vec::with_capacity(n);
        for i in 0..n {
            log::info!("[gguf_loader] dequantizing layer {}/{n}", i + 1);
            layers.push(LayerWeights {
                input_layernorm:          self.tensor(&format!("blk.{i}.attn_norm.weight"))?,
                q_proj:                   self.tensor(&format!("blk.{i}.attn_q.weight"))?,
                k_proj:                   self.tensor(&format!("blk.{i}.attn_k.weight"))?,
                v_proj:                   self.tensor(&format!("blk.{i}.attn_v.weight"))?,
                o_proj:                   self.tensor(&format!("blk.{i}.attn_output.weight"))?,
                post_attention_layernorm: self.tensor(&format!("blk.{i}.ffn_norm.weight"))?,
                gate_proj:                self.tensor(&format!("blk.{i}.ffn_gate.weight"))?,
                up_proj:                  self.tensor(&format!("blk.{i}.ffn_up.weight"))?,
                down_proj:                self.tensor(&format!("blk.{i}.ffn_down.weight"))?,
            });
        }

        Ok(ModelWeights { config: config.clone(), embed_tokens, layers, norm, lm_head })
    }
}

/// Open a GGUF file, extract config, dequantize all weights to f16.
pub fn load_gguf_weights(path: &Path) -> Result<ModelWeights, GgufError> {
    let gguf = GgufFile::open(path)?;
    let config = gguf.extract_config()?;
    gguf.load_weights(&config)
}

// ─── Dequantization implementations ──────────────────────────────────────────

fn deq_f32(data: &[u8], n: usize) -> Vec<f16> {
    data.chunks_exact(4).take(n)
        .map(|b| f16::from_f32(f32::from_le_bytes(b.try_into().unwrap())))
        .collect()
}

fn deq_f16(data: &[u8], n: usize) -> Vec<f16> {
    data.chunks_exact(2).take(n)
        .map(|b| f16::from_bits(u16::from_le_bytes(b.try_into().unwrap())))
        .collect()
}

fn deq_bf16(data: &[u8], n: usize) -> Vec<f16> {
    data.chunks_exact(2).take(n)
        .map(|b| f16::from_f32(bf16::from_bits(u16::from_le_bytes(b.try_into().unwrap())).to_f32()))
        .collect()
}

// Q4_0: [f16 d, u8 qs[16]]  per 32 values
fn deq_q4_0(data: &[u8], n: usize) -> Vec<f16> {
    const B: usize = 32; const BY: usize = 18;
    let mut out = Vec::with_capacity(n);
    for blk in data.chunks_exact(BY).take(n / B) {
        let d = f16::from_bits(u16::from_le_bytes([blk[0], blk[1]])).to_f32();
        for &q in &blk[2..18] {
            out.push(f16::from_f32(((q & 0xf) as i8 - 8) as f32 * d));
            out.push(f16::from_f32(((q >> 4)  as i8 - 8) as f32 * d));
        }
    }
    out
}

// Q4_1: [f16 d, f16 m, u8 qs[16]]  per 32 values
fn deq_q4_1(data: &[u8], n: usize) -> Vec<f16> {
    const B: usize = 32; const BY: usize = 20;
    let mut out = Vec::with_capacity(n);
    for blk in data.chunks_exact(BY).take(n / B) {
        let d = f16::from_bits(u16::from_le_bytes([blk[0], blk[1]])).to_f32();
        let m = f16::from_bits(u16::from_le_bytes([blk[2], blk[3]])).to_f32();
        for &q in &blk[4..20] {
            out.push(f16::from_f32((q & 0xf) as f32 * d + m));
            out.push(f16::from_f32((q >> 4)  as f32 * d + m));
        }
    }
    out
}

// Q8_0: [f16 d, i8 qs[32]]  per 32 values
fn deq_q8_0(data: &[u8], n: usize) -> Vec<f16> {
    const B: usize = 32; const BY: usize = 34;
    let mut out = Vec::with_capacity(n);
    for blk in data.chunks_exact(BY).take(n / B) {
        let d = f16::from_bits(u16::from_le_bytes([blk[0], blk[1]])).to_f32();
        for &q in &blk[2..34] {
            out.push(f16::from_f32(q as i8 as f32 * d));
        }
    }
    out
}

// Q4_K: [f16 d, f16 dmin, u8 scales[12], u8 qs[128]]  per 256 values
fn deq_q4_k(data: &[u8], n: usize) -> Vec<f16> {
    const QK: usize = 256; const BY: usize = 144;
    let mut out = Vec::with_capacity(n);
    for blk in data.chunks_exact(BY).take(n / QK) {
        let d    = f16::from_bits(u16::from_le_bytes([blk[0], blk[1]])).to_f32();
        let dmin = f16::from_bits(u16::from_le_bytes([blk[2], blk[3]])).to_f32();
        let sc   = &blk[4..16];
        let qs   = &blk[16..144];

        let start = out.len();
        out.resize(start + QK, f16::ZERO);
        let o = &mut out[start..];

        let mut is = 0usize;
        let mut qi = 0usize;
        let mut oi = 0usize;
        for _ in (0..QK).step_by(64) {
            let (s0, m0) = scale_min_k4(is,     sc);
            let (s1, m1) = scale_min_k4(is + 1, sc);
            let d1 = d * s0 as f32; let m1v = dmin * m0 as f32;
            let d2 = d * s1 as f32; let m2v = dmin * m1 as f32;
            for l in 0..32 {
                o[oi + l]      = f16::from_f32(d1 * (qs[qi + l] & 0xf) as f32 - m1v);
                o[oi + 32 + l] = f16::from_f32(d2 * (qs[qi + l] >> 4)  as f32 - m2v);
            }
            qi += 32; is += 2; oi += 64;
        }
    }
    out
}

// Q5_K: [f16 d, f16 dmin, u8 scales[12], u8 qh[32], u8 qs[128]]  per 256 values
fn deq_q5_k(data: &[u8], n: usize) -> Vec<f16> {
    const QK: usize = 256; const BY: usize = 176;
    let mut out = Vec::with_capacity(n);
    for blk in data.chunks_exact(BY).take(n / QK) {
        let d    = f16::from_bits(u16::from_le_bytes([blk[0], blk[1]])).to_f32();
        let dmin = f16::from_bits(u16::from_le_bytes([blk[2], blk[3]])).to_f32();
        let sc   = &blk[4..16];
        let qh   = &blk[16..48];
        let qs   = &blk[48..176];

        let start = out.len();
        out.resize(start + QK, f16::ZERO);
        let o = &mut out[start..];

        let mut is = 0usize;
        let mut qi = 0usize;
        let mut oi = 0usize;
        let mut u1 = 1u8;
        let mut u2 = 2u8;
        for _ in (0..QK).step_by(64) {
            let (s0, m0) = scale_min_k4(is,     sc);
            let (s1, m1) = scale_min_k4(is + 1, sc);
            let d1 = d * s0 as f32; let m1v = dmin * m0 as f32;
            let d2 = d * s1 as f32; let m2v = dmin * m1 as f32;
            for l in 0..32 {
                let hi0 = if qh[l] & u1 != 0 { 16.0f32 } else { 0.0 };
                let hi1 = if qh[l] & u2 != 0 { 16.0f32 } else { 0.0 };
                o[oi + l]      = f16::from_f32(d1 * ((qs[qi + l] & 0xf) as f32 + hi0) - m1v);
                o[oi + 32 + l] = f16::from_f32(d2 * ((qs[qi + l] >> 4)  as f32 + hi1) - m2v);
            }
            qi += 32; is += 2; oi += 64;
            u1 = u1.wrapping_shl(2);
            u2 = u2.wrapping_shl(2);
        }
    }
    out
}

// Q6_K: [u8 ql[128], u8 qh[64], i8 scales[16], f16 d]  per 256 values
fn deq_q6_k(data: &[u8], n: usize) -> Vec<f16> {
    const QK: usize = 256; const BY: usize = 210;
    let mut out = Vec::with_capacity(n);
    for blk in data.chunks_exact(BY).take(n / QK) {
        let ql = &blk[0..128];
        let qh = &blk[128..192];
        let sc = &blk[192..208];   // i8 scales
        let d  = f16::from_bits(u16::from_le_bytes([blk[208], blk[209]])).to_f32();

        let start = out.len();
        out.resize(start + QK, f16::ZERO);
        let o = &mut out[start..];

        let mut ib = 0usize;
        let mut qli = 0usize;
        let mut qhi = 0usize;
        let mut oi  = 0usize;
        for _ in (0..QK).step_by(128) {
            for l in 0..32 {
                let q1 = ((ql[qli+l]    & 0xf) | ((qh[qhi+l] & 0x03) << 4)) as i8 - 32;
                let q2 = ((ql[qli+l+32] & 0xf) | (((qh[qhi+l]>>2) & 0x03) << 4)) as i8 - 32;
                let q3 = ((ql[qli+l]    >> 4)  | (((qh[qhi+l]>>4) & 0x03) << 4)) as i8 - 32;
                let q4 = ((ql[qli+l+32] >> 4)  | (((qh[qhi+l]>>6) & 0x03) << 4)) as i8 - 32;
                o[oi+l]    = f16::from_f32(d * sc[ib]   as i8 as f32 * q1 as f32);
                o[oi+32+l] = f16::from_f32(d * sc[ib+1] as i8 as f32 * q2 as f32);
                o[oi+64+l] = f16::from_f32(d * sc[ib+2] as i8 as f32 * q3 as f32);
                o[oi+96+l] = f16::from_f32(d * sc[ib+3] as i8 as f32 * q4 as f32);
            }
            qli += 64; qhi += 32; ib += 4; oi += 128;
        }
    }
    out
}

// ─── K-quantization scale/min helper ─────────────────────────────────────────
// Matches ggml-quants.c: get_scale_min_k4

#[inline(always)]
fn scale_min_k4(j: usize, q: &[u8]) -> (u8, u8) {
    if j < 4 {
        (q[j] & 63, q[j + 4] & 63)
    } else {
        (
            (q[j + 4] & 0x0f) | ((q[j - 4] >> 6) << 4),
            (q[j + 4] >> 4)   | ((q[j]     >> 6) << 4),
        )
    }
}
