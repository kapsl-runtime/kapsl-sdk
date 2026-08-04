//! GGUF native backend — GGUF weight loading + llama.cpp tokenizer + custom CUDA kernels.
//!
//! Differs from `GgufBackend` (which delegates all compute to llama.cpp):
//! llama.cpp is used **only** for tokenization; the full transformer forward pass
//! runs through kapsl's own paged-attention CUDA kernels, exactly as `NativeBackend`
//! does for safetensors models.
//!
//! Input / output contract (matches `GgufBackend`):
//!   input  — `dtype = Uint8`, shape `[len]`, data = UTF-8 prompt bytes
//!   output — `dtype = Uint8`, shape `[len]`, data = UTF-8 generated text bytes

#[cfg(feature = "gguf-native")]
mod inner {
    use std::collections::HashMap;
    use std::path::Path;
    use std::sync::{Arc, Mutex, OnceLock};

    use async_stream::stream;
    use async_trait::async_trait;
    use cudarc::cublas::{CudaBlas, Gemm};
    use cudarc::driver::{CudaDevice, CudaSlice, DeviceSlice};
    use half::f16;
    use llama_cpp_2::{
        llama_backend::LlamaBackend,
        model::{params::LlamaModelParams, AddBos, LlamaModel, Special},
    };
    use rand::{Rng, SeedableRng};

    use kapsl_engine_api::{
        BatchingPolicy, BinaryTensorPacket, Engine, EngineError, EngineMetrics, EngineModelInfo,
        EngineStream, InferenceRequest, RequestMetadata, TensorDtype,
    };
    use kapsl_hal::gpu_arena::{GpuBlockPool, GpuPoolHandle};
    use kapsl_loader::weights::DType;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use kapsl_kernels::cuda_quant_kernels::{launch_q4_k_gemv, launch_q8_0_gemv, QuantGemvParams};
    use kapsl_kernels::cuda_kernels::{
        launch_argmax, launch_batch_argmax, launch_batch_decode_rope, launch_batch_kv_write,
        launch_batch_rope, launch_fused_swiglu, launch_paged_attention, launch_prefill_attention,
        launch_residual_add, launch_rms_norm,
        ArgmaxParams, BatchArgmaxParams, BatchDecodeRopeParams, BatchKvWriteParams, BatchRopeParams,
        PagedAttentionParams, PrefillAttnParams, RmsNormParams,
    };
    use kapsl_loader::{load_gguf_weights, ModelConfig, TensorData};

    // ── GPU weights ──────────────────────────────────────────────────────────

    /// A weight matrix on the GPU — either fp16 or in a native quantized format.
    enum GpuWeight {
        F16(CudaSlice<f16>),
        Q8_0(CudaSlice<u8>),
        Q4_K(CudaSlice<u8>),
    }

    struct GpuLayerWeights {
        // Norm weights are element-wise scalars — always f16.
        input_layernorm:          CudaSlice<f16>,
        post_attention_layernorm: CudaSlice<f16>,
        // Projection weights may be quantized.
        q_proj:    GpuWeight,
        k_proj:    GpuWeight,
        v_proj:    GpuWeight,
        o_proj:    GpuWeight,
        gate_proj: GpuWeight,
        up_proj:   GpuWeight,
        down_proj: GpuWeight,
    }

    struct GpuModelWeights {
        embed_tokens: CudaSlice<f16>,
        layers: Vec<GpuLayerWeights>,
        norm: CudaSlice<f16>,
        lm_head: CudaSlice<f16>,
    }

    fn upload_f16(device: &Arc<CudaDevice>, t: &TensorData) -> Result<CudaSlice<f16>, EngineError> {
        device
            .htod_sync_copy(&t.to_f16_vec())
            .map_err(|e| EngineError::backend(format!("GPU upload f16: {e}")))
    }

    fn upload_weight(device: &Arc<CudaDevice>, t: &TensorData) -> Result<GpuWeight, EngineError> {
        match t.dtype {
            DType::Q8_0 => device
                .htod_sync_copy::<u8>(&t.bytes)
                .map(GpuWeight::Q8_0)
                .map_err(|e| EngineError::backend(format!("GPU upload Q8_0: {e}"))),
            DType::Q4_K => device
                .htod_sync_copy::<u8>(&t.bytes)
                .map(GpuWeight::Q4_K)
                .map_err(|e| EngineError::backend(format!("GPU upload Q4_K: {e}"))),
            _ => upload_f16(device, t).map(GpuWeight::F16),
        }
    }

    fn upload_weights(
        device: &Arc<CudaDevice>,
        w: &kapsl_loader::ModelWeights,
    ) -> Result<GpuModelWeights, EngineError> {
        let embed_tokens = upload_f16(device, &w.embed_tokens)?;
        let norm         = upload_f16(device, &w.norm)?;
        let lm_head      = upload_f16(device, &w.lm_head)?;
        let mut layers = Vec::with_capacity(w.layers.len());
        for (i, l) in w.layers.iter().enumerate() {
            log::info!("[gguf-native] Uploading layer {}/{}", i + 1, w.layers.len());
            layers.push(GpuLayerWeights {
                input_layernorm:          upload_f16(device, &l.input_layernorm)?,
                post_attention_layernorm: upload_f16(device, &l.post_attention_layernorm)?,
                q_proj:    upload_weight(device, &l.q_proj)?,
                k_proj:    upload_weight(device, &l.k_proj)?,
                v_proj:    upload_weight(device, &l.v_proj)?,
                o_proj:    upload_weight(device, &l.o_proj)?,
                gate_proj: upload_weight(device, &l.gate_proj)?,
                up_proj:   upload_weight(device, &l.up_proj)?,
                down_proj: upload_weight(device, &l.down_proj)?,
            });
        }
        Ok(GpuModelWeights { embed_tokens, layers, norm, lm_head })
    }

    // ── Sampling ─────────────────────────────────────────────────────────────

    struct SampleParams {
        temperature: f32,
        top_k: usize,
        top_p: f32,
    }

    impl SampleParams {
        fn from_meta(meta: Option<&RequestMetadata>) -> Self {
            let m = match meta { Some(m) => m, None => return Self::greedy() };
            Self {
                temperature: m.temperature.unwrap_or(0.0),
                top_k: m.top_k.unwrap_or(0) as usize,
                top_p: m.top_p.unwrap_or(1.0),
            }
        }
        fn greedy() -> Self { Self { temperature: 0.0, top_k: 0, top_p: 1.0 } }
    }

    fn env_usize(name: &str) -> Option<usize> {
        std::env::var(name)
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok())
            .filter(|v| *v > 0)
    }

    fn parse_byte_size(value: &str) -> Option<usize> {
        let lower = value.trim().to_ascii_lowercase();
        if lower.is_empty() {
            return None;
        }
        let (number, multiplier) = if let Some(number) = lower.strip_suffix("gib") {
            (number, 1024usize.pow(3))
        } else if let Some(number) = lower.strip_suffix("mib") {
            (number, 1024usize.pow(2))
        } else if let Some(number) = lower.strip_suffix("gb") {
            (number, 1000usize.pow(3))
        } else if let Some(number) = lower.strip_suffix("mb") {
            (number, 1000usize.pow(2))
        } else if let Some(number) = lower.strip_suffix('b') {
            (number, 1)
        } else {
            (lower.as_str(), 1)
        };
        number.trim().parse::<usize>().ok().and_then(|n| n.checked_mul(multiplier))
    }

    fn env_byte_size(name: &str) -> Option<usize> {
        std::env::var(name)
            .ok()
            .and_then(|v| parse_byte_size(&v))
            .filter(|v| *v > 0)
    }

    fn kv_max_ctx(config: &ModelConfig) -> usize {
        env_usize("KAPSL_KV_MAX_CTX")
            .unwrap_or(config.max_position_embeddings)
            .min(config.max_position_embeddings)
            .max(1)
    }

    fn kv_pool_block_count(config: &ModelConfig, block_size: usize) -> usize {
        let num_layers = config.num_hidden_layers.max(1);
        if let Some(blocks) = env_usize("KAPSL_GGUF_NATIVE_KV_POOL_BLOCKS") {
            return blocks.max(num_layers);
        }

        let bytes_per_block = 2usize
            .saturating_mul(config.num_kv_heads())
            .saturating_mul(block_size)
            .saturating_mul(config.head_dim())
            .saturating_mul(std::mem::size_of::<f16>());
        if let Some(bytes) = env_byte_size("KAPSL_GGUF_NATIVE_KV_POOL_BYTES") {
            return (bytes / bytes_per_block.max(1)).max(num_layers);
        }
        if let Some(mib) = env_usize("KAPSL_GGUF_NATIVE_KV_POOL_MIB") {
            let bytes = mib.saturating_mul(1024 * 1024);
            return (bytes / bytes_per_block.max(1)).max(num_layers);
        }

        let max_sequences = env_usize("KAPSL_GGUF_NATIVE_KV_POOL_SEQS").unwrap_or(8);
        let blocks_per_sequence = kv_max_ctx(config).div_ceil(block_size);
        num_layers
            .saturating_mul(max_sequences.max(1))
            .saturating_mul(blocks_per_sequence.max(1))
    }

    // ── Session state ─────────────────────────────────────────────────────────

    struct SessionState {
        block_tables: Vec<Vec<i32>>,
        context_len: usize,
    }

    // ── Prefill scratch ───────────────────────────────────────────────────────

    struct PrefillScratch {
        cap: usize,
        hidden: CudaSlice<f16>,
        norm: CudaSlice<f16>,
        residual: CudaSlice<f16>,
        q_all: CudaSlice<f16>,
        k_all: CudaSlice<f16>,
        v_all: CudaSlice<f16>,
        attn_out: CudaSlice<f16>,
        gate_out: CudaSlice<f16>,
        up_out: CudaSlice<f16>,
        swiglu_out: CudaSlice<f16>,
        ffn_input: CudaSlice<f16>,
        ffn_out: CudaSlice<f16>,
        o_out: CudaSlice<f16>,
    }

    impl PrefillScratch {
        fn new(device: &Arc<CudaDevice>, h: usize, q_dim: usize, kv_dim: usize, inter: usize)
            -> Result<Self, EngineError>
        {
            let a = |n: usize| device.alloc_zeros::<f16>(n)
                .map_err(|e| EngineError::backend(format!("prefill scratch: {e}")));
            Ok(Self {
                cap: 1,
                hidden:     a(h)?,
                norm:       a(h)?,
                residual:   a(h)?,
                q_all:      a(q_dim)?,
                k_all:      a(kv_dim)?,
                v_all:      a(kv_dim)?,
                attn_out:   a(q_dim)?,
                gate_out:   a(inter)?,
                up_out:     a(inter)?,
                swiglu_out: a(inter)?,
                ffn_input:  a(h)?,
                ffn_out:    a(h)?,
                o_out:      a(h)?,
            })
        }
    }

    // ── BackendInner ──────────────────────────────────────────────────────────

    struct BackendInner {
        device: Arc<CudaDevice>,
        blas: Arc<CudaBlas>,
        config: ModelConfig,
        weights: GpuModelWeights,
        block_pool: Arc<GpuBlockPool>,
        /// Shared cap; other backends on the same device may lower this via an atomic store.
        pool_cap: Arc<AtomicUsize>,
        /// Physical blocks currently held across all active sessions of this backend.
        allocated_blocks: usize,
        // Tokenizer (llama.cpp — model loaded for token↔text conversion only)
        llm_backend: Arc<LlamaBackend>,
        llm_model: Arc<LlamaModel>,
        eos_token: i32,
        // Pre-allocated single-token activation buffers
        hidden_buf: CudaSlice<f16>,
        norm_buf: CudaSlice<f16>,
        residual_buf: CudaSlice<f16>,
        q_buf: CudaSlice<f16>,
        k_buf: CudaSlice<f16>,
        v_buf: CudaSlice<f16>,
        attn_buf: CudaSlice<f16>,
        gate_buf: CudaSlice<f16>,
        up_buf: CudaSlice<f16>,
        swiglu_buf: CudaSlice<f16>,
        ffn_input_buf: CudaSlice<f16>,
        ffn_out_buf: CudaSlice<f16>,
        o_proj_buf: CudaSlice<f16>,
        logits_buf: CudaSlice<f16>,
        ctx_scalar_buf: CudaSlice<i32>,
        gpu_block_tables: Vec<CudaSlice<i32>>,
        gpu_block_table_len: usize,
        sessions: HashMap<String, SessionState>,
        rng: rand::rngs::SmallRng,
        prefill: PrefillScratch,
        argmax_buf: CudaSlice<u32>,
        // Batch decode GPU buffers — grow-only (written by GPU kernels, sized to dec_batch_cap)
        // Small per-step i32 uploads (positions, phys blocks, block tables) are
        // allocated fresh each call via htod_sync_copy to avoid size-mismatch panics.
        dec_batch_cap:    usize,
        dec_batch_logits: CudaSlice<f16>,   // [dec_batch_cap * vocab]
        dec_batch_norm:   CudaSlice<f16>,   // [dec_batch_cap * h]
    }

    impl BackendInner {
        // ── Block management ─────────────────────────────────────────────────

        fn ensure_block(&mut self, block_tables: &mut Vec<Vec<i32>>, position: usize)
            -> Result<(), EngineError>
        {
            let block_size = self.block_pool.block_size();
            let num_layers = self.config.num_hidden_layers;
            let logical = position / block_size;
            if block_tables.is_empty() {
                block_tables.resize(num_layers, Vec::new());
            }
            let current_blocks = block_tables.first().map_or(0, Vec::len);
            if block_tables.iter().any(|table| table.len() != current_blocks) {
                return Err(EngineError::backend(
                    "inconsistent KV block table lengths; refusing to allocate",
                ));
            }
            if current_blocks <= logical {
                let cap = self.pool_cap.load(Ordering::Relaxed);
                if self.allocated_blocks + num_layers > cap {
                    return Err(EngineError::backend(format!(
                        "KV block quota exceeded: {}/{} blocks used",
                        self.allocated_blocks, cap,
                    )));
                }

                let mut allocated = Vec::with_capacity(num_layers);
                for _ in 0..num_layers {
                    match self.block_pool.alloc_block() {
                        Ok(phys) => allocated.push(phys),
                        Err(e) => {
                            for phys in allocated {
                                self.block_pool.free_block(phys);
                            }
                            return Err(EngineError::backend(format!("block alloc: {e}")));
                        }
                    }
                }

                for (table, phys) in block_tables.iter_mut().zip(allocated) {
                    table.push(phys as i32);
                }
                self.allocated_blocks += num_layers;
            }
            Ok(())
        }

        fn free_block_tables(&mut self, block_tables: &[Vec<i32>]) {
            for lt in block_tables {
                for &p in lt {
                    self.block_pool.free_block(p as u32);
                    self.allocated_blocks = self.allocated_blocks.saturating_sub(1);
                }
            }
        }

        fn sync_gpu_block_tables(&mut self, block_tables: &[Vec<i32>])
            -> Result<(), EngineError>
        {
            let cpu_len = block_tables.first().map_or(0, |v| v.len());
            if cpu_len == self.gpu_block_table_len { return Ok(()); }
            if self.gpu_block_tables.len() != block_tables.len() {
                self.gpu_block_tables = Vec::with_capacity(block_tables.len());
                for bt in block_tables {
                    let sl = self.device.htod_sync_copy(bt)
                        .map_err(|e| EngineError::backend(format!("bt upload: {e}")))?;
                    self.gpu_block_tables.push(sl);
                }
            } else {
                for (gpu_bt, cpu_bt) in self.gpu_block_tables.iter_mut().zip(block_tables) {
                    *gpu_bt = self.device.htod_sync_copy(cpu_bt)
                        .map_err(|e| EngineError::backend(format!("bt upload: {e}")))?;
                }
            }
            self.gpu_block_table_len = cpu_len;
            Ok(())
        }

        // ── Prefill scratch management ───────────────────────────────────────

        fn ensure_prefill_scratch(&mut self, n: usize) -> Result<(), EngineError> {
            if n <= self.prefill.cap { return Ok(()); }
            let device = self.device.clone();
            let h      = self.config.hidden_size;
            let q_dim  = self.config.num_attention_heads * self.config.head_dim();
            let kv_dim = self.config.num_kv_heads() * self.config.head_dim();
            let inter  = self.config.intermediate_size;
            let a = |sz: usize| device.alloc_zeros::<f16>(sz)
                .map_err(|e| EngineError::backend(format!("prefill grow: {e}")));
            self.prefill.hidden     = a(n * h)?;
            self.prefill.norm       = a(n * h)?;
            self.prefill.residual   = a(n * h)?;
            self.prefill.q_all      = a(n * q_dim)?;
            self.prefill.k_all      = a(n * kv_dim)?;
            self.prefill.v_all      = a(n * kv_dim)?;
            self.prefill.attn_out   = a(n * q_dim)?;
            self.prefill.gate_out   = a(n * inter)?;
            self.prefill.up_out     = a(n * inter)?;
            self.prefill.swiglu_out = a(n * inter)?;
            self.prefill.ffn_input  = a(n * h)?;
            self.prefill.ffn_out    = a(n * h)?;
            self.prefill.o_out      = a(n * h)?;
            self.prefill.cap = n;
            Ok(())
        }

        // ── Batch decode scratch management ──────────────────────────────────

        fn ensure_dec_scratch(&mut self, batch: usize) -> Result<(), EngineError> {
            if batch <= self.dec_batch_cap { return Ok(()); }
            let dev   = self.device.clone();
            let vocab = self.config.vocab_size;
            let h     = self.config.hidden_size;
            let ae    = |tag: &'static str| move |e| EngineError::backend(format!("{tag}: {e}"));
            self.dec_batch_logits = dev.alloc_zeros::<f16>(batch * vocab).map_err(ae("dec_logits"))?;
            self.dec_batch_norm   = dev.alloc_zeros::<f16>(batch * h).map_err(ae("dec_norm"))?;
            self.dec_batch_cap = batch;
            Ok(())
        }

        // ── Sampling ─────────────────────────────────────────────────────────

        fn greedy(logits: &[f32]) -> u32 {
            logits.iter().enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i as u32)
                .unwrap_or(0)
        }

        fn sample(&mut self, logits: &[f32], p: &SampleParams) -> u32 {
            if p.temperature < 1e-6 { return Self::greedy(logits); }
            let inv_t = 1.0 / p.temperature;
            let mut scores: Vec<f32> = logits.iter().map(|&l| l * inv_t).collect();
            if p.top_k > 0 && p.top_k < scores.len() {
                let mut sorted = scores.clone();
                sorted.sort_unstable_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
                let thresh = sorted[p.top_k - 1];
                for s in &mut scores { if *s < thresh { *s = f32::NEG_INFINITY; } }
            }
            let max_s = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut probs: Vec<f32> = scores.iter().map(|&s| (s - max_s).exp()).collect();
            let sum: f32 = probs.iter().sum();
            if sum <= 0.0 { return Self::greedy(logits); }
            for p2 in &mut probs { *p2 /= sum; }
            if p.top_p < 1.0 {
                let mut order: Vec<usize> = (0..probs.len()).collect();
                order.sort_unstable_by(|&a, &b| {
                    probs[b].partial_cmp(&probs[a]).unwrap_or(std::cmp::Ordering::Equal)
                });
                let mut cum = 0.0f32;
                let mut cutoff = 0.0f32;
                for &i in &order {
                    cum += probs[i];
                    if cum >= p.top_p { cutoff = probs[i]; break; }
                }
                for pr in &mut probs { if *pr < cutoff { *pr = 0.0; } }
                let new_sum: f32 = probs.iter().sum();
                if new_sum > 0.0 { for pr in &mut probs { *pr /= new_sum; } }
            }
            let r: f32 = self.rng.gen();
            let mut cum = 0.0f32;
            for (i, &pr) in probs.iter().enumerate() {
                cum += pr;
                if r <= cum { return i as u32; }
            }
            (probs.len() - 1) as u32
        }

        // ── cuBLAS GEMM helper (fp16 weights) ────────────────────────────────

        fn gemm(
            blas: &CudaBlas,
            out_dim: i32, batch: i32, in_dim: i32,
            weight: &CudaSlice<f16>, lda: i32,
            input: &CudaSlice<f16>,  ldb: i32,
            out: &mut CudaSlice<f16>, ldc: i32,
            label: &str,
        ) -> Result<(), EngineError> {
            unsafe {
                blas.gemm(
                    cudarc::cublas::GemmConfig {
                        transa: cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_T,
                        transb: cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_N,
                        m: out_dim,
                        n: batch,
                        k: in_dim,
                        alpha: f16::from_f32(1.0),
                        lda,
                        ldb,
                        beta: f16::from_f32(0.0),
                        ldc,
                    },
                    weight,
                    input,
                    out,
                )
                .map_err(|e| EngineError::backend(format!("{label} gemm: {e}")))
            }
        }

        // ── Dispatching GEMM — fp16 via cuBLAS, quantized via fused kernel ───

        fn gemm_w(
            device: &Arc<CudaDevice>,
            blas: &CudaBlas,
            out_dim: i32, batch: i32, in_dim: i32,
            weight: &GpuWeight,
            input: &CudaSlice<f16>,
            out: &mut CudaSlice<f16>,
            label: &str,
        ) -> Result<(), EngineError> {
            match weight {
                GpuWeight::F16(w) => Self::gemm(
                    blas, out_dim, batch, in_dim,
                    w, in_dim, input, in_dim, out, out_dim, label,
                ),
                GpuWeight::Q8_0(w) => launch_q8_0_gemv(device, &mut QuantGemvParams {
                    out, w, x: input,
                    m: out_dim as u32, k: in_dim as u32, b: batch as u32,
                }).map_err(|e| EngineError::backend(format!("{label} q8_0: {e}"))),
                GpuWeight::Q4_K(w) => launch_q4_k_gemv(device, &mut QuantGemvParams {
                    out, w, x: input,
                    m: out_dim as u32, k: in_dim as u32, b: batch as u32,
                }).map_err(|e| EngineError::backend(format!("{label} q4_k: {e}"))),
            }
        }

        // ── Batch prefill ─────────────────────────────────────────────────────

        fn prefill_compute(
            &mut self,
            token_ids: &[u32],
            start_position: u32,
            block_tables: &[Vec<i32>],
        ) -> Result<(), EngineError> {
            let n = token_ids.len();
            let h = self.config.hidden_size;
            let num_q = self.config.num_attention_heads;
            let num_kv = self.config.num_kv_heads();
            let head_dim = self.config.head_dim();
            let inter = self.config.intermediate_size;
            let eps = self.config.rms_norm_eps as f32;
            let rope_theta = self.config.rope_theta as f32;
            let scale = 1.0 / (head_dim as f32).sqrt();
            let block_size = self.block_pool.block_size();
            let vocab = self.config.vocab_size;
            let e = |s: String| EngineError::backend(s);

            self.ensure_prefill_scratch(n)?;

            for (i, &tok) in token_ids.iter().enumerate() {
                let off = tok as usize * h;
                self.device.dtod_copy(
                    &self.weights.embed_tokens.slice(off..off + h),
                    &mut self.prefill.hidden.slice_mut(i * h..(i + 1) * h),
                ).map_err(|err| e(format!("embed: {err}")))?;
            }

            let pos_in_blk_host: Vec<i32> = (0..n)
                .map(|pos| (pos % block_size) as i32)
                .collect();

            let blas = Arc::clone(&self.blas);
            for layer_idx in 0..self.weights.layers.len() {
                let layer = &self.weights.layers[layer_idx];

                launch_rms_norm(&self.device, &mut RmsNormParams {
                    out: &mut self.prefill.norm, input: self.prefill.hidden.slice(..),
                    weight: &layer.input_layernorm,
                    rows: n as u32, dim: h as u32, eps,
                }).map_err(e)?;

                Self::gemm_w(&self.device, &blas, (num_q * head_dim) as i32, n as i32, h as i32,
                    &layer.q_proj, &self.prefill.norm,
                    &mut self.prefill.q_all, "Q")?;
                Self::gemm_w(&self.device, &blas, (num_kv * head_dim) as i32, n as i32, h as i32,
                    &layer.k_proj, &self.prefill.norm,
                    &mut self.prefill.k_all, "K")?;
                Self::gemm_w(&self.device, &blas, (num_kv * head_dim) as i32, n as i32, h as i32,
                    &layer.v_proj, &self.prefill.norm,
                    &mut self.prefill.v_all, "V")?;

                launch_batch_rope(&self.device, &mut BatchRopeParams {
                    q: &mut self.prefill.q_all, k: &mut self.prefill.k_all,
                    seq_len: n as u32,
                    num_q_heads: num_q as u32, num_kv_heads: num_kv as u32,
                    head_dim: head_dim as u32,
                    position_offset: start_position,
                    theta: rope_theta,
                }).map_err(e)?;

                let phys_dev = self.device.htod_sync_copy(&{
                    (0..n).map(|pos| block_tables[layer_idx][pos / block_size])
                        .collect::<Vec<i32>>()
                }).map_err(|err| EngineError::backend(format!("phys_dev: {err}")))?;
                let pos_dev = self.device.htod_sync_copy(&pos_in_blk_host)
                    .map_err(|err| EngineError::backend(format!("pos_dev: {err}")))?;

                launch_batch_kv_write(&self.device, &mut BatchKvWriteParams {
                    kv_cache: self.block_pool.storage_mut(),
                    k: &self.prefill.k_all, v: &self.prefill.v_all,
                    physical_blocks: &phys_dev, pos_in_blocks: &pos_dev,
                    seq_len: n as u32, num_kv_heads: num_kv as u32,
                    block_size: block_size as u32, head_dim: head_dim as u32,
                }).map_err(e)?;

                launch_prefill_attention(&self.device, &mut PrefillAttnParams {
                    out: &mut self.prefill.attn_out,
                    q: &self.prefill.q_all, k: &self.prefill.k_all, v: &self.prefill.v_all,
                    scale, seq_len: n as u32,
                    num_q_heads: num_q as u32, num_kv_heads: num_kv as u32,
                    head_dim: head_dim as u32,
                }).map_err(e)?;

                let layer = &self.weights.layers[layer_idx];
                Self::gemm_w(&self.device, &blas, h as i32, n as i32, (num_q * head_dim) as i32,
                    &layer.o_proj, &self.prefill.attn_out,
                    &mut self.prefill.o_out, "O")?;

                launch_residual_add(&self.device, &mut self.prefill.residual,
                    &self.prefill.hidden, &self.prefill.o_out, (n * h) as u32).map_err(e)?;

                let layer = &self.weights.layers[layer_idx];
                launch_rms_norm(&self.device, &mut RmsNormParams {
                    out: &mut self.prefill.ffn_input, input: self.prefill.residual.slice(..),
                    weight: &layer.post_attention_layernorm,
                    rows: n as u32, dim: h as u32, eps,
                }).map_err(e)?;

                let layer = &self.weights.layers[layer_idx];
                Self::gemm_w(&self.device, &blas, inter as i32, n as i32, h as i32,
                    &layer.gate_proj, &self.prefill.ffn_input,
                    &mut self.prefill.gate_out, "gate")?;
                Self::gemm_w(&self.device, &blas, inter as i32, n as i32, h as i32,
                    &layer.up_proj, &self.prefill.ffn_input,
                    &mut self.prefill.up_out, "up")?;

                launch_fused_swiglu(&self.device, &mut self.prefill.swiglu_out,
                    &self.prefill.gate_out, &self.prefill.up_out, (n * inter) as u32).map_err(e)?;

                let layer = &self.weights.layers[layer_idx];
                Self::gemm_w(&self.device, &blas, h as i32, n as i32, inter as i32,
                    &layer.down_proj, &self.prefill.swiglu_out,
                    &mut self.prefill.ffn_out, "down")?;

                launch_residual_add(&self.device, &mut self.prefill.hidden,
                    &self.prefill.residual, &self.prefill.ffn_out, (n * h) as u32).map_err(e)?;
            }

            let last_off = (n - 1) * h;
            launch_rms_norm(&self.device, &mut RmsNormParams {
                out: &mut self.norm_buf, input: self.prefill.hidden.slice(last_off..last_off + h),
                weight: &self.weights.norm,
                rows: 1, dim: h as u32, eps,
            }).map_err(e)?;

            Self::gemm(&blas, vocab as i32, 1, h as i32,
                &self.weights.lm_head, h as i32, &self.norm_buf, h as i32,
                &mut self.logits_buf, vocab as i32, "lm_head")?;

            Ok(())
        }

        fn forward_prefill(
            &mut self,
            token_ids: &[u32],
            start_position: u32,
            block_tables: &[Vec<i32>],
        ) -> Result<Vec<f32>, EngineError> {
            self.prefill_compute(token_ids, start_position, block_tables)?;
            let f16v: Vec<f16> = self.device.dtoh_sync_copy(&self.logits_buf)
                .map_err(|err| EngineError::backend(format!("logits dl: {err}")))?;
            Ok(f16v.iter().map(|v| v.to_f32()).collect())
        }

        fn forward_prefill_greedy(
            &mut self,
            token_ids: &[u32],
            start_position: u32,
            block_tables: &[Vec<i32>],
        ) -> Result<u32, EngineError> {
            let vocab = self.config.vocab_size;
            self.prefill_compute(token_ids, start_position, block_tables)?;
            launch_argmax(&self.device, &mut ArgmaxParams {
                input: &self.logits_buf,
                output: &mut self.argmax_buf,
                vocab_size: vocab as u32,
            }).map_err(|s| EngineError::backend(s))?;
            let ids: Vec<u32> = self.device.dtoh_sync_copy(&self.argmax_buf)
                .map_err(|err| EngineError::backend(format!("argmax dl: {err}")))?;
            Ok(ids[0])
        }

        // ── Single-token decode ───────────────────────────────────────────────

        fn one_token_compute(
            &mut self,
            token_id: u32,
            block_tables: &[Vec<i32>],
            context_len: usize,
            position: u32,
        ) -> Result<(), EngineError> {
            let h = self.config.hidden_size;
            let num_q = self.config.num_attention_heads;
            let num_kv = self.config.num_kv_heads();
            let head_dim = self.config.head_dim();
            let inter = self.config.intermediate_size;
            let eps = self.config.rms_norm_eps as f32;
            let rope_theta = self.config.rope_theta as f32;
            let scale = 1.0 / (head_dim as f32).sqrt();
            let block_size = self.block_pool.block_size();
            let vocab = self.config.vocab_size;
            let e = |s: String| EngineError::backend(s);

            let embed_off = token_id as usize * h;
            self.device.dtod_copy(
                &self.weights.embed_tokens.slice(embed_off..embed_off + h),
                &mut self.hidden_buf,
            ).map_err(|err| e(format!("embed: {err}")))?;

            self.device.htod_sync_copy_into(&[context_len as i32], &mut self.ctx_scalar_buf)
                .map_err(|err| e(format!("ctx_dev: {err}")))?;

            let pos_in_seq = context_len - 1;
            let pos_in_block = pos_in_seq % block_size;
            self.sync_gpu_block_tables(block_tables)?;

            let q_dim = num_q * head_dim;
            let kv_dim = num_kv * head_dim;
            let blas = Arc::clone(&self.blas);

            for layer_idx in 0..self.weights.layers.len() {
                let layer = &self.weights.layers[layer_idx];

                launch_rms_norm(&self.device, &mut RmsNormParams {
                    out: &mut self.norm_buf, input: self.hidden_buf.slice(..),
                    weight: &layer.input_layernorm,
                    rows: 1, dim: h as u32, eps,
                }).map_err(e)?;

                Self::gemm_w(&self.device, &blas, q_dim as i32, 1, h as i32,
                    &layer.q_proj, &self.norm_buf,
                    &mut self.q_buf, "Q")?;
                Self::gemm_w(&self.device, &blas, kv_dim as i32, 1, h as i32,
                    &layer.k_proj, &self.norm_buf,
                    &mut self.k_buf, "K")?;
                Self::gemm_w(&self.device, &blas, kv_dim as i32, 1, h as i32,
                    &layer.v_proj, &self.norm_buf,
                    &mut self.v_buf, "V")?;

                use kapsl_kernels::cuda_kernels::{launch_rope, RopeParams};
                launch_rope(&self.device, &mut RopeParams {
                    q: &mut self.q_buf, k: &mut self.k_buf,
                    num_q_heads: num_q as u32, num_kv_heads: num_kv as u32,
                    head_dim: head_dim as u32, position, theta: rope_theta,
                }).map_err(e)?;

                use kapsl_kernels::cuda_kernels::{launch_kv_write, KvWriteParams};
                let physical_block = block_tables[layer_idx][pos_in_seq / block_size];
                launch_kv_write(&self.device, &mut KvWriteParams {
                    kv_cache: self.block_pool.storage_mut(),
                    k_vec: &self.k_buf, v_vec: &self.v_buf,
                    physical_block: physical_block as u32,
                    pos_in_block: pos_in_block as u32,
                    num_kv_heads: num_kv as u32,
                    block_size: block_size as u32,
                    head_dim: head_dim as u32,
                }).map_err(e)?;

                let max_blocks = self.gpu_block_tables[layer_idx].len() as u32;
                launch_paged_attention(&self.device, &mut PagedAttentionParams {
                    out: &mut self.attn_buf, q: &self.q_buf,
                    kv_cache: self.block_pool.storage(),
                    block_tables: &self.gpu_block_tables[layer_idx],
                    context_lens: &self.ctx_scalar_buf,
                    scale, batch_size: 1,
                    num_q_heads: num_q as u32, num_kv_heads: num_kv as u32,
                    head_dim: head_dim as u32, block_size: block_size as u32,
                    max_blocks_per_seq: max_blocks,
                }).map_err(e)?;

                let layer = &self.weights.layers[layer_idx];
                Self::gemm_w(&self.device, &blas, h as i32, 1, q_dim as i32,
                    &layer.o_proj, &self.attn_buf,
                    &mut self.o_proj_buf, "O")?;

                launch_residual_add(&self.device, &mut self.residual_buf,
                    &self.hidden_buf, &self.o_proj_buf, h as u32).map_err(e)?;

                let layer = &self.weights.layers[layer_idx];
                launch_rms_norm(&self.device, &mut RmsNormParams {
                    out: &mut self.ffn_input_buf, input: self.residual_buf.slice(..),
                    weight: &layer.post_attention_layernorm,
                    rows: 1, dim: h as u32, eps,
                }).map_err(e)?;

                let layer = &self.weights.layers[layer_idx];
                Self::gemm_w(&self.device, &blas, inter as i32, 1, h as i32,
                    &layer.gate_proj, &self.ffn_input_buf,
                    &mut self.gate_buf, "gate")?;
                Self::gemm_w(&self.device, &blas, inter as i32, 1, h as i32,
                    &layer.up_proj, &self.ffn_input_buf,
                    &mut self.up_buf, "up")?;

                launch_fused_swiglu(&self.device, &mut self.swiglu_buf,
                    &self.gate_buf, &self.up_buf, inter as u32).map_err(e)?;

                let layer = &self.weights.layers[layer_idx];
                Self::gemm_w(&self.device, &blas, h as i32, 1, inter as i32,
                    &layer.down_proj, &self.swiglu_buf,
                    &mut self.ffn_out_buf, "down")?;

                launch_residual_add(&self.device, &mut self.hidden_buf,
                    &self.residual_buf, &self.ffn_out_buf, h as u32).map_err(e)?;
            }

            launch_rms_norm(&self.device, &mut RmsNormParams {
                out: &mut self.norm_buf, input: self.hidden_buf.slice(..),
                weight: &self.weights.norm, rows: 1, dim: h as u32, eps,
            }).map_err(e)?;

            Self::gemm(&blas, vocab as i32, 1, h as i32,
                &self.weights.lm_head, h as i32, &self.norm_buf, h as i32,
                &mut self.logits_buf, vocab as i32, "lm_head")?;

            Ok(())
        }

        fn forward_one_token(
            &mut self,
            token_id: u32,
            block_tables: &[Vec<i32>],
            context_len: usize,
            position: u32,
        ) -> Result<Vec<f32>, EngineError> {
            self.one_token_compute(token_id, block_tables, context_len, position)?;
            let f16v: Vec<f16> = self.device.dtoh_sync_copy(&self.logits_buf)
                .map_err(|err| EngineError::backend(format!("logits dl: {err}")))?;
            Ok(f16v.iter().map(|v| v.to_f32()).collect())
        }

        fn forward_one_token_greedy(
            &mut self,
            token_id: u32,
            block_tables: &[Vec<i32>],
            context_len: usize,
            position: u32,
        ) -> Result<u32, EngineError> {
            let vocab = self.config.vocab_size;
            self.one_token_compute(token_id, block_tables, context_len, position)?;
            launch_argmax(&self.device, &mut ArgmaxParams {
                input: &self.logits_buf,
                output: &mut self.argmax_buf,
                vocab_size: vocab as u32,
            }).map_err(|s| EngineError::backend(s))?;
            let ids: Vec<u32> = self.device.dtoh_sync_copy(&self.argmax_buf)
                .map_err(|err| EngineError::backend(format!("argmax dl: {err}")))?;
            Ok(ids[0])
        }

        // ── Batched single-step decode ────────────────────────────────────────

        /// Run one decode step for B sequences simultaneously.
        /// `block_tables[b][layer][block_idx]` — per-sequence per-layer blocks.
        /// `context_lens[b]` — length AFTER this token (already incremented by caller).
        /// Returns next-token ids in the same order.
        fn forward_batch_decode_greedy(
            &mut self,
            tokens: &[u32],
            block_tables: &[&Vec<Vec<i32>>],
            context_lens: &[usize],
        ) -> Result<Vec<u32>, EngineError> {
            let b = tokens.len();
            let h       = self.config.hidden_size;
            let num_q   = self.config.num_attention_heads;
            let num_kv  = self.config.num_kv_heads();
            let head_dim = self.config.head_dim();
            let inter   = self.config.intermediate_size;
            let eps     = self.config.rms_norm_eps as f32;
            let theta   = self.config.rope_theta as f32;
            let scale   = 1.0 / (head_dim as f32).sqrt();
            let bs      = self.block_pool.block_size();
            let vocab   = self.config.vocab_size;
            let nl      = self.config.num_hidden_layers;
            let e = |s: String| EngineError::backend(s);

            let max_blks = block_tables.iter()
                .map(|bt| bt.first().map_or(0, |l| l.len()))
                .max().unwrap_or(1)
                .max(1);

            self.ensure_prefill_scratch(b)?;
            self.ensure_dec_scratch(b)?;

            // Embed
            for (i, &tok) in tokens.iter().enumerate() {
                let off = tok as usize * h;
                self.device.dtod_copy(
                    &self.weights.embed_tokens.slice(off..off + h),
                    &mut self.prefill.hidden.slice_mut(i * h..(i + 1) * h),
                ).map_err(|err| e(format!("embed: {err}")))?;
            }

            // Upload per-step small CPU→GPU arrays (fresh alloc avoids size-mismatch panics)
            let pos_cpu: Vec<i32> = context_lens.iter().map(|&c| (c - 1) as i32).collect();
            let ctx_cpu: Vec<i32> = context_lens.iter().map(|&c| c as i32).collect();
            let pib_cpu: Vec<i32> = context_lens.iter()
                .map(|&c| ((c - 1) % bs) as i32).collect();
            let pos_dev = self.device.htod_sync_copy(&pos_cpu)
                .map_err(|err| e(format!("pos up: {err}")))?;
            let ctx_dev = self.device.htod_sync_copy(&ctx_cpu)
                .map_err(|err| e(format!("ctx up: {err}")))?;
            let pib_dev = self.device.htod_sync_copy(&pib_cpu)
                .map_err(|err| e(format!("pib up: {err}")))?;

            let blas = Arc::clone(&self.blas);
            for li in 0..nl {
                let layer = &self.weights.layers[li];

                launch_rms_norm(&self.device, &mut RmsNormParams {
                    out: &mut self.prefill.norm, input: self.prefill.hidden.slice(..),
                    weight: &layer.input_layernorm,
                    rows: b as u32, dim: h as u32, eps,
                }).map_err(e)?;

                Self::gemm_w(&self.device, &blas, (num_q * head_dim) as i32, b as i32, h as i32,
                    &layer.q_proj, &self.prefill.norm,
                    &mut self.prefill.q_all, "Q")?;
                Self::gemm_w(&self.device, &blas, (num_kv * head_dim) as i32, b as i32, h as i32,
                    &layer.k_proj, &self.prefill.norm,
                    &mut self.prefill.k_all, "K")?;
                Self::gemm_w(&self.device, &blas, (num_kv * head_dim) as i32, b as i32, h as i32,
                    &layer.v_proj, &self.prefill.norm,
                    &mut self.prefill.v_all, "V")?;

                launch_batch_decode_rope(&self.device, &mut BatchDecodeRopeParams {
                    q: &mut self.prefill.q_all, k: &mut self.prefill.k_all,
                    positions: &pos_dev,
                    batch_size: b as u32, num_q_heads: num_q as u32,
                    num_kv_heads: num_kv as u32, head_dim: head_dim as u32, theta,
                }).map_err(e)?;

                // Per-layer physical block for each sequence's current KV write position
                let phys_cpu: Vec<i32> = (0..b).map(|s| {
                    block_tables[s][li][(context_lens[s] - 1) / bs]
                }).collect();
                let phys_dev = self.device.htod_sync_copy(&phys_cpu)
                    .map_err(|err| e(format!("phys up: {err}")))?;

                launch_batch_kv_write(&self.device, &mut BatchKvWriteParams {
                    kv_cache: self.block_pool.storage_mut(),
                    k: &self.prefill.k_all, v: &self.prefill.v_all,
                    physical_blocks: &phys_dev,
                    pos_in_blocks: &pib_dev,
                    seq_len: b as u32, num_kv_heads: num_kv as u32,
                    block_size: bs as u32, head_dim: head_dim as u32,
                }).map_err(e)?;

                // Flat block table [B * max_blks] for this layer's paged attention
                let mut flat_bt = vec![0i32; b * max_blks];
                for (s, bt) in block_tables.iter().enumerate() {
                    for (bi, &blk) in bt[li].iter().enumerate() {
                        flat_bt[s * max_blks + bi] = blk;
                    }
                }
                let bt_dev = self.device.htod_sync_copy(&flat_bt)
                    .map_err(|err| e(format!("bt up: {err}")))?;

                launch_paged_attention(&self.device, &mut PagedAttentionParams {
                    out: &mut self.prefill.attn_out, q: &self.prefill.q_all,
                    kv_cache: self.block_pool.storage(),
                    block_tables: &bt_dev,
                    context_lens: &ctx_dev,
                    scale, batch_size: b as u32,
                    num_q_heads: num_q as u32, num_kv_heads: num_kv as u32,
                    head_dim: head_dim as u32, block_size: bs as u32,
                    max_blocks_per_seq: max_blks as u32,
                }).map_err(e)?;

                let layer = &self.weights.layers[li];
                Self::gemm_w(&self.device, &blas, h as i32, b as i32, (num_q * head_dim) as i32,
                    &layer.o_proj, &self.prefill.attn_out,
                    &mut self.prefill.o_out, "O")?;

                launch_residual_add(&self.device, &mut self.prefill.residual,
                    &self.prefill.hidden, &self.prefill.o_out, (b * h) as u32).map_err(e)?;

                let layer = &self.weights.layers[li];
                launch_rms_norm(&self.device, &mut RmsNormParams {
                    out: &mut self.prefill.ffn_input, input: self.prefill.residual.slice(..),
                    weight: &layer.post_attention_layernorm,
                    rows: b as u32, dim: h as u32, eps,
                }).map_err(e)?;

                let layer = &self.weights.layers[li];
                Self::gemm_w(&self.device, &blas, inter as i32, b as i32, h as i32,
                    &layer.gate_proj, &self.prefill.ffn_input,
                    &mut self.prefill.gate_out, "gate")?;
                Self::gemm_w(&self.device, &blas, inter as i32, b as i32, h as i32,
                    &layer.up_proj, &self.prefill.ffn_input,
                    &mut self.prefill.up_out, "up")?;

                launch_fused_swiglu(&self.device, &mut self.prefill.swiglu_out,
                    &self.prefill.gate_out, &self.prefill.up_out, (b * inter) as u32).map_err(e)?;

                let layer = &self.weights.layers[li];
                Self::gemm_w(&self.device, &blas, h as i32, b as i32, inter as i32,
                    &layer.down_proj, &self.prefill.swiglu_out,
                    &mut self.prefill.ffn_out, "down")?;

                launch_residual_add(&self.device, &mut self.prefill.hidden,
                    &self.prefill.residual, &self.prefill.ffn_out, (b * h) as u32).map_err(e)?;
            }

            // Final norm: all B rows
            launch_rms_norm(&self.device, &mut RmsNormParams {
                out: &mut self.dec_batch_norm, input: self.prefill.hidden.slice(..),
                weight: &self.weights.norm, rows: b as u32, dim: h as u32, eps,
            }).map_err(e)?;

            // lm_head: [vocab, B]
            Self::gemm(&blas, vocab as i32, b as i32, h as i32,
                &self.weights.lm_head, h as i32, &self.dec_batch_norm, h as i32,
                &mut self.dec_batch_logits, vocab as i32, "lm_head")?;

            // Batch argmax — allocate exactly-B output so dtoh_sync_copy returns B elements
            let mut argmax_dev = self.device.alloc_zeros::<u32>(b)
                .map_err(|err| e(format!("argmax alloc: {err}")))?;
            launch_batch_argmax(&self.device, &mut BatchArgmaxParams {
                input: &self.dec_batch_logits,
                output: &mut argmax_dev,
                batch_size: b as u32, vocab_size: vocab as u32,
            }).map_err(|s| EngineError::backend(s))?;

            self.device.dtoh_sync_copy(&argmax_dev)
                .map_err(|err| e(format!("dec argmax dl: {err}")))
        }

        // ── Decode loop ───────────────────────────────────────────────────────

        fn token_to_str(&self, tok: u32) -> String {
            use llama_cpp_2::token::LlamaToken;
            self.llm_model
                .token_to_str(LlamaToken(tok as i32), Special::Tokenize)
                .unwrap_or_default()
        }

        /// Runs prefill + decode, yielding detokenized text pieces.
        ///
        /// If `tx` is `Some`, each piece is sent immediately for streaming.
        /// For stateless calls, pass a local `SessionState` and free its blocks after return.
        fn run_decode(
            &mut self,
            prompt_ids: &[u32],
            session: &mut SessionState,
            max_new_tokens: u32,
            sp: &SampleParams,
            cancel: Option<&kapsl_engine_api::CancellationToken>,
            tx: Option<&std::sync::mpsc::Sender<Result<String, EngineError>>>,
        ) -> Result<String, EngineError> {
            if prompt_ids.is_empty() { return Ok(String::new()); }

            let block_tables = &mut session.block_tables;
            let context_len_ref = &mut session.context_len;

            for i in 0..prompt_ids.len() {
                self.ensure_block(block_tables, *context_len_ref + i)?;
            }

            let start_position = *context_len_ref as u32;
            let greedy = sp.temperature < 1e-6;

            let mut next = if greedy {
                self.forward_prefill_greedy(prompt_ids, start_position, block_tables)?
            } else {
                let logits = self.forward_prefill(prompt_ids, start_position, block_tables)?;
                self.sample(&logits, sp)
            };
            *context_len_ref += prompt_ids.len();

            let eos = self.eos_token as u32;
            let mut out = String::new();

            for _ in 0..max_new_tokens {
                if cancel.map_or(false, |c| c.is_cancelled()) { break; }
                if next == eos { break; }

                let piece = self.token_to_str(next);
                if let Some(tx) = tx {
                    if tx.send(Ok(piece.clone())).is_err() { break; }
                }
                out.push_str(&piece);

                self.ensure_block(block_tables, *context_len_ref)?;
                *context_len_ref += 1;
                let position = (*context_len_ref - 1) as u32;

                next = if greedy {
                    self.forward_one_token_greedy(next, block_tables, *context_len_ref, position)?
                } else {
                    let logits = self.forward_one_token(next, block_tables, *context_len_ref, position)?;
                    self.sample(&logits, sp)
                };
            }

            Ok(out)
        }
    }

    // llama.cpp backend is a global singleton — can only be initialized once per process.
    static LLAMA_BACKEND: OnceLock<Result<Arc<LlamaBackend>, String>> = OnceLock::new();

    fn get_llama_backend() -> Result<Arc<LlamaBackend>, EngineError> {
        LLAMA_BACKEND
            .get_or_init(|| {
                LlamaBackend::init()
                    .map(Arc::new)
                    .map_err(|e| format!("llama backend: {e}"))
            })
            .as_ref()
            .map(Arc::clone)
            .map_err(|e| EngineError::backend(e.clone()))
    }

    // ── Batch decode coordinator ──────────────────────────────────────────────

    struct DecodeState {
        current_token: u32,
        block_tables:  Vec<Vec<i32>>,   // [num_layers][num_blocks]
        context_len:   usize,
        max_remaining: u32,
        eos:           u32,
        session_id:    Option<String>,
        tx:            std::sync::mpsc::SyncSender<Result<String, EngineError>>,
    }

    struct BatchDecodeCoordinator {
        submit_tx:    std::sync::mpsc::Sender<DecodeState>,
        active_count: Arc<AtomicUsize>,
    }

    impl BatchDecodeCoordinator {
        fn spawn(inner: Arc<Mutex<Option<BackendInner>>>) -> Self {
            let (submit_tx, submit_rx) = std::sync::mpsc::channel::<DecodeState>();
            let active_count = Arc::new(AtomicUsize::new(0));
            let active_count_thread = Arc::clone(&active_count);
            std::thread::Builder::new()
                .name("gguf-native-dec".into())
                .spawn(move || {
                    let mut active: Vec<DecodeState> = Vec::new();
                    loop {
                        // Wait for at least one slot when idle
                        if active.is_empty() {
                            match submit_rx.recv() {
                                Ok(s) => active.push(s),
                                Err(_) => return,
                            }
                        }
                        // Drain any additional newly submitted slots
                        loop {
                            match submit_rx.try_recv() {
                                Ok(s) => active.push(s),
                                Err(_) => break,
                            }
                        }
                        active_count_thread.store(active.len(), Ordering::Relaxed);

                        let mut guard = inner.lock().unwrap();
                        match guard.as_mut() {
                            Some(b) => Self::step(b, &mut active),
                            None => {
                                for s in active.drain(..) {
                                    let _ = s.tx.send(Err(EngineError::ModelNotLoaded));
                                }
                            }
                        }
                        active_count_thread.store(active.len(), Ordering::Relaxed);
                    }
                })
                .expect("spawn gguf-native-dec");
            Self { submit_tx, active_count }
        }

        fn active_count(&self) -> usize {
            self.active_count.load(Ordering::Relaxed)
        }

        fn submit(&self, state: DecodeState) {
            let _ = self.submit_tx.send(state);
        }

        fn step(b: &mut BackendInner, active: &mut Vec<DecodeState>) {
            // Remove completed slots (EOS or exhausted budget)
            let mut i = 0;
            while i < active.len() {
                let s = &active[i];
                if s.current_token == s.eos || s.max_remaining == 0 {
                    let s = active.remove(i);
                    if let Some(sid) = s.session_id {
                        b.sessions.insert(sid, SessionState {
                            block_tables: s.block_tables, context_len: s.context_len,
                        });
                    } else {
                        b.free_block_tables(&s.block_tables);
                    }
                } else {
                    i += 1;
                }
            }
            if active.is_empty() { return; }

            // Emit current tokens and prepare next step
            let mut failed: Vec<usize> = Vec::new();
            for (i, s) in active.iter_mut().enumerate() {
                let piece = b.token_to_str(s.current_token);
                if s.tx.send(Ok(piece)).is_err() {
                    failed.push(i); continue;
                }
                match b.ensure_block(&mut s.block_tables, s.context_len) {
                    Ok(()) => { s.context_len += 1; s.max_remaining -= 1; }
                    Err(e) => { let _ = s.tx.send(Err(e)); failed.push(i); }
                }
            }
            for i in failed.into_iter().rev() {
                let s = active.remove(i);
                b.free_block_tables(&s.block_tables);
            }
            if active.is_empty() { return; }

            // Run one batched decode step
            let tokens: Vec<u32>           = active.iter().map(|s| s.current_token).collect();
            let bts:    Vec<&Vec<Vec<i32>>> = active.iter().map(|s| &s.block_tables).collect();
            let ctxs:   Vec<usize>          = active.iter().map(|s| s.context_len).collect();
            match b.forward_batch_decode_greedy(&tokens, &bts, &ctxs) {
                Ok(next) => {
                    for (s, nt) in active.iter_mut().zip(next) {
                        s.current_token = nt;
                    }
                }
                Err(e) => {
                    let msg = format!("{e}");
                    for s in active.drain(..) {
                        let _ = s.tx.send(Err(EngineError::backend(msg.clone())));
                        b.free_block_tables(&s.block_tables);
                    }
                }
            }
        }
    }

    // ── GgufNativeBackend ─────────────────────────────────────────────────────

    pub struct GgufNativeBackend {
        device_id: i32,
        inner: Arc<Mutex<Option<BackendInner>>>,
        /// Pool handle to inject before load(); also populated after load() for sharing.
        pool_slot: Arc<Mutex<Option<GpuPoolHandle>>>,
        batch_dec: Arc<BatchDecodeCoordinator>,
    }

    impl GgufNativeBackend {
        pub fn new(device_id: i32) -> Result<Self, EngineError> {
            CudaDevice::new(device_id as usize)
                .map_err(|e| EngineError::backend(format!("CUDA device {device_id}: {e}")))?;
            let inner = Arc::new(Mutex::new(None));
            let batch_dec = Arc::new(BatchDecodeCoordinator::spawn(Arc::clone(&inner)));
            Ok(Self {
                device_id,
                inner,
                pool_slot: Arc::new(Mutex::new(None)),
                batch_dec,
            })
        }

        /// Inject a shared pool handle before calling load(). If the pool geometry
        /// is incompatible with the model, load() will create a private pool instead.
        pub fn with_pool_handle(self, handle: GpuPoolHandle) -> Self {
            *self.pool_slot.lock().unwrap() = Some(handle);
            self
        }

        /// Return the active pool handle after load(), for registration and sharing.
        pub fn pool_handle(&self) -> Option<GpuPoolHandle> {
            self.pool_slot.lock().unwrap().clone()
        }

        fn extract_prompt(request: &InferenceRequest) -> Result<String, EngineError> {
            String::from_utf8(request.input.data.clone())
                .map_err(|e| EngineError::invalid_input(format!("Input is not valid UTF-8: {e}")))
        }

        fn decode_params(request: &InferenceRequest) -> (u32, SampleParams) {
            let meta = request.metadata.as_ref();
            let max_new = meta.and_then(|m| m.max_new_tokens).unwrap_or(512);
            (max_new, SampleParams::from_meta(meta))
        }

        fn text_to_packet(text: String) -> Result<BinaryTensorPacket, EngineError> {
            let data = text.into_bytes();
            let len = data.len() as i64;
            BinaryTensorPacket::new(vec![1, len], TensorDtype::Uint8, data)
                .map_err(|e| EngineError::backend(format!("output packet: {e}")))
        }
    }

    #[async_trait]
    impl Engine for GgufNativeBackend {
        async fn load(&mut self, model_path: &Path) -> Result<(), EngineError> {
            let path = model_path.to_owned();
            let device_id = self.device_id;
            let inner_arc = Arc::clone(&self.inner);
            let pool_slot = Arc::clone(&self.pool_slot);

            tokio::task::spawn_blocking(move || {
                // ── Load GGUF weights → dequantize to f16 ──────────────────
                log::info!("[gguf-native] Loading GGUF weights from {:?}", path);
                let cpu_weights = load_gguf_weights(&path)
                    .map_err(|e| EngineError::backend(format!("GGUF load: {e}")))?;
                let config = cpu_weights.config.clone();
                log::info!(
                    "[gguf-native] {} layers, {}Q/{}KV heads, h={}, vocab={}",
                    config.num_hidden_layers, config.num_attention_heads,
                    config.num_kv_heads(), config.hidden_size, config.vocab_size,
                );

                // ── Initialise llama.cpp backend for tokenization only ──────
                log::info!("[gguf-native] Loading llama.cpp model for tokenization");
                let llm_backend = get_llama_backend()?;
                let params = LlamaModelParams::default().with_n_gpu_layers(0); // CPU-only; no inference
                let llm_model = LlamaModel::load_from_file(&llm_backend, &path, &params)
                    .map_err(|e| EngineError::backend(format!("llama model (tokenizer): {e}")))?;
                let eos_token = llm_model.token_eos().0;

                // ── Upload weights to GPU ────────────────────────────────────
                let device = CudaDevice::new(device_id as usize)
                    .map_err(|e| EngineError::backend(format!("CUDA: {e}")))?;
                let blas = Arc::new(CudaBlas::new(device.clone())
                    .map_err(|e| EngineError::backend(format!("cuBLAS: {e}")))?);
                let weights = upload_weights(&device, &cpu_weights)?;
                drop(cpu_weights);

                // ── Block pool ───────────────────────────────────────────────
                let block_size = 16usize;
                let (block_pool, pool_cap): (Arc<GpuBlockPool>, Arc<AtomicUsize>) = {
                    let mut slot = pool_slot.lock().unwrap();
                    if let Some(ref handle) = *slot {
                        if handle.pool.is_compatible(config.num_kv_heads(), config.head_dim()) {
                            log::info!("[gguf-native] Attaching to shared GpuBlockPool ({} free, cap {})",
                                handle.pool.free_count(), handle.cap());
                            (handle.pool.clone(), handle.blocks_per_engine.clone())
                        } else {
                            log::warn!("[gguf-native] Pool geometry mismatch ({}h×{}d vs {}h×{}d), creating private pool",
                                handle.pool.num_kv_heads(), handle.pool.head_dim(),
                                config.num_kv_heads(), config.head_dim());
                            let num_blocks = kv_pool_block_count(&config, block_size);
                            let p = Arc::new(GpuBlockPool::new(device.clone(), num_blocks, block_size,
                                config.num_kv_heads(), config.head_dim())
                                .map_err(|e| EngineError::backend(format!("block pool: {e}")))?);
                            let h = GpuPoolHandle::private(p.clone());
                            let cap = h.blocks_per_engine.clone();
                            *slot = Some(h);
                            (p, cap)
                        }
                    } else {
                        let num_blocks = kv_pool_block_count(&config, block_size);
                        let p = Arc::new(GpuBlockPool::new(device.clone(), num_blocks, block_size,
                            config.num_kv_heads(), config.head_dim())
                            .map_err(|e| EngineError::backend(format!("block pool: {e}")))?);
                        let h = GpuPoolHandle::private(p.clone());
                        let cap = h.blocks_per_engine.clone();
                        *slot = Some(h);
                        (p, cap)
                    }
                };

                let h = config.hidden_size;
                let nq = config.num_attention_heads;
                let nkv = config.num_kv_heads();
                let hd = config.head_dim();
                let inter = config.intermediate_size;
                let vocab = config.vocab_size;
                let ctx_scalar_buf = device.htod_sync_copy(&[0i32])
                    .map_err(|e| EngineError::backend(format!("ctx buf: {e}")))?;
                let prefill = PrefillScratch::new(&device, h, nq * hd, nkv * hd, inter)?;
                let argmax_buf = device.alloc_zeros::<u32>(1)
                    .map_err(|e| EngineError::backend(format!("argmax buf: {e}")))?;
                // Batch decode GPU buffers — seeded at size 1; grow on first use
                let dae = |tag: &'static str| move |e| EngineError::backend(format!("dec {tag}: {e}"));
                let dec_batch_logits = device.alloc_zeros::<f16>(vocab).map_err(dae("logits"))?;
                let dec_batch_norm   = device.alloc_zeros::<f16>(h).map_err(dae("norm"))?;
                let hidden_buf    = device.alloc_zeros::<f16>(h)        .map_err(|e| EngineError::backend(format!("alloc: {e}")))?;
                let norm_buf      = device.alloc_zeros::<f16>(h)        .map_err(|e| EngineError::backend(format!("alloc: {e}")))?;
                let residual_buf  = device.alloc_zeros::<f16>(h)        .map_err(|e| EngineError::backend(format!("alloc: {e}")))?;
                let q_buf         = device.alloc_zeros::<f16>(nq * hd)  .map_err(|e| EngineError::backend(format!("alloc: {e}")))?;
                let k_buf         = device.alloc_zeros::<f16>(nkv * hd) .map_err(|e| EngineError::backend(format!("alloc: {e}")))?;
                let v_buf         = device.alloc_zeros::<f16>(nkv * hd) .map_err(|e| EngineError::backend(format!("alloc: {e}")))?;
                let attn_buf      = device.alloc_zeros::<f16>(nq * hd)  .map_err(|e| EngineError::backend(format!("alloc: {e}")))?;
                let gate_buf      = device.alloc_zeros::<f16>(inter)    .map_err(|e| EngineError::backend(format!("alloc: {e}")))?;
                let up_buf        = device.alloc_zeros::<f16>(inter)    .map_err(|e| EngineError::backend(format!("alloc: {e}")))?;
                let swiglu_buf    = device.alloc_zeros::<f16>(inter)    .map_err(|e| EngineError::backend(format!("alloc: {e}")))?;
                let ffn_input_buf = device.alloc_zeros::<f16>(h)        .map_err(|e| EngineError::backend(format!("alloc: {e}")))?;
                let ffn_out_buf   = device.alloc_zeros::<f16>(h)        .map_err(|e| EngineError::backend(format!("alloc: {e}")))?;
                let o_proj_buf    = device.alloc_zeros::<f16>(h)        .map_err(|e| EngineError::backend(format!("alloc: {e}")))?;
                let logits_buf    = device.alloc_zeros::<f16>(vocab)    .map_err(|e| EngineError::backend(format!("alloc: {e}")))?;

                let backend = BackendInner {
                    device, blas, config, weights, block_pool, pool_cap, allocated_blocks: 0,
                    llm_backend,
                    llm_model: Arc::new(llm_model),
                    eos_token,
                    hidden_buf, norm_buf, residual_buf, q_buf, k_buf, v_buf, attn_buf,
                    gate_buf, up_buf, swiglu_buf, ffn_input_buf, ffn_out_buf, o_proj_buf, logits_buf,
                    ctx_scalar_buf,
                    gpu_block_tables:    Vec::new(),
                    gpu_block_table_len: 0,
                    sessions:       HashMap::new(),
                    rng:            rand::rngs::SmallRng::from_entropy(),
                    prefill,
                    argmax_buf,
                    dec_batch_cap: 1, dec_batch_logits, dec_batch_norm,
                };
                *inner_arc.lock().unwrap() = Some(backend);
                log::info!("[gguf-native] Ready");
                Ok(())
            })
            .await
            .map_err(|e| EngineError::backend(format!("load task: {e}")))?
        }

        fn infer(&self, request: &InferenceRequest) -> Result<BinaryTensorPacket, EngineError> {
            let prompt = Self::extract_prompt(request)?;
            let (max_new, sp) = Self::decode_params(request);
            let greedy = sp.temperature < 1e-6;
            let sid = request.session_id.clone();

            // Non-greedy: single-sequence path (holds lock throughout)
            if !greedy {
                let mut guard = self.inner.lock().unwrap();
                let b = guard.as_mut().ok_or(EngineError::ModelNotLoaded)?;
                let prompt_ids = b.llm_model.str_to_token(&prompt, AddBos::Always)
                    .map_err(|e| EngineError::invalid_input(format!("tokenize: {e}")))?
                    .into_iter().map(|t| t.0 as u32).collect::<Vec<_>>();
                let text = if let Some(ref sid_str) = sid {
                    let mut sess = b.sessions.remove(sid_str.as_str())
                        .unwrap_or(SessionState { block_tables: Vec::new(), context_len: 0 });
                    let r = b.run_decode(&prompt_ids, &mut sess, max_new, &sp,
                        request.cancellation.as_ref(), None);
                    b.sessions.insert(sid_str.clone(), sess);
                    r?
                } else {
                    let mut tmp = SessionState { block_tables: Vec::new(), context_len: 0 };
                    let r = b.run_decode(&prompt_ids, &mut tmp, max_new, &sp,
                        request.cancellation.as_ref(), None);
                    b.free_block_tables(&tmp.block_tables);
                    r?
                };
                return Self::text_to_packet(text);
            }

            // Greedy: prefill under lock, then hand off decode to coordinator
            let (first_token, block_tables, context_len, eos) = {
                let mut guard = self.inner.lock().unwrap();
                let b = guard.as_mut().ok_or(EngineError::ModelNotLoaded)?;
                let prompt_ids = b.llm_model.str_to_token(&prompt, AddBos::Always)
                    .map_err(|e| EngineError::invalid_input(format!("tokenize: {e}")))?
                    .into_iter().map(|t| t.0 as u32).collect::<Vec<_>>();
                let (mut block_tables, ctx) = if let Some(ref s) = sid {
                    let sess = b.sessions.remove(s.as_str())
                        .unwrap_or(SessionState { block_tables: Vec::new(), context_len: 0 });
                    (sess.block_tables, sess.context_len)
                } else {
                    (Vec::new(), 0usize)
                };
                for i in 0..prompt_ids.len() {
                    b.ensure_block(&mut block_tables, ctx + i)?;
                }
                let ft = b.forward_prefill_greedy(&prompt_ids, ctx as u32, &block_tables)?;
                let new_ctx = ctx + prompt_ids.len();
                let eos = b.eos_token as u32;
                (ft, block_tables, new_ctx, eos)
            };

            if max_new == 0 {
                let mut guard = self.inner.lock().unwrap();
                if let Some(b) = guard.as_mut() {
                    if let Some(ref s) = sid {
                        b.sessions.insert(s.clone(), SessionState { block_tables, context_len });
                    } else {
                        b.free_block_tables(&block_tables);
                    }
                }
                return Self::text_to_packet(String::new());
            }

            let (tx, rx) = std::sync::mpsc::sync_channel::<Result<String, EngineError>>(128);
            self.batch_dec.submit(DecodeState {
                current_token: first_token, block_tables, context_len,
                max_remaining: max_new, eos, session_id: sid, tx,
            });

            let mut out = String::new();
            for piece in rx { out.push_str(&piece?); }
            Self::text_to_packet(out)
        }

        fn infer_stream(&self, request: &InferenceRequest) -> EngineStream {
            let inner_arc = Arc::clone(&self.inner);
            let batch_dec = Arc::clone(&self.batch_dec);

            let prompt = match Self::extract_prompt(request) {
                Ok(p) => p,
                Err(e) => return Box::pin(stream! { yield Err(e); }),
            };
            let (max_new, sp) = Self::decode_params(request);
            let greedy = sp.temperature < 1e-6;
            let sid    = request.session_id.clone();
            let cancel = request.cancellation.clone();

            let (tx, rx) = std::sync::mpsc::channel::<Result<String, EngineError>>();

            std::thread::spawn(move || {
                // Non-greedy: single-sequence path
                if !greedy {
                    let mut guard = match inner_arc.lock() {
                        Ok(g) => g,
                        Err(_) => { let _ = tx.send(Err(EngineError::backend("mutex poisoned"))); return; }
                    };
                    let b = match guard.as_mut() {
                        Some(b) => b,
                        None => { let _ = tx.send(Err(EngineError::ModelNotLoaded)); return; }
                    };
                    let prompt_ids = match b.llm_model.str_to_token(&prompt, AddBos::Always) {
                        Ok(ids) => ids.into_iter().map(|t| t.0 as u32).collect::<Vec<_>>(),
                        Err(e) => { let _ = tx.send(Err(EngineError::invalid_input(format!("tokenize: {e}")))); return; }
                    };
                    let result = if let Some(ref sid_str) = sid {
                        let mut sess = b.sessions.remove(sid_str.as_str())
                            .unwrap_or(SessionState { block_tables: Vec::new(), context_len: 0 });
                        let r = b.run_decode(&prompt_ids, &mut sess, max_new, &sp, cancel.as_ref(), Some(&tx));
                        b.sessions.insert(sid_str.clone(), sess);
                        r
                    } else {
                        let mut tmp = SessionState { block_tables: Vec::new(), context_len: 0 };
                        let r = b.run_decode(&prompt_ids, &mut tmp, max_new, &sp, cancel.as_ref(), Some(&tx));
                        b.free_block_tables(&tmp.block_tables);
                        r
                    };
                    if let Err(e) = result { let _ = tx.send(Err(e)); }
                    return;
                }

                // Greedy: prefill under lock, then coordinator
                let result: Result<(), EngineError> = (|| {
                    let (first_token, block_tables, context_len, eos) = {
                        let mut guard = inner_arc.lock().unwrap();
                        let b = guard.as_mut().ok_or(EngineError::ModelNotLoaded)?;
                        let prompt_ids = b.llm_model.str_to_token(&prompt, AddBos::Always)
                            .map_err(|e| EngineError::invalid_input(format!("tokenize: {e}")))?
                            .into_iter().map(|t| t.0 as u32).collect::<Vec<_>>();
                        let (mut block_tables, ctx) = if let Some(ref s) = sid {
                            let sess = b.sessions.remove(s.as_str())
                                .unwrap_or(SessionState { block_tables: Vec::new(), context_len: 0 });
                            (sess.block_tables, sess.context_len)
                        } else {
                            (Vec::new(), 0usize)
                        };
                        for i in 0..prompt_ids.len() {
                            b.ensure_block(&mut block_tables, ctx + i)?;
                        }
                        let ft = b.forward_prefill_greedy(&prompt_ids, ctx as u32, &block_tables)?;
                        let new_ctx = ctx + prompt_ids.len();
                        let eos = b.eos_token as u32;
                        Ok::<_, EngineError>((ft, block_tables, new_ctx, eos))
                    }?;

                    if max_new == 0 {
                        let mut guard = inner_arc.lock().unwrap();
                        if let Some(b) = guard.as_mut() {
                            if let Some(ref s) = sid {
                                b.sessions.insert(s.clone(), SessionState { block_tables, context_len });
                            } else {
                                b.free_block_tables(&block_tables);
                            }
                        }
                        return Ok(());
                    }

                    let (dec_tx, dec_rx) = std::sync::mpsc::sync_channel::<Result<String, EngineError>>(128);
                    batch_dec.submit(DecodeState {
                        current_token: first_token, block_tables, context_len,
                        max_remaining: max_new, eos, session_id: sid, tx: dec_tx,
                    });
                    for piece in dec_rx {
                        match piece {
                            Ok(p)  => { if tx.send(Ok(p)).is_err() { break; } }
                            Err(e) => { let _ = tx.send(Err(e)); break; }
                        }
                    }
                    Ok(())
                })();
                if let Err(e) = result { let _ = tx.send(Err(e)); }
            });

            // Bridge blocking std::mpsc → async stream.
            let (tok_tx, mut tok_rx) = tokio::sync::mpsc::channel::<Result<String, EngineError>>(64);
            std::thread::spawn(move || {
                for piece in rx {
                    if tok_tx.blocking_send(piece).is_err() { break; }
                }
            });

            Box::pin(stream! {
                while let Some(result) = tok_rx.recv().await {
                    let piece = result?;
                    let data = piece.into_bytes();
                    let len = data.len() as i64;
                    yield BinaryTensorPacket::new(vec![1, len], TensorDtype::Uint8, data)
                        .map_err(|e| EngineError::backend(format!("output packet: {e}")));
                }
            })
        }

        fn unload(&mut self) {
            *self.inner.lock().unwrap() = None;
            log::info!("[gguf-native] Unloaded");
        }

        fn metrics(&self) -> EngineMetrics {
            let active = self.batch_dec.active_count();
            let g = self.inner.lock().unwrap();
            let (total, free, sessions, used_bytes, capacity_bytes) = g.as_ref()
                .map(|b| {
                    (
                        b.block_pool.total_blocks(),
                        b.block_pool.free_count(),
                        b.sessions.len(),
                        b.block_pool.used_bytes(),
                        b.block_pool.capacity_bytes(),
                    )
                })
                .unwrap_or((0, 0, 0, 0, 0));
            EngineMetrics {
                memory_usage: used_bytes,
                batch_size: active.max(1),
                kv_cache_bytes_used: used_bytes,
                kv_cache_bytes_capacity: capacity_bytes,
                kv_cache_blocks_total: total, kv_cache_blocks_free: free,
                kv_cache_sequences: sessions + active,
                kv_cache_evicted_blocks: 0, kv_cache_evicted_sequences: 0,
                kv_cache_packed_layers: 0,
                ..EngineMetrics::new()
            }
        }

        fn batching_policy(&self) -> BatchingPolicy {
            BatchingPolicy::continuous(self.batch_dec.active_count().max(1))
        }

        fn health_check(&self) -> Result<(), EngineError> {
            if self.inner.lock().unwrap().is_some() { Ok(()) }
            else { Err(EngineError::ModelNotLoaded) }
        }

        fn model_info(&self) -> Option<EngineModelInfo> {
            let g = self.inner.lock().unwrap();
            let b = g.as_ref()?;
            let arch = b.config.architectures.first().cloned().unwrap_or_else(|| "gguf".into());
            let nl  = b.config.num_hidden_layers;
            let bs  = b.block_pool.block_size();
            let bps = (b.config.max_position_embeddings + bs - 1) / bs;
            let peak = (b.block_pool.total_blocks() / (nl * bps).max(1)).max(1) as u32;
            Some(EngineModelInfo {
                input_names: vec!["text".into()],
                output_names: vec!["text".into()],
                input_shapes: vec![vec![-1]],
                output_shapes: vec![vec![-1]],
                input_dtypes: vec!["uint8".into()],
                output_dtypes: vec!["uint8".into()],
                framework: Some("gguf-native".into()),
                model_version: Some(arch),
                peak_concurrency: Some(peak),
            })
        }
    }
}

#[cfg(feature = "gguf-native")]
pub use inner::GgufNativeBackend;
