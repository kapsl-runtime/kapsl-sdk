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
    use std::sync::{Arc, Mutex};

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
        BinaryTensorPacket, Engine, EngineError, EngineMetrics, EngineModelInfo, EngineStream,
        InferenceRequest, RequestMetadata, TensorDtype,
    };
    use kapsl_hal::gpu_arena::GpuBlockPool;
    use kapsl_kernels::cuda_kernels::{
        launch_argmax, launch_batch_kv_write, launch_batch_rope, launch_fused_swiglu,
        launch_paged_attention, launch_prefill_attention, launch_residual_add, launch_rms_norm,
        ArgmaxParams, BatchKvWriteParams, BatchRopeParams, PagedAttentionParams, PrefillAttnParams,
        RmsNormParams,
    };
    use kapsl_loader::{load_gguf_weights, ModelConfig, TensorData};

    // ── GPU weights ──────────────────────────────────────────────────────────

    struct GpuLayerWeights {
        input_layernorm: CudaSlice<f16>,
        q_proj: CudaSlice<f16>,
        k_proj: CudaSlice<f16>,
        v_proj: CudaSlice<f16>,
        o_proj: CudaSlice<f16>,
        post_attention_layernorm: CudaSlice<f16>,
        gate_proj: CudaSlice<f16>,
        up_proj: CudaSlice<f16>,
        down_proj: CudaSlice<f16>,
    }

    struct GpuModelWeights {
        embed_tokens: CudaSlice<f16>,
        layers: Vec<GpuLayerWeights>,
        norm: CudaSlice<f16>,
        lm_head: CudaSlice<f16>,
    }

    fn upload_tensor(device: &Arc<CudaDevice>, t: &TensorData) -> Result<CudaSlice<f16>, EngineError> {
        device
            .htod_sync_copy(&t.to_f16_vec())
            .map_err(|e| EngineError::backend(format!("GPU upload: {e}")))
    }

    fn upload_weights(
        device: &Arc<CudaDevice>,
        w: &kapsl_loader::ModelWeights,
    ) -> Result<GpuModelWeights, EngineError> {
        let embed_tokens = upload_tensor(device, &w.embed_tokens)?;
        let norm = upload_tensor(device, &w.norm)?;
        let lm_head = upload_tensor(device, &w.lm_head)?;
        let mut layers = Vec::with_capacity(w.layers.len());
        for (i, l) in w.layers.iter().enumerate() {
            log::info!("[gguf-native] Uploading layer {}/{}", i + 1, w.layers.len());
            layers.push(GpuLayerWeights {
                input_layernorm:          upload_tensor(device, &l.input_layernorm)?,
                q_proj:                   upload_tensor(device, &l.q_proj)?,
                k_proj:                   upload_tensor(device, &l.k_proj)?,
                v_proj:                   upload_tensor(device, &l.v_proj)?,
                o_proj:                   upload_tensor(device, &l.o_proj)?,
                post_attention_layernorm: upload_tensor(device, &l.post_attention_layernorm)?,
                gate_proj:                upload_tensor(device, &l.gate_proj)?,
                up_proj:                  upload_tensor(device, &l.up_proj)?,
                down_proj:                upload_tensor(device, &l.down_proj)?,
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
        block_pool: GpuBlockPool,
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
            if block_tables[0].len() <= logical {
                for l in 0..num_layers {
                    let phys = self.block_pool.alloc_block()
                        .map_err(|e| EngineError::backend(format!("block alloc: {e}")))?;
                    block_tables[l].push(phys as i32);
                }
            }
            Ok(())
        }

        fn free_block_tables(&mut self, block_tables: &[Vec<i32>]) {
            for lt in block_tables {
                for &p in lt { self.block_pool.free_block(p as u32); }
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

        // ── cuBLAS GEMM helper ────────────────────────────────────────────────

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
                    cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_T,
                    cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_N,
                    out_dim, batch, in_dim,
                    &f16::from_f32(1.0),
                    weight, lda,
                    input,  ldb,
                    &f16::from_f32(0.0),
                    out,    ldc,
                )
                .map_err(|e| EngineError::backend(format!("{label} gemm: {e}")))
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
                    out: &mut self.prefill.norm, input: &self.prefill.hidden,
                    weight: &layer.input_layernorm,
                    rows: n as u32, dim: h as u32, eps,
                }).map_err(e)?;

                Self::gemm(&blas, (num_q * head_dim) as i32, n as i32, h as i32,
                    &layer.q_proj, h as i32, &self.prefill.norm, h as i32,
                    &mut self.prefill.q_all, (num_q * head_dim) as i32, "Q")?;
                Self::gemm(&blas, (num_kv * head_dim) as i32, n as i32, h as i32,
                    &layer.k_proj, h as i32, &self.prefill.norm, h as i32,
                    &mut self.prefill.k_all, (num_kv * head_dim) as i32, "K")?;
                Self::gemm(&blas, (num_kv * head_dim) as i32, n as i32, h as i32,
                    &layer.v_proj, h as i32, &self.prefill.norm, h as i32,
                    &mut self.prefill.v_all, (num_kv * head_dim) as i32, "V")?;

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
                Self::gemm(&blas, h as i32, n as i32, (num_q * head_dim) as i32,
                    &layer.o_proj, (num_q * head_dim) as i32,
                    &self.prefill.attn_out, (num_q * head_dim) as i32,
                    &mut self.prefill.o_out, h as i32, "O")?;

                launch_residual_add(&self.device, &mut self.prefill.residual,
                    &self.prefill.hidden, &self.prefill.o_out, (n * h) as u32).map_err(e)?;

                let layer = &self.weights.layers[layer_idx];
                launch_rms_norm(&self.device, &mut RmsNormParams {
                    out: &mut self.prefill.ffn_input, input: &self.prefill.residual,
                    weight: &layer.post_attention_layernorm,
                    rows: n as u32, dim: h as u32, eps,
                }).map_err(e)?;

                let layer = &self.weights.layers[layer_idx];
                Self::gemm(&blas, inter as i32, n as i32, h as i32,
                    &layer.gate_proj, h as i32, &self.prefill.ffn_input, h as i32,
                    &mut self.prefill.gate_out, inter as i32, "gate")?;
                Self::gemm(&blas, inter as i32, n as i32, h as i32,
                    &layer.up_proj, h as i32, &self.prefill.ffn_input, h as i32,
                    &mut self.prefill.up_out, inter as i32, "up")?;

                launch_fused_swiglu(&self.device, &mut self.prefill.swiglu_out,
                    &self.prefill.gate_out, &self.prefill.up_out, (n * inter) as u32).map_err(e)?;

                let layer = &self.weights.layers[layer_idx];
                Self::gemm(&blas, h as i32, n as i32, inter as i32,
                    &layer.down_proj, inter as i32, &self.prefill.swiglu_out, inter as i32,
                    &mut self.prefill.ffn_out, h as i32, "down")?;

                launch_residual_add(&self.device, &mut self.prefill.hidden,
                    &self.prefill.residual, &self.prefill.ffn_out, (n * h) as u32).map_err(e)?;
            }

            let last_off = (n - 1) * h;
            let last_hidden = self.prefill.hidden.slice(last_off..last_off + h);
            launch_rms_norm(&self.device, &mut RmsNormParams {
                out: &mut self.norm_buf, input: &last_hidden,
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
                    out: &mut self.norm_buf, input: &self.hidden_buf,
                    weight: &layer.input_layernorm,
                    rows: 1, dim: h as u32, eps,
                }).map_err(e)?;

                Self::gemm(&blas, q_dim as i32, 1, h as i32,
                    &layer.q_proj, h as i32, &self.norm_buf, h as i32,
                    &mut self.q_buf, q_dim as i32, "Q")?;
                Self::gemm(&blas, kv_dim as i32, 1, h as i32,
                    &layer.k_proj, h as i32, &self.norm_buf, h as i32,
                    &mut self.k_buf, kv_dim as i32, "K")?;
                Self::gemm(&blas, kv_dim as i32, 1, h as i32,
                    &layer.v_proj, h as i32, &self.norm_buf, h as i32,
                    &mut self.v_buf, kv_dim as i32, "V")?;

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
                Self::gemm(&blas, h as i32, 1, q_dim as i32,
                    &layer.o_proj, q_dim as i32, &self.attn_buf, q_dim as i32,
                    &mut self.o_proj_buf, h as i32, "O")?;

                launch_residual_add(&self.device, &mut self.residual_buf,
                    &self.hidden_buf, &self.o_proj_buf, h as u32).map_err(e)?;

                let layer = &self.weights.layers[layer_idx];
                launch_rms_norm(&self.device, &mut RmsNormParams {
                    out: &mut self.ffn_input_buf, input: &self.residual_buf,
                    weight: &layer.post_attention_layernorm,
                    rows: 1, dim: h as u32, eps,
                }).map_err(e)?;

                let layer = &self.weights.layers[layer_idx];
                Self::gemm(&blas, inter as i32, 1, h as i32,
                    &layer.gate_proj, h as i32, &self.ffn_input_buf, h as i32,
                    &mut self.gate_buf, inter as i32, "gate")?;
                Self::gemm(&blas, inter as i32, 1, h as i32,
                    &layer.up_proj, h as i32, &self.ffn_input_buf, h as i32,
                    &mut self.up_buf, inter as i32, "up")?;

                launch_fused_swiglu(&self.device, &mut self.swiglu_buf,
                    &self.gate_buf, &self.up_buf, inter as u32).map_err(e)?;

                let layer = &self.weights.layers[layer_idx];
                Self::gemm(&blas, h as i32, 1, inter as i32,
                    &layer.down_proj, inter as i32, &self.swiglu_buf, inter as i32,
                    &mut self.ffn_out_buf, h as i32, "down")?;

                launch_residual_add(&self.device, &mut self.hidden_buf,
                    &self.residual_buf, &self.ffn_out_buf, h as u32).map_err(e)?;
            }

            launch_rms_norm(&self.device, &mut RmsNormParams {
                out: &mut self.norm_buf, input: &self.hidden_buf,
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

    // ── GgufNativeBackend ─────────────────────────────────────────────────────

    pub struct GgufNativeBackend {
        device_id: i32,
        inner: Arc<Mutex<Option<BackendInner>>>,
    }

    impl GgufNativeBackend {
        pub fn new(device_id: i32) -> Result<Self, EngineError> {
            CudaDevice::new(device_id as usize)
                .map_err(|e| EngineError::backend(format!("CUDA device {device_id}: {e}")))?;
            Ok(Self { device_id, inner: Arc::new(Mutex::new(None)) })
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
                let llm_backend = LlamaBackend::init()
                    .map_err(|e| EngineError::backend(format!("llama backend: {e}")))?;
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
                let bps = (config.max_position_embeddings + block_size - 1) / block_size;
                let num_blocks = config.num_hidden_layers * 8 * bps;
                let block_pool = GpuBlockPool::new(
                    device.clone(), num_blocks, block_size,
                    config.num_kv_heads(), config.head_dim(),
                ).map_err(|e| EngineError::backend(format!("block pool: {e}")))?;

                let h = config.hidden_size;
                let nq = config.num_attention_heads;
                let nkv = config.num_kv_heads();
                let hd = config.head_dim();
                let inter = config.intermediate_size;
                let vocab = config.vocab_size;
                let alloc = |n: usize| device.alloc_zeros::<f16>(n)
                    .map_err(|e| EngineError::backend(format!("alloc: {e}")));

                let ctx_scalar_buf = device.htod_sync_copy(&[0i32])
                    .map_err(|e| EngineError::backend(format!("ctx buf: {e}")))?;
                let prefill = PrefillScratch::new(&device, h, nq * hd, nkv * hd, inter)?;
                let argmax_buf = device.alloc_zeros::<u32>(1)
                    .map_err(|e| EngineError::backend(format!("argmax buf: {e}")))?;

                let backend = BackendInner {
                    device, blas, config, weights, block_pool,
                    llm_backend: Arc::new(llm_backend),
                    llm_model: Arc::new(llm_model),
                    eos_token,
                    hidden_buf:     alloc(h)?,
                    norm_buf:       alloc(h)?,
                    residual_buf:   alloc(h)?,
                    q_buf:          alloc(nq * hd)?,
                    k_buf:          alloc(nkv * hd)?,
                    v_buf:          alloc(nkv * hd)?,
                    attn_buf:       alloc(nq * hd)?,
                    gate_buf:       alloc(inter)?,
                    up_buf:         alloc(inter)?,
                    swiglu_buf:     alloc(inter)?,
                    ffn_input_buf:  alloc(h)?,
                    ffn_out_buf:    alloc(h)?,
                    o_proj_buf:     alloc(h)?,
                    logits_buf:     alloc(vocab)?,
                    ctx_scalar_buf,
                    gpu_block_tables:    Vec::new(),
                    gpu_block_table_len: 0,
                    sessions:       HashMap::new(),
                    rng:            rand::rngs::SmallRng::from_entropy(),
                    prefill,
                    argmax_buf,
                };
                *inner_arc.lock().unwrap() = Some(backend);
                log::info!("[gguf-native] Ready");
                Ok(())
            })
            .await
            .map_err(|e| EngineError::backend(format!("load task: {e}")))?
        }

        fn infer(&self, request: &InferenceRequest) -> Result<BinaryTensorPacket, EngineError> {
            let mut guard = self.inner.lock().unwrap();
            let inner = guard.as_mut().ok_or(EngineError::ModelNotLoaded)?;

            let prompt = Self::extract_prompt(request)?;
            let (max_new, sp) = Self::decode_params(request);
            let sid = request.session_id.as_deref();

            let prompt_ids: Vec<u32> = inner.llm_model
                .str_to_token(&prompt, AddBos::Always)
                .map_err(|e| EngineError::invalid_input(format!("tokenize: {e}")))?
                .into_iter().map(|t| t.0 as u32).collect();

            let text = if let Some(sid) = sid {
                let mut session = inner.sessions.remove(sid)
                    .unwrap_or_else(|| SessionState { block_tables: Vec::new(), context_len: 0 });
                let r = inner.run_decode(&prompt_ids, &mut session, max_new, &sp,
                    request.cancellation.as_ref(), None);
                inner.sessions.insert(sid.to_string(), session);
                r?
            } else {
                let mut tmp = SessionState { block_tables: Vec::new(), context_len: 0 };
                let r = inner.run_decode(&prompt_ids, &mut tmp, max_new, &sp,
                    request.cancellation.as_ref(), None);
                inner.free_block_tables(&tmp.block_tables);
                r?
            };

            Self::text_to_packet(text)
        }

        fn infer_stream(&self, request: &InferenceRequest) -> EngineStream {
            let inner_arc = Arc::clone(&self.inner);

            let prompt = match Self::extract_prompt(request) {
                Ok(p) => p,
                Err(e) => return Box::pin(stream! { yield Err(e); }),
            };
            let (max_new, sp) = Self::decode_params(request);
            let sid = request.session_id.clone();
            let cancel = request.cancellation.clone();

            let (tx, rx) = std::sync::mpsc::channel::<Result<String, EngineError>>();

            std::thread::spawn(move || {
                let mut guard = match inner_arc.lock() {
                    Ok(g) => g,
                    Err(_) => { let _ = tx.send(Err(EngineError::backend("mutex poisoned"))); return; }
                };
                let b = match guard.as_mut() {
                    Some(b) => b,
                    None => { let _ = tx.send(Err(EngineError::ModelNotLoaded)); return; }
                };

                let prompt_ids: Vec<u32> = match b.llm_model.str_to_token(&prompt, AddBos::Always) {
                    Ok(ids) => ids.into_iter().map(|t| t.0 as u32).collect(),
                    Err(e) => {
                        let _ = tx.send(Err(EngineError::invalid_input(format!("tokenize: {e}"))));
                        return;
                    }
                };

                let result = if let Some(ref sid) = sid {
                    let mut session = b.sessions.remove(sid.as_str())
                        .unwrap_or_else(|| SessionState { block_tables: Vec::new(), context_len: 0 });
                    let r = b.run_decode(&prompt_ids, &mut session, max_new, &sp,
                        cancel.as_ref(), Some(&tx));
                    b.sessions.insert(sid.clone(), session);
                    r
                } else {
                    let mut tmp = SessionState { block_tables: Vec::new(), context_len: 0 };
                    let r = b.run_decode(&prompt_ids, &mut tmp, max_new, &sp,
                        cancel.as_ref(), Some(&tx));
                    b.free_block_tables(&tmp.block_tables);
                    r
                };

                if let Err(e) = result {
                    let _ = tx.send(Err(e));
                }
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
            let g = self.inner.lock().unwrap();
            let (total, free, sessions) = g.as_ref()
                .map(|b| (b.block_pool.total_blocks(), b.block_pool.free_count(), b.sessions.len()))
                .unwrap_or((0, 0, 0));
            EngineMetrics {
                inference_time: 0.0, memory_usage: 0, gpu_utilization: 0.0,
                throughput: 0.0, batch_size: 1, queue_depth: 0, error_rate: 0.0,
                collected_at_ms: 0, kv_cache_bytes_used: 0, kv_cache_bytes_capacity: 0,
                kv_cache_blocks_total: total, kv_cache_blocks_free: free,
                kv_cache_sequences: sessions,
                kv_cache_evicted_blocks: 0, kv_cache_evicted_sequences: 0,
            }
        }

        fn health_check(&self) -> Result<(), EngineError> {
            if self.inner.lock().unwrap().is_some() { Ok(()) }
            else { Err(EngineError::ModelNotLoaded) }
        }

        fn model_info(&self) -> Option<EngineModelInfo> {
            let g = self.inner.lock().unwrap();
            let cfg = g.as_ref().map(|b| b.config.clone())?;
            let arch = cfg.architectures.first().cloned().unwrap_or_else(|| "gguf".into());
            Some(EngineModelInfo {
                input_names: vec!["text".into()],
                output_names: vec!["text".into()],
                input_shapes: vec![vec![-1]],
                output_shapes: vec![vec![-1]],
                input_dtypes: vec!["uint8".into()],
                output_dtypes: vec!["uint8".into()],
                framework: Some("gguf-native".into()),
                model_version: Some(arch),
                peak_concurrency: Some(1),
            })
        }
    }
}

#[cfg(feature = "gguf-native")]
pub use inner::GgufNativeBackend;
