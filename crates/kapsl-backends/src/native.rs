//! Native CUDA inference backend — full greedy + sampled decode.
//!
//! # What this implements
//!
//! * **Batch prefill**: all prompt tokens run in a single batched GEMM +
//!   causal prefill-attention kernel, not N separate decode steps.
//! * **Paged decode**: single-token decode using paged KV cache.
//! * **Sampling**: greedy, temperature, top-k, top-p (nucleus).
//! * **Per-token streaming**: `infer_stream` runs the decode loop in a
//!   `spawn_blocking` thread and yields tokens via a channel.
//! * **Multi-turn sessions**: block tables and context length are preserved
//!   across `infer` calls for the same `session_id`.
//!
//! # Input / output contract
//!
//! Input `BinaryTensorPacket`: `dtype = Int32`, shape `[prompt_len]`,
//! data = little-endian `i32` token IDs.
//!
//! Output `BinaryTensorPacket`: `dtype = Int32`, shape `[num_generated]`,
//! data = little-endian `i32` generated token IDs.

#[cfg(feature = "native")]
mod inner {
    use std::collections::HashMap;
    use std::path::Path;
    use std::sync::{Arc, Mutex};

    use async_trait::async_trait;
    use cudarc::cublas::{CudaBlas, Gemm};
    use cudarc::driver::{CudaDevice, CudaSlice, DeviceSlice};
    use half::f16;
    use rand::{Rng, SeedableRng};

    use kapsl_engine_api::{
        BinaryTensorPacket, EngineError, EngineMetrics, EngineModelInfo, EngineStream,
        InferenceRequest, RequestMetadata, TensorDtype,
    };
    use kapsl_hal::gpu_arena::GpuBlockPool;
    use kapsl_kernels::cuda_kernels::{
        launch_batch_kv_write, launch_batch_rope, launch_fused_swiglu, launch_paged_attention,
        launch_prefill_attention, launch_residual_add, launch_rms_norm, BatchKvWriteParams,
        BatchRopeParams, PagedAttentionParams, PrefillAttnParams, RmsNormParams,
    };
    use kapsl_loader::{load_safetensors, ModelConfig, ModelWeights, TensorData};

    // ── GPU weights ──────────────────────────────────────────────────────

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

    fn upload_weights(device: &Arc<CudaDevice>, w: &ModelWeights) -> Result<GpuModelWeights, EngineError> {
        let embed_tokens = upload_tensor(device, &w.embed_tokens)?;
        let norm = upload_tensor(device, &w.norm)?;
        let lm_head = upload_tensor(device, &w.lm_head)?;
        let mut layers = Vec::with_capacity(w.layers.len());
        for (i, l) in w.layers.iter().enumerate() {
            log::info!("Uploading layer {}/{}", i + 1, w.layers.len());
            layers.push(GpuLayerWeights {
                input_layernorm: upload_tensor(device, &l.input_layernorm)?,
                q_proj: upload_tensor(device, &l.q_proj)?,
                k_proj: upload_tensor(device, &l.k_proj)?,
                v_proj: upload_tensor(device, &l.v_proj)?,
                o_proj: upload_tensor(device, &l.o_proj)?,
                post_attention_layernorm: upload_tensor(device, &l.post_attention_layernorm)?,
                gate_proj: upload_tensor(device, &l.gate_proj)?,
                up_proj: upload_tensor(device, &l.up_proj)?,
                down_proj: upload_tensor(device, &l.down_proj)?,
            });
        }
        Ok(GpuModelWeights { embed_tokens, layers, norm, lm_head })
    }

    // ── Sampling ─────────────────────────────────────────────────────────

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

    // ── Session state ─────────────────────────────────────────────────────

    struct SessionState {
        /// [layer_idx][logical_block] = physical_block_id
        block_tables: Vec<Vec<i32>>,
        context_len: usize,
    }

    // ── BackendInner ──────────────────────────────────────────────────────

    struct BackendInner {
        device: Arc<CudaDevice>,
        blas: Arc<CudaBlas>,
        config: ModelConfig,
        weights: GpuModelWeights,
        block_pool: GpuBlockPool,
        /// Pre-allocated single-token activation buffers.
        hidden_buf: CudaSlice<f16>,
        norm_buf: CudaSlice<f16>,
        residual_buf: CudaSlice<f16>,
        q_buf: CudaSlice<f16>,
        k_buf: CudaSlice<f16>,
        v_buf: CudaSlice<f16>,
        /// Session ID → preserved KV state for multi-turn conversations.
        sessions: HashMap<String, SessionState>,
        rng: rand::rngs::SmallRng,
        /// Next model weights pre-loaded into CPU RAM, ready for fast GPU swap.
        staged: Option<ModelWeights>,
    }

    impl BackendInner {
        // ── Block management ─────────────────────────────────────────────

        /// Ensure `block_tables` has physical blocks covering `position`.
        /// All layers are allocated in sync so logical indices stay consistent.
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

        // ── Hot-swap ──────────────────────────────────────────────────────────

        /// Transfer staged CPU weights to the GPU and replace the live weights.
        /// Invalidates all sessions — the KV cache layout is tied to the old weights.
        fn activate_staged(&mut self) -> Result<(), EngineError> {
            let staged = self.staged.take()
                .ok_or_else(|| EngineError::backend("no model staged; call stage() first"))?;

            // Configs must be compatible (same arch, heads, hidden size).
            let sc = &staged.config;
            let cc = &self.config;
            if sc.num_hidden_layers != cc.num_hidden_layers
                || sc.hidden_size != cc.hidden_size
                || sc.num_attention_heads != cc.num_attention_heads
            {
                return Err(EngineError::backend(format!(
                    "staged model architecture mismatch: \
                     layers {}/{}, hidden {}/{}, heads {}/{}",
                    sc.num_hidden_layers, cc.num_hidden_layers,
                    sc.hidden_size, cc.hidden_size,
                    sc.num_attention_heads, cc.num_attention_heads,
                )));
            }

            log::info!("NativeBackend: activating staged weights (PCIe transfer)…");
            let new_weights = upload_weights(&self.device, &staged)?;
            self.weights = new_weights;
            self.config = staged.config;

            // Invalidate all KV state — block contents reference old weights.
            let all_bts: Vec<_> = self.sessions.values()
                .map(|s| s.block_tables.clone())
                .collect();
            for bt in all_bts {
                self.free_block_tables(&bt);
            }
            self.sessions.clear();

            log::info!("NativeBackend: swap complete");
            Ok(())
        }

        // ── Sampling ─────────────────────────────────────────────────────

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

            // Top-k
            if p.top_k > 0 && p.top_k < scores.len() {
                let mut sorted = scores.clone();
                sorted.sort_unstable_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
                let thresh = sorted[p.top_k - 1];
                for s in &mut scores { if *s < thresh { *s = f32::NEG_INFINITY; } }
            }

            // Softmax
            let max_s = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut probs: Vec<f32> = scores.iter().map(|&s| (s - max_s).exp()).collect();
            let sum: f32 = probs.iter().sum();
            if sum <= 0.0 { return Self::greedy(logits); }
            for p2 in &mut probs { *p2 /= sum; }

            // Top-p (nucleus)
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

            // Multinomial
            let r: f32 = self.rng.gen();
            let mut cum = 0.0f32;
            for (i, &pr) in probs.iter().enumerate() {
                cum += pr;
                if r <= cum { return i as u32; }
            }
            (probs.len() - 1) as u32
        }

        // ── cuBLAS GEMM helper ────────────────────────────────────────────
        // Computes: C = weight^T · input
        //   weight: [out_dim, in_dim] row-major
        //   input:  [in_dim, batch]   col-major (= [batch, in_dim] row-major)
        //   C:      [out_dim, batch]  col-major (= [batch, out_dim] row-major)

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

        // ── Batch prefill ─────────────────────────────────────────────────

        /// Run a batched forward pass for `N = token_ids.len()` tokens.
        ///
        /// Block tables must already be fully allocated for positions
        /// `0 .. token_ids.len()`. Returns f32 logits for the **last** token.
        fn forward_prefill(
            &mut self,
            token_ids: &[u32],
            start_position: u32,
            block_tables: &[Vec<i32>],
        ) -> Result<Vec<f32>, EngineError> {
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

            // Build embedding matrix [N, h].
            let mut hidden = self.device.alloc_zeros::<f16>(n * h)
                .map_err(|err| e(format!("alloc hidden: {err}")))?;
            for (i, &tok) in token_ids.iter().enumerate() {
                let off = tok as usize * h;
                self.device.dtod_copy(
                    &self.weights.embed_tokens.slice(off..off + h),
                    &mut hidden.slice_mut(i * h..(i + 1) * h),
                ).map_err(|err| e(format!("embed: {err}")))?;
            }

            let mut norm = self.device.alloc_zeros::<f16>(n * h)
                .map_err(|err| e(format!("alloc norm: {err}")))?;
            let mut residual = self.device.alloc_zeros::<f16>(n * h)
                .map_err(|err| e(format!("alloc residual: {err}")))?;
            let mut q_all = self.device.alloc_zeros::<f16>(n * num_q * head_dim)
                .map_err(|err| e(format!("alloc q: {err}")))?;
            let mut k_all = self.device.alloc_zeros::<f16>(n * num_kv * head_dim)
                .map_err(|err| e(format!("alloc k: {err}")))?;
            let mut v_all = self.device.alloc_zeros::<f16>(n * num_kv * head_dim)
                .map_err(|err| e(format!("alloc v: {err}")))?;
            let mut attn_out = self.device.alloc_zeros::<f16>(n * num_q * head_dim)
                .map_err(|err| e(format!("alloc attn_out: {err}")))?;
            let mut gate_out = self.device.alloc_zeros::<f16>(n * inter)
                .map_err(|err| e(format!("alloc gate: {err}")))?;
            let mut up_out = self.device.alloc_zeros::<f16>(n * inter)
                .map_err(|err| e(format!("alloc up: {err}")))?;
            let mut swiglu_out = self.device.alloc_zeros::<f16>(n * inter)
                .map_err(|err| e(format!("alloc swiglu: {err}")))?;
            let mut ffn_input = self.device.alloc_zeros::<f16>(n * h)
                .map_err(|err| e(format!("alloc ffn_in: {err}")))?;
            let mut ffn_out = self.device.alloc_zeros::<f16>(n * h)
                .map_err(|err| e(format!("alloc ffn_out: {err}")))?;
            let mut o_out = self.device.alloc_zeros::<f16>(n * h)
                .map_err(|err| e(format!("alloc o_out: {err}")))?;

            // Build block-pool position arrays for batch KV write.
            let mut phys_blocks_host: Vec<i32> = Vec::with_capacity(n);
            let mut pos_in_blk_host:  Vec<i32> = Vec::with_capacity(n);
            for pos in 0..n {
                let logical = pos / block_size;
                phys_blocks_host.push(block_tables[0][logical]);
                pos_in_blk_host.push((pos % block_size) as i32);
            }
            // (per-layer these differ only when layers have different block tables,
            //  which is currently always the case — we upload per layer in the loop.)

            for layer_idx in 0..self.weights.layers.len() {
                let layer = &self.weights.layers[layer_idx];

                // RMSNorm [N, h]
                launch_rms_norm(&self.device, &mut RmsNormParams {
                    out: &mut norm, input: &hidden,
                    weight: &layer.input_layernorm,
                    rows: n as u32, dim: h as u32, eps,
                }).map_err(e)?;

                // QKV — batched GEMMs (n column vectors at once)
                let blas = Arc::clone(&self.blas);
                Self::gemm(&blas, (num_q * head_dim) as i32, n as i32, h as i32,
                    &layer.q_proj, h as i32, &norm, h as i32,
                    &mut q_all, (num_q * head_dim) as i32, "Q")?;
                Self::gemm(&blas, (num_kv * head_dim) as i32, n as i32, h as i32,
                    &layer.k_proj, h as i32, &norm, h as i32,
                    &mut k_all, (num_kv * head_dim) as i32, "K")?;
                Self::gemm(&blas, (num_kv * head_dim) as i32, n as i32, h as i32,
                    &layer.v_proj, h as i32, &norm, h as i32,
                    &mut v_all, (num_kv * head_dim) as i32, "V")?;

                // Batch RoPE
                launch_batch_rope(&self.device, &mut BatchRopeParams {
                    q: &mut q_all, k: &mut k_all,
                    seq_len: n as u32,
                    num_q_heads: num_q as u32, num_kv_heads: num_kv as u32,
                    head_dim: head_dim as u32,
                    position_offset: start_position,
                    theta: rope_theta,
                }).map_err(e)?;

                // Batch KV write — per layer (each layer has its own physical blocks)
                let phys_dev = self.device.htod_sync_copy(&{
                    let logical_to_phys: Vec<i32> = (0..n)
                        .map(|pos| block_tables[layer_idx][pos / block_size])
                        .collect();
                    logical_to_phys
                }).map_err(|err| EngineError::backend(format!("phys_dev: {err}")))?;
                let pos_dev = self.device.htod_sync_copy(&pos_in_blk_host)
                    .map_err(|err| EngineError::backend(format!("pos_dev: {err}")))?;

                launch_batch_kv_write(&self.device, &mut BatchKvWriteParams {
                    kv_cache: self.block_pool.storage_mut(),
                    k: &k_all, v: &v_all,
                    physical_blocks: &phys_dev,
                    pos_in_blocks: &pos_dev,
                    seq_len: n as u32,
                    num_kv_heads: num_kv as u32,
                    block_size: block_size as u32,
                    head_dim: head_dim as u32,
                }).map_err(e)?;

                // Prefill causal attention
                launch_prefill_attention(&self.device, &mut PrefillAttnParams {
                    out: &mut attn_out, q: &q_all, k: &k_all, v: &v_all,
                    scale, seq_len: n as u32,
                    num_q_heads: num_q as u32, num_kv_heads: num_kv as u32,
                    head_dim: head_dim as u32,
                }).map_err(e)?;

                // Output projection [N, q_dim] → [N, h]
                let layer = &self.weights.layers[layer_idx];
                Self::gemm(&blas, h as i32, n as i32, (num_q * head_dim) as i32,
                    &layer.o_proj, (num_q * head_dim) as i32,
                    &attn_out, (num_q * head_dim) as i32,
                    &mut o_out, h as i32, "O")?;

                // Residual: residual = hidden + o_out
                launch_residual_add(&self.device, &mut residual, &hidden, &o_out, (n * h) as u32)
                    .map_err(e)?;

                // Post-attn RMSNorm
                let layer = &self.weights.layers[layer_idx];
                launch_rms_norm(&self.device, &mut RmsNormParams {
                    out: &mut ffn_input, input: &residual,
                    weight: &layer.post_attention_layernorm,
                    rows: n as u32, dim: h as u32, eps,
                }).map_err(e)?;

                // FFN gate/up
                let layer = &self.weights.layers[layer_idx];
                Self::gemm(&blas, inter as i32, n as i32, h as i32,
                    &layer.gate_proj, h as i32, &ffn_input, h as i32,
                    &mut gate_out, inter as i32, "gate")?;
                Self::gemm(&blas, inter as i32, n as i32, h as i32,
                    &layer.up_proj, h as i32, &ffn_input, h as i32,
                    &mut up_out, inter as i32, "up")?;

                // SwiGLU
                let layer = &self.weights.layers[layer_idx];
                {
                    use kapsl_kernels::cuda_kernels::launch_fused_swiglu;
                    launch_fused_swiglu(&self.device, &mut swiglu_out, &gate_out, &up_out, (n * inter) as u32)
                        .map_err(e)?;
                }

                // Down projection
                let layer = &self.weights.layers[layer_idx];
                Self::gemm(&blas, h as i32, n as i32, inter as i32,
                    &layer.down_proj, inter as i32, &swiglu_out, inter as i32,
                    &mut ffn_out, h as i32, "down")?;

                // FFN residual: hidden = residual + ffn_out
                launch_residual_add(&self.device, &mut hidden, &residual, &ffn_out, (n * h) as u32)
                    .map_err(e)?;
            }

            // Final RMSNorm on last token only: slice hidden[(N-1)*h .. N*h]
            let last_off = (n - 1) * h;
            let last_hidden = hidden.slice(last_off..last_off + h);
            let mut last_norm = self.device.alloc_zeros::<f16>(h)
                .map_err(|err| e(format!("alloc last_norm: {err}")))?;
            launch_rms_norm(&self.device, &mut RmsNormParams {
                out: &mut last_norm, input: &last_hidden,
                weight: &self.weights.norm,
                rows: 1, dim: h as u32, eps,
            }).map_err(e)?;

            // LM head
            let blas = Arc::clone(&self.blas);
            let mut logits_dev = self.device.alloc_zeros::<f16>(vocab)
                .map_err(|err| e(format!("alloc logits: {err}")))?;
            Self::gemm(&blas, vocab as i32, 1, h as i32,
                &self.weights.lm_head, h as i32, &last_norm, h as i32,
                &mut logits_dev, vocab as i32, "lm_head")?;

            let f16v: Vec<f16> = self.device.dtoh_sync_copy(&logits_dev)
                .map_err(|err| e(format!("logits dl: {err}")))?;
            Ok(f16v.iter().map(|v| v.to_f32()).collect())
        }

        // ── Single-token decode ───────────────────────────────────────────

        fn forward_one_token(
            &mut self,
            token_id: u32,
            block_tables: &[Vec<i32>],
            context_len: usize,
            position: u32,
        ) -> Result<Vec<f32>, EngineError> {
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

            let ctx_dev = self.device.htod_sync_copy(&[context_len as i32])
                .map_err(|err| e(format!("ctx_dev: {err}")))?;

            let pos_in_seq = context_len - 1;
            let pos_in_block = pos_in_seq % block_size;

            let mut attn_out = self.device.alloc_zeros::<f16>(num_q * head_dim)
                .map_err(|err| e(format!("{err}")))?;
            let mut gate_out = self.device.alloc_zeros::<f16>(inter)
                .map_err(|err| e(format!("{err}")))?;
            let mut up_out = self.device.alloc_zeros::<f16>(inter)
                .map_err(|err| e(format!("{err}")))?;
            let mut swiglu_out = self.device.alloc_zeros::<f16>(inter)
                .map_err(|err| e(format!("{err}")))?;
            let mut ffn_input = self.device.alloc_zeros::<f16>(h)
                .map_err(|err| e(format!("{err}")))?;
            let mut ffn_out = self.device.alloc_zeros::<f16>(h)
                .map_err(|err| e(format!("{err}")))?;
            let mut o_proj_out = self.device.alloc_zeros::<f16>(h)
                .map_err(|err| e(format!("{err}")))?;

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

                // RoPE (single token)
                use kapsl_kernels::cuda_kernels::{launch_rope, RopeParams};
                launch_rope(&self.device, &mut RopeParams {
                    q: &mut self.q_buf, k: &mut self.k_buf,
                    num_q_heads: num_q as u32, num_kv_heads: num_kv as u32,
                    head_dim: head_dim as u32, position, theta: rope_theta,
                }).map_err(e)?;

                // Write K/V to pool
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

                // Paged attention
                let bt_dev = self.device.htod_sync_copy(&block_tables[layer_idx])
                    .map_err(|err| e(format!("bt: {err}")))?;
                launch_paged_attention(&self.device, &mut PagedAttentionParams {
                    out: &mut attn_out, q: &self.q_buf,
                    kv_cache: self.block_pool.storage(),
                    block_tables: &bt_dev, context_lens: &ctx_dev,
                    scale, batch_size: 1,
                    num_q_heads: num_q as u32, num_kv_heads: num_kv as u32,
                    head_dim: head_dim as u32, block_size: block_size as u32,
                    max_blocks_per_seq: block_tables[layer_idx].len() as u32,
                }).map_err(e)?;

                // Output projection
                let layer = &self.weights.layers[layer_idx];
                Self::gemm(&blas, h as i32, 1, q_dim as i32,
                    &layer.o_proj, q_dim as i32, &attn_out, q_dim as i32,
                    &mut o_proj_out, h as i32, "O")?;

                launch_residual_add(&self.device, &mut self.residual_buf,
                    &self.hidden_buf, &o_proj_out, h as u32).map_err(e)?;

                let layer = &self.weights.layers[layer_idx];
                launch_rms_norm(&self.device, &mut RmsNormParams {
                    out: &mut ffn_input, input: &self.residual_buf,
                    weight: &layer.post_attention_layernorm,
                    rows: 1, dim: h as u32, eps,
                }).map_err(e)?;

                let layer = &self.weights.layers[layer_idx];
                Self::gemm(&blas, inter as i32, 1, h as i32,
                    &layer.gate_proj, h as i32, &ffn_input, h as i32,
                    &mut gate_out, inter as i32, "gate")?;
                Self::gemm(&blas, inter as i32, 1, h as i32,
                    &layer.up_proj, h as i32, &ffn_input, h as i32,
                    &mut up_out, inter as i32, "up")?;

                launch_fused_swiglu(&self.device, &mut swiglu_out, &gate_out, &up_out, inter as u32)
                    .map_err(e)?;

                let layer = &self.weights.layers[layer_idx];
                Self::gemm(&blas, h as i32, 1, inter as i32,
                    &layer.down_proj, inter as i32, &swiglu_out, inter as i32,
                    &mut ffn_out, h as i32, "down")?;

                launch_residual_add(&self.device, &mut self.hidden_buf,
                    &self.residual_buf, &ffn_out, h as u32).map_err(e)?;
            }

            launch_rms_norm(&self.device, &mut RmsNormParams {
                out: &mut self.norm_buf, input: &self.hidden_buf,
                weight: &self.weights.norm, rows: 1, dim: h as u32, eps,
            }).map_err(e)?;

            let mut logits_dev = self.device.alloc_zeros::<f16>(vocab)
                .map_err(|err| e(format!("alloc logits: {err}")))?;
            Self::gemm(&blas, vocab as i32, 1, h as i32,
                &self.weights.lm_head, h as i32, &self.norm_buf, h as i32,
                &mut logits_dev, vocab as i32, "lm_head")?;

            let f16v: Vec<f16> = self.device.dtoh_sync_copy(&logits_dev)
                .map_err(|err| e(format!("logits dl: {err}")))?;
            Ok(f16v.iter().map(|v| v.to_f32()).collect())
        }

        // ── Decode loop ───────────────────────────────────────────────────

        /// Runs prefill + greedy/sampled decode.
        ///
        /// If `tx` is `Some`, each generated token is sent immediately
        /// (streaming). Either way the full list is returned.
        fn run_decode(
            &mut self,
            prompt_ids: &[u32],
            session: Option<&mut SessionState>,
            max_new_tokens: u32,
            eos: Option<u32>,
            sp: &SampleParams,
            cancel: Option<&kapsl_engine_api::CancellationToken>,
            tx: Option<&tokio::sync::mpsc::Sender<Result<BinaryTensorPacket, EngineError>>>,
        ) -> Result<Vec<u32>, EngineError> {
            if prompt_ids.is_empty() { return Ok(Vec::new()); }

            // Resolve or create session block state.
            let owned_state;
            let (block_tables, context_len_ref) = if let Some(s) = session {
                (&mut s.block_tables, &mut s.context_len)
            } else {
                owned_state = Some(SessionState { block_tables: Vec::new(), context_len: 0 });
                let s = owned_state.as_mut().unwrap();
                (&mut s.block_tables, &mut s.context_len)
            };

            // Ensure blocks for all prompt positions.
            for i in 0..prompt_ids.len() {
                self.ensure_block(block_tables, *context_len_ref + i)?;
            }

            let start_position = *context_len_ref as u32;

            // Batch prefill
            let mut last_logits = self.forward_prefill(
                prompt_ids,
                start_position,
                block_tables,
            )?;
            *context_len_ref += prompt_ids.len();

            let mut generated = Vec::new();
            let mut next = self.sample(&last_logits, sp);

            for _ in 0..max_new_tokens {
                if cancel.map_or(false, |c| c.is_cancelled()) { break; }
                if eos.map_or(false, |e| next == e) { break; }

                generated.push(next);

                if let Some(tx) = tx {
                    let pkt = token_to_packet(next);
                    if tx.blocking_send(Ok(pkt)).is_err() { break; }
                }

                self.ensure_block(block_tables, *context_len_ref)?;
                *context_len_ref += 1;
                let position = (*context_len_ref - 1) as u32;

                last_logits = self.forward_one_token(next, block_tables, *context_len_ref, position)?;
                next = self.sample(&last_logits, sp);
            }

            // For stateless calls, free the blocks we used.
            if tx.is_none() {
                // owned_state is Some only in stateless path; block_tables ref
                // points into it, which we can't free here — caller frees.
            }

            Ok(generated)
        }
    }

    fn token_to_packet(token_id: u32) -> BinaryTensorPacket {
        let data = (token_id as i32).to_le_bytes().to_vec();
        BinaryTensorPacket::new(vec![1], TensorDtype::Int32, data)
            .expect("valid packet")
    }

    // ── NativeBackend ─────────────────────────────────────────────────────

    pub struct NativeBackend {
        device_id: i32,
        inner: Arc<Mutex<Option<BackendInner>>>,
    }

    impl NativeBackend {
        pub fn new(device_id: i32) -> Result<Self, EngineError> {
            CudaDevice::new(device_id as usize)
                .map_err(|e| EngineError::backend(format!("CUDA device {device_id}: {e}")))?;
            Ok(Self { device_id, inner: Arc::new(Mutex::new(None)) })
        }

        fn extract_token_ids(req: &InferenceRequest) -> Result<Vec<u32>, EngineError> {
            let p = &req.input;
            if p.dtype != TensorDtype::Int32 {
                return Err(EngineError::InvalidInput {
                    message: format!("NativeBackend needs Int32, got {:?}", p.dtype),
                    source: None,
                });
            }
            Ok(p.data.chunks_exact(4).map(|b| {
                i32::from_le_bytes(b.try_into().unwrap()) as u32
            }).collect())
        }

        fn decode_params(req: &InferenceRequest) -> (u32, Option<u32>, SampleParams) {
            let meta = req.metadata.as_ref();
            let max_new = meta.and_then(|m| m.max_new_tokens).unwrap_or(128);
            let eos = meta
                .and_then(|m| m.stop_token_ids.as_ref())
                .and_then(|v| v.first().copied());
            (max_new, eos, SampleParams::from_meta(meta))
        }
    }

    #[async_trait]
    impl kapsl_engine_api::Engine for NativeBackend {
        async fn load(&mut self, model_path: &Path) -> Result<(), EngineError> {
            let dir = if model_path.is_dir() {
                model_path.to_path_buf()
            } else {
                model_path.parent().unwrap_or(model_path).to_path_buf()
            };

            let cpu = load_safetensors(&dir)
                .map_err(|e| EngineError::backend(format!("safetensors: {e}")))?;
            let config = cpu.config.clone();
            log::info!(
                "NativeBackend: {} layers, {}Q/{}KV heads, h={}, vocab={}",
                config.num_hidden_layers, config.num_attention_heads,
                config.num_kv_heads(), config.hidden_size, config.vocab_size,
            );

            let device = CudaDevice::new(self.device_id as usize)
                .map_err(|e| EngineError::backend(format!("CUDA: {e}")))?;
            let blas = Arc::new(CudaBlas::new(device.clone())
                .map_err(|e| EngineError::backend(format!("cuBLAS: {e}")))?);

            let weights = upload_weights(&device, &cpu)?;
            drop(cpu);

            let block_size = 16usize;
            let bps = (config.max_position_embeddings + block_size - 1) / block_size;
            let num_blocks = config.num_hidden_layers * 8 * bps; // 8 concurrent sessions
            let block_pool = GpuBlockPool::new(
                device.clone(), num_blocks, block_size,
                config.num_kv_heads(), config.head_dim(),
            ).map_err(|e| EngineError::backend(format!("block pool: {e}")))?;

            let h = config.hidden_size;
            let nq = config.num_attention_heads;
            let nkv = config.num_kv_heads();
            let hd = config.head_dim();
            let alloc = |n: usize| device.alloc_zeros::<f16>(n)
                .map_err(|e| EngineError::backend(format!("alloc: {e}")));

            let backend = BackendInner {
                device, blas, weights, block_pool, config,
                hidden_buf:   alloc(h)?,
                norm_buf:     alloc(h)?,
                residual_buf: alloc(h)?,
                q_buf:        alloc(nq * hd)?,
                k_buf:        alloc(nkv * hd)?,
                v_buf:        alloc(nkv * hd)?,
                sessions:     HashMap::new(),
                rng:          rand::rngs::SmallRng::from_entropy(),
                staged:       None,
            };
            *self.inner.lock().unwrap() = Some(backend);
            log::info!("NativeBackend: ready");
            Ok(())
        }

        fn infer(&self, req: &InferenceRequest) -> Result<BinaryTensorPacket, EngineError> {
            let mut guard = self.inner.lock().unwrap();
            let inner = guard.as_mut().ok_or(EngineError::ModelNotLoaded)?;

            let prompt = Self::extract_token_ids(req)?;
            let (max_new, eos, sp) = Self::decode_params(req);
            let sid = req.session_id.as_deref();

            // Fetch or create session outside the mutable borrow.
            let (generated, stateless_blocks) = if let Some(sid) = sid {
                let session = inner.sessions.entry(sid.to_string())
                    .or_insert_with(|| SessionState { block_tables: Vec::new(), context_len: 0 });
                let gen = inner.run_decode(&prompt, Some(session), max_new, eos, &sp,
                    req.cancellation.as_ref(), None)?;
                (gen, None)
            } else {
                let mut tmp = SessionState { block_tables: Vec::new(), context_len: 0 };
                let gen = inner.run_decode(&prompt, Some(&mut tmp), max_new, eos, &sp,
                    req.cancellation.as_ref(), None)?;
                (gen, Some(tmp.block_tables))
            };

            // Free blocks for stateless calls.
            if let Some(bt) = stateless_blocks {
                inner.free_block_tables(&bt);
            }

            pack_tokens(&generated)
        }

        fn infer_stream(&self, req: &InferenceRequest) -> EngineStream {
            let inner = Arc::clone(&self.inner);

            let prompt = match Self::extract_token_ids(req) {
                Ok(v) => v,
                Err(e) => return Box::pin(futures::stream::once(async { Err(e) })),
            };
            let (max_new, eos, sp) = Self::decode_params(req);
            let sid = req.session_id.clone();
            let cancel = req.cancellation.clone();

            let (tx, rx) = tokio::sync::mpsc::channel::<Result<BinaryTensorPacket, EngineError>>(64);

            tokio::task::spawn_blocking(move || {
                let mut guard = match inner.lock() {
                    Ok(g) => g,
                    Err(_) => { let _ = tx.blocking_send(Err(EngineError::backend("mutex poisoned"))); return; }
                };
                let b = match guard.as_mut() {
                    Some(b) => b,
                    None => { let _ = tx.blocking_send(Err(EngineError::ModelNotLoaded)); return; }
                };

                let result = if let Some(ref sid) = sid {
                    let session = b.sessions.entry(sid.clone())
                        .or_insert_with(|| SessionState { block_tables: Vec::new(), context_len: 0 });
                    b.run_decode(&prompt, Some(session), max_new, eos, &sp, cancel.as_ref(), Some(&tx))
                } else {
                    let mut tmp = SessionState { block_tables: Vec::new(), context_len: 0 };
                    let r = b.run_decode(&prompt, Some(&mut tmp), max_new, eos, &sp, cancel.as_ref(), Some(&tx));
                    b.free_block_tables(&tmp.block_tables);
                    r
                };

                if let Err(e) = result {
                    let _ = tx.blocking_send(Err(e));
                }
            });

            Box::pin(futures::stream::unfold(rx, |mut rx| async move {
                rx.recv().await.map(|item| (item, rx))
            }))
        }

        fn unload(&mut self) {
            *self.inner.lock().unwrap() = None;
            log::info!("NativeBackend: unloaded");
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

        fn model_info(&self) -> Option<EngineModelInfo> {
            let g = self.inner.lock().unwrap();
            let cfg = g.as_ref().map(|b| b.config.clone())?;
            let version = cfg.architectures.first().cloned()
                .unwrap_or_else(|| "native-cuda".into());
            Some(EngineModelInfo {
                input_names: vec!["input_ids".into()],
                output_names: vec!["output_ids".into()],
                input_shapes: vec![vec![-1]],
                output_shapes: vec![vec![-1]],
                input_dtypes: vec!["int32".into()],
                output_dtypes: vec!["int32".into()],
                framework: Some("native-cuda".into()),
                model_version: Some(version),
                peak_concurrency: Some(1),
            })
        }

        fn health_check(&self) -> Result<(), EngineError> {
            if self.inner.lock().unwrap().is_some() { Ok(()) }
            else { Err(EngineError::ModelNotLoaded) }
        }

        fn supports_swap(&self) -> bool { true }

        fn is_staged(&self) -> bool {
            self.inner.lock().unwrap()
                .as_ref()
                .map(|b| b.staged.is_some())
                .unwrap_or(false)
        }

        async fn stage(&self, path: &Path) -> Result<(), EngineError> {
            let path = path.to_owned();
            let inner = Arc::clone(&self.inner);

            tokio::task::spawn_blocking(move || {
                let dir = if path.is_dir() {
                    path.clone()
                } else {
                    path.parent().unwrap_or(&path).to_path_buf()
                };
                log::info!("NativeBackend: staging model from {:?} into CPU RAM…", dir);
                let weights = load_safetensors(&dir)
                    .map_err(|e| EngineError::backend(format!("stage load: {e}")))?;
                log::info!(
                    "NativeBackend: staged {} layers into CPU RAM",
                    weights.layers.len()
                );
                let mut guard = inner.lock().unwrap();
                let b = guard.as_mut().ok_or(EngineError::ModelNotLoaded)?;
                b.staged = Some(weights);
                Ok(())
            })
            .await
            .map_err(|e| EngineError::backend(format!("stage task: {e}")))?
        }

        async fn swap(&self) -> Result<(), EngineError> {
            let inner = Arc::clone(&self.inner);
            tokio::task::spawn_blocking(move || {
                let mut guard = inner.lock().unwrap();
                let b = guard.as_mut().ok_or(EngineError::ModelNotLoaded)?;
                b.activate_staged()
            })
            .await
            .map_err(|e| EngineError::backend(format!("swap task: {e}")))?
        }
    }

    fn pack_tokens(ids: &[u32]) -> Result<BinaryTensorPacket, EngineError> {
        let mut data = Vec::with_capacity(ids.len() * 4);
        for &id in ids { data.extend_from_slice(&(id as i32).to_le_bytes()); }
        BinaryTensorPacket::new(vec![ids.len() as i64], TensorDtype::Int32, data)
            .map_err(|e| EngineError::backend(format!("pack: {e}")))
    }
}

#[cfg(feature = "native")]
pub use inner::NativeBackend;
