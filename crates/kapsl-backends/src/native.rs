//! Native CUDA inference backend — continuous-batching scheduler.
//!
//! # Architecture
//!
//! A dedicated OS thread ("scheduler") owns `BackendInner` exclusively and
//! runs a tight loop:
//!
//! 1. Drain pending `SchedulerCmd::Request` messages from the inbox channel.
//!    For each new request run prefill and add the sequence to the active set.
//! 2. If the active set is non-empty, call `batch_decode_compute` once to
//!    advance ALL active sequences by one token in a single batched forward pass.
//! 3. Emit generated tokens, check EOS / max-tokens, remove finished sequences.
//!
//! `NativeBackend::infer()` and `infer_stream()` post requests to the scheduler
//! and wait on a `tokio::sync::oneshot` or `mpsc` channel for results.
//! The mutex is held for < 1 µs (to clone the sender), never during GPU compute.

#[cfg(feature = "native")]
mod inner {
    use std::collections::{HashMap, HashSet};
    use std::path::Path;
    use std::sync::{Arc, Mutex};
    use std::sync::atomic::{AtomicBool, Ordering};

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
        launch_argmax, launch_batch_kv_write, launch_batch_rope, launch_fused_swiglu,
        launch_paged_attention, launch_prefill_attention, launch_residual_add, launch_rms_norm,
        ArgmaxParams, BatchKvWriteParams, BatchRopeParams, PagedAttentionParams, PrefillAttnParams,
        RmsNormParams,
        launch_batch_decode_rope, launch_batch_argmax,
        BatchDecodeRopeParams, BatchArgmaxParams,
    };
    use kapsl_loader::{load_safetensors, ModelConfig, ModelWeights, TensorData};

    /// Maximum sequences decoded simultaneously in one batched forward pass.
    const MAX_BATCH: usize = 32;

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

    fn upload_f16(device: &Arc<CudaDevice>, v: &[f16]) -> Result<CudaSlice<f16>, EngineError> {
        device.htod_sync_copy(v)
            .map_err(|e| EngineError::backend(format!("GPU upload: {e}")))
    }

    fn upload_staged(device: &Arc<CudaDevice>, s: &StagedF16Weights) -> Result<GpuModelWeights, EngineError> {
        let embed_tokens = upload_f16(device, &s.embed_tokens)?;
        let norm = upload_f16(device, &s.norm)?;
        let lm_head = upload_f16(device, &s.lm_head)?;
        let mut layers = Vec::with_capacity(s.layers.len());
        for (i, l) in s.layers.iter().enumerate() {
            log::info!("Uploading staged layer {}/{}", i + 1, s.layers.len());
            layers.push(GpuLayerWeights {
                input_layernorm:          upload_f16(device, &l.input_layernorm)?,
                q_proj:                   upload_f16(device, &l.q_proj)?,
                k_proj:                   upload_f16(device, &l.k_proj)?,
                v_proj:                   upload_f16(device, &l.v_proj)?,
                o_proj:                   upload_f16(device, &l.o_proj)?,
                post_attention_layernorm: upload_f16(device, &l.post_attention_layernorm)?,
                gate_proj:                upload_f16(device, &l.gate_proj)?,
                up_proj:                  upload_f16(device, &l.up_proj)?,
                down_proj:                upload_f16(device, &l.down_proj)?,
            });
        }
        Ok(GpuModelWeights { embed_tokens, layers, norm, lm_head })
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
        fn is_greedy(&self) -> bool { self.temperature < 1e-6 }
    }

    // ── Session state ─────────────────────────────────────────────────────

    struct SessionState {
        block_tables: Vec<Vec<i32>>,
        context_len: usize,
    }

    // ── Prefill scratch buffers ───────────────────────────────────────────

    struct PrefillScratch {
        cap:       usize,
        hidden:    CudaSlice<f16>,
        norm:      CudaSlice<f16>,
        residual:  CudaSlice<f16>,
        q_all:     CudaSlice<f16>,
        k_all:     CudaSlice<f16>,
        v_all:     CudaSlice<f16>,
        attn_out:  CudaSlice<f16>,
        gate_out:  CudaSlice<f16>,
        up_out:    CudaSlice<f16>,
        swiglu_out:CudaSlice<f16>,
        ffn_input: CudaSlice<f16>,
        ffn_out:   CudaSlice<f16>,
        o_out:     CudaSlice<f16>,
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

    // ── Batch decode scratch ──────────────────────────────────────────────

    struct BatchDecodeScratch {
        /// Capacity in sequences.
        cap:       usize,
        hidden:    CudaSlice<f16>,   // [cap * h]
        norm:      CudaSlice<f16>,   // [cap * h]
        residual:  CudaSlice<f16>,   // [cap * h]
        q_buf:     CudaSlice<f16>,   // [cap * q_dim]
        k_buf:     CudaSlice<f16>,   // [cap * kv_dim]
        v_buf:     CudaSlice<f16>,   // [cap * kv_dim]
        attn_buf:  CudaSlice<f16>,   // [cap * q_dim]
        gate_buf:  CudaSlice<f16>,   // [cap * inter]
        up_buf:    CudaSlice<f16>,   // [cap * inter]
        swiglu_buf:CudaSlice<f16>,   // [cap * inter]
        ffn_input: CudaSlice<f16>,   // [cap * h]
        ffn_out:   CudaSlice<f16>,   // [cap * h]
        o_proj:    CudaSlice<f16>,   // [cap * h]
        logits:    CudaSlice<f16>,   // [cap * vocab]
        argmax:    CudaSlice<u32>,   // [cap]
        positions: CudaSlice<i32>,   // [cap]
        ctx_lens:  CudaSlice<i32>,   // [cap]
    }

    impl BatchDecodeScratch {
        fn new(
            device: &Arc<CudaDevice>,
            cap: usize, h: usize, q_dim: usize, kv_dim: usize, inter: usize, vocab: usize,
        ) -> Result<Self, EngineError> {
            let a = |n: usize| device.alloc_zeros::<f16>(n)
                .map_err(|e| EngineError::backend(format!("batch scratch: {e}")));
            Ok(Self {
                cap,
                hidden:     a(cap * h)?,
                norm:       a(cap * h)?,
                residual:   a(cap * h)?,
                q_buf:      a(cap * q_dim)?,
                k_buf:      a(cap * kv_dim)?,
                v_buf:      a(cap * kv_dim)?,
                attn_buf:   a(cap * q_dim)?,
                gate_buf:   a(cap * inter)?,
                up_buf:     a(cap * inter)?,
                swiglu_buf: a(cap * inter)?,
                ffn_input:  a(cap * h)?,
                ffn_out:    a(cap * h)?,
                o_proj:     a(cap * h)?,
                logits:     a(cap * vocab)?,
                argmax:     device.alloc_zeros::<u32>(cap)
                    .map_err(|e| EngineError::backend(format!("batch argmax: {e}")))?,
                positions:  device.alloc_zeros::<i32>(cap)
                    .map_err(|e| EngineError::backend(format!("batch positions: {e}")))?,
                ctx_lens:   device.alloc_zeros::<i32>(cap)
                    .map_err(|e| EngineError::backend(format!("batch ctx_lens: {e}")))?,
            })
        }
    }

    // ── Pre-converted f16 weights (staging) ───────────────────────────────

    struct StagedF16Weights {
        config: ModelConfig,
        embed_tokens: Vec<f16>,
        norm: Vec<f16>,
        lm_head: Vec<f16>,
        layers: Vec<StagedF16Layer>,
    }

    struct StagedF16Layer {
        input_layernorm: Vec<f16>,
        q_proj: Vec<f16>,
        k_proj: Vec<f16>,
        v_proj: Vec<f16>,
        o_proj: Vec<f16>,
        post_attention_layernorm: Vec<f16>,
        gate_proj: Vec<f16>,
        up_proj: Vec<f16>,
        down_proj: Vec<f16>,
    }

    // ── BackendInner ──────────────────────────────────────────────────────

    struct BackendInner {
        device: Arc<CudaDevice>,
        blas: Arc<CudaBlas>,
        config: ModelConfig,
        weights: GpuModelWeights,
        block_pool: GpuBlockPool,
        // Prefill path reuses these small single-row buffers.
        norm_buf: CudaSlice<f16>,     // [h]
        logits_buf: CudaSlice<f16>,   // [vocab]
        argmax_buf: CudaSlice<u32>,   // [1]
        // Batch decode scratch (MAX_BATCH capacity).
        batch: BatchDecodeScratch,
        // Multi-turn session KV state (owned by scheduler thread).
        sessions: HashMap<String, SessionState>,
        rng: rand::rngs::SmallRng,
        prefill: PrefillScratch,
        staged: Option<StagedF16Weights>,
    }

    impl BackendInner {
        // ── Block management ─────────────────────────────────────────────

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

        // ── Hot-swap ──────────────────────────────────────────────────────

        fn to_staged(w: ModelWeights) -> StagedF16Weights {
            StagedF16Weights {
                config: w.config,
                embed_tokens: w.embed_tokens.to_f16_vec(),
                norm: w.norm.to_f16_vec(),
                lm_head: w.lm_head.to_f16_vec(),
                layers: w.layers.into_iter().map(|l| StagedF16Layer {
                    input_layernorm:         l.input_layernorm.to_f16_vec(),
                    q_proj:                  l.q_proj.to_f16_vec(),
                    k_proj:                  l.k_proj.to_f16_vec(),
                    v_proj:                  l.v_proj.to_f16_vec(),
                    o_proj:                  l.o_proj.to_f16_vec(),
                    post_attention_layernorm: l.post_attention_layernorm.to_f16_vec(),
                    gate_proj:               l.gate_proj.to_f16_vec(),
                    up_proj:                 l.up_proj.to_f16_vec(),
                    down_proj:               l.down_proj.to_f16_vec(),
                }).collect(),
            }
        }

        fn activate_staged(&mut self) -> Result<(), EngineError> {
            let staged = self.staged.take()
                .ok_or_else(|| EngineError::backend("no model staged; call stage() first"))?;

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

            log::info!("NativeBackend: activating staged weights (PCIe transfer only)…");
            let new_weights = upload_staged(&self.device, &staged)?;
            self.weights = new_weights;
            self.config = staged.config;

            let all_bts: Vec<_> = self.sessions.values()
                .map(|s| s.block_tables.clone())
                .collect();
            for bt in all_bts { self.free_block_tables(&bt); }
            self.sessions.clear();
            log::info!("NativeBackend: swap complete");
            Ok(())
        }

        // ── Prefill scratch management ────────────────────────────────────

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

        // ── Sampling ─────────────────────────────────────────────────────

        fn greedy(logits: &[f32]) -> u32 {
            logits.iter().enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i as u32)
                .unwrap_or(0)
        }

        fn sample(&mut self, logits: &[f32], p: &SampleParams) -> u32 {
            if p.is_greedy() { return Self::greedy(logits); }

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

        // ── cuBLAS GEMM helper ────────────────────────────────────────────
        // C = weight^T · input  (cuBLAS column-major convention)
        //   weight [out_dim, in_dim] row-major  → treated as [in_dim, out_dim] column-major
        //   input  [in_dim, batch]  column-major (= [batch, in_dim] row-major)
        //   C      [out_dim, batch] column-major (= [batch, out_dim] row-major)

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

            let pos_in_blk_host: Vec<i32> = (0..n).map(|p| (p % block_size) as i32).collect();
            // Upload once — pos_in_block is the same for every layer.
            let pos_dev = self.device.htod_sync_copy(&pos_in_blk_host)
                .map_err(|err| e(format!("pos_dev: {err}")))?;

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

        fn forward_prefill_greedy(
            &mut self,
            token_ids: &[u32],
            start_position: u32,
            block_tables: &[Vec<i32>],
        ) -> Result<u32, EngineError> {
            let vocab = self.config.vocab_size;
            let e = |s: String| EngineError::backend(s);
            self.prefill_compute(token_ids, start_position, block_tables)?;
            launch_argmax(&self.device, &mut ArgmaxParams {
                input: &self.logits_buf,
                output: &mut self.argmax_buf,
                vocab_size: vocab as u32,
            }).map_err(|s| EngineError::backend(s))?;
            let ids: Vec<u32> = self.device.dtoh_sync_copy(&self.argmax_buf)
                .map_err(|err| e(format!("argmax dl: {err}")))?;
            Ok(ids[0])
        }

        fn forward_prefill(
            &mut self,
            token_ids: &[u32],
            start_position: u32,
            block_tables: &[Vec<i32>],
        ) -> Result<Vec<f32>, EngineError> {
            let e = |s: String| EngineError::backend(s);
            self.prefill_compute(token_ids, start_position, block_tables)?;
            let f16v: Vec<f16> = self.device.dtoh_sync_copy(&self.logits_buf)
                .map_err(|err| e(format!("logits dl: {err}")))?;
            Ok(f16v.iter().map(|v| v.to_f32()).collect())
        }

        // ── Batched decode step ───────────────────────────────────────────
        //
        // Processes ALL active sequences in one forward pass.
        //
        // Each seq contributes exactly one decode token.  Block tables and
        // context_len in `seqs` must already be updated for this step before
        // calling (context_len = old + 1, block for position context_len-1
        // already allocated).
        //
        // Returns one output token per sequence (in the same order).

        fn batch_decode_compute(&mut self, seqs: &[ActiveDecodeSeq]) -> Result<Vec<u32>, EngineError> {
            let b = seqs.len();
            debug_assert!(b > 0 && b <= MAX_BATCH);

            let h      = self.config.hidden_size;
            let num_q  = self.config.num_attention_heads;
            let num_kv = self.config.num_kv_heads();
            let hd     = self.config.head_dim();
            let inter  = self.config.intermediate_size;
            let eps    = self.config.rms_norm_eps as f32;
            let theta  = self.config.rope_theta as f32;
            let scale  = 1.0 / (hd as f32).sqrt();
            let bs     = self.block_pool.block_size();
            let vocab  = self.config.vocab_size;
            let e = |s: String| EngineError::backend(s);
            let blas = Arc::clone(&self.blas);

            // Upload positions (= context_len - 1) and context_lens.
            let positions_host: Vec<i32> = seqs.iter().map(|s| (s.context_len - 1) as i32).collect();
            let ctx_lens_host: Vec<i32>  = seqs.iter().map(|s| s.context_len as i32).collect();
            self.device.htod_sync_copy_into(&positions_host, &mut self.batch.positions)
                .map_err(|err| e(format!("positions upload: {err}")))?;
            self.device.htod_sync_copy_into(&ctx_lens_host, &mut self.batch.ctx_lens)
                .map_err(|err| e(format!("ctx_lens upload: {err}")))?;

            // Embed lookup: copy one embed row per sequence into batch.hidden.
            for (i, seq) in seqs.iter().enumerate() {
                let off = seq.next_token as usize * h;
                self.device.dtod_copy(
                    &self.weights.embed_tokens.slice(off..off + h),
                    &mut self.batch.hidden.slice_mut(i * h..(i + 1) * h),
                ).map_err(|err| e(format!("embed: {err}")))?;
            }

            let q_dim  = num_q  * hd;
            let kv_dim = num_kv * hd;

            for layer_idx in 0..self.weights.layers.len() {
                let layer = &self.weights.layers[layer_idx];

                // RMS norm over B rows.
                launch_rms_norm(&self.device, &mut RmsNormParams {
                    out: &mut self.batch.norm, input: &self.batch.hidden,
                    weight: &layer.input_layernorm,
                    rows: b as u32, dim: h as u32, eps,
                }).map_err(e)?;

                // Q / K / V projections: [B, h] → [B, q_dim / kv_dim].
                Self::gemm(&blas, q_dim as i32, b as i32, h as i32,
                    &layer.q_proj, h as i32, &self.batch.norm, h as i32,
                    &mut self.batch.q_buf, q_dim as i32, "bQ")?;
                Self::gemm(&blas, kv_dim as i32, b as i32, h as i32,
                    &layer.k_proj, h as i32, &self.batch.norm, h as i32,
                    &mut self.batch.k_buf, kv_dim as i32, "bK")?;
                Self::gemm(&blas, kv_dim as i32, b as i32, h as i32,
                    &layer.v_proj, h as i32, &self.batch.norm, h as i32,
                    &mut self.batch.v_buf, kv_dim as i32, "bV")?;

                // Per-sequence RoPE (each seq at its own absolute position).
                launch_batch_decode_rope(&self.device, &mut BatchDecodeRopeParams {
                    q: &mut self.batch.q_buf, k: &mut self.batch.k_buf,
                    positions: &self.batch.positions,
                    batch_size: b as u32,
                    num_q_heads: num_q as u32, num_kv_heads: num_kv as u32,
                    head_dim: hd as u32, theta,
                }).map_err(e)?;

                // Write K/V for each seq's current token to the paged pool.
                let mut phys_host = Vec::with_capacity(b);
                let mut pos_blk_host = Vec::with_capacity(b);
                for seq in seqs.iter() {
                    let pos = seq.context_len - 1;
                    phys_host.push(seq.block_tables[layer_idx][pos / bs]);
                    pos_blk_host.push((pos % bs) as i32);
                }
                let phys_dev = self.device.htod_sync_copy(&phys_host)
                    .map_err(|err| e(format!("phys_dev: {err}")))?;
                let pos_dev = self.device.htod_sync_copy(&pos_blk_host)
                    .map_err(|err| e(format!("pos_dev: {err}")))?;

                launch_batch_kv_write(&self.device, &mut BatchKvWriteParams {
                    kv_cache: self.block_pool.storage_mut(),
                    k: &self.batch.k_buf, v: &self.batch.v_buf,
                    physical_blocks: &phys_dev, pos_in_blocks: &pos_dev,
                    seq_len: b as u32, num_kv_heads: num_kv as u32,
                    block_size: bs as u32, head_dim: hd as u32,
                }).map_err(e)?;

                // Build flattened batch block table [B, max_blocks_per_seq] for this layer.
                let max_blks = seqs.iter().map(|s| s.block_tables[layer_idx].len()).max().unwrap_or(1);
                let mut bt_host = vec![0i32; b * max_blks];
                for (i, seq) in seqs.iter().enumerate() {
                    let src = &seq.block_tables[layer_idx];
                    bt_host[i * max_blks..i * max_blks + src.len()].copy_from_slice(src);
                }
                let bt_dev = self.device.htod_sync_copy(&bt_host)
                    .map_err(|err| e(format!("bt upload: {err}")))?;

                launch_paged_attention(&self.device, &mut PagedAttentionParams {
                    out: &mut self.batch.attn_buf,
                    q: &self.batch.q_buf,
                    kv_cache: self.block_pool.storage(),
                    block_tables: &bt_dev,
                    context_lens: &self.batch.ctx_lens,
                    scale,
                    batch_size: b as u32,
                    num_q_heads: num_q as u32, num_kv_heads: num_kv as u32,
                    head_dim: hd as u32, block_size: bs as u32,
                    max_blocks_per_seq: max_blks as u32,
                }).map_err(e)?;

                let layer = &self.weights.layers[layer_idx];
                Self::gemm(&blas, h as i32, b as i32, q_dim as i32,
                    &layer.o_proj, q_dim as i32, &self.batch.attn_buf, q_dim as i32,
                    &mut self.batch.o_proj, h as i32, "bO")?;

                launch_residual_add(&self.device, &mut self.batch.residual,
                    &self.batch.hidden, &self.batch.o_proj, (b * h) as u32).map_err(e)?;

                let layer = &self.weights.layers[layer_idx];
                launch_rms_norm(&self.device, &mut RmsNormParams {
                    out: &mut self.batch.ffn_input, input: &self.batch.residual,
                    weight: &layer.post_attention_layernorm,
                    rows: b as u32, dim: h as u32, eps,
                }).map_err(e)?;

                let layer = &self.weights.layers[layer_idx];
                Self::gemm(&blas, inter as i32, b as i32, h as i32,
                    &layer.gate_proj, h as i32, &self.batch.ffn_input, h as i32,
                    &mut self.batch.gate_buf, inter as i32, "bgate")?;
                Self::gemm(&blas, inter as i32, b as i32, h as i32,
                    &layer.up_proj, h as i32, &self.batch.ffn_input, h as i32,
                    &mut self.batch.up_buf, inter as i32, "bup")?;

                launch_fused_swiglu(&self.device, &mut self.batch.swiglu_buf,
                    &self.batch.gate_buf, &self.batch.up_buf, (b * inter) as u32).map_err(e)?;

                let layer = &self.weights.layers[layer_idx];
                Self::gemm(&blas, h as i32, b as i32, inter as i32,
                    &layer.down_proj, inter as i32, &self.batch.swiglu_buf, inter as i32,
                    &mut self.batch.ffn_out, h as i32, "bdown")?;

                launch_residual_add(&self.device, &mut self.batch.hidden,
                    &self.batch.residual, &self.batch.ffn_out, (b * h) as u32).map_err(e)?;
            }

            // Final norm + LM head.
            launch_rms_norm(&self.device, &mut RmsNormParams {
                out: &mut self.batch.norm, input: &self.batch.hidden,
                weight: &self.weights.norm,
                rows: b as u32, dim: h as u32, eps,
            }).map_err(e)?;

            Self::gemm(&blas, vocab as i32, b as i32, h as i32,
                &self.weights.lm_head, h as i32, &self.batch.norm, h as i32,
                &mut self.batch.logits, vocab as i32, "blm_head")?;

            // Sampling: GPU argmax when all are greedy, CPU otherwise.
            let any_sampled = seqs.iter().any(|s| !s.sp.is_greedy());
            if !any_sampled {
                launch_batch_argmax(&self.device, &mut BatchArgmaxParams {
                    input: &self.batch.logits, output: &mut self.batch.argmax,
                    batch_size: b as u32, vocab_size: vocab as u32,
                }).map_err(|s| EngineError::backend(s))?;
                let winners: Vec<u32> = self.device.dtoh_sync_copy(&self.batch.argmax)
                    .map_err(|err| e(format!("argmax dl: {err}")))?;
                Ok(winners[..b].to_vec())
            } else {
                let all_f16: Vec<f16> = self.device.dtoh_sync_copy(&self.batch.logits)
                    .map_err(|err| e(format!("logits dl: {err}")))?;
                // Collect into f32 rows, then sample each.  We can't call self.sample()
                // inside the iterator because it borrows self mutably; collect first.
                let rows: Vec<Vec<f32>> = (0..b)
                    .map(|i| all_f16[i * vocab..(i + 1) * vocab].iter().map(|v| v.to_f32()).collect())
                    .collect();
                let mut tokens = Vec::with_capacity(b);
                for (row, seq) in rows.iter().zip(seqs.iter()) {
                    tokens.push(self.sample(row, &seq.sp));
                }
                Ok(tokens)
            }
        }
    }

    // ── Active sequence in the scheduler ─────────────────────────────────

    struct ActiveDecodeSeq {
        /// Token to emit this step AND use as input for this step's forward pass.
        next_token: u32,
        block_tables: Vec<Vec<i32>>,
        /// Number of KV positions currently filled (attention attends to [0..context_len-1]).
        context_len: usize,
        generated: Vec<u32>,
        max_new_tokens: u32,
        eos: Option<u32>,
        sp: SampleParams,
        session_id: Option<String>,
        cancel: Option<kapsl_engine_api::CancellationToken>,
        stream_tx: Option<tokio::sync::mpsc::Sender<Result<BinaryTensorPacket, EngineError>>>,
        result_tx: tokio::sync::oneshot::Sender<Result<Vec<u32>, EngineError>>,
    }

    impl ActiveDecodeSeq {
        fn finish(self, result: Result<Vec<u32>, EngineError>) {
            let _ = self.result_tx.send(result);
        }
    }

    /// Save session KV state (if stateful) and finish the sequence.
    /// Frees paged blocks for stateless sequences.
    fn finish_seq(
        done: ActiveDecodeSeq,
        inner: &mut BackendInner,
        active_sessions: &mut HashSet<String>,
    ) {
        if let Some(sid) = &done.session_id { active_sessions.remove(sid); }
        let generated = done.generated.clone();
        if let Some(sid) = done.session_id.clone() {
            inner.sessions.insert(sid, SessionState {
                block_tables: done.block_tables.clone(),
                context_len: done.context_len,
            });
            done.finish(Ok(generated));
        } else {
            let bt = done.block_tables.clone();
            done.finish(Ok(generated));
            inner.free_block_tables(&bt);
        }
    }

    // ── Scheduler protocol ────────────────────────────────────────────────

    struct SchedulerRequest {
        prompt_ids: Vec<u32>,
        max_new_tokens: u32,
        eos: Option<u32>,
        sp: SampleParams,
        session_id: Option<String>,
        cancel: Option<kapsl_engine_api::CancellationToken>,
        stream_tx: Option<tokio::sync::mpsc::Sender<Result<BinaryTensorPacket, EngineError>>>,
        result_tx: tokio::sync::oneshot::Sender<Result<Vec<u32>, EngineError>>,
    }

    enum SchedulerCmd {
        Request(SchedulerRequest),
        /// Store f16-converted CPU weights ready for GPU swap.
        StoreStaged {
            staged: StagedF16Weights,
            reply: tokio::sync::oneshot::Sender<()>,
        },
        /// Activate staged weights; fails all active sequences.
        Swap {
            reply: tokio::sync::oneshot::Sender<Result<(), EngineError>>,
        },
    }

    // ── Scheduler thread ──────────────────────────────────────────────────

    /// Run prefill for a new request.  Returns `None` and sends the error
    /// via `req.result_tx` if prefill fails.
    fn prefill_request(
        inner: &mut BackendInner,
        req: SchedulerRequest,
    ) -> Option<ActiveDecodeSeq> {
        if req.prompt_ids.is_empty() {
            let _ = req.result_tx.send(Ok(Vec::new()));
            return None;
        }

        // Resolve or create session state.
        let session_id = req.session_id.clone();
        let mut session = session_id.as_deref()
            .and_then(|sid| inner.sessions.remove(sid))
            .unwrap_or(SessionState { block_tables: Vec::new(), context_len: 0 });

        // Ensure blocks for all prompt positions.
        for i in 0..req.prompt_ids.len() {
            if let Err(e) = inner.ensure_block(&mut session.block_tables, session.context_len + i) {
                inner.free_block_tables(&session.block_tables);
                let _ = req.result_tx.send(Err(e));
                return None;
            }
        }

        let start_pos = session.context_len as u32;
        let first_token = if req.sp.is_greedy() {
            inner.forward_prefill_greedy(&req.prompt_ids, start_pos, &session.block_tables)
        } else {
            // Two separate statements so the &mut inner borrow from forward_prefill
            // is fully released before inner.sample() borrows it again.
            let logits = inner.forward_prefill(&req.prompt_ids, start_pos, &session.block_tables);
            logits.map(|ls| inner.sample(&ls, &req.sp))
        };

        let first_token = match first_token {
            Ok(t) => t,
            Err(e) => {
                inner.free_block_tables(&session.block_tables);
                let _ = req.result_tx.send(Err(e));
                return None;
            }
        };

        session.context_len += req.prompt_ids.len();

        Some(ActiveDecodeSeq {
            next_token: first_token,
            block_tables: session.block_tables,
            context_len: session.context_len,
            generated: Vec::new(),
            max_new_tokens: req.max_new_tokens,
            eos: req.eos,
            sp: req.sp,
            session_id,
            cancel: req.cancel,
            stream_tx: req.stream_tx,
            result_tx: req.result_tx,
        })
    }

    fn dispatch_cmd(
        cmd: SchedulerCmd,
        inner: &mut BackendInner,
        active: &mut Vec<ActiveDecodeSeq>,
        active_sessions: &mut HashSet<String>,
    ) {
        match cmd {
            SchedulerCmd::Request(req) => {
                if let Some(sid) = &req.session_id {
                    if active_sessions.contains(sid) {
                        let _ = req.result_tx.send(Err(EngineError::backend(
                            "session is already active; previous generation still in progress",
                        )));
                        return;
                    }
                }
                if let Some(seq) = prefill_request(inner, req) {
                    if let Some(sid) = &seq.session_id {
                        active_sessions.insert(sid.clone());
                    }
                    active.push(seq);
                }
            }
            SchedulerCmd::StoreStaged { staged, reply } => {
                inner.staged = Some(staged);
                let _ = reply.send(());
            }
            SchedulerCmd::Swap { reply } => {
                for seq in active.drain(..) {
                    if let Some(sid) = &seq.session_id { active_sessions.remove(sid); }
                    seq.finish(Err(EngineError::backend("model swapped; retry request")));
                }
                let _ = reply.send(inner.activate_staged());
            }
        }
    }

    fn run_scheduler(mut inner: BackendInner, inbox: std::sync::mpsc::Receiver<SchedulerCmd>) {
        let mut active: Vec<ActiveDecodeSeq> = Vec::new();
        // Sessions currently held by an active decode sequence (cannot be re-entered).
        let mut active_sessions: HashSet<String> = HashSet::new();

        loop {
            // ── Drain pending commands ────────────────────────────────────

            // When idle, block until at least one command arrives.
            if active.is_empty() {
                match inbox.recv() {
                    Ok(cmd) => dispatch_cmd(cmd, &mut inner, &mut active, &mut active_sessions),
                    Err(_) => return, // all senders dropped = shutdown
                }
            }

            // Non-blocking drain of the remaining backlog (up to MAX_BATCH).
            while active.len() < MAX_BATCH {
                match inbox.try_recv() {
                    Ok(cmd) => dispatch_cmd(cmd, &mut inner, &mut active, &mut active_sessions),
                    Err(_) => break,
                }
            }

            if active.is_empty() {
                continue;
            }

            // ── Pre-decode: EOS / cancel check ───────────────────────────
            //
            // If next_token is EOS or the request is cancelled, finish the
            // sequence without emitting the EOS token (matches original semantics).

            let mut i = 0;
            while i < active.len() {
                let seq = &active[i];
                let is_eos = seq.eos.map_or(false, |e| seq.next_token == e);
                let is_cancelled = seq.cancel.as_ref().map_or(false, |c| c.is_cancelled());

                if is_eos || is_cancelled {
                    let done = active.swap_remove(i);
                    finish_seq(done, &mut inner, &mut active_sessions);
                } else {
                    i += 1;
                }
            }

            if active.is_empty() { continue; }

            // ── Ensure blocks for this decode step ────────────────────────
            //
            // Each seq needs a block at position `context_len` (where
            // next_token will be written by the forward pass).

            i = 0;
            while i < active.len() {
                let pos = active[i].context_len;
                let bt = &mut active[i].block_tables;
                if let Err(e) = inner.ensure_block(bt, pos) {
                    let done = active.swap_remove(i);
                    if let Some(sid) = &done.session_id { active_sessions.remove(sid); }
                    let bt = done.block_tables.clone();
                    done.finish(Err(e));
                    inner.free_block_tables(&bt);
                } else {
                    active[i].context_len += 1;
                    i += 1;
                }
            }

            if active.is_empty() { continue; }

            // ── Batched forward pass ──────────────────────────────────────
            //
            // Processes each seq's next_token, writes K/V to paged pool,
            // returns one output token per sequence.

            let output_tokens = match inner.batch_decode_compute(&active) {
                Ok(t) => t,
                Err(e) => {
                    let err_str = format!("{e}");
                    for done in active.drain(..) {
                        if let Some(sid) = &done.session_id { active_sessions.remove(sid); }
                        let bt = done.block_tables.clone();
                        done.finish(Err(EngineError::backend(err_str.clone())));
                        inner.free_block_tables(&bt);
                    }
                    continue;
                }
            };

            // ── Post-decode: emit input token, update next_token ──────────
            //
            // We emit the INPUT token (next_token before the forward pass)
            // since that is what the forward pass just placed into the KV cache.
            // The OUTPUT token becomes the next next_token.

            i = 0;
            while i < active.len() {
                let input_tok  = active[i].next_token;
                let output_tok = output_tokens[i];

                // Emit the input token.
                active[i].generated.push(input_tok);
                let stream_closed = if let Some(tx) = &active[i].stream_tx {
                    tx.blocking_send(Ok(token_to_packet(input_tok))).is_err()
                } else {
                    false
                };

                // Advance to the output token for the next step.
                active[i].next_token = output_tok;

                let is_done = stream_closed
                    || active[i].generated.len() >= active[i].max_new_tokens as usize
                    || active[i].cancel.as_ref().map_or(false, |c| c.is_cancelled());

                if is_done {
                    let done = active.swap_remove(i);
                    finish_seq(done, &mut inner, &mut active_sessions);
                } else {
                    i += 1;
                }
            }
        }
    }

    // ── NativeBackend ─────────────────────────────────────────────────────

    struct BackendState {
        tx: std::sync::mpsc::SyncSender<SchedulerCmd>,
        thread: Option<std::thread::JoinHandle<()>>,
        is_staged: Arc<AtomicBool>,
    }

    pub struct NativeBackend {
        device_id: i32,
        state: Arc<Mutex<Option<BackendState>>>,
    }

    impl NativeBackend {
        pub fn new(device_id: i32) -> Result<Self, EngineError> {
            CudaDevice::new(device_id as usize)
                .map_err(|e| EngineError::backend(format!("CUDA device {device_id}: {e}")))?;
            Ok(Self { device_id, state: Arc::new(Mutex::new(None)) })
        }

        fn get_tx(&self) -> Result<std::sync::mpsc::SyncSender<SchedulerCmd>, EngineError> {
            self.state.lock().unwrap()
                .as_ref()
                .map(|bs| bs.tx.clone())
                .ok_or(EngineError::ModelNotLoaded)
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
            let num_blocks = config.num_hidden_layers * MAX_BATCH * bps;
            let block_pool = GpuBlockPool::new(
                device.clone(), num_blocks, block_size,
                config.num_kv_heads(), config.head_dim(),
            ).map_err(|e| EngineError::backend(format!("block pool: {e}")))?;

            let h     = config.hidden_size;
            let nq    = config.num_attention_heads;
            let nkv   = config.num_kv_heads();
            let hd    = config.head_dim();
            let inter = config.intermediate_size;
            let vocab = config.vocab_size;

            let alloc1 = |n: usize| device.alloc_zeros::<f16>(n)
                .map_err(|e| EngineError::backend(format!("alloc: {e}")));

            let prefill = PrefillScratch::new(&device, h, nq * hd, nkv * hd, inter)?;
            let batch   = BatchDecodeScratch::new(&device, MAX_BATCH, h, nq * hd, nkv * hd, inter, vocab)?;

            let inner = BackendInner {
                device: device.clone(), blas, weights, block_pool, config,
                norm_buf:   alloc1(h)?,
                logits_buf: alloc1(vocab)?,
                argmax_buf: device.alloc_zeros::<u32>(1)
                    .map_err(|e| EngineError::backend(format!("argmax buf: {e}")))?,
                batch,
                sessions: HashMap::new(),
                rng: rand::rngs::SmallRng::from_entropy(),
                prefill,
                staged: None,
            };

            let (tx, rx) = std::sync::mpsc::sync_channel::<SchedulerCmd>(256);
            let handle = std::thread::spawn(move || run_scheduler(inner, rx));

            let mut guard = self.state.lock().unwrap();
            *guard = Some(BackendState {
                tx,
                thread: Some(handle),
                is_staged: Arc::new(AtomicBool::new(false)),
            });
            log::info!("NativeBackend: scheduler started (MAX_BATCH={})", MAX_BATCH);
            Ok(())
        }

        fn infer(&self, req: &InferenceRequest) -> Result<BinaryTensorPacket, EngineError> {
            let tx = self.get_tx()?;
            let prompt = Self::extract_token_ids(req)?;
            let (max_new, eos, sp) = Self::decode_params(req);
            let (result_tx, result_rx) = tokio::sync::oneshot::channel();

            tx.send(SchedulerCmd::Request(SchedulerRequest {
                prompt_ids: prompt,
                max_new_tokens: max_new,
                eos,
                sp,
                session_id: req.session_id.clone(),
                cancel: req.cancellation.clone(),
                stream_tx: None,
                result_tx,
            })).map_err(|_| EngineError::backend("scheduler not running"))?;

            let generated = result_rx.blocking_recv()
                .map_err(|_| EngineError::backend("scheduler crashed"))??;
            pack_tokens(&generated)
        }

        fn infer_stream(&self, req: &InferenceRequest) -> EngineStream {
            let tx = match self.get_tx() {
                Ok(t) => t,
                Err(e) => return Box::pin(futures::stream::once(async { Err(e) })),
            };

            let prompt = match Self::extract_token_ids(req) {
                Ok(v) => v,
                Err(e) => return Box::pin(futures::stream::once(async { Err(e) })),
            };
            let (max_new, eos, sp) = Self::decode_params(req);
            let (result_tx, result_rx) = tokio::sync::oneshot::channel::<Result<Vec<u32>, EngineError>>();
            let (stream_tx, stream_rx) = tokio::sync::mpsc::channel::<Result<BinaryTensorPacket, EngineError>>(64);

            let send_result = tx.send(SchedulerCmd::Request(SchedulerRequest {
                prompt_ids: prompt,
                max_new_tokens: max_new,
                eos,
                sp,
                session_id: req.session_id.clone(),
                cancel: req.cancellation.clone(),
                stream_tx: Some(stream_tx),
                result_tx,
            }));

            if let Err(_) = send_result {
                let e = EngineError::backend("scheduler not running");
                return Box::pin(futures::stream::once(async { Err(e) }));
            }

            // Drop result_rx — we only care about the stream of individual tokens.
            drop(result_rx);

            Box::pin(futures::stream::unfold(stream_rx, |mut rx| async move {
                rx.recv().await.map(|item| (item, rx))
            }))
        }

        fn unload(&mut self) {
            let state = self.state.lock().unwrap().take();
            if let Some(mut bs) = state {
                // Dropping the sender causes the scheduler to see a closed channel and exit.
                drop(bs.tx);
                if let Some(t) = bs.thread.take() {
                    let _ = t.join();
                }
            }
            log::info!("NativeBackend: unloaded");
        }

        fn metrics(&self) -> EngineMetrics {
            // Metrics are approximate — reading from the scheduler thread would require
            // a round-trip.  Return zeros for now; a future PR can add atomic counters.
            EngineMetrics {
                inference_time: 0.0, memory_usage: 0, gpu_utilization: 0.0,
                throughput: 0.0, batch_size: MAX_BATCH as u32, queue_depth: 0, error_rate: 0.0,
                collected_at_ms: 0, kv_cache_bytes_used: 0, kv_cache_bytes_capacity: 0,
                kv_cache_blocks_total: 0, kv_cache_blocks_free: 0,
                kv_cache_sequences: 0,
                kv_cache_evicted_blocks: 0, kv_cache_evicted_sequences: 0,
            }
        }

        fn model_info(&self) -> Option<EngineModelInfo> {
            // We can't query the scheduler thread synchronously from a non-blocking fn.
            // Return None when unloaded; when loaded we return static info.
            let loaded = self.state.lock().unwrap().is_some();
            if !loaded { return None; }
            Some(EngineModelInfo {
                input_names: vec!["input_ids".into()],
                output_names: vec!["output_ids".into()],
                input_shapes: vec![vec![-1]],
                output_shapes: vec![vec![-1]],
                input_dtypes: vec!["int32".into()],
                output_dtypes: vec!["int32".into()],
                framework: Some("native-cuda".into()),
                model_version: Some("native-cuda".into()),
                peak_concurrency: Some(MAX_BATCH as u32),
            })
        }

        fn health_check(&self) -> Result<(), EngineError> {
            if self.state.lock().unwrap().is_some() { Ok(()) }
            else { Err(EngineError::ModelNotLoaded) }
        }

        fn supports_swap(&self) -> bool { true }

        fn is_staged(&self) -> bool {
            self.state.lock().unwrap()
                .as_ref()
                .map(|bs| bs.is_staged.load(Ordering::Relaxed))
                .unwrap_or(false)
        }

        async fn stage(&self, path: &Path) -> Result<(), EngineError> {
            let path = path.to_owned();
            let tx = self.get_tx()?;
            let is_staged_flag = self.state.lock().unwrap()
                .as_ref()
                .map(|bs| Arc::clone(&bs.is_staged))
                .ok_or(EngineError::ModelNotLoaded)?;

            let staged = tokio::task::spawn_blocking(move || {
                let dir = if path.is_dir() { path.clone() } else {
                    path.parent().unwrap_or(&path).to_path_buf()
                };
                log::info!("NativeBackend: staging model from {:?}…", dir);
                let weights = load_safetensors(&dir)
                    .map_err(|e| EngineError::backend(format!("stage load: {e}")))?;
                log::info!(
                    "NativeBackend: converting {} layers to f16 during staging…",
                    weights.layers.len()
                );
                Ok::<_, EngineError>(BackendInner::to_staged(weights))
            })
            .await
            .map_err(|e| EngineError::backend(format!("stage task: {e}")))??;

            let (reply_tx, reply_rx) = tokio::sync::oneshot::channel();
            tx.send(SchedulerCmd::StoreStaged { staged, reply: reply_tx })
                .map_err(|_| EngineError::backend("scheduler not running"))?;
            reply_rx.await
                .map_err(|_| EngineError::backend("scheduler crashed"))?;
            is_staged_flag.store(true, Ordering::Relaxed);
            Ok(())
        }

        async fn swap(&self) -> Result<(), EngineError> {
            let tx = self.get_tx()?;
            let is_staged_flag = self.state.lock().unwrap()
                .as_ref()
                .map(|bs| Arc::clone(&bs.is_staged))
                .ok_or(EngineError::ModelNotLoaded)?;

            let (reply_tx, reply_rx) = tokio::sync::oneshot::channel();
            tx.send(SchedulerCmd::Swap { reply: reply_tx })
                .map_err(|_| EngineError::backend("scheduler not running"))?;
            let result = reply_rx.await
                .map_err(|_| EngineError::backend("scheduler crashed"))?;
            if result.is_ok() {
                is_staged_flag.store(false, Ordering::Relaxed);
            }
            result
        }
    }

    fn token_to_packet(token_id: u32) -> BinaryTensorPacket {
        let data = (token_id as i32).to_le_bytes().to_vec();
        BinaryTensorPacket::new(vec![1], TensorDtype::Int32, data)
            .expect("valid packet")
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

#[cfg(all(feature = "native", test))]
#[path = "native_hotswap_tests.rs"]
mod native_hotswap_tests;
