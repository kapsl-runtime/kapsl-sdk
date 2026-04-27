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
        launch_argmax, launch_batch_kv_write, launch_batch_rope, launch_fused_swiglu,
        launch_paged_attention, launch_prefill_attention, launch_residual_add, launch_rms_norm,
        ArgmaxParams, BatchKvWriteParams, BatchRopeParams, PagedAttentionParams, PrefillAttnParams,
        RmsNormParams,
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

    fn upload_f16(device: &Arc<CudaDevice>, v: &[f16]) -> Result<CudaSlice<f16>, EngineError> {
        device.htod_sync_copy(v)
            .map_err(|e| EngineError::backend(format!("GPU upload: {e}")))
    }

    /// Upload pre-converted f16 staged weights — no dtype conversion, pure PCIe transfer.
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
    }

    // ── Session state ─────────────────────────────────────────────────────

    struct SessionState {
        /// [layer_idx][logical_block] = physical_block_id
        block_tables: Vec<Vec<i32>>,
        context_len: usize,
    }

    // ── Prefill scratch buffers ───────────────────────────────────────────

    /// GPU scratch buffers for batch prefill.  Capacity grows on demand and is
    /// never shrunk — reused across calls when the new prompt fits.
    struct PrefillScratch {
        /// Current capacity in tokens.
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

    // ── BackendInner ──────────────────────────────────────────────────────

    struct BackendInner {
        device: Arc<CudaDevice>,
        blas: Arc<CudaBlas>,
        config: ModelConfig,
        weights: GpuModelWeights,
        block_pool: GpuBlockPool,
        // ── Pre-allocated decode activation buffers (size = config dims) ──
        hidden_buf: CudaSlice<f16>,
        norm_buf: CudaSlice<f16>,
        residual_buf: CudaSlice<f16>,
        q_buf: CudaSlice<f16>,
        k_buf: CudaSlice<f16>,
        v_buf: CudaSlice<f16>,
        /// Attention output [num_q_heads * head_dim]
        attn_buf: CudaSlice<f16>,
        /// FFN gate / up / swiglu / ffn_input / ffn_out / o_proj [intermediate or hidden]
        gate_buf: CudaSlice<f16>,
        up_buf: CudaSlice<f16>,
        swiglu_buf: CudaSlice<f16>,
        ffn_input_buf: CudaSlice<f16>,
        ffn_out_buf: CudaSlice<f16>,
        o_proj_buf: CudaSlice<f16>,
        /// Logits buffer [vocab_size] — reused every decode step
        logits_buf: CudaSlice<f16>,
        /// context-length scalar for paged attention [1]
        ctx_scalar_buf: CudaSlice<i32>,
        // ── Block-table GPU cache ──────────────────────────────────────────
        /// Per-layer cached GPU block tables. Re-uploaded only when a new
        /// physical block is allocated (every `block_size` decode steps).
        gpu_block_tables: Vec<CudaSlice<i32>>,
        /// Number of logical blocks currently reflected in `gpu_block_tables`.
        /// When this is less than `block_tables[0].len()`, the cache is stale.
        gpu_block_table_len: usize,
        // ── Session state ─────────────────────────────────────────────────
        /// Session ID → preserved KV state for multi-turn conversations.
        sessions: HashMap<String, SessionState>,
        rng: rand::rngs::SmallRng,
        // ── Prefill scratch (capacity-tracked, never shrunk) ─────────────
        prefill: PrefillScratch,
        /// Pre-allocated [1] buffer: receives the GPU argmax output for greedy decode.
        argmax_buf: CudaSlice<u32>,
        // ── Hot-swap staging ─────────────────────────────────────────────
        /// Next model weights pre-converted to f16 in CPU RAM, ready for a
        /// fast GPU swap (PCIe transfer only — disk I/O already done by stage()).
        /// Layout: flat Vec per tensor in the same order as upload_weights().
        staged: Option<StagedF16Weights>,
    }

    /// Pre-converted f16 weights in CPU RAM — the output of stage(), input to swap().
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

        /// Upload block tables to GPU if the CPU-side has grown since the last upload.
        /// Returns a slice of the current GPU block-table slices for use in attention.
        fn sync_gpu_block_tables(&mut self, block_tables: &[Vec<i32>])
            -> Result<(), EngineError>
        {
            let cpu_len = block_tables.first().map_or(0, |v| v.len());
            if cpu_len == self.gpu_block_table_len {
                return Ok(());  // cache is current
            }
            // Resize the GPU cache vector if needed (first call or layer count changed).
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

        // ── Hot-swap ──────────────────────────────────────────────────────────

        /// Pre-convert ModelWeights → StagedF16Weights (CPU only, no GPU involved).
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

            log::info!("NativeBackend: activating staged weights (PCIe transfer only — f16 already converted)…");
            let new_weights = upload_staged(&self.device, &staged)?;
            self.weights = new_weights;
            self.config = staged.config;
            // Invalidate GPU block-table cache — sessions are about to be cleared anyway.
            self.gpu_block_tables.clear();
            self.gpu_block_table_len = 0;

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

        /// Core prefill forward pass — leaves the last-token logits in `self.logits_buf`.
        /// Grows `self.prefill` scratch buffers on demand; never frees them.
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

            let mut phys_blocks_host: Vec<i32> = Vec::with_capacity(n);
            let mut pos_in_blk_host:  Vec<i32> = Vec::with_capacity(n);
            for pos in 0..n {
                let logical = pos / block_size;
                phys_blocks_host.push(block_tables[0][logical]);
                pos_in_blk_host.push((pos % block_size) as i32);
            }
            // (per-layer these differ only when layers have different block tables,
            //  which is currently always the case — we upload per layer in the loop.)

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

        /// Prefill + full logits download.  Use only when temperature > 0 (sampling path).
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

        /// Prefill + GPU argmax — transfers only 4 bytes instead of the full logits vector.
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

        // ── Single-token decode ───────────────────────────────────────────

        /// Core single-token forward pass — leaves logits in `self.logits_buf`.
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

            // Embed lookup — single dtod_copy, no allocation.
            let embed_off = token_id as usize * h;
            self.device.dtod_copy(
                &self.weights.embed_tokens.slice(embed_off..embed_off + h),
                &mut self.hidden_buf,
            ).map_err(|err| e(format!("embed: {err}")))?;

            // Update ctx scalar in-place to avoid allocation.
            self.device.htod_sync_copy_into(&[context_len as i32], &mut self.ctx_scalar_buf)
                .map_err(|err| e(format!("ctx_dev: {err}")))?;

            let pos_in_seq = context_len - 1;
            let pos_in_block = pos_in_seq % block_size;

            // Sync GPU block table cache (uploads only when a new block was allocated).
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

                // Paged attention — use cached GPU block table (no per-layer H2D copy).
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

                // Output projection — reuse pre-allocated o_proj_buf.
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

        /// Single-token forward + full logits download.  Use only when temperature > 0.
        fn forward_one_token(
            &mut self,
            token_id: u32,
            block_tables: &[Vec<i32>],
            context_len: usize,
            position: u32,
        ) -> Result<Vec<f32>, EngineError> {
            let e = |s: String| EngineError::backend(s);
            self.one_token_compute(token_id, block_tables, context_len, position)?;
            let f16v: Vec<f16> = self.device.dtoh_sync_copy(&self.logits_buf)
                .map_err(|err| e(format!("logits dl: {err}")))?;
            Ok(f16v.iter().map(|v| v.to_f32()).collect())
        }

        /// Single-token forward + GPU argmax — transfers only 4 bytes per step.
        fn forward_one_token_greedy(
            &mut self,
            token_id: u32,
            block_tables: &[Vec<i32>],
            context_len: usize,
            position: u32,
        ) -> Result<u32, EngineError> {
            let vocab = self.config.vocab_size;
            let e = |s: String| EngineError::backend(s);
            self.one_token_compute(token_id, block_tables, context_len, position)?;
            launch_argmax(&self.device, &mut ArgmaxParams {
                input: &self.logits_buf,
                output: &mut self.argmax_buf,
                vocab_size: vocab as u32,
            }).map_err(|s| EngineError::backend(s))?;
            let ids: Vec<u32> = self.device.dtoh_sync_copy(&self.argmax_buf)
                .map_err(|err| e(format!("argmax dl: {err}")))?;
            Ok(ids[0])
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
            let greedy = sp.temperature < 1e-6;

            let mut next = if greedy {
                self.forward_prefill_greedy(prompt_ids, start_position, block_tables)?
            } else {
                let logits = self.forward_prefill(prompt_ids, start_position, block_tables)?;
                self.sample(&logits, sp)
            };
            *context_len_ref += prompt_ids.len();

            let mut generated = Vec::new();

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

                next = if greedy {
                    self.forward_one_token_greedy(next, block_tables, *context_len_ref, position)?
                } else {
                    let logits = self.forward_one_token(next, block_tables, *context_len_ref, position)?;
                    self.sample(&logits, sp)
                };
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
                device, blas, weights, block_pool, config,
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
                staged:         None,
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

            let (generated, stateless_blocks) = if let Some(sid) = sid {
                let mut session = inner.sessions.remove(sid)
                    .unwrap_or_else(|| SessionState { block_tables: Vec::new(), context_len: 0 });
                let gen = inner.run_decode(&prompt, Some(&mut session), max_new, eos, &sp,
                    req.cancellation.as_ref(), None);
                inner.sessions.insert(sid.to_string(), session);
                let gen = gen?;
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
                    let mut session = b.sessions.remove(sid.as_str())
                        .unwrap_or_else(|| SessionState { block_tables: Vec::new(), context_len: 0 });
                    let r = b.run_decode(&prompt, Some(&mut session), max_new, eos, &sp, cancel.as_ref(), Some(&tx));
                    b.sessions.insert(sid.clone(), session);
                    r
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
                    "NativeBackend: converting {} layers to f16 during staging \
                     (swap will be PCIe-transfer-only)…",
                    weights.layers.len()
                );
                // Pre-convert to f16 now so activate_staged() only does PCIe transfer.
                let staged = BackendInner::to_staged(weights);
                log::info!("NativeBackend: staged {} layers ready", staged.layers.len());
                let mut guard = inner.lock().unwrap();
                let b = guard.as_mut().ok_or(EngineError::ModelNotLoaded)?;
                b.staged = Some(staged);
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

#[cfg(all(feature = "native", test))]
#[path = "native_hotswap_tests.rs"]
mod native_hotswap_tests;
