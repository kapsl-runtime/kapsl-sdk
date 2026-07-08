use crate::onnx::{ExecutionProvider, OnnxBackend, OnnxBackendBuilder};
use crate::provider_compat::{resolve_onnx_accelerator_provider, OnnxAcceleratorProvider};
#[cfg(feature = "gguf-native")]
use crate::gguf_native::GgufNativeBackend;
#[cfg(feature = "native")]
use crate::native::NativeBackend;
use kapsl_core::loader::Manifest;
use kapsl_core::EngineKind;
use kapsl_core::HardwareRequirements;
use kapsl_engine_api::Engine;
#[cfg(any(feature = "gguf-native", feature = "native", feature = "gguf-cuda-shared-kv"))]
use kapsl_hal::gpu_arena::GpuPoolHandle;
use kapsl_hal::device::DeviceInfo;
use kapsl_llm::llm_backend::LLMBackend;
use kapsl_llm::GgufBackend;
#[cfg(target_os = "windows")]
use ort::execution_providers::DirectMLExecutionProvider;
use ort::execution_providers::ExecutionProvider as _;
use ort::execution_providers::{
    CUDAExecutionProvider, CoreMLExecutionProvider, OpenVINOExecutionProvider,
    ROCmExecutionProvider, TensorRTExecutionProvider,
};
use ort::session::builder::GraphOptimizationLevel;

pub struct BackendFactory;

#[derive(Debug, Clone, Default)]
pub struct OnnxRuntimeTuning {
    pub memory_pattern: Option<bool>,
    pub disable_cpu_mem_arena: Option<bool>,
    pub session_buckets: Option<usize>,
    pub bucket_dim_granularity: Option<usize>,
    pub bucket_max_dims: Option<usize>,
    pub peak_concurrency_hint: Option<u32>,
}

/// Error for a declared-but-unimplemented engine cell (e.g. ONNX classification).
/// The triple is valid; there is just no backend yet.
fn unimplemented_engine_error(kind: EngineKind) -> String {
    format!(
        "model kind `{}` is declared but not yet implemented by a backend",
        kind.label()
    )
}

/// Read a boolean knob from `metadata.<section>.<key>`, defaulting to `default`.
fn manifest_metadata_bool(manifest: &Manifest, section: &str, key: &str, default: bool) -> bool {
    manifest
        .metadata
        .as_ref()
        .and_then(|m| m.get(section))
        .and_then(|s| s.get(key))
        .and_then(|v| v.as_bool())
        .unwrap_or(default)
}

/// Wrap a freshly built ONNX engine in the task post-processor the manifest asks
/// for (embedding pooling, classification softmax, or detection box-decode+NMS);
/// otherwise return it unchanged.
fn maybe_wrap_onnx_postprocess(
    engine_kind: EngineKind,
    manifest: &Manifest,
    inner: Box<dyn Engine>,
) -> Result<Box<dyn Engine>, String> {
    match engine_kind {
        EngineKind::OnnxEmbed => {
            // Default true: cosine == dot product for vector stores.
            let normalize = manifest_metadata_bool(manifest, "embed", "normalize", true);
            Ok(Box::new(crate::onnx_embed::OnnxEmbedBackend::new(
                inner, normalize,
            )))
        }
        EngineKind::OnnxClassify => {
            // Default true: most classifier exports emit raw logits.
            let apply_softmax = manifest_metadata_bool(manifest, "classify", "apply_softmax", true);
            Ok(Box::new(crate::onnx_classify::OnnxClassifyBackend::new(
                inner,
                apply_softmax,
            )))
        }
        EngineKind::OnnxDetect => {
            let spec = manifest
                .metadata
                .as_ref()
                .and_then(|m| m.get("detect"))
                .ok_or_else(|| {
                    "detect task requires a `metadata.detect` block (num_classes at minimum)"
                        .to_string()
                })?;
            let cfg: crate::onnx_detect::DetectConfig = serde_yaml::from_value(spec.clone())
                .map_err(|e| format!("invalid metadata.detect: {e}"))?;
            let backend = crate::onnx_detect::OnnxDetectBackend::new(inner, cfg)
                .map_err(|e| format!("detector init failed: {e}"))?;
            Ok(Box::new(backend))
        }
        EngineKind::OnnxTranscribe => {
            // metadata.transcribe is optional (defaults to blank-0, collapse on).
            let cfg = match manifest.metadata.as_ref().and_then(|m| m.get("transcribe")) {
                Some(spec) => serde_yaml::from_value(spec.clone())
                    .map_err(|e| format!("invalid metadata.transcribe: {e}"))?,
                None => crate::onnx_transcribe::TranscribeConfig::default(),
            };
            Ok(Box::new(crate::onnx_transcribe::OnnxTranscribeBackend::new(
                inner, cfg,
            )))
        }
        _ => Ok(inner),
    }
}

/// Wrap an ONNX engine in the input *pre*-processing stage its manifest asks for
/// (`metadata.preprocess`), if any. Only ONNX-session engines are eligible.
///
/// This is applied *outside* the post-processor so the runtime pipeline reads
/// `preprocess -> onnx forward -> postprocess`: e.g. an image classifier decodes
/// and normalizes the incoming JPEG, runs the forward pass, then softmaxes.
fn maybe_wrap_onnx_preprocess(
    engine_kind: EngineKind,
    manifest: &Manifest,
    engine: Box<dyn Engine>,
) -> Result<Box<dyn Engine>, String> {
    if !engine_kind.uses_onnx_session() {
        return Ok(engine);
    }
    let Some(spec) = manifest.metadata.as_ref().and_then(|m| m.get("preprocess")) else {
        return Ok(engine);
    };
    match spec.get("kind").and_then(|k| k.as_str()) {
        Some("vision") => {
            let cfg: crate::preprocess::vision::VisionConfig = serde_yaml::from_value(spec.clone())
                .map_err(|e| format!("invalid metadata.preprocess (vision): {e}"))?;
            let pre = crate::preprocess::VisionPreprocessor::new(cfg)
                .map_err(|e| format!("vision preprocess init failed: {e}"))?;
            log::info!("✓ Wrapping ONNX engine with vision input preprocessing");
            Ok(Box::new(crate::preprocess::PreprocessBackend::new(
                engine,
                Box::new(pre),
            )))
        }
        Some("audio") => {
            let cfg: crate::preprocess::audio::AudioConfig = serde_yaml::from_value(spec.clone())
                .map_err(|e| format!("invalid metadata.preprocess (audio): {e}"))?;
            let pre = crate::preprocess::AudioPreprocessor::new(cfg)
                .map_err(|e| format!("audio preprocess init failed: {e}"))?;
            log::info!("✓ Wrapping ONNX engine with audio (log-mel) input preprocessing");
            Ok(Box::new(crate::preprocess::PreprocessBackend::new(
                engine,
                Box::new(pre),
            )))
        }
        Some(other) => Err(format!("unknown metadata.preprocess kind `{other}`")),
        None => Err("metadata.preprocess is set but missing a `kind`".to_string()),
    }
}

/// Apply the ONNX task post-processor then the input pre-processor, yielding the
/// full serving pipeline for a freshly built ONNX engine.
fn finalize_onnx_pipeline(
    engine_kind: EngineKind,
    manifest: &Manifest,
    inner: Box<dyn Engine>,
) -> Result<Box<dyn Engine>, String> {
    let post = maybe_wrap_onnx_postprocess(engine_kind, manifest, inner)?;
    maybe_wrap_onnx_preprocess(engine_kind, manifest, post)
}

pub fn parse_optimization_level(level: Option<&String>) -> Result<GraphOptimizationLevel, String> {
    match level.as_ref().map(|s| s.as_str()) {
        Some("disable") | Some("0") => Ok(GraphOptimizationLevel::Disable),
        Some("basic") | Some("1") => Ok(GraphOptimizationLevel::Level1),
        Some("extended") | Some("2") => Ok(GraphOptimizationLevel::Level2),
        Some("all") | Some("3") | Some("99") | None => Ok(GraphOptimizationLevel::Level3),
        _ => Err("Unknown optimization level".to_string()),
    }
}

impl BackendFactory {
    fn apply_onnx_tuning(
        mut builder: OnnxBackendBuilder,
        tuning: &OnnxRuntimeTuning,
    ) -> OnnxBackendBuilder {
        if let Some(v) = tuning.memory_pattern {
            builder = builder.with_memory_pattern(v);
        }
        if let Some(v) = tuning.disable_cpu_mem_arena {
            builder = builder.with_disable_cpu_mem_arena(v);
        }
        if let Some(v) = tuning.session_buckets {
            builder = builder.with_max_bucket_sessions(v);
        }
        if let Some(v) = tuning.bucket_dim_granularity {
            builder = builder.with_bucket_dim_granularity(v);
        }
        if let Some(v) = tuning.bucket_max_dims {
            builder = builder.with_bucket_max_dims(v);
        }
        if let Some(v) = tuning.peak_concurrency_hint {
            builder = builder.with_peak_concurrency_hint(v);
        }
        builder
    }

    fn build_onnx_backend(
        provider: ExecutionProvider,
        opt_level: GraphOptimizationLevel,
        device_id: i32,
        tuning: &OnnxRuntimeTuning,
    ) -> Result<Box<dyn Engine>, String> {
        let mut builder = OnnxBackend::builder()
            .with_provider(provider)
            .with_optimization_level(opt_level);
        if !matches!(provider, ExecutionProvider::CPU) {
            builder = builder.with_device_id(device_id)?;
        }
        builder = Self::apply_onnx_tuning(builder, tuning);
        Ok(Box::new(builder.build()))
    }

    fn push_unique_provider(providers: &mut Vec<String>, provider: &str) {
        if providers
            .iter()
            .any(|candidate| candidate.eq_ignore_ascii_case(provider))
        {
            return;
        }
        providers.push(provider.to_string());
    }

    fn provider_policy() -> String {
        std::env::var("KAPSL_PROVIDER_POLICY")
            .or_else(|_| std::env::var("KAPSL_PROVIDER_POLICY"))
            .unwrap_or_else(|_| "fastest".to_string())
            .trim()
            .to_ascii_lowercase()
    }

    fn should_append_fastest_candidates(providers: &[String]) -> bool {
        if Self::provider_policy() == "manifest" {
            return false;
        }

        providers.is_empty()
            || providers
                .iter()
                .all(|provider| matches!(provider.trim().to_ascii_lowercase().as_str(), "" | "cpu"))
    }

    fn append_fastest_candidates(device_info: &DeviceInfo, providers: &mut Vec<String>) {
        if device_info.has_cuda {
            Self::push_unique_provider(providers, "tensorrt");
            Self::push_unique_provider(providers, "cuda");
        }
        if device_info.has_metal {
            Self::push_unique_provider(providers, "coreml");
        }
        if device_info.has_rocm {
            Self::push_unique_provider(providers, "rocm");
        }
        if device_info.has_directml {
            Self::push_unique_provider(providers, "directml");
        }
        Self::push_unique_provider(providers, "cpu");
    }

    /// Create the best available backend based on manifest requirements and available hardware
    pub fn create_best_backend(
        manifest: &Manifest,
        device_info: &DeviceInfo,
    ) -> Result<Box<dyn Engine>, String> {
        Self::create_best_backend_with_tuning(manifest, device_info, &OnnxRuntimeTuning::default())
    }

    pub fn create_best_backend_with_tuning(
        manifest: &Manifest,
        device_info: &DeviceInfo,
        tuning: &OnnxRuntimeTuning,
    ) -> Result<Box<dyn Engine>, String> {
        let engine_kind = EngineKind::resolve(manifest);

        // GGUF: prefer native CUDA kernels when gguf-native feature is compiled in.
        #[cfg(feature = "gguf-native")]
        if engine_kind.is_gguf() {
            let device_id = manifest.hardware_requirements.device_id.unwrap_or(0);
            log::info!("✓ Using GgufNativeBackend (GGUF loader + native CUDA), device {}", device_id);
            return crate::gguf_native::GgufNativeBackend::new(device_id as i32)
                .map(|b| Box::new(b) as Box<dyn Engine>)
                .map_err(|e| format!("GgufNativeBackend init failed: {e}"));
        }

        #[cfg(all(feature = "gguf-cuda-shared-kv", not(feature = "gguf-native")))]
        if engine_kind.is_gguf() {
            let device_id = manifest.hardware_requirements.device_id.unwrap_or(0);
            // Reuse the device pool registered with ORT (if any) so GGUF KV
            // and ONNX sessions draw from the same memory budget.
            let pool = crate::ort_pool_allocator::registered_pool_handle(device_id as i32);
            log::info!(
                "✓ Using GgufBackend (llama.cpp CUDA + Kapsl shared KV{}), device {}",
                if pool.is_some() { ", pool shared with ONNX" } else { "" },
                device_id
            );
            return Ok(Box::new(GgufBackend::new_cuda_shared_kv(
                device_id as usize,
                pool,
            )));
        }

        // GGUF: fallback to llama.cpp-backed GgufBackend.
        #[cfg(not(any(feature = "gguf-native", feature = "gguf-cuda-shared-kv")))]
        if engine_kind.is_gguf() {
            log::info!("✓ Using GgufBackend (llama.cpp)");
            return Ok(Box::new(GgufBackend::new()));
        }

        // Native CUDA: safetensors models use custom kernel backend.
        #[cfg(feature = "native")]
        if engine_kind == EngineKind::Native {
            let device_id = manifest.hardware_requirements.device_id.unwrap_or(0);
            log::info!("✓ Using NativeBackend (custom CUDA kernels), device {}", device_id);
            return crate::native::NativeBackend::new(device_id)
                .map(|b| Box::new(b) as Box<dyn Engine>)
                .map_err(|e| format!("NativeBackend init failed: {e}"));
        }

        // ONNX generative path (LLMBackend): autoregressive decode over an ONNX graph.
        if engine_kind.is_onnx_generate() {
            let requirements = &manifest.hardware_requirements;
            if Self::provider_policy() == "manifest" {
                if let Some(provider) = requirements.preferred_provider.clone() {
                    let device_id = requirements.device_id.unwrap_or(0);
                    log::info!(
                        "✓ Using LLMBackend with manifest provider override: {}",
                        provider
                    );
                    return Ok(Box::new(LLMBackend::with_device(provider, device_id)));
                }
            }
            log::info!("✓ Using LLMBackend with runtime fastest-provider selection");
            return Ok(Box::new(LLMBackend::new()));
        }

        if !engine_kind.is_implemented() {
            return Err(unimplemented_engine_error(engine_kind));
        }

        let requirements = &manifest.hardware_requirements;

        log::info!("🔍 Selecting backend based on requirements:");
        log::info!("   Preferred: {:?}", requirements.preferred_provider);
        log::info!("   Fallbacks: {:?}", requirements.fallback_providers);
        log::info!(
            "   Graph Optimization: {:?}",
            requirements.graph_optimization_level
        );

        // Parse and validate optimization level early (fail-fast)
        let opt_level = parse_optimization_level(requirements.graph_optimization_level.as_ref())
            .map_err(|e| format!("Invalid graph optimization level in manifest: {}", e))?;

        log::info!("   Graph Optimization: {:?}", opt_level);

        let mut providers_to_try = Vec::new();
        if let Some(preferred) = &requirements.preferred_provider {
            Self::push_unique_provider(&mut providers_to_try, preferred);
        }
        for provider in &requirements.fallback_providers {
            Self::push_unique_provider(&mut providers_to_try, provider);
        }

        if Self::should_append_fastest_candidates(&providers_to_try) {
            log::info!("⚡ Provider policy `fastest`: appending hardware-accelerated providers");
            Self::append_fastest_candidates(device_info, &mut providers_to_try);
        }

        for provider in &providers_to_try {
            let device_id = requirements.device_id.unwrap_or(0);
            match Self::try_create_provider(provider, device_info, opt_level, device_id, tuning) {
                Ok(backend) => {
                    log::info!("✓ Using provider: {}", provider);
                    return finalize_onnx_pipeline(engine_kind, manifest, backend);
                }
                Err(err) => {
                    log::warn!("⚠ Provider '{}' not available: {}", provider, err);
                }
            }
        }

        // Last resort: CPU
        log::info!("⚠ Using last-resort CPU backend");
        let opt_cpu = parse_optimization_level(requirements.graph_optimization_level.as_ref())
            .unwrap_or(GraphOptimizationLevel::Level3);
        let inner = Self::build_onnx_backend(ExecutionProvider::CPU, opt_cpu, 0, tuning)?;
        finalize_onnx_pipeline(engine_kind, manifest, inner)
    }

    /// Create a backend for a specific device
    pub fn create_backend_for_device(
        manifest: &Manifest,
        provider: &str,
        device_id: usize,
        device_info: &DeviceInfo,
    ) -> Result<Box<dyn Engine>, String> {
        Self::create_backend_for_device_with_tuning(
            manifest,
            provider,
            device_id,
            device_info,
            &OnnxRuntimeTuning::default(),
        )
    }

    pub fn create_backend_for_device_with_tuning(
        manifest: &Manifest,
        provider: &str,
        device_id: usize,
        device_info: &DeviceInfo,
        tuning: &OnnxRuntimeTuning,
    ) -> Result<Box<dyn Engine>, String> {
        let engine_kind = EngineKind::resolve(manifest);

        // Native safetensors: route to custom CUDA kernel backend
        #[cfg(feature = "native")]
        if engine_kind == EngineKind::Native {
            log::info!("✓ Using NativeBackend (custom CUDA kernels) on device {}", device_id);
            return NativeBackend::new(device_id as i32)
                .map(|b| Box::new(b) as Box<dyn Engine>)
                .map_err(|e| format!("NativeBackend init failed: {e}"));
        }

        // GGUF: prefer native CUDA kernels when gguf-native feature is compiled in.
        #[cfg(feature = "gguf-native")]
        if engine_kind.is_gguf() {
            log::info!("✓ Using GgufNativeBackend (GGUF loader + native CUDA), device {}", device_id);
            return crate::gguf_native::GgufNativeBackend::new(device_id as i32)
                .map(|b| Box::new(b) as Box<dyn Engine>)
                .map_err(|e| format!("GgufNativeBackend init failed: {e}"));
        }

        #[cfg(all(feature = "gguf-cuda-shared-kv", not(feature = "gguf-native")))]
        if engine_kind.is_gguf() {
            log::info!(
                "✓ Using GgufBackend (llama.cpp CUDA + Kapsl shared KV) on device {}",
                device_id
            );
            return Ok(Box::new(GgufBackend::new_cuda_shared_kv(device_id, None)));
        }

        // GGUF: fallback to llama.cpp-backed GgufBackend.
        #[cfg(not(any(feature = "gguf-native", feature = "gguf-cuda-shared-kv")))]
        if engine_kind.is_gguf() {
            log::info!("✓ Using GgufBackend (llama.cpp)");
            return Ok(Box::new(GgufBackend::new()));
        }

        // ONNX generative path (LLMBackend): autoregressive decode over an ONNX graph.
        if engine_kind.is_onnx_generate() {
            if Self::provider_policy() == "manifest" {
                log::info!(
                    "✓ Using LLMBackend with manifest provider override: {}",
                    provider
                );
                return Ok(Box::new(LLMBackend::with_device(
                    provider.to_string(),
                    device_id as i32,
                )));
            }

            log::info!(
                "✓ Using LLMBackend with device pinning and runtime provider auto-selection"
            );
            return Ok(Box::new(LLMBackend::with_device_id(device_id as i32)));
        }

        if !engine_kind.is_implemented() {
            return Err(unimplemented_engine_error(engine_kind));
        }

        let requirements = &manifest.hardware_requirements;
        let opt_level = parse_optimization_level(requirements.graph_optimization_level.as_ref())
            .map_err(|e| format!("Invalid graph optimization level in manifest: {}", e))?;
        let inner =
            Self::try_create_provider(provider, device_info, opt_level, device_id as i32, tuning)?;
        finalize_onnx_pipeline(engine_kind, manifest, inner)
    }

    fn try_create_provider(
        provider: &str,
        device_info: &DeviceInfo,
        opt_level: GraphOptimizationLevel,
        device_id: i32,
        tuning: &OnnxRuntimeTuning,
    ) -> Result<Box<dyn Engine>, String> {
        let provider_lower = provider.to_lowercase();

        match provider_lower.as_str() {
            "cuda" => {
                if !device_info.has_cuda {
                    return Err("CUDA not available on this system".to_string());
                }
                if !CUDAExecutionProvider::default()
                    .is_available()
                    .unwrap_or(false)
                {
                    return Err(
                        "CUDA execution provider is not available in ONNX Runtime".to_string()
                    );
                }
                let cuda_version = device_info
                    .devices
                    .iter()
                    .find(|d| matches!(d.backend, kapsl_hal::device::DeviceBackend::Cuda))
                    .and_then(|d| d.cuda_version.as_ref())
                    .map(|s| s.as_str())
                    .unwrap_or("unknown");
                let acceptance = resolve_onnx_accelerator_provider(
                    OnnxAcceleratorProvider::Cuda,
                    device_info,
                    device_id,
                )?;
                log::info!(
                    "   CUDA accepted: version {:?}, bundle={}, device={}",
                    cuda_version,
                    acceptance.bundle,
                    acceptance.device_summary
                );
                Self::build_onnx_backend(ExecutionProvider::CUDA, opt_level, device_id, tuning)
            }

            "tensorrt" => {
                if !device_info.has_cuda {
                    return Err("TensorRT requires CUDA-capable GPU".to_string());
                }
                if !TensorRTExecutionProvider::default()
                    .is_available()
                    .unwrap_or(false)
                {
                    return Err(
                        "TensorRT execution provider is not available in ONNX Runtime".to_string(),
                    );
                }
                let acceptance = resolve_onnx_accelerator_provider(
                    OnnxAcceleratorProvider::TensorRt,
                    device_info,
                    device_id,
                )?;
                log::info!(
                    "   TensorRT accepted: bundle={}, device={}",
                    acceptance.bundle,
                    acceptance.device_summary
                );
                Self::build_onnx_backend(ExecutionProvider::TensorRT, opt_level, device_id, tuning)
            }

            "metal" | "coreml" => {
                if !device_info.has_metal {
                    return Err(format!(
                        "{} not available on this system",
                        if provider_lower == "metal" {
                            "Metal"
                        } else {
                            "CoreML"
                        }
                    ));
                }
                if !CoreMLExecutionProvider::default()
                    .is_available()
                    .unwrap_or(false)
                {
                    return Err("CoreML execution provider is not available".to_string());
                }
                if provider_lower == "metal" {
                    log::info!("   Metal available on macOS");
                    log::info!("   Using CoreML execution provider for Metal");
                } else {
                    log::info!("   CoreML available on macOS");
                }
                // CoreML performs best with basic optimization; aggressive levels
                // can cause layout issues and runtime errors on Apple Silicon.
                let coreml_opt_level = match opt_level {
                    GraphOptimizationLevel::Level2 | GraphOptimizationLevel::Level3 => {
                        log::info!("   Capping optimization level to Level1 for CoreML backend");
                        GraphOptimizationLevel::Level1
                    }
                    other => other,
                };
                Self::build_onnx_backend(
                    ExecutionProvider::CoreML,
                    coreml_opt_level,
                    device_id,
                    tuning,
                )
            }
            "rocm" => {
                if !device_info.has_rocm {
                    return Err("ROCm not available on this system".to_string());
                }
                if !ROCmExecutionProvider::default()
                    .is_available()
                    .unwrap_or(false)
                {
                    return Err("ROCm execution provider is not available".to_string());
                }
                log::info!("   ROCm available");
                Self::build_onnx_backend(ExecutionProvider::ROCm, opt_level, device_id, tuning)
            }
            "directml" => {
                #[cfg(target_os = "windows")]
                {
                    if !device_info.has_directml {
                        return Err("DirectML not available on this system".to_string());
                    }
                    if !DirectMLExecutionProvider::default()
                        .is_available()
                        .unwrap_or(false)
                    {
                        return Err("DirectML execution provider is not available".to_string());
                    }
                    log::info!("   DirectML available");
                    Self::build_onnx_backend(
                        ExecutionProvider::DirectML,
                        opt_level,
                        device_id,
                        tuning,
                    )
                }
                #[cfg(not(target_os = "windows"))]
                {
                    Err("DirectML is only supported on Windows".to_string())
                }
            }
            "openvino" => {
                if !OpenVINOExecutionProvider::default()
                    .is_available()
                    .unwrap_or(false)
                {
                    return Err("OpenVINO execution provider is not available".to_string());
                }
                log::info!("   OpenVINO available");
                Self::build_onnx_backend(ExecutionProvider::OpenVINO, opt_level, device_id, tuning)
            }

            "cpu" => {
                log::info!("   Using CPU execution");
                Self::build_onnx_backend(ExecutionProvider::CPU, opt_level, 0, tuning)
            }

            _ => Err(format!("Unknown provider: {}", provider)),
        }
    }

    /// Create a GgufNativeBackend with an optional pre-shared pool handle.
    ///
    /// Returns the concrete (unboxed) backend so callers can call
    /// `backend.pool_handle()` after `load()` to register the pool for sharing.
    #[cfg(feature = "gguf-native")]
    pub fn create_gguf_native(
        device_id: i32,
        handle: Option<GpuPoolHandle>,
    ) -> Result<GgufNativeBackend, String> {
        let mut b = GgufNativeBackend::new(device_id)
            .map_err(|e| format!("GgufNativeBackend init failed: {e}"))?;
        if let Some(h) = handle {
            b = b.with_pool_handle(h);
        }
        Ok(b)
    }

    /// Create a llama.cpp CUDA backend backed by the Kapsl shared KV pool.
    #[cfg(feature = "gguf-cuda-shared-kv")]
    pub fn create_gguf_cuda_shared_kv(
        device_id: i32,
        handle: Option<GpuPoolHandle>,
    ) -> Result<GgufBackend, String> {
        Ok(GgufBackend::new_cuda_shared_kv(device_id.max(0) as usize, handle))
    }

    /// Create a NativeBackend with an optional pre-shared pool handle.
    ///
    /// Returns the concrete (unboxed) backend so callers can call
    /// `backend.pool_handle()` after `load()` to register the pool for sharing.
    #[cfg(feature = "native")]
    pub fn create_native(
        device_id: i32,
        handle: Option<GpuPoolHandle>,
    ) -> Result<NativeBackend, String> {
        let mut b = NativeBackend::new(device_id)
            .map_err(|e| format!("NativeBackend init failed: {e}"))?;
        if let Some(h) = handle {
            b = b.with_pool_handle(h);
        }
        Ok(b)
    }

    /// Validate that hardware meets minimum requirements
    pub fn validate_requirements(
        requirements: &HardwareRequirements,
        device_info: &DeviceInfo,
    ) -> Result<(), String> {
        // Validation logic for CPU memory
        if let Some(min_mem_mb) = requirements.min_memory_mb {
            let available_mb = device_info.total_memory / (1024 * 1024);
            if available_mb < min_mem_mb {
                return Err(format!(
                    "Insufficient memory: need {}MB, have {}MB",
                    min_mem_mb, available_mb
                ));
            }
        }

        // Collect all providers to check (preferred + fallbacks)
        let mut providers_to_check = Vec::new();
        if let Some(preferred) = &requirements.preferred_provider {
            providers_to_check.push(preferred.clone());
        }
        providers_to_check.extend(requirements.fallback_providers.clone());

        // We only fail if NONE of the providers are valid/present
        let mut reasons = Vec::new();
        let mut has_valid_provider = false;

        let strategy = requirements
            .strategy
            .as_deref()
            .unwrap_or("")
            .to_ascii_lowercase();
        let allow_multi = matches!(
            strategy.as_str(),
            "pool"
                | "round-robin"
                | "data-parallel"
                | "pipeline"
                | "pipeline-parallel"
                | "tensor-parallel"
                | "auto"
        );

        for provider in &providers_to_check {
            let provider_lower = provider.to_lowercase();
            let backend_key = match provider_lower.as_str() {
                "tensorrt" => "cuda",
                "coreml" => "metal",
                other => other,
            };
            let is_cpu = backend_key == "cpu";

            if is_cpu {
                // CPU is always valid if memory check passed (which is global above, though strictly
                // memory check should maybe be per-provider if requirements differed, but here it's global)
                has_valid_provider = true;
                break;
            }

            // GPU checks
            if !device_info.has_provider(backend_key) {
                reasons.push(format!("Provider {} not available", provider));
                continue;
            }

            let device_meets = |device: &kapsl_hal::device::Device| -> bool {
                if backend_key != "cpu" {
                    if let Some(min_vram) = requirements.min_vram_mb {
                        if device.memory_mb < min_vram {
                            return false;
                        }
                    }
                    if backend_key == "cuda" {
                        if let Some(min_ver) = &requirements.min_cuda_version {
                            if let Some(dev_ver) = &device.cuda_version {
                                if dev_ver < min_ver {
                                    return false;
                                }
                            } else {
                                return false;
                            }
                        }
                    }
                }
                true
            };

            if allow_multi {
                let mut candidates = device_info
                    .devices
                    .iter()
                    .filter(|d| d.backend.to_string().to_lowercase() == backend_key);

                if candidates.any(device_meets) {
                    has_valid_provider = true;
                    break;
                }

                reasons.push(format!(
                    "No devices meet requirements for provider {}",
                    provider
                ));
                continue;
            }

            // Find the device
            let dev_id = requirements.device_id.unwrap_or(0) as usize;
            // Note: device_id 0 is usually the first GPU if provider is GPU.
            if let Some(device) = device_info
                .devices
                .iter()
                .find(|d| d.id == dev_id && d.backend.to_string().to_lowercase() == backend_key)
            {
                // Check VRAM
                if let Some(min_vram) = requirements.min_vram_mb {
                    if device.memory_mb < min_vram {
                        reasons.push(format!(
                            "Provider {} (Device {}) has insufficient VRAM: {}MB < required {}MB",
                            provider, dev_id, device.memory_mb, min_vram
                        ));
                        continue;
                    }
                }

                // Check CUDA version
                if backend_key == "cuda" {
                    if let Some(min_ver) = &requirements.min_cuda_version {
                        if let Some(dev_ver) = &device.cuda_version {
                            if dev_ver < min_ver {
                                reasons.push(format!(
                                    "CUDA version too old: {} < required {}",
                                    dev_ver, min_ver
                                ));
                                continue;
                            }
                        } else {
                            reasons.push("Unknown CUDA version on device".to_string());
                            continue;
                        }
                    }
                }

                has_valid_provider = true;
                break;
            } else {
                reasons.push(format!(
                    "Device ID {} not found for provider {}",
                    dev_id, provider
                ));
            }
        }

        if !has_valid_provider {
            if providers_to_check.is_empty() {
                // No requirements?
                return Ok(());
            }
            // If we have CPU in list and it wasn't caught above, it means something weird happened.
            // But usually CPU works.
            return Err(format!(
                "No compatible provider found. Reasons: {:?}",
                reasons
            ));
        }

        Ok(())
    }
}

#[path = "factory_tests.rs"]
mod factory_tests;
