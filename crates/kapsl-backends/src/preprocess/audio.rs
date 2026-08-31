//! Audio input preprocessing: PCM waveform -> log-mel spectrogram.
//!
//! The acoustic front-end for the CTC ASR cell ([`crate::onnx_transcribe`]) and
//! any ONNX audio model that consumes mel features. The client sends a `Float32`
//! packet of mono PCM samples (already resampled to `sample_rate`); this stage
//! frames the signal, applies a windowed FFT, projects the power spectrum onto a
//! mel filterbank, and takes the log — the standard `[1, n_mels, n_frames]`
//! feature tensor.
//!
//! Decoding compressed containers (WAV/FLAC/MP3) is out of scope for this slice
//! to avoid a heavy codec dependency: callers pass raw f32 samples, matching the
//! `raw_speech` input contract of Hugging Face feature extractors.
//!
//! Configuration comes from the manifest's `metadata.preprocess` block, e.g.
//! (a typical 80-bin 16 kHz setup):
//!
//! ```yaml
//! metadata:
//!   preprocess:
//!     kind: audio
//!     sample_rate: 16000
//!     n_fft: 400
//!     hop_length: 160
//!     n_mels: 80
//!     f_min: 0.0
//!     f_max: 8000.0        # defaults to sample_rate / 2
//!     mel_scale: htk       # or `slaney` (librosa default)
//!     norm: none           # or `slaney` (area-normalize filters)
//!     log: log10           # `none` | `ln` | `log10`
//!     power: 2.0           # 1.0 magnitude, 2.0 power
//!     center: true         # reflect-pad n_fft/2 each side (librosa default)
//!     normalize: per_feature # `none` (default) or mean/std over time per mel bin
//!     normalize_eps: 1.0e-5  # added to each feature standard deviation
//!     layout: mel_time     # [n_mels, n_frames]; or `time_mel`
//!     length_input: length # optional derived [1] frame-count model input
//!     length_dtype: int64  # `int64` (default) or `int32`
//! ```

use kapsl_engine_api::{BinaryTensorPacket, EngineError, NamedTensor, TensorDtype};
use rustfft::{num_complex::Complex, Fft, FftPlanner};
use serde::Deserialize;
use std::sync::Arc;

use super::Preprocessor;

/// Mel warping curve used to space the filterbank centers.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MelScale {
    /// `2595 * log10(1 + hz/700)`.
    #[default]
    Htk,
    /// Linear below 1 kHz, log above (librosa/torchaudio default).
    Slaney,
}

/// Filter normalization.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MelNorm {
    /// Unit-height triangular filters.
    #[default]
    None,
    /// Area-normalized (`2 / (right - left)`), as librosa `norm="slaney"`.
    Slaney,
}

/// Log compression applied to the mel energies.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum LogKind {
    /// Leave mel energies linear.
    None,
    /// Natural log.
    Ln,
    /// Base-10 log.
    #[default]
    Log10,
}

/// Layout of the emitted feature tensor.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum AudioLayout {
    /// `[1, n_mels, n_frames]`.
    #[default]
    MelTime,
    /// `[1, n_frames, n_mels]`.
    TimeMel,
}

/// Normalization applied to the completed mel spectrogram.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FeatureNormalization {
    /// Leave features on their original scale.
    #[default]
    None,
    /// Normalize every mel bin independently over the time axis.
    PerFeature,
}

fn default_sample_rate() -> u32 {
    16000
}
fn default_n_fft() -> usize {
    400
}
fn default_hop_length() -> usize {
    160
}
fn default_n_mels() -> usize {
    80
}
fn default_power() -> f32 {
    2.0
}
fn default_center() -> bool {
    true
}
fn default_log_eps() -> f32 {
    1e-10
}
fn default_normalize_eps() -> f32 {
    // Matches NeMo's CONSTANT used by normalize_batch.
    1e-5
}
fn default_length_dtype() -> TensorDtype {
    TensorDtype::Int64
}

/// Parsed `metadata.preprocess` block for an audio model.
#[derive(Debug, Clone, Deserialize)]
pub struct AudioConfig {
    #[serde(default = "default_sample_rate")]
    pub sample_rate: u32,
    #[serde(default = "default_n_fft")]
    pub n_fft: usize,
    #[serde(default = "default_hop_length")]
    pub hop_length: usize,
    #[serde(default = "default_n_mels")]
    pub n_mels: usize,
    #[serde(default)]
    pub f_min: f32,
    /// Upper edge of the mel filters. Defaults to `sample_rate / 2`.
    #[serde(default)]
    pub f_max: Option<f32>,
    #[serde(default)]
    pub mel_scale: MelScale,
    #[serde(default)]
    pub norm: MelNorm,
    #[serde(default)]
    pub log: LogKind,
    #[serde(default = "default_power")]
    pub power: f32,
    #[serde(default = "default_center")]
    pub center: bool,
    #[serde(default)]
    pub layout: AudioLayout,
    /// Floor added before a log to avoid `log(0)`.
    #[serde(default = "default_log_eps")]
    pub log_eps: f32,
    /// Feature normalization after log compression. `normalize_type` is
    /// accepted as an alias for model metadata that uses NeMo's field name.
    #[serde(default, alias = "normalize_type")]
    pub normalize: FeatureNormalization,
    /// Value added to each standard deviation during normalization.
    #[serde(default = "default_normalize_eps")]
    pub normalize_eps: f32,
    /// Optional named model input that receives the emitted feature-frame
    /// count as a one-element tensor.
    #[serde(default)]
    pub length_input: Option<String>,
    /// Integer dtype for `length_input`.
    #[serde(default = "default_length_dtype")]
    pub length_dtype: TensorDtype,
}

impl Default for AudioConfig {
    fn default() -> Self {
        Self {
            sample_rate: default_sample_rate(),
            n_fft: default_n_fft(),
            hop_length: default_hop_length(),
            n_mels: default_n_mels(),
            f_min: 0.0,
            f_max: None,
            mel_scale: MelScale::default(),
            norm: MelNorm::default(),
            log: LogKind::default(),
            power: default_power(),
            center: default_center(),
            layout: AudioLayout::default(),
            log_eps: default_log_eps(),
            normalize: FeatureNormalization::default(),
            normalize_eps: default_normalize_eps(),
            length_input: None,
            length_dtype: default_length_dtype(),
        }
    }
}

impl AudioConfig {
    fn validate(&self) -> Result<(), EngineError> {
        if self.n_fft == 0 || self.hop_length == 0 || self.n_mels == 0 {
            return Err(EngineError::backend(
                "audio preprocess: n_fft, hop_length, and n_mels must be non-zero",
            ));
        }
        if self.sample_rate == 0 {
            return Err(EngineError::backend(
                "audio preprocess: sample_rate must be non-zero",
            ));
        }
        if !self.normalize_eps.is_finite() || self.normalize_eps <= 0.0 {
            return Err(EngineError::backend(
                "audio preprocess: normalize_eps must be finite and greater than zero",
            ));
        }
        if self
            .length_input
            .as_ref()
            .is_some_and(|name| name.trim().is_empty())
        {
            return Err(EngineError::backend(
                "audio preprocess: length_input must not be empty",
            ));
        }
        if !matches!(self.length_dtype, TensorDtype::Int32 | TensorDtype::Int64) {
            return Err(EngineError::backend(
                "audio preprocess: length_dtype must be int32 or int64",
            ));
        }
        Ok(())
    }

    fn f_max(&self) -> f32 {
        self.f_max.unwrap_or(self.sample_rate as f32 / 2.0)
    }
}

fn hz_to_mel(hz: f32, scale: MelScale) -> f32 {
    match scale {
        MelScale::Htk => 2595.0 * (1.0 + hz / 700.0).log10(),
        MelScale::Slaney => {
            let f_sp = 200.0 / 3.0;
            let min_log_hz = 1000.0;
            let min_log_mel = min_log_hz / f_sp;
            let logstep = (6.4f32).ln() / 27.0;
            if hz < min_log_hz {
                hz / f_sp
            } else {
                min_log_mel + (hz / min_log_hz).ln() / logstep
            }
        }
    }
}

fn mel_to_hz(mel: f32, scale: MelScale) -> f32 {
    match scale {
        MelScale::Htk => 700.0 * (10f32.powf(mel / 2595.0) - 1.0),
        MelScale::Slaney => {
            let f_sp = 200.0 / 3.0;
            let min_log_hz = 1000.0;
            let min_log_mel = min_log_hz / f_sp;
            let logstep = (6.4f32).ln() / 27.0;
            if mel < min_log_mel {
                mel * f_sp
            } else {
                min_log_hz * (logstep * (mel - min_log_mel)).exp()
            }
        }
    }
}

/// Periodic Hann window of length `n` (matches torch/librosa STFT default).
fn hann_window(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| 0.5 - 0.5 * (2.0 * std::f32::consts::PI * i as f32 / n as f32).cos())
        .collect()
}

/// Triangular mel filterbank, flat `[n_mels * n_freqs]` row-major.
fn mel_filterbank(cfg: &AudioConfig, n_freqs: usize) -> Vec<f32> {
    let mel_min = hz_to_mel(cfg.f_min, cfg.mel_scale);
    let mel_max = hz_to_mel(cfg.f_max(), cfg.mel_scale);
    // n_mels + 2 band edges in mel space, mapped back to Hz.
    let edges: Vec<f32> = (0..cfg.n_mels + 2)
        .map(|i| {
            let mel = mel_min + (mel_max - mel_min) * i as f32 / (cfg.n_mels + 1) as f32;
            mel_to_hz(mel, cfg.mel_scale)
        })
        .collect();

    let bin_hz = |k: usize| k as f32 * cfg.sample_rate as f32 / cfg.n_fft as f32;

    let mut fb = vec![0f32; cfg.n_mels * n_freqs];
    for m in 0..cfg.n_mels {
        let (left, center, right) = (edges[m], edges[m + 1], edges[m + 2]);
        let ld = (center - left).max(f32::EPSILON);
        let rd = (right - center).max(f32::EPSILON);
        for k in 0..n_freqs {
            let f = bin_hz(k);
            let w = if f >= left && f <= center {
                (f - left) / ld
            } else if f > center && f <= right {
                (right - f) / rd
            } else {
                0.0
            };
            fb[m * n_freqs + k] = w;
        }
        if cfg.norm == MelNorm::Slaney {
            let enorm = 2.0 / (right - left).max(f32::EPSILON);
            for k in 0..n_freqs {
                fb[m * n_freqs + k] *= enorm;
            }
        }
    }
    fb
}

/// Decodes PCM samples into a log-mel spectrogram.
pub struct AudioPreprocessor {
    cfg: AudioConfig,
    window: Vec<f32>,
    /// `[n_mels * n_freqs]`, `n_freqs = n_fft/2 + 1`.
    mel_fb: Vec<f32>,
    n_freqs: usize,
    fft: Arc<dyn Fft<f32>>,
}

impl AudioPreprocessor {
    pub fn new(cfg: AudioConfig) -> Result<Self, EngineError> {
        cfg.validate()?;
        let n_freqs = cfg.n_fft / 2 + 1;
        let window = hann_window(cfg.n_fft);
        let mel_fb = mel_filterbank(&cfg, n_freqs);
        let fft = FftPlanner::new().plan_fft_forward(cfg.n_fft);
        Ok(Self {
            cfg,
            window,
            mel_fb,
            n_freqs,
            fft,
        })
    }

    fn compress(&self, e: f32) -> f32 {
        match self.cfg.log {
            LogKind::None => e,
            LogKind::Ln => (e + self.cfg.log_eps).ln(),
            LogKind::Log10 => (e + self.cfg.log_eps).log10(),
        }
    }

    /// NeMo-compatible per-feature normalization: for each mel bin, subtract
    /// its mean over time and divide by the bias-corrected sample standard
    /// deviation plus epsilon.
    fn normalize(&self, mels: &mut [f32], n_frames: usize) {
        if self.cfg.normalize != FeatureNormalization::PerFeature || n_frames == 0 {
            return;
        }

        for row in mels.chunks_exact_mut(n_frames) {
            let mean = row.iter().copied().sum::<f32>() / n_frames as f32;
            let variance = if n_frames > 1 {
                row.iter()
                    .map(|&value| {
                        let centered = value - mean;
                        centered * centered
                    })
                    .sum::<f32>()
                    / (n_frames - 1) as f32
            } else {
                0.0
            };
            let denominator = variance.sqrt() + self.cfg.normalize_eps;
            for value in row {
                *value = (*value - mean) / denominator;
            }
        }
    }

    fn frame_count(&self, output: &BinaryTensorPacket) -> Result<i64, EngineError> {
        let time_axis = match self.cfg.layout {
            AudioLayout::MelTime => 2,
            AudioLayout::TimeMel => 1,
        };
        output.shape.get(time_axis).copied().ok_or_else(|| {
            EngineError::backend(format!(
                "audio preprocess: cannot derive length from output shape {:?}",
                output.shape
            ))
        })
    }

    /// Reflect-pad `n_fft/2` samples on each side (librosa `center=True`).
    fn maybe_center(&self, samples: &[f32]) -> Vec<f32> {
        if !self.cfg.center {
            return samples.to_vec();
        }
        let pad = self.cfg.n_fft / 2;
        let n = samples.len();
        let mut out = Vec::with_capacity(n + 2 * pad);
        // Reflect without repeating the edge sample (numpy 'reflect').
        for i in 0..pad {
            let idx = reflect_index((pad - i) as isize, n);
            out.push(samples[idx]);
        }
        out.extend_from_slice(samples);
        for i in 0..pad {
            let idx = reflect_index(n as isize - 2 - i as isize, n);
            out.push(samples[idx]);
        }
        out
    }
}

/// Clamp an index into `[0, n)` by reflection (for short signals).
fn reflect_index(i: isize, n: usize) -> usize {
    if n == 1 {
        return 0;
    }
    let mut idx = i;
    let m = n as isize;
    // Fold repeatedly for signals shorter than the pad.
    while idx < 0 || idx >= m {
        if idx < 0 {
            idx = -idx;
        }
        if idx >= m {
            idx = 2 * (m - 1) - idx;
        }
    }
    idx as usize
}

impl Preprocessor for AudioPreprocessor {
    fn apply(&self, input: &BinaryTensorPacket) -> Result<BinaryTensorPacket, EngineError> {
        if input.dtype != TensorDtype::Float32 {
            return Err(EngineError::invalid_input(format!(
                "audio preprocess expects Float32 PCM samples, got {:?}",
                input.dtype
            )));
        }
        let samples: Vec<f32> = input
            .data
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect();
        if samples.is_empty() {
            return Err(EngineError::invalid_input(
                "audio preprocess received an empty waveform",
            ));
        }

        let signal = self.maybe_center(&samples);
        let (n_fft, hop) = (self.cfg.n_fft, self.cfg.hop_length);
        let n_frames = if signal.len() >= n_fft {
            1 + (signal.len() - n_fft) / hop
        } else {
            0
        };

        let n_mels = self.cfg.n_mels;
        let mut mels = vec![0f32; n_mels * n_frames];
        let mut buf = vec![Complex::<f32>::new(0.0, 0.0); n_fft];
        for f in 0..n_frames {
            let start = f * hop;
            for i in 0..n_fft {
                buf[i] = Complex::new(signal[start + i] * self.window[i], 0.0);
            }
            self.fft.process(&mut buf);

            // Power (or magnitude) spectrum over the non-redundant bins.
            for m in 0..n_mels {
                let row = &self.mel_fb[m * self.n_freqs..m * self.n_freqs + self.n_freqs];
                let mut acc = 0f32;
                for (k, &wgt) in row.iter().enumerate() {
                    if wgt == 0.0 {
                        continue;
                    }
                    let c = buf[k];
                    let p = if self.cfg.power == 1.0 {
                        (c.re * c.re + c.im * c.im).sqrt()
                    } else {
                        c.re * c.re + c.im * c.im // power = 2
                    };
                    acc += wgt * p;
                }
                mels[m * n_frames + f] = self.compress(acc);
            }
        }

        self.normalize(&mut mels, n_frames);

        // Emit in the requested layout.
        let (shape, data) = match self.cfg.layout {
            AudioLayout::MelTime => {
                let shape = vec![1, n_mels as i64, n_frames as i64];
                (shape, mels)
            }
            AudioLayout::TimeMel => {
                let mut t = vec![0f32; n_mels * n_frames];
                for m in 0..n_mels {
                    for f in 0..n_frames {
                        t[f * n_mels + m] = mels[m * n_frames + f];
                    }
                }
                (vec![1, n_frames as i64, n_mels as i64], t)
            }
        };

        let bytes = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        BinaryTensorPacket::new(shape, TensorDtype::Float32, bytes)
    }

    fn name(&self) -> &'static str {
        "audio"
    }

    fn derived_inputs(&self, output: &BinaryTensorPacket) -> Result<Vec<NamedTensor>, EngineError> {
        let Some(name) = self.cfg.length_input.as_ref() else {
            return Ok(Vec::new());
        };
        let n_frames = self.frame_count(output)?;
        let data = match self.cfg.length_dtype {
            TensorDtype::Int64 => n_frames.to_le_bytes().to_vec(),
            TensorDtype::Int32 => i32::try_from(n_frames)
                .map_err(|_| {
                    EngineError::backend(format!(
                        "audio preprocess: frame count {n_frames} does not fit int32"
                    ))
                })?
                .to_le_bytes()
                .to_vec(),
            _ => unreachable!("length dtype validated during construction"),
        };
        let tensor = BinaryTensorPacket::new(vec![1], self.cfg.length_dtype, data)?;
        Ok(vec![NamedTensor {
            name: name.clone(),
            tensor,
        }])
    }
}
