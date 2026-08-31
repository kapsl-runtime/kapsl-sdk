//! Unit tests for the audio log-mel preprocessing stage.

use super::audio::{
    AudioConfig, AudioLayout, AudioPreprocessor, FeatureNormalization, LogKind, MelScale,
};
use super::Preprocessor;
use kapsl_engine_api::{BinaryTensorPacket, TensorDtype};

fn f32_packet(samples: &[f32]) -> BinaryTensorPacket {
    let data: Vec<u8> = samples.iter().flat_map(|v| v.to_le_bytes()).collect();
    BinaryTensorPacket::new(vec![samples.len() as i64], TensorDtype::Float32, data).unwrap()
}

fn out_f32(p: &BinaryTensorPacket) -> Vec<f32> {
    p.data
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect()
}

fn sine(freq: f32, sr: u32, n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| (2.0 * std::f32::consts::PI * freq * i as f32 / sr as f32).sin())
        .collect()
}

/// Small non-centered config with predictable framing.
fn small_cfg() -> AudioConfig {
    AudioConfig {
        n_fft: 8,
        hop_length: 4,
        n_mels: 4,
        center: false,
        log: LogKind::None,
        ..Default::default()
    }
}

#[test]
fn frame_count_and_shape_no_center() {
    let pre = AudioPreprocessor::new(small_cfg()).unwrap();
    // 16 samples, n_fft 8, hop 4 -> 1 + (16-8)/4 = 3 frames.
    let out = pre.apply(&f32_packet(&[0.0; 16])).unwrap();
    assert_eq!(out.shape, vec![1, 4, 3]);
}

#[test]
fn center_padding_adds_frames() {
    let mut cfg = small_cfg();
    cfg.center = true;
    let pre = AudioPreprocessor::new(cfg).unwrap();
    // pad 4 each side -> signal 24 -> 1 + (24-8)/4 = 5 frames.
    let out = pre.apply(&f32_packet(&[0.1; 16])).unwrap();
    assert_eq!(out.shape, vec![1, 4, 5]);
}

#[test]
fn silence_is_zero_energy_without_log() {
    let pre = AudioPreprocessor::new(small_cfg()).unwrap();
    let out = pre.apply(&f32_packet(&[0.0; 32])).unwrap();
    assert!(out_f32(&out).iter().all(|&v| v.abs() < 1e-9));
}

#[test]
fn time_mel_layout_transposes_shape() {
    let mut cfg = small_cfg();
    cfg.layout = AudioLayout::TimeMel;
    let pre = AudioPreprocessor::new(cfg).unwrap();
    let out = pre.apply(&f32_packet(&[0.0; 16])).unwrap();
    // [1, n_frames, n_mels] = [1, 3, 4]
    assert_eq!(out.shape, vec![1, 3, 4]);
}

/// A higher-frequency tone should peak in a higher mel bin than a low tone —
/// validates FFT + mel mapping direction without brittle exact-bin math.
#[test]
fn tone_frequency_maps_to_expected_mel_bin_order() {
    let sr = 16000;
    let cfg = AudioConfig {
        sample_rate: sr,
        n_fft: 512,
        hop_length: 256,
        n_mels: 40,
        center: false,
        log: LogKind::None,
        mel_scale: MelScale::Htk,
        ..Default::default()
    };
    let pre = AudioPreprocessor::new(cfg).unwrap();

    let peak_bin = |freq: f32| -> usize {
        let out = pre.apply(&f32_packet(&sine(freq, sr, 4096))).unwrap();
        let n_mels = 40usize;
        let n_frames = out.shape[2] as usize;
        let v = out_f32(&out);
        // Average energy per mel bin across frames, then argmax.
        let mut best = 0usize;
        let mut best_e = f32::NEG_INFINITY;
        for m in 0..n_mels {
            let mut acc = 0f32;
            for f in 0..n_frames {
                acc += v[m * n_frames + f];
            }
            if acc > best_e {
                best_e = acc;
                best = m;
            }
        }
        best
    };

    let low = peak_bin(500.0);
    let high = peak_bin(4000.0);
    assert!(
        low < high,
        "500Hz peaked at mel {low}, 4000Hz at {high}; expected low < high"
    );
}

#[test]
fn rejects_non_f32_input() {
    let pre = AudioPreprocessor::new(small_cfg()).unwrap();
    let bad = BinaryTensorPacket::new(vec![1], TensorDtype::Uint8, vec![0]).unwrap();
    assert!(pre.apply(&bad).is_err());
}

#[test]
fn rejects_empty_waveform() {
    let pre = AudioPreprocessor::new(small_cfg()).unwrap();
    // Construct directly: BinaryTensorPacket::new rejects an empty payload at the
    // constructor, so bypass it to exercise the preprocessor's own guard.
    let empty = BinaryTensorPacket {
        shape: vec![0],
        dtype: TensorDtype::Float32,
        data: vec![],
    };
    assert!(pre.apply(&empty).is_err());
}

#[test]
fn zero_config_rejected() {
    let cfg = AudioConfig {
        n_fft: 0,
        ..Default::default()
    };
    assert!(AudioPreprocessor::new(cfg).is_err());
}

#[test]
fn log10_compression_applied_to_silence() {
    // With log10 and default eps 1e-10, silence -> log10(1e-10) = -10.
    let mut cfg = small_cfg();
    cfg.log = LogKind::Log10;
    let pre = AudioPreprocessor::new(cfg).unwrap();
    let out = pre.apply(&f32_packet(&[0.0; 16])).unwrap();
    assert!(out_f32(&out).iter().all(|&v| (v + 10.0).abs() < 1e-3));
}

#[test]
fn per_feature_normalization_matches_nemo_formula() {
    let samples: Vec<f32> = (0..512)
        .map(|i| {
            let x = i as f32;
            (0.071 * x).sin() + 0.35 * (0.0009 * x * x).cos()
        })
        .collect();
    let base = AudioConfig {
        n_fft: 64,
        hop_length: 16,
        n_mels: 8,
        center: false,
        ..Default::default()
    };

    let raw = AudioPreprocessor::new(base.clone())
        .unwrap()
        .apply(&f32_packet(&samples))
        .unwrap();
    let normalized = AudioPreprocessor::new(AudioConfig {
        normalize: FeatureNormalization::PerFeature,
        ..base
    })
    .unwrap()
    .apply(&f32_packet(&samples))
    .unwrap();

    assert_eq!(normalized.shape, raw.shape);
    let n_frames = raw.shape[2] as usize;
    let raw_values = out_f32(&raw);
    let normalized_values = out_f32(&normalized);
    for mel in 0..8 {
        let row = &raw_values[mel * n_frames..(mel + 1) * n_frames];
        let mean = row.iter().copied().sum::<f32>() / n_frames as f32;
        let sample_variance = row
            .iter()
            .map(|&value| {
                let centered = value - mean;
                centered * centered
            })
            .sum::<f32>()
            / (n_frames - 1) as f32;
        let denominator = sample_variance.sqrt() + 1e-5;

        for frame in 0..n_frames {
            let expected = (row[frame] - mean) / denominator;
            let actual = normalized_values[mel * n_frames + frame];
            assert!(
                (actual - expected).abs() < 1e-5,
                "mel {mel}, frame {frame}: expected {expected}, got {actual}"
            );
        }
    }
}

#[test]
fn per_feature_normalization_handles_one_frame() {
    let cfg = AudioConfig {
        n_fft: 16,
        hop_length: 8,
        n_mels: 4,
        center: false,
        normalize: FeatureNormalization::PerFeature,
        ..Default::default()
    };
    let pre = AudioPreprocessor::new(cfg).unwrap();
    let out = pre.apply(&f32_packet(&sine(440.0, 16000, 16))).unwrap();

    assert_eq!(out.shape, vec![1, 4, 1]);
    assert!(out_f32(&out).iter().all(|&value| value == 0.0));
}

#[test]
fn derives_feature_length_from_emitted_time_axis() {
    let cfg = AudioConfig {
        layout: AudioLayout::TimeMel,
        length_input: Some("length".to_string()),
        ..small_cfg()
    };
    let pre = AudioPreprocessor::new(cfg).unwrap();
    let output = pre.apply(&f32_packet(&[0.0; 16])).unwrap();
    let derived = pre.derived_inputs(&output).unwrap();

    assert_eq!(output.shape, vec![1, 3, 4]);
    assert_eq!(derived.len(), 1);
    assert_eq!(derived[0].name, "length");
    assert_eq!(derived[0].tensor.shape, vec![1]);
    assert_eq!(derived[0].tensor.dtype, TensorDtype::Int64);
    assert_eq!(
        i64::from_le_bytes(derived[0].tensor.data.as_slice().try_into().unwrap()),
        3
    );
}

#[test]
fn audio_config_accepts_nemo_normalize_type_alias() {
    let cfg: AudioConfig = serde_yaml::from_str(
        "normalize_type: per_feature\nlength_input: length\nlength_dtype: int32\n",
    )
    .unwrap();

    assert_eq!(cfg.normalize, FeatureNormalization::PerFeature);
    assert_eq!(cfg.length_input.as_deref(), Some("length"));
    assert_eq!(cfg.length_dtype, TensorDtype::Int32);
}

#[test]
fn invalid_normalization_and_length_settings_are_rejected() {
    assert!(AudioPreprocessor::new(AudioConfig {
        normalize_eps: 0.0,
        ..Default::default()
    })
    .is_err());
    assert!(AudioPreprocessor::new(AudioConfig {
        length_input: Some(" ".to_string()),
        ..Default::default()
    })
    .is_err());
    assert!(AudioPreprocessor::new(AudioConfig {
        length_dtype: TensorDtype::Float32,
        ..Default::default()
    })
    .is_err());
}
