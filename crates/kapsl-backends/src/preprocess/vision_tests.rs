//! Unit tests for the vision preprocessing stage.

use super::vision::{Layout, ResizeMode, VisionConfig, VisionPreprocessor};
use super::Preprocessor;
use image::{Rgb, RgbImage};
use kapsl_engine_api::{BinaryTensorPacket, TensorDtype};
use std::io::Cursor;

/// Encode a solid-color `w x h` image to PNG bytes.
fn solid_png(w: u32, h: u32, color: [u8; 3]) -> Vec<u8> {
    let img = RgbImage::from_pixel(w, h, Rgb(color));
    let mut buf = Vec::new();
    image::DynamicImage::ImageRgb8(img)
        .write_to(&mut Cursor::new(&mut buf), image::ImageFormat::Png)
        .expect("encode png");
    buf
}

fn uint8_packet(bytes: Vec<u8>) -> BinaryTensorPacket {
    BinaryTensorPacket::new(vec![bytes.len() as i64], TensorDtype::Uint8, bytes).unwrap()
}

fn output_f32(packet: &BinaryTensorPacket) -> Vec<f32> {
    packet
        .data
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

#[test]
fn nchw_solid_color_normalizes_per_channel() {
    let cfg = VisionConfig {
        width: 4,
        height: 4,
        resize: ResizeMode::Stretch,
        layout: Layout::Nchw,
        ..Default::default()
    };
    let pre = VisionPreprocessor::new(cfg).unwrap();

    // Solid red 8x8 -> stretched to 4x4.
    let packet = uint8_packet(solid_png(8, 8, [255, 0, 0]));
    let out = pre.apply(&packet).unwrap();

    assert_eq!(out.dtype, TensorDtype::Float32);
    assert_eq!(out.shape, vec![1, 3, 4, 4]);

    let values = output_f32(&out);
    let plane = 16; // 4 * 4
    // Default scale 1/255, mean 0, std 1: R plane == 1.0, G/B planes == 0.0.
    for &r in &values[0..plane] {
        assert!((r - 1.0).abs() < 1e-6, "R channel expected 1.0, got {r}");
    }
    for &g in &values[plane..2 * plane] {
        assert!(g.abs() < 1e-6, "G channel expected 0.0, got {g}");
    }
    for &b in &values[2 * plane..3 * plane] {
        assert!(b.abs() < 1e-6, "B channel expected 0.0, got {b}");
    }
}

#[test]
fn mean_std_normalization_applied() {
    let cfg = VisionConfig {
        width: 2,
        height: 2,
        mean: [0.5, 0.0, 0.0],
        std: [0.5, 1.0, 1.0],
        ..Default::default()
    };
    let pre = VisionPreprocessor::new(cfg).unwrap();

    let packet = uint8_packet(solid_png(2, 2, [255, 0, 0]));
    let values = output_f32(&pre.apply(&packet).unwrap());
    // R = (255/255 - 0.5) / 0.5 = 1.0
    assert!((values[0] - 1.0).abs() < 1e-6, "got {}", values[0]);
}

#[test]
fn nhwc_layout_is_interleaved() {
    let cfg = VisionConfig {
        width: 1,
        height: 1,
        layout: Layout::Nhwc,
        ..Default::default()
    };
    let pre = VisionPreprocessor::new(cfg).unwrap();

    let packet = uint8_packet(solid_png(1, 1, [255, 128, 0]));
    let out = pre.apply(&packet).unwrap();
    assert_eq!(out.shape, vec![1, 1, 1, 3]);
    let v = output_f32(&out);
    assert!((v[0] - 1.0).abs() < 1e-6); // R
    assert!((v[1] - 128.0 / 255.0).abs() < 1e-6); // G
    assert!(v[2].abs() < 1e-6); // B
}

#[test]
fn letterbox_pads_non_square_input() {
    let cfg = VisionConfig {
        width: 10,
        height: 10,
        resize: ResizeMode::Letterbox,
        pad: 0,
        ..Default::default()
    };
    let pre = VisionPreprocessor::new(cfg).unwrap();

    // Wide 20x10 white image -> fits to 10x5 centered, top/bottom rows padded 0.
    let packet = uint8_packet(solid_png(20, 10, [255, 255, 255]));
    let out = pre.apply(&packet).unwrap();
    assert_eq!(out.shape, vec![1, 3, 10, 10]);

    let v = output_f32(&out);
    let plane = 100; // 10 * 10, R channel
    // Top-left corner (row 0) is padding -> 0.0; center row is image -> ~1.0.
    assert!(v[0].abs() < 1e-6, "expected padded corner 0.0, got {}", v[0]);
    let center = 5 * 10 + 5; // row 5, col 5
    assert!(
        (v[center] - 1.0).abs() < 1e-6,
        "expected image center ~1.0, got {}",
        v[center]
    );
    let _ = plane;
}

#[test]
fn rejects_non_uint8_input() {
    let pre = VisionPreprocessor::new(VisionConfig::default()).unwrap();
    let bad = BinaryTensorPacket::new(vec![1], TensorDtype::Float32, vec![0, 0, 0, 0]).unwrap();
    assert!(pre.apply(&bad).is_err());
}

#[test]
fn rejects_undecodable_bytes() {
    let pre = VisionPreprocessor::new(VisionConfig::default()).unwrap();
    let junk = uint8_packet(vec![1, 2, 3, 4, 5]);
    assert!(pre.apply(&junk).is_err());
}

#[test]
fn zero_std_config_rejected() {
    let cfg = VisionConfig {
        std: [0.0, 1.0, 1.0],
        ..Default::default()
    };
    assert!(VisionPreprocessor::new(cfg).is_err());
}
