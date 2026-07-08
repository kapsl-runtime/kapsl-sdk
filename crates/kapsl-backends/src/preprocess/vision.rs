//! Vision input preprocessing: encoded image bytes -> normalized image tensor.
//!
//! The client sends a `Uint8` 1-D packet holding the raw bytes of an encoded
//! image file (JPEG or PNG). This preprocessor decodes it, resizes it to the
//! model's fixed input size, normalizes the pixels, and emits a `Float32` tensor
//! in the requested channel layout — the standard front-end for image
//! classifiers, embedders, and detector backbones (ResNet, ViT, YOLO, CLIP image
//! tower, …).
//!
//! Configuration comes from the package manifest's
//! `metadata.preprocess` block, e.g. (ImageNet-normalized 224² classifier):
//!
//! ```yaml
//! metadata:
//!   preprocess:
//!     kind: vision
//!     width: 224
//!     height: 224
//!     resize: stretch          # or `letterbox` to preserve aspect ratio
//!     layout: nchw             # or `nhwc`
//!     scale: 0.00392156862     # multiply raw 0..255 pixels (default 1/255)
//!     mean: [0.485, 0.456, 0.406]
//!     std:  [0.229, 0.224, 0.225]
//! ```
//!
//! Per-channel output is `(pixel * scale - mean[c]) / std[c]`. With the defaults
//! (`scale = 1/255`, `mean = 0`, `std = 1`) this is a plain `pixel / 255`.

use image::{imageops::FilterType, RgbImage};
use kapsl_engine_api::{BinaryTensorPacket, EngineError, TensorDtype};
use serde::Deserialize;

use super::Preprocessor;

/// How the decoded image is fit to the model's fixed input size.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ResizeMode {
    /// Resize directly to `width x height`, distorting aspect ratio.
    Stretch,
    /// Resize preserving aspect ratio to fit within `width x height`, then pad
    /// the remainder with `pad` (letterboxing, as used by YOLO-family models).
    Letterbox,
}

impl Default for ResizeMode {
    fn default() -> Self {
        ResizeMode::Stretch
    }
}

/// Channel layout of the emitted tensor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Layout {
    /// `[1, channels, height, width]` — the common PyTorch/ONNX export layout.
    Nchw,
    /// `[1, height, width, channels]` — some TensorFlow-origin exports.
    Nhwc,
}

impl Default for Layout {
    fn default() -> Self {
        Layout::Nchw
    }
}

fn default_width() -> u32 {
    224
}
fn default_height() -> u32 {
    224
}
fn default_scale() -> f32 {
    1.0 / 255.0
}
fn default_mean() -> [f32; 3] {
    [0.0, 0.0, 0.0]
}
fn default_std() -> [f32; 3] {
    [1.0, 1.0, 1.0]
}
fn default_pad() -> u8 {
    0
}

/// Parsed `metadata.preprocess` block for a vision model.
#[derive(Debug, Clone, Deserialize)]
pub struct VisionConfig {
    #[serde(default = "default_width")]
    pub width: u32,
    #[serde(default = "default_height")]
    pub height: u32,
    #[serde(default)]
    pub resize: ResizeMode,
    #[serde(default)]
    pub layout: Layout,
    /// Multiplier applied to raw 0..=255 pixels before mean/std. Default `1/255`.
    #[serde(default = "default_scale")]
    pub scale: f32,
    #[serde(default = "default_mean")]
    pub mean: [f32; 3],
    #[serde(default = "default_std")]
    pub std: [f32; 3],
    /// Letterbox padding value in raw 0..=255 pixel space. Default `0`.
    #[serde(default = "default_pad")]
    pub pad: u8,
}

impl Default for VisionConfig {
    fn default() -> Self {
        Self {
            width: default_width(),
            height: default_height(),
            resize: ResizeMode::default(),
            layout: Layout::default(),
            scale: default_scale(),
            mean: default_mean(),
            std: default_std(),
            pad: default_pad(),
        }
    }
}

impl VisionConfig {
    fn validate(&self) -> Result<(), EngineError> {
        if self.width == 0 || self.height == 0 {
            return Err(EngineError::backend(format!(
                "vision preprocess: width and height must be non-zero (got {}x{})",
                self.width, self.height
            )));
        }
        if self.std.iter().any(|s| *s == 0.0) {
            return Err(EngineError::backend(
                "vision preprocess: std values must be non-zero (division by zero)",
            ));
        }
        Ok(())
    }
}

/// Decodes and normalizes encoded images into a model input tensor.
pub struct VisionPreprocessor {
    cfg: VisionConfig,
}

impl VisionPreprocessor {
    pub fn new(cfg: VisionConfig) -> Result<Self, EngineError> {
        cfg.validate()?;
        Ok(Self { cfg })
    }

    /// Resize the decoded image to the configured `width x height` per the resize
    /// mode, returning an owned `width x height` RGB buffer.
    fn fit(&self, img: &RgbImage) -> RgbImage {
        let (w, h) = (self.cfg.width, self.cfg.height);
        match self.cfg.resize {
            ResizeMode::Stretch => image::imageops::resize(img, w, h, FilterType::Triangle),
            ResizeMode::Letterbox => {
                // Scale to fit within the box, preserving aspect ratio.
                let (iw, ih) = (img.width() as f32, img.height() as f32);
                let ratio = (w as f32 / iw).min(h as f32 / ih);
                let nw = (iw * ratio).round().max(1.0) as u32;
                let nh = (ih * ratio).round().max(1.0) as u32;
                let resized = image::imageops::resize(img, nw, nh, FilterType::Triangle);

                let pad = image::Rgb([self.cfg.pad, self.cfg.pad, self.cfg.pad]);
                let mut canvas = RgbImage::from_pixel(w, h, pad);
                // Center the resized image within the canvas.
                let ox = ((w - nw) / 2) as i64;
                let oy = ((h - nh) / 2) as i64;
                image::imageops::overlay(&mut canvas, &resized, ox, oy);
                canvas
            }
        }
    }

    /// Normalize an already-resized `width x height` RGB image into an f32 vector
    /// laid out per `self.cfg.layout`.
    fn normalize(&self, img: &RgbImage) -> Vec<f32> {
        let (w, h) = (self.cfg.width as usize, self.cfg.height as usize);
        let VisionConfig {
            scale, mean, std, ..
        } = self.cfg;
        let mut out = vec![0f32; 3 * w * h];
        match self.cfg.layout {
            Layout::Nchw => {
                // channel-major: out[c*h*w + y*w + x]
                let plane = w * h;
                for (i, px) in img.pixels().enumerate() {
                    // pixels() iterates row-major (y, then x), matching y*w + x.
                    for c in 0..3 {
                        out[c * plane + i] = (px[c] as f32 * scale - mean[c]) / std[c];
                    }
                }
            }
            Layout::Nhwc => {
                // interleaved: out[(y*w + x)*3 + c]
                for (i, px) in img.pixels().enumerate() {
                    for c in 0..3 {
                        out[i * 3 + c] = (px[c] as f32 * scale - mean[c]) / std[c];
                    }
                }
            }
        }
        out
    }
}

impl Preprocessor for VisionPreprocessor {
    fn apply(&self, input: &BinaryTensorPacket) -> Result<BinaryTensorPacket, EngineError> {
        if input.dtype != TensorDtype::Uint8 {
            return Err(EngineError::invalid_input(format!(
                "vision preprocess expects encoded image bytes as a Uint8 packet, got {:?}",
                input.dtype
            )));
        }
        if input.data.is_empty() {
            return Err(EngineError::invalid_input(
                "vision preprocess received an empty image payload",
            ));
        }

        let decoded = image::load_from_memory(&input.data)
            .map_err(|e| EngineError::invalid_input(format!("failed to decode image: {e}")))?
            .to_rgb8();

        let fitted = self.fit(&decoded);
        let values = self.normalize(&fitted);

        let (w, h) = (self.cfg.width as i64, self.cfg.height as i64);
        let shape = match self.cfg.layout {
            Layout::Nchw => vec![1, 3, h, w],
            Layout::Nhwc => vec![1, h, w, 3],
        };
        let data = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        BinaryTensorPacket::new(shape, TensorDtype::Float32, data)
    }

    fn name(&self) -> &'static str {
        "vision"
    }
}
