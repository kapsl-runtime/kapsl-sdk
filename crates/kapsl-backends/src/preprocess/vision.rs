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
//!     channel_order: rgb       # or `bgr` for models trained on cv2-loaded images (YOLOX)
//! ```
//!
//! Per-channel output is `(pixel * scale - mean[c]) / std[c]`, where `c` indexes
//! the *output* channels (after any `channel_order` swap). With the defaults
//! (`scale = 1/255`, `mean = 0`, `std = 1`) this is a plain `pixel / 255`.

use image::{imageops::FilterType, RgbImage};
use kapsl_engine_api::{BinaryTensorPacket, EngineError, TensorDtype};
use serde::Deserialize;

use super::Preprocessor;

/// How the decoded image is fit to the model's fixed input size.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ResizeMode {
    /// Resize directly to `width x height`, distorting aspect ratio.
    #[default]
    Stretch,
    /// Resize preserving aspect ratio to fit within `width x height`, then pad
    /// the remainder with `pad` (letterboxing, as used by YOLO-family models).
    Letterbox,
}

/// Channel layout of the emitted tensor.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Layout {
    /// `[1, channels, height, width]` — the common PyTorch/ONNX export layout.
    #[default]
    Nchw,
    /// `[1, height, width, channels]` — some TensorFlow-origin exports.
    Nhwc,
}

/// Order of the three color channels in the emitted tensor.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ChannelOrder {
    /// Red, green, blue — the order the image is decoded in.
    #[default]
    Rgb,
    /// Blue, green, red — for models trained on OpenCV-loaded images.
    Bgr,
}

impl ChannelOrder {
    /// Index into a decoded RGB pixel for output channel `c`.
    fn source_channel(self, c: usize) -> usize {
        match self {
            Self::Rgb => c,
            Self::Bgr => 2 - c,
        }
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
    /// Channel order of the emitted tensor. Default `rgb`.
    #[serde(default)]
    pub channel_order: ChannelOrder,
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
            channel_order: ChannelOrder::default(),
        }
    }
}

impl VisionConfig {
    /// Resize/pad geometry for a `src_w x src_h` source image.
    pub fn geometry(&self, src_w: u32, src_h: u32) -> FitGeometry {
        let (w, h) = (self.width, self.height);
        match self.resize {
            ResizeMode::Stretch => FitGeometry {
                src_w,
                src_h,
                fit_w: w,
                fit_h: h,
                pad_x: 0,
                pad_y: 0,
            },
            ResizeMode::Letterbox => {
                // Scale to fit within the box, preserving aspect ratio.
                let (iw, ih) = (src_w as f32, src_h as f32);
                let ratio = (w as f32 / iw).min(h as f32 / ih);
                let fit_w = ((iw * ratio).round().max(1.0) as u32).min(w);
                let fit_h = ((ih * ratio).round().max(1.0) as u32).min(h);
                FitGeometry {
                    src_w,
                    src_h,
                    fit_w,
                    fit_h,
                    // Centered within the canvas.
                    pad_x: (w - fit_w) / 2,
                    pad_y: (h - fit_h) / 2,
                }
            }
        }
    }

    fn validate(&self) -> Result<(), EngineError> {
        if self.width == 0 || self.height == 0 {
            return Err(EngineError::backend(format!(
                "vision preprocess: width and height must be non-zero (got {}x{})",
                self.width, self.height
            )));
        }
        if self.std.contains(&0.0) {
            return Err(EngineError::backend(
                "vision preprocess: std values must be non-zero (division by zero)",
            ));
        }
        Ok(())
    }
}

/// Where the source image lands inside the model input: the single source of
/// truth for resize, scale and pad, shared by [`VisionPreprocessor`] (which
/// applies it) and the detection post-processor (which inverts it to report boxes
/// in original-image coordinates).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FitGeometry {
    /// Source (original image) size in pixels.
    pub src_w: u32,
    pub src_h: u32,
    /// Size the source is resized to before padding. Equals the model input for
    /// `stretch`; smaller on one axis for `letterbox`.
    pub fit_w: u32,
    pub fit_h: u32,
    /// Top-left of the resized image inside the model input (letterbox padding).
    pub pad_x: u32,
    pub pad_y: u32,
}

impl FitGeometry {
    /// Horizontal and vertical factor from source pixels to model-input pixels
    /// (the exact ratio of the rounded resize, not the nominal letterbox ratio).
    pub fn scale(&self) -> (f32, f32) {
        (
            self.fit_w as f32 / self.src_w as f32,
            self.fit_h as f32 / self.src_h as f32,
        )
    }

    /// Map a point from model-input space back to source-image space (unclipped).
    pub fn to_source(&self, x: f32, y: f32) -> (f32, f32) {
        let (sx, sy) = self.scale();
        ((x - self.pad_x as f32) / sx, (y - self.pad_y as f32) / sy)
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
        let geo = self.cfg.geometry(img.width(), img.height());
        let resized = image::imageops::resize(img, geo.fit_w, geo.fit_h, FilterType::Triangle);
        match self.cfg.resize {
            ResizeMode::Stretch => resized,
            ResizeMode::Letterbox => {
                let pad = image::Rgb([self.cfg.pad, self.cfg.pad, self.cfg.pad]);
                let mut canvas = RgbImage::from_pixel(w, h, pad);
                image::imageops::overlay(&mut canvas, &resized, geo.pad_x as i64, geo.pad_y as i64);
                canvas
            }
        }
    }

    /// Normalize an already-resized `width x height` RGB image into an f32 vector
    /// laid out per `self.cfg.layout`.
    fn normalize(&self, img: &RgbImage) -> Vec<f32> {
        let (w, h) = (self.cfg.width as usize, self.cfg.height as usize);
        let VisionConfig {
            scale,
            mean,
            std,
            channel_order,
            ..
        } = self.cfg;
        let mut out = vec![0f32; 3 * w * h];
        match self.cfg.layout {
            Layout::Nchw => {
                // channel-major: out[c*h*w + y*w + x]
                let plane = w * h;
                for (i, px) in img.pixels().enumerate() {
                    // pixels() iterates row-major (y, then x), matching y*w + x.
                    for c in 0..3 {
                        out[c * plane + i] =
                            (px[channel_order.source_channel(c)] as f32 * scale - mean[c]) / std[c];
                    }
                }
            }
            Layout::Nhwc => {
                // interleaved: out[(y*w + x)*3 + c]
                for (i, px) in img.pixels().enumerate() {
                    for c in 0..3 {
                        out[i * 3 + c] =
                            (px[channel_order.source_channel(c)] as f32 * scale - mean[c]) / std[c];
                    }
                }
            }
        }
        out
    }
}

/// Pixel dimensions of an encoded image packet, read from the header without
/// decoding the pixels. Matches what [`VisionPreprocessor`] decodes.
pub fn encoded_image_dimensions(input: &BinaryTensorPacket) -> Result<(u32, u32), EngineError> {
    if input.dtype != TensorDtype::Uint8 {
        return Err(EngineError::invalid_input(format!(
            "expected encoded image bytes as a Uint8 packet, got {:?}",
            input.dtype
        )));
    }
    image::ImageReader::new(std::io::Cursor::new(&input.data))
        .with_guessed_format()
        .and_then(|r| r.into_dimensions().map_err(std::io::Error::other))
        .map_err(|e| EngineError::invalid_input(format!("failed to read image dimensions: {e}")))
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
