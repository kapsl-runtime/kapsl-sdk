//! ONNX object-detection backend (`EngineKind::OnnxDetect`).
//!
//! Detection is the same ONNX forward pass as [`crate::onnx::OnnxBackend`]
//! producing a raw prediction grid, followed by box decoding and non-max
//! suppression (NMS). This backend wraps an inner ONNX engine and post-processes
//! its output into a compact list of surviving detections.
//!
//! Output layout is `[num_detections, 6]`, each row `[x1, y1, x2, y2, score,
//! class_id]` in the model's input pixel space (i.e. relative to the tensor the
//! model actually saw — pair this with the vision preprocessor's `width`/`height`
//! and letterbox settings to map back to the original image).
//!
//! The decoder targets the common single-tensor YOLO family and is configured
//! from the manifest's `metadata.detect` block, e.g.:
//!
//! ```yaml
//! metadata:
//!   detect:
//!     num_classes: 80
//!     score_threshold: 0.25
//!     iou_threshold: 0.45
//!     max_detections: 300
//!     box_format: xywh        # center form (default); or `xyxy`
//!     objectness: true        # YOLOv5 has an objectness channel; YOLOv8 does not
//!     transposed: false       # YOLOv8 emits [1, 4+nc, anchors]; set true for it
//! ```

use crate::tensor_util::{bytes_to_f32, dim_usize, f32_packet};
use async_trait::async_trait;
use kapsl_engine_api::{
    BatchingPolicy, BinaryTensorPacket, Engine, EngineError, EngineMetrics, EngineModelInfo,
    EngineStream, InferenceRequest, TensorDtype,
};
use serde::Deserialize;
use std::path::Path;

/// Encoding of the four box coordinates in the raw prediction.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum BoxFormat {
    /// `(cx, cy, w, h)` — center + size (YOLO default).
    #[default]
    Xywh,
    /// `(x1, y1, x2, y2)` — corner coordinates.
    Xyxy,
}

fn default_score_threshold() -> f32 {
    0.25
}
fn default_iou_threshold() -> f32 {
    0.45
}
fn default_max_detections() -> usize {
    300
}
fn default_objectness() -> bool {
    true
}

/// Parsed `metadata.detect` block.
#[derive(Debug, Clone, Deserialize)]
pub struct DetectConfig {
    /// Number of object classes the model predicts. Required.
    pub num_classes: usize,
    #[serde(default = "default_score_threshold")]
    pub score_threshold: f32,
    #[serde(default = "default_iou_threshold")]
    pub iou_threshold: f32,
    #[serde(default = "default_max_detections")]
    pub max_detections: usize,
    #[serde(default)]
    pub box_format: BoxFormat,
    /// Whether the prediction has a separate objectness channel (multiplied into
    /// the class score). True for YOLOv5-style, false for YOLOv8-style.
    #[serde(default = "default_objectness")]
    pub objectness: bool,
    /// Whether the raw output is channel-major `[.., channels, anchors]`
    /// (YOLOv8) rather than anchor-major `[.., anchors, channels]` (YOLOv5).
    #[serde(default)]
    pub transposed: bool,
    /// Run NMS across all classes together instead of per class.
    #[serde(default)]
    pub class_agnostic: bool,
}

impl DetectConfig {
    /// Channels per anchor implied by the config: 4 box coords + optional
    /// objectness + one score per class.
    fn expected_channels(&self) -> usize {
        4 + usize::from(self.objectness) + self.num_classes
    }

    fn validate(&self) -> Result<(), EngineError> {
        if self.num_classes == 0 {
            return Err(EngineError::backend("detect: num_classes must be non-zero"));
        }
        Ok(())
    }
}

/// A single decoded detection prior to NMS.
#[derive(Debug, Clone, Copy)]
struct Candidate {
    x1: f32,
    y1: f32,
    x2: f32,
    y2: f32,
    score: f32,
    class: usize,
}

/// Wraps an inner ONNX engine and turns a raw prediction grid into detections.
pub struct OnnxDetectBackend {
    inner: Box<dyn Engine>,
    cfg: DetectConfig,
}

impl OnnxDetectBackend {
    pub fn new(inner: Box<dyn Engine>, cfg: DetectConfig) -> Result<Self, EngineError> {
        cfg.validate()?;
        Ok(Self { inner, cfg })
    }
}

#[async_trait]
impl Engine for OnnxDetectBackend {
    fn planned_external_device_memory(
        &self,
        model_path: &Path,
    ) -> Result<kapsl_engine_api::ExternalDeviceMemoryReport, EngineError> {
        self.inner.planned_external_device_memory(model_path)
    }

    async fn load(&mut self, model_path: &Path) -> Result<(), EngineError> {
        self.inner.load(model_path).await
    }

    fn actual_external_device_memory(&self) -> kapsl_engine_api::ExternalDeviceMemoryReport {
        self.inner.actual_external_device_memory()
    }

    fn infer(&self, request: &InferenceRequest) -> Result<BinaryTensorPacket, EngineError> {
        let output = self.inner.infer(request)?;
        detect_from_output(&output, &self.cfg)
    }

    fn infer_batch(
        &self,
        requests: &[InferenceRequest],
    ) -> Result<Vec<BinaryTensorPacket>, EngineError> {
        let outputs = self.inner.infer_batch(requests)?;
        if outputs.len() != requests.len() {
            return Err(EngineError::backend(format!(
                "detector batch result length mismatch: expected {}, got {}",
                requests.len(),
                outputs.len()
            )));
        }
        outputs
            .into_iter()
            .map(|output| detect_from_output(&output, &self.cfg))
            .collect()
    }

    fn max_batch(&self) -> usize {
        self.inner.max_batch()
    }

    fn self_batches(&self) -> bool {
        self.inner.self_batches()
    }

    fn batching_policy(&self) -> BatchingPolicy {
        self.inner.batching_policy()
    }

    fn infer_stream(&self, request: &InferenceRequest) -> EngineStream {
        let result = self.infer(request);
        Box::pin(futures::stream::once(async move { result }))
    }

    fn unload(&mut self) {
        self.inner.unload();
    }

    fn metrics(&self) -> EngineMetrics {
        self.inner.metrics()
    }

    fn model_info(&self) -> Option<EngineModelInfo> {
        self.inner.model_info()
    }

    fn health_check(&self) -> Result<(), EngineError> {
        self.inner.health_check()
    }
}

/// Decode a raw YOLO-style prediction tensor into `[num_detections, 6]`.
fn detect_from_output(
    output: &BinaryTensorPacket,
    cfg: &DetectConfig,
) -> Result<BinaryTensorPacket, EngineError> {
    if output.dtype != TensorDtype::Float32 {
        return Err(EngineError::backend(format!(
            "detector output dtype {:?} is not supported (expected float32)",
            output.dtype
        )));
    }
    let values = bytes_to_f32(&output.data);

    // Reduce shape to the two grid dims, dropping a leading batch of 1.
    let (d0, d1) = match output.shape.as_slice() {
        [a, c] => (dim_usize(*a), dim_usize(*c)),
        [1, a, c] => (dim_usize(*a), dim_usize(*c)),
        other => {
            return Err(EngineError::backend(format!(
                "detector expects a 2-D [anchors, channels] or 3-D [1, anchors, channels] \
                 output (channels/anchors transposed when metadata.detect.transposed=true), \
                 got shape {:?}",
                other
            )));
        }
    };

    // Map the two dims to (anchors, channels) per the transpose flag.
    let (num_anchors, channels) = if cfg.transposed { (d1, d0) } else { (d0, d1) };

    let expected = cfg.expected_channels();
    if channels != expected {
        return Err(EngineError::backend(format!(
            "detector output has {} channels per anchor but config implies {} \
             (4 box + {} objectness + {} classes); check num_classes/objectness/transposed",
            channels,
            expected,
            usize::from(cfg.objectness),
            cfg.num_classes
        )));
    }
    if values.len() != num_anchors * channels {
        return Err(EngineError::backend(format!(
            "detector output has {} values but shape implies {}x{}={}",
            values.len(),
            num_anchors,
            channels,
            num_anchors * channels
        )));
    }

    // Accessor for channel `ch` of anchor `a` given the memory layout.
    let at = |a: usize, ch: usize| -> f32 {
        let idx = if cfg.transposed {
            ch * num_anchors + a
        } else {
            a * channels + ch
        };
        values[idx]
    };

    let class_offset = 4 + usize::from(cfg.objectness);
    let mut candidates: Vec<Candidate> = Vec::new();
    for a in 0..num_anchors {
        // Best class and its score for this anchor.
        let mut best_class = 0usize;
        let mut best_class_score = f32::NEG_INFINITY;
        for c in 0..cfg.num_classes {
            let s = at(a, class_offset + c);
            if s > best_class_score {
                best_class_score = s;
                best_class = c;
            }
        }
        let score = if cfg.objectness {
            at(a, 4) * best_class_score
        } else {
            best_class_score
        };
        if score < cfg.score_threshold {
            continue;
        }

        let (b0, b1, b2, b3) = (at(a, 0), at(a, 1), at(a, 2), at(a, 3));
        let (x1, y1, x2, y2) = match cfg.box_format {
            BoxFormat::Xywh => (b0 - b2 / 2.0, b1 - b3 / 2.0, b0 + b2 / 2.0, b1 + b3 / 2.0),
            BoxFormat::Xyxy => (b0, b1, b2, b3),
        };
        candidates.push(Candidate {
            x1,
            y1,
            x2,
            y2,
            score,
            class: best_class,
        });
    }

    let kept = non_max_suppression(candidates, cfg.iou_threshold, cfg.class_agnostic);
    let kept = &kept[..kept.len().min(cfg.max_detections)];

    let mut out = Vec::with_capacity(kept.len() * 6);
    for d in kept {
        out.extend_from_slice(&[d.x1, d.y1, d.x2, d.y2, d.score, d.class as f32]);
    }
    Ok(f32_packet(vec![kept.len() as i64, 6], out))
}

/// Greedy non-max suppression. Returns survivors sorted by descending score.
/// When `class_agnostic` is false, boxes only suppress others of the same class.
fn non_max_suppression(
    mut candidates: Vec<Candidate>,
    iou_threshold: f32,
    class_agnostic: bool,
) -> Vec<Candidate> {
    candidates.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut kept: Vec<Candidate> = Vec::new();
    'outer: for cand in candidates {
        for k in &kept {
            if (class_agnostic || k.class == cand.class) && iou(k, &cand) > iou_threshold {
                continue 'outer;
            }
        }
        kept.push(cand);
    }
    kept
}

/// Intersection-over-union of two axis-aligned boxes.
fn iou(a: &Candidate, b: &Candidate) -> f32 {
    let ix1 = a.x1.max(b.x1);
    let iy1 = a.y1.max(b.y1);
    let ix2 = a.x2.min(b.x2);
    let iy2 = a.y2.min(b.y2);
    let iw = (ix2 - ix1).max(0.0);
    let ih = (iy2 - iy1).max(0.0);
    let inter = iw * ih;
    let area_a = (a.x2 - a.x1).max(0.0) * (a.y2 - a.y1).max(0.0);
    let area_b = (b.x2 - b.x1).max(0.0) * (b.y2 - b.y1).max(0.0);
    let union = area_a + area_b - inter;
    if union <= 0.0 {
        0.0
    } else {
        inter / union
    }
}

#[cfg(test)]
#[path = "onnx_detect_tests.rs"]
mod onnx_detect_tests;
