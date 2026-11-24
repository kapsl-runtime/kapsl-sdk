//! ONNX object-detection backend (`EngineKind::OnnxDetect`).
//!
//! Detection is the same ONNX forward pass as [`crate::onnx::OnnxBackend`]
//! producing a raw prediction grid, followed by box decoding and non-max
//! suppression (NMS). This backend wraps an inner ONNX engine and post-processes
//! its output into a compact list of surviving detections.
//!
//! Output layout is `[num_detections, 6]`, each row `[x1, y1, x2, y2, score,
//! class_id]`. When the package declares a vision preprocessor the boxes are
//! mapped back to the *original image's* pixel coordinates (undoing the resize,
//! scale and letterbox pad the preprocessor applied) and clipped to the image;
//! set `coordinates: model` to keep them in the model-input pixel space instead.
//! Without a vision preprocessor (tensor-in) boxes are in model-input space and
//! unclipped, since the image size is unknown.
//!
//! With `output_format: json` (which requires `labels`) the output is instead a
//! UTF-8 JSON array (dtype `Utf8`, shape `[byte_len]`) of `{"box": [x1,y1,x2,y2],
//! "score": f, "class_id": n, "label": "name"}` objects, since a numeric tensor
//! cannot carry names. With the default `tensor`, `labels` (if given) is only
//! validated and does not change the output.
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
//!     coordinates: original   # or `model` (see above)
//!     labels: [person, bicycle, ...]   # optional; length must equal num_classes
//!     output_format: tensor   # default; or `json` (needs `labels`)
//! ```
//!
//! Unknown keys are rejected at load time so a typo cannot silently change
//! behavior.

use crate::preprocess::vision::{encoded_image_dimensions, FitGeometry, VisionConfig};
use crate::tensor_util::{bytes_to_f32, dim_usize, f32_packet};
use async_trait::async_trait;
use kapsl_engine_api::{
    BatchingPolicy, BinaryTensorPacket, Engine, EngineError, EngineMetrics, EngineModelInfo,
    EngineStream, InferenceRequest, TensorDtype,
};
use serde::{Deserialize, Serialize};
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

/// Coordinate space of reported boxes when the package has a vision preprocessor.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Coordinates {
    /// Pixels of the original image sent by the client (resize/pad undone).
    #[default]
    Original,
    /// Pixels of the model input tensor (after resize and letterbox pad).
    Model,
}

/// Wire format of the detection result.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum OutputFormat {
    /// `[N, 6]` float32 rows `[x1, y1, x2, y2, score, class_id]`.
    #[default]
    Tensor,
    /// UTF-8 JSON array of objects that include the class label.
    Json,
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
#[serde(deny_unknown_fields)]
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
    /// Class names indexed by class id. When set, its length must equal
    /// `num_classes`. Used in the output only when `output_format` is `json`.
    #[serde(default)]
    pub labels: Option<Vec<String>>,
    /// Box coordinate space when a vision preprocessor is configured.
    #[serde(default)]
    pub coordinates: Coordinates,
    /// Result encoding: `tensor` (default) or `json`.
    #[serde(default)]
    pub output_format: OutputFormat,
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
        if let Some(labels) = &self.labels {
            if labels.len() != self.num_classes {
                return Err(EngineError::backend(format!(
                    "detect: labels has {} entries but num_classes is {}",
                    labels.len(),
                    self.num_classes
                )));
            }
        }
        if self.output_format == OutputFormat::Json && self.labels.is_none() {
            return Err(EngineError::backend(
                "detect: output_format `json` requires `labels` (the JSON carries class names)",
            ));
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
///
/// When built [`with_image_input`](Self::with_image_input), `inner` must accept
/// the client's raw encoded-image request (i.e. it includes the vision
/// preprocessor), so the original image size can be read from the request and
/// boxes mapped back to it.
pub struct OnnxDetectBackend {
    inner: Box<dyn Engine>,
    cfg: DetectConfig,
    image_input: Option<VisionConfig>,
}

impl OnnxDetectBackend {
    pub fn new(inner: Box<dyn Engine>, cfg: DetectConfig) -> Result<Self, EngineError> {
        cfg.validate()?;
        Ok(Self {
            inner,
            cfg,
            image_input: None,
        })
    }

    /// Declare that requests carry encoded images preprocessed with `vision`, so
    /// boxes can be mapped to original-image coordinates and clipped.
    pub fn with_image_input(mut self, vision: VisionConfig) -> Self {
        self.image_input = Some(vision);
        self
    }

    /// Turn one raw model output into the response packet for `request`.
    fn finish(
        &self,
        request: &InferenceRequest,
        output: &BinaryTensorPacket,
    ) -> Result<BinaryTensorPacket, EngineError> {
        let mut dets = decode_detections(output, &self.cfg)?;
        if let Some(vision) = &self.image_input {
            let (w, h) = encoded_image_dimensions(&request.input)?;
            let geo = vision.geometry(w, h);
            dets = match self.cfg.coordinates {
                Coordinates::Original => place_boxes(dets, Some(&geo), w as f32, h as f32),
                Coordinates::Model => {
                    place_boxes(dets, None, vision.width as f32, vision.height as f32)
                }
            };
        }
        encode_detections(&dets, &self.cfg)
    }
}

#[async_trait]
impl Engine for OnnxDetectBackend {
    fn kv_capabilities(&self) -> kapsl_engine_api::KvBackendCapabilities {
        self.inner.kv_capabilities()
    }

    fn kv_topology(&self) -> Option<kapsl_engine_api::KvTopology> {
        self.inner.kv_topology()
    }

    fn planned_memory(
        &self,
        model_path: &Path,
    ) -> Result<kapsl_engine_api::MemoryReport, EngineError> {
        self.inner.planned_memory(model_path)
    }

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

    fn actual_memory(&self) -> kapsl_engine_api::MemoryReport {
        self.inner.actual_memory()
    }

    fn infer(&self, request: &InferenceRequest) -> Result<BinaryTensorPacket, EngineError> {
        let output = self.inner.infer(request)?;
        self.finish(request, &output)
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
        requests
            .iter()
            .zip(&outputs)
            .map(|(request, output)| self.finish(request, output))
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

/// Decode a raw YOLO-style prediction tensor into the response packet, with
/// boxes left in model-input space.
#[cfg(test)]
fn detect_from_output(
    output: &BinaryTensorPacket,
    cfg: &DetectConfig,
) -> Result<BinaryTensorPacket, EngineError> {
    encode_detections(&decode_detections(output, cfg)?, cfg)
}

/// Move boxes into the reported coordinate space and clip them to `bound_w x
/// bound_h`. With `geometry`, boxes are first mapped from model-input space back
/// to the source image. Boxes left with no area after clipping (entirely in the
/// letterbox padding or outside the image) are dropped.
fn place_boxes(
    dets: Vec<Candidate>,
    geometry: Option<&FitGeometry>,
    bound_w: f32,
    bound_h: f32,
) -> Vec<Candidate> {
    dets.into_iter()
        .filter_map(|mut d| {
            if let Some(g) = geometry {
                (d.x1, d.y1) = g.to_source(d.x1, d.y1);
                (d.x2, d.y2) = g.to_source(d.x2, d.y2);
            }
            d.x1 = d.x1.clamp(0.0, bound_w);
            d.x2 = d.x2.clamp(0.0, bound_w);
            d.y1 = d.y1.clamp(0.0, bound_h);
            d.y2 = d.y2.clamp(0.0, bound_h);
            (d.x2 > d.x1 && d.y2 > d.y1).then_some(d)
        })
        .collect()
}

#[derive(Serialize)]
struct DetectionJson<'a> {
    #[serde(rename = "box")]
    bbox: [f32; 4],
    score: f32,
    class_id: usize,
    label: &'a str,
}

/// Encode detections as `[N, 6]` float32 rows (`tensor`) or as a labeled JSON
/// array (`json`).
fn encode_detections(
    dets: &[Candidate],
    cfg: &DetectConfig,
) -> Result<BinaryTensorPacket, EngineError> {
    if cfg.output_format == OutputFormat::Json {
        // validate() guarantees labels are present and cover every class id.
        let labels = cfg.labels.as_deref().unwrap_or_default();
        let rows: Vec<DetectionJson<'_>> = dets
            .iter()
            .map(|d| DetectionJson {
                bbox: [d.x1, d.y1, d.x2, d.y2],
                score: d.score,
                class_id: d.class,
                label: &labels[d.class],
            })
            .collect();
        let json = serde_json::to_vec(&rows)
            .map_err(|e| EngineError::backend(format!("detect: encoding output failed: {e}")))?;
        return BinaryTensorPacket::new(vec![json.len() as i64], TensorDtype::Utf8, json);
    }
    let mut out = Vec::with_capacity(dets.len() * 6);
    for d in dets {
        out.extend_from_slice(&[d.x1, d.y1, d.x2, d.y2, d.score, d.class as f32]);
    }
    Ok(f32_packet(vec![dets.len() as i64, 6], out))
}

/// Decode a raw YOLO-style prediction tensor into post-NMS detections in
/// model-input space.
fn decode_detections(
    output: &BinaryTensorPacket,
    cfg: &DetectConfig,
) -> Result<Vec<Candidate>, EngineError> {
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
    let mut kept = kept;
    kept.truncate(cfg.max_detections);
    Ok(kept)
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
#[path = "detect_tests.rs"]
mod onnx_detect_tests;
