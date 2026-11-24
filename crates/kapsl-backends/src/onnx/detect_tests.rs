//! Tests for the ONNX detection post-processor (box decode and NMS).

use super::*;

fn cfg(num_classes: usize) -> DetectConfig {
    DetectConfig {
        num_classes,
        score_threshold: 0.25,
        iou_threshold: 0.45,
        max_detections: 300,
        box_format: BoxFormat::Xyxy,
        objectness: false,
        transposed: false,
        class_agnostic: false,
        labels: None,
        coordinates: Coordinates::default(),
        output_format: OutputFormat::default(),
    }
}

/// Build a raw anchor-major `[anchors, channels]` f32 packet from rows.
fn raw_packet(rows: &[Vec<f32>]) -> BinaryTensorPacket {
    let anchors = rows.len() as i64;
    let channels = rows[0].len() as i64;
    let flat: Vec<f32> = rows.iter().flatten().copied().collect();
    f32_packet(vec![anchors, channels], flat)
}

fn decode_rows(out: &BinaryTensorPacket) -> Vec<[f32; 6]> {
    assert_eq!(out.dtype, TensorDtype::Float32);
    assert_eq!(out.shape[1], 6);
    bytes_to_f32(&out.data).as_chunks::<6>().0.to_vec()
}

#[test]
fn thresholds_out_low_scores() {
    // 2 classes, xyxy, no objectness. One strong box, one below threshold.
    let out = detect_from_output(
        &raw_packet(&[
            vec![0.0, 0.0, 10.0, 10.0, 0.9, 0.1],   // class 0 @ 0.9
            vec![50.0, 50.0, 60.0, 60.0, 0.1, 0.2], // best 0.2 < 0.25 -> dropped
        ]),
        &cfg(2),
    )
    .unwrap();
    let dets = decode_rows(&out);
    assert_eq!(dets.len(), 1);
    assert_eq!(dets[0], [0.0, 0.0, 10.0, 10.0, 0.9, 0.0]);
}

#[test]
fn nms_suppresses_overlapping_same_class() {
    // Two near-identical class-0 boxes -> only the higher-scoring one survives.
    let out = detect_from_output(
        &raw_packet(&[
            vec![0.0, 0.0, 10.0, 10.0, 0.9, 0.0],
            vec![0.5, 0.5, 10.5, 10.5, 0.8, 0.0], // IoU high with the first
        ]),
        &cfg(2),
    )
    .unwrap();
    let dets = decode_rows(&out);
    assert_eq!(dets.len(), 1);
    assert_eq!(dets[0][4], 0.9);
}

#[test]
fn nms_keeps_overlapping_different_classes() {
    // Same location, different class -> per-class NMS keeps both.
    let out = detect_from_output(
        &raw_packet(&[
            vec![0.0, 0.0, 10.0, 10.0, 0.9, 0.1], // class 0
            vec![0.5, 0.5, 10.5, 10.5, 0.1, 0.8], // class 1
        ]),
        &cfg(2),
    )
    .unwrap();
    let mut dets = decode_rows(&out);
    dets.sort_by(|a, b| a[5].partial_cmp(&b[5]).unwrap());
    assert_eq!(dets.len(), 2);
    assert_eq!(dets[0][5], 0.0);
    assert_eq!(dets[1][5], 1.0);
}

#[test]
fn class_agnostic_suppresses_across_classes() {
    let mut c = cfg(2);
    c.class_agnostic = true;
    let out = detect_from_output(
        &raw_packet(&[
            vec![0.0, 0.0, 10.0, 10.0, 0.9, 0.1],
            vec![0.5, 0.5, 10.5, 10.5, 0.1, 0.8],
        ]),
        &c,
    )
    .unwrap();
    assert_eq!(decode_rows(&out).len(), 1);
}

#[test]
fn objectness_multiplies_class_score() {
    // YOLOv5 style: channels = 4 box + 1 obj + 2 classes.
    let mut c = cfg(2);
    c.objectness = true;
    // obj=0.5, class0=0.4 -> score 0.2 (< 0.25) dropped; obj=0.9, class1=0.9 -> 0.81 kept
    let out = detect_from_output(
        &raw_packet(&[
            vec![0.0, 0.0, 10.0, 10.0, 0.5, 0.4, 0.1],
            vec![20.0, 20.0, 30.0, 30.0, 0.9, 0.1, 0.9],
        ]),
        &c,
    )
    .unwrap();
    let dets = decode_rows(&out);
    assert_eq!(dets.len(), 1);
    assert_eq!(dets[0][5], 1.0);
    assert!((dets[0][4] - 0.81).abs() < 1e-6);
}

#[test]
fn xywh_converts_to_corners() {
    let mut c = cfg(1);
    c.box_format = BoxFormat::Xywh;
    // center (10,10), size 4x6 -> x1=8,y1=7,x2=12,y2=13
    let out = detect_from_output(&raw_packet(&[vec![10.0, 10.0, 4.0, 6.0, 0.9]]), &c).unwrap();
    let d = decode_rows(&out)[0];
    assert_eq!([d[0], d[1], d[2], d[3]], [8.0, 7.0, 12.0, 13.0]);
}

#[test]
fn transposed_layout_channel_major() {
    // YOLOv8 style [channels, anchors]: 4 box + 2 classes, 2 anchors.
    let mut c = cfg(2);
    c.transposed = true;
    // Column-major columns are anchors. Anchor0: box(0,0,10,10) cls(0.9,0.1);
    // Anchor1: box(20,20,30,30) cls(0.1,0.8).
    // Channel-major flat: [x's, y's, x2's, y2's, cls0's, cls1's]
    let flat = vec![
        0.0, 20.0, // channel 0 (x1) for anchors 0,1
        0.0, 20.0, // channel 1 (y1)
        10.0, 30.0, // channel 2 (x2)
        10.0, 30.0, // channel 3 (y2)
        0.9, 0.1, // channel 4 (class 0)
        0.1, 0.8, // channel 5 (class 1)
    ];
    let packet = f32_packet(vec![6, 2], flat);
    let mut dets = decode_rows(&detect_from_output(&packet, &c).unwrap());
    dets.sort_by(|a, b| a[5].partial_cmp(&b[5]).unwrap());
    assert_eq!(dets.len(), 2);
    assert_eq!(
        [dets[0][0], dets[0][1], dets[0][2], dets[0][3]],
        [0.0, 0.0, 10.0, 10.0]
    );
    assert_eq!(dets[0][5], 0.0);
    assert_eq!(
        [dets[1][0], dets[1][1], dets[1][2], dets[1][3]],
        [20.0, 20.0, 30.0, 30.0]
    );
    assert_eq!(dets[1][5], 1.0);
}

#[test]
fn channel_count_mismatch_errors() {
    // Config expects 4+0+2=6 channels; give 5.
    let err = detect_from_output(&raw_packet(&[vec![0.0, 0.0, 10.0, 10.0, 0.9]]), &cfg(2));
    assert!(err.is_err());
}

#[test]
fn max_detections_caps_output() {
    let mut c = cfg(1);
    c.max_detections = 2;
    // Four non-overlapping high-score boxes; cap to 2 highest.
    let out = detect_from_output(
        &raw_packet(&[
            vec![0.0, 0.0, 5.0, 5.0, 0.5],
            vec![10.0, 10.0, 15.0, 15.0, 0.9],
            vec![20.0, 20.0, 25.0, 25.0, 0.7],
            vec![30.0, 30.0, 35.0, 35.0, 0.6],
        ]),
        &c,
    )
    .unwrap();
    let dets = decode_rows(&out);
    assert_eq!(dets.len(), 2);
    assert_eq!(dets[0][4], 0.9);
    assert_eq!(dets[1][4], 0.7);
}

#[test]
fn empty_when_all_below_threshold() {
    let out =
        detect_from_output(&raw_packet(&[vec![0.0, 0.0, 10.0, 10.0, 0.01]]), &cfg(1)).unwrap();
    assert_eq!(out.shape, vec![0, 6]);
    assert!(decode_rows(&out).is_empty());
}

// ---- labels, strict config, coordinate mapping, clipping ----

use crate::preprocess::vision::{ResizeMode, VisionConfig};
use async_trait::async_trait;
use kapsl_engine_api::{EngineMetrics, KvBackendCapabilities};

fn names(n: usize) -> Vec<String> {
    (0..n).map(|i| format!("class{i}")).collect()
}

#[test]
fn labels_length_must_match_num_classes() {
    let mut c = cfg(3);
    c.labels = Some(names(2));
    let err = c.validate().unwrap_err().to_string();
    assert!(
        err.contains("labels has 2 entries but num_classes is 3"),
        "{err}"
    );
    c.labels = Some(Vec::new());
    assert!(
        c.validate().is_err(),
        "empty labels array is a length mismatch"
    );
    c.labels = Some(names(3));
    assert!(c.validate().is_ok());
}

#[test]
fn backend_new_rejects_mismatched_labels_at_load() {
    let mut c = cfg(3);
    c.labels = Some(names(4));
    assert!(
        OnnxDetectBackend::new(Box::new(FixedEngine::new(raw_packet(&[vec![0.0; 7]]))), c).is_err()
    );
}

#[test]
fn unknown_detect_keys_are_rejected() {
    // A typo must fail loudly instead of being ignored.
    let err = serde_yaml::from_str::<DetectConfig>("num_classes: 2\nlables: [a, b]").unwrap_err();
    assert!(err.to_string().contains("unknown field `lables`"), "{err}");
    assert!(serde_yaml::from_str::<DetectConfig>("num_classes: 2\nclass_agnostic: true").is_ok());
}

#[test]
fn coordinates_parse_and_default_to_original() {
    let c: DetectConfig = serde_yaml::from_str("num_classes: 1").unwrap();
    assert_eq!(c.coordinates, Coordinates::Original);
    let c: DetectConfig = serde_yaml::from_str("num_classes: 1\ncoordinates: model").unwrap();
    assert_eq!(c.coordinates, Coordinates::Model);
    assert!(serde_yaml::from_str::<DetectConfig>("num_classes: 1\ncoordinates: bogus").is_err());
}

#[test]
fn json_output_format_includes_class_id_and_label() {
    let mut c = cfg(2);
    c.output_format = OutputFormat::Json;
    c.labels = Some(vec!["cat".into(), "dog \"rex\"".into()]);
    let out = detect_from_output(
        &raw_packet(&[
            vec![0.0, 0.0, 10.0, 10.0, 0.1, 0.9],
            vec![50.0, 50.0, 60.0, 60.0, 0.8, 0.1],
        ]),
        &c,
    )
    .unwrap();
    assert_eq!(out.dtype, TensorDtype::Utf8);
    assert_eq!(out.shape, vec![out.data.len() as i64]);
    let v: serde_json::Value = serde_json::from_slice(&out.data).unwrap();
    let rows = v.as_array().unwrap();
    assert_eq!(rows.len(), 2);
    assert_eq!(rows[0]["class_id"], 1);
    assert_eq!(rows[0]["label"], "dog \"rex\""); // escaping survives
    assert_eq!(rows[0]["box"], serde_json::json!([0.0, 0.0, 10.0, 10.0]));
    assert!((rows[0]["score"].as_f64().unwrap() - 0.9).abs() < 1e-6);
    assert_eq!(rows[1]["class_id"], 0);
    assert_eq!(rows[1]["label"], "cat");
}

#[test]
fn empty_detections_in_json_format_encode_as_empty_json_array() {
    let mut c = cfg(1);
    c.output_format = OutputFormat::Json;
    c.labels = Some(names(1));
    let out = detect_from_output(&raw_packet(&[vec![0.0, 0.0, 10.0, 10.0, 0.01]]), &c).unwrap();
    assert_eq!(out.dtype, TensorDtype::Utf8);
    assert_eq!(out.data, b"[]");
}

/// Engine that ignores the request and returns a fixed raw prediction tensor.
struct FixedEngine {
    output: BinaryTensorPacket,
}

impl FixedEngine {
    fn new(output: BinaryTensorPacket) -> Self {
        Self { output }
    }
}

#[async_trait]
impl Engine for FixedEngine {
    fn kv_capabilities(&self) -> KvBackendCapabilities {
        KvBackendCapabilities::opaque_connected()
    }
    async fn load(&mut self, _model_path: &Path) -> Result<(), EngineError> {
        Ok(())
    }
    fn infer(&self, _request: &InferenceRequest) -> Result<BinaryTensorPacket, EngineError> {
        Ok(self.output.clone())
    }
    fn infer_stream(&self, request: &InferenceRequest) -> EngineStream {
        let result = self.infer(request);
        Box::pin(futures::stream::once(async move { result }))
    }
    fn unload(&mut self) {}
    fn metrics(&self) -> EngineMetrics {
        EngineMetrics::new()
    }
    fn health_check(&self) -> Result<(), EngineError> {
        Ok(())
    }
}

fn png_request(w: u32, h: u32) -> InferenceRequest {
    let mut buf = Vec::new();
    image::DynamicImage::ImageRgb8(image::RgbImage::new(w, h))
        .write_to(&mut std::io::Cursor::new(&mut buf), image::ImageFormat::Png)
        .unwrap();
    InferenceRequest::new(
        BinaryTensorPacket::new(vec![buf.len() as i64], TensorDtype::Uint8, buf).unwrap(),
    )
}

fn vision_100(resize: ResizeMode) -> VisionConfig {
    VisionConfig {
        width: 100,
        height: 100,
        resize,
        ..Default::default()
    }
}

fn backend_with(rows: &[Vec<f32>], c: DetectConfig, v: VisionConfig) -> OnnxDetectBackend {
    OnnxDetectBackend::new(Box::new(FixedEngine::new(raw_packet(rows))), c)
        .unwrap()
        .with_image_input(v)
}

#[test]
fn letterbox_boxes_map_back_to_original_image_pixels() {
    // 200x100 source into a 100x100 box: scale 0.5, fit 100x50, pad_y 25.
    // Model-space box (10,35)-(60,60) -> source (20,20)-(120,70).
    let b = backend_with(
        &[vec![10.0, 35.0, 60.0, 60.0, 0.9]],
        cfg(1),
        vision_100(ResizeMode::Letterbox),
    );
    let out = b.infer(&png_request(200, 100)).unwrap();
    assert_eq!(decode_rows(&out), vec![[20.0, 20.0, 120.0, 70.0, 0.9, 0.0]]);
}

#[test]
fn stretch_boxes_map_back_with_independent_axis_scales() {
    // 200x50 stretched to 100x100: x scale 0.5, y scale 2.
    let b = backend_with(
        &[vec![10.0, 20.0, 50.0, 80.0, 0.9]],
        cfg(1),
        vision_100(ResizeMode::Stretch),
    );
    let out = b.infer(&png_request(200, 50)).unwrap();
    assert_eq!(decode_rows(&out), vec![[20.0, 10.0, 100.0, 40.0, 0.9, 0.0]]);
}

#[test]
fn boxes_are_clipped_to_the_original_image() {
    // 200x100 source, letterbox. This box overhangs every edge in model space.
    let b = backend_with(
        &[vec![-20.0, -10.0, 130.0, 120.0, 0.9]],
        cfg(1),
        vision_100(ResizeMode::Letterbox),
    );
    let out = b.infer(&png_request(200, 100)).unwrap();
    assert_eq!(decode_rows(&out), vec![[0.0, 0.0, 200.0, 100.0, 0.9, 0.0]]);
}

#[test]
fn boxes_entirely_in_the_letterbox_padding_are_dropped() {
    // Padding occupies model y in [0,25) and [75,100) for a 200x100 source.
    let b = backend_with(
        &[
            vec![10.0, 2.0, 60.0, 20.0, 0.9],  // fully in the top pad
            vec![10.0, 30.0, 60.0, 60.0, 0.8], // real content
        ],
        cfg(1),
        vision_100(ResizeMode::Letterbox),
    );
    let out = b.infer(&png_request(200, 100)).unwrap();
    let rows = decode_rows(&out);
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0][4], 0.8);
}

#[test]
fn model_coordinates_flag_keeps_model_space_and_clips_to_the_input() {
    let mut c = cfg(1);
    c.coordinates = Coordinates::Model;
    let b = backend_with(
        &[vec![-5.0, 35.0, 60.0, 130.0, 0.9]],
        c,
        vision_100(ResizeMode::Letterbox),
    );
    let out = b.infer(&png_request(200, 100)).unwrap();
    assert_eq!(decode_rows(&out), vec![[0.0, 35.0, 60.0, 100.0, 0.9, 0.0]]);
}

#[test]
fn without_image_input_boxes_stay_in_model_space_unclipped() {
    let b = OnnxDetectBackend::new(
        Box::new(FixedEngine::new(raw_packet(&[vec![
            -5.0, 0.0, 600.0, 10.0, 0.9,
        ]]))),
        cfg(1),
    )
    .unwrap();
    let req = InferenceRequest::new(f32_packet(vec![1, 1], vec![0.0]));
    assert_eq!(
        decode_rows(&b.infer(&req).unwrap()),
        vec![[-5.0, 0.0, 600.0, 10.0, 0.9, 0.0]]
    );
}

#[test]
fn image_input_rejects_a_non_image_request() {
    let b = backend_with(
        &[vec![0.0, 0.0, 10.0, 10.0, 0.9]],
        cfg(1),
        vision_100(ResizeMode::Letterbox),
    );
    let junk = InferenceRequest::new(
        BinaryTensorPacket::new(vec![4], TensorDtype::Uint8, vec![1, 2, 3, 4]).unwrap(),
    );
    assert!(b.infer(&junk).is_err());
}

#[test]
fn batch_maps_each_request_with_its_own_image_size() {
    let b = backend_with(
        &[vec![10.0, 35.0, 60.0, 60.0, 0.9]],
        cfg(1),
        vision_100(ResizeMode::Letterbox),
    );
    let out = b
        .infer_batch(&[png_request(200, 100), png_request(100, 200)])
        .unwrap();
    // 200x100: scale 0.5, pad_y 25 -> (20,20,120,70).
    assert_eq!(
        decode_rows(&out[0]),
        vec![[20.0, 20.0, 120.0, 70.0, 0.9, 0.0]]
    );
    // 100x200: scale 0.5, pad_x 25 -> x=(10-25)/0.5<0 clips to 0, x2=(60-25)/0.5=70;
    // y=(35-0)/0.5=70, y2=120.
    assert_eq!(
        decode_rows(&out[1]),
        vec![[0.0, 70.0, 70.0, 120.0, 0.9, 0.0]]
    );
}

#[test]
fn tensor_output_format_is_default_and_ignores_labels() {
    let c: DetectConfig = serde_yaml::from_str("num_classes: 2").unwrap();
    assert_eq!(c.output_format, OutputFormat::Tensor);

    // Labels are validated but must not change the tensor output.
    let rows = [
        vec![0.0, 0.0, 10.0, 10.0, 0.1, 0.9],
        vec![50.0, 50.0, 60.0, 60.0, 0.8, 0.1],
    ];
    let plain = detect_from_output(&raw_packet(&rows), &cfg(2)).unwrap();
    let mut labeled = cfg(2);
    labeled.labels = Some(names(2));
    labeled.output_format = OutputFormat::Tensor;
    let with_labels = detect_from_output(&raw_packet(&rows), &labeled).unwrap();
    assert_eq!(with_labels.dtype, TensorDtype::Float32);
    assert_eq!(with_labels.shape, plain.shape);
    assert_eq!(with_labels.data, plain.data);

    // And mismatched labels are still rejected under `tensor`.
    labeled.labels = Some(names(3));
    assert!(labeled.validate().is_err());
}

#[test]
fn output_format_parses_and_unknown_value_is_rejected() {
    let c: DetectConfig =
        serde_yaml::from_str("num_classes: 1\nlabels: [a]\noutput_format: json").unwrap();
    assert_eq!(c.output_format, OutputFormat::Json);
    let c: DetectConfig = serde_yaml::from_str("num_classes: 1\noutput_format: tensor").unwrap();
    assert_eq!(c.output_format, OutputFormat::Tensor);
    for bad in ["xml", "JSON", "labeled", "\"\""] {
        let err = serde_yaml::from_str::<DetectConfig>(&format!(
            "num_classes: 1\nlabels: [a]\noutput_format: {bad}"
        ))
        .unwrap_err();
        assert!(err.to_string().contains("unknown variant"), "{bad}: {err}");
    }
}

#[test]
fn json_output_format_requires_labels_at_load() {
    let mut c = cfg(2);
    c.output_format = OutputFormat::Json;
    let err = c.validate().unwrap_err().to_string();
    assert!(err.contains("requires `labels`"), "{err}");
    assert!(
        OnnxDetectBackend::new(Box::new(FixedEngine::new(raw_packet(&[vec![0.0; 7]]))), c).is_err()
    );
}

#[test]
fn json_format_works_with_original_coordinates_end_to_end() {
    let mut c = cfg(1);
    c.labels = Some(vec!["thing".into()]);
    c.output_format = OutputFormat::Json;
    let b = backend_with(
        &[vec![10.0, 35.0, 60.0, 60.0, 0.9]],
        c,
        vision_100(ResizeMode::Letterbox),
    );
    let out = b.infer(&png_request(200, 100)).unwrap();
    let v: serde_json::Value = serde_json::from_slice(&out.data).unwrap();
    assert_eq!(v[0]["box"], serde_json::json!([20.0, 20.0, 120.0, 70.0]));
    assert_eq!(v[0]["label"], "thing");
}
