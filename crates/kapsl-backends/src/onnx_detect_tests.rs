//! Unit tests for the ONNX detection post-processor (box decode + NMS).

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
    bytes_to_f32(&out.data)
        .chunks_exact(6)
        .map(|c| [c[0], c[1], c[2], c[3], c[4], c[5]])
        .collect()
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
