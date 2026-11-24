//! Opt-in end-to-end run of a real YOLOX-nano model through the real
//! `BackendFactory` pipeline (`onnx -> vision preprocess -> detect`), built from
//! a package manifest exactly as the engine does. Ignored by default because the
//! weights and sample images are not committed (YOLOX is Apache-2.0; see
//! `claude/vision-detect-report.md` for provenance and how to produce them).
//!
//! ```text
//! KAPSL_DETECT_E2E_DIR=/path/to/vision-detect \
//!   cargo test -p kapsl-backends --test onnx_detect_real_model -- --ignored --nocapture
//! ```
//!
//! The directory must hold `model/yolox_nano_decoded.onnx` (BGR input, decoded
//! boxes), `model/coco80_labels.json` and `images/{bus,dog,pizza}.jpg`. Raw
//! results are written to `$KAPSL_DETECT_E2E_DIR/out/<image>.sdk.json`.

use kapsl_backends::{BackendFactory, OnnxRuntimeTuning};
use kapsl_core::Manifest;
use kapsl_engine_api::{BinaryTensorPacket, InferenceRequest, TensorDtype};
use kapsl_hal::device::{Device, DeviceBackend, DeviceInfo};
use serde_json::{json, Value};
use std::path::PathBuf;

fn cpu_only_device_info() -> DeviceInfo {
    DeviceInfo {
        cpu_cores: 8,
        total_memory: 16 * 1024 * 1024 * 1024,
        os_type: "test".to_string(),
        os_release: "0".to_string(),
        has_cuda: false,
        has_metal: false,
        has_rocm: false,
        has_directml: false,
        devices: vec![Device {
            id: 0,
            name: "CPU".to_string(),
            backend: DeviceBackend::Cpu,
            memory_mb: 16 * 1024,
            compute_units: 8,
            pci_bus_id: None,
            driver_version: None,
            compute_capability: None,
            utilization_gpu_pct: None,
            temperature_c: None,
            supports_fp16: false,
            supports_int8: false,
            cuda_version: None,
            partition_id: None,
        }],
    }
}

fn manifest(labels: &Value) -> Manifest {
    serde_json::from_value(json!({
        "project_name": "yolox-nano",
        "framework": "onnx",
        "version": "1.0.0",
        "created_at": "0",
        "model_file": "yolox_nano_decoded.onnx",
        "format": "onnx",
        "model_type": "detector",
        "task": "detect",
        "metadata": {
            "detect": {
                "num_classes": 80, "objectness": true, "transposed": false,
                "box_format": "xywh", "score_threshold": 0.25, "iou_threshold": 0.45,
                "max_detections": 100, "class_agnostic": true, "labels": labels,
                "output_format": "json"
            },
            "preprocess": {
                "kind": "vision", "width": 416, "height": 416, "resize": "letterbox",
                "layout": "nchw", "scale": 1.0, "mean": [0, 0, 0], "std": [1, 1, 1],
                "pad": 114, "channel_order": "bgr"
            }
        },
        "hardware_requirements": { "preferred_provider": "cpu" }
    }))
    .expect("manifest")
}

/// Expected top detection for one sample image (original-image pixels).
struct Case {
    name: &'static str,
    size: (f64, f64),
    top_label: &'static str,
    min_score: f64,
    top_box: [f64; 4],
}

#[tokio::test]
#[ignore = "needs YOLOX weights and sample images; set KAPSL_DETECT_E2E_DIR"]
async fn yolox_nano_detects_real_images_in_original_coordinates_with_labels() {
    let root = PathBuf::from(std::env::var("KAPSL_DETECT_E2E_DIR").expect("KAPSL_DETECT_E2E_DIR"));
    let labels: Value =
        serde_json::from_slice(&std::fs::read(root.join("model/coco80_labels.json")).unwrap())
            .unwrap();
    let mut engine = BackendFactory::create_best_backend_with_tuning(
        &manifest(&labels),
        &cpu_only_device_info(),
        &OnnxRuntimeTuning::default(),
    )
    .expect("factory pipeline");
    engine
        .load(&root.join("model/yolox_nano_decoded.onnx"))
        .await
        .expect("load");

    let cases = [
        Case {
            name: "bus",
            size: (1280.0, 850.0),
            top_label: "person",
            min_score: 0.7,
            top_box: [1080.0, 470.0, 1195.0, 745.0],
        },
        Case {
            name: "dog",
            size: (1280.0, 960.0),
            top_label: "dog",
            min_score: 0.8,
            top_box: [35.0, 0.0, 1011.0, 777.0],
        },
        Case {
            name: "pizza",
            size: (1280.0, 960.0),
            top_label: "pizza",
            min_score: 0.85,
            top_box: [315.0, 465.0, 1077.0, 750.0],
        },
    ];
    for Case {
        name,
        size: (w, h),
        top_label,
        min_score,
        top_box: want_box,
    } in cases
    {
        let bytes = std::fs::read(root.join(format!("images/{name}.jpg"))).unwrap();
        let req = InferenceRequest::new(
            BinaryTensorPacket::new(vec![bytes.len() as i64], TensorDtype::Uint8, bytes).unwrap(),
        );
        let out = engine.infer(&req).expect("infer");
        assert_eq!(
            out.dtype,
            TensorDtype::Utf8,
            "{name}: labels => JSON output"
        );
        let _ = std::fs::write(root.join(format!("out/{name}.sdk.json")), &out.data);
        let dets: Vec<Value> = serde_json::from_slice(&out.data).unwrap();
        assert!(!dets.is_empty(), "{name}: no detections");

        for d in &dets {
            let b: Vec<f64> = d["box"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_f64().unwrap())
                .collect();
            assert!(
                b[0] >= 0.0 && b[1] >= 0.0 && b[2] <= w && b[3] <= h,
                "{name}: box outside image: {b:?}"
            );
            assert!(b[2] > b[0] && b[3] > b[1], "{name}: degenerate box {b:?}");
            let id = d["class_id"].as_u64().unwrap() as usize;
            assert_eq!(
                d["label"], labels[id],
                "{name}: label does not match class_id"
            );
        }

        let top = &dets[0];
        assert_eq!(top["label"], top_label, "{name}: top detection is {top}");
        assert!(
            top["score"].as_f64().unwrap() >= min_score,
            "{name}: score {top}"
        );
        let b: Vec<f64> = top["box"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect();
        for (got, want) in b.iter().zip(want_box) {
            assert!(
                (got - want).abs() < 15.0,
                "{name}: box {b:?} differs from expected {want_box:?}"
            );
        }
    }
}
