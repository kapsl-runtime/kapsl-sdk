//! Unit tests for the ONNX CTC transcription post-processor.

use super::*;

/// Build a `[time, vocab]` f32 logit packet from per-frame rows.
fn logits(rows: &[Vec<f32>]) -> BinaryTensorPacket {
    let time = rows.len() as i64;
    let vocab = rows[0].len() as i64;
    let flat: Vec<f32> = rows.iter().flatten().copied().collect();
    let mut data = Vec::with_capacity(flat.len() * 4);
    for v in &flat {
        data.extend_from_slice(&v.to_le_bytes());
    }
    BinaryTensorPacket {
        shape: vec![time, vocab],
        dtype: TensorDtype::Float32,
        data,
    }
}

/// One-hot-ish frame: put the peak on `id` within `vocab` classes.
fn frame(id: usize, vocab: usize) -> Vec<f32> {
    let mut v = vec![0.0; vocab];
    v[id] = 1.0;
    v
}

fn decode_ids(out: &BinaryTensorPacket) -> Vec<i32> {
    assert_eq!(out.dtype, TensorDtype::Int32);
    out.data
        .chunks_exact(4)
        .map(|b| i32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

fn cfg() -> TranscribeConfig {
    TranscribeConfig::default() // blank_id 0, collapse_repeats true
}

#[test]
fn collapses_repeats_and_drops_blank() {
    // vocab 4, blank=0. Frames: h h _ e l l l _ o  (h=1,e=2,l=3,o... use ids)
    // Sequence of frame argmaxes: 1 1 0 2 3 3 3 0 1 -> tokens 1,2,3,1
    let v = 4;
    let out = transcribe_from_output(
        &logits(&[
            frame(1, v),
            frame(1, v),
            frame(0, v), // blank
            frame(2, v),
            frame(3, v),
            frame(3, v),
            frame(3, v),
            frame(0, v), // blank
            frame(1, v),
        ]),
        &cfg(),
    )
    .unwrap();
    assert_eq!(decode_ids(&out), vec![1, 2, 3, 1]);
}

#[test]
fn repeat_separated_by_blank_yields_two_tokens() {
    // 1 1 0 1 -> collapse gives [1, (blank), 1] -> tokens 1,1 (the blank resets).
    let v = 3;
    let out = transcribe_from_output(
        &logits(&[frame(1, v), frame(1, v), frame(0, v), frame(1, v)]),
        &cfg(),
    )
    .unwrap();
    assert_eq!(decode_ids(&out), vec![1, 1]);
}

#[test]
fn adjacent_repeat_without_blank_is_single_token() {
    // 2 2 2 -> single token 2.
    let v = 3;
    let out =
        transcribe_from_output(&logits(&[frame(2, v), frame(2, v), frame(2, v)]), &cfg()).unwrap();
    assert_eq!(decode_ids(&out), vec![2]);
}

#[test]
fn collapse_disabled_keeps_raw_nonblank_argmax() {
    let mut c = cfg();
    c.collapse_repeats = false;
    let v = 3;
    // 2 2 0 2 -> without collapse, drop only blanks -> 2,2,2
    let out = transcribe_from_output(
        &logits(&[frame(2, v), frame(2, v), frame(0, v), frame(2, v)]),
        &c,
    )
    .unwrap();
    assert_eq!(decode_ids(&out), vec![2, 2, 2]);
}

#[test]
fn non_zero_blank_id() {
    // blank at last index (vocab 3 -> blank 2). Frames 0 2 1 -> tokens 0,1.
    let mut c = cfg();
    c.blank_id = 2;
    let v = 3;
    let out =
        transcribe_from_output(&logits(&[frame(0, v), frame(2, v), frame(1, v)]), &c).unwrap();
    assert_eq!(decode_ids(&out), vec![0, 1]);
}

#[test]
fn all_blank_yields_empty() {
    let v = 3;
    let out =
        transcribe_from_output(&logits(&[frame(0, v), frame(0, v), frame(0, v)]), &cfg()).unwrap();
    assert_eq!(out.shape, vec![0]);
    assert!(decode_ids(&out).is_empty());
}

#[test]
fn accepts_3d_batch_one_shape() {
    let v = 3;
    let mut p = logits(&[frame(1, v), frame(2, v)]);
    p.shape = vec![1, 2, 3];
    let out = transcribe_from_output(&p, &cfg()).unwrap();
    assert_eq!(decode_ids(&out), vec![1, 2]);
}

#[test]
fn blank_id_out_of_range_errors() {
    let mut c = cfg();
    c.blank_id = 9;
    let v = 3;
    assert!(transcribe_from_output(&logits(&[frame(1, v)]), &c).is_err());
}

#[test]
fn wrong_dtype_errors() {
    let bad = BinaryTensorPacket {
        shape: vec![1, 3],
        dtype: TensorDtype::Int32,
        data: vec![0; 12],
    };
    assert!(transcribe_from_output(&bad, &cfg()).is_err());
}
