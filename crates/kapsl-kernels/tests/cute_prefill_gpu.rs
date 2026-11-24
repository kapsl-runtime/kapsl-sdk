//! Checks the CuTe prefill-attention kernel on a real GPU: against the scalar
//! NVRTC kernel, and against a CPU reference at a sequence length the scalar
//! kernel cannot run. Ignored by default because it needs an NVIDIA GPU:
//!
//! ```text
//! cargo test -p kapsl-kernels --test cute_prefill_gpu \
//!     --features cute,cudarc/cuda-version-from-build-system -- --ignored
//! ```
//!
//! `tests/cute_host/` checks the same kernel on the CPU, without a GPU.
#![cfg(feature = "cute")]

use cudarc::driver::CudaDevice;
use half::f16;
use kapsl_kernels::cuda_kernels::{
    launch_prefill_attention_cute, launch_prefill_attention_scalar, PrefillAttnParams,
};
use std::sync::Arc;

/// Outputs are O(1) and stored as f16; the CuTe kernel also rounds the
/// attention weights to f16 before multiplying by V.
const TOLERANCE: f32 = 4e-3;

/// Deterministic values in `[-range, range]`.
fn random_f16(len: usize, range: f32, seed: &mut u64) -> Vec<f16> {
    (0..len)
        .map(|_| {
            *seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let unit = (*seed >> 40) as f32 / (1u64 << 24) as f32; // [0, 1)
            f16::from_f32((unit * 2.0 - 1.0) * range)
        })
        .collect()
}

struct Case {
    seq_len: usize,
    num_q_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    q: Vec<f16>,
    k: Vec<f16>,
    v: Vec<f16>,
}

impl Case {
    fn new(seq_len: usize, num_q_heads: usize, num_kv_heads: usize, head_dim: usize) -> Self {
        let mut seed = (seq_len * 31 + head_dim) as u64;
        Self {
            seq_len,
            num_q_heads,
            num_kv_heads,
            head_dim,
            q: random_f16(seq_len * num_q_heads * head_dim, 2.0, &mut seed),
            k: random_f16(seq_len * num_kv_heads * head_dim, 2.0, &mut seed),
            v: random_f16(seq_len * num_kv_heads * head_dim, 1.0, &mut seed),
        }
    }

    fn scale(&self) -> f32 {
        1.0 / (self.head_dim as f32).sqrt()
    }

    /// Runs one of the two kernels. `None` means the CuTe kernel declined
    /// (unsupported device or head dim).
    fn run(&self, device: &Arc<CudaDevice>, cute: bool) -> Option<Vec<f16>> {
        let q = device.htod_sync_copy(&self.q).unwrap();
        let k = device.htod_sync_copy(&self.k).unwrap();
        let v = device.htod_sync_copy(&self.v).unwrap();
        let mut out = device.alloc_zeros::<f16>(self.q.len()).unwrap();
        let mut params = PrefillAttnParams {
            out: &mut out,
            q: &q,
            k: &k,
            v: &v,
            scale: self.scale(),
            seq_len: self.seq_len as u32,
            num_q_heads: self.num_q_heads as u32,
            num_kv_heads: self.num_kv_heads as u32,
            head_dim: self.head_dim as u32,
        };
        if cute {
            if !launch_prefill_attention_cute(device, &mut params).unwrap() {
                return None;
            }
        } else {
            launch_prefill_attention_scalar(device, &mut params).unwrap();
        }
        Some(device.dtoh_sync_copy(&out).unwrap())
    }

    /// Causal attention output for one (row, head), in f64.
    fn reference_row(&self, row: usize, head: usize) -> Vec<f64> {
        let d = self.head_dim;
        let kv_head = head * self.num_kv_heads / self.num_q_heads;
        let q = &self.q[(row * self.num_q_heads + head) * d..][..d];
        let scores: Vec<f64> = (0..=row)
            .map(|j| {
                let k = &self.k[(j * self.num_kv_heads + kv_head) * d..][..d];
                let dot: f64 = q.iter().zip(k).map(|(a, b)| a.to_f64() * b.to_f64()).sum();
                dot * self.scale() as f64
            })
            .collect();
        let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let weights: Vec<f64> = scores.iter().map(|s| (s - max).exp()).collect();
        let sum: f64 = weights.iter().sum();
        let mut out = vec![0.0; d];
        for (j, w) in weights.iter().enumerate() {
            let v = &self.v[(j * self.num_kv_heads + kv_head) * d..][..d];
            for (o, x) in out.iter_mut().zip(v) {
                *o += w / sum * x.to_f64();
            }
        }
        out
    }
}

fn max_abs_diff(a: &[f16], b: &[f16]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x.to_f32() - y.to_f32()).abs())
        .fold(0.0, f32::max)
}

#[test]
#[ignore = "needs an NVIDIA GPU (sm_80 or newer for the CuTe kernel)"]
fn cute_matches_scalar_kernel() {
    let device = CudaDevice::new(0).expect("no CUDA device");
    // (seq_len, q heads, kv heads, head_dim); lengths straddle the 64-row tiles.
    for (seq_len, hq, hkv, d) in [
        (1, 2, 2, 64),
        (63, 4, 2, 64),
        (65, 3, 1, 64),
        (129, 4, 2, 32),
        (200, 2, 2, 96),
        (777, 8, 2, 128),
        (4096, 4, 4, 128),
    ] {
        let case = Case::new(seq_len, hq, hkv, d);
        let Some(cute) = case.run(&device, true) else {
            eprintln!("skipped: the CuTe kernel does not support this device");
            return;
        };
        let scalar = case.run(&device, false).unwrap();
        let diff = max_abs_diff(&cute, &scalar);
        println!("seq_len={seq_len} heads={hq}/{hkv} head_dim={d}: max_abs_diff={diff:.2e}");
        assert!(cute.iter().all(|x| x.is_finite()), "non-finite output");
        assert!(diff < TOLERANCE, "kernels disagree by {diff}");
    }
}

/// The scalar kernel needs shared memory proportional to `seq_len` and cannot
/// launch at this length; the CuTe kernel must still be right.
#[test]
#[ignore = "needs an NVIDIA GPU (sm_80 or newer for the CuTe kernel)"]
fn cute_handles_long_prompts() {
    let device = CudaDevice::new(0).expect("no CUDA device");
    let case = Case::new(16_384, 2, 1, 64);
    let Some(out) = case.run(&device, true) else {
        eprintln!("skipped: the CuTe kernel does not support this device");
        return;
    };
    assert!(out.iter().all(|x| x.is_finite()), "non-finite output");

    let d = case.head_dim;
    for row in [0, 63, 64, 5_000, 16_320, 16_383] {
        for head in 0..case.num_q_heads {
            let expected = case.reference_row(row, head);
            let got = &out[(row * case.num_q_heads + head) * d..][..d];
            let diff = got
                .iter()
                .zip(&expected)
                .map(|(g, e)| (g.to_f64() - e).abs())
                .fold(0.0, f64::max);
            assert!(
                diff < TOLERANCE as f64,
                "row {row} head {head}: off by {diff}"
            );
        }
    }
}
