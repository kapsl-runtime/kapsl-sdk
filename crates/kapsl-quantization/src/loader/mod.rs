use crate::tensor::{AwqTensor, GptqTensor, QuantizedTensor};
use anyhow::{bail, Context, Result};
use half::{bf16, f16};
use safetensors::tensor::TensorView;
use safetensors::{Dtype, SafeTensors};
use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::sync::Arc;

pub trait ModelLoader {
    fn load(&self, model_path: &Path) -> Result<HashMap<String, QuantizedTensor>>;
}

pub mod awq;
pub mod gptq;
pub mod int8;

pub use awq::AwqLoader;
pub use gptq::GptqLoader;
pub use int8::Int8Loader;

#[derive(Debug, Clone, Copy)]
pub(crate) enum QuantizedFormat {
    Awq,
    Gptq,
}

/// Scan every `*.safetensors` file under `model_path` for `<base>.qweight`
/// entries that also carry `<base>.qzeros` and `<base>.scales`, decode the
/// little-endian payloads, and build the requested quantized tensor variant.
///
/// AWQ and GPTQ store weights in the same layout and differ only in which
/// `QuantizedTensor` variant they produce, so both share this walk.
pub(crate) fn load_quantized_safetensors(
    model_path: &Path,
    format: QuantizedFormat,
) -> Result<HashMap<String, QuantizedTensor>> {
    let mut tensors = HashMap::new();

    visit_safetensor_files(model_path, |path, safetensors| {
        for name in safetensors.names() {
            let Some(base_name) = name.strip_suffix(".qweight") else {
                continue;
            };

            let qweight = safetensors.tensor(name)?;
            let shape = qweight.shape().to_vec();
            let qzeros_name = format!("{}.qzeros", base_name);
            let scales_name = format!("{}.scales", base_name);
            let qzeros = safetensors.tensor(&qzeros_name).with_context(|| {
                format!("{name} in {} is missing {qzeros_name}", path.display())
            })?;
            let scales = safetensors.tensor(&scales_name).with_context(|| {
                format!("{name} in {} is missing {scales_name}", path.display())
            })?;
            let qweight = Arc::new(decode_u32_words(&qweight, name)?);
            let qzeros = Arc::new(decode_u32_words(&qzeros, &qzeros_name)?);
            let scales = Arc::new(decode_float_values(&scales, &scales_name)?);
            let tensor = match format {
                QuantizedFormat::Awq => QuantizedTensor::Awq(AwqTensor {
                    qweight,
                    qzeros,
                    scales,
                    shape,
                }),
                QuantizedFormat::Gptq => {
                    let g_idx_name = format!("{}.g_idx", base_name);
                    let bias_name = format!("{}.bias", base_name);
                    let g_idx = safetensors
                        .tensor(&g_idx_name)
                        .ok()
                        .map(|tensor| decode_u32_words(&tensor, &g_idx_name).map(Arc::new))
                        .transpose()?;
                    let bias = safetensors
                        .tensor(&bias_name)
                        .ok()
                        .map(|tensor| decode_float_values(&tensor, &bias_name).map(Arc::new))
                        .transpose()?;
                    QuantizedTensor::Gptq(GptqTensor {
                        qweight,
                        qzeros,
                        scales,
                        g_idx,
                        bias,
                        shape,
                    })
                }
            };

            if tensors.insert(base_name.to_string(), tensor).is_some() {
                bail!(
                    "duplicate quantized tensor {base_name} while loading {}",
                    path.display()
                );
            }
        }
        Ok(())
    })?;

    Ok(tensors)
}

pub(crate) fn visit_safetensor_files(
    model_path: &Path,
    mut visit: impl FnMut(&Path, &SafeTensors<'_>) -> Result<()>,
) -> Result<()> {
    let entries = fs::read_dir(model_path)
        .with_context(|| format!("failed to read model directory {}", model_path.display()))?;
    let mut paths = Vec::new();
    for entry in entries {
        let path = entry
            .with_context(|| format!("failed to read an entry in {}", model_path.display()))?
            .path();
        if path.is_file() && path.extension().is_some_and(|ext| ext == "safetensors") {
            paths.push(path);
        }
    }
    paths.sort();

    for path in paths {
        let file_content =
            fs::read(&path).with_context(|| format!("failed to read {}", path.display()))?;
        let safetensors = SafeTensors::deserialize(&file_content)
            .with_context(|| format!("failed to parse {}", path.display()))?;
        visit(&path, &safetensors)?;
    }
    Ok(())
}

pub(crate) fn decode_u32_words(tensor: &TensorView<'_>, name: &str) -> Result<Vec<u32>> {
    if !matches!(tensor.dtype(), Dtype::I32 | Dtype::U32) {
        bail!(
            "tensor {name} must use I32 or U32 storage, got {:?}",
            tensor.dtype()
        );
    }
    let data = tensor.data();
    if !data.len().is_multiple_of(4) {
        bail!(
            "tensor {name} byte length {} is not divisible by 4",
            data.len()
        );
    }
    Ok(data
        .as_chunks::<4>()
        .0
        .iter()
        .map(|bytes| u32::from_le_bytes(*bytes))
        .collect())
}

pub(crate) fn decode_float_values(tensor: &TensorView<'_>, name: &str) -> Result<Vec<f16>> {
    let data = tensor.data();
    match tensor.dtype() {
        Dtype::F16 => {
            ensure_element_width(data, 2, name)?;
            Ok(data
                .as_chunks::<2>()
                .0
                .iter()
                .map(|bytes| f16::from_le_bytes(*bytes))
                .collect())
        }
        Dtype::BF16 => {
            ensure_element_width(data, 2, name)?;
            Ok(data
                .as_chunks::<2>()
                .0
                .iter()
                .map(|bytes| bf16::from_le_bytes(*bytes))
                .map(|value| f16::from_f32(value.to_f32()))
                .collect())
        }
        Dtype::F32 => {
            ensure_element_width(data, 4, name)?;
            Ok(data
                .as_chunks::<4>()
                .0
                .iter()
                .map(|bytes| f32::from_le_bytes(*bytes))
                .map(f16::from_f32)
                .collect())
        }
        dtype => bail!("tensor {name} must use F16, BF16, or F32 storage, got {dtype:?}"),
    }
}

pub(crate) fn decode_f32_scalar(tensor: &TensorView<'_>, name: &str) -> Result<f32> {
    ensure_scalar(tensor, name)?;
    let data = tensor.data();
    match tensor.dtype() {
        Dtype::F16 => Ok(f16::from_le_bytes([data[0], data[1]]).to_f32()),
        Dtype::BF16 => Ok(bf16::from_le_bytes([data[0], data[1]]).to_f32()),
        Dtype::F32 => Ok(f32::from_le_bytes([data[0], data[1], data[2], data[3]])),
        dtype => bail!("tensor {name} must use F16, BF16, or F32 storage, got {dtype:?}"),
    }
}

pub(crate) fn decode_i32_scalar(tensor: &TensorView<'_>, name: &str) -> Result<i32> {
    ensure_scalar(tensor, name)?;
    let data = tensor.data();
    match tensor.dtype() {
        Dtype::I8 => Ok(i32::from(data[0] as i8)),
        Dtype::U8 => Ok(i32::from(data[0])),
        Dtype::I32 => Ok(i32::from_le_bytes([data[0], data[1], data[2], data[3]])),
        Dtype::U32 => i32::try_from(u32::from_le_bytes([data[0], data[1], data[2], data[3]]))
            .context("unsigned zero point does not fit in i32"),
        dtype => bail!("tensor {name} must use I8, U8, I32, or U32 storage, got {dtype:?}"),
    }
}

fn ensure_element_width(data: &[u8], width: usize, name: &str) -> Result<()> {
    if !data.len().is_multiple_of(width) {
        bail!(
            "tensor {name} byte length {} is not divisible by {width}",
            data.len()
        );
    }
    Ok(())
}

fn ensure_scalar(tensor: &TensorView<'_>, name: &str) -> Result<()> {
    let elements = tensor
        .shape()
        .iter()
        .try_fold(1_usize, |count, dimension| count.checked_mul(*dimension))
        .context("tensor element count overflow")?;
    if elements != 1 {
        bail!("tensor {name} must contain exactly one value");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use safetensors::tensor::{serialize, TensorView};
    use std::collections::BTreeMap;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};

    static NEXT_TEST_DIR: AtomicU64 = AtomicU64::new(0);

    struct TestModelDir(PathBuf);

    impl TestModelDir {
        fn new() -> Self {
            let sequence = NEXT_TEST_DIR.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir().join(format!(
                "kapsl-quantization-{}-{sequence}",
                std::process::id()
            ));
            fs::create_dir_all(&path).expect("create test model directory");
            Self(path)
        }

        fn path(&self) -> &Path {
            &self.0
        }

        fn write(&self, name: &str, tensors: BTreeMap<&str, TensorView<'_>>) {
            let bytes = serialize(tensors, &None).expect("serialize safetensors fixture");
            fs::write(self.0.join(name), bytes).expect("write safetensors fixture");
        }
    }

    impl Drop for TestModelDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn u32_bytes(values: &[u32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect()
    }

    #[test]
    fn gptq_loader_reads_required_and_optional_tensors() {
        let directory = TestModelDir::new();
        let qweight = u32_bytes(&[1, 2]);
        let qzeros = u32_bytes(&[3]);
        let scales = f16::from_f32(0.25).to_le_bytes();
        let g_idx = u32_bytes(&[0, 1]);
        let bias = f16::from_f32(0.5).to_le_bytes();
        let mut tensors = BTreeMap::new();
        tensors.insert(
            "layer.bias",
            TensorView::new(Dtype::F16, vec![1], &bias).unwrap(),
        );
        tensors.insert(
            "layer.g_idx",
            TensorView::new(Dtype::I32, vec![2], &g_idx).unwrap(),
        );
        tensors.insert(
            "layer.qweight",
            TensorView::new(Dtype::I32, vec![2], &qweight).unwrap(),
        );
        tensors.insert(
            "layer.qzeros",
            TensorView::new(Dtype::I32, vec![1], &qzeros).unwrap(),
        );
        tensors.insert(
            "layer.scales",
            TensorView::new(Dtype::F16, vec![1], &scales).unwrap(),
        );
        directory.write("model.safetensors", tensors);

        let loaded = GptqLoader.load(directory.path()).unwrap();
        let QuantizedTensor::Gptq(tensor) = &loaded["layer"] else {
            panic!()
        };

        assert_eq!(tensor.qweight.as_slice(), &[1, 2]);
        assert_eq!(tensor.qzeros.as_slice(), &[3]);
        assert_eq!(
            tensor.g_idx.as_deref().map(Vec::as_slice),
            Some(&[0, 1][..])
        );
        assert_eq!(
            tensor.bias.as_deref().map(Vec::as_slice),
            Some(&[f16::from_f32(0.5)][..])
        );
    }

    #[test]
    fn quantized_loader_rejects_missing_companion_tensor() {
        let directory = TestModelDir::new();
        let qweight = u32_bytes(&[1]);
        let mut tensors = BTreeMap::new();
        tensors.insert(
            "layer.qweight",
            TensorView::new(Dtype::I32, vec![1], &qweight).unwrap(),
        );
        directory.write("model.safetensors", tensors);

        let error = AwqLoader.load(directory.path()).unwrap_err().to_string();

        assert!(error.contains("missing layer.qzeros"));
    }

    #[test]
    fn int8_loader_reads_exact_scalar_metadata() {
        let directory = TestModelDir::new();
        let weights = [254_u8, 3];
        let scale = 0.125_f32.to_le_bytes();
        let zero_point = [7_u8];
        let mut tensors = BTreeMap::new();
        tensors.insert(
            "layer.scale",
            TensorView::new(Dtype::F32, vec![1], &scale).unwrap(),
        );
        tensors.insert(
            "layer.weight",
            TensorView::new(Dtype::I8, vec![2], &weights).unwrap(),
        );
        tensors.insert(
            "layer.zero_point",
            TensorView::new(Dtype::U8, vec![1], &zero_point).unwrap(),
        );
        directory.write("model.safetensors", tensors);

        let loaded = Int8Loader.load(directory.path()).unwrap();
        let QuantizedTensor::Int8(tensor) = &loaded["layer"] else {
            panic!()
        };

        assert_eq!(tensor.weight.as_slice(), &[-2, 3]);
        assert_eq!(tensor.scale, 0.125);
        assert_eq!(tensor.zero_point, 7);
        assert!(!tensor.symmetric);
    }

    #[test]
    fn int8_loader_rejects_non_scalar_scale() {
        let directory = TestModelDir::new();
        let weights = [1_u8];
        let scales = [0.25_f32, 0.5_f32]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect::<Vec<_>>();
        let mut tensors = BTreeMap::new();
        tensors.insert(
            "layer.scale",
            TensorView::new(Dtype::F32, vec![2], &scales).unwrap(),
        );
        tensors.insert(
            "layer.weight",
            TensorView::new(Dtype::I8, vec![1], &weights).unwrap(),
        );
        directory.write("model.safetensors", tensors);

        let error = Int8Loader.load(directory.path()).unwrap_err().to_string();

        assert!(error.contains("exactly one value"));
    }
}
