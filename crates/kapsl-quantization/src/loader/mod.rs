use crate::tensor::QuantizedTensor;
use anyhow::{Context, Result};
use half::f16;
use safetensors::SafeTensors;
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

/// The three parallel arrays a `.qweight` group decodes to, plus its shape.
pub(crate) struct QuantParts {
    pub qweight: Arc<Vec<u32>>,
    pub qzeros: Arc<Vec<u32>>,
    pub scales: Arc<Vec<f16>>,
    pub shape: Vec<usize>,
}

/// Scan every `*.safetensors` file under `model_path` for `<base>.qweight`
/// entries that also carry `<base>.qzeros` and `<base>.scales`, decode the
/// little-endian payloads, and hand each group to `build`.
///
/// AWQ and GPTQ store weights in the same layout and differ only in which
/// `QuantizedTensor` variant they produce, so both share this walk.
pub(crate) fn load_quantized_safetensors(
    model_path: &Path,
    build: impl Fn(QuantParts) -> QuantizedTensor,
) -> Result<HashMap<String, QuantizedTensor>> {
    let mut tensors = HashMap::new();

    let entries = fs::read_dir(model_path).context("Failed to read model directory")?;
    for entry in entries {
        let entry = entry?;
        let path = entry.path();
        if path.extension().is_none_or(|ext| ext != "safetensors") {
            continue;
        }

        let file_content = fs::read(&path)?;
        let safetensors = SafeTensors::deserialize(&file_content)?;

        for name in safetensors.names() {
            let Some(base_name) = name.strip_suffix(".qweight") else {
                continue;
            };

            let qweight = safetensors.tensor(name)?;
            let qzeros_name = format!("{}.qzeros", base_name);
            let scales_name = format!("{}.scales", base_name);
            let (Ok(qzeros), Ok(scales)) = (
                safetensors.tensor(&qzeros_name),
                safetensors.tensor(&scales_name),
            ) else {
                continue;
            };

            let parts = QuantParts {
                qweight: Arc::new(u32_le(qweight.data())),
                qzeros: Arc::new(u32_le(qzeros.data())),
                scales: Arc::new(
                    scales
                        .data()
                        .chunks_exact(2)
                        .map(|c| f16::from_le_bytes(c.try_into().unwrap()))
                        .collect(),
                ),
                shape: qweight.shape().to_vec(),
            };

            tensors.insert(base_name.to_string(), build(parts));
        }
    }

    Ok(tensors)
}

fn u32_le(data: &[u8]) -> Vec<u32> {
    data.chunks_exact(4)
        .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}
