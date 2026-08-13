use super::{load_quantized_safetensors, ModelLoader};
use crate::tensor::{GptqTensor, QuantizedTensor};
use anyhow::Result;
use std::collections::HashMap;
use std::path::Path;

pub struct GptqLoader;

impl ModelLoader for GptqLoader {
    fn load(&self, model_path: &Path) -> Result<HashMap<String, QuantizedTensor>> {
        load_quantized_safetensors(model_path, |p| {
            QuantizedTensor::Gptq(GptqTensor {
                qweight: p.qweight,
                qzeros: p.qzeros,
                scales: p.scales,
                // Not present in the base layout; populated by callers that have them.
                g_idx: None,
                bias: None,
                shape: p.shape,
            })
        })
    }
}
