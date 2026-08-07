use super::{load_quantized_safetensors, ModelLoader};
use crate::tensor::{AwqTensor, QuantizedTensor};
use anyhow::Result;
use std::collections::HashMap;
use std::path::Path;

pub struct AwqLoader;

impl ModelLoader for AwqLoader {
    fn load(&self, model_path: &Path) -> Result<HashMap<String, QuantizedTensor>> {
        load_quantized_safetensors(model_path, |p| {
            QuantizedTensor::Awq(AwqTensor {
                qweight: p.qweight,
                qzeros: p.qzeros,
                scales: p.scales,
                shape: p.shape,
            })
        })
    }
}
