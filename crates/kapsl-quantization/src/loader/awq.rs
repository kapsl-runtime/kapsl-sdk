use super::{load_quantized_safetensors, ModelLoader, QuantizedFormat};
use crate::tensor::QuantizedTensor;
use anyhow::Result;
use std::collections::HashMap;
use std::path::Path;

pub struct AwqLoader;

impl ModelLoader for AwqLoader {
    fn load(&self, model_path: &Path) -> Result<HashMap<String, QuantizedTensor>> {
        load_quantized_safetensors(model_path, QuantizedFormat::Awq)
    }
}
