use super::{decode_f32_scalar, decode_i32_scalar, visit_safetensor_files, ModelLoader};
use crate::tensor::{Int8Tensor, QuantizedTensor};
use anyhow::{bail, Result};
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

pub struct Int8Loader;

impl ModelLoader for Int8Loader {
    fn load(&self, model_path: &Path) -> Result<HashMap<String, QuantizedTensor>> {
        let mut tensors = HashMap::new();

        visit_safetensor_files(model_path, |path, safetensors| {
            for name in safetensors.names() {
                let Some(base_name) = name.strip_suffix(".weight") else {
                    continue;
                };
                let tensor = safetensors.tensor(name)?;
                if tensor.dtype() != safetensors::Dtype::I8 {
                    continue;
                }

                let scale_name = format!("{}.scale", base_name);
                let zero_point_name = format!("{}.zero_point", base_name);
                let scale = safetensors
                    .tensor(&scale_name)
                    .ok()
                    .map(|tensor| decode_f32_scalar(&tensor, &scale_name))
                    .transpose()?
                    .unwrap_or(1.0);
                if !scale.is_finite() || scale <= 0.0 {
                    bail!("tensor {scale_name} must contain a positive finite scale");
                }
                let zero_point = safetensors
                    .tensor(&zero_point_name)
                    .ok()
                    .map(|tensor| decode_i32_scalar(&tensor, &zero_point_name))
                    .transpose()?
                    .unwrap_or(0);
                let quantized = QuantizedTensor::Int8(Int8Tensor {
                    weight: Arc::new(tensor.data().iter().map(|&byte| byte as i8).collect()),
                    scale,
                    zero_point,
                    symmetric: zero_point == 0,
                    shape: tensor.shape().to_vec(),
                });

                if tensors.insert(base_name.to_string(), quantized).is_some() {
                    bail!(
                        "duplicate Int8 tensor {base_name} while loading {}",
                        path.display()
                    );
                }
            }
            Ok(())
        })?;

        Ok(tensors)
    }
}
