//! Shared NVRTC compilation settings for the runtime-compiled CUDA kernels.

use cudarc::driver::{CudaFunction, CudaStream, DriverError};
use cudarc::nvrtc::{CompileOptions, Ptx};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, Weak};

/// NVRTC options pointing at the local CUDA headers.
///
/// Honours `CUDA_PATH` then `CUDA_HOME`, falling back to the default
/// `/usr/local/cuda` install prefix.
pub(crate) fn cuda_compile_opts() -> CompileOptions {
    let cuda_include = std::env::var("CUDA_PATH")
        .or_else(|_| std::env::var("CUDA_HOME"))
        .map(|p| format!("{p}/include"))
        .unwrap_or_else(|_| "/usr/local/cuda/include".to_string());
    CompileOptions {
        include_paths: vec![cuda_include],
        ..Default::default()
    }
}

/// Functions loaded through one device handle, keyed by (module, function).
struct DeviceModules {
    device: Weak<CudaStream>,
    functions: HashMap<(&'static str, &'static str), CudaFunction>,
}

// cudarc 0.12 kept loaded modules on the `CudaDevice` and unloaded them with
// it; 0.19 hands them to the caller. Keep them per device handle here and
// release them once that handle is gone.
static MODULES: Mutex<Vec<DeviceModules>> = Mutex::new(Vec::new());

fn is_device(entry: &DeviceModules, device: &Arc<CudaStream>) -> bool {
    std::ptr::eq(entry.device.as_ptr(), Arc::as_ptr(device))
}

/// Counterpart of cudarc 0.12's `CudaDevice::get_func`.
pub(crate) fn get_func(
    device: &Arc<CudaStream>,
    module_name: &'static str,
    func_name: &'static str,
) -> Option<CudaFunction> {
    let modules = MODULES.lock().unwrap();
    modules
        .iter()
        .find(|entry| is_device(entry, device))
        .and_then(|entry| entry.functions.get(&(module_name, func_name)).cloned())
}

/// Counterpart of cudarc 0.12's `CudaDevice::load_ptx`.
pub(crate) fn load_ptx(
    device: &Arc<CudaStream>,
    ptx: Ptx,
    module_name: &'static str,
    func_names: &[&'static str],
) -> Result<(), DriverError> {
    let module = device.context().load_module(ptx)?;
    let mut loaded = Vec::with_capacity(func_names.len());
    for &func_name in func_names {
        loaded.push(((module_name, func_name), module.load_function(func_name)?));
    }

    let stale: Vec<DeviceModules>;
    {
        let mut modules = MODULES.lock().unwrap();
        stale = modules
            .extract_if(.., |entry| entry.device.strong_count() == 0)
            .collect();
        let index = match modules.iter().position(|entry| is_device(entry, device)) {
            Some(index) => index,
            None => {
                modules.push(DeviceModules {
                    device: Arc::downgrade(device),
                    functions: HashMap::new(),
                });
                modules.len() - 1
            }
        };
        modules[index].functions.extend(loaded);
    }
    // Unloading a module binds its CUDA context; do it outside the lock.
    drop(stale);
    Ok(())
}
