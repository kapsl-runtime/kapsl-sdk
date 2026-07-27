use std::env;
use std::path::{Path, PathBuf};

pub const PROVIDER_PATH_ENV: &str = "KAPSL_PROVIDER_PATH";
pub const ALLOW_UNMANAGED_PROVIDERS_ENV: &str = "KAPSL_ALLOW_UNMANAGED_PROVIDERS";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AcceleratorProviderPack {
    Cuda,
    TensorRt,
    Rocm,
}

impl AcceleratorProviderPack {
    fn directory_prefix(self) -> &'static str {
        match self {
            Self::Cuda => "cuda",
            Self::TensorRt => "tensorrt",
            Self::Rocm => "rocm",
        }
    }

    fn manifest_prefix(self) -> &'static str {
        match self {
            Self::Cuda => "kapsl-provider-cuda",
            Self::TensorRt => "kapsl-provider-tensorrt",
            Self::Rocm => "kapsl-provider-rocm",
        }
    }

    pub fn display_name(self) -> &'static str {
        match self {
            Self::Cuda => "CUDA",
            Self::TensorRt => "TensorRT",
            Self::Rocm => "ROCm",
        }
    }
}

pub fn accelerator_provider_pack_installed(provider: AcceleratorProviderPack) -> bool {
    if read_env_flag(ALLOW_UNMANAGED_PROVIDERS_ENV) {
        return true;
    }

    accelerator_provider_pack_installed_in(provider, &provider_search_roots())
}

fn provider_search_roots() -> Vec<PathBuf> {
    let mut roots = Vec::new();

    if let Some(paths) = env::var_os(PROVIDER_PATH_ENV) {
        roots.extend(env::split_paths(&paths));
    }

    if let Ok(executable) = env::current_exe() {
        if let Some(parent) = executable.parent() {
            roots.push(parent.to_path_buf());
            roots.push(parent.join("providers"));
            roots.push(parent.join("resources").join("runtime"));
        }
    }

    roots.sort();
    roots.dedup();
    roots
}

fn accelerator_provider_pack_installed_in(
    provider: AcceleratorProviderPack,
    roots: &[PathBuf],
) -> bool {
    let provider_present = roots
        .iter()
        .any(|root| root_has_provider_manifest(root, provider));

    if provider != AcceleratorProviderPack::TensorRt {
        return provider_present;
    }

    provider_present
        && roots
            .iter()
            .any(|root| root_has_provider_manifest(root, AcceleratorProviderPack::Cuda))
}

fn root_has_provider_manifest(root: &Path, provider: AcceleratorProviderPack) -> bool {
    if directory_has_provider_manifest(root, provider) {
        return true;
    }

    let Ok(entries) = std::fs::read_dir(root) else {
        return false;
    };

    entries.filter_map(Result::ok).any(|entry| {
        let path = entry.path();
        path.is_dir()
            && entry
                .file_name()
                .to_string_lossy()
                .to_ascii_lowercase()
                .starts_with(provider.directory_prefix())
            && directory_has_provider_manifest(&path, provider)
    })
}

fn directory_has_provider_manifest(directory: &Path, provider: AcceleratorProviderPack) -> bool {
    let Ok(entries) = std::fs::read_dir(directory) else {
        return false;
    };

    entries.filter_map(Result::ok).any(|entry| {
        let path = entry.path();
        if !path.is_file() {
            return false;
        }
        let name = entry.file_name().to_string_lossy().to_ascii_lowercase();
        name.starts_with(provider.manifest_prefix()) && name.ends_with(".json")
    })
}

fn read_env_flag(name: &str) -> bool {
    env::var(name)
        .ok()
        .map(|value| {
            matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        })
        .unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::{accelerator_provider_pack_installed_in, AcceleratorProviderPack};

    #[test]
    fn cuda_pack_requires_its_manifest() {
        let temp = tempfile::tempdir().expect("temp provider root");
        let roots = vec![temp.path().to_path_buf()];

        assert!(!accelerator_provider_pack_installed_in(
            AcceleratorProviderPack::Cuda,
            &roots
        ));

        std::fs::write(
            temp.path().join("kapsl-provider-cuda12.json"),
            br#"{"provider":"cuda"}"#,
        )
        .expect("write CUDA marker");

        assert!(accelerator_provider_pack_installed_in(
            AcceleratorProviderPack::Cuda,
            &roots
        ));
    }

    #[test]
    fn versioned_provider_subdirectories_are_discovered() {
        let temp = tempfile::tempdir().expect("temp provider root");
        let cuda_dir = temp.path().join("cuda12");
        std::fs::create_dir(&cuda_dir).expect("create CUDA provider directory");
        std::fs::write(
            cuda_dir.join("kapsl-provider-cuda12.json"),
            br#"{"provider":"cuda"}"#,
        )
        .expect("write CUDA marker");

        assert!(accelerator_provider_pack_installed_in(
            AcceleratorProviderPack::Cuda,
            &[temp.path().to_path_buf()]
        ));
    }

    #[test]
    fn tensorrt_pack_also_requires_cuda_pack() {
        let temp = tempfile::tempdir().expect("temp provider root");
        let roots = vec![temp.path().to_path_buf()];
        std::fs::write(
            temp.path().join("kapsl-provider-tensorrt10.json"),
            br#"{"provider":"tensorrt"}"#,
        )
        .expect("write TensorRT marker");

        assert!(!accelerator_provider_pack_installed_in(
            AcceleratorProviderPack::TensorRt,
            &roots
        ));

        std::fs::write(
            temp.path().join("kapsl-provider-cuda12.json"),
            br#"{"provider":"cuda"}"#,
        )
        .expect("write CUDA marker");

        assert!(accelerator_provider_pack_installed_in(
            AcceleratorProviderPack::TensorRt,
            &roots
        ));
    }
}
