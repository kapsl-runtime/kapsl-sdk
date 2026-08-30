use std::collections::HashMap;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use kapsl_rag_sdk::manifest::{ConnectorManifest, ConnectorRuntime as ManifestRuntime};
use kapsl_rag_sdk::types::ConnectorConfig;
use serde::Deserialize;
use serde_json::Value;

use crate::runtime::{
    ConnectorClient, ConnectorRuntime as RuntimeTrait, SidecarConnectorRuntime, WasiPermissions,
    WasmConnectorRuntime,
};

#[derive(thiserror::Error, Debug)]
pub enum ExtensionError {
    #[error("io error: {0}")]
    Io(String),
    #[error("manifest not found in {0}")]
    ManifestMissing(String),
    #[error("invalid manifest: {0}")]
    InvalidManifest(String),
    #[error("invalid config: {0}")]
    InvalidConfig(String),
    #[error("extension not installed: {0}")]
    NotInstalled(String),
    #[error("runtime error: {0}")]
    Runtime(String),
}

impl From<io::Error> for ExtensionError {
    fn from(err: io::Error) -> Self {
        ExtensionError::Io(err.to_string())
    }
}

#[derive(Debug, Clone)]
pub struct InstalledExtension {
    pub manifest: ConnectorManifest,
    pub path: PathBuf,
}

#[derive(Debug, Clone)]
pub struct ExtensionRegistry {
    pub root: PathBuf,
}

impl ExtensionRegistry {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    pub fn discover(&self) -> Result<Vec<InstalledExtension>, ExtensionError> {
        let mut extensions = Vec::new();
        if !self.root.exists() {
            return Ok(extensions);
        }
        let mut paths = fs::read_dir(&self.root)?
            .map(|entry| entry.map(|entry| entry.path()))
            .collect::<Result<Vec<_>, _>>()?;
        paths.sort();
        for path in paths {
            if path
                .file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.starts_with('.'))
            {
                continue;
            }
            let metadata = fs::symlink_metadata(&path)?;
            if metadata.file_type().is_symlink() || !metadata.is_dir() {
                continue;
            }
            if let Ok(manifest) = load_manifest(&path) {
                if resolve_manifest_entrypoint(&path, &manifest).is_ok() {
                    extensions.push(InstalledExtension { manifest, path });
                }
            }
        }
        Ok(extensions)
    }

    pub fn install_from_dir(&self, source: &Path) -> Result<InstalledExtension, ExtensionError> {
        let source_metadata = fs::symlink_metadata(source)?;
        if source_metadata.file_type().is_symlink() || !source_metadata.is_dir() {
            return Err(ExtensionError::InvalidConfig(
                "extension source must be a regular directory".to_string(),
            ));
        }
        let manifest = load_manifest(source)?;
        resolve_manifest_entrypoint(source, &manifest)?;
        fs::create_dir_all(&self.root)?;
        let source_canonical = fs::canonicalize(source)?;
        let root_canonical = fs::canonicalize(&self.root)?;
        if root_canonical.starts_with(&source_canonical) {
            return Err(ExtensionError::InvalidConfig(
                "extension source cannot contain the installation root".to_string(),
            ));
        }
        let target = self.root.join(&manifest.id);
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |duration| duration.as_nanos());
        let staging = self.root.join(format!(
            ".{}.installing-{}-{timestamp}",
            manifest.id,
            std::process::id()
        ));
        let backup = self.root.join(format!(
            ".{}.backup-{}-{timestamp}",
            manifest.id,
            std::process::id()
        ));
        if fs::symlink_metadata(&staging).is_ok() || fs::symlink_metadata(&backup).is_ok() {
            return Err(ExtensionError::InvalidConfig(
                "extension staging path already exists".to_string(),
            ));
        }
        let install_result = copy_dir_all(source, &staging);
        if let Err(error) = install_result {
            let _ = fs::remove_dir_all(&staging);
            return Err(error);
        }
        let had_existing_target = match fs::symlink_metadata(&target) {
            Ok(_) => {
                fs::rename(&target, &backup)?;
                true
            }
            Err(error) if error.kind() == io::ErrorKind::NotFound => false,
            Err(error) => return Err(error.into()),
        };
        if let Err(error) = fs::rename(&staging, &target) {
            if had_existing_target {
                let _ = fs::rename(&backup, &target);
            }
            let _ = remove_path(&staging);
            return Err(error.into());
        }
        if had_existing_target {
            remove_path(&backup)?;
        }
        Ok(InstalledExtension {
            manifest,
            path: target,
        })
    }

    pub fn uninstall(&self, extension_id: &str) -> Result<(), ExtensionError> {
        crate::validation::extension_id(extension_id).map_err(ExtensionError::InvalidConfig)?;
        let target = self.root.join(extension_id);
        match fs::symlink_metadata(&target) {
            Ok(_) => remove_path(&target)?,
            Err(error) if error.kind() == io::ErrorKind::NotFound => {
                return Err(ExtensionError::NotInstalled(extension_id.to_string()));
            }
            Err(error) => return Err(error.into()),
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct ExtensionManager {
    pub registry: ExtensionRegistry,
    pub config_root: PathBuf,
}

impl ExtensionManager {
    pub fn new(registry: ExtensionRegistry, config_root: impl Into<PathBuf>) -> Self {
        Self {
            registry,
            config_root: config_root.into(),
        }
    }

    pub fn set_workspace_config(
        &self,
        workspace_id: &str,
        extension_id: &str,
        config: &ConnectorConfig,
    ) -> Result<(), ExtensionError> {
        let path = self.workspace_config_path(workspace_id, extension_id)?;
        fs::create_dir_all(path.parent().expect("config path always has a parent"))?;
        let data = serde_json::to_vec_pretty(config)
            .map_err(|e| ExtensionError::InvalidConfig(e.to_string()))?;
        fs::write(path, data)?;
        Ok(())
    }

    pub fn get_workspace_config(
        &self,
        workspace_id: &str,
        extension_id: &str,
    ) -> Result<Option<ConnectorConfig>, ExtensionError> {
        let path = self.workspace_config_path(workspace_id, extension_id)?;
        let data = match fs::read_to_string(path) {
            Ok(data) => data,
            Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(None),
            Err(error) => return Err(error.into()),
        };
        let config = serde_json::from_str(&data)
            .map_err(|e| ExtensionError::InvalidConfig(e.to_string()))?;
        Ok(Some(config))
    }

    pub fn get_workspace_connector_config(
        &self,
        workspace_id: &str,
        extension_id: &str,
    ) -> Result<Option<ConnectorConfig>, ExtensionError> {
        let config = self.get_workspace_config(workspace_id, extension_id)?;
        Ok(config.map(strip_wasi_block))
    }

    pub fn get_workspace_wasi_permissions(
        &self,
        workspace_id: &str,
        extension_id: &str,
    ) -> Result<WasiPermissions, ExtensionError> {
        let config = self.get_workspace_config(workspace_id, extension_id)?;
        wasi_permissions_from_config(config.as_ref())
    }

    pub fn launch_connector(
        &self,
        workspace_id: &str,
        extension: &InstalledExtension,
    ) -> Result<ConnectorClient<ConnectorRuntimeHandle>, ExtensionError> {
        let entrypoint = resolve_entrypoint(extension)?;
        let runtime = match extension.manifest.runtime {
            ManifestRuntime::Wasm => {
                let permissions =
                    self.get_workspace_wasi_permissions(workspace_id, &extension.manifest.id)?;
                ConnectorRuntimeHandle::Wasm(
                    WasmConnectorRuntime::spawn_with_permissions(&entrypoint, permissions)
                        .map_err(|e| ExtensionError::Runtime(e.to_string()))?,
                )
            }
            ManifestRuntime::Sidecar => {
                let runtime = SidecarConnectorRuntime::spawn(&entrypoint)
                    .map_err(|e| ExtensionError::Runtime(e.to_string()))?;
                ConnectorRuntimeHandle::Sidecar(runtime)
            }
        };
        Ok(ConnectorClient::new(runtime))
    }

    fn workspace_config_path(
        &self,
        workspace_id: &str,
        extension_id: &str,
    ) -> Result<PathBuf, ExtensionError> {
        crate::validation::path_component(workspace_id, "workspace id")
            .map_err(ExtensionError::InvalidConfig)?;
        crate::validation::extension_id(extension_id).map_err(ExtensionError::InvalidConfig)?;
        Ok(self
            .config_root
            .join(workspace_id)
            .join(format!("{extension_id}.json")))
    }
}

pub enum ConnectorRuntimeHandle {
    Wasm(WasmConnectorRuntime),
    Sidecar(SidecarConnectorRuntime),
}

impl RuntimeTrait for ConnectorRuntimeHandle {
    fn send(
        &mut self,
        request: kapsl_rag_sdk::protocol::ConnectorRequest,
    ) -> Result<kapsl_rag_sdk::protocol::ConnectorResponse, crate::runtime::RuntimeError> {
        match self {
            ConnectorRuntimeHandle::Wasm(runtime) => runtime.send(request),
            ConnectorRuntimeHandle::Sidecar(runtime) => runtime.send(request),
        }
    }

    fn close(&mut self) -> Result<(), crate::runtime::RuntimeError> {
        match self {
            ConnectorRuntimeHandle::Wasm(runtime) => runtime.close(),
            ConnectorRuntimeHandle::Sidecar(runtime) => runtime.close(),
        }
    }
}

#[derive(Debug, Deserialize, Default)]
struct WasiConfig {
    #[serde(default)]
    env: HashMap<String, String>,
    #[serde(default)]
    preopen_dirs: Vec<WasiDirConfig>,
}

#[derive(Debug, Deserialize)]
struct WasiDirConfig {
    host_path: String,
    guest_path: String,
    #[serde(default)]
    read_only: bool,
}

fn wasi_permissions_from_config(
    config: Option<&ConnectorConfig>,
) -> Result<WasiPermissions, ExtensionError> {
    let Some(config) = config else {
        return Ok(WasiPermissions::default());
    };
    let obj = match config {
        serde_json::Value::Object(_) => config,
        _ => return Ok(WasiPermissions::default()),
    };

    let Some(wasi_value) = obj.get("wasi") else {
        return Ok(WasiPermissions::default());
    };
    let parsed: WasiConfig = serde_json::from_value(wasi_value.clone())
        .map_err(|e| ExtensionError::InvalidConfig(e.to_string()))?;

    let mut permissions = WasiPermissions::default();
    for (key, value) in parsed.env {
        crate::validation::env_key_value(&key, &value).map_err(ExtensionError::InvalidConfig)?;
        permissions = permissions.with_env(key, value);
    }

    for dir in parsed.preopen_dirs {
        crate::validation::host_path(Path::new(&dir.host_path))
            .map_err(ExtensionError::InvalidConfig)?;
        crate::validation::guest_path(&dir.guest_path).map_err(ExtensionError::InvalidConfig)?;
        permissions =
            permissions.allow_dir(PathBuf::from(dir.host_path), dir.guest_path, dir.read_only);
    }

    Ok(permissions)
}

fn strip_wasi_block(config: ConnectorConfig) -> ConnectorConfig {
    match config {
        Value::Object(mut map) => {
            map.remove("wasi");
            Value::Object(map)
        }
        other => other,
    }
}

fn load_manifest(dir: &Path) -> Result<ConnectorManifest, ExtensionError> {
    let toml_path = dir.join("rag-extension.toml");
    let json_path = dir.join("rag-extension.json");

    if toml_path.exists() {
        let data = fs::read_to_string(&toml_path)?;
        let manifest: ConnectorManifest =
            toml::from_str(&data).map_err(|e| ExtensionError::InvalidManifest(e.to_string()))?;
        validate_manifest(&manifest)?;
        return Ok(manifest);
    }

    if json_path.exists() {
        let data = fs::read_to_string(&json_path)?;
        let manifest: ConnectorManifest = serde_json::from_str(&data)
            .map_err(|e| ExtensionError::InvalidManifest(e.to_string()))?;
        validate_manifest(&manifest)?;
        return Ok(manifest);
    }

    Err(ExtensionError::ManifestMissing(dir.display().to_string()))
}

fn validate_manifest(manifest: &ConnectorManifest) -> Result<(), ExtensionError> {
    crate::validation::extension_id(&manifest.id).map_err(ExtensionError::InvalidManifest)?;
    if manifest.name.trim().is_empty() {
        return Err(ExtensionError::InvalidManifest(
            "connector name cannot be empty".to_string(),
        ));
    }
    if manifest.version.trim().is_empty() {
        return Err(ExtensionError::InvalidManifest(
            "connector version cannot be empty".to_string(),
        ));
    }
    if let Some(entrypoint) = manifest.entrypoint.as_deref() {
        crate::validation::relative_path(entrypoint, "connector entrypoint")
            .map_err(ExtensionError::InvalidManifest)?;
    }
    Ok(())
}

fn resolve_entrypoint(extension: &InstalledExtension) -> Result<PathBuf, ExtensionError> {
    resolve_manifest_entrypoint(&extension.path, &extension.manifest)
}

fn resolve_manifest_entrypoint(
    extension_root: &Path,
    manifest: &ConnectorManifest,
) -> Result<PathBuf, ExtensionError> {
    let default_entry = match manifest.runtime {
        ManifestRuntime::Wasm => "connector.wasm",
        ManifestRuntime::Sidecar => "connector",
    };
    let entry = manifest.entrypoint.as_deref().unwrap_or(default_entry);
    crate::validation::relative_path(entry, "connector entrypoint")
        .map_err(ExtensionError::InvalidConfig)?;
    let resolved = extension_root.join(entry);
    let metadata = fs::symlink_metadata(&resolved).map_err(|error| {
        ExtensionError::InvalidConfig(format!(
            "cannot inspect entrypoint {}: {error}",
            resolved.display()
        ))
    })?;
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        return Err(ExtensionError::InvalidConfig(format!(
            "entrypoint must be a regular file inside the extension: {}",
            resolved.display()
        )));
    }
    Ok(resolved)
}

fn copy_dir_all(src: &Path, dst: &Path) -> Result<(), ExtensionError> {
    fs::create_dir_all(dst)?;
    for entry in fs::read_dir(src)? {
        let entry = entry?;
        let ty = entry.file_type()?;
        let src_path = entry.path();
        let dst_path = dst.join(entry.file_name());
        if ty.is_dir() {
            copy_dir_all(&src_path, &dst_path)?;
        } else if ty.is_file() {
            fs::copy(&src_path, &dst_path)?;
        } else {
            return Err(ExtensionError::InvalidConfig(format!(
                "extension contains unsupported filesystem entry: {}",
                src_path.display()
            )));
        }
    }
    Ok(())
}

fn remove_path(path: &Path) -> Result<(), ExtensionError> {
    let metadata = match fs::symlink_metadata(path) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(()),
        Err(error) => return Err(error.into()),
    };
    if metadata.is_dir() && !metadata.file_type().is_symlink() {
        fs::remove_dir_all(path)?;
    } else {
        fs::remove_file(path)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::sync::atomic::{AtomicU64, Ordering};

    static NEXT_TEST_DIR: AtomicU64 = AtomicU64::new(0);

    struct TestDirectory(PathBuf);

    impl TestDirectory {
        fn new() -> Self {
            let sequence = NEXT_TEST_DIR.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir().join(format!(
                "kapsl-rag-extension-{}-{sequence}",
                std::process::id()
            ));
            fs::create_dir_all(&path).unwrap();
            Self(path)
        }

        fn path(&self) -> &Path {
            &self.0
        }

        fn extension_source(&self, id: &str, entrypoint: &str) -> PathBuf {
            let source = self.0.join("source");
            fs::create_dir_all(&source).unwrap();
            fs::write(
                source.join("rag-extension.toml"),
                format!(
                    "id = {id:?}\nname = \"Test Connector\"\nversion = \"1.0.0\"\nruntime = \"sidecar\"\ncapabilities = [\"sync\"]\nauth = [\"none\"]\npermissions = []\nentrypoint = {entrypoint:?}\n"
                ),
            )
            .unwrap();
            source
        }
    }

    impl Drop for TestDirectory {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn registry_installs_discovers_replaces_and_uninstalls() {
        let directory = TestDirectory::new();
        let source = directory.extension_source("connector.test", "connector");
        fs::write(source.join("connector"), "first").unwrap();
        let registry = ExtensionRegistry::new(directory.path().join("installed"));

        let installed = registry.install_from_dir(&source).unwrap();
        assert_eq!(installed.manifest.id, "connector.test");
        assert_eq!(
            fs::read_to_string(installed.path.join("connector")).unwrap(),
            "first"
        );
        assert_eq!(registry.discover().unwrap().len(), 1);

        fs::write(source.join("connector"), "second").unwrap();
        let replaced = registry.install_from_dir(&source).unwrap();
        assert_eq!(
            fs::read_to_string(replaced.path.join("connector")).unwrap(),
            "second"
        );

        registry.uninstall("connector.test").unwrap();
        assert!(registry.discover().unwrap().is_empty());
    }

    #[test]
    fn registry_rejects_unsafe_manifest_paths() {
        let directory = TestDirectory::new();
        let unsafe_id = directory.extension_source("../escape", "connector");
        fs::write(unsafe_id.join("connector"), "connector").unwrap();
        let registry = ExtensionRegistry::new(directory.path().join("installed"));
        assert!(matches!(
            registry.install_from_dir(&unsafe_id),
            Err(ExtensionError::InvalidManifest(_))
        ));

        let absolute_entrypoint = directory.extension_source("connector.test", "/bin/connector");
        assert!(matches!(
            registry.install_from_dir(&absolute_entrypoint),
            Err(ExtensionError::InvalidManifest(_))
        ));
    }

    #[test]
    fn workspace_config_separates_connector_values_from_wasi_permissions() {
        let directory = TestDirectory::new();
        let manager = ExtensionManager::new(
            ExtensionRegistry::new(directory.path().join("installed")),
            directory.path().join("config"),
        );
        let config = json!({
            "token": "secret",
            "wasi": {
                "env": {"CONNECTOR_MODE": "test"},
                "preopen_dirs": [{
                    "host_path": directory.path().to_string_lossy(),
                    "guest_path": "/data",
                    "read_only": true
                }]
            }
        });

        manager
            .set_workspace_config("workspace", "connector.test", &config)
            .unwrap();

        let connector_config = manager
            .get_workspace_connector_config("workspace", "connector.test")
            .unwrap()
            .unwrap();
        assert_eq!(connector_config, json!({"token": "secret"}));
        let permissions = manager
            .get_workspace_wasi_permissions("workspace", "connector.test")
            .unwrap();
        assert_eq!(
            permissions.env.get("CONNECTOR_MODE").map(String::as_str),
            Some("test")
        );
        assert_eq!(permissions.preopen_dirs.len(), 1);
        assert!(permissions.preopen_dirs[0].read_only);

        assert!(manager
            .set_workspace_config("../escape", "connector.test", &json!({}))
            .is_err());
    }
}
