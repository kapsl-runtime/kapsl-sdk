#[cfg(test)]
mod tests {
    use super::super::{LoaderError, Manifest, PackageLoader};
    use crate::HardwareRequirements;
    use flate2::write::GzEncoder;
    use flate2::Compression;
    use std::fs::File;
    use std::path::{Path, PathBuf};
    use std::sync::{Mutex, OnceLock};
    use tar::Builder;
    use tempfile::TempDir;

    fn build_package(entries: Vec<(&str, Vec<u8>)>) -> (TempDir, PathBuf) {
        let temp_dir = tempfile::tempdir().expect("create temp dir");
        let tar_path = temp_dir.path().join("package.tar.gz");
        let tar_file = File::create(&tar_path).expect("create tar.gz");
        let encoder = GzEncoder::new(tar_file, Compression::default());
        let mut builder = Builder::new(encoder);

        for (path, data) in entries {
            let mut header = tar::Header::new_gnu();
            header.set_path(path).expect("set path");
            header.set_size(data.len() as u64);
            header.set_mode(0o644);
            header.set_cksum();
            builder.append(&header, data.as_slice()).expect("append");
        }

        let encoder = builder.into_inner().expect("finish tar builder");
        encoder.finish().expect("finish gzip");

        (temp_dir, tar_path)
    }

    fn default_manifest(model_file: &str) -> Manifest {
        Manifest {
            project_name: "test-project".to_string(),
            framework: "onnx".to_string(),
            version: "1.0.0".to_string(),
            created_at: "2024-01-01T00:00:00Z".to_string(),
            model_file: model_file.to_string(),
            format: None,
            model_type: None,
            task: None,
            metadata: None,
            hardware_requirements: HardwareRequirements::default(),
            cron_jobs: Vec::new(),
        }
    }

    fn env_lock() -> &'static Mutex<()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
    }

    fn clear_model_cache_env() {
        for key in [
            "KAPSL_MODEL_CACHE_DIR",
            "KAPSL_MODEL_CACHE_MAX_BYTES",
            "KAPSL_MODEL_CACHE_MAX_MIB",
            "KAPSL_MODEL_CACHE_RESERVED_FREE_BYTES",
            "KAPSL_MODEL_CACHE_RESERVED_FREE_MIB",
            "KAPSL_LITE_MODEL_CACHE_DIR",
            "KAPSL_LITE_MODEL_CACHE_MAX_BYTES",
            "KAPSL_LITE_MODEL_CACHE_MAX_MIB",
            "KAPSL_LITE_MODEL_CACHE_RESERVED_FREE_BYTES",
            "KAPSL_LITE_MODEL_CACHE_RESERVED_FREE_MIB",
            "KAPSL_DISCARD_PACKAGE_AFTER_LOAD",
            "KAPSL_LITE_DISCARD_PACKAGE_AFTER_LOAD",
        ] {
            std::env::remove_var(key);
        }
    }

    /// Builds a package at a caller-chosen path so a test can watch what
    /// happens to the file itself.
    fn build_package_at(package_path: &Path, model_bytes: Vec<u8>) {
        let manifest = default_manifest("model.onnx");
        let manifest_bytes = serde_json::to_vec(&manifest).expect("serialize manifest");
        let tar_file = File::create(package_path).expect("create tar.gz");
        let encoder = GzEncoder::new(tar_file, Compression::default());
        let mut builder = Builder::new(encoder);

        for (path, data) in [
            ("metadata.json", manifest_bytes),
            ("model.onnx", model_bytes),
        ] {
            let mut header = tar::Header::new_gnu();
            header.set_path(path).expect("set path");
            header.set_size(data.len() as u64);
            header.set_mode(0o644);
            header.set_cksum();
            builder.append(&header, data.as_slice()).expect("append");
        }

        let encoder = builder.into_inner().expect("finish tar builder");
        encoder.finish().expect("finish gzip");
    }

    #[test]
    fn test_load_success() {
        let _guard = env_lock().lock().expect("acquire env lock");
        clear_model_cache_env();

        let manifest = default_manifest("model.onnx");
        let manifest_bytes = serde_json::to_vec(&manifest).expect("serialize manifest");
        let model_bytes = vec![1u8, 2u8, 3u8];

        let (_temp_dir, package_path) = build_package(vec![
            ("metadata.json", manifest_bytes),
            ("model.onnx", model_bytes),
        ]);

        let loader = PackageLoader::load(&package_path).expect("load package");
        assert_eq!(loader.manifest.project_name, "test-project");
        assert_eq!(loader.manifest.model_file, "model.onnx");

        let model_path = loader.get_model_path();
        assert!(Path::new(&model_path).exists());
    }

    #[test]
    fn test_load_missing_manifest() {
        let (_temp_dir, package_path) = build_package(vec![("model.onnx", vec![0u8])]);

        let res = PackageLoader::load(&package_path);
        assert!(matches!(res, Err(LoaderError::ManifestMissing)));
    }

    #[test]
    fn test_load_invalid_manifest_json() {
        let (_temp_dir, package_path) =
            build_package(vec![("metadata.json", b"{not-json}".to_vec())]);

        let res = PackageLoader::load(&package_path);
        assert!(matches!(res, Err(LoaderError::Json(_))));
    }

    #[test]
    fn test_load_persists_model_into_cache_dir_override() {
        let _guard = env_lock().lock().expect("acquire env lock");
        clear_model_cache_env();

        let package_dir = tempfile::tempdir().expect("create package dir");
        let cache_dir = tempfile::tempdir().expect("create cache dir");
        std::env::set_var("KAPSL_MODEL_CACHE_DIR", cache_dir.path());

        let manifest = default_manifest("model.onnx");
        let manifest_bytes = serde_json::to_vec(&manifest).expect("serialize manifest");
        let package_path = package_dir.path().join("with-sidecar.aimod");
        let tar_file = File::create(&package_path).expect("create tar.gz");
        let encoder = GzEncoder::new(tar_file, Compression::default());
        let mut builder = Builder::new(encoder);

        for (path, data) in [
            ("metadata.json", manifest_bytes),
            ("model.onnx", vec![7u8; 1024]),
            ("model.onnx_data", vec![9u8; 256]),
        ] {
            let mut header = tar::Header::new_gnu();
            header.set_path(path).expect("set path");
            header.set_size(data.len() as u64);
            header.set_mode(0o644);
            header.set_cksum();
            builder.append(&header, data.as_slice()).expect("append");
        }
        let encoder = builder.into_inner().expect("finish tar builder");
        encoder.finish().expect("finish gzip");

        let loader = PackageLoader::load(&package_path).expect("load package");
        let model_path = loader.get_model_path();
        assert!(
            model_path.starts_with(cache_dir.path()),
            "model path should be in cache dir override"
        );
        assert!(model_path.exists(), "cached model should exist");
        let sidecar = model_path
            .parent()
            .expect("cache dir exists")
            .join("model.onnx_data");
        assert!(sidecar.exists(), "sidecar asset should also be copied");

        clear_model_cache_env();
    }

    #[test]
    fn test_cache_evicts_old_entries_when_over_max_bytes() {
        let _guard = env_lock().lock().expect("acquire env lock");
        clear_model_cache_env();

        let cache_dir = tempfile::tempdir().expect("create cache dir");
        std::env::set_var("KAPSL_MODEL_CACHE_DIR", cache_dir.path());
        std::env::set_var("KAPSL_MODEL_CACHE_MAX_BYTES", "2048");

        let create_pkg = |name: &str, byte: u8| -> (TempDir, PathBuf) {
            let temp_dir = tempfile::tempdir().expect("create temp dir");
            let package_path = temp_dir.path().join(format!("{name}.aimod"));
            let tar_file = File::create(&package_path).expect("create tar.gz");
            let encoder = GzEncoder::new(tar_file, Compression::default());
            let mut builder = Builder::new(encoder);
            let manifest = Manifest {
                project_name: name.to_string(),
                framework: "onnx".to_string(),
                version: "1.0.0".to_string(),
                created_at: "2024-01-01T00:00:00Z".to_string(),
                model_file: "model.onnx".to_string(),
                format: None,
                model_type: None,
                task: None,
                metadata: None,
                hardware_requirements: HardwareRequirements::default(),
                cron_jobs: Vec::new(),
            };
            let manifest_bytes = serde_json::to_vec(&manifest).expect("serialize manifest");
            for (path, data) in [
                ("metadata.json", manifest_bytes),
                ("model.onnx", vec![byte; 1024]),
            ] {
                let mut header = tar::Header::new_gnu();
                header.set_path(path).expect("set path");
                header.set_size(data.len() as u64);
                header.set_mode(0o644);
                header.set_cksum();
                builder.append(&header, data.as_slice()).expect("append");
            }
            let encoder = builder.into_inner().expect("finish tar builder");
            encoder.finish().expect("finish gzip");
            (temp_dir, package_path)
        };

        let (_pkg1_dir, pkg1) = create_pkg("p1", 1);
        let loader1 = PackageLoader::load(&pkg1).expect("load first package");
        let first_cached_path = loader1.get_model_path();
        assert!(first_cached_path.exists(), "first cache entry should exist");

        let (_pkg2_dir, pkg2) = create_pkg("p2", 2);
        let loader2 = PackageLoader::load(&pkg2).expect("load second package");
        let second_cached_path = loader2.get_model_path();
        assert!(
            second_cached_path.exists(),
            "second cache entry should exist"
        );

        assert!(
            !first_cached_path.exists(),
            "oldest cache entry should be evicted to satisfy max size"
        );

        clear_model_cache_env();
    }

    // Caching the model must not spend its bytes a second time. The container
    // case is the one that hurts: a writable layer is often RAM, so a copy of
    // the weights is charged against the same budget as the weights.
    #[cfg(unix)]
    #[test]
    fn test_cached_model_is_hard_linked_to_the_extracted_file() {
        use std::os::unix::fs::MetadataExt;

        let _guard = env_lock().lock().expect("acquire env lock");
        clear_model_cache_env();

        let package_dir = tempfile::tempdir().expect("create package dir");
        let package_path = package_dir.path().join("model.aimod");
        build_package_at(&package_path, vec![3u8; 4096]);

        let loader = PackageLoader::load(&package_path).expect("load package");
        let cached = loader.get_model_path();
        let cached_meta = std::fs::metadata(&cached).expect("stat cached model");

        assert_eq!(
            cached_meta.nlink(),
            2,
            "cached model should share an inode with the extracted file, not duplicate it"
        );

        let extracted_meta =
            std::fs::metadata(loader.extracted_path.join("model.onnx")).expect("stat extracted");
        assert_eq!(
            (cached_meta.dev(), cached_meta.ino()),
            (extracted_meta.dev(), extracted_meta.ino()),
            "cached model should be the same inode as the extracted file"
        );
    }

    // The link is only sound if the bytes outlive the temp directory that
    // supplied them, which is deleted as soon as the loader is dropped.
    #[test]
    fn test_cached_model_outlives_the_extracted_package() {
        let _guard = env_lock().lock().expect("acquire env lock");
        clear_model_cache_env();

        let package_dir = tempfile::tempdir().expect("create package dir");
        let package_path = package_dir.path().join("model.aimod");
        build_package_at(&package_path, vec![5u8; 2048]);

        let loader = PackageLoader::load(&package_path).expect("load package");
        let cached = loader.get_model_path();
        let extracted = loader.extracted_path.clone();
        drop(loader);

        assert!(
            !extracted.exists(),
            "the package temp directory should be cleaned up on drop"
        );
        assert_eq!(
            std::fs::read(&cached).expect("read cached model"),
            vec![5u8; 2048],
            "cached model should still be readable once the extract is gone"
        );
    }

    // Deleting the caller's own file is never the default: for anyone running
    // `kapsl run --model ./my-model.aimod` the package is theirs to keep.
    #[test]
    fn test_package_is_kept_after_load_by_default() {
        let _guard = env_lock().lock().expect("acquire env lock");
        clear_model_cache_env();

        let package_dir = tempfile::tempdir().expect("create package dir");
        let package_path = package_dir.path().join("model.aimod");
        build_package_at(&package_path, vec![1u8; 512]);

        let _loader = PackageLoader::load(&package_path).expect("load package");

        assert!(package_path.exists(), "package should be left alone");
    }

    // A runtime that pulled the package solely to load it, onto a filesystem
    // that is RAM, can say so.
    #[test]
    fn test_package_is_discarded_after_load_when_enabled() {
        let _guard = env_lock().lock().expect("acquire env lock");
        clear_model_cache_env();
        std::env::set_var("KAPSL_DISCARD_PACKAGE_AFTER_LOAD", "1");

        let package_dir = tempfile::tempdir().expect("create package dir");
        let package_path = package_dir.path().join("model.aimod");
        build_package_at(&package_path, vec![7u8; 1024]);

        let loader = PackageLoader::load(&package_path).expect("load package");
        let cached = loader.get_model_path();

        assert!(!package_path.exists(), "package should be discarded");
        assert_eq!(
            std::fs::read(&cached).expect("read cached model"),
            vec![7u8; 1024],
            "discarding the package must not disturb the cached model"
        );

        clear_model_cache_env();
    }

    #[test]
    fn test_load_errors_when_reserved_free_space_unavailable() {
        let _guard = env_lock().lock().expect("acquire env lock");
        clear_model_cache_env();

        let cache_dir = tempfile::tempdir().expect("create cache dir");
        std::env::set_var("KAPSL_MODEL_CACHE_DIR", cache_dir.path());
        std::env::set_var(
            "KAPSL_MODEL_CACHE_RESERVED_FREE_BYTES",
            u64::MAX.to_string(),
        );

        let manifest = default_manifest("model.onnx");
        let manifest_bytes = serde_json::to_vec(&manifest).expect("serialize manifest");
        let (_temp_dir, package_path) = build_package(vec![
            ("metadata.json", manifest_bytes),
            ("model.onnx", vec![1u8; 128]),
        ]);

        let result = PackageLoader::load(&package_path);
        assert!(
            matches!(result, Err(LoaderError::InsufficientDiskSpace { .. })),
            "reserved free space guard should reject load when impossible"
        );

        clear_model_cache_env();
    }

    fn manifest_with_metadata(metadata: &str) -> Manifest {
        let mut manifest = default_manifest("model.onnx");
        manifest.metadata = Some(serde_yaml::from_str(metadata).expect("parse metadata"));
        manifest
    }

    #[test]
    fn preprocess_kind_reads_the_front_end_a_package_asks_for() {
        let vision = manifest_with_metadata("preprocess:\n  kind: vision\n  width: 224\n");
        assert_eq!(vision.preprocess_kind().as_deref(), Some("vision"));

        let audio = manifest_with_metadata("preprocess:\n  kind: audio\n  n_mels: 80\n");
        assert_eq!(audio.preprocess_kind().as_deref(), Some("audio"));
    }

    // A package with no preprocessing takes tensors directly, and one with a
    // malformed block must not be reported as having a front-end it lacks.
    #[test]
    fn preprocess_kind_is_absent_without_a_usable_block() {
        assert_eq!(default_manifest("model.onnx").preprocess_kind(), None);
        assert_eq!(
            manifest_with_metadata("labels: [cat, dog]\n").preprocess_kind(),
            None
        );
        assert_eq!(
            manifest_with_metadata("preprocess:\n  width: 224\n").preprocess_kind(),
            None
        );
    }
}
