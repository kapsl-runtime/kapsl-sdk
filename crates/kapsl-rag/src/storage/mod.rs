use std::fs;
use std::io;
use std::path::PathBuf;

#[derive(thiserror::Error, Debug)]
pub enum DocStoreError {
    #[error("io error: {0}")]
    Io(String),
}

impl From<io::Error> for DocStoreError {
    fn from(err: io::Error) -> Self {
        DocStoreError::Io(err.to_string())
    }
}

#[derive(Debug, Clone)]
pub struct DocKey {
    pub tenant_id: String,
    pub workspace_id: String,
    pub source_id: String,
    pub doc_id: String,
}

pub trait DocStore: Send + Sync {
    fn put(&self, key: &DocKey, bytes: &[u8]) -> Result<PathBuf, DocStoreError>;
    fn get(&self, key: &DocKey) -> Result<Vec<u8>, DocStoreError>;
    fn delete(&self, key: &DocKey) -> Result<(), DocStoreError>;
}

#[derive(Debug, Clone)]
pub struct FsDocStore {
    pub root: PathBuf,
}

impl FsDocStore {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    fn path_for(&self, key: &DocKey) -> Result<PathBuf, DocStoreError> {
        for (value, label) in [
            (&key.tenant_id, "tenant id"),
            (&key.workspace_id, "workspace id"),
            (&key.source_id, "source id"),
        ] {
            crate::validation::path_component(value, label).map_err(DocStoreError::Io)?;
        }
        crate::validation::relative_path(&key.doc_id, "document id").map_err(DocStoreError::Io)?;
        Ok(self
            .root
            .join(&key.tenant_id)
            .join(&key.workspace_id)
            .join(&key.source_id)
            .join(&key.doc_id))
    }
}

impl DocStore for FsDocStore {
    fn put(&self, key: &DocKey, bytes: &[u8]) -> Result<PathBuf, DocStoreError> {
        let path = self.path_for(key)?;
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(&path, bytes)?;
        Ok(path)
    }

    fn get(&self, key: &DocKey) -> Result<Vec<u8>, DocStoreError> {
        let path = self.path_for(key)?;
        let data = fs::read(path)?;
        Ok(data)
    }

    fn delete(&self, key: &DocKey) -> Result<(), DocStoreError> {
        let path = self.path_for(key)?;
        match fs::remove_file(path) {
            Ok(()) => {}
            Err(error) if error.kind() == io::ErrorKind::NotFound => {}
            Err(error) => return Err(error.into()),
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    static NEXT_TEST_DIR: AtomicU64 = AtomicU64::new(0);

    fn store() -> FsDocStore {
        let sequence = NEXT_TEST_DIR.fetch_add(1, Ordering::Relaxed);
        FsDocStore::new(std::env::temp_dir().join(format!(
            "kapsl-rag-doc-store-{}-{sequence}",
            std::process::id()
        )))
    }

    fn key(doc_id: &str) -> DocKey {
        DocKey {
            tenant_id: "tenant".to_string(),
            workspace_id: "workspace".to_string(),
            source_id: "source".to_string(),
            doc_id: doc_id.to_string(),
        }
    }

    #[test]
    fn document_store_roundtrips_nested_document_ids() {
        let store = store();
        let key = key("folder/document.txt");

        let path = store.put(&key, b"document").unwrap();

        assert!(path.starts_with(&store.root));
        assert_eq!(store.get(&key).unwrap(), b"document");
        store.delete(&key).unwrap();
        assert!(!path.exists());
        let _ = fs::remove_dir_all(&store.root);
    }

    #[test]
    fn document_store_rejects_path_traversal() {
        let store = store();

        assert!(store.put(&key("../../escape"), b"document").is_err());
    }
}
