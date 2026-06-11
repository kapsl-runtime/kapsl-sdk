mod ann;

use ann::ScopeIndex;
pub use ann::DEFAULT_ANN_THRESHOLD;
use async_trait::async_trait;
use rusqlite::{params, Connection, Row};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::sync::Mutex;

#[derive(thiserror::Error, Debug)]
pub enum VectorStoreError {
    #[error("db error: {0}")]
    Db(String),
    #[error("serialization error: {0}")]
    Serialization(String),
    #[error("invalid input: {0}")]
    InvalidInput(String),
}

impl From<rusqlite::Error> for VectorStoreError {
    fn from(err: rusqlite::Error) -> Self {
        VectorStoreError::Db(err.to_string())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AccessControl {
    pub allow_users: Vec<String>,
    pub allow_groups: Vec<String>,
    pub deny_users: Vec<String>,
    pub deny_groups: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddedChunk {
    pub id: String,
    pub tenant_id: String,
    pub workspace_id: String,
    pub source_id: String,
    pub doc_id: String,
    pub chunk_index: i64,
    pub text: String,
    pub embedding: Vec<f32>,
    pub metadata: HashMap<String, String>,
    pub acl: AccessControl,
}

#[derive(Debug, Clone)]
pub struct VectorQuery {
    pub query_embedding: Vec<f32>,
    pub top_k: usize,
    pub tenant_id: String,
    pub workspace_id: String,
    pub source_ids: Option<Vec<String>>,
    pub allowed_users: Vec<String>,
    pub allowed_groups: Vec<String>,
    pub min_score: f32,
}

#[derive(Debug, Clone)]
pub struct VectorSearchResult {
    pub chunk: EmbeddedChunk,
    pub score: f32,
}

#[async_trait]
pub trait VectorStore: Send + Sync {
    async fn upsert(&self, chunks: Vec<EmbeddedChunk>) -> Result<(), VectorStoreError>;
    async fn delete_by_doc(
        &self,
        tenant_id: &str,
        workspace_id: &str,
        source_id: &str,
        doc_id: &str,
    ) -> Result<(), VectorStoreError>;
    async fn query(
        &self,
        request: VectorQuery,
    ) -> Result<Vec<VectorSearchResult>, VectorStoreError>;
}

type ScopeKey = (String, String);

pub struct SqliteVectorStore {
    conn: Mutex<Connection>,
    /// Lazily built HNSW indexes, one per (tenant_id, workspace_id).
    /// Lock order: `indexes` before `conn`; writers release `conn` before
    /// touching `indexes`, so the order never inverts.
    indexes: Mutex<HashMap<ScopeKey, ScopeIndex>>,
    /// Scopes with fewer live chunks than this are scanned exactly.
    ann_threshold: usize,
}

const SELECT_COLUMNS: &str =
    "id, tenant_id, workspace_id, source_id, doc_id, chunk_index, text, embedding,
     metadata_json, acl_allow_users, acl_allow_groups, acl_deny_users, acl_deny_groups";

impl SqliteVectorStore {
    pub fn open(path: &Path) -> Result<Self, VectorStoreError> {
        let conn = Connection::open(path)?;
        let store = Self {
            conn: Mutex::new(conn),
            indexes: Mutex::new(HashMap::new()),
            ann_threshold: DEFAULT_ANN_THRESHOLD,
        };
        store.init()?;
        Ok(store)
    }

    /// Override the scope size above which queries go through the HNSW index.
    pub fn with_ann_threshold(mut self, threshold: usize) -> Self {
        self.ann_threshold = threshold.max(1);
        self
    }

    fn init(&self) -> Result<(), VectorStoreError> {
        let conn = self
            .conn
            .lock()
            .map_err(|_| VectorStoreError::Db("vector store mutex poisoned".to_string()))?;
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS rag_chunks (
                id TEXT PRIMARY KEY,
                tenant_id TEXT NOT NULL,
                workspace_id TEXT NOT NULL,
                source_id TEXT NOT NULL,
                doc_id TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                text TEXT NOT NULL,
                embedding BLOB NOT NULL,
                metadata_json TEXT,
                acl_allow_users TEXT,
                acl_allow_groups TEXT,
                acl_deny_users TEXT,
                acl_deny_groups TEXT,
                updated_at INTEGER
            );
            CREATE INDEX IF NOT EXISTS idx_rag_chunks_scope
                ON rag_chunks (tenant_id, workspace_id, source_id, doc_id);
            ",
        )?;
        Ok(())
    }

    /// Exact scan over every chunk in scope — the recall baseline. Used for
    /// small scopes and as the fallback when ANN candidates are filtered
    /// below `top_k`.
    fn scan_exact(
        &self,
        request: &VectorQuery,
    ) -> Result<Vec<VectorSearchResult>, VectorStoreError> {
        let conn = self
            .conn
            .lock()
            .map_err(|_| VectorStoreError::Db("vector store mutex poisoned".to_string()))?;

        let mut sql = format!(
            "SELECT {SELECT_COLUMNS} FROM rag_chunks WHERE tenant_id = ? AND workspace_id = ?"
        );
        if let Some(source_ids) = &request.source_ids {
            if !source_ids.is_empty() {
                let placeholders = vec!["?"; source_ids.len()].join(", ");
                sql.push_str(" AND source_id IN (");
                sql.push_str(&placeholders);
                sql.push(')');
            }
        }

        let mut stmt = conn.prepare(&sql)?;
        let mut params_vec: Vec<&dyn rusqlite::ToSql> = Vec::new();
        params_vec.push(&request.tenant_id as &dyn rusqlite::ToSql);
        params_vec.push(&request.workspace_id as &dyn rusqlite::ToSql);
        if let Some(source_ids) = &request.source_ids {
            for source_id in source_ids {
                params_vec.push(source_id as &dyn rusqlite::ToSql);
            }
        }

        let mut rows = stmt.query(params_vec.as_slice())?;
        let mut results = Vec::new();
        while let Some(row) = rows.next()? {
            if let Some(result) = score_row(row, request)? {
                results.push(result);
            }
        }

        rank_and_truncate(&mut results, request.top_k);
        Ok(results)
    }

    /// ANN candidate search. Returns `None` when the exact scan should run
    /// instead: scope below threshold, query dimension mismatch, or filters
    /// cut the candidate set below `top_k`.
    fn query_ann(
        &self,
        request: &VectorQuery,
    ) -> Result<Option<Vec<VectorSearchResult>>, VectorStoreError> {
        let key: ScopeKey = (request.tenant_id.clone(), request.workspace_id.clone());
        let mut indexes = self
            .indexes
            .lock()
            .map_err(|_| VectorStoreError::Db("vector index mutex poisoned".to_string()))?;

        if indexes.get(&key).is_some_and(|idx| idx.needs_rebuild()) {
            indexes.remove(&key);
        }

        if !indexes.contains_key(&key) {
            let Some(index) = self.build_scope_index(request)? else {
                return Ok(None);
            };
            indexes.insert(key.clone(), index);
        }
        let index = &indexes[&key];

        if index.dim() != request.query_embedding.len() || index.live_count() < self.ann_threshold
        {
            return Ok(None);
        }

        // Over-fetch so source/ACL/min_score filtering can't silently shrink
        // the result set below top_k without triggering the exact fallback.
        let fetch = request.top_k * 4 + 16;
        let candidate_ids = index.search(&request.query_embedding, fetch);
        if candidate_ids.is_empty() {
            return Ok(None);
        }

        let mut results = self.fetch_and_score(&candidate_ids, request)?;
        if results.len() < request.top_k && results.len() < index.live_count() {
            // Filters were too selective for this candidate set; let the
            // exact scan guarantee recall.
            return Ok(None);
        }
        rank_and_truncate(&mut results, request.top_k);
        Ok(Some(results))
    }

    /// Load every embedding in scope and build its HNSW index. Returns `None`
    /// when the scope is too small to be worth indexing.
    fn build_scope_index(
        &self,
        request: &VectorQuery,
    ) -> Result<Option<ScopeIndex>, VectorStoreError> {
        let conn = self
            .conn
            .lock()
            .map_err(|_| VectorStoreError::Db("vector store mutex poisoned".to_string()))?;

        let count: usize = conn.query_row(
            "SELECT COUNT(*) FROM rag_chunks WHERE tenant_id = ?1 AND workspace_id = ?2",
            params![request.tenant_id, request.workspace_id],
            |row| row.get(0),
        )?;
        if count < self.ann_threshold {
            return Ok(None);
        }

        let mut stmt = conn.prepare(
            "SELECT id, embedding FROM rag_chunks WHERE tenant_id = ?1 AND workspace_id = ?2",
        )?;
        let mut rows = stmt.query(params![request.tenant_id, request.workspace_id])?;
        let mut index = ScopeIndex::new(request.query_embedding.len(), count);
        while let Some(row) = rows.next()? {
            let id: String = row.get(0)?;
            let blob: Vec<u8> = row.get(1)?;
            index.upsert(&id, &deserialize_embedding(&blob));
        }
        log::debug!(
            "[vector] built HNSW index for {}/{}: {} vectors",
            request.tenant_id,
            request.workspace_id,
            index.live_count()
        );
        Ok(Some(index))
    }

    /// Fetch candidate rows by id and apply the same filtering and exact
    /// scoring as the brute-force scan.
    fn fetch_and_score(
        &self,
        candidate_ids: &[String],
        request: &VectorQuery,
    ) -> Result<Vec<VectorSearchResult>, VectorStoreError> {
        let conn = self
            .conn
            .lock()
            .map_err(|_| VectorStoreError::Db("vector store mutex poisoned".to_string()))?;

        let placeholders = vec!["?"; candidate_ids.len()].join(", ");
        let sql = format!(
            "SELECT {SELECT_COLUMNS} FROM rag_chunks
             WHERE tenant_id = ? AND workspace_id = ? AND id IN ({placeholders})"
        );
        let mut stmt = conn.prepare(&sql)?;
        let mut params_vec: Vec<&dyn rusqlite::ToSql> = Vec::new();
        params_vec.push(&request.tenant_id as &dyn rusqlite::ToSql);
        params_vec.push(&request.workspace_id as &dyn rusqlite::ToSql);
        for id in candidate_ids {
            params_vec.push(id as &dyn rusqlite::ToSql);
        }

        let source_filter: Option<HashSet<&String>> = request
            .source_ids
            .as_ref()
            .filter(|ids| !ids.is_empty())
            .map(|ids| ids.iter().collect());

        let mut rows = stmt.query(params_vec.as_slice())?;
        let mut results = Vec::new();
        while let Some(row) = rows.next()? {
            if let Some(filter) = &source_filter {
                let source_id: String = row.get(3)?;
                if !filter.contains(&source_id) {
                    continue;
                }
            }
            if let Some(result) = score_row(row, request)? {
                results.push(result);
            }
        }
        Ok(results)
    }
}

#[async_trait]
impl VectorStore for SqliteVectorStore {
    async fn upsert(&self, chunks: Vec<EmbeddedChunk>) -> Result<(), VectorStoreError> {
        {
            let mut conn = self
                .conn
                .lock()
                .map_err(|_| VectorStoreError::Db("vector store mutex poisoned".to_string()))?;
            let tx = conn.transaction()?;
            for chunk in &chunks {
                let metadata_json = serde_json::to_string(&chunk.metadata)
                    .map_err(|e| VectorStoreError::Serialization(e.to_string()))?;
                let acl_allow_users = serde_json::to_string(&chunk.acl.allow_users)
                    .map_err(|e| VectorStoreError::Serialization(e.to_string()))?;
                let acl_allow_groups = serde_json::to_string(&chunk.acl.allow_groups)
                    .map_err(|e| VectorStoreError::Serialization(e.to_string()))?;
                let acl_deny_users = serde_json::to_string(&chunk.acl.deny_users)
                    .map_err(|e| VectorStoreError::Serialization(e.to_string()))?;
                let acl_deny_groups = serde_json::to_string(&chunk.acl.deny_groups)
                    .map_err(|e| VectorStoreError::Serialization(e.to_string()))?;
                let embedding_blob = serialize_embedding(&chunk.embedding);
                tx.execute(
                    "INSERT OR REPLACE INTO rag_chunks (
                        id, tenant_id, workspace_id, source_id, doc_id, chunk_index,
                        text, embedding, metadata_json, acl_allow_users, acl_allow_groups,
                        acl_deny_users, acl_deny_groups, updated_at
                    ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, strftime('%s','now'))",
                    params![
                        chunk.id,
                        chunk.tenant_id,
                        chunk.workspace_id,
                        chunk.source_id,
                        chunk.doc_id,
                        chunk.chunk_index,
                        chunk.text,
                        embedding_blob,
                        metadata_json,
                        acl_allow_users,
                        acl_allow_groups,
                        acl_deny_users,
                        acl_deny_groups,
                    ],
                )?;
            }
            tx.commit()?;
        }

        // Keep any already-built scope indexes in sync.
        let mut indexes = self
            .indexes
            .lock()
            .map_err(|_| VectorStoreError::Db("vector index mutex poisoned".to_string()))?;
        for chunk in &chunks {
            let key: ScopeKey = (chunk.tenant_id.clone(), chunk.workspace_id.clone());
            if let Some(index) = indexes.get_mut(&key) {
                index.upsert(&chunk.id, &chunk.embedding);
            }
        }
        Ok(())
    }

    async fn delete_by_doc(
        &self,
        tenant_id: &str,
        workspace_id: &str,
        source_id: &str,
        doc_id: &str,
    ) -> Result<(), VectorStoreError> {
        let deleted_ids: Vec<String> = {
            let conn = self
                .conn
                .lock()
                .map_err(|_| VectorStoreError::Db("vector store mutex poisoned".to_string()))?;
            let mut stmt = conn.prepare(
                "SELECT id FROM rag_chunks WHERE tenant_id = ?1 AND workspace_id = ?2 AND source_id = ?3 AND doc_id = ?4",
            )?;
            let ids = stmt
                .query_map(params![tenant_id, workspace_id, source_id, doc_id], |row| {
                    row.get(0)
                })?
                .collect::<Result<Vec<String>, _>>()?;
            conn.execute(
                "DELETE FROM rag_chunks WHERE tenant_id = ?1 AND workspace_id = ?2 AND source_id = ?3 AND doc_id = ?4",
                params![tenant_id, workspace_id, source_id, doc_id],
            )?;
            ids
        };

        if !deleted_ids.is_empty() {
            let mut indexes = self
                .indexes
                .lock()
                .map_err(|_| VectorStoreError::Db("vector index mutex poisoned".to_string()))?;
            let key: ScopeKey = (tenant_id.to_string(), workspace_id.to_string());
            if let Some(index) = indexes.get_mut(&key) {
                for id in &deleted_ids {
                    index.remove(id);
                }
            }
        }
        Ok(())
    }

    async fn query(
        &self,
        request: VectorQuery,
    ) -> Result<Vec<VectorSearchResult>, VectorStoreError> {
        if request.query_embedding.is_empty() {
            return Err(VectorStoreError::InvalidInput(
                "query embedding is empty".to_string(),
            ));
        }
        if let Some(results) = self.query_ann(&request)? {
            return Ok(results);
        }
        self.scan_exact(&request)
    }
}

/// Apply ACL filtering and exact cosine scoring to one row. Returns `None`
/// when the row is filtered out. Column order must match [`SELECT_COLUMNS`].
fn score_row(
    row: &Row<'_>,
    request: &VectorQuery,
) -> Result<Option<VectorSearchResult>, VectorStoreError> {
    let embedding_blob: Vec<u8> = row.get(7)?;
    let embedding = deserialize_embedding(&embedding_blob);
    if embedding.len() != request.query_embedding.len() {
        return Ok(None);
    }

    let acl_allow_users: String = row.get(9)?;
    let acl_allow_groups: String = row.get(10)?;
    let acl_deny_users: String = row.get(11)?;
    let acl_deny_groups: String = row.get(12)?;

    let allow_users: Vec<String> = parse_json_list(&acl_allow_users);
    let allow_groups: Vec<String> = parse_json_list(&acl_allow_groups);
    let deny_users: Vec<String> = parse_json_list(&acl_deny_users);
    let deny_groups: Vec<String> = parse_json_list(&acl_deny_groups);

    let allowed_users: HashSet<String> = request.allowed_users.iter().cloned().collect();
    let allowed_groups: HashSet<String> = request.allowed_groups.iter().cloned().collect();

    if !is_allowed(&allowed_users, &allowed_groups, &allow_users, &allow_groups) {
        return Ok(None);
    }
    if is_denied(&allowed_users, &allowed_groups, &deny_users, &deny_groups) {
        return Ok(None);
    }

    let score = cosine_similarity(&request.query_embedding, &embedding);
    if score < request.min_score {
        return Ok(None);
    }

    let metadata_json: String = row.get(8)?;
    let metadata: HashMap<String, String> =
        serde_json::from_str(&metadata_json).unwrap_or_else(|_| HashMap::new());

    let chunk = EmbeddedChunk {
        id: row.get(0)?,
        tenant_id: row.get(1)?,
        workspace_id: row.get(2)?,
        source_id: row.get(3)?,
        doc_id: row.get(4)?,
        chunk_index: row.get(5)?,
        text: row.get(6)?,
        embedding,
        metadata,
        acl: AccessControl {
            allow_users,
            allow_groups,
            deny_users,
            deny_groups,
        },
    };

    Ok(Some(VectorSearchResult { chunk, score }))
}

fn rank_and_truncate(results: &mut Vec<VectorSearchResult>, top_k: usize) {
    results.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    results.truncate(top_k);
}

fn serialize_embedding(embedding: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(embedding.len() * 4);
    for val in embedding {
        bytes.extend_from_slice(&val.to_le_bytes());
    }
    bytes
}

fn deserialize_embedding(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}

fn parse_json_list(value: &str) -> Vec<String> {
    match serde_json::from_str::<Value>(value) {
        Ok(Value::Array(items)) => items
            .into_iter()
            .filter_map(|v| v.as_str().map(|s| s.to_string()))
            .collect(),
        _ => Vec::new(),
    }
}

fn is_allowed(
    allowed_users: &HashSet<String>,
    allowed_groups: &HashSet<String>,
    acl_users: &[String],
    acl_groups: &[String],
) -> bool {
    if acl_users.is_empty() && acl_groups.is_empty() {
        return true;
    }
    acl_users.iter().any(|u| allowed_users.contains(u))
        || acl_groups.iter().any(|g| allowed_groups.contains(g))
}

fn is_denied(
    allowed_users: &HashSet<String>,
    allowed_groups: &HashSet<String>,
    deny_users: &[String],
    deny_groups: &[String],
) -> bool {
    deny_users.iter().any(|u| allowed_users.contains(u))
        || deny_groups.iter().any(|g| allowed_groups.contains(g))
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0f32;
    let mut norm_a = 0.0f32;
    let mut norm_b = 0.0f32;
    for (x, y) in a.iter().zip(b.iter()) {
        dot += x * y;
        norm_a += x * x;
        norm_b += y * y;
    }
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    dot / (norm_a.sqrt() * norm_b.sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;

    const DIM: usize = 32;

    fn pseudo_vec(seed: u64) -> Vec<f32> {
        let mut x = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (0..DIM)
            .map(|_| {
                x = x
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((x >> 33) as f32 / u32::MAX as f32) - 0.5
            })
            .collect()
    }

    fn chunk(id: &str, doc: &str, embedding: Vec<f32>) -> EmbeddedChunk {
        EmbeddedChunk {
            id: id.to_string(),
            tenant_id: "t1".to_string(),
            workspace_id: "w1".to_string(),
            source_id: "s1".to_string(),
            doc_id: doc.to_string(),
            chunk_index: 0,
            text: format!("text for {id}"),
            embedding,
            metadata: HashMap::new(),
            acl: AccessControl::default(),
        }
    }

    fn query(embedding: Vec<f32>, top_k: usize) -> VectorQuery {
        VectorQuery {
            query_embedding: embedding,
            top_k,
            tenant_id: "t1".to_string(),
            workspace_id: "w1".to_string(),
            source_ids: None,
            allowed_users: vec!["alice".to_string()],
            allowed_groups: Vec::new(),
            min_score: -1.0,
        }
    }

    fn temp_store(name: &str, ann_threshold: usize) -> SqliteVectorStore {
        let path = std::env::temp_dir().join(format!(
            "kapsl_vec_test_{}_{}.sqlite3",
            name,
            std::process::id()
        ));
        let _ = std::fs::remove_file(&path);
        SqliteVectorStore::open(&path)
            .expect("open store")
            .with_ann_threshold(ann_threshold)
    }

    /// 300 random vectors + 1 near-duplicate of the query: the ANN path must
    /// surface the near-duplicate first, with the same exact-cosine score the
    /// brute-force path would give it.
    #[tokio::test]
    async fn ann_path_finds_nearest() {
        let store = temp_store("ann_nearest", 16);
        let target_query = pseudo_vec(999_999);
        let mut near = target_query.clone();
        near[0] += 0.01;

        let mut chunks: Vec<EmbeddedChunk> =
            (0..300).map(|i| chunk(&format!("c{i}"), "doc", pseudo_vec(i))).collect();
        chunks.push(chunk("target", "doc", near.clone()));
        store.upsert(chunks).await.unwrap();

        let results = store.query(query(target_query.clone(), 5)).await.unwrap();
        assert_eq!(results.len(), 5);
        assert_eq!(results[0].chunk.id, "target");
        let exact = cosine_similarity(&target_query, &near);
        assert!((results[0].score - exact).abs() < 1e-6);
    }

    /// ANN and exact scan must agree on the top result for the same data.
    #[tokio::test]
    async fn ann_matches_exact_scan() {
        let store_ann = temp_store("ann_vs_exact_a", 16);
        let store_exact = temp_store("ann_vs_exact_b", usize::MAX);
        let chunks: Vec<EmbeddedChunk> =
            (0..200).map(|i| chunk(&format!("c{i}"), "doc", pseudo_vec(i))).collect();
        store_ann.upsert(chunks.clone()).await.unwrap();
        store_exact.upsert(chunks).await.unwrap();

        let q = pseudo_vec(42); // exact duplicate of c42 — unambiguous winner
        let ann = store_ann.query(query(q.clone(), 1)).await.unwrap();
        let exact = store_exact.query(query(q, 1)).await.unwrap();
        assert_eq!(ann[0].chunk.id, exact[0].chunk.id);
        assert!((ann[0].score - exact[0].score).abs() < 1e-6);
    }

    #[tokio::test]
    async fn delete_tombstones_ann_index() {
        let store = temp_store("ann_delete", 16);
        let target_query = pseudo_vec(777_777);
        let mut chunks: Vec<EmbeddedChunk> =
            (0..100).map(|i| chunk(&format!("c{i}"), "doc", pseudo_vec(i))).collect();
        chunks.push(chunk("target", "target-doc", target_query.clone()));
        store.upsert(chunks).await.unwrap();

        // Build the index, confirm the target is found.
        let results = store.query(query(target_query.clone(), 1)).await.unwrap();
        assert_eq!(results[0].chunk.id, "target");

        store.delete_by_doc("t1", "w1", "s1", "target-doc").await.unwrap();
        let results = store.query(query(target_query, 1)).await.unwrap();
        assert!(results.iter().all(|r| r.chunk.id != "target"));
    }

    #[tokio::test]
    async fn upsert_replaces_embedding_in_ann_index() {
        let store = temp_store("ann_replace", 16);
        let old_emb = pseudo_vec(111_111);
        let new_emb = pseudo_vec(222_222);
        let mut chunks: Vec<EmbeddedChunk> =
            (0..100).map(|i| chunk(&format!("c{i}"), "doc", pseudo_vec(i))).collect();
        chunks.push(chunk("target", "doc", old_emb.clone()));
        store.upsert(chunks).await.unwrap();

        // Build the index against the old embedding.
        let results = store.query(query(old_emb.clone(), 1)).await.unwrap();
        assert_eq!(results[0].chunk.id, "target");

        store.upsert(vec![chunk("target", "doc", new_emb.clone())]).await.unwrap();
        let results = store.query(query(new_emb, 1)).await.unwrap();
        assert_eq!(results[0].chunk.id, "target");
        assert!(results[0].score > 0.999);
    }

    /// A denied chunk must never come back through the ANN path, and the
    /// shortfall must be refilled via the exact fallback.
    #[tokio::test]
    async fn ann_respects_acl_deny() {
        let store = temp_store("ann_acl", 16);
        let target_query = pseudo_vec(555_555);
        let mut denied = chunk("denied", "doc", target_query.clone());
        denied.acl.deny_users = vec!["alice".to_string()];
        let mut chunks: Vec<EmbeddedChunk> =
            (0..100).map(|i| chunk(&format!("c{i}"), "doc", pseudo_vec(i))).collect();
        chunks.push(denied);
        store.upsert(chunks).await.unwrap();

        let results = store.query(query(target_query, 10)).await.unwrap();
        assert_eq!(results.len(), 10);
        assert!(results.iter().all(|r| r.chunk.id != "denied"));
    }

    #[tokio::test]
    async fn small_scope_uses_exact_scan() {
        let store = temp_store("exact_small", 1024);
        let chunks: Vec<EmbeddedChunk> =
            (0..10).map(|i| chunk(&format!("c{i}"), "doc", pseudo_vec(i))).collect();
        store.upsert(chunks).await.unwrap();

        let results = store.query(query(pseudo_vec(3), 3)).await.unwrap();
        assert_eq!(results.len(), 3);
        assert_eq!(results[0].chunk.id, "c3");
        assert!(results[0].score > 0.999);
    }
}
