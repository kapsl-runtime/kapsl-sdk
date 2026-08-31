mod ann;

use ann::ScopeIndex;
pub use ann::DEFAULT_ANN_THRESHOLD;
use async_trait::async_trait;
use rusqlite::{params, Connection, Row};
use serde::{Deserialize, Serialize};
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

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct AccessControl {
    pub allow_users: Vec<String>,
    pub allow_groups: Vec<String>,
    pub deny_users: Vec<String>,
    pub deny_groups: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
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

#[derive(Debug, Clone, PartialEq)]
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

#[derive(Debug, Clone, PartialEq)]
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
    "storage_id, id, tenant_id, workspace_id, source_id, doc_id, chunk_index, text, embedding,
     metadata_json, acl_allow_users, acl_allow_groups, acl_deny_users, acl_deny_groups";
const MAX_SOURCE_FILTERS: usize = 512;
// Stay below SQLite's conservative 999 bound-parameter limit after the two
// scope parameters are included.
const MAX_ANN_CANDIDATES: usize = 900;

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
        let mut conn = self
            .conn
            .lock()
            .map_err(|_| VectorStoreError::Db("vector store mutex poisoned".to_string()))?;
        let table_exists: bool = conn.query_row(
            "SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'rag_chunks')",
            [],
            |row| row.get(0),
        )?;
        if table_exists && !table_has_column(&conn, "rag_chunks", "storage_id")? {
            migrate_legacy_schema(&mut conn)?;
        }
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS rag_chunks (
                storage_id TEXT PRIMARY KEY,
                id TEXT NOT NULL,
                tenant_id TEXT NOT NULL,
                workspace_id TEXT NOT NULL,
                source_id TEXT NOT NULL,
                doc_id TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                text TEXT NOT NULL,
                embedding BLOB NOT NULL,
                metadata_json TEXT NOT NULL,
                acl_allow_users TEXT NOT NULL,
                acl_allow_groups TEXT NOT NULL,
                acl_deny_users TEXT NOT NULL,
                acl_deny_groups TEXT NOT NULL,
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
        let principal = QueryPrincipal::new(request);
        while let Some(row) = rows.next()? {
            if let Some(result) = score_row(row, request, &principal)? {
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

        if index.dim() != request.query_embedding.len() || index.live_count() < self.ann_threshold {
            return Ok(None);
        }

        // Over-fetch so source/ACL/min_score filtering can't silently shrink
        // the result set below top_k without triggering the exact fallback.
        let fetch = request
            .top_k
            .saturating_mul(4)
            .saturating_add(16)
            .min(index.live_count())
            .min(MAX_ANN_CANDIDATES);
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
            "SELECT storage_id, embedding FROM rag_chunks WHERE tenant_id = ?1 AND workspace_id = ?2",
        )?;
        let mut rows = stmt.query(params![request.tenant_id, request.workspace_id])?;
        let mut index = ScopeIndex::new(request.query_embedding.len(), count);
        while let Some(row) = rows.next()? {
            let id: String = row.get(0)?;
            let blob: Vec<u8> = row.get(1)?;
            let embedding = deserialize_embedding(&blob)?;
            index.upsert(&id, &embedding);
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
             WHERE tenant_id = ? AND workspace_id = ? AND storage_id IN ({placeholders})"
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
        let principal = QueryPrincipal::new(request);
        while let Some(row) = rows.next()? {
            if let Some(filter) = &source_filter {
                let source_id: String = row.get(4)?;
                if !filter.contains(&source_id) {
                    continue;
                }
            }
            if let Some(result) = score_row(row, request, &principal)? {
                results.push(result);
            }
        }
        Ok(results)
    }
}

fn table_has_column(
    conn: &Connection,
    table: &str,
    expected_column: &str,
) -> Result<bool, VectorStoreError> {
    let mut statement = conn.prepare(&format!("PRAGMA table_info({table})"))?;
    let mut rows = statement.query([])?;
    while let Some(row) = rows.next()? {
        let column: String = row.get(1)?;
        if column == expected_column {
            return Ok(true);
        }
    }
    Ok(false)
}

fn migrate_legacy_schema(conn: &mut Connection) -> Result<(), VectorStoreError> {
    let transaction = conn.transaction()?;
    transaction.execute_batch(
        "CREATE TABLE rag_chunks_v2 (
            storage_id TEXT PRIMARY KEY,
            id TEXT NOT NULL,
            tenant_id TEXT NOT NULL,
            workspace_id TEXT NOT NULL,
            source_id TEXT NOT NULL,
            doc_id TEXT NOT NULL,
            chunk_index INTEGER NOT NULL,
            text TEXT NOT NULL,
            embedding BLOB NOT NULL,
            metadata_json TEXT NOT NULL,
            acl_allow_users TEXT NOT NULL,
            acl_allow_groups TEXT NOT NULL,
            acl_deny_users TEXT NOT NULL,
            acl_deny_groups TEXT NOT NULL,
            updated_at INTEGER
        );
        INSERT INTO rag_chunks_v2 (
            storage_id, id, tenant_id, workspace_id, source_id, doc_id, chunk_index,
            text, embedding, metadata_json, acl_allow_users, acl_allow_groups,
            acl_deny_users, acl_deny_groups, updated_at
        )
        SELECT
            lower(hex(tenant_id)) || ':' || lower(hex(workspace_id)) || ':' ||
                lower(hex(source_id)) || ':' || lower(hex(id)),
            id, tenant_id, workspace_id, source_id, doc_id, chunk_index, text, embedding,
            coalesce(metadata_json, '{}'), coalesce(acl_allow_users, '[]'),
            coalesce(acl_allow_groups, '[]'), coalesce(acl_deny_users, '[]'),
            coalesce(acl_deny_groups, '[]'), updated_at
        FROM rag_chunks;
        DROP TABLE rag_chunks;
        ALTER TABLE rag_chunks_v2 RENAME TO rag_chunks;",
    )?;
    transaction.commit()?;
    Ok(())
}

fn storage_id(chunk: &EmbeddedChunk) -> String {
    storage_id_for(
        &chunk.tenant_id,
        &chunk.workspace_id,
        &chunk.source_id,
        &chunk.id,
    )
}

fn storage_id_for(tenant_id: &str, workspace_id: &str, source_id: &str, id: &str) -> String {
    [tenant_id, workspace_id, source_id, id]
        .into_iter()
        .map(hex_string)
        .collect::<Vec<_>>()
        .join(":")
}

fn hex_string(value: &str) -> String {
    let mut encoded = String::with_capacity(value.len().saturating_mul(2));
    for byte in value.as_bytes() {
        use std::fmt::Write as _;
        let _ = write!(encoded, "{byte:02x}");
    }
    encoded
}

#[async_trait]
impl VectorStore for SqliteVectorStore {
    async fn upsert(&self, chunks: Vec<EmbeddedChunk>) -> Result<(), VectorStoreError> {
        for chunk in &chunks {
            validate_chunk(chunk)?;
        }
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
                let storage_id = storage_id(chunk);
                tx.execute(
                    "INSERT OR REPLACE INTO rag_chunks (
                        storage_id, id, tenant_id, workspace_id, source_id, doc_id, chunk_index,
                        text, embedding, metadata_json, acl_allow_users, acl_allow_groups,
                        acl_deny_users, acl_deny_groups, updated_at
                    ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, strftime('%s','now'))",
                    params![
                        storage_id,
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
                index.upsert(&storage_id(chunk), &chunk.embedding);
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
                "SELECT storage_id FROM rag_chunks WHERE tenant_id = ?1 AND workspace_id = ?2 AND source_id = ?3 AND doc_id = ?4",
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
        validate_query(&request)?;
        if let Some(results) = self.query_ann(&request)? {
            return Ok(results);
        }
        self.scan_exact(&request)
    }
}

fn validate_chunk(chunk: &EmbeddedChunk) -> Result<(), VectorStoreError> {
    for (value, label) in [
        (&chunk.id, "chunk id"),
        (&chunk.tenant_id, "tenant id"),
        (&chunk.workspace_id, "workspace id"),
        (&chunk.source_id, "source id"),
        (&chunk.doc_id, "document id"),
    ] {
        if value.trim().is_empty() {
            return Err(VectorStoreError::InvalidInput(format!(
                "{label} cannot be empty"
            )));
        }
    }
    validate_embedding(&chunk.embedding, "chunk embedding")
}

fn validate_query(request: &VectorQuery) -> Result<(), VectorStoreError> {
    validate_embedding(&request.query_embedding, "query embedding")?;
    if request.top_k == 0 {
        return Err(VectorStoreError::InvalidInput(
            "top_k must be greater than zero".to_string(),
        ));
    }
    if !request.min_score.is_finite() {
        return Err(VectorStoreError::InvalidInput(
            "min_score must be finite".to_string(),
        ));
    }
    for (value, label) in [
        (&request.tenant_id, "tenant id"),
        (&request.workspace_id, "workspace id"),
    ] {
        if value.trim().is_empty() {
            return Err(VectorStoreError::InvalidInput(format!(
                "{label} cannot be empty"
            )));
        }
    }
    if let Some(source_ids) = &request.source_ids {
        if source_ids.len() > MAX_SOURCE_FILTERS {
            return Err(VectorStoreError::InvalidInput(format!(
                "source filter exceeds {MAX_SOURCE_FILTERS} entries"
            )));
        }
        if source_ids
            .iter()
            .any(|source_id| source_id.trim().is_empty())
        {
            return Err(VectorStoreError::InvalidInput(
                "source filter contains an empty id".to_string(),
            ));
        }
    }
    Ok(())
}

fn validate_embedding(embedding: &[f32], label: &str) -> Result<(), VectorStoreError> {
    if embedding.is_empty() {
        return Err(VectorStoreError::InvalidInput(format!(
            "{label} cannot be empty"
        )));
    }
    if !embedding.iter().all(|value| value.is_finite()) {
        return Err(VectorStoreError::InvalidInput(format!(
            "{label} must contain only finite values"
        )));
    }
    Ok(())
}

struct QueryPrincipal<'a> {
    users: HashSet<&'a str>,
    groups: HashSet<&'a str>,
}

impl<'a> QueryPrincipal<'a> {
    fn new(request: &'a VectorQuery) -> Self {
        Self {
            users: request.allowed_users.iter().map(String::as_str).collect(),
            groups: request.allowed_groups.iter().map(String::as_str).collect(),
        }
    }
}

/// Apply ACL filtering and exact cosine scoring to one row. Returns `None`
/// when the row is filtered out. Column order must match [`SELECT_COLUMNS`].
fn score_row(
    row: &Row<'_>,
    request: &VectorQuery,
    principal: &QueryPrincipal<'_>,
) -> Result<Option<VectorSearchResult>, VectorStoreError> {
    let embedding_blob: Vec<u8> = row.get(8)?;
    let embedding = deserialize_embedding(&embedding_blob)?;
    if embedding.len() != request.query_embedding.len() {
        return Ok(None);
    }

    let acl_allow_users: String = row.get(10)?;
    let acl_allow_groups: String = row.get(11)?;
    let acl_deny_users: String = row.get(12)?;
    let acl_deny_groups: String = row.get(13)?;

    let allow_users = parse_json_list(&acl_allow_users, "acl_allow_users")?;
    let allow_groups = parse_json_list(&acl_allow_groups, "acl_allow_groups")?;
    let deny_users = parse_json_list(&acl_deny_users, "acl_deny_users")?;
    let deny_groups = parse_json_list(&acl_deny_groups, "acl_deny_groups")?;

    if !is_allowed(
        &principal.users,
        &principal.groups,
        &allow_users,
        &allow_groups,
    ) {
        return Ok(None);
    }
    if is_denied(
        &principal.users,
        &principal.groups,
        &deny_users,
        &deny_groups,
    ) {
        return Ok(None);
    }

    let score = cosine_similarity(&request.query_embedding, &embedding);
    if score < request.min_score {
        return Ok(None);
    }

    let metadata_json: String = row.get(9)?;
    let metadata: HashMap<String, String> = serde_json::from_str(&metadata_json)
        .map_err(|error| VectorStoreError::Serialization(error.to_string()))?;

    let chunk = EmbeddedChunk {
        id: row.get(1)?,
        tenant_id: row.get(2)?,
        workspace_id: row.get(3)?,
        source_id: row.get(4)?,
        doc_id: row.get(5)?,
        chunk_index: row.get(6)?,
        text: row.get(7)?,
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
    let mut bytes = Vec::with_capacity(embedding.len().saturating_mul(4));
    for val in embedding {
        bytes.extend_from_slice(&val.to_le_bytes());
    }
    bytes
}

fn deserialize_embedding(bytes: &[u8]) -> Result<Vec<f32>, VectorStoreError> {
    if !bytes.len().is_multiple_of(4) {
        return Err(VectorStoreError::Serialization(format!(
            "embedding byte length {} is not divisible by four",
            bytes.len()
        )));
    }
    let embedding = bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|chunk| f32::from_le_bytes(*chunk))
        .collect::<Vec<_>>();
    validate_embedding(&embedding, "stored embedding")?;
    Ok(embedding)
}

fn parse_json_list(value: &str, column: &str) -> Result<Vec<String>, VectorStoreError> {
    serde_json::from_str(value)
        .map_err(|error| VectorStoreError::Serialization(format!("invalid {column}: {error}")))
}

fn is_allowed(
    allowed_users: &HashSet<&str>,
    allowed_groups: &HashSet<&str>,
    acl_users: &[String],
    acl_groups: &[String],
) -> bool {
    if acl_users.is_empty() && acl_groups.is_empty() {
        return true;
    }
    acl_users
        .iter()
        .any(|user| allowed_users.contains(user.as_str()))
        || acl_groups
            .iter()
            .any(|group| allowed_groups.contains(group.as_str()))
}

fn is_denied(
    allowed_users: &HashSet<&str>,
    allowed_groups: &HashSet<&str>,
    deny_users: &[String],
    deny_groups: &[String],
) -> bool {
    deny_users
        .iter()
        .any(|user| allowed_users.contains(user.as_str()))
        || deny_groups
            .iter()
            .any(|group| allowed_groups.contains(group.as_str()))
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

    fn temp_store(_name: &str, ann_threshold: usize) -> SqliteVectorStore {
        SqliteVectorStore::open(Path::new(":memory:"))
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

        let mut chunks: Vec<EmbeddedChunk> = (0..300)
            .map(|i| chunk(&format!("c{i}"), "doc", pseudo_vec(i)))
            .collect();
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
        let chunks: Vec<EmbeddedChunk> = (0..200)
            .map(|i| chunk(&format!("c{i}"), "doc", pseudo_vec(i)))
            .collect();
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
        let mut chunks: Vec<EmbeddedChunk> = (0..100)
            .map(|i| chunk(&format!("c{i}"), "doc", pseudo_vec(i)))
            .collect();
        chunks.push(chunk("target", "target-doc", target_query.clone()));
        store.upsert(chunks).await.unwrap();

        // Build the index, confirm the target is found.
        let results = store.query(query(target_query.clone(), 1)).await.unwrap();
        assert_eq!(results[0].chunk.id, "target");

        store
            .delete_by_doc("t1", "w1", "s1", "target-doc")
            .await
            .unwrap();
        let results = store.query(query(target_query, 1)).await.unwrap();
        assert!(results.iter().all(|r| r.chunk.id != "target"));
    }

    #[tokio::test]
    async fn upsert_replaces_embedding_in_ann_index() {
        let store = temp_store("ann_replace", 16);
        let old_emb = pseudo_vec(111_111);
        let new_emb = pseudo_vec(222_222);
        let mut chunks: Vec<EmbeddedChunk> = (0..100)
            .map(|i| chunk(&format!("c{i}"), "doc", pseudo_vec(i)))
            .collect();
        chunks.push(chunk("target", "doc", old_emb.clone()));
        store.upsert(chunks).await.unwrap();

        // Build the index against the old embedding.
        let results = store.query(query(old_emb.clone(), 1)).await.unwrap();
        assert_eq!(results[0].chunk.id, "target");

        store
            .upsert(vec![chunk("target", "doc", new_emb.clone())])
            .await
            .unwrap();
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
        let mut chunks: Vec<EmbeddedChunk> = (0..100)
            .map(|i| chunk(&format!("c{i}"), "doc", pseudo_vec(i)))
            .collect();
        chunks.push(denied);
        store.upsert(chunks).await.unwrap();

        let results = store.query(query(target_query, 10)).await.unwrap();
        assert_eq!(results.len(), 10);
        assert!(results.iter().all(|r| r.chunk.id != "denied"));
    }

    #[tokio::test]
    async fn small_scope_uses_exact_scan() {
        let store = temp_store("exact_small", 1024);
        let chunks: Vec<EmbeddedChunk> = (0..10)
            .map(|i| chunk(&format!("c{i}"), "doc", pseudo_vec(i)))
            .collect();
        store.upsert(chunks).await.unwrap();

        let results = store.query(query(pseudo_vec(3), 3)).await.unwrap();
        assert_eq!(results.len(), 3);
        assert_eq!(results[0].chunk.id, "c3");
        assert!(results[0].score > 0.999);
    }

    #[tokio::test]
    async fn logical_chunk_ids_are_isolated_by_tenant_and_source() {
        let store = temp_store("scoped_ids", usize::MAX);
        let embedding = pseudo_vec(7);
        let mut source_two = chunk("shared", "doc", embedding.clone());
        source_two.source_id = "s2".to_string();
        source_two.text = "source two".to_string();
        let mut tenant_two = chunk("shared", "doc", embedding.clone());
        tenant_two.tenant_id = "t2".to_string();
        tenant_two.text = "tenant two".to_string();

        store
            .upsert(vec![
                chunk("shared", "doc", embedding.clone()),
                source_two,
                tenant_two,
            ])
            .await
            .unwrap();

        let tenant_one = store.query(query(embedding.clone(), 10)).await.unwrap();
        assert_eq!(tenant_one.len(), 2);
        assert!(tenant_one
            .iter()
            .any(|result| result.chunk.source_id == "s1"));
        assert!(tenant_one
            .iter()
            .any(|result| result.chunk.source_id == "s2"));

        let mut tenant_two_query = query(embedding, 10);
        tenant_two_query.tenant_id = "t2".to_string();
        let tenant_two = store.query(tenant_two_query).await.unwrap();
        assert_eq!(tenant_two.len(), 1);
        assert_eq!(tenant_two[0].chunk.text, "tenant two");
    }

    #[tokio::test]
    async fn query_and_upsert_reject_invalid_embeddings_and_limits() {
        let store = temp_store("invalid_input", usize::MAX);
        let mut invalid_chunk = chunk("invalid", "doc", vec![f32::NAN; DIM]);
        assert!(matches!(
            store.upsert(vec![invalid_chunk.clone()]).await,
            Err(VectorStoreError::InvalidInput(_))
        ));

        invalid_chunk.embedding = pseudo_vec(1);
        store.upsert(vec![invalid_chunk]).await.unwrap();
        let mut invalid_query = query(pseudo_vec(1), 0);
        assert!(matches!(
            store.query(invalid_query.clone()).await,
            Err(VectorStoreError::InvalidInput(_))
        ));
        invalid_query.top_k = 1;
        invalid_query.min_score = f32::NAN;
        assert!(matches!(
            store.query(invalid_query).await,
            Err(VectorStoreError::InvalidInput(_))
        ));
    }

    #[tokio::test]
    async fn malformed_acl_data_fails_closed() {
        let store = temp_store("malformed_acl", usize::MAX);
        store
            .upsert(vec![chunk("chunk", "doc", pseudo_vec(1))])
            .await
            .unwrap();
        store
            .conn
            .lock()
            .unwrap()
            .execute("UPDATE rag_chunks SET acl_allow_users = 'not-json'", [])
            .unwrap();

        assert!(matches!(
            store.query(query(pseudo_vec(1), 1)).await,
            Err(VectorStoreError::Serialization(_))
        ));
    }

    #[tokio::test]
    async fn legacy_database_schema_is_migrated_without_losing_chunks() {
        let path = std::env::temp_dir().join(format!(
            "kapsl_vec_test_legacy_migration_{}.sqlite3",
            std::process::id()
        ));
        let _ = std::fs::remove_file(&path);
        let legacy = Connection::open(&path).unwrap();
        legacy
            .execute_batch(
                "CREATE TABLE rag_chunks (
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
                );",
            )
            .unwrap();
        let embedding = pseudo_vec(17);
        legacy
            .execute(
                "INSERT INTO rag_chunks (
                    id, tenant_id, workspace_id, source_id, doc_id, chunk_index, text,
                    embedding, metadata_json, acl_allow_users, acl_allow_groups,
                    acl_deny_users, acl_deny_groups
                ) VALUES (?1, 't1', 'w1', 's1', 'doc', 0, 'legacy', ?2, '{}', '[]', '[]', '[]', '[]')",
                params!["legacy", serialize_embedding(&embedding)],
            )
            .unwrap();
        drop(legacy);

        let store = SqliteVectorStore::open(&path).unwrap();
        assert!(table_has_column(&store.conn.lock().unwrap(), "rag_chunks", "storage_id").unwrap());
        let results = store.query(query(embedding, 1)).await.unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].chunk.id, "legacy");
        assert_eq!(results[0].chunk.text, "legacy");
        drop(store);
        let _ = std::fs::remove_file(path);
    }
}
