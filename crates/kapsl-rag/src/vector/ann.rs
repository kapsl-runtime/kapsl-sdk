//! In-memory HNSW indexes layered over the SQLite vector store.
//!
//! One index per (tenant_id, workspace_id) scope, built lazily from SQLite on
//! the first query that crosses the ANN threshold. SQLite stays the source of
//! truth; the index only proposes candidate chunk ids, and the store re-ranks
//! candidates with exact cosine similarity so scores are identical to the
//! brute-force path.
//!
//! Deletes are tombstoned; the index is dropped (and lazily rebuilt) once
//! tombstones exceed [`TOMBSTONE_REBUILD_FRACTION`] of inserted points.

use hnsw_rs::prelude::*;
use std::collections::{HashMap, HashSet};

const HNSW_MAX_NB_CONNECTION: usize = 32;
const HNSW_NB_LAYER: usize = 16;
const HNSW_EF_CONSTRUCTION: usize = 200;
const TOMBSTONE_REBUILD_FRACTION: f32 = 0.2;

/// Scopes with fewer live vectors than this are scanned exactly.
pub const DEFAULT_ANN_THRESHOLD: usize = 1024;

pub struct ScopeIndex {
    hnsw: Hnsw<'static, f32, DistCosine>,
    /// data_id (usize handed to hnsw) -> chunk id
    ids: Vec<String>,
    /// chunk id -> data_id, for tombstoning replaced/deleted chunks
    live: HashMap<String, usize>,
    tombstones: HashSet<usize>,
    dim: usize,
}

impl ScopeIndex {
    pub fn new(dim: usize, capacity_hint: usize) -> Self {
        let capacity = capacity_hint.max(DEFAULT_ANN_THRESHOLD) * 2;
        Self {
            hnsw: Hnsw::new(
                HNSW_MAX_NB_CONNECTION,
                capacity,
                HNSW_NB_LAYER,
                HNSW_EF_CONSTRUCTION,
                DistCosine {},
            ),
            ids: Vec::new(),
            live: HashMap::new(),
            tombstones: HashSet::new(),
            dim,
        }
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn live_count(&self) -> usize {
        self.live.len()
    }

    /// Insert or replace a chunk's embedding. Vectors of the wrong dimension
    /// are skipped, matching the brute-force path which skips them at query
    /// time.
    pub fn upsert(&mut self, chunk_id: &str, embedding: &Vec<f32>) {
        if embedding.len() != self.dim {
            return;
        }
        if let Some(old) = self.live.remove(chunk_id) {
            self.tombstones.insert(old);
        }
        let data_id = self.ids.len();
        self.hnsw.insert((embedding, data_id));
        self.ids.push(chunk_id.to_string());
        self.live.insert(chunk_id.to_string(), data_id);
    }

    pub fn remove(&mut self, chunk_id: &str) {
        if let Some(data_id) = self.live.remove(chunk_id) {
            self.tombstones.insert(data_id);
        }
    }

    /// True once enough points are dead that a rebuild is cheaper than
    /// filtering them out of every search.
    pub fn needs_rebuild(&self) -> bool {
        let total = self.ids.len();
        total > 0 && self.tombstones.len() as f32 > total as f32 * TOMBSTONE_REBUILD_FRACTION
    }

    /// Return up to `k` live candidate chunk ids, nearest first.
    pub fn search(&self, query: &[f32], k: usize) -> Vec<String> {
        if query.len() != self.dim {
            return Vec::new();
        }
        // Over-fetch so tombstones don't eat into the requested k.
        let fetch = k + self.tombstones.len().min(k);
        // Search beam width. A small ef can miss an obvious nearest neighbor when
        // it sits among unrelated vectors (the greedy traversal never routes to
        // its node), and nondeterministic graph construction makes that flaky
        // across runs/platforms. Keep ef generous and floored at the construction
        // beam so recall of a clear top hit is reliable.
        let ef_search = (fetch * 4).max(HNSW_EF_CONSTRUCTION);
        self.hnsw
            .search(query, fetch, ef_search)
            .into_iter()
            .filter(|n| !self.tombstones.contains(&n.d_id))
            .take(k)
            .map(|n| self.ids[n.d_id].clone())
            .collect()
    }
}
