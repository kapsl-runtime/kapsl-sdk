use async_trait::async_trait;
use base64::engine::general_purpose::STANDARD as BASE64;
use base64::Engine;
use std::collections::HashMap;

use kapsl_rag_sdk::types::DocumentPayload;

use crate::vector::{AccessControl, EmbeddedChunk, VectorStore, VectorStoreError};

#[derive(thiserror::Error, Debug)]
pub enum IngestionError {
    #[error("invalid input: {0}")]
    InvalidInput(String),
    #[error("parse error: {0}")]
    Parse(String),
    #[error("embedding error: {0}")]
    Embedding(String),
    #[error("store error: {0}")]
    Store(String),
}

impl From<VectorStoreError> for IngestionError {
    fn from(err: VectorStoreError) -> Self {
        IngestionError::Store(err.to_string())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParsedDocument {
    pub id: String,
    pub text: String,
    pub metadata: HashMap<String, String>,
    pub content_type: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Chunk {
    pub id: String,
    pub index: i64,
    pub text: String,
    pub metadata: HashMap<String, String>,
}

#[async_trait]
pub trait DocumentParser: Send + Sync {
    async fn parse(&self, payload: &DocumentPayload) -> Result<ParsedDocument, IngestionError>;
}

#[async_trait]
pub trait Chunker: Send + Sync {
    async fn chunk(&self, document: &ParsedDocument) -> Result<Vec<Chunk>, IngestionError>;
}

#[async_trait]
pub trait Embedder: Send + Sync {
    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, IngestionError>;
}

pub struct IngestionContext {
    pub tenant_id: String,
    pub workspace_id: String,
    pub source_id: String,
    pub doc_id: String,
    pub acl: AccessControl,
}

pub struct IngestionPipeline<P, C, E, V>
where
    P: DocumentParser,
    C: Chunker,
    E: Embedder,
    V: VectorStore,
{
    parser: P,
    chunker: C,
    embedder: E,
    vector_store: V,
}

impl<P, C, E, V> IngestionPipeline<P, C, E, V>
where
    P: DocumentParser,
    C: Chunker,
    E: Embedder,
    V: VectorStore,
{
    pub fn new(parser: P, chunker: C, embedder: E, vector_store: V) -> Self {
        Self {
            parser,
            chunker,
            embedder,
            vector_store,
        }
    }

    pub async fn ingest(
        &self,
        ctx: IngestionContext,
        payload: DocumentPayload,
    ) -> Result<(), IngestionError> {
        let parsed = self.parser.parse(&payload).await?;
        let chunks = self.chunker.chunk(&parsed).await?;
        if chunks.is_empty() {
            return Ok(());
        }
        let texts: Vec<String> = chunks.iter().map(|c| c.text.clone()).collect();
        let embeddings = self.embedder.embed(&texts).await?;
        if embeddings.len() != chunks.len() {
            return Err(IngestionError::Embedding(
                "embedding count mismatch".to_string(),
            ));
        }
        if embeddings.iter().any(|embedding| {
            embedding.is_empty() || !embedding.iter().all(|value| value.is_finite())
        }) {
            return Err(IngestionError::Embedding(
                "embeddings must be non-empty and contain only finite values".to_string(),
            ));
        }

        let mut embedded = Vec::with_capacity(chunks.len());
        for (chunk, embedding) in chunks.into_iter().zip(embeddings.into_iter()) {
            embedded.push(EmbeddedChunk {
                id: chunk.id,
                tenant_id: ctx.tenant_id.clone(),
                workspace_id: ctx.workspace_id.clone(),
                source_id: ctx.source_id.clone(),
                doc_id: ctx.doc_id.clone(),
                chunk_index: chunk.index,
                text: chunk.text,
                embedding,
                metadata: chunk.metadata,
                acl: ctx.acl.clone(),
            });
        }

        self.vector_store.upsert(embedded).await?;
        Ok(())
    }
}

pub struct PlainTextParser;

#[async_trait]
impl DocumentParser for PlainTextParser {
    async fn parse(&self, payload: &DocumentPayload) -> Result<ParsedDocument, IngestionError> {
        let bytes = BASE64
            .decode(&payload.bytes_b64)
            .map_err(|e| IngestionError::Parse(e.to_string()))?;
        let text = String::from_utf8(bytes).map_err(|e| IngestionError::Parse(e.to_string()))?;
        Ok(ParsedDocument {
            id: payload.id.clone(),
            text,
            metadata: payload.metadata.clone(),
            content_type: payload.content_type.clone(),
        })
    }
}

pub struct SimpleChunker {
    pub chunk_size: usize,
    pub overlap: usize,
}

impl SimpleChunker {
    pub fn new(chunk_size: usize, overlap: usize) -> Result<Self, IngestionError> {
        validate_chunk_window(chunk_size, overlap)?;
        Ok(Self {
            chunk_size,
            overlap,
        })
    }
}

#[async_trait]
impl Chunker for SimpleChunker {
    async fn chunk(&self, document: &ParsedDocument) -> Result<Vec<Chunk>, IngestionError> {
        validate_chunk_window(self.chunk_size, self.overlap)?;
        let tokens: Vec<&str> = document.text.split_whitespace().collect();
        if tokens.is_empty() {
            return Ok(Vec::new());
        }
        let mut chunks = Vec::new();
        let mut start = 0usize;
        let mut index = 0i64;
        while start < tokens.len() {
            let end = (start + self.chunk_size).min(tokens.len());
            let slice = tokens[start..end].join(" ");
            let mut metadata = document.metadata.clone();
            metadata.insert("chunk_index".to_string(), index.to_string());
            chunks.push(Chunk {
                id: format!("{}:{}", document.id, index),
                index,
                text: slice,
                metadata,
            });
            if end == tokens.len() {
                break;
            }
            start = end - self.overlap;
            index += 1;
        }
        Ok(chunks)
    }
}

fn validate_chunk_window(chunk_size: usize, overlap: usize) -> Result<(), IngestionError> {
    if chunk_size == 0 {
        return Err(IngestionError::InvalidInput(
            "chunk_size must be greater than zero".to_string(),
        ));
    }
    if overlap >= chunk_size {
        return Err(IngestionError::InvalidInput(
            "overlap must be smaller than chunk_size".to_string(),
        ));
    }
    Ok(())
}

pub struct DummyEmbedder {
    pub dimension: usize,
}

#[async_trait]
impl Embedder for DummyEmbedder {
    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, IngestionError> {
        if self.dimension == 0 {
            return Err(IngestionError::InvalidInput(
                "embedding dimension must be greater than zero".to_string(),
            ));
        }
        let mut embeddings = Vec::with_capacity(texts.len());
        for _ in texts {
            embeddings.push(vec![0.0; self.dimension]);
        }
        Ok(embeddings)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn document(text: &str) -> ParsedDocument {
        ParsedDocument {
            id: "doc".to_string(),
            text: text.to_string(),
            metadata: HashMap::new(),
            content_type: "text/plain".to_string(),
        }
    }

    #[tokio::test]
    async fn chunker_rejects_non_advancing_windows() {
        assert!(SimpleChunker::new(0, 0).is_err());
        assert!(SimpleChunker::new(4, 4).is_err());

        let invalid = SimpleChunker {
            chunk_size: 4,
            overlap: 5,
        };
        assert!(invalid.chunk(&document("one two three")).await.is_err());
    }

    #[tokio::test]
    async fn chunker_builds_deterministic_overlapping_windows() {
        let chunker = SimpleChunker::new(3, 1).unwrap();

        let chunks = chunker
            .chunk(&document("one two three four five"))
            .await
            .unwrap();

        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].text, "one two three");
        assert_eq!(chunks[1].text, "three four five");
        assert_eq!(chunks[0].id, "doc:0");
        assert_eq!(chunks[1].id, "doc:1");
    }

    #[tokio::test]
    async fn dummy_embedder_rejects_zero_dimension() {
        let embedder = DummyEmbedder { dimension: 0 };

        assert!(embedder.embed(&["text".to_string()]).await.is_err());
    }
}
