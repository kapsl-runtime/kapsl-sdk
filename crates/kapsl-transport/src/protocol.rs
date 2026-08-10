//! Wire protocol shared by the stream-based transport clients.
//!
//! TCP and IPC (Unix socket / named pipe) speak the same framing, so the
//! request/response exchange lives here once and is generic over the stream
//! rather than being reimplemented per transport.

use crate::{RequestMetadata, ResponseMetadata, TransportError};
use kapsl_engine_api::BinaryTensorPacket;
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};

/// Perform one inference round-trip over `conn`.
///
/// Request framing:  `[metadata_len: u32][metadata][input_len: u32][input]`
/// Response framing: `[metadata_len: u32][metadata][output_len: u32][output]`
pub async fn infer_over_stream<S>(
    conn: &mut S,
    model_id: u32,
    input: BinaryTensorPacket,
) -> Result<BinaryTensorPacket, TransportError>
where
    S: AsyncRead + AsyncWrite + Unpin + ?Sized,
{
    // TransportClient carries no per-client counter, so derive a request id
    // from the wall clock.
    let request_id = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos() as u64;

    let metadata = RequestMetadata::new(request_id, model_id, 0, false);

    let metadata_bytes =
        bincode::serialize(&metadata).map_err(|e| TransportError::Serialization(e.to_string()))?;
    let input_bytes =
        bincode::serialize(&input).map_err(|e| TransportError::Serialization(e.to_string()))?;

    conn.write_u32(metadata_bytes.len() as u32).await?;
    conn.write_all(&metadata_bytes).await?;
    conn.write_u32(input_bytes.len() as u32).await?;
    conn.write_all(&input_bytes).await?;
    conn.flush().await?;

    let resp_metadata_len = conn.read_u32().await?;
    let mut resp_metadata_buf = vec![0u8; resp_metadata_len as usize];
    conn.read_exact(&mut resp_metadata_buf).await?;

    let resp_metadata: ResponseMetadata = bincode::deserialize(&resp_metadata_buf)
        .map_err(|e| TransportError::Serialization(e.to_string()))?;

    if !resp_metadata.is_success() {
        // Drain the body the server still framed for us before reporting.
        let output_len = conn.read_u32().await?;
        let mut output_buf = vec![0u8; output_len as usize];
        conn.read_exact(&mut output_buf).await?;
        return Err(TransportError::ServerError(format!(
            "Remote error (status {})",
            resp_metadata.status
        )));
    }

    let output_len = conn.read_u32().await?;
    let mut output_buf = vec![0u8; output_len as usize];
    conn.read_exact(&mut output_buf).await?;

    bincode::deserialize(&output_buf).map_err(|e| TransportError::Serialization(e.to_string()))
}
