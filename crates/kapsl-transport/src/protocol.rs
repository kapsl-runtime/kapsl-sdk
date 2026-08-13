//! Canonical framing for Kapsl's stream transports.
//!
//! Unix sockets, named pipes, and TCP all use the deployed little-endian
//! framing defined here. Keeping byte I/O in this lower-level crate lets the
//! IPC server, Rust clients, and language bindings share one implementation.

use crate::TransportError;
use kapsl_engine_api::{BinaryTensorPacket, InferenceRequest, NamedTensor};
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use std::io;
use thiserror::Error;
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};

/// Default allocation ceiling for a single peer-supplied frame payload.
pub const DEFAULT_MAX_FRAME_PAYLOAD_BYTES: usize = 1024 * 1024 * 1024;

pub const OP_INFER: u32 = 1;
pub const OP_INFER_STREAM: u32 = 2;
pub const OP_METRICS: u32 = 3;
pub const OP_HYBRID_INFER: u32 = 4;

pub const STATUS_OK: u32 = 0;
pub const STATUS_ERR: u32 = 1;
pub const STATUS_STREAM_CHUNK: u32 = 2;
pub const STATUS_STREAM_END: u32 = 3;

const REQUEST_HEADER_BYTES: usize = 12;
const RESPONSE_HEADER_BYTES: usize = 8;

/// Fixed 12-byte request header. Headers are encoded explicitly, not with
/// bincode, so their wire representation cannot change with serde settings.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct RequestHeader {
    pub model_id: u32,
    pub op_code: u32,
    pub payload_size: u32,
}

/// Fixed 8-byte response header.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResponseHeader {
    pub status: u32,
    pub payload_size: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RequestFrame {
    pub header: RequestHeader,
    pub payload: Vec<u8>,
}

impl RequestFrame {
    pub fn deserialize<T: DeserializeOwned>(&self) -> Result<T, CodecError> {
        deserialize_value(&self.payload)
    }

    pub fn decode_inference_request(&self) -> Result<InferenceRequest, CodecError> {
        decode_inference_request(&self.payload)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResponseFrame {
    pub header: ResponseHeader,
    pub payload: Vec<u8>,
}

impl ResponseFrame {
    pub fn deserialize<T: DeserializeOwned>(&self) -> Result<T, CodecError> {
        deserialize_value(&self.payload)
    }

    pub fn remote_error(&self) -> CodecError {
        CodecError::Remote(String::from_utf8_lossy(&self.payload).into_owned())
    }

    pub fn ensure_status(&self, expected: u32) -> Result<(), CodecError> {
        if self.header.status == expected {
            Ok(())
        } else if self.header.status == STATUS_ERR {
            Err(self.remote_error())
        } else {
            Err(CodecError::UnexpectedStatus(self.header.status))
        }
    }
}

#[derive(Debug, Clone)]
pub enum StreamResponse {
    Chunk(BinaryTensorPacket),
    End,
}

#[derive(Debug, Error)]
pub enum CodecError {
    #[error("I/O error: {0}")]
    Io(#[from] io::Error),
    #[error("serialization error: {0}")]
    Serialize(String),
    #[error("deserialization error: {0}")]
    Deserialize(String),
    #[error("frame payload is {size} bytes; maximum is {max} bytes")]
    PayloadTooLarge { size: usize, max: usize },
    #[error("frame payload length {size} does not fit in a u32")]
    LengthOverflow { size: usize },
    #[error("unexpected response status {0}")]
    UnexpectedStatus(u32),
    #[error("remote error: {0}")]
    Remote(String),
}

impl CodecError {
    pub fn is_io(&self) -> bool {
        matches!(self, Self::Io(_))
    }

    pub fn into_io_error(self) -> io::Error {
        match self {
            Self::Io(error) => error,
            other => io::Error::new(io::ErrorKind::InvalidData, other),
        }
    }

    pub fn into_transport_error(self) -> TransportError {
        match self {
            Self::Io(error) => TransportError::Io(error),
            Self::Serialize(message) | Self::Deserialize(message) => {
                TransportError::Serialization(message)
            }
            Self::Remote(message) => TransportError::ServerError(message),
            Self::PayloadTooLarge { .. }
            | Self::LengthOverflow { .. }
            | Self::UnexpectedStatus(_) => TransportError::InvalidRequest(self.to_string()),
        }
    }
}

impl From<CodecError> for TransportError {
    fn from(error: CodecError) -> Self {
        error.into_transport_error()
    }
}

#[derive(Debug, Serialize, Deserialize)]
struct LegacyInferenceRequestV1 {
    input: BinaryTensorPacket,
    #[serde(default)]
    additional_inputs: Vec<NamedTensor>,
    #[serde(default)]
    session_id: Option<String>,
}

pub fn serialize_value<T: Serialize + ?Sized>(value: &T) -> Result<Vec<u8>, CodecError> {
    bincode::serialize(value).map_err(|error| CodecError::Serialize(error.to_string()))
}

pub fn deserialize_value<T: DeserializeOwned>(payload: &[u8]) -> Result<T, CodecError> {
    bincode::deserialize(payload).map_err(|error| CodecError::Deserialize(error.to_string()))
}

pub fn encode_inference_request(request: &InferenceRequest) -> Result<Vec<u8>, CodecError> {
    serialize_value(request)
}

/// Decode the current request layout, with the server's pre-metadata layout as
/// a compatibility fallback. This keeps every ingress on the same policy.
pub fn decode_inference_request(payload: &[u8]) -> Result<InferenceRequest, CodecError> {
    match bincode::deserialize::<InferenceRequest>(payload) {
        Ok(request) => Ok(request),
        Err(primary_error) => match bincode::deserialize::<LegacyInferenceRequestV1>(payload) {
            Ok(legacy) => Ok(InferenceRequest {
                input: legacy.input,
                additional_inputs: legacy.additional_inputs,
                session_id: legacy.session_id,
                metadata: None,
                cancellation: None,
            }),
            Err(_) => Err(CodecError::Deserialize(primary_error.to_string())),
        },
    }
}

fn checked_payload_len(size: usize, max: usize) -> Result<u32, CodecError> {
    if size > max {
        return Err(CodecError::PayloadTooLarge { size, max });
    }
    u32::try_from(size).map_err(|_| CodecError::LengthOverflow { size })
}

fn validate_incoming_len(size: u32, max: usize) -> Result<usize, CodecError> {
    let size = size as usize;
    if size > max {
        Err(CodecError::PayloadTooLarge { size, max })
    } else {
        Ok(size)
    }
}

fn encode_request_header(header: RequestHeader) -> [u8; REQUEST_HEADER_BYTES] {
    let mut bytes = [0; REQUEST_HEADER_BYTES];
    bytes[0..4].copy_from_slice(&header.model_id.to_le_bytes());
    bytes[4..8].copy_from_slice(&header.op_code.to_le_bytes());
    bytes[8..12].copy_from_slice(&header.payload_size.to_le_bytes());
    bytes
}

fn decode_request_header(bytes: [u8; REQUEST_HEADER_BYTES]) -> RequestHeader {
    RequestHeader {
        model_id: u32::from_le_bytes(bytes[0..4].try_into().expect("fixed slice")),
        op_code: u32::from_le_bytes(bytes[4..8].try_into().expect("fixed slice")),
        payload_size: u32::from_le_bytes(bytes[8..12].try_into().expect("fixed slice")),
    }
}

fn encode_response_header(header: ResponseHeader) -> [u8; RESPONSE_HEADER_BYTES] {
    let mut bytes = [0; RESPONSE_HEADER_BYTES];
    bytes[0..4].copy_from_slice(&header.status.to_le_bytes());
    bytes[4..8].copy_from_slice(&header.payload_size.to_le_bytes());
    bytes
}

fn decode_response_header(bytes: [u8; RESPONSE_HEADER_BYTES]) -> ResponseHeader {
    ResponseHeader {
        status: u32::from_le_bytes(bytes[0..4].try_into().expect("fixed slice")),
        payload_size: u32::from_le_bytes(bytes[4..8].try_into().expect("fixed slice")),
    }
}

pub mod blocking {
    use super::*;
    use std::io::{Read, Write};

    pub fn write_request_value<W, T>(
        writer: &mut W,
        model_id: u32,
        op_code: u32,
        value: &T,
    ) -> Result<(), CodecError>
    where
        W: Write + ?Sized,
        T: Serialize + ?Sized,
    {
        write_request_bytes(writer, model_id, op_code, &serialize_value(value)?)
    }

    pub fn write_request_bytes<W: Write + ?Sized>(
        writer: &mut W,
        model_id: u32,
        op_code: u32,
        payload: &[u8],
    ) -> Result<(), CodecError> {
        let payload_size = checked_payload_len(payload.len(), DEFAULT_MAX_FRAME_PAYLOAD_BYTES)?;
        writer.write_all(&encode_request_header(RequestHeader {
            model_id,
            op_code,
            payload_size,
        }))?;
        writer.write_all(payload)?;
        writer.flush()?;
        Ok(())
    }

    pub fn read_request_frame<R: Read + ?Sized>(
        reader: &mut R,
        max_payload_bytes: usize,
    ) -> Result<RequestFrame, CodecError> {
        let mut header_bytes = [0; REQUEST_HEADER_BYTES];
        reader.read_exact(&mut header_bytes)?;
        let header = decode_request_header(header_bytes);
        let payload_size = validate_incoming_len(header.payload_size, max_payload_bytes)?;
        let mut payload = vec![0; payload_size];
        reader.read_exact(&mut payload)?;
        Ok(RequestFrame { header, payload })
    }

    pub fn write_response_value<W, T>(
        writer: &mut W,
        status: u32,
        value: &T,
    ) -> Result<(), CodecError>
    where
        W: Write + ?Sized,
        T: Serialize + ?Sized,
    {
        write_response_bytes(writer, status, &serialize_value(value)?)
    }

    pub fn write_response_bytes<W: Write + ?Sized>(
        writer: &mut W,
        status: u32,
        payload: &[u8],
    ) -> Result<(), CodecError> {
        let payload_size = checked_payload_len(payload.len(), DEFAULT_MAX_FRAME_PAYLOAD_BYTES)?;
        writer.write_all(&encode_response_header(ResponseHeader {
            status,
            payload_size,
        }))?;
        writer.write_all(payload)?;
        writer.flush()?;
        Ok(())
    }

    pub fn read_response_frame<R: Read + ?Sized>(
        reader: &mut R,
        max_payload_bytes: usize,
    ) -> Result<ResponseFrame, CodecError> {
        let mut header_bytes = [0; RESPONSE_HEADER_BYTES];
        reader.read_exact(&mut header_bytes)?;
        let header = decode_response_header(header_bytes);
        let payload_size = validate_incoming_len(header.payload_size, max_payload_bytes)?;
        let mut payload = vec![0; payload_size];
        reader.read_exact(&mut payload)?;
        Ok(ResponseFrame { header, payload })
    }

    pub fn read_response_value<R, T>(
        reader: &mut R,
        max_payload_bytes: usize,
    ) -> Result<T, CodecError>
    where
        R: Read + ?Sized,
        T: DeserializeOwned,
    {
        let frame = read_response_frame(reader, max_payload_bytes)?;
        frame.ensure_status(STATUS_OK)?;
        frame.deserialize()
    }

    pub fn infer_request_over_stream<S: Read + Write + ?Sized>(
        conn: &mut S,
        model_id: u32,
        request: &InferenceRequest,
    ) -> Result<BinaryTensorPacket, CodecError> {
        infer_request_over_stream_with_limit(
            conn,
            model_id,
            request,
            DEFAULT_MAX_FRAME_PAYLOAD_BYTES,
        )
    }

    pub fn infer_request_over_stream_with_limit<S: Read + Write + ?Sized>(
        conn: &mut S,
        model_id: u32,
        request: &InferenceRequest,
        max_payload_bytes: usize,
    ) -> Result<BinaryTensorPacket, CodecError> {
        write_request_value(conn, model_id, OP_INFER, request)?;
        read_response_value(conn, max_payload_bytes)
    }

    pub fn read_stream_packet<R: Read + ?Sized>(
        reader: &mut R,
        max_payload_bytes: usize,
    ) -> Result<StreamResponse, CodecError> {
        let frame = read_response_frame(reader, max_payload_bytes)?;
        match frame.header.status {
            STATUS_STREAM_CHUNK => Ok(StreamResponse::Chunk(frame.deserialize()?)),
            STATUS_STREAM_END => Ok(StreamResponse::End),
            STATUS_ERR => Err(frame.remote_error()),
            status => Err(CodecError::UnexpectedStatus(status)),
        }
    }
}

pub mod asynchronous {
    use super::*;

    pub async fn write_request_value<W, T>(
        writer: &mut W,
        model_id: u32,
        op_code: u32,
        value: &T,
    ) -> Result<(), CodecError>
    where
        W: AsyncWrite + Unpin + ?Sized,
        T: Serialize + ?Sized,
    {
        write_request_bytes(writer, model_id, op_code, &serialize_value(value)?).await
    }

    pub async fn write_request_bytes<W: AsyncWrite + Unpin + ?Sized>(
        writer: &mut W,
        model_id: u32,
        op_code: u32,
        payload: &[u8],
    ) -> Result<(), CodecError> {
        let payload_size = checked_payload_len(payload.len(), DEFAULT_MAX_FRAME_PAYLOAD_BYTES)?;
        writer
            .write_all(&encode_request_header(RequestHeader {
                model_id,
                op_code,
                payload_size,
            }))
            .await?;
        writer.write_all(payload).await?;
        writer.flush().await?;
        Ok(())
    }

    pub async fn read_request_frame<R: AsyncRead + Unpin + ?Sized>(
        reader: &mut R,
        max_payload_bytes: usize,
    ) -> Result<RequestFrame, CodecError> {
        let mut header_bytes = [0; REQUEST_HEADER_BYTES];
        reader.read_exact(&mut header_bytes).await?;
        let header = decode_request_header(header_bytes);
        let payload_size = validate_incoming_len(header.payload_size, max_payload_bytes)?;
        let mut payload = vec![0; payload_size];
        reader.read_exact(&mut payload).await?;
        Ok(RequestFrame { header, payload })
    }

    /// Like `read_request_frame`, but distinguishes a clean close before the
    /// next frame from a truncated header.
    pub async fn read_request_frame_or_eof<R: AsyncRead + Unpin + ?Sized>(
        reader: &mut R,
        max_payload_bytes: usize,
    ) -> Result<Option<RequestFrame>, CodecError> {
        let mut header_bytes = [0; REQUEST_HEADER_BYTES];
        match reader.read(&mut header_bytes[..1]).await? {
            0 => return Ok(None),
            1 => {}
            _ => unreachable!("one-byte read"),
        }
        reader.read_exact(&mut header_bytes[1..]).await?;
        let header = decode_request_header(header_bytes);
        let payload_size = validate_incoming_len(header.payload_size, max_payload_bytes)?;
        let mut payload = vec![0; payload_size];
        reader.read_exact(&mut payload).await?;
        Ok(Some(RequestFrame { header, payload }))
    }

    pub async fn write_response_value<W, T>(
        writer: &mut W,
        status: u32,
        value: &T,
    ) -> Result<(), CodecError>
    where
        W: AsyncWrite + Unpin + ?Sized,
        T: Serialize + ?Sized,
    {
        write_response_bytes(writer, status, &serialize_value(value)?).await
    }

    pub async fn write_response_bytes<W: AsyncWrite + Unpin + ?Sized>(
        writer: &mut W,
        status: u32,
        payload: &[u8],
    ) -> Result<(), CodecError> {
        let payload_size = checked_payload_len(payload.len(), DEFAULT_MAX_FRAME_PAYLOAD_BYTES)?;
        writer
            .write_all(&encode_response_header(ResponseHeader {
                status,
                payload_size,
            }))
            .await?;
        writer.write_all(payload).await?;
        writer.flush().await?;
        Ok(())
    }

    pub async fn read_response_frame<R: AsyncRead + Unpin + ?Sized>(
        reader: &mut R,
        max_payload_bytes: usize,
    ) -> Result<ResponseFrame, CodecError> {
        let mut header_bytes = [0; RESPONSE_HEADER_BYTES];
        reader.read_exact(&mut header_bytes).await?;
        let header = decode_response_header(header_bytes);
        let payload_size = validate_incoming_len(header.payload_size, max_payload_bytes)?;
        let mut payload = vec![0; payload_size];
        reader.read_exact(&mut payload).await?;
        Ok(ResponseFrame { header, payload })
    }

    pub async fn read_response_value<R, T>(
        reader: &mut R,
        max_payload_bytes: usize,
    ) -> Result<T, CodecError>
    where
        R: AsyncRead + Unpin + ?Sized,
        T: DeserializeOwned,
    {
        let frame = read_response_frame(reader, max_payload_bytes).await?;
        frame.ensure_status(STATUS_OK)?;
        frame.deserialize()
    }

    pub async fn infer_request_over_stream<S: AsyncRead + AsyncWrite + Unpin + ?Sized>(
        conn: &mut S,
        model_id: u32,
        request: &InferenceRequest,
    ) -> Result<BinaryTensorPacket, CodecError> {
        infer_request_over_stream_with_limit(
            conn,
            model_id,
            request,
            DEFAULT_MAX_FRAME_PAYLOAD_BYTES,
        )
        .await
    }

    pub async fn infer_request_over_stream_with_limit<
        S: AsyncRead + AsyncWrite + Unpin + ?Sized,
    >(
        conn: &mut S,
        model_id: u32,
        request: &InferenceRequest,
        max_payload_bytes: usize,
    ) -> Result<BinaryTensorPacket, CodecError> {
        write_request_value(conn, model_id, OP_INFER, request).await?;
        read_response_value(conn, max_payload_bytes).await
    }

    pub async fn read_stream_packet<R: AsyncRead + Unpin + ?Sized>(
        reader: &mut R,
        max_payload_bytes: usize,
    ) -> Result<StreamResponse, CodecError> {
        let frame = read_response_frame(reader, max_payload_bytes).await?;
        match frame.header.status {
            STATUS_STREAM_CHUNK => Ok(StreamResponse::Chunk(frame.deserialize()?)),
            STATUS_STREAM_END => Ok(StreamResponse::End),
            STATUS_ERR => Err(frame.remote_error()),
            status => Err(CodecError::UnexpectedStatus(status)),
        }
    }
}

/// Backward-compatible `TransportClient` helper. Its behavior is corrected to
/// speak the same request-header protocol as `IpcServer` and `TcpServer`.
pub async fn infer_over_stream<S>(
    conn: &mut S,
    model_id: u32,
    input: BinaryTensorPacket,
) -> Result<BinaryTensorPacket, TransportError>
where
    S: AsyncRead + AsyncWrite + Unpin + ?Sized,
{
    let request = InferenceRequest::new(input);
    asynchronous::infer_request_over_stream(conn, model_id, &request)
        .await
        .map_err(TransportError::from)
}

#[cfg(test)]
mod tests {
    use super::*;
    use kapsl_engine_api::TensorDtype;
    use std::io::Cursor;
    use tokio::io::duplex;

    fn packet() -> BinaryTensorPacket {
        BinaryTensorPacket {
            shape: vec![1],
            dtype: TensorDtype::Float32,
            data: 1.5f32.to_le_bytes().to_vec(),
        }
    }

    #[test]
    fn request_header_is_exact_little_endian_wire_format() {
        let mut wire = Vec::new();
        blocking::write_request_bytes(&mut wire, 0x0102_0304, OP_INFER, &[7, 8]).unwrap();
        assert_eq!(&wire[..12], &[4, 3, 2, 1, 1, 0, 0, 0, 2, 0, 0, 0]);

        let frame =
            blocking::read_request_frame(&mut Cursor::new(wire), DEFAULT_MAX_FRAME_PAYLOAD_BYTES)
                .unwrap();
        assert_eq!(frame.header.model_id, 0x0102_0304);
        assert_eq!(frame.payload, vec![7, 8]);
    }

    #[test]
    fn response_header_is_exact_little_endian_wire_format() {
        let mut wire = Vec::new();
        blocking::write_response_bytes(&mut wire, STATUS_STREAM_CHUNK, &[7, 8]).unwrap();
        assert_eq!(&wire[..8], &[2, 0, 0, 0, 2, 0, 0, 0]);

        let frame =
            blocking::read_response_frame(&mut Cursor::new(wire), DEFAULT_MAX_FRAME_PAYLOAD_BYTES)
                .unwrap();
        assert_eq!(frame.header.status, STATUS_STREAM_CHUNK);
        assert_eq!(frame.payload, vec![7, 8]);
    }

    #[test]
    fn legacy_request_layout_decodes_in_one_shared_place() {
        let legacy = LegacyInferenceRequestV1 {
            input: packet(),
            additional_inputs: Vec::new(),
            session_id: Some("legacy-session".to_string()),
        };
        let payload = serialize_value(&legacy).unwrap();

        let decoded = decode_inference_request(&payload).unwrap();
        assert_eq!(decoded.input.data, packet().data);
        assert_eq!(decoded.session_id.as_deref(), Some("legacy-session"));
        assert!(decoded.metadata.is_none());
    }

    #[test]
    fn oversized_incoming_payload_is_rejected_before_allocation() {
        let mut wire = Vec::new();
        wire.extend_from_slice(&7u32.to_le_bytes());
        wire.extend_from_slice(&OP_INFER.to_le_bytes());
        wire.extend_from_slice(&11u32.to_le_bytes());

        let error = blocking::read_request_frame(&mut Cursor::new(wire), 10).unwrap_err();
        assert!(matches!(
            error,
            CodecError::PayloadTooLarge { size: 11, max: 10 }
        ));
    }

    #[tokio::test]
    async fn unary_helper_matches_deployed_frame_and_drains_remote_errors() {
        let (mut client, mut server) = duplex(4096);
        let server_task = tokio::spawn(async move {
            let first =
                asynchronous::read_request_frame(&mut server, DEFAULT_MAX_FRAME_PAYLOAD_BYTES)
                    .await
                    .unwrap();
            assert_eq!(first.header.model_id, 9);
            assert_eq!(first.header.op_code, OP_INFER);
            assert_eq!(
                first.decode_inference_request().unwrap().input.data,
                packet().data
            );
            asynchronous::write_response_bytes(&mut server, STATUS_ERR, b"try again")
                .await
                .unwrap();

            let second =
                asynchronous::read_request_frame(&mut server, DEFAULT_MAX_FRAME_PAYLOAD_BYTES)
                    .await
                    .unwrap();
            assert_eq!(second.header.model_id, 9);
            asynchronous::write_response_value(&mut server, STATUS_OK, &packet())
                .await
                .unwrap();
        });

        let error = infer_over_stream(&mut client, 9, packet())
            .await
            .unwrap_err();
        assert!(matches!(error, TransportError::ServerError(message) if message == "try again"));
        let output = infer_over_stream(&mut client, 9, packet()).await.unwrap();
        assert_eq!(output.data, packet().data);
        server_task.await.unwrap();
    }

    #[tokio::test]
    async fn stream_reader_handles_chunk_and_end() {
        let (mut client, mut server) = duplex(4096);
        let server_task = tokio::spawn(async move {
            asynchronous::write_response_value(&mut server, STATUS_STREAM_CHUNK, &packet())
                .await
                .unwrap();
            asynchronous::write_response_bytes(&mut server, STATUS_STREAM_END, &[])
                .await
                .unwrap();
        });

        match asynchronous::read_stream_packet(&mut client, DEFAULT_MAX_FRAME_PAYLOAD_BYTES)
            .await
            .unwrap()
        {
            StreamResponse::Chunk(value) => assert_eq!(value.data, packet().data),
            StreamResponse::End => panic!("expected chunk"),
        }
        assert!(matches!(
            asynchronous::read_stream_packet(&mut client, DEFAULT_MAX_FRAME_PAYLOAD_BYTES)
                .await
                .unwrap(),
            StreamResponse::End
        ));
        server_task.await.unwrap();
    }

    #[tokio::test]
    async fn clean_close_before_a_request_is_not_a_truncated_frame() {
        let (client, mut server) = duplex(64);
        drop(client);

        let frame =
            asynchronous::read_request_frame_or_eof(&mut server, DEFAULT_MAX_FRAME_PAYLOAD_BYTES)
                .await
                .unwrap();
        assert!(frame.is_none());
    }
}
