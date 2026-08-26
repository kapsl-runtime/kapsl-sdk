//! Canonical framing for Kapsl's stream transports.
//!
//! Unix sockets, named pipes, and TCP all use the deployed little-endian
//! framing defined here. Keeping byte I/O in this lower-level crate lets the
//! IPC server, Rust clients, and language bindings share one implementation.

use crate::TransportError;
use kapsl_engine_api::{
    BinaryTensorPacket, InferenceRequest, NamedTensor, OpenAiWireFormat, OpenAiWireRequest,
    OpenAiWireResponse, OpenAiWireResponseHead,
};
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use std::io;
use thiserror::Error;
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};

/// Default allocation ceiling for a single peer-supplied frame payload.
pub const DEFAULT_MAX_FRAME_PAYLOAD_BYTES: usize = 1024 * 1024 * 1024;
/// Exact ceiling for the complete versioned OpenAI wire request envelope.
///
/// This includes the preamble, serialized request metadata, body, and optional
/// transport credential so paired clients cannot accept a request that the
/// IPC server will reject before authentication.
pub const MAX_OPENAI_WIRE_REQUEST_PAYLOAD_BYTES: usize = 16 * 1024 * 1024;

pub const OP_INFER: u32 = 1;
pub const OP_INFER_STREAM: u32 = 2;
pub const OP_METRICS: u32 = 3;
pub const OP_HYBRID_INFER: u32 = 4;
pub const OP_OPENAI_WIRE: u32 = 5;
pub const OP_OPENAI_WIRE_STREAM: u32 = 6;

/// Version of the transport-only envelope carried by OpenAI wire request
/// operation payloads. This preamble is checked before bincode deserializes the
/// engine-facing request.
pub const OPENAI_WIRE_TRANSPORT_VERSION: u16 = 1;

pub const STATUS_OK: u32 = 0;
pub const STATUS_ERR: u32 = 1;
pub const STATUS_STREAM_CHUNK: u32 = 2;
pub const STATUS_STREAM_END: u32 = 3;
pub const STATUS_OPENAI_WIRE_HEAD: u32 = 4;
pub const STATUS_OPENAI_WIRE_CHUNK: u32 = 5;

const REQUEST_HEADER_BYTES: usize = 12;
const RESPONSE_HEADER_BYTES: usize = 8;
const OPENAI_WIRE_PREAMBLE_MAGIC: [u8; 4] = *b"KOWR";
const OPENAI_WIRE_PREAMBLE_BYTES: usize =
    OPENAI_WIRE_PREAMBLE_MAGIC.len() + std::mem::size_of::<u16>();

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

    pub fn decode_openai_wire_envelope(&self) -> Result<OpenAiWireTransportEnvelope, CodecError> {
        if !matches!(self.header.op_code, OP_OPENAI_WIRE | OP_OPENAI_WIRE_STREAM) {
            return Err(CodecError::Deserialize(format!(
                "unexpected OpenAI wire request operation {}",
                self.header.op_code
            )));
        }
        let envelope = decode_openai_wire_transport_envelope(&self.payload)?;
        envelope
            .request
            .validate(self.payload.len())
            .map_err(|error| CodecError::Deserialize(error.to_string()))?;
        Ok(envelope)
    }

    pub fn decode_openai_wire_request(&self) -> Result<OpenAiWireRequest, CodecError> {
        Ok(self.decode_openai_wire_envelope()?.request)
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OpenAiWireStreamFrame {
    Head(OpenAiWireResponseHead),
    Chunk(Vec<u8>),
    End,
}

/// Transport-only wrapper for authenticating a wire operation. Its bincode
/// representation follows the fixed OpenAI wire transport preamble; it is
/// never written directly as a frame payload. The server consumes `auth_token`
/// before dispatching `request`, whose engine-facing type cannot represent
/// credentials.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAiWireTransportEnvelope {
    pub request: OpenAiWireRequest,
    #[serde(default)]
    pub auth_token: Option<String>,
}

impl OpenAiWireTransportEnvelope {
    pub fn unauthenticated(request: OpenAiWireRequest) -> Self {
        Self {
            request,
            auth_token: None,
        }
    }

    pub fn authenticated(request: OpenAiWireRequest, auth_token: impl Into<String>) -> Self {
        Self {
            request,
            auth_token: Some(auth_token.into()),
        }
    }
}

fn encode_openai_wire_transport_envelope(
    envelope: &OpenAiWireTransportEnvelope,
) -> Result<Vec<u8>, CodecError> {
    let encoded = serialize_value(envelope)?;
    let capacity = OPENAI_WIRE_PREAMBLE_BYTES
        .checked_add(encoded.len())
        .ok_or(CodecError::LengthOverflow {
            size: encoded.len(),
        })?;
    checked_payload_len(capacity, MAX_OPENAI_WIRE_REQUEST_PAYLOAD_BYTES)?;
    let mut payload = Vec::with_capacity(capacity);
    payload.extend_from_slice(&OPENAI_WIRE_PREAMBLE_MAGIC);
    payload.extend_from_slice(&OPENAI_WIRE_TRANSPORT_VERSION.to_le_bytes());
    payload.extend_from_slice(&encoded);
    Ok(payload)
}

fn decode_openai_wire_transport_envelope(
    payload: &[u8],
) -> Result<OpenAiWireTransportEnvelope, CodecError> {
    if payload.len() < OPENAI_WIRE_PREAMBLE_BYTES {
        return Err(CodecError::Deserialize(format!(
            "OpenAI wire transport payload is {} bytes; preamble requires {} bytes",
            payload.len(),
            OPENAI_WIRE_PREAMBLE_BYTES,
        )));
    }
    if payload[..OPENAI_WIRE_PREAMBLE_MAGIC.len()] != OPENAI_WIRE_PREAMBLE_MAGIC {
        return Err(CodecError::Deserialize(
            "OpenAI wire transport payload has an invalid preamble".to_string(),
        ));
    }
    let version_offset = OPENAI_WIRE_PREAMBLE_MAGIC.len();
    let version = u16::from_le_bytes(
        payload[version_offset..OPENAI_WIRE_PREAMBLE_BYTES]
            .try_into()
            .expect("fixed OpenAI wire version slice"),
    );
    if version != OPENAI_WIRE_TRANSPORT_VERSION {
        return Err(CodecError::Deserialize(format!(
            "unsupported OpenAI wire transport version {version}; expected {OPENAI_WIRE_TRANSPORT_VERSION}",
        )));
    }
    deserialize_value(&payload[OPENAI_WIRE_PREAMBLE_BYTES..])
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

    fn write_openai_wire_envelope<W: Write + ?Sized>(
        writer: &mut W,
        model_id: u32,
        op_code: u32,
        envelope: &OpenAiWireTransportEnvelope,
    ) -> Result<(), CodecError> {
        let payload = encode_openai_wire_transport_envelope(envelope)?;
        write_request_bytes(writer, model_id, op_code, &payload)
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

    pub fn openai_wire_over_stream<S: Read + Write + ?Sized>(
        conn: &mut S,
        model_id: u32,
        request: &OpenAiWireRequest,
    ) -> Result<OpenAiWireResponse, CodecError> {
        openai_wire_over_stream_with_auth(conn, model_id, request, None)
    }

    pub fn openai_wire_over_stream_authenticated<S: Read + Write + ?Sized>(
        conn: &mut S,
        model_id: u32,
        request: &OpenAiWireRequest,
        auth_token: &str,
    ) -> Result<OpenAiWireResponse, CodecError> {
        openai_wire_over_stream_with_auth(conn, model_id, request, Some(auth_token))
    }

    fn openai_wire_over_stream_with_auth<S: Read + Write + ?Sized>(
        conn: &mut S,
        model_id: u32,
        request: &OpenAiWireRequest,
        auth_token: Option<&str>,
    ) -> Result<OpenAiWireResponse, CodecError> {
        if request.format != OpenAiWireFormat::Json {
            return Err(CodecError::Serialize(
                "non-streaming OpenAI wire operation requires JSON format".to_string(),
            ));
        }
        request
            .validate(MAX_OPENAI_WIRE_REQUEST_PAYLOAD_BYTES)
            .map_err(|error| CodecError::Serialize(error.to_string()))?;
        let envelope = OpenAiWireTransportEnvelope {
            request: request.clone(),
            auth_token: auth_token.map(str::to_string),
        };
        write_openai_wire_envelope(conn, model_id, OP_OPENAI_WIRE, &envelope)?;
        let response: OpenAiWireResponse =
            read_response_value(conn, DEFAULT_MAX_FRAME_PAYLOAD_BYTES)?;
        response
            .head
            .validate()
            .map_err(|error| CodecError::Deserialize(error.to_string()))?;
        Ok(response)
    }

    pub fn write_openai_wire_stream_request<W: Write + ?Sized>(
        writer: &mut W,
        model_id: u32,
        request: &OpenAiWireRequest,
    ) -> Result<(), CodecError> {
        write_openai_wire_stream_request_with_auth(writer, model_id, request, None)
    }

    pub fn write_openai_wire_stream_request_authenticated<W: Write + ?Sized>(
        writer: &mut W,
        model_id: u32,
        request: &OpenAiWireRequest,
        auth_token: &str,
    ) -> Result<(), CodecError> {
        write_openai_wire_stream_request_with_auth(writer, model_id, request, Some(auth_token))
    }

    fn write_openai_wire_stream_request_with_auth<W: Write + ?Sized>(
        writer: &mut W,
        model_id: u32,
        request: &OpenAiWireRequest,
        auth_token: Option<&str>,
    ) -> Result<(), CodecError> {
        if request.format != OpenAiWireFormat::ServerSentEvents {
            return Err(CodecError::Serialize(
                "streaming OpenAI wire operation requires SSE format".to_string(),
            ));
        }
        request
            .validate(MAX_OPENAI_WIRE_REQUEST_PAYLOAD_BYTES)
            .map_err(|error| CodecError::Serialize(error.to_string()))?;
        let envelope = OpenAiWireTransportEnvelope {
            request: request.clone(),
            auth_token: auth_token.map(str::to_string),
        };
        write_openai_wire_envelope(writer, model_id, OP_OPENAI_WIRE_STREAM, &envelope)
    }

    pub fn read_openai_wire_stream_frame<R: Read + ?Sized>(
        reader: &mut R,
        max_payload_bytes: usize,
    ) -> Result<OpenAiWireStreamFrame, CodecError> {
        let frame = read_response_frame(reader, max_payload_bytes)?;
        match frame.header.status {
            STATUS_OPENAI_WIRE_HEAD => {
                let head: OpenAiWireResponseHead = frame.deserialize()?;
                head.validate()
                    .map_err(|error| CodecError::Deserialize(error.to_string()))?;
                Ok(OpenAiWireStreamFrame::Head(head))
            }
            STATUS_OPENAI_WIRE_CHUNK => Ok(OpenAiWireStreamFrame::Chunk(frame.payload)),
            STATUS_STREAM_END => Ok(OpenAiWireStreamFrame::End),
            STATUS_ERR => Err(frame.remote_error()),
            status => Err(CodecError::UnexpectedStatus(status)),
        }
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

    async fn write_openai_wire_envelope<W: AsyncWrite + Unpin + ?Sized>(
        writer: &mut W,
        model_id: u32,
        op_code: u32,
        envelope: &OpenAiWireTransportEnvelope,
    ) -> Result<(), CodecError> {
        let payload = encode_openai_wire_transport_envelope(envelope)?;
        write_request_bytes(writer, model_id, op_code, &payload).await
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
        read_request_frame_or_eof_with_operation_limits(reader, max_payload_bytes, &[]).await
    }

    /// Read a request while applying tighter pre-allocation ceilings to
    /// selected operation codes. The fixed header is decoded first, so a
    /// restricted operation is rejected before reserving or reading its
    /// peer-controlled payload.
    pub async fn read_request_frame_or_eof_with_operation_limits<R: AsyncRead + Unpin + ?Sized>(
        reader: &mut R,
        default_max_payload_bytes: usize,
        operation_limits: &[(u32, usize)],
    ) -> Result<Option<RequestFrame>, CodecError> {
        let mut header_bytes = [0; REQUEST_HEADER_BYTES];
        match reader.read(&mut header_bytes[..1]).await? {
            0 => return Ok(None),
            1 => {}
            _ => unreachable!("one-byte read"),
        }
        reader.read_exact(&mut header_bytes[1..]).await?;
        let header = decode_request_header(header_bytes);
        let max_payload_bytes = operation_limits
            .iter()
            .find_map(|(operation, limit)| (*operation == header.op_code).then_some(*limit))
            .unwrap_or(default_max_payload_bytes)
            .min(default_max_payload_bytes);
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

    pub async fn openai_wire_over_stream<S: AsyncRead + AsyncWrite + Unpin + ?Sized>(
        conn: &mut S,
        model_id: u32,
        request: &OpenAiWireRequest,
    ) -> Result<OpenAiWireResponse, CodecError> {
        openai_wire_over_stream_with_auth(conn, model_id, request, None).await
    }

    pub async fn openai_wire_over_stream_authenticated<
        S: AsyncRead + AsyncWrite + Unpin + ?Sized,
    >(
        conn: &mut S,
        model_id: u32,
        request: &OpenAiWireRequest,
        auth_token: &str,
    ) -> Result<OpenAiWireResponse, CodecError> {
        openai_wire_over_stream_with_auth(conn, model_id, request, Some(auth_token)).await
    }

    async fn openai_wire_over_stream_with_auth<S: AsyncRead + AsyncWrite + Unpin + ?Sized>(
        conn: &mut S,
        model_id: u32,
        request: &OpenAiWireRequest,
        auth_token: Option<&str>,
    ) -> Result<OpenAiWireResponse, CodecError> {
        if request.format != OpenAiWireFormat::Json {
            return Err(CodecError::Serialize(
                "non-streaming OpenAI wire operation requires JSON format".to_string(),
            ));
        }
        request
            .validate(MAX_OPENAI_WIRE_REQUEST_PAYLOAD_BYTES)
            .map_err(|error| CodecError::Serialize(error.to_string()))?;
        let envelope = OpenAiWireTransportEnvelope {
            request: request.clone(),
            auth_token: auth_token.map(str::to_string),
        };
        write_openai_wire_envelope(conn, model_id, OP_OPENAI_WIRE, &envelope).await?;
        let response: OpenAiWireResponse =
            read_response_value(conn, DEFAULT_MAX_FRAME_PAYLOAD_BYTES).await?;
        response
            .head
            .validate()
            .map_err(|error| CodecError::Deserialize(error.to_string()))?;
        Ok(response)
    }

    pub async fn write_openai_wire_stream_request<W: AsyncWrite + Unpin + ?Sized>(
        writer: &mut W,
        model_id: u32,
        request: &OpenAiWireRequest,
    ) -> Result<(), CodecError> {
        write_openai_wire_stream_request_with_auth(writer, model_id, request, None).await
    }

    pub async fn write_openai_wire_stream_request_authenticated<W: AsyncWrite + Unpin + ?Sized>(
        writer: &mut W,
        model_id: u32,
        request: &OpenAiWireRequest,
        auth_token: &str,
    ) -> Result<(), CodecError> {
        write_openai_wire_stream_request_with_auth(writer, model_id, request, Some(auth_token))
            .await
    }

    async fn write_openai_wire_stream_request_with_auth<W: AsyncWrite + Unpin + ?Sized>(
        writer: &mut W,
        model_id: u32,
        request: &OpenAiWireRequest,
        auth_token: Option<&str>,
    ) -> Result<(), CodecError> {
        if request.format != OpenAiWireFormat::ServerSentEvents {
            return Err(CodecError::Serialize(
                "streaming OpenAI wire operation requires SSE format".to_string(),
            ));
        }
        request
            .validate(MAX_OPENAI_WIRE_REQUEST_PAYLOAD_BYTES)
            .map_err(|error| CodecError::Serialize(error.to_string()))?;
        let envelope = OpenAiWireTransportEnvelope {
            request: request.clone(),
            auth_token: auth_token.map(str::to_string),
        };
        write_openai_wire_envelope(writer, model_id, OP_OPENAI_WIRE_STREAM, &envelope).await
    }

    pub async fn read_openai_wire_stream_frame<R: AsyncRead + Unpin + ?Sized>(
        reader: &mut R,
        max_payload_bytes: usize,
    ) -> Result<OpenAiWireStreamFrame, CodecError> {
        let frame = read_response_frame(reader, max_payload_bytes).await?;
        match frame.header.status {
            STATUS_OPENAI_WIRE_HEAD => {
                let head: OpenAiWireResponseHead = frame.deserialize()?;
                head.validate()
                    .map_err(|error| CodecError::Deserialize(error.to_string()))?;
                Ok(OpenAiWireStreamFrame::Head(head))
            }
            STATUS_OPENAI_WIRE_CHUNK => Ok(OpenAiWireStreamFrame::Chunk(frame.payload)),
            STATUS_STREAM_END => Ok(OpenAiWireStreamFrame::End),
            STATUS_ERR => Err(frame.remote_error()),
            status => Err(CodecError::UnexpectedStatus(status)),
        }
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
    use kapsl_engine_api::{
        OpenAiWireEndpoint, OpenAiWireHeader, OpenAiWireHeaderName, TensorDtype,
    };
    use std::io::Cursor;
    use tokio::io::{duplex, AsyncReadExt};

    // Full request frame for an unauthenticated v1 chat/SSE request with body
    // `{}` and model ID 0x01020304. This freezes both the explicit transport
    // preamble and the v1 bincode envelope that follows it.
    const OPENAI_WIRE_V1_STREAM_FRAME: &[u8] = &[
        0x04, 0x03, 0x02, 0x01, // model ID, little endian
        0x06, 0x00, 0x00, 0x00, // OP_OPENAI_WIRE_STREAM
        0x1d, 0x00, 0x00, 0x00, // 29-byte payload
        b'K', b'O', b'W', b'R', // transport payload magic
        0x01, 0x00, // outer transport version, little endian
        0x01, 0x00, // engine-facing request version
        0x00, 0x00, 0x00, 0x00, // ChatCompletions
        0x01, 0x00, 0x00, 0x00, // ServerSentEvents
        0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // body length
        b'{', b'}', 0x00, // body, no session ID
        0x00, // no policy metadata
        0x00, // no transport auth token
    ];

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
    fn blocking_openai_wire_v1_request_matches_frozen_frame() {
        let request = OpenAiWireRequest::new(
            OpenAiWireEndpoint::ChatCompletions,
            OpenAiWireFormat::ServerSentEvents,
            b"{}".to_vec(),
        );
        let mut encoded = Vec::new();

        blocking::write_openai_wire_stream_request(&mut encoded, 0x0102_0304, &request).unwrap();

        assert_eq!(encoded, OPENAI_WIRE_V1_STREAM_FRAME);
        let frame = blocking::read_request_frame(
            &mut Cursor::new(OPENAI_WIRE_V1_STREAM_FRAME),
            DEFAULT_MAX_FRAME_PAYLOAD_BYTES,
        )
        .unwrap();
        assert_eq!(frame.decode_openai_wire_request().unwrap().body, b"{}");
    }

    #[tokio::test]
    async fn asynchronous_openai_wire_v1_request_matches_frozen_frame() {
        let request = OpenAiWireRequest::new(
            OpenAiWireEndpoint::ChatCompletions,
            OpenAiWireFormat::ServerSentEvents,
            b"{}".to_vec(),
        );
        let (mut writer, mut reader) = duplex(128);

        asynchronous::write_openai_wire_stream_request(&mut writer, 0x0102_0304, &request)
            .await
            .unwrap();
        let mut encoded = vec![0; OPENAI_WIRE_V1_STREAM_FRAME.len()];
        reader.read_exact(&mut encoded).await.unwrap();

        assert_eq!(encoded, OPENAI_WIRE_V1_STREAM_FRAME);
    }

    #[test]
    fn openai_wire_client_limits_the_complete_encoded_envelope() {
        let request = OpenAiWireRequest::new(
            OpenAiWireEndpoint::ChatCompletions,
            OpenAiWireFormat::ServerSentEvents,
            vec![b'x'; MAX_OPENAI_WIRE_REQUEST_PAYLOAD_BYTES],
        );
        let mut encoded = Vec::new();

        let error = blocking::write_openai_wire_stream_request(&mut encoded, 7, &request)
            .expect_err("the request body leaves no room for its versioned envelope");

        assert!(matches!(
            error,
            CodecError::PayloadTooLarge { size, max }
                if size > max && max == MAX_OPENAI_WIRE_REQUEST_PAYLOAD_BYTES
        ));
        assert!(encoded.is_empty());
    }

    #[test]
    fn unknown_openai_wire_outer_version_is_rejected_before_bincode() {
        let mut payload = OPENAI_WIRE_PREAMBLE_MAGIC.to_vec();
        payload.extend_from_slice(&(OPENAI_WIRE_TRANSPORT_VERSION + 1).to_le_bytes());
        // Deliberately omit a bincode envelope. The stable version error proves
        // the outer preamble is inspected before deserialization is attempted.
        let frame = RequestFrame {
            header: RequestHeader {
                model_id: 7,
                op_code: OP_OPENAI_WIRE,
                payload_size: payload.len() as u32,
            },
            payload,
        };

        let error = frame.decode_openai_wire_envelope().unwrap_err();
        assert!(matches!(
            error,
            CodecError::Deserialize(message)
                if message == "unsupported OpenAI wire transport version 2; expected 1"
        ));
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
    async fn operation_limit_rejects_wire_payload_before_reading_or_allocating_it() {
        let (mut peer, mut server) = duplex(64);
        peer.write_all(&encode_request_header(RequestHeader {
            model_id: 1,
            op_code: OP_OPENAI_WIRE,
            payload_size: 17,
        }))
        .await
        .unwrap();

        let error = tokio::time::timeout(
            std::time::Duration::from_millis(100),
            asynchronous::read_request_frame_or_eof_with_operation_limits(
                &mut server,
                DEFAULT_MAX_FRAME_PAYLOAD_BYTES,
                &[(OP_OPENAI_WIRE, 16)],
            ),
        )
        .await
        .expect("restricted frame must fail from its header without waiting for payload bytes")
        .unwrap_err();
        assert!(matches!(
            error,
            CodecError::PayloadTooLarge { size: 17, max: 16 }
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
    async fn openai_wire_stream_uses_distinct_operation_and_raw_byte_frames() {
        let (mut client, mut server) = duplex(4096);
        let request = OpenAiWireRequest::new(
            OpenAiWireEndpoint::ChatCompletions,
            OpenAiWireFormat::ServerSentEvents,
            br#"{"model":"served","stream":true}"#.to_vec(),
        );
        let server_task = tokio::spawn(async move {
            let frame =
                asynchronous::read_request_frame(&mut server, DEFAULT_MAX_FRAME_PAYLOAD_BYTES)
                    .await
                    .unwrap();
            assert_eq!(frame.header.model_id, 17);
            assert_eq!(frame.header.op_code, OP_OPENAI_WIRE_STREAM);
            let decoded = frame.decode_openai_wire_request().unwrap();
            assert_eq!(decoded.body, br#"{"model":"served","stream":true}"#);

            let head = OpenAiWireResponseHead::new(
                200,
                vec![OpenAiWireHeader::new(
                    OpenAiWireHeaderName::ContentType,
                    b"text/event-stream".to_vec(),
                )
                .unwrap()],
            )
            .unwrap();
            asynchronous::write_response_value(&mut server, STATUS_OPENAI_WIRE_HEAD, &head)
                .await
                .unwrap();
            asynchronous::write_response_bytes(
                &mut server,
                STATUS_OPENAI_WIRE_CHUNK,
                b"data: {\"delta\":\"a\"}\n\n\0raw",
            )
            .await
            .unwrap();
            asynchronous::write_response_bytes(&mut server, STATUS_STREAM_END, &[])
                .await
                .unwrap();
        });

        asynchronous::write_openai_wire_stream_request(&mut client, 17, &request)
            .await
            .unwrap();
        assert!(matches!(
            asynchronous::read_openai_wire_stream_frame(
                &mut client,
                DEFAULT_MAX_FRAME_PAYLOAD_BYTES
            )
            .await
            .unwrap(),
            OpenAiWireStreamFrame::Head(head) if head.status == 200
        ));
        assert!(matches!(
            asynchronous::read_openai_wire_stream_frame(
                &mut client,
                DEFAULT_MAX_FRAME_PAYLOAD_BYTES
            )
            .await
            .unwrap(),
            OpenAiWireStreamFrame::Chunk(bytes)
                if bytes == b"data: {\"delta\":\"a\"}\n\n\0raw"
        ));
        assert!(matches!(
            asynchronous::read_openai_wire_stream_frame(
                &mut client,
                DEFAULT_MAX_FRAME_PAYLOAD_BYTES
            )
            .await
            .unwrap(),
            OpenAiWireStreamFrame::End
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
