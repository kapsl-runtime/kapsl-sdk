pub use kapsl_transport::protocol::{
    RequestHeader, ResponseHeader, OP_HYBRID_INFER, OP_INFER, OP_INFER_STREAM, OP_METRICS,
    OP_OPENAI_WIRE, OP_OPENAI_WIRE_STREAM, STATUS_ERR, STATUS_OK, STATUS_OPENAI_WIRE_CHUNK,
    STATUS_OPENAI_WIRE_HEAD, STATUS_STREAM_CHUNK, STATUS_STREAM_END,
};
use kapsl_transport::{RequestMetadata, ResponseMetadata};

use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug)]
pub struct HybridRequest {
    pub metadata: RequestMetadata,
    pub shm_offset: u64,
    pub shm_size: u64,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct HybridResponse {
    pub metadata: ResponseMetadata,
    pub shm_offset: u64,
    pub shm_size: u64,
}
