pub use kapsl_transport::protocol::{
    RequestHeader, ResponseHeader, OP_HYBRID_INFER, OP_INFER, OP_INFER_STREAM, OP_METRICS,
    STATUS_ERR, STATUS_OK, STATUS_STREAM_CHUNK, STATUS_STREAM_END,
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
