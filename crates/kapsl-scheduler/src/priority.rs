use kapsl_engine_api::{InferenceRequest, OpenAiWireMetadata, OpenAiWireRequest, RequestMetadata};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Priority {
    LatencyCritical,
    Throughput,
}

impl Priority {
    /// Engine metadata convention: 0 = latency-critical, 1 = throughput.
    pub(crate) fn engine_metadata_value(self) -> u8 {
        match self {
            Self::LatencyCritical => 0,
            Self::Throughput => 1,
        }
    }
}

pub(crate) fn stamp_engine_priority(request: &mut InferenceRequest, priority: Priority) {
    request
        .metadata
        .get_or_insert_with(RequestMetadata::default)
        .priority = Some(priority.engine_metadata_value());
}

pub(crate) fn stamp_openai_wire_priority(request: &mut OpenAiWireRequest, priority: Priority) {
    request
        .metadata
        .get_or_insert_with(OpenAiWireMetadata::default)
        .priority = Some(priority.engine_metadata_value());
}
