use kapsl_engine_api::{InferenceRequest, OpenAiWireMetadata, OpenAiWireRequest, RequestMetadata};

/// Defined in `kapsl-engine-api` so transports can use it without this crate.
pub use kapsl_engine_api::Priority;

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
