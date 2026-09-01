//! Legacy llama.cpp native-pack ABI version 1.
//!
//! These names remain re-exported from the crate root so existing engine and
//! pack releases can adopt the published crate without a source or layout
//! change. New backend packs should use the backend-neutral API in
//! [`crate::KapslBackendApiV1`].

use std::ffi::c_void;

use crate::{KapslFreeBufferFn, KapslLogFn, KapslOwnedBuffer, KapslRequestCancelledFn, KapslSlice};

pub const KAPSL_LLAMA_CPP_ABI_VERSION: u32 = 1;
pub const KAPSL_LLAMA_CPP_ENTRYPOINT_MAGIC: u32 = 0x4b4c_4c4d; // KLLM
pub const KAPSL_LLAMA_CPP_ENTRYPOINT_SYMBOL: &[u8] = b"kapsl_llama_cpp_backend_v1\0";
pub const KAPSL_LLAMA_CPP_WIRE_FORMAT_JSON_V1: u32 = 1;

pub const KAPSL_LLAMA_CAP_CPU: u64 = 1 << 0;
pub const KAPSL_LLAMA_CAP_CUDA: u64 = 1 << 1;
pub const KAPSL_LLAMA_CAP_NATIVE_KV: u64 = 1 << 2;
pub const KAPSL_LLAMA_CAP_SHARED_POOL: u64 = 1 << 3;
pub const KAPSL_LLAMA_CAP_STREAMING: u64 = 1 << 4;
pub const KAPSL_LLAMA_CAP_CANCELLATION: u64 = 1 << 5;
pub const KAPSL_LLAMA_CAP_MEMORY_REPORTING: u64 = 1 << 6;

pub const KAPSL_LLAMA_PROFILE_CPU: u32 = 1;
pub const KAPSL_LLAMA_PROFILE_CUDA12: u32 = 2;

/// Geometry requested by a shared-pool-capable llama.cpp pack.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct KapslSharedPoolGeometryV1 {
    pub struct_size: u32,
    pub device_id: u32,
    pub requested_blocks: u64,
    pub block_size_tokens: u32,
    pub num_layers: u32,
    pub num_kv_heads: u32,
    pub key_head_dim: u32,
    pub value_head_dim: u32,
    pub element_bytes: u32,
    pub max_sequences: u32,
    pub max_blocks_per_sequence: u32,
    pub model_fingerprint: u64,
}

pub type KapslPoolReserveFn = unsafe extern "C" fn(
    pool_context: *mut c_void,
    session_id: u64,
    tokens_needed: u32,
    block_table_device_out: *mut *mut u32,
    blocks_out: *mut u32,
) -> u32;
pub type KapslPoolReserveSequenceFn = KapslPoolReserveFn;
pub type KapslPoolCommitSequencesFn =
    unsafe extern "C" fn(pool_context: *mut c_void, block_table_device_out: *mut *mut u32) -> u32;
pub type KapslPoolReleaseFn = unsafe extern "C" fn(pool_context: *mut c_void, session_id: u64);
pub type KapslPoolTouchFn = unsafe extern "C" fn(pool_context: *mut c_void, session_id: u64) -> u32;

/// Runtime-owned storage and allocation callbacks. A pack may retain this only
/// until the matching host `destroy_shared_pool` callback returns.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct KapslSharedPoolDescriptorV1 {
    pub struct_size: u32,
    pub pool_context: *mut c_void,
    pub device_base: *mut c_void,
    pub addressable_blocks: u64,
    pub block_table_device: *mut u32,
    pub block_table_layer_stride: u32,
    pub block_table_sequence_stride: u32,
    pub sequence_slots: u32,
    pub reserve: Option<KapslPoolReserveFn>,
    pub reserve_sequence: Option<KapslPoolReserveSequenceFn>,
    pub commit_sequences: Option<KapslPoolCommitSequencesFn>,
    pub release: Option<KapslPoolReleaseFn>,
    pub touch: Option<KapslPoolTouchFn>,
}

pub type KapslCreateSharedPoolFn = unsafe extern "C" fn(
    user_data: *mut c_void,
    geometry: *const KapslSharedPoolGeometryV1,
    descriptor_out: *mut KapslSharedPoolDescriptorV1,
    error_out: *mut KapslOwnedBuffer,
) -> i32;
pub type KapslDestroySharedPoolFn =
    unsafe extern "C" fn(user_data: *mut c_void, pool_context: *mut c_void);
pub type KapslSharedPoolBytesFn =
    unsafe extern "C" fn(user_data: *mut c_void, pool_context: *mut c_void) -> u64;

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct KapslLlamaHostCallbacksV1 {
    pub struct_size: u32,
    pub user_data: *mut c_void,
    pub log: Option<KapslLogFn>,
    pub create_shared_pool: Option<KapslCreateSharedPoolFn>,
    pub destroy_shared_pool: Option<KapslDestroySharedPoolFn>,
    pub shared_pool_bytes: Option<KapslSharedPoolBytesFn>,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct KapslLlamaConfigV1 {
    pub struct_size: u32,
    pub profile: u32,
    pub device_id: u32,
    pub model_id: u32,
    pub replica_id: u32,
    /// Fail initialization unless all KV allocations use host callbacks.
    pub require_shared_pool: u32,
    /// This callback table and its context remain live until `shutdown` returns.
    pub host: *const KapslLlamaHostCallbacksV1,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct KapslLlamaRequestV1 {
    pub struct_size: u32,
    pub wire_format: u32,
    pub request_id: u64,
    pub request_json: KapslSlice,
    pub cancellation_context: *mut c_void,
    pub is_cancelled: Option<KapslRequestCancelledFn>,
}

pub type KapslLlamaStreamChunkFn =
    unsafe extern "C" fn(user_data: *mut c_void, request_id: u64, packet_json: KapslSlice) -> i32;

pub type KapslInitializeFn = unsafe extern "C" fn(
    config: *const KapslLlamaConfigV1,
    handle_out: *mut *mut c_void,
    error_out: *mut KapslOwnedBuffer,
) -> i32;
pub type KapslPathReportFn = unsafe extern "C" fn(
    handle: *mut c_void,
    model_path: KapslSlice,
    report_out: *mut KapslOwnedBuffer,
    error_out: *mut KapslOwnedBuffer,
) -> i32;
pub type KapslLoadModelFn = unsafe extern "C" fn(
    handle: *mut c_void,
    model_path: KapslSlice,
    error_out: *mut KapslOwnedBuffer,
) -> i32;
pub type KapslRequestReportFn = unsafe extern "C" fn(
    handle: *mut c_void,
    request: *const KapslLlamaRequestV1,
    report_out: *mut KapslOwnedBuffer,
    error_out: *mut KapslOwnedBuffer,
) -> i32;
pub type KapslInferFn = unsafe extern "C" fn(
    handle: *mut c_void,
    request: *const KapslLlamaRequestV1,
    packet_out: *mut KapslOwnedBuffer,
    error_out: *mut KapslOwnedBuffer,
) -> i32;
pub type KapslInferStreamFn = unsafe extern "C" fn(
    handle: *mut c_void,
    request: *const KapslLlamaRequestV1,
    user_data: *mut c_void,
    on_chunk: Option<KapslLlamaStreamChunkFn>,
    error_out: *mut KapslOwnedBuffer,
) -> i32;
pub type KapslCancelFn = unsafe extern "C" fn(handle: *mut c_void, request_id: u64) -> i32;
pub type KapslJsonReportFn = unsafe extern "C" fn(
    handle: *mut c_void,
    report_out: *mut KapslOwnedBuffer,
    error_out: *mut KapslOwnedBuffer,
) -> i32;
pub type KapslHealthFn =
    unsafe extern "C" fn(handle: *mut c_void, error_out: *mut KapslOwnedBuffer) -> i32;
pub type KapslUnloadFn = unsafe extern "C" fn(handle: *mut c_void);
pub type KapslShutdownFn = unsafe extern "C" fn(handle: *mut c_void);

/// Versioned function table exported by every llama.cpp native pack.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct KapslLlamaCppApiV1 {
    pub magic: u32,
    pub abi_version: u32,
    pub struct_size: u32,
    pub wire_format: u32,
    pub capabilities: u64,
    pub initialize: Option<KapslInitializeFn>,
    pub planned_memory: Option<KapslPathReportFn>,
    pub load_model: Option<KapslLoadModelFn>,
    pub planned_request_memory: Option<KapslRequestReportFn>,
    pub infer: Option<KapslInferFn>,
    pub infer_stream: Option<KapslInferStreamFn>,
    pub cancel: Option<KapslCancelFn>,
    pub actual_memory: Option<KapslJsonReportFn>,
    pub metrics: Option<KapslJsonReportFn>,
    pub model_info: Option<KapslJsonReportFn>,
    pub kv_capabilities: Option<KapslJsonReportFn>,
    pub kv_topology: Option<KapslJsonReportFn>,
    pub batching_policy: Option<KapslJsonReportFn>,
    pub health_check: Option<KapslHealthFn>,
    pub unload: Option<KapslUnloadFn>,
    pub shutdown: Option<KapslShutdownFn>,
    pub free_buffer: Option<KapslFreeBufferFn>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn legacy_layout_and_versions_remain_stable() {
        assert_eq!(KAPSL_LLAMA_CPP_ABI_VERSION, 1);
        assert_eq!(KAPSL_LLAMA_CPP_WIRE_FORMAT_JSON_V1, 1);
        assert!(std::mem::size_of::<KapslLlamaCppApiV1>() >= 16 * std::mem::size_of::<usize>());
        assert!(std::mem::size_of::<KapslLlamaConfigV1>() >= 6 * std::mem::size_of::<u32>());
    }
}
