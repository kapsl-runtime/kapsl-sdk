//! Backend-neutral native plugin ABI version 1.

use std::ffi::c_void;

use crate::{KapslFreeBufferFn, KapslLogFn, KapslOwnedBuffer, KapslRequestCancelledFn, KapslSlice};

pub const KAPSL_BACKEND_ABI_VERSION: u32 = 1;
pub const KAPSL_BACKEND_ENTRYPOINT_MAGIC: u32 = 0x4b42_4e44; // KBND
pub const KAPSL_BACKEND_ENTRYPOINT_SYMBOL: &[u8] = b"kapsl_backend_v1\0";
pub const KAPSL_BACKEND_WIRE_FORMAT_TENSORS_V1: u32 = 1;

pub type KapslBackendEntrypointV1 = unsafe extern "C" fn() -> *const KapslBackendApiV1;

pub const KAPSL_BACKEND_CAP_CPU: u64 = 1 << 0;
pub const KAPSL_BACKEND_CAP_CUDA: u64 = 1 << 1;
pub const KAPSL_BACKEND_CAP_TENSORRT: u64 = 1 << 2;
pub const KAPSL_BACKEND_CAP_BATCHING: u64 = 1 << 3;
pub const KAPSL_BACKEND_CAP_STREAMING: u64 = 1 << 4;
pub const KAPSL_BACKEND_CAP_CANCELLATION: u64 = 1 << 5;
pub const KAPSL_BACKEND_CAP_MEMORY_REPORTING: u64 = 1 << 6;
pub const KAPSL_BACKEND_CAP_GOVERNED_DEVICE_ALLOCATOR: u64 = 1 << 7;
pub const KAPSL_BACKEND_CAP_KV_PARTICIPANT: u64 = 1 << 8;
pub const KAPSL_BACKEND_CAP_CONCURRENT_INFERENCE: u64 = 1 << 9;

pub const KAPSL_MEMORY_HOST: u32 = 1;
pub const KAPSL_MEMORY_HOST_PINNED: u32 = 2;
pub const KAPSL_MEMORY_CUDA: u32 = 3;
pub const KAPSL_MEMORY_PROVIDER: u32 = 4;

pub const KAPSL_DTYPE_BOOL: u32 = 1;
pub const KAPSL_DTYPE_U8: u32 = 2;
pub const KAPSL_DTYPE_I8: u32 = 3;
pub const KAPSL_DTYPE_U16: u32 = 4;
pub const KAPSL_DTYPE_I16: u32 = 5;
pub const KAPSL_DTYPE_U32: u32 = 6;
pub const KAPSL_DTYPE_I32: u32 = 7;
pub const KAPSL_DTYPE_U64: u32 = 8;
pub const KAPSL_DTYPE_I64: u32 = 9;
pub const KAPSL_DTYPE_F16: u32 = 10;
pub const KAPSL_DTYPE_BF16: u32 = 11;
pub const KAPSL_DTYPE_F32: u32 = 12;
pub const KAPSL_DTYPE_F64: u32 = 13;

pub const KAPSL_TENSOR_FLAG_CONTIGUOUS: u32 = 1 << 0;
pub const KAPSL_TENSOR_FLAG_READ_ONLY: u32 = 1 << 1;

pub const KAPSL_ALLOCATION_CLASS_WEIGHTS: u32 = 1;
pub const KAPSL_ALLOCATION_CLASS_WORKSPACE: u32 = 2;
pub const KAPSL_ALLOCATION_CLASS_KV: u32 = 3;
pub const KAPSL_ALLOCATION_CLASS_REQUEST: u32 = 4;
pub const KAPSL_ALLOCATION_CLASS_OTHER: u32 = 5;

/// One governed device allocation requested by a native backend.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct KapslDeviceAllocationRequestV1 {
    pub struct_size: u32,
    pub device_id: u32,
    pub memory_kind: u32,
    pub allocation_class: u32,
    pub model_id: u32,
    pub replica_id: u32,
    pub flags: u32,
    pub reserved: u32,
    pub bytes: u64,
    pub alignment: u64,
}

impl KapslDeviceAllocationRequestV1 {
    pub const fn new(
        device_id: u32,
        memory_kind: u32,
        allocation_class: u32,
        model_id: u32,
        replica_id: u32,
        bytes: u64,
        alignment: u64,
    ) -> Self {
        Self {
            struct_size: std::mem::size_of::<Self>() as u32,
            device_id,
            memory_kind,
            allocation_class,
            model_id,
            replica_id,
            flags: 0,
            reserved: 0,
            bytes,
            alignment,
        }
    }
}

/// Allocation identity returned by the Kapsl host.
///
/// The backend may copy this value, but must return the exact identity to the
/// matching host `free_device` callback. `allocation_id` is opaque and is the
/// authority for release; the pointer is repeated so the host can reject a
/// mismatched or corrupted free.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct KapslDeviceAllocationV1 {
    pub struct_size: u32,
    pub reserved: u32,
    pub allocation_id: u64,
    pub device_ptr: *mut c_void,
    pub granted_bytes: u64,
}

impl KapslDeviceAllocationV1 {
    pub const fn empty() -> Self {
        Self {
            struct_size: std::mem::size_of::<Self>() as u32,
            reserved: 0,
            allocation_id: 0,
            device_ptr: std::ptr::null_mut(),
            granted_bytes: 0,
        }
    }
}

impl Default for KapslDeviceAllocationV1 {
    fn default() -> Self {
        Self::empty()
    }
}

pub type KapslDeviceAllocateFn = unsafe extern "C" fn(
    user_data: *mut c_void,
    request: *const KapslDeviceAllocationRequestV1,
    allocation_out: *mut KapslDeviceAllocationV1,
) -> i32;

pub type KapslDeviceFreeFn =
    unsafe extern "C" fn(user_data: *mut c_void, allocation: *const KapslDeviceAllocationV1) -> i32;

pub type KapslDeviceSynchronizeFn =
    unsafe extern "C" fn(user_data: *mut c_void, device_id: u32) -> i32;

/// Host services whose storage remains valid until adapter `shutdown` returns.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct KapslBackendHostV1 {
    pub struct_size: u32,
    pub abi_version: u32,
    pub user_data: *mut c_void,
    pub log: Option<KapslLogFn>,
    pub allocate_device: Option<KapslDeviceAllocateFn>,
    pub free_device: Option<KapslDeviceFreeFn>,
    pub synchronize_device: Option<KapslDeviceSynchronizeFn>,
}

/// Configuration for one model/replica backend instance.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct KapslBackendConfigV1 {
    pub struct_size: u32,
    pub device_id: u32,
    pub model_id: u32,
    pub replica_id: u32,
    /// A governed profile must fail initialization unless all device allocation
    /// required by its certified path uses the supplied host callbacks.
    pub require_governed_device_memory: u32,
    pub reserved: u32,
    pub profile: KapslSlice,
    pub manifest_json: KapslSlice,
    pub options_json: KapslSlice,
    pub host: *const KapslBackendHostV1,
}

/// Borrowed tensor storage. Shape and stride values are element counts, while
/// `byte_len` is the bounded size of the data view. A null `strides` pointer
/// means standard contiguous strides. Device pointers are never dereferenced by
/// the host unless the declared memory kind is supported by that host.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct KapslTensorViewV1 {
    pub struct_size: u32,
    pub dtype: u32,
    pub memory_kind: u32,
    pub flags: u32,
    pub device_id: i32,
    pub rank: u32,
    pub shape: *const i64,
    pub strides: *const i64,
    pub data: *const c_void,
    pub byte_len: u64,
}

impl KapslTensorViewV1 {
    pub const fn empty() -> Self {
        Self {
            struct_size: std::mem::size_of::<Self>() as u32,
            dtype: 0,
            memory_kind: 0,
            flags: 0,
            device_id: -1,
            rank: 0,
            shape: std::ptr::null(),
            strides: std::ptr::null(),
            data: std::ptr::null(),
            byte_len: 0,
        }
    }
}

impl Default for KapslTensorViewV1 {
    fn default() -> Self {
        Self::empty()
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct KapslNamedTensorViewV1 {
    pub struct_size: u32,
    pub reserved: u32,
    pub name: KapslSlice,
    pub tensor: KapslTensorViewV1,
}

/// Synchronous inference request. Input storage and shape arrays remain valid
/// until `infer`, `infer_batch`, or `infer_stream` returns. The backend may poll
/// `is_cancelled` during that call but may not retain the callback context.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct KapslInferenceRequestV1 {
    pub struct_size: u32,
    pub wire_format: u32,
    pub request_id: u64,
    pub inputs: *const KapslNamedTensorViewV1,
    pub input_count: u32,
    pub reserved: u32,
    pub metadata_json: KapslSlice,
    pub cancellation_context: *mut c_void,
    pub is_cancelled: Option<KapslRequestCancelledFn>,
}

/// Adapter-owned inference result. All referenced storage remains valid until
/// the host calls `release_result` from the same backend function table.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct KapslInferenceResultV1 {
    pub struct_size: u32,
    pub output_count: u32,
    pub outputs: *const KapslNamedTensorViewV1,
    pub metadata_json: KapslSlice,
    pub owner_context: *mut c_void,
}

impl KapslInferenceResultV1 {
    pub const fn empty() -> Self {
        Self {
            struct_size: std::mem::size_of::<Self>() as u32,
            output_count: 0,
            outputs: std::ptr::null(),
            metadata_json: KapslSlice::empty(),
            owner_context: std::ptr::null_mut(),
        }
    }
}

impl Default for KapslInferenceResultV1 {
    fn default() -> Self {
        Self::empty()
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct KapslInferenceBatchV1 {
    pub struct_size: u32,
    pub request_count: u32,
    pub requests: *const KapslInferenceRequestV1,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct KapslInferenceBatchResultV1 {
    pub struct_size: u32,
    pub result_count: u32,
    pub results: *const KapslInferenceResultV1,
    pub owner_context: *mut c_void,
}

impl KapslInferenceBatchResultV1 {
    pub const fn empty() -> Self {
        Self {
            struct_size: std::mem::size_of::<Self>() as u32,
            result_count: 0,
            results: std::ptr::null(),
            owner_context: std::ptr::null_mut(),
        }
    }
}

impl Default for KapslInferenceBatchResultV1 {
    fn default() -> Self {
        Self::empty()
    }
}

pub type KapslBackendDescribeFn = unsafe extern "C" fn(
    descriptor_json_out: *mut KapslOwnedBuffer,
    error_out: *mut KapslOwnedBuffer,
) -> i32;

pub type KapslBackendInitializeFn = unsafe extern "C" fn(
    config: *const KapslBackendConfigV1,
    handle_out: *mut *mut c_void,
    error_out: *mut KapslOwnedBuffer,
) -> i32;

pub type KapslBackendPathReportFn = unsafe extern "C" fn(
    handle: *mut c_void,
    model_path: KapslSlice,
    report_json_out: *mut KapslOwnedBuffer,
    error_out: *mut KapslOwnedBuffer,
) -> i32;

pub type KapslBackendLoadModelFn = unsafe extern "C" fn(
    handle: *mut c_void,
    model_path: KapslSlice,
    error_out: *mut KapslOwnedBuffer,
) -> i32;

pub type KapslBackendRequestReportFn = unsafe extern "C" fn(
    handle: *mut c_void,
    request: *const KapslInferenceRequestV1,
    report_json_out: *mut KapslOwnedBuffer,
    error_out: *mut KapslOwnedBuffer,
) -> i32;

pub type KapslBackendInferFn = unsafe extern "C" fn(
    handle: *mut c_void,
    request: *const KapslInferenceRequestV1,
    result_out: *mut KapslInferenceResultV1,
    error_out: *mut KapslOwnedBuffer,
) -> i32;

pub type KapslBackendInferBatchFn = unsafe extern "C" fn(
    handle: *mut c_void,
    batch: *const KapslInferenceBatchV1,
    result_out: *mut KapslInferenceBatchResultV1,
    error_out: *mut KapslOwnedBuffer,
) -> i32;

/// Streamed results are borrowed for the duration of `on_chunk`. The callback
/// must copy anything it needs after returning and must not call
/// `release_result` for a borrowed chunk.
pub type KapslBackendStreamChunkFn = unsafe extern "C" fn(
    user_data: *mut c_void,
    request_id: u64,
    result: *const KapslInferenceResultV1,
) -> i32;

pub type KapslBackendInferStreamFn = unsafe extern "C" fn(
    handle: *mut c_void,
    request: *const KapslInferenceRequestV1,
    user_data: *mut c_void,
    on_chunk: Option<KapslBackendStreamChunkFn>,
    error_out: *mut KapslOwnedBuffer,
) -> i32;

pub type KapslBackendCancelFn = unsafe extern "C" fn(handle: *mut c_void, request_id: u64) -> i32;

pub type KapslBackendJsonReportFn = unsafe extern "C" fn(
    handle: *mut c_void,
    report_json_out: *mut KapslOwnedBuffer,
    error_out: *mut KapslOwnedBuffer,
) -> i32;

pub type KapslBackendHealthFn =
    unsafe extern "C" fn(handle: *mut c_void, error_out: *mut KapslOwnedBuffer) -> i32;

pub type KapslBackendUnloadFn =
    unsafe extern "C" fn(handle: *mut c_void, error_out: *mut KapslOwnedBuffer) -> i32;

pub type KapslBackendShutdownFn = unsafe extern "C" fn(handle: *mut c_void);

pub type KapslBackendReleaseResultFn =
    unsafe extern "C" fn(handle: *mut c_void, result: *mut KapslInferenceResultV1);

pub type KapslBackendReleaseBatchResultFn =
    unsafe extern "C" fn(handle: *mut c_void, result: *mut KapslInferenceBatchResultV1);

/// Backend-neutral function table exported by native adapter packs.
///
/// New fields may only be appended. A host must check `struct_size` before
/// reading an optional tail field and must validate the capability bit before
/// calling its function.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct KapslBackendApiV1 {
    pub magic: u32,
    pub abi_version: u32,
    pub struct_size: u32,
    pub wire_format: u32,
    pub capabilities: u64,
    pub describe: Option<KapslBackendDescribeFn>,
    pub initialize: Option<KapslBackendInitializeFn>,
    pub planned_memory: Option<KapslBackendPathReportFn>,
    pub load_model: Option<KapslBackendLoadModelFn>,
    pub planned_request_memory: Option<KapslBackendRequestReportFn>,
    pub infer: Option<KapslBackendInferFn>,
    pub infer_batch: Option<KapslBackendInferBatchFn>,
    pub infer_stream: Option<KapslBackendInferStreamFn>,
    pub cancel: Option<KapslBackendCancelFn>,
    pub actual_memory: Option<KapslBackendJsonReportFn>,
    pub metrics: Option<KapslBackendJsonReportFn>,
    pub model_info: Option<KapslBackendJsonReportFn>,
    pub kv_capabilities: Option<KapslBackendJsonReportFn>,
    pub kv_topology: Option<KapslBackendJsonReportFn>,
    pub batching_policy: Option<KapslBackendJsonReportFn>,
    pub health_check: Option<KapslBackendHealthFn>,
    pub unload: Option<KapslBackendUnloadFn>,
    pub shutdown: Option<KapslBackendShutdownFn>,
    pub release_result: Option<KapslBackendReleaseResultFn>,
    pub release_batch_result: Option<KapslBackendReleaseBatchResultFn>,
    pub free_buffer: Option<KapslFreeBufferFn>,
}

impl KapslBackendApiV1 {
    /// Structural compatibility check that does not call adapter code.
    pub fn is_compatible(&self) -> bool {
        self.magic == KAPSL_BACKEND_ENTRYPOINT_MAGIC
            && self.abi_version == KAPSL_BACKEND_ABI_VERSION
            && self.wire_format == KAPSL_BACKEND_WIRE_FORMAT_TENSORS_V1
            && self.struct_size >= std::mem::size_of::<Self>() as u32
    }

    /// Functions required by every loadable backend. Optional batching,
    /// streaming, cancellation, and KV functions remain capability-gated.
    pub fn has_required_functions(&self) -> bool {
        self.describe.is_some()
            && self.initialize.is_some()
            && self.planned_memory.is_some()
            && self.load_model.is_some()
            && self.planned_request_memory.is_some()
            && self.infer.is_some()
            && self.actual_memory.is_some()
            && self.metrics.is_some()
            && self.model_info.is_some()
            && self.batching_policy.is_some()
            && self.health_check.is_some()
            && self.unload.is_some()
            && self.shutdown.is_some()
            && self.release_result.is_some()
            && self.free_buffer.is_some()
    }

    /// Verify that every advertised optional capability has the corresponding
    /// callable surface. A host should reject inconsistent tables before
    /// invoking `initialize`.
    pub fn capabilities_are_consistent(&self) -> bool {
        (self.capabilities & KAPSL_BACKEND_CAP_BATCHING == 0
            || (self.infer_batch.is_some() && self.release_batch_result.is_some()))
            && (self.capabilities & KAPSL_BACKEND_CAP_STREAMING == 0 || self.infer_stream.is_some())
            && (self.capabilities & KAPSL_BACKEND_CAP_CANCELLATION == 0 || self.cancel.is_some())
            && (self.capabilities & KAPSL_BACKEND_CAP_KV_PARTICIPANT == 0
                || (self.kv_capabilities.is_some() && self.kv_topology.is_some()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    unsafe extern "C" fn describe(
        _descriptor_json_out: *mut KapslOwnedBuffer,
        _error_out: *mut KapslOwnedBuffer,
    ) -> i32 {
        crate::KAPSL_STATUS_OK
    }

    unsafe extern "C" fn initialize(
        _config: *const KapslBackendConfigV1,
        _handle_out: *mut *mut c_void,
        _error_out: *mut KapslOwnedBuffer,
    ) -> i32 {
        crate::KAPSL_STATUS_OK
    }

    unsafe extern "C" fn path_report(
        _handle: *mut c_void,
        _model_path: KapslSlice,
        _report_json_out: *mut KapslOwnedBuffer,
        _error_out: *mut KapslOwnedBuffer,
    ) -> i32 {
        crate::KAPSL_STATUS_OK
    }

    unsafe extern "C" fn load_model(
        _handle: *mut c_void,
        _model_path: KapslSlice,
        _error_out: *mut KapslOwnedBuffer,
    ) -> i32 {
        crate::KAPSL_STATUS_OK
    }

    unsafe extern "C" fn request_report(
        _handle: *mut c_void,
        _request: *const KapslInferenceRequestV1,
        _report_json_out: *mut KapslOwnedBuffer,
        _error_out: *mut KapslOwnedBuffer,
    ) -> i32 {
        crate::KAPSL_STATUS_OK
    }

    unsafe extern "C" fn infer(
        _handle: *mut c_void,
        _request: *const KapslInferenceRequestV1,
        _result_out: *mut KapslInferenceResultV1,
        _error_out: *mut KapslOwnedBuffer,
    ) -> i32 {
        crate::KAPSL_STATUS_OK
    }

    unsafe extern "C" fn report(
        _handle: *mut c_void,
        _report_json_out: *mut KapslOwnedBuffer,
        _error_out: *mut KapslOwnedBuffer,
    ) -> i32 {
        crate::KAPSL_STATUS_OK
    }

    unsafe extern "C" fn health(_handle: *mut c_void, _error_out: *mut KapslOwnedBuffer) -> i32 {
        crate::KAPSL_STATUS_OK
    }

    unsafe extern "C" fn unload(_handle: *mut c_void, _error_out: *mut KapslOwnedBuffer) -> i32 {
        crate::KAPSL_STATUS_OK
    }

    unsafe extern "C" fn shutdown(_handle: *mut c_void) {}

    unsafe extern "C" fn release_result(
        _handle: *mut c_void,
        _result: *mut KapslInferenceResultV1,
    ) {
    }

    unsafe extern "C" fn free_buffer(_buffer: KapslOwnedBuffer) {}

    fn complete_api() -> KapslBackendApiV1 {
        KapslBackendApiV1 {
            magic: KAPSL_BACKEND_ENTRYPOINT_MAGIC,
            abi_version: KAPSL_BACKEND_ABI_VERSION,
            struct_size: std::mem::size_of::<KapslBackendApiV1>() as u32,
            wire_format: KAPSL_BACKEND_WIRE_FORMAT_TENSORS_V1,
            capabilities: KAPSL_BACKEND_CAP_CPU | KAPSL_BACKEND_CAP_MEMORY_REPORTING,
            describe: Some(describe),
            initialize: Some(initialize),
            planned_memory: Some(path_report),
            load_model: Some(load_model),
            planned_request_memory: Some(request_report),
            infer: Some(infer),
            infer_batch: None,
            infer_stream: None,
            cancel: None,
            actual_memory: Some(report),
            metrics: Some(report),
            model_info: Some(report),
            kv_capabilities: None,
            kv_topology: None,
            batching_policy: Some(report),
            health_check: Some(health),
            unload: Some(unload),
            shutdown: Some(shutdown),
            release_result: Some(release_result),
            release_batch_result: None,
            free_buffer: Some(free_buffer),
        }
    }

    #[test]
    fn complete_table_is_compatible() {
        let api = complete_api();
        assert!(api.is_compatible());
        assert!(api.has_required_functions());
    }

    #[test]
    fn incompatible_version_and_truncated_table_are_rejected() {
        let mut api = complete_api();
        api.abi_version += 1;
        assert!(!api.is_compatible());

        api.abi_version = KAPSL_BACKEND_ABI_VERSION;
        api.struct_size -= 1;
        assert!(!api.is_compatible());
    }

    #[test]
    fn missing_required_function_is_rejected() {
        let mut api = complete_api();
        api.release_result = None;
        assert!(!api.has_required_functions());
    }

    #[test]
    fn advertised_capabilities_require_their_functions() {
        let mut api = complete_api();
        api.capabilities |= KAPSL_BACKEND_CAP_BATCHING;
        assert!(!api.capabilities_are_consistent());

        api.capabilities &= !KAPSL_BACKEND_CAP_BATCHING;
        api.capabilities |= KAPSL_BACKEND_CAP_CANCELLATION;
        assert!(!api.capabilities_are_consistent());

        api.capabilities &= !KAPSL_BACKEND_CAP_CANCELLATION;
        assert!(api.capabilities_are_consistent());
    }

    #[test]
    fn empty_owned_values_have_no_ambient_owner() {
        let allocation = KapslDeviceAllocationV1::empty();
        assert_eq!(allocation.allocation_id, 0);
        assert!(allocation.device_ptr.is_null());

        let result = KapslInferenceResultV1::empty();
        assert_eq!(result.output_count, 0);
        assert!(result.outputs.is_null());
        assert!(result.owner_context.is_null());
    }

    #[test]
    #[cfg(target_pointer_width = "64")]
    fn c_abi_layout_is_frozen_on_64_bit_targets() {
        use std::mem::{offset_of, size_of};

        assert_eq!(size_of::<KapslSlice>(), 16);
        assert_eq!(size_of::<KapslOwnedBuffer>(), 24);
        assert_eq!(size_of::<KapslDeviceAllocationRequestV1>(), 48);
        assert_eq!(size_of::<KapslDeviceAllocationV1>(), 32);
        assert_eq!(size_of::<KapslBackendHostV1>(), 48);
        assert_eq!(size_of::<KapslBackendConfigV1>(), 80);
        assert_eq!(size_of::<KapslTensorViewV1>(), 56);
        assert_eq!(size_of::<KapslNamedTensorViewV1>(), 80);
        assert_eq!(size_of::<KapslInferenceRequestV1>(), 64);
        assert_eq!(size_of::<KapslInferenceResultV1>(), 40);
        assert_eq!(size_of::<KapslInferenceBatchV1>(), 16);
        assert_eq!(size_of::<KapslInferenceBatchResultV1>(), 24);
        assert_eq!(offset_of!(KapslBackendApiV1, describe), 24);
        assert_eq!(size_of::<KapslBackendApiV1>(), 192);
    }
}
