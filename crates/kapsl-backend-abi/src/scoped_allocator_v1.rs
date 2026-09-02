//! Request-aware extension for the backend ABI v1 host allocator.
//!
//! The published [`crate::KapslBackendHostV1`] prefix remains frozen. Hosts
//! that support request-aware allocation pass a
//! [`KapslBackendHostScopedAllocatorV1`] whose `base` field starts at offset
//! zero and whose base `struct_size` covers the complete extension. Older
//! adapters continue to read only the v1 prefix.

use std::ffi::c_void;

use crate::{KapslBackendHostV1, KapslDeviceAllocationV1, KAPSL_BACKEND_ABI_VERSION};

pub const KAPSL_SCOPED_DEVICE_ALLOCATOR_VERSION: u32 = 1;

pub const KAPSL_ALLOCATION_SCOPE_MODEL: u32 = 1;
pub const KAPSL_ALLOCATION_SCOPE_REPLICA: u32 = 2;
pub const KAPSL_ALLOCATION_SCOPE_REQUEST: u32 = 3;
pub const KAPSL_ALLOCATION_SCOPE_REQUEST_BATCH: u32 = 4;

/// Logical ownership active while a backend asks the host for device memory.
///
/// `scope_id` is non-zero and unique for the lifetime of one backend instance.
/// Model and replica scopes have no request IDs, request scopes have exactly
/// one, and request-batch scopes have at least two. The request-ID slice is
/// borrowed only for the synchronous allocation callback. The model/replica
/// pair is the durable charge owner; request IDs identify the operation that
/// caused the allocation and do not shorten the allocation's lifetime. Only
/// the matching host free callback releases the allocation.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct KapslDeviceAllocationScopeV1 {
    pub struct_size: u32,
    pub scope_kind: u32,
    pub scope_id: u64,
    pub model_id: u32,
    pub replica_id: u32,
    pub request_count: u32,
    pub reserved: u32,
    pub request_ids: *const u64,
}

impl KapslDeviceAllocationScopeV1 {
    pub fn new(
        scope_kind: u32,
        scope_id: u64,
        model_id: u32,
        replica_id: u32,
        request_ids: &[u64],
    ) -> Self {
        Self {
            struct_size: std::mem::size_of::<Self>() as u32,
            scope_kind,
            scope_id,
            model_id,
            replica_id,
            request_count: request_ids.len() as u32,
            reserved: 0,
            request_ids: if request_ids.is_empty() {
                std::ptr::null()
            } else {
                request_ids.as_ptr()
            },
        }
    }

    /// Validate the fixed header and the pointer/count rules without
    /// dereferencing backend-provided storage.
    pub fn is_well_formed(&self) -> bool {
        if self.struct_size < std::mem::size_of::<Self>() as u32
            || self.scope_id == 0
            || self.reserved != 0
        {
            return false;
        }
        match self.scope_kind {
            KAPSL_ALLOCATION_SCOPE_MODEL | KAPSL_ALLOCATION_SCOPE_REPLICA => {
                self.request_count == 0 && self.request_ids.is_null()
            }
            KAPSL_ALLOCATION_SCOPE_REQUEST => {
                self.request_count == 1 && !self.request_ids.is_null()
            }
            KAPSL_ALLOCATION_SCOPE_REQUEST_BATCH => {
                self.request_count >= 2 && !self.request_ids.is_null()
            }
            _ => false,
        }
    }

    /// Borrow the request IDs for the duration of the host callback.
    ///
    /// # Safety
    ///
    /// The backend must keep `request_ids` readable for `request_count`
    /// elements until the synchronous allocation callback returns.
    pub unsafe fn request_ids(&self) -> Option<&[u64]> {
        if self.request_count == 0 {
            return self.request_ids.is_null().then_some(&[]);
        }
        if self.request_ids.is_null() {
            return None;
        }
        // SAFETY: upheld by the caller of this method.
        Some(unsafe { std::slice::from_raw_parts(self.request_ids, self.request_count as usize) })
    }
}

/// One request-aware governed device allocation.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct KapslScopedDeviceAllocationRequestV1 {
    pub struct_size: u32,
    pub device_id: u32,
    pub memory_kind: u32,
    pub allocation_class: u32,
    pub scope: KapslDeviceAllocationScopeV1,
    pub flags: u32,
    pub reserved: u32,
    pub bytes: u64,
    pub alignment: u64,
}

impl KapslScopedDeviceAllocationRequestV1 {
    pub const fn new(
        device_id: u32,
        memory_kind: u32,
        allocation_class: u32,
        scope: KapslDeviceAllocationScopeV1,
        bytes: u64,
        alignment: u64,
    ) -> Self {
        Self {
            struct_size: std::mem::size_of::<Self>() as u32,
            device_id,
            memory_kind,
            allocation_class,
            scope,
            flags: 0,
            reserved: 0,
            bytes,
            alignment,
        }
    }

    pub fn is_well_formed(&self) -> bool {
        self.struct_size >= std::mem::size_of::<Self>() as u32
            && self.flags == 0
            && self.reserved == 0
            && self.bytes != 0
            && self.alignment.is_power_of_two()
            && self.scope.is_well_formed()
    }
}

pub type KapslScopedDeviceAllocateFn = unsafe extern "C" fn(
    user_data: *mut c_void,
    request: *const KapslScopedDeviceAllocationRequestV1,
    allocation_out: *mut KapslDeviceAllocationV1,
) -> i32;

/// Backward-compatible host extension for request-aware governed allocation.
///
/// Pass `&host.base` through [`crate::KapslBackendConfigV1::host`]. The base
/// table retains ABI version 1 and advertises the extension through its larger
/// `struct_size`. A backend that advertises
/// [`crate::KAPSL_BACKEND_CAP_SCOPED_DEVICE_ALLOCATOR`] must require this
/// extension and the exact scoped-allocator version during initialization.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct KapslBackendHostScopedAllocatorV1 {
    pub base: KapslBackendHostV1,
    pub scoped_allocator_version: u32,
    pub reserved: u32,
    pub allocate_device_scoped: Option<KapslScopedDeviceAllocateFn>,
}

impl KapslBackendHostScopedAllocatorV1 {
    pub const fn new(
        mut base: KapslBackendHostV1,
        allocate_device_scoped: KapslScopedDeviceAllocateFn,
    ) -> Self {
        base.struct_size = std::mem::size_of::<Self>() as u32;
        base.abi_version = KAPSL_BACKEND_ABI_VERSION;
        Self {
            base,
            scoped_allocator_version: KAPSL_SCOPED_DEVICE_ALLOCATOR_VERSION,
            reserved: 0,
            allocate_device_scoped: Some(allocate_device_scoped),
        }
    }

    pub fn is_well_formed(&self) -> bool {
        self.base.struct_size >= std::mem::size_of::<Self>() as u32
            && self.base.abi_version == KAPSL_BACKEND_ABI_VERSION
            && self.scoped_allocator_version == KAPSL_SCOPED_DEVICE_ALLOCATOR_VERSION
            && self.reserved == 0
            && self.allocate_device_scoped.is_some()
            && self.base.allocate_device.is_some()
            && self.base.free_device.is_some()
            && self.base.synchronize_device.is_some()
    }

    /// Read a scoped allocator extension through the ABI-v1 base pointer.
    ///
    /// # Safety
    ///
    /// `base` must point to host-owned storage that remains readable through
    /// backend shutdown. If its `struct_size` advertises the extension, that
    /// storage must cover the complete extended table.
    pub unsafe fn from_base<'a>(base: *const KapslBackendHostV1) -> Option<&'a Self> {
        if base.is_null() {
            return None;
        }
        // SAFETY: the caller guarantees a readable ABI-v1 prefix.
        let struct_size = unsafe { base.cast::<u32>().read() };
        if struct_size < std::mem::size_of::<Self>() as u32 {
            return None;
        }
        // SAFETY: the advertised size and caller contract cover the extension.
        let extension = unsafe { &*base.cast::<Self>() };
        extension.is_well_formed().then_some(extension)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{KAPSL_ALLOCATION_CLASS_WORKSPACE, KAPSL_MEMORY_CUDA, KAPSL_STATUS_OK};

    unsafe extern "C" fn allocate(
        _user_data: *mut c_void,
        _request: *const KapslScopedDeviceAllocationRequestV1,
        _allocation_out: *mut KapslDeviceAllocationV1,
    ) -> i32 {
        KAPSL_STATUS_OK
    }

    unsafe extern "C" fn allocate_legacy(
        _user_data: *mut c_void,
        _request: *const crate::KapslDeviceAllocationRequestV1,
        _allocation_out: *mut KapslDeviceAllocationV1,
    ) -> i32 {
        KAPSL_STATUS_OK
    }

    unsafe extern "C" fn free(
        _user_data: *mut c_void,
        _allocation: *const KapslDeviceAllocationV1,
    ) -> i32 {
        KAPSL_STATUS_OK
    }

    unsafe extern "C" fn synchronize(_user_data: *mut c_void, _device_id: u32) -> i32 {
        KAPSL_STATUS_OK
    }

    #[test]
    fn scope_kinds_have_unambiguous_request_cardinality() {
        let request_ids = [41, 42];
        let model = KapslDeviceAllocationScopeV1::new(KAPSL_ALLOCATION_SCOPE_MODEL, 1, 7, 3, &[]);
        assert!(model.is_well_formed());

        let request = KapslDeviceAllocationScopeV1::new(
            KAPSL_ALLOCATION_SCOPE_REQUEST,
            2,
            7,
            3,
            &request_ids[..1],
        );
        assert!(request.is_well_formed());
        // SAFETY: request_ids outlives this test assertion.
        assert_eq!(unsafe { request.request_ids() }, Some(&request_ids[..1]));

        let batch = KapslDeviceAllocationScopeV1::new(
            KAPSL_ALLOCATION_SCOPE_REQUEST_BATCH,
            3,
            7,
            3,
            &request_ids,
        );
        assert!(batch.is_well_formed());

        let contradictory = KapslDeviceAllocationScopeV1::new(
            KAPSL_ALLOCATION_SCOPE_REQUEST,
            4,
            7,
            3,
            &request_ids,
        );
        assert!(!contradictory.is_well_formed());
    }

    #[test]
    fn scoped_request_and_host_extension_validate_as_one_contract() {
        let ids = [99];
        let scope =
            KapslDeviceAllocationScopeV1::new(KAPSL_ALLOCATION_SCOPE_REQUEST, 8, 12, 4, &ids);
        let request = KapslScopedDeviceAllocationRequestV1::new(
            0,
            KAPSL_MEMORY_CUDA,
            KAPSL_ALLOCATION_CLASS_WORKSPACE,
            scope,
            4096,
            256,
        );
        assert!(request.is_well_formed());

        let host = KapslBackendHostScopedAllocatorV1::new(
            KapslBackendHostV1 {
                struct_size: 0,
                abi_version: 0,
                user_data: std::ptr::null_mut(),
                log: None,
                allocate_device: Some(allocate_legacy),
                free_device: Some(free),
                synchronize_device: Some(synchronize),
            },
            allocate,
        );
        assert!(host.is_well_formed());
        assert_eq!(host.base.struct_size, std::mem::size_of_val(&host) as u32);
        // SAFETY: host owns a complete extension for the duration of the call.
        let recovered = unsafe { KapslBackendHostScopedAllocatorV1::from_base(&host.base) };
        assert!(recovered.is_some());

        let legacy = KapslBackendHostV1 {
            struct_size: std::mem::size_of::<KapslBackendHostV1>() as u32,
            ..host.base
        };
        // SAFETY: legacy owns a readable v1 prefix and advertises no extension.
        assert!(unsafe { KapslBackendHostScopedAllocatorV1::from_base(&legacy) }.is_none());
    }

    #[test]
    #[cfg(target_pointer_width = "64")]
    fn scoped_extension_layout_is_frozen_on_64_bit_targets() {
        use std::mem::{offset_of, size_of};

        assert_eq!(size_of::<KapslDeviceAllocationScopeV1>(), 40);
        assert_eq!(size_of::<KapslScopedDeviceAllocationRequestV1>(), 80);
        assert_eq!(offset_of!(KapslBackendHostScopedAllocatorV1, base), 0);
        assert_eq!(
            offset_of!(KapslBackendHostScopedAllocatorV1, scoped_allocator_version),
            48
        );
        assert_eq!(size_of::<KapslBackendHostScopedAllocatorV1>(), 64);
    }
}
