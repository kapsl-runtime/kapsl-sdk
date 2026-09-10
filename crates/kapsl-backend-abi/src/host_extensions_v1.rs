//! Discovery of optional, independently versioned host contracts.
//!
//! The ABI-v1 host and scoped-allocator prefixes retain their published layouts.
//! An adapter reads this tail only after checking the advertised storage size.
//! Extension names identify contracts, never backend products. For example, a
//! native KV participant queries the contract published by `kapsl-kv-abi`.

use std::ffi::c_void;

use crate::{
    KapslBackendHostScopedAllocatorV1, KapslBackendHostV1, KapslSlice, KAPSL_BACKEND_ABI_VERSION,
    KAPSL_SCOPED_DEVICE_ALLOCATOR_VERSION,
};

pub const KAPSL_HOST_EXTENSION_QUERY_VERSION: u32 = 1;

/// Return an immutable borrowed function table for the exact name/version, or
/// null when unsupported. The name is borrowed for this call only. A returned
/// table and its context remain alive until adapter shutdown returns. The
/// adapter validates that contract's size, version and required callbacks.
/// Queries must not allocate model memory or change host ownership.
pub type KapslHostQueryExtensionFn =
    unsafe extern "C" fn(user_data: *mut c_void, name: KapslSlice, version: u32) -> *const c_void;

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct KapslBackendHostExtensionsV1 {
    pub base: KapslBackendHostScopedAllocatorV1,
    pub extension_query_version: u32,
    pub reserved: u32,
    pub extension_user_data: *mut c_void,
    pub query_extension: Option<KapslHostQueryExtensionFn>,
}

impl KapslBackendHostExtensionsV1 {
    pub const fn new(
        mut base: KapslBackendHostScopedAllocatorV1,
        extension_user_data: *mut c_void,
        query_extension: KapslHostQueryExtensionFn,
    ) -> Self {
        base.base.struct_size = std::mem::size_of::<Self>() as u32;
        Self {
            base,
            extension_query_version: KAPSL_HOST_EXTENSION_QUERY_VERSION,
            reserved: 0,
            extension_user_data,
            query_extension: Some(query_extension),
        }
    }

    pub fn is_well_formed(&self) -> bool {
        self.base.base.struct_size >= std::mem::size_of::<Self>() as u32
            && self.base.base.abi_version == KAPSL_BACKEND_ABI_VERSION
            && self.base.scoped_allocator_version == KAPSL_SCOPED_DEVICE_ALLOCATOR_VERSION
            && self.base.reserved == 0
            && self.extension_query_version == KAPSL_HOST_EXTENSION_QUERY_VERSION
            && self.reserved == 0
            && self.query_extension.is_some()
    }

    /// # Safety
    /// `base` points to a readable `struct_size` field. If it advertises this
    /// extension, that size accurately describes readable host-owned storage
    /// retained through shutdown.
    pub unsafe fn from_base<'a>(base: *const KapslBackendHostV1) -> Option<&'a Self> {
        if base.is_null() {
            return None;
        }
        // SAFETY: the caller guarantees the readable prefix.
        if unsafe { base.cast::<u32>().read() } < std::mem::size_of::<Self>() as u32 {
            return None;
        }
        // SAFETY: the checked size covers this complete extension.
        let extension = unsafe { &*base.cast::<Self>() };
        extension.is_well_formed().then_some(extension)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    unsafe extern "C" fn query(
        _user_data: *mut c_void,
        _name: KapslSlice,
        _version: u32,
    ) -> *const c_void {
        std::ptr::null()
    }

    fn table() -> KapslBackendHostExtensionsV1 {
        KapslBackendHostExtensionsV1::new(
            KapslBackendHostScopedAllocatorV1 {
                base: KapslBackendHostV1 {
                    struct_size: std::mem::size_of::<KapslBackendHostV1>() as u32,
                    abi_version: KAPSL_BACKEND_ABI_VERSION,
                    user_data: std::ptr::null_mut(),
                    log: None,
                    allocate_device: None,
                    free_device: None,
                    synchronize_device: None,
                },
                scoped_allocator_version: KAPSL_SCOPED_DEVICE_ALLOCATOR_VERSION,
                reserved: 0,
                allocate_device_scoped: None,
            },
            std::ptr::null_mut(),
            query,
        )
    }

    #[test]
    fn discovery_preserves_both_published_prefixes_and_allows_cpu_hosts() {
        let table = table();
        assert_eq!(std::mem::offset_of!(KapslBackendHostExtensionsV1, base), 0);
        #[cfg(target_pointer_width = "64")]
        {
            assert_eq!(std::mem::size_of::<KapslBackendHostExtensionsV1>(), 88);
            assert_eq!(
                std::mem::offset_of!(KapslBackendHostExtensionsV1, query_extension),
                80
            );
        }
        assert!(table.is_well_formed());
        let extension =
            unsafe { KapslBackendHostExtensionsV1::from_base(&table.base.base) }.unwrap();
        assert!(unsafe {
            extension.query_extension.unwrap()(
                extension.extension_user_data,
                KapslSlice::from_bytes(b"unknown-contract"),
                1,
            )
        }
        .is_null());
    }

    #[test]
    fn older_and_truncated_hosts_are_rejected_before_reading_the_tail() {
        let header_only = std::mem::size_of::<KapslBackendHostV1>() as u32;
        assert!(unsafe {
            KapslBackendHostExtensionsV1::from_base(
                (&header_only as *const u32).cast::<KapslBackendHostV1>(),
            )
        }
        .is_none());
        assert!(unsafe { KapslBackendHostExtensionsV1::from_base(std::ptr::null()) }.is_none());
        let mut table = table();
        table.base.base.struct_size -= 1;
        assert!(!table.is_well_formed());
    }

    #[test]
    fn incompatible_or_incomplete_extension_tables_fail_closed() {
        let mut candidate = table();
        candidate.base.base.abi_version += 1;
        assert!(!candidate.is_well_formed());
        let mut candidate = table();
        candidate.base.scoped_allocator_version += 1;
        assert!(!candidate.is_well_formed());
        let mut candidate = table();
        candidate.base.reserved = 1;
        assert!(!candidate.is_well_formed());
        let mut candidate = table();
        candidate.extension_query_version += 1;
        assert!(!candidate.is_well_formed());
        let mut candidate = table();
        candidate.reserved = 1;
        assert!(!candidate.is_well_formed());
        let mut candidate = table();
        candidate.query_extension = None;
        assert!(!candidate.is_well_formed());
    }
}
