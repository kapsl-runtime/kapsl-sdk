//! Stable native-backend boundary shared by Kapsl runtimes and backend packs.
//!
//! This crate deliberately exposes C-compatible values only. Rust enums, trait
//! objects, collections, futures, and ownership-bearing standard-library types
//! must not cross the dynamic-library boundary. Backend-specific contracts may
//! build on the common primitives, but new native integrations should implement
//! [`KapslBackendApiV1`].

#![deny(unsafe_op_in_unsafe_fn)]

use std::ffi::c_void;

mod backend_v1;
mod llama_cpp_v1;

pub use backend_v1::*;
pub use llama_cpp_v1::*;

pub const KAPSL_STATUS_OK: i32 = 0;
pub const KAPSL_STATUS_INVALID_ARGUMENT: i32 = 1;
pub const KAPSL_STATUS_INCOMPATIBLE_ABI: i32 = 2;
pub const KAPSL_STATUS_UNSUPPORTED: i32 = 3;
pub const KAPSL_STATUS_BACKEND_ERROR: i32 = 4;
pub const KAPSL_STATUS_CANCELLED: i32 = 5;
pub const KAPSL_STATUS_PANIC: i32 = 6;

pub const KAPSL_LOG_ERROR: u32 = 1;
pub const KAPSL_LOG_WARN: u32 = 2;
pub const KAPSL_LOG_INFO: u32 = 3;
pub const KAPSL_LOG_DEBUG: u32 = 4;
pub const KAPSL_LOG_TRACE: u32 = 5;

/// Borrowed bytes. The receiver must not retain `ptr` after the enclosing call
/// returns unless that call's contract explicitly grants a longer lifetime.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KapslSlice {
    pub ptr: *const u8,
    pub len: usize,
}

impl KapslSlice {
    pub const fn empty() -> Self {
        Self {
            ptr: std::ptr::null(),
            len: 0,
        }
    }

    pub fn from_bytes(bytes: &[u8]) -> Self {
        Self {
            ptr: bytes.as_ptr(),
            len: bytes.len(),
        }
    }

    /// # Safety
    ///
    /// `ptr` must be readable for `len` bytes for the returned borrow. The
    /// allocation must remain alive for the lifetime selected by the caller.
    pub unsafe fn as_bytes(&self) -> Option<&[u8]> {
        if self.len == 0 {
            return Some(&[]);
        }
        if self.ptr.is_null() {
            return None;
        }
        // SAFETY: upheld by the caller of this method.
        Some(unsafe { std::slice::from_raw_parts(self.ptr, self.len) })
    }
}

impl Default for KapslSlice {
    fn default() -> Self {
        Self::empty()
    }
}

/// Bytes allocated by a backend pack. They must be returned through the
/// `free_buffer` function from the same function table that produced them.
#[repr(C)]
#[derive(Debug, PartialEq, Eq)]
pub struct KapslOwnedBuffer {
    pub ptr: *mut u8,
    pub len: usize,
    pub capacity: usize,
}

impl KapslOwnedBuffer {
    pub const fn empty() -> Self {
        Self {
            ptr: std::ptr::null_mut(),
            len: 0,
            capacity: 0,
        }
    }
}

impl Default for KapslOwnedBuffer {
    fn default() -> Self {
        Self::empty()
    }
}

pub type KapslLogFn = unsafe extern "C" fn(user_data: *mut c_void, level: u32, message: KapslSlice);

pub type KapslRequestCancelledFn =
    unsafe extern "C" fn(user_data: *mut c_void, request_id: u64) -> u32;

pub type KapslFreeBufferFn = unsafe extern "C" fn(buffer: KapslOwnedBuffer);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_slices_and_buffers_are_canonical() {
        let slice = KapslSlice::empty();
        assert!(slice.ptr.is_null());
        assert_eq!(slice.len, 0);

        let buffer = KapslOwnedBuffer::empty();
        assert!(buffer.ptr.is_null());
        assert_eq!(buffer.len, 0);
        assert_eq!(buffer.capacity, 0);
    }

    #[test]
    fn nonempty_null_slice_is_invalid() {
        let slice = KapslSlice {
            ptr: std::ptr::null(),
            len: 1,
        };
        // SAFETY: this deliberately exercises validation before dereference.
        assert!(unsafe { slice.as_bytes() }.is_none());
    }
}
