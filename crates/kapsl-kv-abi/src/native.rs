//! In-process shared-KV host contract. This extension contains memory operations
//! only; adapter discovery, lifecycle and inference use the backend ABI.
//!
//! Query `KAPSL_NATIVE_KV_EXTENSION_NAME` at version 1 through the generic host
//! extension table. Storage and callbacks remain valid through adapter shutdown.
//! No Rust-owned value, trait, collection or unwinding exception crosses this
//! boundary. All sizes and pointer/count headers are checked before tail reads.

use std::ffi::c_void;

pub const KAPSL_NATIVE_KV_EXTENSION_NAME: &[u8] = b"kapsl-kv-native";
pub const KAPSL_NATIVE_KV_VERSION: u32 = 1;
pub const KAPSL_KV_STATUS_OK: i32 = 0;
pub const KAPSL_KV_STATUS_INVALID_ARGUMENT: i32 = 1;
pub const KAPSL_KV_STATUS_INCOMPATIBLE_ABI: i32 = 2;
pub const KAPSL_KV_STATUS_UNSUPPORTED: i32 = 3;
pub const KAPSL_KV_STATUS_BACKEND_ERROR: i32 = 4;
pub const KAPSL_KV_STATUS_CANCELLED: i32 = 5;
pub const KAPSL_KV_STATUS_PANIC: i32 = 6;
pub const KAPSL_KV_RESERVE_UPLOAD: u32 = 1;

/// One model/replica's view into governed KV storage. Pool creation is allowed
/// only during an admitted model lifecycle call. The host table, not geometry,
/// determines its model/replica owner. `requested_blocks` counts physical blocks
/// across layers; one block stores K and V for one layer.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct KapslKvGeometryV1 {
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
    pub flags: u32,
    pub reserved: u32,
}

impl KapslKvGeometryV1 {
    pub fn is_well_formed(&self) -> bool {
        if self.struct_size < std::mem::size_of::<Self>() as u32
            || self.flags != 0
            || self.reserved != 0
            || self.requested_blocks < u64::from(self.num_layers)
            || [
                self.block_size_tokens,
                self.num_layers,
                self.num_kv_heads,
                self.key_head_dim,
                self.value_head_dim,
                self.max_sequences,
                self.max_blocks_per_sequence,
            ]
            .contains(&0)
            || !matches!(self.element_bytes, 1 | 2 | 4 | 8)
        {
            return false;
        }
        self.num_layers
            .checked_mul(self.max_blocks_per_sequence)
            .is_some()
            && self.storage_bytes().is_some()
    }

    pub fn storage_bytes(&self) -> Option<u64> {
        let heads = u64::from(self.key_head_dim).checked_add(u64::from(self.value_head_dim))?;
        self.requested_blocks
            .checked_mul(u64::from(self.block_size_tokens))?
            .checked_mul(u64::from(self.num_kv_heads))?
            .checked_mul(heads)?
            .checked_mul(u64::from(self.element_bytes))
    }
}

/// Host-issued, nonzero pool ID, unique for the host process lifetime. A stale
/// ID may never identify a later pool even when device addresses are recycled.
/// All pointers are borrowed until successful destroy. `device_base` names an
/// address space; addressable capacity is not ownership of other models' blocks.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct KapslKvPoolV1 {
    pub struct_size: u32,
    pub reserved: u32,
    pub pool_id: u64,
    pub device_base: *mut c_void,
    pub addressable_blocks: u64,
    pub block_table_device: *mut u32,
    pub block_table_layer_stride: u32,
    pub block_table_sequence_stride: u32,
    pub sequence_slots: u32,
    pub reserved_tail: u32,
}

impl KapslKvPoolV1 {
    pub fn is_well_formed(&self) -> bool {
        self.struct_size >= std::mem::size_of::<Self>() as u32
            && self.reserved == 0
            && self.reserved_tail == 0
            && self.pool_id != 0
            && !self.device_base.is_null()
            && !self.block_table_device.is_null()
            && self.addressable_blocks != 0
            && self.block_table_layer_stride != 0
            && self.block_table_sequence_stride >= self.block_table_layer_stride
            && self.sequence_slots != 0
    }
}

/// Grow a sequence's reservation under an active engine request. Sequence IDs
/// may include zero but are never reused within a pool; sequence slots may be
/// reused after successful release. Request IDs must be nonzero and engine-issued.
/// The host validates model/replica, active request, cancellation,
/// pool ownership, sequence ownership and slot exclusivity before allocating.
/// Existing reservations are grow-only and retain attribution until release.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct KapslKvReserveRequestV1 {
    pub struct_size: u32,
    /// `KAPSL_KV_RESERVE_UPLOAD` publishes the table before returning. Without
    /// it the adapter calls commit before device work can read the table.
    pub flags: u32,
    pub model_id: u32,
    pub replica_id: u32,
    pub request_id: u64,
    pub sequence_id: u64,
    pub sequence_slot: u32,
    pub tokens_needed: u32,
}

impl KapslKvReserveRequestV1 {
    pub fn is_well_formed(&self) -> bool {
        self.struct_size >= std::mem::size_of::<Self>() as u32
            && self.flags & !KAPSL_KV_RESERVE_UPLOAD == 0
            && self.request_id != 0
            && self.tokens_needed != 0
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct KapslKvReservationV1 {
    pub struct_size: u32,
    pub logical_blocks: u32,
    pub block_table_device: *mut u32,
}

pub type KapslKvCreatePoolFn = unsafe extern "C" fn(
    user_data: *mut c_void,
    geometry: *const KapslKvGeometryV1,
    pool_out: *mut KapslKvPoolV1,
) -> i32;
pub type KapslKvReserveFn = unsafe extern "C" fn(
    user_data: *mut c_void,
    pool_id: u64,
    request: *const KapslKvReserveRequestV1,
    reservation_out: *mut KapslKvReservationV1,
) -> i32;
pub type KapslKvCommitFn = unsafe extern "C" fn(
    user_data: *mut c_void,
    pool_id: u64,
    block_table_device_out: *mut *mut u32,
) -> i32;
pub type KapslKvSequenceFn =
    unsafe extern "C" fn(user_data: *mut c_void, pool_id: u64, sequence_id: u64) -> i32;
pub type KapslKvPoolBytesFn =
    unsafe extern "C" fn(user_data: *mut c_void, pool_id: u64, bytes_out: *mut u64) -> i32;
pub type KapslKvDestroyPoolFn = unsafe extern "C" fn(user_data: *mut c_void, pool_id: u64) -> i32;

/// One immutable, owner-bound shared-KV host. Callbacks validate opaque IDs
/// before lookup and synchronize before frees, rollback or successful destroy.
/// Failed synchronization/frees preserve storage and accounting for retry.
/// Errors leave output structs untouched and are reported through status plus
/// the host's diagnostics; no host buffer is freed by an adapter allocator.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct KapslNativeKvHostV1 {
    pub struct_size: u32,
    pub abi_version: u32,
    pub device_id: u32,
    pub model_id: u32,
    pub replica_id: u32,
    pub reserved: u32,
    pub user_data: *mut c_void,
    pub create_pool: Option<KapslKvCreatePoolFn>,
    pub reserve: Option<KapslKvReserveFn>,
    pub commit: Option<KapslKvCommitFn>,
    pub release: Option<KapslKvSequenceFn>,
    pub touch: Option<KapslKvSequenceFn>,
    pub pool_bytes: Option<KapslKvPoolBytesFn>,
    pub destroy_pool: Option<KapslKvDestroyPoolFn>,
}

impl KapslNativeKvHostV1 {
    pub fn is_well_formed(&self) -> bool {
        self.struct_size >= std::mem::size_of::<Self>() as u32
            && self.abi_version == KAPSL_NATIVE_KV_VERSION
            && self.reserved == 0
            && !self.user_data.is_null()
            && self.create_pool.is_some()
            && self.reserve.is_some()
            && self.commit.is_some()
            && self.release.is_some()
            && self.touch.is_some()
            && self.pool_bytes.is_some()
            && self.destroy_pool.is_some()
    }

    /// # Safety
    /// A non-null pointer must have a readable size field. If it advertises the
    /// complete table, the host guarantees that storage through shutdown.
    pub unsafe fn from_ptr<'a>(pointer: *const c_void) -> Option<&'a Self> {
        if pointer.is_null() {
            return None;
        }
        // SAFETY: the caller guarantees the readable size field.
        if unsafe { pointer.cast::<u32>().read() } < std::mem::size_of::<Self>() as u32 {
            return None;
        }
        // SAFETY: the checked advertised size covers the complete table.
        let table = unsafe { &*pointer.cast::<Self>() };
        table.is_well_formed().then_some(table)
    }
}

#[cfg(test)]
#[path = "tests/native.rs"]
mod tests;
