use kapsl_transport::{RequestMetadata, ResponseMetadata};

/// Number of entries reserved for each shared-memory metadata queue.
pub const SHM_QUEUE_CAPACITY: usize = 1024;

/// Current fixed-layout shared-memory request/response protocol version.
pub const SHM_PROTOCOL_VERSION: u16 = 2;

/// Request metadata exchanged through the shared-memory request queue.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct ShmRequest {
    pub metadata: RequestMetadata,
    pub tensor_offset: u64,
    pub tensor_size: u64,
    /// Process-shared lease token protecting the input tensor slot.
    pub tensor_lease: u64,
    /// Mailbox reserved exclusively for this request's response.
    pub response_mailbox: u32,
    /// Version of this fixed-layout request.
    pub protocol_version: u16,
    pub _padding: [u8; 2],
}

/// Response metadata exchanged through the shared-memory response queue.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct ShmResponse {
    pub metadata: ResponseMetadata,
    pub result_offset: u64,
    pub result_size: u64,
    /// Offset of an encoded error message, or zero for successful responses.
    pub error_offset: u64,
    /// Process-shared lease token protecting either result or error payload.
    pub payload_lease: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn queue_entries_fit_the_reserved_control_regions() {
        assert!(std::mem::size_of::<ShmRequest>() <= 64);
        assert!(std::mem::size_of::<ShmResponse>() <= 64);
        assert!(std::mem::align_of::<ShmRequest>() >= std::mem::align_of::<usize>());
        assert!(std::mem::align_of::<ShmResponse>() >= std::mem::align_of::<usize>());
    }
}
