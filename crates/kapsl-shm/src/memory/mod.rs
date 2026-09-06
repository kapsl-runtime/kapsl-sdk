use crate::allocator::SharedShmAllocator;
use crate::mailbox::{
    mailbox_alignment, mailbox_bytes, ResponseMailboxRegistry, RESPONSE_MAILBOX_COUNT,
};
use crate::protocol::{ShmRequest, SHM_QUEUE_CAPACITY};
use shared_memory::*;
use std::sync::atomic::{AtomicU64, Ordering};

const MAGIC_NUMBER: u32 = 0x41494D4F; // "AIMO"
const VERSION: u32 = 3;
const CONTROL_PAGE_BYTES: usize = 4 * 1024;
const QUEUE_REGION_BYTES: usize = 64 * 1024;
const LEASE_SEQUENCE_OFFSET: usize = 512;
const REQUEST_SEQUENCE_OFFSET: usize = LEASE_SEQUENCE_OFFSET + std::mem::size_of::<AtomicU64>();
const MIN_TENSOR_POOL_BYTES: usize = 64 * 1024;
const FEATURE_ROUTED_RESPONSES: u64 = 1 << 0;
const FEATURE_SHARED_TENSOR_LEASES: u64 = 1 << 1;
const REQUIRED_FEATURES: u64 = FEATURE_ROUTED_RESPONSES | FEATURE_SHARED_TENSOR_LEASES;

/// Header structure at the beginning of shared memory.
///
/// Fields are immutable after creation. Mutable process-shared counters live at
/// the advertised aligned offsets so reading this header never races an atomic
/// update.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct ShmHeader {
    pub magic: u32,
    pub version: u32,
    pub request_queue_offset: u64,
    pub tensor_pool_offset: u64,
    pub max_tensor_size: u64,
    pub features: u64,
    pub response_mailbox_offset: u64,
    pub response_mailbox_count: u32,
    pub _padding: u32,
    pub lease_sequence_offset: u64,
    pub request_sequence_offset: u64,
}

/// Tensor header in shared memory.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct TensorHeader {
    pub ndim: u32,
    pub dtype: u8,
    pub _padding: [u8; 3],
    pub shape: [i64; 8],
    pub data_size: u64,
}

/// Owns or connects to a versioned shared-memory communication region.
pub struct ShmManager {
    shmem: Shmem,
    size: usize,
}

// SAFETY: the mapping lifetime is owned by this value. Mutable shared control
// fields use process-shared atomics, while payload access is protected by the
// lease and mailbox protocols.
unsafe impl Send for ShmManager {}
unsafe impl Sync for ShmManager {}

impl ShmManager {
    /// Create and initialize a new shared-memory region.
    pub fn create(name: &str, size: usize) -> Result<Self, ShmError> {
        let layout = RegionLayout::for_size(size)?;

        let _ = ShmemConf::new().os_id(name).force_create_flink();
        let shmem = match ShmemConf::new().size(size).os_id(name).create() {
            Ok(shmem) => shmem,
            Err(_) => {
                let _ = ShmemConf::new().os_id(name).force_create_flink();
                std::thread::sleep(std::time::Duration::from_millis(100));
                ShmemConf::new().size(size).os_id(name).create()?
            }
        };

        let header = ShmHeader {
            magic: MAGIC_NUMBER,
            version: VERSION,
            request_queue_offset: layout.request_queue_offset as u64,
            tensor_pool_offset: layout.tensor_pool_offset as u64,
            max_tensor_size: layout.tensor_pool_bytes as u64,
            features: REQUIRED_FEATURES,
            response_mailbox_offset: layout.response_mailbox_offset as u64,
            response_mailbox_count: RESPONSE_MAILBOX_COUNT as u32,
            _padding: 0,
            lease_sequence_offset: LEASE_SEQUENCE_OFFSET as u64,
            request_sequence_offset: REQUEST_SEQUENCE_OFFSET as u64,
        };

        // SAFETY: this process exclusively initializes the newly created map.
        unsafe {
            std::ptr::write(shmem.as_ptr().cast::<ShmHeader>(), header);
            std::ptr::write(
                shmem
                    .as_ptr()
                    .add(LEASE_SEQUENCE_OFFSET)
                    .cast::<AtomicU64>(),
                AtomicU64::new(0),
            );
            std::ptr::write(
                shmem
                    .as_ptr()
                    .add(REQUEST_SEQUENCE_OFFSET)
                    .cast::<AtomicU64>(),
                AtomicU64::new(0),
            );
        }

        let manager = Self { shmem, size };
        ResponseMailboxRegistry::initialize(&manager);
        SharedShmAllocator::initialize(&manager);

        log::info!(
            "Created shared memory region '{}' (size={} tensor_pool={}..{} mailboxes={})",
            name,
            size,
            layout.tensor_pool_offset,
            size,
            RESPONSE_MAILBOX_COUNT
        );
        Ok(manager)
    }

    /// Connect to an existing compatible shared-memory region.
    pub fn connect(name: &str) -> Result<Self, ShmError> {
        let shmem = ShmemConf::new().os_id(name).open()?;
        let manager = Self {
            size: shmem.len(),
            shmem,
        };
        manager.validate_header()?;

        log::info!(
            "Connected to shared memory region '{}' of size {} bytes",
            name,
            manager.size
        );
        Ok(manager)
    }

    /// Get the base pointer to the mapped region.
    pub fn as_ptr(&self) -> *mut u8 {
        self.shmem.as_ptr()
    }

    /// Get the size of the mapped region.
    pub fn size(&self) -> usize {
        self.size
    }

    /// Get the request queue offset.
    pub fn request_queue_offset(&self) -> usize {
        self.header().request_queue_offset as usize
    }

    /// Get the response mailbox array offset.
    pub fn response_mailbox_offset(&self) -> usize {
        self.header().response_mailbox_offset as usize
    }

    /// Get the number of independently routed response mailboxes.
    pub fn response_mailbox_count(&self) -> usize {
        self.header().response_mailbox_count as usize
    }

    /// Get the first byte used by shared tensor slots.
    pub fn tensor_pool_offset(&self) -> usize {
        self.header().tensor_pool_offset as usize
    }

    /// Get the total number of bytes available to shared tensor slots.
    pub fn max_tensor_size(&self) -> usize {
        self.header().max_tensor_size as usize
    }

    /// Allocate a region-wide request id unique across connected clients.
    pub fn next_request_id(&self) -> u64 {
        self.next_nonzero_sequence(self.header().request_sequence_offset as usize)
    }

    /// Allocate the non-zero generation embedded in a tensor lease token.
    pub(crate) fn next_lease_sequence(&self) -> u32 {
        loop {
            let sequence =
                self.next_nonzero_sequence(self.header().lease_sequence_offset as usize) as u32;
            if sequence != 0 {
                return sequence;
            }
        }
    }

    fn header(&self) -> ShmHeader {
        // SAFETY: the header is immutable after successful region creation.
        unsafe { std::ptr::read(self.as_ptr().cast::<ShmHeader>()) }
    }

    fn next_nonzero_sequence(&self, offset: usize) -> u64 {
        // SAFETY: header validation guarantees an aligned AtomicU64 control
        // word at this in-bounds offset.
        let sequence = unsafe { &*self.as_ptr().add(offset).cast::<AtomicU64>() };
        loop {
            let next = sequence.fetch_add(1, Ordering::Relaxed).wrapping_add(1);
            if next != 0 {
                return next;
            }
        }
    }

    fn validate_header(&self) -> Result<(), ShmError> {
        if self.size < std::mem::size_of::<ShmHeader>() {
            return Err(ShmError::InvalidLayout(
                "region is smaller than the SHM header".to_string(),
            ));
        }
        let header = self.header();
        if header.magic != MAGIC_NUMBER {
            return Err(ShmError::InvalidMagic);
        }
        if header.version != VERSION {
            return Err(ShmError::VersionMismatch {
                expected: VERSION,
                found: header.version,
            });
        }
        if header.features & REQUIRED_FEATURES != REQUIRED_FEATURES {
            return Err(ShmError::InvalidLayout(
                "region does not advertise routed responses and shared tensor leases".to_string(),
            ));
        }
        if header.response_mailbox_count as usize != RESPONSE_MAILBOX_COUNT {
            return Err(ShmError::InvalidLayout(format!(
                "expected {RESPONSE_MAILBOX_COUNT} response mailboxes, found {}",
                header.response_mailbox_count
            )));
        }
        if !(header.response_mailbox_offset as usize).is_multiple_of(mailbox_alignment()) {
            return Err(ShmError::InvalidLayout(
                "response mailbox array is not cache-line aligned".to_string(),
            ));
        }
        if header.lease_sequence_offset as usize != LEASE_SEQUENCE_OFFSET
            || header.request_sequence_offset as usize != REQUEST_SEQUENCE_OFFSET
        {
            return Err(ShmError::InvalidLayout(
                "shared sequence counters are at unexpected offsets".to_string(),
            ));
        }
        let expected = RegionLayout::for_size(self.size)?;
        if header.request_queue_offset != expected.request_queue_offset as u64
            || header.response_mailbox_offset != expected.response_mailbox_offset as u64
            || header.tensor_pool_offset != expected.tensor_pool_offset as u64
            || header.max_tensor_size != expected.tensor_pool_bytes as u64
        {
            return Err(ShmError::InvalidLayout(
                "region offsets do not match the protocol layout for its mapped size".to_string(),
            ));
        }

        validate_region(
            self.size,
            header.request_queue_offset as usize,
            QUEUE_REGION_BYTES,
            "request queue",
        )?;
        validate_region(
            self.size,
            header.response_mailbox_offset as usize,
            mailbox_bytes(header.response_mailbox_count as usize),
            "response mailboxes",
        )?;
        validate_region(
            self.size,
            header.tensor_pool_offset as usize,
            header.max_tensor_size as usize,
            "tensor pool",
        )?;
        validate_atomic_offset(self.size, header.lease_sequence_offset as usize)?;
        validate_atomic_offset(self.size, header.request_sequence_offset as usize)?;
        Ok(())
    }
}

#[derive(Debug)]
pub enum ShmError {
    ShmemError(shared_memory::ShmemError),
    InvalidMagic,
    VersionMismatch { expected: u32, found: u32 },
    RegionTooSmall { required: usize, actual: usize },
    InvalidLayout(String),
}

impl From<shared_memory::ShmemError> for ShmError {
    fn from(error: shared_memory::ShmemError) -> Self {
        Self::ShmemError(error)
    }
}

impl std::fmt::Display for ShmError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ShmemError(error) => write!(formatter, "Shared memory error: {error}"),
            Self::InvalidMagic => formatter.write_str("Invalid magic number in shared memory"),
            Self::VersionMismatch { expected, found } => write!(
                formatter,
                "Shared memory version mismatch: expected {expected}, found {found}"
            ),
            Self::RegionTooSmall { required, actual } => write!(
                formatter,
                "Shared memory region is too small: required at least {required} bytes, got {actual}"
            ),
            Self::InvalidLayout(message) => {
                write!(formatter, "Invalid shared memory layout: {message}")
            }
        }
    }
}

impl std::error::Error for ShmError {}

#[derive(Debug, Clone, Copy)]
struct RegionLayout {
    request_queue_offset: usize,
    response_mailbox_offset: usize,
    tensor_pool_offset: usize,
    tensor_pool_bytes: usize,
}

impl RegionLayout {
    fn for_size(size: usize) -> Result<Self, ShmError> {
        let request_queue_bytes = std::mem::size_of::<ShmRequest>() * SHM_QUEUE_CAPACITY;
        if request_queue_bytes > QUEUE_REGION_BYTES {
            return Err(ShmError::InvalidLayout(format!(
                "request queue requires {request_queue_bytes} bytes but its control region reserves {QUEUE_REGION_BYTES}"
            )));
        }
        let request_queue_offset = CONTROL_PAGE_BYTES;
        let response_mailbox_offset = request_queue_offset + QUEUE_REGION_BYTES;
        let tensor_pool_offset = align_up(
            response_mailbox_offset + mailbox_bytes(RESPONSE_MAILBOX_COUNT),
            CONTROL_PAGE_BYTES,
        );
        let required = tensor_pool_offset + MIN_TENSOR_POOL_BYTES;
        if size < required {
            return Err(ShmError::RegionTooSmall {
                required,
                actual: size,
            });
        }
        Ok(Self {
            request_queue_offset,
            response_mailbox_offset,
            tensor_pool_offset,
            tensor_pool_bytes: size - tensor_pool_offset,
        })
    }
}

fn validate_region(
    region_size: usize,
    offset: usize,
    byte_len: usize,
    label: &str,
) -> Result<(), ShmError> {
    let end = offset
        .checked_add(byte_len)
        .ok_or_else(|| ShmError::InvalidLayout(format!("{label} range overflow")))?;
    if end > region_size {
        return Err(ShmError::InvalidLayout(format!(
            "{label} range {offset}..{end} exceeds region size {region_size}"
        )));
    }
    Ok(())
}

fn validate_atomic_offset(region_size: usize, offset: usize) -> Result<(), ShmError> {
    if !offset.is_multiple_of(std::mem::align_of::<AtomicU64>()) {
        return Err(ShmError::InvalidLayout(format!(
            "atomic counter offset {offset} is not aligned"
        )));
    }
    validate_region(
        region_size,
        offset,
        std::mem::size_of::<AtomicU64>(),
        "atomic counter",
    )
}

const fn align_up(value: usize, alignment: usize) -> usize {
    value.div_ceil(alignment) * alignment
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_and_connect() {
        let name = format!("/test_shm_create_connect_{}", std::process::id());
        let size = 1024 * 1024;
        let manager1 = match ShmManager::create(&name, size) {
            Ok(manager) => manager,
            Err(ShmError::ShmemError(shared_memory::ShmemError::MapCreateFailed(_))) => {
                eprintln!("Skipping shared memory test (mapping creation failed)");
                return;
            }
            Err(error) => panic!("Failed to create shared memory: {error}"),
        };
        let manager2 = ShmManager::connect(&name).expect("connect to shared memory");

        assert_eq!(manager1.size(), size);
        assert_eq!(manager2.size(), size);
        assert_eq!(
            manager1.request_queue_offset(),
            manager2.request_queue_offset()
        );
        assert_eq!(
            manager1.response_mailbox_offset(),
            manager2.response_mailbox_offset()
        );
        assert_eq!(manager1.tensor_pool_offset(), manager2.tensor_pool_offset());
        assert_ne!(manager1.next_request_id(), manager2.next_request_id());
    }

    #[test]
    fn rejects_regions_without_tensor_capacity() {
        let error = RegionLayout::for_size(64 * 1024).expect_err("small region must fail");
        assert!(matches!(error, ShmError::RegionTooSmall { .. }));
    }
}

#[cfg(test)]
#[path = "tests.rs"]
mod additional_tests;
