use kapsl_communication::shm::allocator::{SharedShmAllocator, SharedShmLease};
use kapsl_communication::shm::memory::{ShmManager, TensorHeader};
use kapsl_engine_api::{BinaryTensorPacket, TensorDtype};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::PyErr;
use std::fmt;
use std::mem::{align_of, size_of};
use std::str::FromStr;
use std::sync::Arc;
use std::time::Duration;

const MAX_TENSOR_RANK: usize = 8;
const MAX_ERROR_MESSAGE_BYTES: usize = 64 * 1024;
const SLOT_LEASE_TTL: Duration = Duration::from_secs(30);

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum ShmTensorError {
    InvalidInput(String),
    InvalidMemory(String),
    PoolExhausted(String),
}

impl fmt::Display for ShmTensorError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidInput(message)
            | Self::InvalidMemory(message)
            | Self::PoolExhausted(message) => formatter.write_str(message),
        }
    }
}

impl From<ShmTensorError> for PyErr {
    fn from(error: ShmTensorError) -> Self {
        match error {
            ShmTensorError::InvalidInput(message) => PyValueError::new_err(message),
            ShmTensorError::InvalidMemory(message) | ShmTensorError::PoolExhausted(message) => {
                PyRuntimeError::new_err(message)
            }
        }
    }
}

/// Owns a shared-memory allocation until the request has completed.
pub(crate) struct StagedTensor {
    allocator: Arc<SharedShmAllocator>,
    lease: Option<SharedShmLease>,
    offset: usize,
    encoded_size: usize,
    lease_token: u64,
}

impl StagedTensor {
    pub(crate) fn offset(&self) -> usize {
        self.offset
    }

    pub(crate) fn encoded_size(&self) -> usize {
        self.encoded_size
    }

    pub(crate) fn lease_token(&self) -> u64 {
        self.lease_token
    }

    /// Transfer responsibility for releasing the input slot to the server.
    pub(crate) fn transfer_to_server(&mut self) {
        self.lease = None;
    }
}

impl Drop for StagedTensor {
    fn drop(&mut self) {
        if let Some(lease) = self.lease.take() {
            let _ = self.allocator.release(lease);
        }
    }
}

struct LeaseReleaseGuard<'a> {
    allocator: &'a SharedShmAllocator,
    lease: Option<SharedShmLease>,
}

/// Best-effort release for a lease copied from response metadata.
///
/// Tensor/error readers renew the exact generation and own their own guard.
/// Keeping this outer guard ensures malformed offsets, sizes, or headers do not
/// leave the original advertised lease live until its timeout.
pub(crate) struct WireLeaseReleaseGuard<'a> {
    allocator: &'a SharedShmAllocator,
    offset: u64,
    token: u64,
}

impl<'a> WireLeaseReleaseGuard<'a> {
    pub(crate) fn new(allocator: &'a SharedShmAllocator, offset: u64, token: u64) -> Option<Self> {
        (offset > 0 && token > 0).then_some(Self {
            allocator,
            offset,
            token,
        })
    }
}

impl Drop for WireLeaseReleaseGuard<'_> {
    fn drop(&mut self) {
        if let Ok(offset) = usize::try_from(self.offset) {
            let _ = self.allocator.release_wire(offset, self.token);
        }
    }
}

impl<'a> LeaseReleaseGuard<'a> {
    fn new(allocator: &'a SharedShmAllocator, lease: SharedShmLease) -> Self {
        Self {
            allocator,
            lease: Some(lease),
        }
    }

    fn lease(&self) -> SharedShmLease {
        self.lease.expect("lease guard is active")
    }
}

impl Drop for LeaseReleaseGuard<'_> {
    fn drop(&mut self) {
        if let Some(lease) = self.lease.take() {
            let _ = self.allocator.release(lease);
        }
    }
}

pub(crate) fn create_tensor_allocator(
    shm: &Arc<ShmManager>,
) -> Result<Arc<SharedShmAllocator>, ShmTensorError> {
    let pool_offset = shm.tensor_pool_offset();
    let pool_size = shm.max_tensor_size();
    checked_region(shm.size(), pool_offset, pool_size, "tensor pool")?;

    Ok(Arc::new(SharedShmAllocator::connect(
        shm.clone(),
        SLOT_LEASE_TTL,
    )))
}

pub(crate) fn parse_shm_dtype(value: &str) -> Result<TensorDtype, ShmTensorError> {
    let dtype = TensorDtype::from_str(value)
        .map_err(|error| ShmTensorError::InvalidInput(error.to_string()))?;
    encode_dtype(dtype)?;
    Ok(dtype)
}

pub(crate) fn stage_tensor(
    shm: &ShmManager,
    allocator: &Arc<SharedShmAllocator>,
    shape: &[i64],
    dtype: TensorDtype,
    data: &[u8],
) -> Result<StagedTensor, ShmTensorError> {
    let header = build_header(shape, dtype, data.len())?;
    let encoded_size = tensor_size(data.len())?;
    let lease = allocator.try_allocate(encoded_size).ok_or_else(|| {
        ShmTensorError::PoolExhausted(format!(
            "SHM tensor pool exhausted (required={} bytes, largest_slot={} bytes, layout={})",
            encoded_size,
            allocator.largest_slot_size(),
            allocator.layout_summary(),
        ))
    })?;

    if let Err(error) = write_tensor(shm, lease.offset(), &header, data) {
        let _ = allocator.release(lease);
        return Err(error);
    }

    Ok(StagedTensor {
        allocator: Arc::clone(allocator),
        lease: Some(lease),
        offset: lease.offset(),
        encoded_size,
        lease_token: lease.token(),
    })
}

pub(crate) fn read_tensor(
    shm: &ShmManager,
    allocator: &SharedShmAllocator,
    offset: usize,
    encoded_size: usize,
    lease_token: u64,
) -> Result<BinaryTensorPacket, ShmTensorError> {
    let header_size = size_of::<TensorHeader>();
    let lease = allocator
        .acquire(offset, encoded_size, lease_token)
        .ok_or_else(|| {
            ShmTensorError::InvalidMemory(
                "SHM response carries an invalid or expired tensor lease".to_string(),
            )
        })?;
    let lease_guard = LeaseReleaseGuard::new(allocator, lease);
    if encoded_size < header_size {
        return Err(ShmTensorError::InvalidMemory(format!(
            "SHM tensor payload is too small: {} bytes (minimum {})",
            encoded_size, header_size
        )));
    }

    // SAFETY: the complete advertised region was checked above. `read_unaligned`
    // avoids assuming that a peer-supplied offset is naturally aligned.
    let header =
        unsafe { std::ptr::read_unaligned(shm.as_ptr().add(offset).cast::<TensorHeader>()) };
    let rank = usize::try_from(header.ndim).map_err(|_| {
        ShmTensorError::InvalidMemory("SHM tensor rank cannot fit in usize".to_string())
    })?;
    if rank > MAX_TENSOR_RANK {
        let error = ShmTensorError::InvalidMemory(format!(
            "SHM tensor rank {} exceeds the protocol maximum of {}",
            rank, MAX_TENSOR_RANK
        ));
        return Err(error);
    }

    let data_size = usize::try_from(header.data_size).map_err(|_| {
        ShmTensorError::InvalidMemory("SHM tensor data size cannot fit in usize".to_string())
    })?;
    let required_size = tensor_size(data_size)?;
    if required_size > encoded_size {
        return Err(ShmTensorError::InvalidMemory(format!(
            "SHM tensor header declares {} bytes, but the response advertises {} bytes",
            required_size, encoded_size
        )));
    }
    if required_size > lease_guard.lease().capacity() {
        return Err(ShmTensorError::InvalidMemory(format!(
            "SHM tensor requires {} bytes but lease capacity is {}",
            required_size,
            lease_guard.lease().capacity()
        )));
    }
    let dtype = decode_dtype(header.dtype)?;

    // SAFETY: `required_size` and the data start were checked against the
    // mapped region, and the data is copied before this function returns.
    let data = unsafe {
        std::slice::from_raw_parts(shm.as_ptr().add(offset + header_size), data_size).to_vec()
    };
    let packet = BinaryTensorPacket {
        shape: header.shape[..rank].to_vec(),
        dtype,
        data,
    };
    Ok(packet)
}

pub(crate) fn read_error_message(
    shm: &ShmManager,
    allocator: &SharedShmAllocator,
    offset: usize,
    lease_token: u64,
) -> Result<String, ShmTensorError> {
    let prefix_size = size_of::<u64>();
    let lease = allocator
        .acquire(offset, prefix_size, lease_token)
        .ok_or_else(|| {
            ShmTensorError::InvalidMemory(
                "SHM error carries an invalid or expired tensor lease".to_string(),
            )
        })?;
    let lease_guard = LeaseReleaseGuard::new(allocator, lease);

    // SAFETY: the prefix range was checked above and unaligned reads are used.
    let byte_len = unsafe { std::ptr::read_unaligned(shm.as_ptr().add(offset).cast::<u64>()) };
    let byte_len = usize::try_from(byte_len).map_err(|_| {
        ShmTensorError::InvalidMemory("SHM error length cannot fit in usize".to_string())
    })?;
    if byte_len > MAX_ERROR_MESSAGE_BYTES {
        let error = ShmTensorError::InvalidMemory(format!(
            "SHM error message exceeds the {} byte limit",
            MAX_ERROR_MESSAGE_BYTES
        ));
        return Err(error);
    }
    let total_size = prefix_size.checked_add(byte_len).ok_or_else(|| {
        ShmTensorError::InvalidMemory("SHM error message size overflow".to_string())
    })?;
    if total_size > lease_guard.lease().capacity() {
        return Err(ShmTensorError::InvalidMemory(
            "SHM error message exceeds its leased slot".to_string(),
        ));
    }

    // SAFETY: the complete error payload was checked against the mapped region.
    let bytes =
        unsafe { std::slice::from_raw_parts(shm.as_ptr().add(offset + prefix_size), byte_len) };
    let message = String::from_utf8_lossy(bytes).into_owned();
    Ok(message)
}

pub(crate) fn validate_region<T>(
    shm: &ShmManager,
    offset: usize,
    capacity: usize,
    label: &str,
) -> Result<(), ShmTensorError> {
    if !offset.is_multiple_of(align_of::<T>()) {
        return Err(ShmTensorError::InvalidMemory(format!(
            "{} offset {} is not aligned to {} bytes",
            label,
            offset,
            align_of::<T>()
        )));
    }
    let byte_len = capacity
        .checked_mul(size_of::<T>())
        .ok_or_else(|| ShmTensorError::InvalidMemory(format!("{} size overflow", label)))?;
    checked_region(shm.size(), offset, byte_len, label)
}

fn build_header(
    shape: &[i64],
    dtype: TensorDtype,
    data_size: usize,
) -> Result<TensorHeader, ShmTensorError> {
    if shape.len() > MAX_TENSOR_RANK {
        return Err(ShmTensorError::InvalidInput(format!(
            "Tensor rank {} exceeds the shared-memory maximum of {}",
            shape.len(),
            MAX_TENSOR_RANK
        )));
    }

    let mut encoded_shape = [0_i64; MAX_TENSOR_RANK];
    encoded_shape[..shape.len()].copy_from_slice(shape);
    Ok(TensorHeader {
        ndim: shape.len() as u32,
        dtype: encode_dtype(dtype)?,
        _padding: [0; 3],
        shape: encoded_shape,
        data_size: data_size as u64,
    })
}

fn write_tensor(
    shm: &ShmManager,
    offset: usize,
    header: &TensorHeader,
    data: &[u8],
) -> Result<(), ShmTensorError> {
    let encoded_size = tensor_size(data.len())?;
    checked_tensor_region(shm, offset, encoded_size)?;

    // SAFETY: the destination range was checked above. The allocator grants
    // this caller exclusive ownership of the slot until `StagedTensor` drops.
    unsafe {
        std::ptr::write_unaligned(shm.as_ptr().add(offset).cast::<TensorHeader>(), *header);
        std::ptr::copy_nonoverlapping(
            data.as_ptr(),
            shm.as_ptr().add(offset + size_of::<TensorHeader>()),
            data.len(),
        );
    }
    Ok(())
}

fn checked_tensor_region(
    shm: &ShmManager,
    offset: usize,
    byte_len: usize,
) -> Result<(), ShmTensorError> {
    if offset < shm.tensor_pool_offset() {
        return Err(ShmTensorError::InvalidMemory(format!(
            "SHM tensor offset {} is before the tensor pool at {}",
            offset,
            shm.tensor_pool_offset()
        )));
    }
    checked_region(shm.size(), offset, byte_len, "tensor payload")
}

fn checked_region(
    region_size: usize,
    offset: usize,
    byte_len: usize,
    label: &str,
) -> Result<(), ShmTensorError> {
    let end = offset
        .checked_add(byte_len)
        .ok_or_else(|| ShmTensorError::InvalidMemory(format!("{} range overflow", label)))?;
    if end > region_size {
        return Err(ShmTensorError::InvalidMemory(format!(
            "{} range {}..{} exceeds shared-memory size {}",
            label, offset, end, region_size
        )));
    }
    Ok(())
}

fn tensor_size(data_size: usize) -> Result<usize, ShmTensorError> {
    size_of::<TensorHeader>()
        .checked_add(data_size)
        .ok_or_else(|| ShmTensorError::InvalidMemory("SHM tensor size overflow".to_string()))
}

fn encode_dtype(dtype: TensorDtype) -> Result<u8, ShmTensorError> {
    match dtype {
        TensorDtype::Float32 => Ok(0),
        TensorDtype::Float64 => Ok(1),
        TensorDtype::Int32 => Ok(2),
        TensorDtype::Int64 => Ok(3),
        unsupported => Err(ShmTensorError::InvalidInput(format!(
            "Shared-memory transport does not support dtype '{}'; use float32, float64, int32, or int64",
            unsupported.as_str()
        ))),
    }
}

fn decode_dtype(value: u8) -> Result<TensorDtype, ShmTensorError> {
    match value {
        0 => Ok(TensorDtype::Float32),
        1 => Ok(TensorDtype::Float64),
        2 => Ok(TensorDtype::Int32),
        3 => Ok(TensorDtype::Int64),
        unsupported => Err(ShmTensorError::InvalidMemory(format!(
            "SHM tensor header contains unsupported dtype code {}",
            unsupported
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn header_preserves_supported_shape_and_dtype() {
        let header = build_header(&[1, 2, 3], TensorDtype::Float64, 48).expect("valid header");

        assert_eq!(header.ndim, 3);
        assert_eq!(header.dtype, 1);
        assert_eq!(&header.shape[..3], &[1, 2, 3]);
        assert_eq!(header.data_size, 48);
    }

    #[test]
    fn header_rejects_rank_above_protocol_limit() {
        let error = build_header(&[1; MAX_TENSOR_RANK + 1], TensorDtype::Float32, 0)
            .expect_err("rank must be rejected");

        assert!(matches!(error, ShmTensorError::InvalidInput(_)));
    }

    #[test]
    fn shm_dtype_rejects_types_the_wire_format_cannot_represent() {
        let error = parse_shm_dtype("float16").expect_err("float16 must be rejected");

        assert!(matches!(error, ShmTensorError::InvalidInput(_)));
    }

    #[test]
    fn staged_tensor_round_trips_through_a_mapped_region() {
        // macOS limits POSIX shared-memory names to 31 bytes.
        let name = format!("/kp3_{}", std::process::id());
        let shm =
            Arc::new(ShmManager::create(&name, 2 * 1024 * 1024).expect("create shared memory"));
        let allocator = create_tensor_allocator(&shm).expect("create allocator");
        let data = [1_u8, 2, 3, 4, 5, 6, 7, 8];
        let staged = stage_tensor(&shm, &allocator, &[1, 2], TensorDtype::Int32, &data)
            .expect("stage tensor");

        let decoded = read_tensor(
            &shm,
            &allocator,
            staged.offset(),
            staged.encoded_size(),
            staged.lease_token(),
        )
        .expect("read staged tensor");

        assert_eq!(decoded.shape, vec![1, 2]);
        assert_eq!(decoded.dtype, TensorDtype::Int32);
        assert_eq!(decoded.data, data);
    }

    #[test]
    fn checked_region_rejects_overflow_and_out_of_bounds_ranges() {
        assert!(checked_region(128, 120, 8, "test").is_ok());
        assert!(checked_region(128, 120, 9, "test").is_err());
        assert!(checked_region(128, usize::MAX, 2, "test").is_err());
    }
}
