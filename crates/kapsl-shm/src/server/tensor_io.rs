use super::*;
use crate::allocator::{SharedShmAllocator, SharedShmLease};

const ERROR_LEN_PREFIX_BYTES: usize = std::mem::size_of::<u64>();

/// Copy a validated leased tensor out of shared memory and release its slot.
pub(super) fn take_request_tensor(
    shm: &ShmManager,
    allocator: &SharedShmAllocator,
    request: &ShmRequest,
) -> Result<BinaryTensorPacket, String> {
    let offset = usize::try_from(request.tensor_offset)
        .map_err(|_| "SHM request tensor offset exceeds platform limits".to_string())?;
    let encoded_size = usize::try_from(request.tensor_size)
        .map_err(|_| "SHM request tensor size exceeds platform limits".to_string())?;
    let lease = allocator
        .acquire(offset, encoded_size, request.tensor_lease)
        .ok_or_else(|| "SHM request carries an invalid or expired tensor lease".to_string())?;

    let result = decode_tensor(shm, lease, encoded_size);
    let _ = allocator.release(lease);
    result
}

/// Encode a tensor into a process-shared leased slot.
pub(super) fn stage_response_tensor(
    shm: &ShmManager,
    allocator: &SharedShmAllocator,
    tensor: &BinaryTensorPacket,
) -> Result<(SharedShmLease, usize), String> {
    let encoded_size = std::mem::size_of::<TensorHeader>()
        .checked_add(tensor.data.len())
        .ok_or_else(|| "SHM response tensor size overflow".to_string())?;
    let lease = allocator.try_allocate(encoded_size).ok_or_else(|| {
        format!(
            "SHM tensor pool exhausted (required={} bytes, largest_slot={} bytes, layout={})",
            encoded_size,
            allocator.largest_slot_size(),
            allocator.layout_summary()
        )
    })?;
    if let Err(error) = encode_tensor(shm, lease, tensor) {
        let _ = allocator.release(lease);
        return Err(error);
    }
    Ok((lease, encoded_size))
}

/// Build an error response, storing the message in a leased payload when space
/// is available. A zero offset/token still delivers a typed error status when
/// the pool itself is exhausted.
pub(super) fn error_response(
    shm: &ShmManager,
    allocator: &SharedShmAllocator,
    request_id: u64,
    latency_ns: u64,
    error: &str,
) -> ShmResponse {
    let bytes = error.as_bytes();
    let required_size = ERROR_LEN_PREFIX_BYTES.saturating_add(bytes.len());
    let payload = allocator.try_allocate(required_size).and_then(|lease| {
        if write_error_payload(shm, lease, bytes).is_ok() {
            Some(lease)
        } else {
            let _ = allocator.release(lease);
            None
        }
    });

    ShmResponse {
        metadata: ResponseMetadata::error(request_id, latency_ns),
        result_offset: 0,
        result_size: 0,
        error_offset: payload.map_or(0, |lease| lease.offset() as u64),
        payload_lease: payload.map_or(0, |lease| lease.token()),
    }
}

fn decode_tensor(
    shm: &ShmManager,
    lease: SharedShmLease,
    encoded_size: usize,
) -> Result<BinaryTensorPacket, String> {
    let header_size = std::mem::size_of::<TensorHeader>();
    if encoded_size < header_size {
        return Err(format!(
            "SHM tensor payload is too small: {encoded_size} bytes"
        ));
    }

    // SAFETY: the complete encoded region is protected by `lease` and bounded
    // by its capacity. Unaligned access avoids trusting a peer's alignment.
    let header = unsafe {
        std::ptr::read_unaligned(shm.as_ptr().add(lease.offset()).cast::<TensorHeader>())
    };
    let rank = header.ndim as usize;
    if rank > header.shape.len() {
        return Err(format!(
            "SHM tensor rank {rank} exceeds maximum {}",
            header.shape.len()
        ));
    }
    let data_size = usize::try_from(header.data_size)
        .map_err(|_| "SHM tensor data size exceeds platform limits".to_string())?;
    let required_size = header_size
        .checked_add(data_size)
        .ok_or_else(|| "SHM tensor encoded size overflow".to_string())?;
    if required_size > encoded_size || required_size > lease.capacity() {
        return Err(format!(
            "SHM tensor header requires {required_size} bytes but lease advertises {encoded_size}"
        ));
    }
    let dtype = decode_dtype(header.dtype)?;
    // SAFETY: required_size was checked against both the lease and mapping.
    let data = unsafe {
        std::slice::from_raw_parts(shm.as_ptr().add(lease.offset() + header_size), data_size)
            .to_vec()
    };
    Ok(BinaryTensorPacket {
        shape: header.shape[..rank].to_vec(),
        dtype,
        data,
    })
}

fn encode_tensor(
    shm: &ShmManager,
    lease: SharedShmLease,
    tensor: &BinaryTensorPacket,
) -> Result<(), String> {
    if tensor.shape.len() > 8 {
        return Err(format!(
            "SHM tensor rank {} exceeds maximum 8",
            tensor.shape.len()
        ));
    }
    let required_size = std::mem::size_of::<TensorHeader>()
        .checked_add(tensor.data.len())
        .ok_or_else(|| "SHM tensor encoded size overflow".to_string())?;
    if required_size > lease.capacity() {
        return Err(format!(
            "SHM tensor requires {required_size} bytes but slot capacity is {}",
            lease.capacity()
        ));
    }

    let mut shape = [0_i64; 8];
    shape[..tensor.shape.len()].copy_from_slice(&tensor.shape);
    let header = TensorHeader {
        ndim: tensor.shape.len() as u32,
        dtype: encode_dtype(tensor.dtype)?,
        _padding: [0; 3],
        shape,
        data_size: tensor.data.len() as u64,
    };
    // SAFETY: this caller exclusively owns `lease`, and both writes fit its
    // capacity inside the mapped tensor pool.
    unsafe {
        std::ptr::write_unaligned(
            shm.as_ptr().add(lease.offset()).cast::<TensorHeader>(),
            header,
        );
        std::ptr::copy_nonoverlapping(
            tensor.data.as_ptr(),
            shm.as_ptr()
                .add(lease.offset() + std::mem::size_of::<TensorHeader>()),
            tensor.data.len(),
        );
    }
    Ok(())
}

fn write_error_payload(
    shm: &ShmManager,
    lease: SharedShmLease,
    bytes: &[u8],
) -> Result<(), String> {
    let required_size = ERROR_LEN_PREFIX_BYTES
        .checked_add(bytes.len())
        .ok_or_else(|| "SHM error payload size overflow".to_string())?;
    if required_size > lease.capacity() {
        return Err("SHM error payload exceeds its leased slot".to_string());
    }
    // SAFETY: the length and bytes fit the exclusively leased slot.
    unsafe {
        std::ptr::write_unaligned(
            shm.as_ptr().add(lease.offset()).cast::<u64>(),
            bytes.len() as u64,
        );
        std::ptr::copy_nonoverlapping(
            bytes.as_ptr(),
            shm.as_ptr().add(lease.offset() + ERROR_LEN_PREFIX_BYTES),
            bytes.len(),
        );
    }
    Ok(())
}

fn encode_dtype(dtype: TensorDtype) -> Result<u8, String> {
    match dtype {
        TensorDtype::Float32 => Ok(0),
        TensorDtype::Float64 => Ok(1),
        TensorDtype::Int32 => Ok(2),
        TensorDtype::Int64 => Ok(3),
        unsupported => Err(format!("Unsupported SHM tensor dtype {unsupported}")),
    }
}

fn decode_dtype(dtype: u8) -> Result<TensorDtype, String> {
    match dtype {
        0 => Ok(TensorDtype::Float32),
        1 => Ok(TensorDtype::Float64),
        2 => Ok(TensorDtype::Int32),
        3 => Ok(TensorDtype::Int64),
        unsupported => Err(format!("Unsupported SHM tensor dtype code {unsupported}")),
    }
}
