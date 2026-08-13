use super::*;

/// Read tensor from shared memory
pub(super) unsafe fn read_tensor_from_shm(base: *mut u8, offset: usize) -> BinaryTensorPacket {
    let header_ptr = base.add(offset) as *const TensorHeader;
    let header = std::ptr::read(header_ptr);

    // Read shape
    let shape: Vec<i64> = header.shape[0..header.ndim as usize].to_vec();

    // Read dtype (simple mapping)
    let dtype = match header.dtype {
        0 => TensorDtype::Float32,
        1 => TensorDtype::Float64,
        2 => TensorDtype::Int32,
        3 => TensorDtype::Int64,
        _ => TensorDtype::Float32,
    };

    // Read data
    let data_ptr = base.add(offset + std::mem::size_of::<TensorHeader>());
    let data = std::slice::from_raw_parts(data_ptr, header.data_size as usize).to_vec();

    BinaryTensorPacket { shape, dtype, data }
}

/// Write tensor to shared memory
pub(super) unsafe fn write_tensor_to_shm(
    base: *mut u8,
    offset: usize,
    tensor: &BinaryTensorPacket,
) {
    // Write header
    let mut shape_array = [0i64; 8];
    for (i, &s) in tensor.shape.iter().enumerate() {
        shape_array[i] = s;
    }

    let dtype_byte = match tensor.dtype {
        TensorDtype::Float32 => 0,
        TensorDtype::Float64 => 1,
        TensorDtype::Int32 => 2,
        TensorDtype::Int64 => 3,
        _ => 0,
    };

    let header = TensorHeader {
        ndim: tensor.shape.len() as u32,
        dtype: dtype_byte,
        _padding: [0; 3],
        shape: shape_array,
        data_size: tensor.data.len() as u64,
    };

    let header_ptr = base.add(offset) as *mut TensorHeader;
    std::ptr::write(header_ptr, header);

    // Write data
    let data_ptr = base.add(offset + std::mem::size_of::<TensorHeader>());
    std::ptr::copy_nonoverlapping(tensor.data.as_ptr(), data_ptr, tensor.data.len());
}

pub(super) unsafe fn push_response_and_notify(shm: &ShmManager, response: ShmResponse) {
    let resp_queue: LockFreeRingBuffer<ShmResponse> = LockFreeRingBuffer::connect(
        shm.as_ptr().add(shm.response_queue_offset()) as *mut ShmResponse,
        1024,
    );
    let _ = resp_queue.push(response);

    // Notify via pipe
    let write_fd = shm.notify_write_fd();
    if write_fd >= 0 {
        let byte: u8 = 1;
        libc::write(write_fd, &byte as *const u8 as *const libc::c_void, 1);
    }
}

#[allow(dead_code)]
pub(super) fn allocate_pool_slot(
    allocator: &(impl ShmPoolAllocator + ?Sized),
    required_size: usize,
    metrics: Option<&ShmPoolMetrics>,
) -> Option<usize> {
    let offset = allocator.try_allocate(required_size);
    if offset.is_none() {
        if let Some(m) = metrics {
            m.on_exhausted();
        }
    }
    if let Some(m) = metrics {
        m.update_from_snapshot(allocator.snapshot());
    }
    offset
}

/// Allocate from the model's dedicated sub-pool (with shared pool fallback).
pub(super) fn allocate_pool_slot_for_model(
    allocator: &DynamicPerModelPool,
    model_id: u32,
    required_size: usize,
    metrics: Option<&ShmPoolMetrics>,
) -> Option<usize> {
    let offset = allocator.try_allocate(model_id, required_size);
    if offset.is_none() {
        if let Some(m) = metrics {
            m.on_exhausted();
        }
    }
    if let Some(m) = metrics {
        m.update_from_snapshot(allocator.snapshot());
    }
    offset
}

/// Write error message to shared memory.
/// Layout: `[u64 error_len][error bytes]`.
#[allow(dead_code)]
pub(super) fn write_error_to_shm(
    base: *mut u8,
    allocator: &(impl ShmPoolAllocator + ?Sized),
    metrics: Option<&ShmPoolMetrics>,
    error: &str,
) -> Option<usize> {
    let bytes = error.as_bytes();
    let total_size = ERROR_LEN_PREFIX_BYTES + bytes.len();
    let offset = allocate_pool_slot(allocator, total_size, metrics)?;
    unsafe {
        let len_ptr = base.add(offset) as *mut u64;
        std::ptr::write(len_ptr, bytes.len() as u64);
        let ptr = base.add(offset + ERROR_LEN_PREFIX_BYTES);
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), ptr, bytes.len());
    }
    Some(offset)
}

/// Write error message using the per-model sub-pool allocator.
pub(super) fn write_error_to_shm_for_model(
    base: *mut u8,
    allocator: &DynamicPerModelPool,
    model_id: u32,
    metrics: Option<&ShmPoolMetrics>,
    error: &str,
) -> Option<usize> {
    let bytes = error.as_bytes();
    let total_size = ERROR_LEN_PREFIX_BYTES + bytes.len();
    let offset = allocate_pool_slot_for_model(allocator, model_id, total_size, metrics)?;
    unsafe {
        let len_ptr = base.add(offset) as *mut u64;
        std::ptr::write(len_ptr, bytes.len() as u64);
        let ptr = base.add(offset + ERROR_LEN_PREFIX_BYTES);
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), ptr, bytes.len());
    }
    Some(offset)
}

pub(super) fn error_response_for_model(
    base: *mut u8,
    allocator: &DynamicPerModelPool,
    model_id: u32,
    request_id: u64,
    latency_ns: u64,
    metrics: Option<&ShmPoolMetrics>,
    error: &str,
) -> ShmResponse {
    let error_offset =
        write_error_to_shm_for_model(base, allocator, model_id, metrics, error).unwrap_or(0);
    ShmResponse {
        metadata: ResponseMetadata::error(request_id, latency_ns),
        result_offset: 0,
        result_size: 0,
        error_offset: error_offset as u64,
    }
}
