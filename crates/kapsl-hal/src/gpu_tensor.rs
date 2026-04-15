//! GPU tensor types wrapping cudarc allocations.
//!
//! These are thin wrappers around `CudaSlice<T>` that carry shape metadata,
//! used by CUDA kernels and cuBLAS calls in the native backend.

#[cfg(feature = "cuda")]
use cudarc::driver::{CudaDevice, CudaSlice};
#[cfg(feature = "cuda")]
use std::sync::Arc;

/// An owned tensor living on a CUDA device.
#[cfg(feature = "cuda")]
pub struct GpuTensor<T: cudarc::driver::DeviceRepr> {
    pub data: CudaSlice<T>,
    pub shape: Vec<usize>,
}

#[cfg(feature = "cuda")]
impl<T: cudarc::driver::DeviceRepr> GpuTensor<T> {
    /// Allocate a zeroed tensor on `device`.
    pub fn zeros(
        device: &Arc<CudaDevice>,
        shape: &[usize],
    ) -> Result<Self, cudarc::driver::DriverError> {
        let numel: usize = shape.iter().product();
        let data = device.alloc_zeros::<T>(numel)?;
        Ok(Self {
            data,
            shape: shape.to_vec(),
        })
    }

    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }
}

#[cfg(feature = "cuda")]
impl<T: cudarc::driver::DeviceRepr + Clone> GpuTensor<T> {
    /// Upload host data to a new GPU tensor.
    pub fn from_host(
        device: &Arc<CudaDevice>,
        host: &[T],
        shape: &[usize],
    ) -> Result<Self, cudarc::driver::DriverError> {
        let data = device.htod_sync_copy(host)?;
        Ok(Self {
            data,
            shape: shape.to_vec(),
        })
    }
}

#[cfg(feature = "cuda")]
impl GpuTensor<half::f16> {
    /// Upload from raw f16 bytes (little-endian, 2 bytes per element).
    pub fn from_f16_bytes(
        device: &Arc<CudaDevice>,
        bytes: &[u8],
        shape: &[usize],
    ) -> Result<Self, cudarc::driver::DriverError> {
        let numel: usize = shape.iter().product();
        assert_eq!(bytes.len(), numel * 2);
        let f16_slice: &[half::f16] =
            unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const half::f16, numel) };
        Self::from_host(device, f16_slice, shape)
    }
}

#[cfg(feature = "cuda")]
impl<T: cudarc::driver::DeviceRepr + Default + Clone> GpuTensor<T> {
    /// Download tensor to host.
    pub fn to_host(
        &self,
        device: &Arc<CudaDevice>,
    ) -> Result<Vec<T>, cudarc::driver::DriverError> {
        device.dtoh_sync_copy(&self.data)
    }
}

#[cfg(feature = "cuda")]
impl<T: cudarc::driver::DeviceRepr> std::fmt::Debug for GpuTensor<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "GpuTensor<{}>({:?})", std::any::type_name::<T>(), self.shape)
    }
}
