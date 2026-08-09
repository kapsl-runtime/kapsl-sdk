//! ONNX Runtime CUDA allocator backed by the runtime-owned `GpuDevicePool`.
//!
//! Registers a custom `OrtAllocator` on the global ORT environment so that
//! CUDA execution-provider sessions draw device memory from the same device
//! pool as GGUF KV views, giving multi-model deployments a single GPU
//! memory budget instead of two independent arenas.
//!
//! # Flow
//!
//! 1. The runtime creates a `GpuDevicePool` for a device and calls
//!    [`register_pool_allocator`] before constructing backend sessions.
//! 2. ORT sessions on that device opt in with the session config entry
//!    `session.use_env_allocators = 1` ([`USE_ENV_ALLOCATORS_KEY`]);
//!    `OnnxBackend` does this automatically when a pool is registered.
//! 3. Each ORT device allocation is served as an aligned byte extent; frees
//!    return that exact allocation to the pool.
//!
//! The registered allocator is matched by ORT against the memory info
//! (`"Cuda"`, device id, `OrtMemTypeDefault`), so CPU-side allocations and
//! pinned buffers are unaffected.

use std::collections::HashMap;
use std::ffi::{c_void, CStr};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::{Arc, Mutex, OnceLock};

use kapsl_hal::gpu_arena::{GpuAllocation, GpuDevicePool, PoolOwner};
use ort::memory::{AllocationDevice, AllocatorType, MemoryInfo, MemoryType};
use ort::sys as ort_sys;
use ort::AsPointer;

/// Session config entry that makes a session use environment-registered
/// allocators (see ORT's "allocator sharing between sessions" docs).
pub const USE_ENV_ALLOCATORS_KEY: &str = "session.use_env_allocators";

/// CUDA device pointers handed to ORT must be at least 256-byte aligned.
const CUDA_ALLOC_ALIGN: usize = 256;

struct AllocState {
    pool: Arc<GpuDevicePool>,
    /// Kept alive for the allocator's lifetime; `Info` returns its pointer.
    mem_info: MemoryInfo,
    device_id: i32,
    /// Device pointer → allocation, so `Free` returns the exact owned extent.
    live: Mutex<HashMap<usize, GpuAllocation>>,
}

#[repr(C)]
struct PoolOrtAllocator {
    /// C vtable. MUST stay the first field: ORT holds `*mut OrtAllocator`
    /// and the callbacks cast it back to `PoolOrtAllocator`.
    ort: ort_sys::OrtAllocator,
    state: AllocState,
}

unsafe extern "system" fn pool_alloc(
    this_: *mut ort_sys::OrtAllocator,
    size: usize,
) -> *mut c_void {
    catch_unwind(AssertUnwindSafe(|| {
        let state = unsafe { &(*(this_ as *const PoolOrtAllocator)).state };
        if size == 0 {
            return std::ptr::null_mut();
        }
        match state.pool.alloc(PoolOwner::Onnx, size, CUDA_ALLOC_ALIGN) {
            Ok(allocation) => {
                let ptr = state.pool.allocation_ptr(&allocation) as usize;
                state.live.lock().unwrap().insert(ptr, allocation);
                ptr as *mut c_void
            }
            Err(e) => {
                log::error!(
                    "ORT pool allocator (device {}): failed to allocate {} bytes: {}",
                    state.device_id,
                    size,
                    e
                );
                std::ptr::null_mut()
            }
        }
    }))
    .unwrap_or(std::ptr::null_mut())
}

unsafe extern "system" fn pool_free(this_: *mut ort_sys::OrtAllocator, p: *mut c_void) {
    let _ = catch_unwind(AssertUnwindSafe(|| {
        if p.is_null() {
            return;
        }
        let state = unsafe { &(*(this_ as *const PoolOrtAllocator)).state };
        let allocation = { state.live.lock().unwrap().remove(&(p as usize)) };
        match allocation {
            Some(allocation) => {
                if let Err(error) = state.pool.free(allocation.clone()) {
                    state
                        .live
                        .lock()
                        .unwrap()
                        .insert(p as usize, allocation);
                    log::error!(
                        "ORT pool allocator (device {}): failed to free {:p}: {}",
                        state.device_id,
                        p,
                        error
                    );
                }
            }
            None => log::warn!(
                "ORT pool allocator (device {}): free of unknown pointer {:p}",
                state.device_id,
                p
            ),
        }
    }));
}

unsafe extern "system" fn pool_info(
    this_: *const ort_sys::OrtAllocator,
) -> *const ort_sys::OrtMemoryInfo {
    let state = unsafe { &(*(this_ as *const PoolOrtAllocator)).state };
    state.mem_info.ptr()
}

unsafe extern "system" fn pool_reserve(
    this_: *const ort_sys::OrtAllocator,
    size: usize,
) -> *mut c_void {
    // Reserve bypasses arena bookkeeping in ORT's own allocators; for the
    // block pool it is the same operation as Alloc.
    unsafe { pool_alloc(this_ as *mut ort_sys::OrtAllocator, size) }
}

struct Registration {
    pool: Arc<GpuDevicePool>,
    /// Pins the ORT environment the allocator was registered on, so it (and
    /// the leaked allocator vtable) outlive every session that may use it.
    _env: Arc<ort::environment::Environment>,
}

static REGISTRY: OnceLock<Mutex<HashMap<i32, Registration>>> = OnceLock::new();

fn registry() -> &'static Mutex<HashMap<i32, Registration>> {
    REGISTRY.get_or_init(|| Mutex::new(HashMap::new()))
}

fn status_to_result(status: ort_sys::OrtStatusPtr) -> Result<(), String> {
    if status.0.is_null() {
        return Ok(());
    }
    unsafe {
        let msg = CStr::from_ptr((ort::api().GetErrorMessage)(status.0))
            .to_string_lossy()
            .into_owned();
        (ort::api().ReleaseStatus)(status.0);
        Err(msg)
    }
}

/// Register the runtime-owned pool as the ORT device allocator for `device_id`.
///
/// Idempotent for the same pool; registering a *different* pool for a device
/// that already has one is an error. After this call, `OnnxBackend` sessions
/// on this device automatically allocate from the shared pool.
pub fn register_pool_allocator(device_id: i32, pool: &Arc<GpuDevicePool>) -> Result<(), String> {
    let mut reg = registry().lock().unwrap();
    if let Some(existing) = reg.get(&device_id) {
        if Arc::ptr_eq(&existing.pool, pool) {
            return Ok(());
        }
        return Err(format!(
            "a different GpuDevicePool is already registered with ORT for device {device_id}"
        ));
    }

    let env = ort::environment::get_environment()
        .map_err(|e| format!("failed to obtain ORT environment: {e}"))?;
    let mem_info = MemoryInfo::new(
        AllocationDevice::CUDA,
        device_id,
        AllocatorType::Device,
        MemoryType::Default,
    )
    .map_err(|e| format!("failed to create ORT memory info: {e}"))?;

    let allocator = Box::new(PoolOrtAllocator {
        ort: ort_sys::OrtAllocator {
            version: ort_sys::ORT_API_VERSION,
            Alloc: Some(pool_alloc),
            Free: Some(pool_free),
            Info: Some(pool_info),
            Reserve: Some(pool_reserve),
        },
        state: AllocState {
            pool: Arc::clone(pool),
            mem_info,
            device_id,
            live: Mutex::new(HashMap::new()),
        },
    });
    // ORT keeps the raw pointer for the environment's lifetime; the
    // registration entry pins the environment, so leak the allocator.
    let allocator: &'static mut PoolOrtAllocator = Box::leak(allocator);

    let status =
        unsafe { (ort::api().RegisterAllocator)(env.ptr().cast_mut(), &mut allocator.ort) };
    status_to_result(status).map_err(|e| format!("ORT RegisterAllocator failed: {e}"))?;

    log::info!(
        "Registered runtime GPU device pool with ORT for device {}: {} MiB",
        device_id,
        pool.capacity_bytes() / (1024 * 1024),
    );
    reg.insert(
        device_id,
        Registration {
            pool: Arc::clone(pool),
            _env: env,
        },
    );
    Ok(())
}

/// Whether a shared pool allocator has been registered for `device_id`.
pub fn is_registered(device_id: i32) -> bool {
    registry().lock().unwrap().contains_key(&device_id)
}
