//! ONNX Runtime CUDA allocator backed by the shared Kapsl `GpuBlockPool`.
//!
//! Registers a custom `OrtAllocator` on the global ORT environment so that
//! CUDA execution-provider sessions draw device memory from the same block
//! pool as the GGUF KV cache, giving multi-model deployments a single GPU
//! memory budget instead of two independent arenas.
//!
//! # Flow
//!
//! 1. The runtime creates (or obtains, e.g. via `GgufBackend::pool_handle()`)
//!    a `GpuPoolHandle` for a device and calls [`register_pool_allocator`].
//! 2. ORT sessions on that device opt in with the session config entry
//!    `session.use_env_allocators = 1` ([`USE_ENV_ALLOCATORS_KEY`]);
//!    `OnnxBackend` does this automatically when a pool is registered.
//! 3. Each ORT device allocation is served as a contiguous run of pool
//!    blocks; frees return the run to the pool.
//!
//! The registered allocator is matched by ORT against the memory info
//! (`"Cuda"`, device id, `OrtMemTypeDefault`), so CPU-side allocations and
//! pinned buffers are unaffected.

use std::collections::HashMap;
use std::ffi::{c_void, CStr};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::{Arc, Mutex, OnceLock};

use kapsl_hal::gpu_arena::{GpuBlockPool, GpuPoolHandle};
use ort::memory::{AllocationDevice, AllocatorType, MemoryInfo, MemoryType};
use ort::sys as ort_sys;
use ort::AsPointer;

/// Session config entry that makes a session use environment-registered
/// allocators (see ORT's "allocator sharing between sessions" docs).
pub const USE_ENV_ALLOCATORS_KEY: &str = "session.use_env_allocators";

/// CUDA device pointers handed to ORT must be at least 256-byte aligned.
const CUDA_ALLOC_ALIGN: usize = 256;

/// A live ORT allocation: a contiguous run of pool blocks.
struct BlockRun {
    first: u32,
    count: usize,
}

struct AllocState {
    pool: Arc<GpuBlockPool>,
    /// Kept alive for the allocator's lifetime; `Info` returns its pointer.
    mem_info: MemoryInfo,
    bytes_per_block: usize,
    device_id: i32,
    /// Device pointer → run, so `Free` can return blocks to the pool.
    live: Mutex<HashMap<usize, BlockRun>>,
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
        let blocks = size.div_ceil(state.bytes_per_block);
        match state.pool.alloc_blocks_contiguous(blocks) {
            Ok(first) => {
                let ptr = state.pool.block_device_ptr(first) as usize;
                state.live.lock().unwrap().insert(ptr, BlockRun { first, count: blocks });
                ptr as *mut c_void
            }
            Err(e) => {
                log::error!(
                    "ORT pool allocator (device {}): failed to allocate {} bytes ({} blocks): {}",
                    state.device_id, size, blocks, e
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
        match state.live.lock().unwrap().remove(&(p as usize)) {
            Some(run) => state.pool.free_blocks_contiguous(run.first, run.count),
            None => log::warn!(
                "ORT pool allocator (device {}): free of unknown pointer {:p}",
                state.device_id, p
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
    handle: GpuPoolHandle,
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

/// Register `handle`'s pool as the ORT device allocator for `device_id`.
///
/// Idempotent for the same pool; registering a *different* pool for a device
/// that already has one is an error. After this call, `OnnxBackend` sessions
/// on this device automatically allocate from the shared pool.
pub fn register_pool_allocator(device_id: i32, handle: &GpuPoolHandle) -> Result<(), String> {
    let pool = handle.pool.clone();
    let bytes_per_block = pool.bytes_per_block();
    if bytes_per_block == 0 || bytes_per_block % CUDA_ALLOC_ALIGN != 0 {
        return Err(format!(
            "pool block size ({bytes_per_block} bytes) is not a multiple of the required \
             {CUDA_ALLOC_ALIGN}-byte CUDA alignment; cannot serve ORT allocations"
        ));
    }

    let mut reg = registry().lock().unwrap();
    if let Some(existing) = reg.get(&device_id) {
        if Arc::ptr_eq(&existing.handle.pool, &pool) {
            return Ok(());
        }
        return Err(format!(
            "a different GpuBlockPool is already registered with ORT for device {device_id}"
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
            pool: pool.clone(),
            mem_info,
            bytes_per_block,
            device_id,
            live: Mutex::new(HashMap::new()),
        },
    });
    // ORT keeps the raw pointer for the environment's lifetime; the
    // registration entry pins the environment, so leak the allocator.
    let allocator: &'static mut PoolOrtAllocator = Box::leak(allocator);

    let status = unsafe {
        (ort::api().RegisterAllocator)(env.ptr().cast_mut(), &mut allocator.ort)
    };
    status_to_result(status).map_err(|e| format!("ORT RegisterAllocator failed: {e}"))?;

    log::info!(
        "Registered shared GPU pool with ORT for device {}: {} blocks × {} KiB ({} MiB)",
        device_id,
        pool.total_blocks(),
        bytes_per_block / 1024,
        pool.capacity_bytes() / (1024 * 1024),
    );
    reg.insert(
        device_id,
        Registration {
            handle: handle.clone(),
            _env: env,
        },
    );
    Ok(())
}

/// Whether a shared pool allocator has been registered for `device_id`.
pub fn is_registered(device_id: i32) -> bool {
    registry().lock().unwrap().contains_key(&device_id)
}

/// The pool handle registered for `device_id`, if any. Lets the factory hand
/// the same pool to GGUF backends created after registration.
pub fn registered_pool_handle(device_id: i32) -> Option<GpuPoolHandle> {
    registry()
        .lock()
        .unwrap()
        .get(&device_id)
        .map(|r| r.handle.clone())
}
