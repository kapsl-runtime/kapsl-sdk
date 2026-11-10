//! Measured memory for ONNX Runtime sessions.
//!
//! `OnnxBackend` used to answer `actual_memory()` with its own plan, so the
//! engine saw "observed == lease" for every ONNX model. This module reads what
//! ORT and the process really hold:
//!
//! * ORT allocator statistics (`OrtAllocator::GetStats`, ORT API 23, keys
//!   `InUse`, `TotalAllocated`, `MaxInUse`, ...) for the session's CPU arena
//!   and, for CUDA, its device arena;
//! * the process resident size around a load, for memory ORT cannot see
//!   (CoreML's compiled model, mapped weights);
//! * the Kapsl GPU pool's per-owner usage, when the pool allocator is
//!   registered (see `pool_allocator.rs`).
//!
//! The pure parts (parsing, aggregation, report building) take plain values so
//! they are tested with fakes. Only [`session_allocator_stats`] and
//! [`process_resident_bytes`] touch ORT or the OS.
//!
//! ## Why `GetStats` is read through a local struct
//!
//! ort-sys 2.0.0-rc.11 advertises ORT API 23, but its `OrtAllocator` ends at
//! `Reserve` and `OrtApi` ends before `AllocatorGetStats` (the same gap
//! `pool_allocator.rs` documents). The C header for v1.23.2
//! (`onnxruntime_c_api.h`, `struct OrtAllocator`) places
//! `GetStats(const OrtAllocator*, OrtKeyValuePairs**)` directly after
//! `Reserve`. [`AllocatorV23`] repeats that one trailing field. It is only read
//! when the allocator's own `version` is at least 23, because older
//! allocators (including the Kapsl pool allocator, which reports 22) keep
//! unrelated state after `Reserve`.

use std::ffi::CStr;
use std::os::raw::c_char;

use ort::memory::{AllocationDevice, Allocator, AllocatorType, MemoryInfo, MemoryType};
use ort::session::Session;
use ort::sys as ort_sys;
use ort::AsPointer;

/// First ORT API version whose `OrtAllocator` carries `GetStats`.
const ORT_ALLOCATOR_GET_STATS_VERSION: u32 = 23;

/// `OrtAllocator` as laid out by ORT 1.23: the ort-sys prefix plus `GetStats`.
#[repr(C)]
struct AllocatorV23 {
    base: ort_sys::OrtAllocator,
    get_stats: Option<
        unsafe extern "system" fn(
            this_: *const ort_sys::OrtAllocator,
            out: *mut *mut ort_sys::OrtKeyValuePairs,
        ) -> ort_sys::OrtStatusPtr,
    >,
}

/// Statistics of one ORT allocator (bytes unless noted). Keys follow the
/// header documentation; a key an allocator does not provide stays 0.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct AllocatorStats {
    pub(crate) in_use: usize,
    pub(crate) total_allocated: usize,
    pub(crate) max_in_use: usize,
    pub(crate) num_allocs: usize,
    pub(crate) num_reserves: usize,
    pub(crate) num_arena_extensions: usize,
}

impl AllocatorStats {
    /// Parse ORT's key/value statistics. `None` when the allocator reported
    /// nothing we recognise (a non-arena allocator returns an empty set).
    pub(crate) fn from_pairs(pairs: &[(String, String)]) -> Option<Self> {
        let get = |key: &str| -> Option<usize> {
            pairs
                .iter()
                .find(|(name, _)| name == key)
                .and_then(|(_, value)| value.trim().parse::<i64>().ok())
                .map(|value| value.max(0) as usize)
        };
        let total_allocated = get("TotalAllocated")?;
        Some(Self {
            in_use: get("InUse").unwrap_or(0),
            total_allocated,
            max_in_use: get("MaxInUse").unwrap_or(0),
            num_allocs: get("NumAllocs").unwrap_or(0),
            num_reserves: get("NumReserves").unwrap_or(0),
            num_arena_extensions: get("NumArenaExtensions").unwrap_or(0),
        })
    }
}

/// Raw key/value statistics of an ORT allocator, `None` if it has none.
///
/// # Safety
/// `allocator` must point at a live `OrtAllocator` created by ORT.
unsafe fn allocator_stat_pairs(
    allocator: *mut ort_sys::OrtAllocator,
) -> Option<Vec<(String, String)>> {
    if allocator.is_null() || unsafe { (*allocator).version } < ORT_ALLOCATOR_GET_STATS_VERSION {
        return None;
    }
    let get_stats = unsafe { (*(allocator as *const AllocatorV23)).get_stats }?;
    let mut kvps: *mut ort_sys::OrtKeyValuePairs = std::ptr::null_mut();
    let status = unsafe { get_stats(allocator, &mut kvps) };
    if !status.0.is_null() {
        unsafe { (ort::api().ReleaseStatus)(status.0) };
        return None;
    }
    if kvps.is_null() {
        return None;
    }
    let mut keys: *const *const c_char = std::ptr::null();
    let mut values: *const *const c_char = std::ptr::null();
    let mut count = 0usize;
    unsafe { (ort::api().GetKeyValuePairs)(kvps, &mut keys, &mut values, &mut count) };
    let mut pairs = Vec::with_capacity(count);
    for index in 0..count {
        let (key, value) = unsafe { (*keys.add(index), *values.add(index)) };
        if key.is_null() || value.is_null() {
            continue;
        }
        pairs.push((
            unsafe { CStr::from_ptr(key) }
                .to_string_lossy()
                .into_owned(),
            unsafe { CStr::from_ptr(value) }
                .to_string_lossy()
                .into_owned(),
        ));
    }
    unsafe { (ort::api().ReleaseKeyValuePairs)(kvps) };
    Some(pairs)
}

/// Statistics of the arena a session uses for `device` / `device_id`.
///
/// Uses the session's own allocator (`CreateAllocator(session, mem_info)`),
/// never the process-default one. `None` when the session has no arena for
/// that device, the arena is disabled, or the allocator does not expose stats.
pub(crate) fn session_allocator_stats(
    session: &Session,
    device: AllocationDevice,
    device_id: i32,
) -> Option<AllocatorStats> {
    let info =
        MemoryInfo::new(device, device_id, AllocatorType::Arena, MemoryType::Default).ok()?;
    let allocator = Allocator::new(session, info).ok()?;
    let pairs = unsafe { allocator_stat_pairs(allocator.ptr() as *mut _) }?;
    AllocatorStats::from_pairs(&pairs)
}

/// Resident set size of this process, the same quantity the engine uses for
/// its host load reconciliation (`sysinfo` `Process::memory`).
pub(crate) fn process_resident_bytes() -> Option<usize> {
    #[cfg(target_os = "macos")]
    {
        let mut info: libc::proc_taskinfo = unsafe { std::mem::zeroed() };
        let size = std::mem::size_of::<libc::proc_taskinfo>() as libc::c_int;
        let written = unsafe {
            libc::proc_pidinfo(
                std::process::id() as libc::c_int,
                libc::PROC_PIDTASKINFO,
                0,
                (&mut info as *mut libc::proc_taskinfo).cast(),
                size,
            )
        };
        (written == size).then_some(info.pti_resident_size as usize)
    }
    #[cfg(target_os = "linux")]
    {
        let statm = std::fs::read_to_string("/proc/self/statm").ok()?;
        let pages: usize = statm.split_whitespace().nth(1)?.parse().ok()?;
        let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
        (page > 0).then(|| pages * page as usize)
    }
    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    {
        None
    }
}

// ---------------------------------------------------------------------------
// Per-session measurements
// ---------------------------------------------------------------------------

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use kapsl_engine_api::{
    MemoryAllocation, MemoryAllocationClass, MemoryAllocationSource, MemoryDomain, MemoryReport,
};

/// A session's arena grows front-loaded (the first runs size it), so the first
/// few runs always refresh; after that a refresh happens at most this often.
const REFRESH_INTERVAL: Duration = Duration::from_millis(250);
const ALWAYS_REFRESH_FIRST_RUNS: u32 = 4;

/// What the meter reads for every session of one backend.
#[derive(Debug, Clone, Copy)]
pub(crate) struct MeterConfig {
    /// Device arena to read besides the CPU arena (CUDA only), as
    /// `(device, CUDA ordinal)`.
    pub(crate) device_arena: Option<(AllocationDevice, i32)>,
    /// Use the process resident-size delta around session creation as the
    /// session's host footprint. Only meaningful when the model's memory lives
    /// in the host (CPU, unified memory): with CUDA the delta is dominated by
    /// the CUDA/cuDNN libraries that get mapped on first use.
    pub(crate) use_creation_rss: bool,
}

/// Overlap detector for session creations. A resident-size delta is only
/// attributable to one session when no other session was created meanwhile.
#[derive(Debug, Default)]
pub(crate) struct CreationTracker {
    active: AtomicUsize,
    started: AtomicU64,
}

/// An open creation window; `finish` says whether it ran alone.
pub(crate) struct CreationWindow<'a> {
    tracker: &'a CreationTracker,
    started_at: u64,
    alone_at_start: bool,
}

impl CreationTracker {
    pub(crate) fn begin(&self) -> CreationWindow<'_> {
        let alone_at_start = self.active.fetch_add(1, Ordering::SeqCst) == 0;
        let started_at = self.started.fetch_add(1, Ordering::SeqCst) + 1;
        CreationWindow {
            tracker: self,
            started_at,
            alone_at_start,
        }
    }
}

impl CreationWindow<'_> {
    /// Close the window. True when no other creation overlapped it.
    pub(crate) fn finish(self) -> bool {
        let alone =
            self.alone_at_start && self.tracker.started.load(Ordering::SeqCst) == self.started_at;
        self.tracker.active.fetch_sub(1, Ordering::SeqCst);
        alone
    }
}

/// Process-wide: creations in different backends in one process overlap too.
pub(crate) fn creation_tracker() -> &'static CreationTracker {
    static TRACKER: CreationTracker = CreationTracker {
        active: AtomicUsize::new(0),
        started: AtomicU64::new(0),
    };
    &TRACKER
}

#[derive(Debug, Clone)]
struct SessionRecord {
    /// Resident-size growth while this session was created; `None` when it
    /// was not measured, not attributable (overlapping creation) or negative.
    creation_rss_delta: Option<usize>,
    cpu_at_load: Option<AllocatorStats>,
    cpu_now: Option<AllocatorStats>,
    device_at_load: Option<AllocatorStats>,
    device_now: Option<AllocatorStats>,
    /// Resident-size growth of the session's first run, kept only for sessions
    /// whose CPU allocator exposes no arena statistics and only when no other
    /// session was created or run meanwhile.
    first_run_rss_delta: Option<usize>,
    runs: u32,
    last_refresh: Instant,
}

/// Sums over every measured session of one backend.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct MeterTotals {
    pub(crate) sessions: usize,
    /// Host bytes the sessions held right after creation (weights, graph,
    /// provider state). Per session: the larger of the creation resident delta
    /// and the CPU arena total; when no clean resident delta exists, the
    /// larger of the model file size and the arena total (the plan's figure is
    /// the floor, never replaced by a smaller guess).
    pub(crate) host_at_load: usize,
    /// How many of those sessions fell back to the file-size floor.
    pub(crate) host_floor_sessions: usize,
    /// CPU arena growth since creation (activations).
    pub(crate) host_arena_growth: usize,
    /// Sessions whose CPU allocator exposes arena statistics. A session
    /// without them (the arena is disabled, or ORT was given an arena-less CPU
    /// provider, as the CoreML path does) cannot have its activation memory
    /// measured.
    pub(crate) host_arena_sessions: usize,
    /// Activation memory of the sessions: arena growth where the arena is
    /// measured; otherwise the clean first-run resident delta, else the
    /// caller's fallback per session that ran. Before any session has run the
    /// caller's fallback is charged once for the whole backend.
    pub(crate) host_workspace: usize,
    pub(crate) host_arena_in_use: usize,
    pub(crate) host_arena_max_in_use: usize,
    /// Device arena bytes at creation / growth since. `None` when no session
    /// exposed device arena statistics.
    pub(crate) device_at_load: Option<usize>,
    pub(crate) device_growth: Option<usize>,
}

/// Collects per-session measurements for one `OnnxBackend`.
#[derive(Debug)]
pub(crate) struct MemoryMeter {
    config: MeterConfig,
    records: Mutex<HashMap<usize, SessionRecord>>,
}

fn session_key(session: &Session) -> usize {
    session.ptr() as usize
}

impl MemoryMeter {
    pub(crate) fn new(config: MeterConfig) -> Self {
        Self {
            config,
            records: Mutex::new(HashMap::new()),
        }
    }

    fn read_arenas(&self, session: &Session) -> (Option<AllocatorStats>, Option<AllocatorStats>) {
        let cpu = session_allocator_stats(session, AllocationDevice::CPU, 0);
        let device = self
            .config
            .device_arena
            .and_then(|(device, id)| session_allocator_stats(session, device, id));
        (cpu, device)
    }

    /// Record a freshly created session. `rss_before`/`rss_after` bracket its
    /// creation; `alone` says no other creation overlapped.
    pub(crate) fn register_creation(
        &self,
        session: &Session,
        rss_before: Option<usize>,
        rss_after: Option<usize>,
        alone: bool,
    ) {
        let (cpu, device) = self.read_arenas(session);
        let creation_rss_delta = if self.config.use_creation_rss && alone {
            rss_before
                .zip(rss_after)
                .and_then(|(before, after)| after.checked_sub(before))
        } else {
            None
        };
        let record = SessionRecord {
            creation_rss_delta,
            cpu_at_load: cpu,
            cpu_now: cpu,
            device_at_load: device,
            device_now: device,
            first_run_rss_delta: None,
            runs: 0,
            last_refresh: Instant::now(),
        };
        if let Ok(mut records) = self.records.lock() {
            records.insert(session_key(session), record);
        }
    }

    /// True when the next run of `session` should be bracketed with resident
    /// size reads: it is the session's first run and its activations cannot be
    /// read from an arena.
    pub(crate) fn wants_first_run_rss(&self, session: &Session) -> bool {
        self.records.lock().is_ok_and(|records| {
            records.get(&session_key(session)).is_some_and(|r| {
                r.runs == 0 && r.cpu_at_load.is_none() && self.config.use_creation_rss
            })
        })
    }

    pub(crate) fn record_first_run(
        &self,
        session: &Session,
        rss_before: Option<usize>,
        rss_after: Option<usize>,
        alone: bool,
    ) {
        if !alone {
            return;
        }
        let delta = rss_before
            .zip(rss_after)
            .and_then(|(b, a)| a.checked_sub(b));
        if let Ok(mut records) = self.records.lock() {
            if let Some(record) = records.get_mut(&session_key(session)) {
                record.first_run_rss_delta = delta;
            }
        }
    }

    /// Re-read a session's arenas after it ran. Throttled except for the first
    /// runs; `force` skips the throttle.
    pub(crate) fn refresh(&self, session: &Session, force: bool) {
        let key = session_key(session);
        {
            let Ok(mut records) = self.records.lock() else {
                return;
            };
            let Some(record) = records.get_mut(&key) else {
                return;
            };
            if !force {
                record.runs = record.runs.saturating_add(1);
            }
            if !force
                && record.runs > ALWAYS_REFRESH_FIRST_RUNS
                && record.last_refresh.elapsed() < REFRESH_INTERVAL
            {
                return;
            }
        }
        // Statistics are read outside the lock: they call into ORT.
        let (cpu, device) = self.read_arenas(session);
        if let Ok(mut records) = self.records.lock() {
            if let Some(record) = records.get_mut(&key) {
                record.cpu_now = cpu.or(record.cpu_now);
                record.device_now = device.or(record.device_now);
                record.last_refresh = Instant::now();
            }
        }
    }

    /// Forget a session that was dropped or discarded.
    pub(crate) fn forget(&self, session: &Session) {
        if let Ok(mut records) = self.records.lock() {
            records.remove(&session_key(session));
        }
    }

    pub(crate) fn totals(&self, model_bytes: usize, fallback_workspace: usize) -> MeterTotals {
        let Ok(records) = self.records.lock() else {
            return MeterTotals::default();
        };
        let mut totals = MeterTotals::default();
        let (mut device_load, mut device_growth, mut any_device) = (0usize, 0usize, false);
        for record in records.values() {
            totals.sessions += 1;
            let arena = record.cpu_at_load.map_or(0, |stats| stats.total_allocated);
            let host = match record.creation_rss_delta {
                Some(delta) => delta.max(arena),
                None => {
                    totals.host_floor_sessions += 1;
                    model_bytes.max(arena)
                }
            };
            totals.host_at_load = totals.host_at_load.saturating_add(host);
            let growth = match (record.cpu_at_load, record.cpu_now) {
                (Some(load), Some(now)) => {
                    let growth = now.total_allocated.saturating_sub(load.total_allocated);
                    totals.host_arena_growth = totals.host_arena_growth.saturating_add(growth);
                    growth
                }
                // No arena to read: the first run's resident delta when it was
                // clean, else the planned fallback. A session that never ran
                // has not allocated activations yet.
                _ if record.runs == 0 => 0,
                _ => record.first_run_rss_delta.unwrap_or(fallback_workspace),
            };
            totals.host_workspace = totals.host_workspace.saturating_add(growth);
            if let Some(now) = record.cpu_now {
                totals.host_arena_sessions += 1;
                totals.host_arena_in_use = totals.host_arena_in_use.saturating_add(now.in_use);
                totals.host_arena_max_in_use =
                    totals.host_arena_max_in_use.saturating_add(now.max_in_use);
            }
            if let (Some(load), Some(now)) = (record.device_at_load, record.device_now) {
                any_device = true;
                device_load = device_load.saturating_add(load.total_allocated);
                device_growth = device_growth
                    .saturating_add(now.total_allocated.saturating_sub(load.total_allocated));
            }
        }
        if !records.values().any(|record| record.runs > 0) {
            // Nothing has run, so no arena has been sized and a measured zero
            // would be "unknown", not "none". Charge the fallback once for the
            // backend (not per prewarmed session) until the first run.
            totals.host_workspace = fallback_workspace;
        }
        if any_device {
            totals.device_at_load = Some(device_load);
            totals.device_growth = Some(device_growth);
        }
        totals
    }
}

// ---------------------------------------------------------------------------
// Placement and report building (pure)
// ---------------------------------------------------------------------------

/// Where an execution provider's model memory physically lives.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Placement {
    /// CPU, or a unified-memory device (Apple silicon CoreML).
    Host,
    /// A dedicated NVIDIA device, keyed by the engine's device id. `pooled`
    /// when the Kapsl pool allocator serves the session's device memory.
    Cuda { device_id: usize, pooled: bool },
    /// A provider whose memory ORT cannot report and the engine has no pool
    /// for: the plan is kept for its device rows.
    Provider { name: String, device_id: usize },
    /// The host or device cannot be told apart (CoreML off Apple silicon).
    Unattributed,
}

pub(crate) const UNATTRIBUTED_PROVIDER: &str = "unattributed";

pub(crate) fn unattributed_domain() -> MemoryDomain {
    MemoryDomain::Provider {
        provider: UNATTRIBUTED_PROVIDER.to_string(),
        device_id: None,
    }
}

/// Memory the Kapsl pool attributes to this model replica (CUDA, pooled).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct PoolUsage {
    pub(crate) weights: usize,
    pub(crate) workspace: usize,
}

fn row(
    id: String,
    domain: MemoryDomain,
    class: MemoryAllocationClass,
    source: MemoryAllocationSource,
    bytes: usize,
) -> MemoryAllocation {
    MemoryAllocation {
        allocation_id: id,
        domain,
        class,
        source,
        bytes,
    }
}

/// Build the post-load report from measurements.
///
/// `plan` is the backend's own plan: it supplies the rows that cannot be
/// measured (a provider ORT does not expose) and the fallback when no session
/// is registered. Allocation ids are the planned ones, so a measured row
/// replaces its planned row one for one.
pub(crate) fn measured_report(
    allocation_id: &str,
    placement: &Placement,
    totals: &MeterTotals,
    pool: Option<PoolUsage>,
    plan: &MemoryReport,
) -> MemoryReport {
    if totals.sessions == 0 {
        return plan.clone();
    }
    let host_session_id = format!("{allocation_id}:host-session");
    let workspace_id = format!("{allocation_id}:workspace");
    let backend = MemoryAllocationSource::BackendManaged;
    let host_at_load = totals.host_at_load;
    let host_workspace = totals.host_workspace;
    // The engine selects the device's provider domain (`metal` for CoreML) and
    // adds its own weights estimate there when no weights row exists. A host
    // report migrates that estimate only if it carries a row of the same
    // class, so host placements always report a zero-byte weights row (the
    // weights are inside the host session row).
    let weights_marker = |report: &mut MemoryReport, domain: MemoryDomain| {
        report.push(row(
            allocation_id.to_string(),
            domain,
            MemoryAllocationClass::PersistentWeights,
            backend,
            0,
        ));
    };
    let mut report = MemoryReport::default();
    let host_rows = |report: &mut MemoryReport, domain: MemoryDomain, workspace_id: String| {
        report.push(row(
            host_session_id.clone(),
            domain.clone(),
            MemoryAllocationClass::ModelSession,
            backend,
            host_at_load,
        ));
        report.push(row(
            workspace_id,
            domain,
            MemoryAllocationClass::TransientWorkspace,
            backend,
            host_workspace,
        ));
    };
    match placement {
        Placement::Host => {
            host_rows(&mut report, MemoryDomain::Host, workspace_id);
            weights_marker(&mut report, MemoryDomain::Host);
        }
        Placement::Unattributed => {
            host_rows(&mut report, unattributed_domain(), workspace_id);
            weights_marker(&mut report, unattributed_domain());
        }
        Placement::Cuda { device_id, pooled } => {
            // Host side: the creation resident-size delta is not used for CUDA
            // (see `MeterConfig::use_creation_rss`), so every session falls
            // back to the file-size floor: this row is the plan, not a
            // measurement.
            host_rows(
                &mut report,
                MemoryDomain::Host,
                format!("{allocation_id}:host-workspace"),
            );
            let domain = MemoryDomain::Cuda {
                device_id: *device_id,
            };
            let planned = |class| {
                plan.allocations
                    .iter()
                    .find(|a| a.domain == domain && a.class == class)
                    .map_or(0, |a| a.bytes)
            };
            let planned_weights = planned(MemoryAllocationClass::PersistentWeights);
            let planned_workspace = planned(MemoryAllocationClass::TransientWorkspace);
            let (measured_weights, measured_workspace, source) = if *pooled {
                let usage = pool.unwrap_or_default();
                (
                    usage.weights,
                    usage.workspace,
                    MemoryAllocationSource::RuntimeManaged,
                )
            } else {
                (
                    totals.device_at_load.unwrap_or(0),
                    totals.device_growth.unwrap_or(0),
                    backend,
                )
            };
            // Floor at the plan until a CUDA run confirms the sources: ORT keeps
            // initializers mostly outside its arena on CPU (MobileNetV2: 6 MB
            // arena for a 14 MB file), pool allocations made on ORT threads
            // without an owner scope are attributed to nobody, and ORT may
            // keep a BFC arena beside the pool. A measurement below the plan
            // is therefore not trusted to be complete.
            let weights = measured_weights.max(planned_weights);
            let workspace = measured_workspace.max(planned_workspace);
            report.push(row(
                allocation_id.to_string(),
                domain.clone(),
                MemoryAllocationClass::PersistentWeights,
                source,
                weights,
            ));
            report.push(row(
                workspace_id,
                domain,
                MemoryAllocationClass::TransientWorkspace,
                source,
                workspace,
            ));
        }
        Placement::Provider { .. } => {
            host_rows(
                &mut report,
                MemoryDomain::Host,
                format!("{allocation_id}:host-workspace"),
            );
            // Device rows of an unmeasurable provider stay as planned.
            report.allocations.extend(
                plan.allocations
                    .iter()
                    .filter(|a| !matches!(a.domain, MemoryDomain::Host))
                    .cloned(),
            );
        }
    }
    report
}

#[cfg(test)]
mod tests {
    use super::*;
    use kapsl_engine_api::MemoryAllocationClass as Class;

    const MIB: usize = 1024 * 1024;

    fn stats(total: usize, in_use: usize) -> AllocatorStats {
        AllocatorStats {
            in_use,
            total_allocated: total,
            max_in_use: in_use,
            ..AllocatorStats::default()
        }
    }

    fn record(
        rss: Option<usize>,
        cpu: Option<(usize, usize)>,
        device: Option<(usize, usize)>,
    ) -> SessionRecord {
        SessionRecord {
            creation_rss_delta: rss,
            cpu_at_load: cpu.map(|(load, _)| stats(load, 0)),
            cpu_now: cpu.map(|(_, now)| stats(now, 0)),
            device_at_load: device.map(|(load, _)| stats(load, 0)),
            device_now: device.map(|(_, now)| stats(now, 0)),
            first_run_rss_delta: None,
            runs: 1,
            last_refresh: Instant::now(),
        }
    }

    fn meter_with(records: Vec<SessionRecord>) -> MemoryMeter {
        let meter = MemoryMeter::new(MeterConfig {
            device_arena: None,
            use_creation_rss: true,
        });
        let mut map = meter.records.lock().unwrap();
        for (index, record) in records.into_iter().enumerate() {
            map.insert(index, record);
        }
        drop(map);
        meter
    }

    fn plan_for(id: &str, domain: MemoryDomain, bytes: usize) -> MemoryReport {
        let mut plan = MemoryReport::single(
            format!("{id}:host-session"),
            MemoryDomain::Host,
            Class::ModelSession,
            bytes,
        );
        if domain != MemoryDomain::Host {
            plan.push(row(
                id.to_string(),
                domain.clone(),
                Class::PersistentWeights,
                MemoryAllocationSource::BackendManaged,
                bytes,
            ));
            plan.push(row(
                format!("{id}:workspace"),
                domain,
                Class::TransientWorkspace,
                MemoryAllocationSource::BackendManaged,
                (bytes / 4).max(64 * MIB),
            ));
        }
        plan
    }

    fn bytes_of(report: &MemoryReport, domain: &MemoryDomain, class: Class) -> Option<usize> {
        report
            .allocations
            .iter()
            .find(|a| &a.domain == domain && a.class == class)
            .map(|a| a.bytes)
    }

    #[test]
    fn totals_sum_sessions_and_take_the_larger_host_figure() {
        let meter = meter_with(vec![
            // rss delta larger than the arena: the delta wins.
            record(Some(30 * MIB), Some((MIB, 16 * MIB)), None),
            // arena larger than a (clean) tiny delta: the arena wins.
            record(Some(MIB), Some((6 * MIB, 20 * MIB)), None),
        ]);
        let totals = meter.totals(MIB, 64 * MIB);
        assert_eq!(totals.sessions, 2);
        assert_eq!(totals.host_at_load, 30 * MIB + 6 * MIB);
        assert_eq!(totals.host_arena_growth, 15 * MIB + 14 * MIB);
        assert_eq!(totals.host_floor_sessions, 0);
        assert_eq!(totals.device_at_load, None);
    }

    #[test]
    fn a_session_without_a_clean_resident_delta_is_floored_at_the_file_size() {
        let meter = meter_with(vec![
            record(None, None, None),
            record(Some(4 * MIB), None, None),
        ]);
        let totals = meter.totals(10 * MIB, 64 * MIB);
        assert_eq!(totals.host_floor_sessions, 1);
        assert_eq!(totals.host_at_load, 10 * MIB + 4 * MIB);
    }

    #[test]
    fn device_arena_totals_are_none_until_a_session_exposes_them() {
        let meter = meter_with(vec![
            record(None, Some((MIB, MIB)), Some((100 * MIB, 130 * MIB))),
            record(None, Some((MIB, MIB)), Some((100 * MIB, 100 * MIB))),
        ]);
        let totals = meter.totals(MIB, 64 * MIB);
        assert_eq!(totals.device_at_load, Some(200 * MIB));
        assert_eq!(totals.device_growth, Some(30 * MIB));
    }

    #[test]
    fn creation_windows_detect_overlap() {
        let tracker = CreationTracker::default();
        let solo = tracker.begin();
        assert!(solo.finish(), "a lone creation is attributable");

        let first = tracker.begin();
        let second = tracker.begin();
        assert!(!second.finish(), "started while another was active");
        assert!(!first.finish(), "another one started during it");

        let after = tracker.begin();
        assert!(
            after.finish(),
            "earlier overlaps do not taint a later window"
        );
    }

    #[test]
    fn a_window_that_ends_before_another_starts_stays_clean() {
        let tracker = CreationTracker::default();
        let first = tracker.begin();
        assert!(first.finish());
        let second = tracker.begin();
        assert!(second.finish());
    }

    #[test]
    fn host_placement_reports_session_and_workspace_rows_with_planned_ids() {
        let meter = meter_with(vec![record(
            Some(54 * MIB),
            Some((6 * MIB, 20 * MIB)),
            None,
        )]);
        let plan = plan_for("onnx-session:0:1", MemoryDomain::Host, 14 * MIB);
        let report = measured_report(
            "onnx-session:0:1",
            &Placement::Host,
            &meter.totals(14 * MIB, 64 * MIB),
            None,
            &plan,
        );
        assert_eq!(
            bytes_of(&report, &MemoryDomain::Host, Class::ModelSession),
            Some(54 * MIB)
        );
        assert_eq!(
            bytes_of(&report, &MemoryDomain::Host, Class::TransientWorkspace),
            Some(14 * MIB)
        );
        let ids: Vec<_> = report
            .allocations
            .iter()
            .map(|a| a.allocation_id.as_str())
            .collect();
        assert_eq!(
            ids,
            [
                "onnx-session:0:1:host-session",
                "onnx-session:0:1:workspace",
                "onnx-session:0:1"
            ]
        );
        assert!(report
            .allocations
            .iter()
            .all(|a| a.domain == MemoryDomain::Host));
    }

    #[test]
    fn a_workspace_row_is_reported_even_when_the_arena_has_not_grown() {
        // The engine can only shrink a fallback workspace claim if the report
        // carries a workspace row, so a zero must still be a row.
        let meter = meter_with(vec![record(Some(MIB), Some((MIB, MIB)), None)]);
        let report = measured_report(
            "a",
            &Placement::Host,
            &meter.totals(MIB, 64 * MIB),
            None,
            &plan_for("a", MemoryDomain::Host, MIB),
        );
        assert_eq!(
            bytes_of(&report, &MemoryDomain::Host, Class::TransientWorkspace),
            Some(0)
        );
    }

    #[test]
    fn sessions_without_a_clean_delta_report_at_least_the_file_size_each() {
        let meter = meter_with(vec![record(None, None, None), record(None, None, None)]);
        let report = measured_report(
            "a",
            &Placement::Host,
            &meter.totals(10 * MIB, 64 * MIB),
            None,
            &plan_for("a", MemoryDomain::Host, 20 * MIB),
        );
        assert_eq!(
            bytes_of(&report, &MemoryDomain::Host, Class::ModelSession),
            Some(20 * MIB)
        );
    }

    #[test]
    fn sessions_without_arena_statistics_are_charged_the_fallback_workspace() {
        // CoreML registers an arena-less CPU provider: activations of the CPU
        // fallback nodes cannot be measured, so each such session keeps the
        // planned workspace instead of reporting a false zero.
        let meter = meter_with(vec![
            record(Some(60 * MIB), None, None),
            record(Some(60 * MIB), Some((MIB, 5 * MIB)), None),
        ]);
        let totals = meter.totals(10 * MIB, 64 * MIB);
        assert_eq!(totals.host_arena_sessions, 1);
        let report = measured_report(
            "a",
            &Placement::Host,
            &totals,
            None,
            &plan_for("a", MemoryDomain::Host, 10 * MIB),
        );
        assert_eq!(
            bytes_of(&report, &MemoryDomain::Host, Class::TransientWorkspace),
            Some(4 * MIB + 64 * MIB),
            "one measured growth plus one fallback"
        );
    }

    #[test]
    fn before_any_run_the_fallback_workspace_is_charged_once_for_the_backend() {
        let mut a = record(Some(10 * MIB), Some((MIB, MIB)), None);
        let mut b = record(Some(10 * MIB), Some((MIB, MIB)), None);
        a.runs = 0;
        b.runs = 0;
        let meter = meter_with(vec![a, b]);
        let totals = meter.totals(MIB, 64 * MIB);
        assert_eq!(totals.sessions, 2);
        assert_eq!(
            totals.host_workspace,
            64 * MIB,
            "once, not once per prewarmed session"
        );
        assert_eq!(totals.host_arena_growth, 0);
    }

    #[test]
    fn once_a_session_has_run_idle_sessions_report_their_measured_growth() {
        let mut idle = record(Some(10 * MIB), Some((MIB, MIB)), None);
        idle.runs = 0;
        let ran = record(Some(10 * MIB), Some((MIB, 15 * MIB)), None);
        let meter = meter_with(vec![idle, ran]);
        assert_eq!(meter.totals(MIB, 64 * MIB).host_workspace, 14 * MIB);
    }

    #[test]
    fn an_arenaless_session_reports_its_clean_first_run_delta_as_workspace() {
        let mut clean = record(Some(60 * MIB), None, None);
        clean.first_run_rss_delta = Some(5 * MIB);
        let meter = meter_with(vec![clean]);
        let totals = meter.totals(10 * MIB, 64 * MIB);
        assert_eq!(totals.host_workspace, 5 * MIB);
    }

    #[test]
    fn no_sessions_means_the_plan_is_all_there_is() {
        let plan = plan_for("a", MemoryDomain::Host, 5 * MIB);
        let report = measured_report("a", &Placement::Host, &MeterTotals::default(), None, &plan);
        assert_eq!(report, plan);
    }

    #[test]
    fn unattributed_placement_never_lands_in_host_or_a_device() {
        let meter = meter_with(vec![record(Some(10 * MIB), Some((MIB, 2 * MIB)), None)]);
        let report = measured_report(
            "a",
            &Placement::Unattributed,
            &meter.totals(MIB, 64 * MIB),
            None,
            &plan_for("a", unattributed_domain(), MIB),
        );
        assert!(report
            .allocations
            .iter()
            .all(|a| a.domain == unattributed_domain()));
        assert_eq!(
            bytes_of(&report, &unattributed_domain(), Class::ModelSession),
            Some(10 * MIB)
        );
    }

    #[test]
    fn cuda_without_the_pool_uses_device_arena_statistics() {
        let device = MemoryDomain::Cuda { device_id: 1 };
        let meter = meter_with(vec![record(
            Some(500 * MIB),
            Some((2 * MIB, 5 * MIB)),
            Some((120 * MIB, 150 * MIB)),
        )]);
        let placement = Placement::Cuda {
            device_id: 1,
            pooled: false,
        };
        let report = measured_report(
            "a",
            &placement,
            &meter.totals(100 * MIB, 64 * MIB),
            None,
            &plan_for("a", device.clone(), 100 * MIB),
        );
        assert_eq!(
            bytes_of(&report, &device, Class::PersistentWeights),
            Some(120 * MIB)
        );
        // Measured growth (30 MiB) is below the 64 MiB plan: floored.
        assert_eq!(
            bytes_of(&report, &device, Class::TransientWorkspace),
            Some(64 * MIB)
        );
        // Host side: the 500 MiB creation delta is CUDA library mapping noise
        // and is not used for CUDA; the CPU arena growth is.
        assert_eq!(
            bytes_of(&report, &MemoryDomain::Host, Class::TransientWorkspace),
            Some(3 * MIB)
        );
        assert!(report
            .allocations
            .iter()
            .filter(|a| a.domain == device)
            .all(|a| a.source == MemoryAllocationSource::BackendManaged));
    }

    #[test]
    fn cuda_with_the_pool_reports_pool_owner_usage_as_runtime_managed() {
        let device = MemoryDomain::Cuda { device_id: 0 };
        let meter = meter_with(vec![record(None, Some((MIB, MIB)), None)]);
        let placement = Placement::Cuda {
            device_id: 0,
            pooled: true,
        };
        let pool = PoolUsage {
            weights: 90 * MIB,
            workspace: 12 * MIB,
        };
        let report = measured_report(
            "a",
            &placement,
            &meter.totals(100 * MIB, 64 * MIB),
            Some(pool),
            &plan_for("a", device.clone(), 100 * MIB),
        );
        // Pool usage below the plan (90 < 100, 12 < 64) is floored at the plan
        // until a GPU run shows the pool owner sees every allocation.
        assert_eq!(
            bytes_of(&report, &device, Class::PersistentWeights),
            Some(100 * MIB)
        );
        assert_eq!(
            bytes_of(&report, &device, Class::TransientWorkspace),
            Some(64 * MIB)
        );
        assert!(report
            .allocations
            .iter()
            .filter(|a| a.domain == device)
            .all(|a| a.source == MemoryAllocationSource::RuntimeManaged));
    }

    #[test]
    fn cuda_measurements_above_the_plan_are_reported_as_measured() {
        let device = MemoryDomain::Cuda { device_id: 0 };
        let meter = meter_with(vec![record(
            None,
            Some((MIB, MIB)),
            Some((120 * MIB, 220 * MIB)),
        )]);
        for (pooled, pool) in [
            (
                true,
                Some(PoolUsage {
                    weights: 130 * MIB,
                    workspace: 70 * MIB,
                }),
            ),
            (false, None),
        ] {
            let placement = Placement::Cuda {
                device_id: 0,
                pooled,
            };
            let report = measured_report(
                "a",
                &placement,
                &meter.totals(100 * MIB, 64 * MIB),
                pool,
                &plan_for("a", device.clone(), 100 * MIB),
            );
            let (weights, workspace) = if pooled {
                (130 * MIB, 70 * MIB)
            } else {
                (120 * MIB, 100 * MIB)
            };
            assert_eq!(
                bytes_of(&report, &device, Class::PersistentWeights),
                Some(weights),
                "pooled={pooled}"
            );
            assert_eq!(
                bytes_of(&report, &device, Class::TransientWorkspace),
                Some(workspace),
                "pooled={pooled}"
            );
        }
    }

    #[test]
    fn host_placements_report_a_zero_weights_row_so_provider_estimates_migrate() {
        let meter = meter_with(vec![record(Some(10 * MIB), Some((MIB, 2 * MIB)), None)]);
        for (placement, domain) in [
            (Placement::Host, MemoryDomain::Host),
            (Placement::Unattributed, unattributed_domain()),
        ] {
            let report = measured_report(
                "a",
                &placement,
                &meter.totals(MIB, 64 * MIB),
                None,
                &plan_for("a", domain.clone(), MIB),
            );
            assert_eq!(
                bytes_of(&report, &domain, Class::PersistentWeights),
                Some(0),
                "{placement:?}"
            );
        }
    }

    #[test]
    fn cuda_without_any_device_measurement_keeps_the_planned_device_rows() {
        let device = MemoryDomain::Cuda { device_id: 0 };
        let meter = meter_with(vec![record(None, Some((MIB, MIB)), None)]);
        for pooled in [false, true] {
            let placement = Placement::Cuda {
                device_id: 0,
                pooled,
            };
            let plan = plan_for("a", device.clone(), 100 * MIB);
            let report = measured_report(
                "a",
                &placement,
                &meter.totals(100 * MIB, 64 * MIB),
                None,
                &plan,
            );
            assert_eq!(
                bytes_of(&report, &device, Class::PersistentWeights),
                Some(100 * MIB),
                "pooled={pooled}"
            );
            assert_eq!(
                bytes_of(&report, &device, Class::TransientWorkspace),
                Some(64 * MIB),
                "pooled={pooled}"
            );
        }
    }

    #[test]
    fn unmeasurable_providers_keep_planned_device_rows_and_measure_the_host_side() {
        let provider = MemoryDomain::Provider {
            provider: "directml".into(),
            device_id: Some(0),
        };
        let meter = meter_with(vec![record(Some(7 * MIB), Some((MIB, 3 * MIB)), None)]);
        let placement = Placement::Provider {
            name: "directml".into(),
            device_id: 0,
        };
        let plan = plan_for("a", provider.clone(), 40 * MIB);
        let report = measured_report(
            "a",
            &placement,
            &meter.totals(40 * MIB, 64 * MIB),
            None,
            &plan,
        );
        assert_eq!(
            bytes_of(&report, &provider, Class::PersistentWeights),
            Some(40 * MIB)
        );
        assert_eq!(
            bytes_of(&report, &MemoryDomain::Host, Class::ModelSession),
            Some(7 * MIB)
        );
    }

    fn pairs(items: &[(&str, &str)]) -> Vec<(String, String)> {
        items
            .iter()
            .map(|(k, v)| ((*k).to_string(), (*v).to_string()))
            .collect()
    }

    #[test]
    fn parses_documented_keys() {
        let stats = AllocatorStats::from_pairs(&pairs(&[
            ("Limit", "-1"),
            ("InUse", "1024"),
            ("TotalAllocated", "4096"),
            ("MaxInUse", "2048"),
            ("NumAllocs", "7"),
            ("NumReserves", "3"),
            ("NumArenaExtensions", "2"),
        ]))
        .expect("stats");
        assert_eq!(stats.in_use, 1024);
        assert_eq!(stats.total_allocated, 4096);
        assert_eq!(stats.max_in_use, 2048);
        assert_eq!(stats.num_allocs, 7);
        assert_eq!(stats.num_reserves, 3);
        assert_eq!(stats.num_arena_extensions, 2);
    }

    #[test]
    fn empty_or_unrecognised_stats_are_not_measurements() {
        assert_eq!(AllocatorStats::from_pairs(&[]), None);
        assert_eq!(
            AllocatorStats::from_pairs(&pairs(&[("InUse", "5")])),
            None,
            "without TotalAllocated the set is not an arena report"
        );
    }

    #[test]
    fn negative_and_garbage_values_do_not_wrap() {
        let stats = AllocatorStats::from_pairs(&pairs(&[
            ("TotalAllocated", "100"),
            ("InUse", "-5"),
            ("MaxInUse", "not-a-number"),
        ]))
        .expect("stats");
        assert_eq!(stats.in_use, 0);
        assert_eq!(stats.max_in_use, 0);
    }

    #[test]
    fn allocator_v23_extends_the_ort_sys_prefix_by_one_pointer() {
        assert_eq!(
            std::mem::size_of::<AllocatorV23>(),
            std::mem::size_of::<ort_sys::OrtAllocator>() + std::mem::size_of::<usize>()
        );
    }

    #[test]
    fn resident_size_is_readable_on_supported_hosts() {
        if cfg!(any(target_os = "macos", target_os = "linux")) {
            assert!(process_resident_bytes().expect("rss") > 1024 * 1024);
        }
    }

    /// Manual probe: `KAPSL_PROBE_MODEL=<onnx> KAPSL_PROBE_SHAPE=1,3,224,224
    /// [KAPSL_PROBE_PROVIDER=cpu|coreml] cargo test -p kapsl-backends
    /// memory_measure::tests::probe -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn probe() {
        use ort::execution_providers::{CoreMLExecutionProvider, ExecutionProvider as _};
        let model = std::env::var("KAPSL_PROBE_MODEL").expect("KAPSL_PROBE_MODEL");
        let shape: Vec<usize> = std::env::var("KAPSL_PROBE_SHAPE")
            .expect("KAPSL_PROBE_SHAPE")
            .split(',')
            .map(|d| d.parse().unwrap())
            .collect();
        let provider = std::env::var("KAPSL_PROBE_PROVIDER").unwrap_or_else(|_| "cpu".into());
        let rss0 = process_resident_bytes().unwrap();
        let mut builder = Session::builder().unwrap();
        if provider == "coreml" {
            assert!(CoreMLExecutionProvider::default().is_available().unwrap());
            builder = builder
                .with_execution_providers([CoreMLExecutionProvider::default().build()])
                .unwrap();
        }
        let mut session = builder.commit_from_file(&model).unwrap();
        let rss1 = process_resident_bytes().unwrap();
        let show = |label: &str, session: &Session| {
            let cpu = session_allocator_stats(session, AllocationDevice::CPU, 0);
            println!(
                "PROBE {label}: rss={} cpu_arena={cpu:?}",
                process_resident_bytes().unwrap()
            );
        };
        println!(
            "PROBE file={} rss_delta_load={}",
            std::fs::metadata(&model).unwrap().len(),
            rss1 - rss0
        );
        show("after-load", &session);
        let len: usize = shape.iter().product();
        let input = ort::value::Tensor::from_array((shape.clone(), vec![0.5f32; len])).unwrap();
        let name = session.inputs()[0].name().to_string();
        for i in 0..3 {
            let _ = session.run(ort::inputs![name.as_str() => &input]).unwrap();
            show(&format!("after-run-{i}"), &session);
        }
    }
}
