//! Where GGUF memory is counted: the rule that maps llama.cpp devices to
//! memory-authority domains.
//!
//! Principle: count memory where it physically lives.
//!
//! * Dedicated memory of an NVIDIA GPU goes to that GPU's `Cuda{device_id}`
//!   domain, found by **PCI bus id** (CUDA ordinals and NVML indices differ under
//!   `CUDA_DEVICE_ORDER` and `CUDA_VISIBLE_DEVICES`).
//! * Dedicated memory of any other GPU (ROCm/HIP, Vulkan, SYCL, OpenCL, ...) goes
//!   to the accounting-only `Provider{<backend>, ordinal}` domain.
//! * CPU, accelerator and **unified-memory** devices (Apple Silicon, integrated
//!   GPUs) share system memory, so their bytes go to `Host`.
//! * A device that cannot be mapped is never guessed: its bytes go to an explicit
//!   `Provider{"unattributed"}` row and a warning names the device.
//!
//! The functions here are pure (no llama.cpp calls) so every rule is covered by
//! tests over fake device lists.

use kapsl_engine_api::{MemoryAllocationClass, MemoryDomain};
use std::collections::BTreeMap;
use std::sync::RwLock;

static ENGINE_DEVICES: RwLock<Vec<EngineDeviceInfo>> = RwLock::new(Vec::new());

/// Tell the GGUF backends which accelerators the engine inventoried, so llama.cpp
/// devices can be matched to engine devices by PCI bus id. The factory calls this
/// with the host's device list; with no list registered a lone CUDA device is
/// still mapped, anything ambiguous is reported as unattributed.
pub fn register_engine_devices(devices: Vec<EngineDeviceInfo>) {
    if let Ok(mut slot) = ENGINE_DEVICES.write() {
        *slot = devices;
    }
}

/// The devices registered with [`register_engine_devices`].
pub fn engine_devices() -> Vec<EngineDeviceInfo> {
    ENGINE_DEVICES.read().map(|d| d.clone()).unwrap_or_default()
}

/// Provider name of the explicit "could not attribute" domain.
pub const UNATTRIBUTED_PROVIDER: &str = "unattributed";

/// ggml backend device type, reduced to what placement needs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LlamaDeviceKind {
    Cpu,
    Accelerator,
    Gpu,
    IntegratedGpu,
    Other,
}

/// One llama.cpp (ggml) backend device.
#[derive(Debug, Clone)]
pub struct LlamaDeviceInfo {
    /// ggml device name (`CUDA0`, `MTL0`, `Vulkan1`, `CPU`, ...).
    pub name: String,
    /// Backend registry name (`CUDA`, `ROCm`, `Vulkan`, `MTL`, `CPU`, ...).
    pub backend: String,
    pub kind: LlamaDeviceKind,
    /// PCI bus id when the backend reports one.
    pub pci_bus_id: Option<String>,
    /// The device's memory is the host's system memory.
    pub unified_memory: bool,
}

impl LlamaDeviceInfo {
    fn is_cuda(&self) -> bool {
        self.backend.eq_ignore_ascii_case("cuda")
    }
}

/// One accelerator the engine inventoried (from `kapsl-hal`).
#[derive(Debug, Clone)]
pub struct EngineDeviceInfo {
    /// Engine device id (NVML index for CUDA).
    pub id: usize,
    /// Backend label (`cuda`, `rocm`, `metal`, ...).
    pub backend: String,
    pub pci_bus_id: Option<String>,
}

impl EngineDeviceInfo {
    fn is_cuda(&self) -> bool {
        self.backend.eq_ignore_ascii_case("cuda")
    }
}

/// Outcome of placing one llama device.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Placement {
    Domain(MemoryDomain),
    /// The device could not be mapped; the string says why.
    Unattributed(String),
}

impl Placement {
    pub fn domain(&self) -> MemoryDomain {
        match self {
            Self::Domain(domain) => domain.clone(),
            Self::Unattributed(_) => unattributed_domain(),
        }
    }
}

pub fn unattributed_domain() -> MemoryDomain {
    MemoryDomain::Provider {
        provider: UNATTRIBUTED_PROVIDER.to_string(),
        device_id: None,
    }
}

/// Canonical PCI bus id; see [`crate::cuda_ordinal::normalize_pci`], the one
/// implementation shared with the engine's CUDA ordinal map.
pub fn normalize_pci_bus_id(raw: &str) -> Option<String> {
    crate::cuda_ordinal::normalize_pci(raw)
}

/// Provider label for a non-CUDA discrete GPU backend.
fn provider_name(backend: &str) -> String {
    match backend.trim().to_ascii_lowercase().as_str() {
        "hip" | "rocm" => "rocm".to_string(),
        "mtl" => "metal".to_string(),
        other => other.to_string(),
    }
}

/// Engine CUDA devices whose PCI bus id equals `pci` (normalized).
fn engine_cuda_by_pci<'a>(engine: &'a [EngineDeviceInfo], pci: &str) -> Vec<&'a EngineDeviceInfo> {
    engine
        .iter()
        .filter(|device| device.is_cuda())
        .filter(|device| {
            device
                .pci_bus_id
                .as_deref()
                .and_then(normalize_pci_bus_id)
                .as_deref()
                == Some(pci)
        })
        .collect()
}

/// Place one llama device. `all` is the full llama device list (needed for
/// ordinals and "exactly one" checks); `legacy_cuda_id` is the engine device id the
/// backend was created for, used only when the engine supplied no device list.
pub fn place_device(
    index: usize,
    all: &[LlamaDeviceInfo],
    engine: &[EngineDeviceInfo],
    legacy_cuda_id: Option<usize>,
) -> Placement {
    let device = &all[index];

    match device.kind {
        LlamaDeviceKind::Cpu | LlamaDeviceKind::Accelerator => {
            return Placement::Domain(MemoryDomain::Host)
        }
        LlamaDeviceKind::IntegratedGpu => return Placement::Domain(MemoryDomain::Host),
        LlamaDeviceKind::Other => {
            return Placement::Unattributed(format!(
                "device {} has an unsupported type for memory attribution",
                device.name
            ))
        }
        LlamaDeviceKind::Gpu => {}
    }
    if device.unified_memory {
        return Placement::Domain(MemoryDomain::Host);
    }

    if device.is_cuda() {
        if let Some(pci) = device.pci_bus_id.as_deref().and_then(normalize_pci_bus_id) {
            let matches = engine_cuda_by_pci(engine, &pci);
            return match matches.as_slice() {
                [one] => Placement::Domain(MemoryDomain::Cuda { device_id: one.id }),
                [] if engine.iter().all(|d| !d.is_cuda()) => {
                    legacy_single_cuda(device, all, legacy_cuda_id)
                }
                [] => Placement::Unattributed(format!(
                    "{} (PCI {pci}) matches no engine CUDA device",
                    device.name
                )),
                many => Placement::Unattributed(format!(
                    "{} (PCI {pci}) matches {} engine devices with the same PCI bus id \
                     (for example the MIG instances of one GPU); the PCI id cannot tell them apart",
                    device.name,
                    many.len()
                )),
            };
        }
        return legacy_single_cuda(device, all, legacy_cuda_id);
    }

    // Any other discrete GPU: accounting-only provider domain, numbered as the
    // backend numbers its own devices.
    let ordinal = all[..index]
        .iter()
        .filter(|other| {
            other.kind == LlamaDeviceKind::Gpu
                && other.backend.eq_ignore_ascii_case(&device.backend)
        })
        .count();
    Placement::Domain(MemoryDomain::Provider {
        provider: provider_name(&device.backend),
        device_id: Some(ordinal),
    })
}

/// No PCI identity is available. Mapping by position is only unambiguous with a
/// single CUDA device, so that is the only case accepted.
fn legacy_single_cuda(
    device: &LlamaDeviceInfo,
    all: &[LlamaDeviceInfo],
    legacy_cuda_id: Option<usize>,
) -> Placement {
    let cuda_devices = all
        .iter()
        .filter(|d| d.is_cuda() && d.kind == LlamaDeviceKind::Gpu)
        .count();
    match (cuda_devices, legacy_cuda_id) {
        (1, Some(id)) => Placement::Domain(MemoryDomain::Cuda { device_id: id }),
        _ => Placement::Unattributed(format!(
            "{} has no PCI bus id to match against the engine's CUDA devices and {} CUDA devices are visible",
            device.name, cuda_devices
        )),
    }
}

/// Index (into `all`) of the llama CUDA device that is the engine's device
/// `engine_id`, matched by PCI bus id. Falls back to the `engine_id`-th CUDA
/// device only when identity is unavailable *and* the choice is unambiguous
/// (one CUDA device); otherwise returns an error naming the problem.
pub fn select_cuda_llama_device(
    all: &[LlamaDeviceInfo],
    engine: &[EngineDeviceInfo],
    engine_id: usize,
) -> Result<usize, String> {
    let cuda: Vec<usize> = all
        .iter()
        .enumerate()
        .filter(|(_, d)| d.is_cuda() && d.kind == LlamaDeviceKind::Gpu)
        .map(|(i, _)| i)
        .collect();
    if cuda.is_empty() {
        return Err(format!(
            "llama.cpp CUDA device {engine_id} is not available"
        ));
    }
    let engine_pci = engine
        .iter()
        .find(|d| d.is_cuda() && d.id == engine_id)
        .and_then(|d| d.pci_bus_id.as_deref())
        .and_then(normalize_pci_bus_id);
    if let Some(pci) = engine_pci {
        let hits: Vec<usize> = cuda
            .iter()
            .copied()
            .filter(|&i| {
                all[i]
                    .pci_bus_id
                    .as_deref()
                    .and_then(normalize_pci_bus_id)
                    .as_deref()
                    == Some(pci.as_str())
            })
            .collect();
        return match hits.as_slice() {
            [one] => Ok(*one),
            [] => Err(format!(
                "engine CUDA device {engine_id} (PCI {pci}) is not visible to llama.cpp \
                 (check CUDA_VISIBLE_DEVICES)"
            )),
            _ => Err(format!(
                "engine CUDA device {engine_id} (PCI {pci}) matches several llama.cpp devices"
            )),
        };
    }
    let engine_cuda = engine.iter().filter(|d| d.is_cuda()).count();
    // Without a PCI id the choice is unambiguous only with one GPU on each side,
    // and only for the device the engine actually has (an empty engine list is a
    // library user with a single GPU, which is device 0).
    let known_device = if engine_cuda == 0 {
        engine_id == 0
    } else {
        engine.iter().any(|d| d.is_cuda() && d.id == engine_id)
    };
    if cuda.len() == 1 && engine_cuda <= 1 && known_device {
        return Ok(cuda[0]);
    }
    let reason = if engine_cuda == 0 {
        "the engine's device inventory was never registered with the GGUF backend \
         (no engine CUDA devices are known)"
    } else if !known_device {
        "the engine has no CUDA device with that id"
    } else {
        "the engine reports no usable PCI bus id for that device"
    };
    Err(format!(
        "cannot identify engine CUDA device {engine_id} among {} llama.cpp CUDA devices: {reason}",
        cuda.len()
    ))
}

/// Every value a CUDA GGUF load hands to llama.cpp to name a device.
///
/// They live in three different numbering spaces, and the engine id is none of
/// them, so none may be filled from it directly:
/// - `llama_device`: index into ggml's backend device list
///   (`LlamaModelParams::with_devices`),
/// - `main_gpu`: index into the device list given to `with_devices`,
/// - `cuda_ordinal`: the CUDA runtime ordinal, which ggml's CUDA backend uses
///   as its own device index (`llama_kapsl_kv_pool_desc.device_id` reaches
///   `ggml_backend_cuda_buffer_from_device_ptr`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CudaLlamaDevicePlan {
    pub llama_device: usize,
    pub main_gpu: i32,
    pub cuda_ordinal: usize,
}

/// Plan the llama.cpp device values for engine CUDA device `engine_id`.
/// `cuda_ordinal` is the engine's own mapping of that device
/// (`cuda_ordinal::to_cuda_ordinal`); the PCI-matched llama device must agree
/// with it when ggml names its devices `CUDA<n>`, otherwise the two mappings
/// disagree about which GPU this is and the load is refused.
pub fn plan_cuda_llama_devices(
    all: &[LlamaDeviceInfo],
    engine: &[EngineDeviceInfo],
    engine_id: usize,
    cuda_ordinal: usize,
) -> Result<CudaLlamaDevicePlan, String> {
    let llama_device = select_cuda_llama_device(all, engine, engine_id)?;
    let name = &all[llama_device].name;
    if let Some(named) = name
        .strip_prefix("CUDA")
        .and_then(|n| n.parse::<usize>().ok())
    {
        if named != cuda_ordinal {
            return Err(format!(
                "engine CUDA device {engine_id} is CUDA ordinal {cuda_ordinal}, but its PCI-matched \
                 llama.cpp device is named {name}"
            ));
        }
    }
    Ok(CudaLlamaDevicePlan {
        llama_device,
        // The model is loaded on exactly one device, listed first.
        main_gpu: 0,
        cuda_ordinal,
    })
}

/// Measured bytes of one llama device.
#[derive(Debug, Clone, Copy, Default)]
pub struct DeviceBytes {
    pub weights: u64,
    pub kv: u64,
    pub compute: u64,
}

/// A row ready to become a `MemoryAllocation`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlacedRow {
    pub class: MemoryAllocationClass,
    pub domain: MemoryDomain,
    pub bytes: usize,
}

#[derive(Debug, Clone, Default)]
pub struct PlacementReport {
    pub rows: Vec<PlacedRow>,
    pub warnings: Vec<String>,
}

fn push_merged(
    merged: &mut BTreeMap<(String, u8), PlacedRow>,
    class: MemoryAllocationClass,
    domain: MemoryDomain,
    bytes: u64,
) {
    if bytes == 0 {
        return;
    }
    let key = (domain_label(&domain), class_order(class));
    let entry = merged.entry(key).or_insert(PlacedRow {
        class,
        domain,
        bytes: 0,
    });
    entry.bytes = entry.bytes.saturating_add(bytes as usize);
}

fn class_order(class: MemoryAllocationClass) -> u8 {
    match class {
        MemoryAllocationClass::PersistentWeights => 0,
        MemoryAllocationClass::KvCache => 1,
        MemoryAllocationClass::TransientWorkspace => 2,
        _ => 3,
    }
}

/// Domain name as the engine prints it in stats (`host`, `cuda:0`, `rocm:1`,
/// `unattributed`).
pub fn domain_label(domain: &MemoryDomain) -> String {
    match domain {
        MemoryDomain::Host => "host".to_string(),
        MemoryDomain::HostPinned {
            provider,
            device_id,
        } => match device_id {
            Some(id) => format!("host-pinned:{provider}:{id}"),
            None => format!("host-pinned:{provider}"),
        },
        MemoryDomain::HostMapped {
            provider,
            device_id,
        } => match device_id {
            Some(id) => format!("host-mapped:{provider}:{id}"),
            None => format!("host-mapped:{provider}"),
        },
        MemoryDomain::Cuda { device_id } => format!("cuda:{device_id}"),
        MemoryDomain::Provider {
            provider,
            device_id,
        } => match device_id {
            Some(id) => format!("{provider}:{id}"),
            None => provider.clone(),
        },
    }
}

/// Place measured bytes. `per_device[i]` is the bytes of `all[i]`; `host` is the
/// `(weights, kv, compute)` that live outside every listed device. Rows with the
/// same domain and class are merged (Metal and CPU both land in `Host`), zero rows
/// are dropped, and each unmappable device contributes a warning.
pub fn place_bytes(
    all: &[LlamaDeviceInfo],
    engine: &[EngineDeviceInfo],
    legacy_cuda_id: Option<usize>,
    per_device: &[(usize, DeviceBytes)],
    host: (u64, u64, u64),
) -> PlacementReport {
    let mut merged = BTreeMap::new();
    let mut warnings = Vec::new();
    for (index, bytes) in per_device {
        let placement = place_device(*index, all, engine, legacy_cuda_id);
        if let Placement::Unattributed(reason) = &placement {
            warnings.push(format!(
                "GGUF memory on {} is counted as unattributed: {reason}",
                all[*index].name
            ));
        }
        let domain = placement.domain();
        push_merged(
            &mut merged,
            MemoryAllocationClass::PersistentWeights,
            domain.clone(),
            bytes.weights,
        );
        push_merged(
            &mut merged,
            MemoryAllocationClass::KvCache,
            domain.clone(),
            bytes.kv,
        );
        push_merged(
            &mut merged,
            MemoryAllocationClass::TransientWorkspace,
            domain,
            bytes.compute,
        );
    }
    push_merged(
        &mut merged,
        MemoryAllocationClass::PersistentWeights,
        MemoryDomain::Host,
        host.0,
    );
    push_merged(
        &mut merged,
        MemoryAllocationClass::KvCache,
        MemoryDomain::Host,
        host.1,
    );
    push_merged(
        &mut merged,
        MemoryAllocationClass::TransientWorkspace,
        MemoryDomain::Host,
        host.2,
    );
    PlacementReport {
        rows: merged.into_values().collect(),
        warnings,
    }
}

/// Inputs for planning before the model is loaded.
#[derive(Debug, Clone, Copy)]
pub struct PlanInput {
    pub weights: usize,
    pub kv: usize,
    pub scratch: usize,
    pub n_layers: usize,
    /// `KAPSL_GGUF_N_GPU_LAYERS`; `None` means llama.cpp's default (all layers).
    pub n_gpu_layers: Option<u32>,
}

/// Plan rows with the same classification as the reported rows. `primary` is
/// the llama device expected to hold the offloaded layers (`None`: no GPU).
/// The offloaded fraction (`n_gpu_layers / n_layers`) of weights and KV goes to
/// the primary device's domain and the rest to `Host`.
pub fn plan_rows(
    primary: Option<usize>,
    all: &[LlamaDeviceInfo],
    engine: &[EngineDeviceInfo],
    legacy_cuda_id: Option<usize>,
    input: PlanInput,
) -> PlacementReport {
    let fraction_num = match (primary, input.n_gpu_layers) {
        (None, _) => 0,
        (Some(_), None) => input.n_layers.max(1),
        (Some(_), Some(ngl)) => (ngl as usize).min(input.n_layers.max(1)),
    };
    let fraction_den = input.n_layers.max(1);
    let split = |bytes: usize| -> (usize, usize) {
        let device = bytes.saturating_mul(fraction_num) / fraction_den;
        (device, bytes.saturating_sub(device))
    };
    let (w_dev, w_host) = split(input.weights);
    let (k_dev, k_host) = split(input.kv);
    let (s_dev, s_host) = split(input.scratch);
    // The scratch estimate is fitted to device-resident compute buffers. Layers
    // that run on the CPU get a CPU compute buffer that measured 1.10x to 1.18x
    // larger than the fitted figure (Qwen2.5-0.5B, `n_gpu_layers=0`), so the host
    // share carries 25% headroom to stay an upper bound.
    let s_host = s_host.saturating_add(s_host / 4);

    let mut per_device = Vec::new();
    if let Some(index) = primary {
        per_device.push((
            index,
            DeviceBytes {
                weights: w_dev as u64,
                kv: k_dev as u64,
                compute: s_dev as u64,
            },
        ));
    }
    place_bytes(
        all,
        engine,
        legacy_cuda_id,
        &per_device,
        (w_host as u64, k_host as u64, s_host as u64),
    )
}

/// Assign allocation ids: the first row of a class (domains ordered by name) keeps
/// the base id, additional domains of the same class get `base@domain`, so one
/// class split across several domains never shares an id.
pub fn allocation_id_for(base: &str, rows: &[PlacedRow], row: &PlacedRow) -> String {
    let mut same_class: Vec<&PlacedRow> = rows.iter().filter(|r| r.class == row.class).collect();
    same_class.sort_by_key(|r| domain_label(&r.domain));
    match same_class.first() {
        Some(first) if first.domain == row.domain => base.to_string(),
        _ => format!("{base}@{}", domain_label(&row.domain)),
    }
}

#[cfg(test)]
#[path = "placement_tests.rs"]
mod tests;
