//! Mapping between the engine's CUDA device ids and the CUDA runtime's ordinals.
//!
//! The engine inventories GPUs through NVML (`kapsl-hal`), and keys budgets,
//! pools and leases by that inventory id. The CUDA runtime numbers the same
//! GPUs differently: fastest-first by default (`CUDA_DEVICE_ORDER`), and
//! `CUDA_VISIBLE_DEVICES` filters and renumbers them. NVIDIA's NVML header says
//! "the NVML index may not correlate with other APIs, such as the CUDA device
//! index" and recommends looking devices up by PCI id or UUID, so the two are
//! joined here **by PCI bus id**, once, for every visible CUDA device.
//!
//! The map is pure (it takes the PCI id of each CUDA ordinal as input), so every
//! case is covered by tests over fake inventories. The engine builds it at
//! startup, [`install`]s it, and every place that hands an engine id to the CUDA
//! runtime converts with [`to_cuda_ordinal`]. With no map installed, or when
//! neither side has PCI ids and the counts agree, the mapping is the identity,
//! which is the behavior single-GPU hosts and containers have always had.

use std::sync::{Arc, RwLock};

/// One CUDA device in the engine's inventory.
#[derive(Debug, Clone)]
pub struct EngineCudaDevice {
    /// Inventory (NVML) id: the key the engine uses for budgets and pools.
    pub id: usize,
    /// PCI bus id as the inventory reports it.
    pub pci_bus_id: Option<String>,
}

/// What became of one engine device.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CudaMatch {
    /// The device is visible to this process as CUDA ordinal `ordinal`.
    Visible { ordinal: usize },
    /// The device exists but this process cannot see it (for example it is
    /// excluded by `CUDA_VISIBLE_DEVICES`).
    Hidden,
    /// The device could not be matched, with the reason.
    Unmatched(String),
}

#[derive(Debug, Clone)]
pub struct CudaOrdinalMap {
    entries: Vec<(usize, CudaMatch, Option<String>)>,
    /// PCI id per CUDA ordinal, as enumerated.
    ordinals: Vec<Option<String>>,
}

/// Canonical PCI bus id (`dddddddd:bb:dd.f`, lowercase) or `None` if malformed.
pub fn normalize_pci(raw: &str) -> Option<String> {
    let trimmed = raw.trim_matches(|c: char| c == '\0' || c.is_whitespace());
    if trimmed.is_empty() {
        return None;
    }
    let lower = trimmed.to_ascii_lowercase();
    let parts: Vec<&str> = lower.split(':').collect();
    let (domain, bus, devfn) = match parts.as_slice() {
        [domain, bus, devfn] => (*domain, *bus, *devfn),
        [bus, devfn] => ("0", *bus, *devfn),
        _ => return None,
    };
    let (dev, func) = devfn.split_once('.')?;
    let hex = |s: &str| !s.is_empty() && s.chars().all(|c| c.is_ascii_hexdigit());
    if ![domain, bus, dev, func].iter().all(|s| hex(s)) {
        return None;
    }
    Some(format!("{domain:0>8}:{bus:0>2}:{dev:0>2}.{func}"))
}

impl CudaOrdinalMap {
    /// Join the engine's CUDA devices to the CUDA runtime's visible ordinals.
    /// `ordinal_pci[i]` is the PCI bus id the CUDA driver reports for ordinal
    /// `i` (`None` if it could not be read); its length is the number of CUDA
    /// devices visible to the process.
    pub fn build(engine: &[EngineCudaDevice], ordinal_pci: &[Option<String>]) -> Self {
        let ordinals: Vec<Option<String>> = ordinal_pci
            .iter()
            .map(|pci| pci.as_deref().and_then(normalize_pci))
            .collect();
        let ordinals_all_known = ordinals.iter().all(Option::is_some);
        let engine_pci: Vec<Option<String>> = engine
            .iter()
            .map(|d| d.pci_bus_id.as_deref().and_then(normalize_pci))
            .collect();
        let engine_all_known = engine_pci.iter().all(Option::is_some);

        let entries = engine
            .iter()
            .zip(&engine_pci)
            .map(|(device, pci)| {
                let result = if engine_all_known && ordinals_all_known {
                    Self::match_by_pci(device, pci.as_deref().unwrap_or(""), engine, &engine_pci, &ordinals)
                } else if !engine_all_known && !ordinals.iter().any(Option::is_some) && engine.len() == ordinals.len() {
                    // Neither side can identify its devices and the counts agree:
                    // the historical id == ordinal behavior.
                    CudaMatch::Visible { ordinal: device.id }
                } else if engine.len() == 1 && ordinals.len() == 1 {
                    // One GPU on each side: nothing to confuse it with.
                    CudaMatch::Visible { ordinal: 0 }
                } else if let (Some(pci), true) = (pci.as_deref(), ordinals_all_known) {
                    // Partial identity: match what is known, refuse to guess the rest.
                    Self::match_by_pci(device, pci, engine, &engine_pci, &ordinals)
                } else {
                    CudaMatch::Unmatched(format!(
                        "no PCI bus id to match engine CUDA device {} against {} visible CUDA devices",
                        device.id,
                        ordinals.len()
                    ))
                };
                (device.id, result, pci.clone())
            })
            .collect();
        Self { entries, ordinals }
    }

    fn match_by_pci(
        device: &EngineCudaDevice,
        pci: &str,
        engine: &[EngineCudaDevice],
        engine_pci: &[Option<String>],
        ordinals: &[Option<String>],
    ) -> CudaMatch {
        let engine_twins = engine
            .iter()
            .zip(engine_pci)
            .filter(|(_, other)| other.as_deref() == Some(pci))
            .count();
        let hits: Vec<usize> = ordinals
            .iter()
            .enumerate()
            .filter(|(_, o)| o.as_deref() == Some(pci))
            .map(|(i, _)| i)
            .collect();
        match (hits.as_slice(), engine_twins) {
            ([ordinal], 1) => CudaMatch::Visible { ordinal: *ordinal },
            ([], _) => CudaMatch::Hidden,
            (_, twins) if twins > 1 => CudaMatch::Unmatched(format!(
                "engine CUDA device {} shares PCI bus id {pci} with {} other engine device(s) \
                 (for example MIG instances); the PCI id cannot tell them apart",
                device.id,
                twins - 1
            )),
            (hits, _) => CudaMatch::Unmatched(format!(
                "PCI bus id {pci} matches {} CUDA ordinals",
                hits.len()
            )),
        }
    }

    /// The outcome for an engine device id (`None`: not an inventory device).
    pub fn status(&self, engine_id: usize) -> Option<&CudaMatch> {
        self.entries
            .iter()
            .find(|(id, _, _)| *id == engine_id)
            .map(|(_, m, _)| m)
    }

    /// CUDA ordinal of a visible engine device.
    pub fn ordinal_for(&self, engine_id: usize) -> Option<usize> {
        match self.status(engine_id) {
            Some(CudaMatch::Visible { ordinal }) => Some(*ordinal),
            _ => None,
        }
    }

    /// Engine id of a visible CUDA ordinal.
    pub fn engine_id_for(&self, ordinal: usize) -> Option<usize> {
        self.entries.iter().find_map(|(id, m, _)| match m {
            CudaMatch::Visible { ordinal: o } if *o == ordinal => Some(*id),
            _ => None,
        })
    }

    /// One human-readable line per engine device, for the startup log.
    pub fn describe(&self) -> Vec<String> {
        self.entries
            .iter()
            .map(|(id, m, pci)| {
                let pci = pci.as_deref().unwrap_or("unknown PCI");
                match m {
                    CudaMatch::Visible { ordinal } => {
                        format!("engine CUDA device {id} ({pci}) -> CUDA ordinal {ordinal}")
                    }
                    CudaMatch::Hidden => format!(
                        "engine CUDA device {id} ({pci}) is not visible to this process (not managed)"
                    ),
                    CudaMatch::Unmatched(reason) => {
                        format!("engine CUDA device {id} ({pci}) could not be matched: {reason} (not managed)")
                    }
                }
            })
            .chain(std::iter::once(format!(
                "CUDA runtime enumerates {} device(s): [{}]",
                self.ordinals.len(),
                self.ordinals
                    .iter()
                    .enumerate()
                    .map(|(i, p)| format!("{i}={}", p.as_deref().unwrap_or("unknown PCI")))
                    .collect::<Vec<_>>()
                    .join(", ")
            )))
            .collect()
    }

    /// True when every visible engine device maps to the same number.
    pub fn is_identity(&self) -> bool {
        self.entries.iter().all(|(id, m, _)| match m {
            CudaMatch::Visible { ordinal } => ordinal == id,
            _ => false,
        })
    }
}

static GLOBAL: RwLock<Option<Arc<CudaOrdinalMap>>> = RwLock::new(None);

/// Make `map` the process-wide mapping used by [`to_cuda_ordinal`].
pub fn install(map: CudaOrdinalMap) -> Arc<CudaOrdinalMap> {
    let map = Arc::new(map);
    if let Ok(mut slot) = GLOBAL.write() {
        *slot = Some(Arc::clone(&map));
    }
    map
}

/// The installed map, if any.
pub fn installed() -> Option<Arc<CudaOrdinalMap>> {
    GLOBAL.read().ok().and_then(|slot| slot.clone())
}

/// Convert an engine CUDA device id to the CUDA runtime ordinal to open or hand
/// to a CUDA-based library (ORT's CUDA/TensorRT provider, cudarc).
///
/// Identity when no map is installed (library users, tests) and for a device the
/// map does not know. A hidden or unmatched device also returns the id, with an
/// error log: the engine never selects such a device (it has no memory
/// authority), so reaching this means a caller bypassed the engine.
pub fn to_cuda_ordinal(engine_id: usize) -> usize {
    match installed() {
        None => engine_id,
        Some(map) => match map.status(engine_id) {
            Some(CudaMatch::Visible { ordinal }) => *ordinal,
            Some(other) => {
                log::error!(
                    "engine CUDA device {engine_id} has no visible CUDA ordinal ({other:?}); using {engine_id} unchanged"
                );
                engine_id
            }
            None => engine_id,
        },
    }
}

/// [`to_cuda_ordinal`] for the `i32` device ids ONNX Runtime options take.
pub fn to_cuda_ordinal_i32(engine_id: i32) -> i32 {
    usize::try_from(engine_id)
        .map(|id| i32::try_from(to_cuda_ordinal(id)).unwrap_or(engine_id))
        .unwrap_or(engine_id)
}

#[cfg(test)]
#[path = "cuda_ordinal_tests.rs"]
mod tests;
