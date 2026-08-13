use crate::allocator::{
    ModelSubPoolConfig, PerModelShmAllocator, ShmAllocatorSnapshot, ShmClassBudget,
};
use crate::memory::TensorHeader;
use kapsl_engine_api::EngineModelInfo;
use kapsl_scheduler::ReplicaScheduler;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

const DEFAULT_TENSOR_SLOT_CLASS_BUDGETS: [ShmClassBudget; 5] = [
    ShmClassBudget {
        slot_size: 256 * 1024,
        weight: 1,
    },
    ShmClassBudget {
        slot_size: 1024 * 1024,
        weight: 1,
    },
    ShmClassBudget {
        slot_size: 4 * 1024 * 1024,
        weight: 1,
    },
    ShmClassBudget {
        slot_size: 16 * 1024 * 1024,
        weight: 1,
    },
    ShmClassBudget {
        slot_size: 64 * 1024 * 1024,
        weight: 1,
    },
];
pub(super) const DEFAULT_TENSOR_SLOT_LEASE_TTL_SECS: u64 = 30;
pub(super) const ERROR_LEN_PREFIX_BYTES: usize = std::mem::size_of::<u64>();
pub(super) const SHM_METRICS_REFRESH_SECS: u64 = 1;
const MODEL_AWARE_MAX_CLASSES: usize = 6;
const MODEL_AWARE_MIN_SLOT_BYTES: usize = 64 * 1024;
const MODEL_AWARE_MAX_SLOT_BYTES: usize = 64 * 1024 * 1024;
const MODEL_AWARE_DYNAMIC_BATCH_FALLBACK: usize = 1;
const MODEL_AWARE_DYNAMIC_DIM_FALLBACK: usize = 256;
const MODEL_AWARE_MAX_ESTIMATED_TENSOR_BYTES: usize = 128 * 1024 * 1024;
const MODEL_AWARE_MAX_MODEL_WEIGHT: u32 = 1_000;
/// Percentage of the tensor pool reserved as a shared overflow for all models.
const MODEL_POOL_SHARED_RESERVE_PCT: usize = 10;

fn dtype_size_bytes(dtype: Option<&str>) -> usize {
    match dtype.map(|v| v.to_ascii_lowercase()) {
        Some(v) if v == "float64" || v == "fp64" || v == "int64" || v == "i64" => 8,
        Some(v) if v == "float16" || v == "fp16" => 2,
        Some(v) if v == "uint8" || v == "u8" || v == "string" || v == "utf8" => 1,
        _ => 4, // default to fp32/i32 when unknown
    }
}

fn estimate_shape_elements(shape: &[i64]) -> Option<usize> {
    if shape.is_empty() {
        return Some(1);
    }
    let mut elements = 1usize;
    for (index, dim) in shape.iter().copied().enumerate() {
        let resolved = if dim > 0 {
            dim as usize
        } else if index == 0 {
            MODEL_AWARE_DYNAMIC_BATCH_FALLBACK
        } else {
            MODEL_AWARE_DYNAMIC_DIM_FALLBACK
        };
        elements = elements.checked_mul(resolved)?;
    }
    Some(elements)
}

fn estimate_tensor_bytes(shape: &[i64], dtype: Option<&str>) -> Option<usize> {
    let elements = estimate_shape_elements(shape)?;
    let elem_size = dtype_size_bytes(dtype);
    let payload_bytes = elements.checked_mul(elem_size)?;
    let total = payload_bytes.checked_add(std::mem::size_of::<TensorHeader>())?;
    Some(total.min(MODEL_AWARE_MAX_ESTIMATED_TENSOR_BYTES))
}

fn bucket_slot_size(bytes: usize, pool_bytes: usize) -> usize {
    let pool_cap = pool_bytes.max(1);
    let min_slot = MODEL_AWARE_MIN_SLOT_BYTES.min(pool_cap);
    let max_slot = MODEL_AWARE_MAX_SLOT_BYTES.min(pool_cap).max(min_slot);
    let clamped = bytes.clamp(min_slot, max_slot);
    clamped
        .checked_next_power_of_two()
        .unwrap_or(max_slot)
        .clamp(min_slot, max_slot)
}

fn add_model_tensor_buckets(
    shapes: &[Vec<i64>],
    dtypes: &[String],
    pool_bytes: usize,
    model_weight: u32,
    buckets: &mut HashMap<usize, u32>,
) {
    for (index, shape) in shapes.iter().enumerate() {
        let dtype = dtypes.get(index).map(String::as_str);
        if let Some(bytes) = estimate_tensor_bytes(shape, dtype) {
            let slot_size = bucket_slot_size(bytes, pool_bytes);
            let entry = buckets.entry(slot_size).or_insert(0);
            *entry = entry.saturating_add(model_weight);
        }
    }
}

fn add_model_info_buckets(
    model_info: &EngineModelInfo,
    pool_bytes: usize,
    model_weight: u32,
    buckets: &mut HashMap<usize, u32>,
) {
    add_model_tensor_buckets(
        &model_info.input_shapes,
        &model_info.input_dtypes,
        pool_bytes,
        model_weight,
        buckets,
    );
    add_model_tensor_buckets(
        &model_info.output_shapes,
        &model_info.output_dtypes,
        pool_bytes,
        model_weight,
        buckets,
    );
}

fn model_peak_weight(model_info: &EngineModelInfo) -> u32 {
    model_info
        .peak_concurrency
        .unwrap_or(1)
        .clamp(1, MODEL_AWARE_MAX_MODEL_WEIGHT)
}

/// Derive model-aware class budgets for a single model given its allocated pool size.
pub(super) fn derive_single_model_class_budgets(
    model_id: u32,
    schedulers: &HashMap<u32, Arc<dyn ReplicaScheduler + Send + Sync>>,
    pool_bytes: usize,
) -> Vec<ShmClassBudget> {
    let Some(info) = schedulers.get(&model_id).and_then(|s| s.model_info()) else {
        return DEFAULT_TENSOR_SLOT_CLASS_BUDGETS.to_vec();
    };

    let mut buckets: HashMap<usize, u32> = HashMap::new();
    let control_slot = bucket_slot_size(ERROR_LEN_PREFIX_BYTES + 4096, pool_bytes);
    buckets.insert(control_slot, 1);
    let model_weight = model_peak_weight(&info);
    add_model_info_buckets(&info, pool_bytes, model_weight, &mut buckets);

    if buckets.len() <= 1 {
        return DEFAULT_TENSOR_SLOT_CLASS_BUDGETS.to_vec();
    }

    let mut weighted: Vec<(usize, u32)> = buckets.into_iter().collect();
    weighted.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    weighted.truncate(MODEL_AWARE_MAX_CLASSES);
    weighted.sort_unstable_by_key(|(s, _)| *s);
    weighted
        .into_iter()
        .map(|(slot_size, weight)| ShmClassBudget {
            slot_size,
            weight: weight.max(1),
        })
        .collect()
}

/// Partition `[base_offset, base_offset + pool_bytes)` into per-model sub-pools.
///
/// Each model's share is proportional to its `peak_concurrency`. Models are ordered
/// by `model_id` for a deterministic layout. A fixed percentage is reserved at the
/// end as a shared overflow pool.
pub(super) fn build_per_model_pool(
    schedulers: &HashMap<u32, Arc<dyn ReplicaScheduler + Send + Sync>>,
    base_offset: usize,
    pool_bytes: usize,
    lease_ttl: std::time::Duration,
) -> PerModelShmAllocator {
    let mut model_weights: Vec<(u32, u32)> = schedulers
        .keys()
        .map(|&id| {
            let weight = schedulers[&id]
                .model_info()
                .map(|info| model_peak_weight(&info))
                .unwrap_or(1);
            (id, weight)
        })
        .collect();
    model_weights.sort_unstable_by_key(|(id, _)| *id);

    let total_weight: u64 = model_weights
        .iter()
        .map(|(_, w)| *w as u64)
        .sum::<u64>()
        .max(1);

    let shared_bytes =
        (pool_bytes * MODEL_POOL_SHARED_RESERVE_PCT / 100).max(MODEL_AWARE_MIN_SLOT_BYTES);
    let model_pool_total = pool_bytes.saturating_sub(shared_bytes);

    let n = model_weights.len();
    let mut model_configs: Vec<ModelSubPoolConfig> = Vec::with_capacity(n);
    let mut allocated = 0usize;

    for (idx, (model_id, weight)) in model_weights.iter().enumerate() {
        let is_last = idx == n - 1;
        let bytes = if is_last {
            model_pool_total.saturating_sub(allocated)
        } else {
            (model_pool_total as u64 * *weight as u64 / total_weight) as usize
        };

        if bytes < MODEL_AWARE_MIN_SLOT_BYTES {
            // Pool slice too small to be useful; requests fall back to the shared pool.
            continue;
        }

        let class_budgets = derive_single_model_class_budgets(*model_id, schedulers, bytes);
        model_configs.push(ModelSubPoolConfig {
            model_id: *model_id,
            pool_bytes: bytes,
            class_budgets,
        });
        allocated = allocated.saturating_add(bytes);
    }

    PerModelShmAllocator::new(base_offset, pool_bytes, model_configs, lease_ttl)
}

struct DynamicPerModelPoolState {
    model_ids: Vec<u32>,
    allocator: PerModelShmAllocator,
}

/// Owns the current model-aware layout and replaces it when the live scheduler
/// registry changes. Replacement is only safe when every response lease has
/// drained; until then newly discovered models use the allocator's shared
/// overflow pool.
pub(super) struct DynamicPerModelPool {
    base_offset: usize,
    pool_bytes: usize,
    lease_ttl: std::time::Duration,
    state: Mutex<DynamicPerModelPoolState>,
}

impl DynamicPerModelPool {
    pub(super) fn new(
        schedulers: &HashMap<u32, Arc<dyn ReplicaScheduler + Send + Sync>>,
        base_offset: usize,
        pool_bytes: usize,
        lease_ttl: std::time::Duration,
    ) -> Self {
        Self {
            base_offset,
            pool_bytes,
            lease_ttl,
            state: Mutex::new(DynamicPerModelPoolState {
                model_ids: sorted_model_ids(schedulers),
                allocator: build_per_model_pool(schedulers, base_offset, pool_bytes, lease_ttl),
            }),
        }
    }

    /// Returns the new layout summary when a registry change was applied.
    pub(super) fn refresh(
        &self,
        schedulers: &HashMap<u32, Arc<dyn ReplicaScheduler + Send + Sync>>,
    ) -> Option<String> {
        let model_ids = sorted_model_ids(schedulers);
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        if state.model_ids == model_ids {
            return None;
        }

        // Repartitioning reuses the same offsets. Never replace a layout while
        // a client may still be reading a response allocated from the old one.
        if state.allocator.snapshot().in_use_slots != 0 {
            return None;
        }

        state.allocator = build_per_model_pool(
            schedulers,
            self.base_offset,
            self.pool_bytes,
            self.lease_ttl,
        );
        state.model_ids = model_ids;
        Some(state.allocator.layout_summary())
    }

    pub(super) fn try_allocate(&self, model_id: u32, required_size: usize) -> Option<usize> {
        // Allocation is performed while holding the layout lock so refresh
        // cannot observe an empty old allocator and replace it immediately
        // before this lease is recorded.
        self.state
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
            .allocator
            .try_allocate(model_id, required_size)
    }

    pub(super) fn snapshot(&self) -> ShmAllocatorSnapshot {
        self.state
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
            .allocator
            .snapshot()
    }

    pub(super) fn layout_summary(&self) -> String {
        self.state
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
            .allocator
            .layout_summary()
    }

    pub(super) fn largest_slot_size_for_model(&self, model_id: u32) -> usize {
        self.state
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
            .allocator
            .largest_slot_size_for_model(model_id)
    }
}

fn sorted_model_ids(
    schedulers: &HashMap<u32, Arc<dyn ReplicaScheduler + Send + Sync>>,
) -> Vec<u32> {
    let mut model_ids: Vec<_> = schedulers.keys().copied().collect();
    model_ids.sort_unstable();
    model_ids
}
