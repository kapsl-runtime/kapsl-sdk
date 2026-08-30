//! Structured KV-cache topology and geometry.

use super::*;

/// Element type stored by a cache group.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum KvElementType {
    F16,
    Bf16,
    F32,
    I8,
    Fp8E4m3,
    Custom { name: String },
}

/// Tensor layout of a structured cache group.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum KvTensorLayout {
    /// Separate K/V planes laid out as layer, sequence block, head, dimension.
    LayerSequenceHeadDim,
    /// Paged-attention block, K/V plane, head, token, dimension.
    BlockKvHeadTokenDim,
    /// A stable backend-defined layout identifier.
    BackendNative { layout_id: String },
}

/// Geometry of one independently managed cache group.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum KvCacheGeometry {
    PagedAttention {
        block_size_tokens: u32,
        kv_heads: u32,
        key_head_dim: u32,
        value_head_dim: u32,
        element_type: KvElementType,
        layout: KvTensorLayout,
    },
    RecurrentState {
        state_bytes_per_sequence: u64,
        element_type: KvElementType,
        layout: KvTensorLayout,
    },
    /// Geometry hints for a KV-connected backend using opaque metadata.
    Opaque {
        layout_id: String,
        block_size_tokens: Option<u32>,
        bytes_per_block: Option<u64>,
    },
}

/// Attention/state policy associated with a cache group.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum KvCachePolicy {
    FullAttention,
    SlidingWindow { window_tokens: u32 },
    Recurrent,
    BackendDefined { policy_id: String },
}

/// Stable logical identity for a layer in a cache group.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct KvLayerId {
    pub index: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

impl KvLayerId {
    pub fn indexed(index: u32) -> Self {
        Self { index, name: None }
    }
}

/// Tensor/pipeline shard that owns the described cache groups.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct KvShard {
    pub tensor_parallel_rank: u32,
    pub tensor_parallel_world_size: u32,
    pub pipeline_parallel_rank: u32,
    pub pipeline_parallel_world_size: u32,
}

impl Default for KvShard {
    fn default() -> Self {
        Self {
            tensor_parallel_rank: 0,
            tensor_parallel_world_size: 1,
            pipeline_parallel_rank: 0,
            pipeline_parallel_world_size: 1,
        }
    }
}

/// One cache group. Groups may use different geometries (for example full
/// attention, sliding-window attention, and recurrent state in one model).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct KvCacheGroup {
    pub group_id: String,
    pub layers: Vec<KvLayerId>,
    pub geometry: KvCacheGeometry,
    pub policy: KvCachePolicy,
}

/// Logical cache topology for one model replica/shard.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct KvTopology {
    pub abi_version: KvAbiVersion,
    pub model_fingerprint: String,
    #[serde(default)]
    pub shard: KvShard,
    pub cache_groups: Vec<KvCacheGroup>,
}

impl KvTopology {
    pub fn validate(&self) -> Result<(), KvContractError> {
        if !KAPSL_KV_ABI_VERSION.accepts(self.abi_version) {
            return Err(KvContractError::VersionMismatch {
                host: KAPSL_KV_ABI_VERSION,
                participant: self.abi_version,
            });
        }
        if self.model_fingerprint.trim().is_empty() {
            return Err(KvContractError::invalid_topology(
                "model_fingerprint must not be empty",
            ));
        }
        validate_shard(self.shard)?;
        if self.cache_groups.is_empty() {
            return Err(KvContractError::invalid_topology(
                "at least one cache group is required",
            ));
        }

        let mut group_ids = BTreeSet::new();
        for group in &self.cache_groups {
            if group.group_id.trim().is_empty() || !group_ids.insert(group.group_id.as_str()) {
                return Err(KvContractError::invalid_topology(
                    "cache group IDs must be non-empty and unique",
                ));
            }
            if group.layers.is_empty() {
                return Err(KvContractError::invalid_topology(format!(
                    "cache group '{}' has no layers",
                    group.group_id
                )));
            }
            let mut layer_indices = BTreeSet::new();
            for layer in &group.layers {
                if layer
                    .name
                    .as_ref()
                    .is_some_and(|name| name.trim().is_empty())
                    || !layer_indices.insert(layer.index)
                {
                    return Err(KvContractError::invalid_topology(
                        "layer indices must be unique and layer names cannot be empty",
                    ));
                }
            }
            validate_geometry(&group.geometry)?;
            validate_policy(&group.policy)?;
            validate_geometry_policy(&group.geometry, &group.policy)?;
        }
        Ok(())
    }
}

pub(crate) fn validate_shard(shard: KvShard) -> Result<(), KvContractError> {
    if shard.tensor_parallel_world_size == 0
        || shard.pipeline_parallel_world_size == 0
        || shard.tensor_parallel_rank >= shard.tensor_parallel_world_size
        || shard.pipeline_parallel_rank >= shard.pipeline_parallel_world_size
    {
        return Err(KvContractError::invalid_topology(
            "parallel world sizes must be non-zero and ranks must be in range",
        ));
    }
    Ok(())
}

fn validate_geometry(geometry: &KvCacheGeometry) -> Result<(), KvContractError> {
    match geometry {
        KvCacheGeometry::PagedAttention {
            block_size_tokens,
            kv_heads,
            key_head_dim,
            value_head_dim,
            element_type,
            layout,
        } => {
            if *block_size_tokens == 0
                || *kv_heads == 0
                || *key_head_dim == 0
                || *value_head_dim == 0
            {
                return Err(KvContractError::invalid_topology(
                    "paged-attention dimensions must be non-zero",
                ));
            }
            validate_element_type(element_type)?;
            validate_layout(layout)?;
        }
        KvCacheGeometry::RecurrentState {
            state_bytes_per_sequence,
            element_type,
            layout,
        } => {
            if *state_bytes_per_sequence == 0 {
                return Err(KvContractError::invalid_topology(
                    "recurrent state size must be non-zero",
                ));
            }
            validate_element_type(element_type)?;
            validate_layout(layout)?;
        }
        KvCacheGeometry::Opaque {
            layout_id,
            block_size_tokens,
            bytes_per_block,
        } => {
            if layout_id.trim().is_empty()
                || block_size_tokens.is_some_and(|value| value == 0)
                || bytes_per_block.is_some_and(|value| value == 0)
            {
                return Err(KvContractError::invalid_topology(
                    "opaque geometry requires a layout ID and non-zero optional hints",
                ));
            }
        }
    }
    Ok(())
}

fn validate_element_type(element_type: &KvElementType) -> Result<(), KvContractError> {
    if matches!(element_type, KvElementType::Custom { name } if name.trim().is_empty()) {
        return Err(KvContractError::invalid_topology(
            "custom element type name must not be empty",
        ));
    }
    Ok(())
}

fn validate_layout(layout: &KvTensorLayout) -> Result<(), KvContractError> {
    if matches!(layout, KvTensorLayout::BackendNative { layout_id } if layout_id.trim().is_empty())
    {
        return Err(KvContractError::invalid_topology(
            "backend-native layout ID must not be empty",
        ));
    }
    Ok(())
}

fn validate_policy(policy: &KvCachePolicy) -> Result<(), KvContractError> {
    match policy {
        KvCachePolicy::SlidingWindow { window_tokens } if *window_tokens == 0 => Err(
            KvContractError::invalid_topology("sliding window must be non-zero"),
        ),
        KvCachePolicy::BackendDefined { policy_id } if policy_id.trim().is_empty() => Err(
            KvContractError::invalid_topology("backend-defined policy ID must not be empty"),
        ),
        _ => Ok(()),
    }
}

fn validate_geometry_policy(
    geometry: &KvCacheGeometry,
    policy: &KvCachePolicy,
) -> Result<(), KvContractError> {
    match (geometry, policy) {
        (KvCacheGeometry::PagedAttention { .. }, KvCachePolicy::Recurrent)
        | (KvCacheGeometry::RecurrentState { .. }, KvCachePolicy::FullAttention)
        | (KvCacheGeometry::RecurrentState { .. }, KvCachePolicy::SlidingWindow { .. }) => Err(
            KvContractError::invalid_topology("cache geometry and policy are incompatible"),
        ),
        _ => Ok(()),
    }
}
