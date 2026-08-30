//! Participant KV-memory capacity and accounting models.

use super::*;

/// Physical memory domain occupied by a participant-owned KV pool.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum KvMemoryDomain {
    Host,
    HostPinned {
        provider: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        device_id: Option<u32>,
    },
    HostMapped {
        provider: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        device_id: Option<u32>,
    },
    Cuda {
        device_id: u32,
    },
    Provider {
        provider: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        device_id: Option<u32>,
    },
}

impl KvMemoryDomain {
    pub(crate) fn validate(&self) -> Result<(), KvContractError> {
        match self {
            Self::Host | Self::Cuda { .. } => Ok(()),
            Self::HostPinned { provider, .. }
            | Self::HostMapped { provider, .. }
            | Self::Provider { provider, .. }
                if provider.trim().is_empty() =>
            {
                Err(KvContractError::invalid_capabilities(
                    "KV memory-domain provider names must not be empty",
                ))
            }
            Self::HostPinned { .. } | Self::HostMapped { .. } | Self::Provider { .. } => Ok(()),
        }
    }
}

/// Logical-to-physical accounting for one cache group. Opaque participants
/// must expose this much information even though their block layout is private.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct KvCapacityGroup {
    pub group_id: String,
    /// Groups with the same pool ID alias one physical allocation pool. Their
    /// reservation cost is the maximum within the pool, not the sum.
    pub pool_id: String,
    /// Tokens covered by one backend allocation unit (normally one page).
    pub allocation_granularity_tokens: u32,
    /// Device bytes consumed by that allocation across every layer in the group.
    pub bytes_per_allocation: u64,
    /// Every physical domain containing this pool. The reservation is charged
    /// once per domain, which models tensor-parallel workers with one allocator
    /// pool on each device without exposing their backend-private block IDs.
    pub memory_domains: Vec<KvMemoryDomain>,
    /// Current ceiling advertised by a backend-owned allocator, or the maximum
    /// block count requested from a runtime-owned shared-pool provisioner.
    /// Required for `shared_pool` registrations.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_allocations: Option<u64>,
}

impl KvCapacityGroup {
    pub fn bytes_for_tokens(&self, tokens: u64) -> Option<u64> {
        if self.allocation_granularity_tokens == 0 || self.bytes_per_allocation == 0 {
            return None;
        }
        let granularity = u64::from(self.allocation_granularity_tokens);
        let allocations = tokens.div_ceil(granularity);
        if self
            .max_allocations
            .is_some_and(|maximum| allocations > maximum)
        {
            return None;
        }
        allocations.checked_mul(self.bytes_per_allocation)
    }

    pub fn bytes_for_reservation(&self, reservation: &KvGroupReservation) -> Option<u64> {
        if reservation.group_id != self.group_id
            || reservation.token_capacity == 0
            || self.allocation_granularity_tokens == 0
            || self.bytes_per_allocation == 0
        {
            return None;
        }
        let token_allocations = u64::from(reservation.token_capacity)
            .div_ceil(u64::from(self.allocation_granularity_tokens));
        let allocations = token_allocations.max(u64::from(reservation.minimum_blocks.unwrap_or(0)));
        if self
            .max_allocations
            .is_some_and(|maximum| allocations > maximum)
        {
            return None;
        }
        allocations.checked_mul(self.bytes_per_allocation)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct KvCapacityModel {
    pub groups: Vec<KvCapacityGroup>,
}

impl KvCapacityModel {
    pub fn validate(&self) -> Result<(), KvContractError> {
        if self.groups.is_empty() {
            return Err(KvContractError::invalid_capabilities(
                "at least one KV capacity group is required",
            ));
        }
        let mut group_ids = BTreeSet::new();
        let mut pool_domains = BTreeMap::<&str, BTreeSet<&KvMemoryDomain>>::new();
        for group in &self.groups {
            let domains = group.memory_domains.iter().collect::<BTreeSet<_>>();
            if group.group_id.trim().is_empty()
                || group.pool_id.trim().is_empty()
                || !group_ids.insert(group.group_id.as_str())
                || group.allocation_granularity_tokens == 0
                || group.bytes_per_allocation == 0
                || group.max_allocations.is_some_and(|value| value == 0)
                || domains.is_empty()
                || domains.len() != group.memory_domains.len()
            {
                return Err(KvContractError::invalid_capabilities(
                    "capacity group IDs and memory domains must be unique and accounting values must be non-zero",
                ));
            }
            for domain in &group.memory_domains {
                domain.validate()?;
            }
            if let Some(existing) = pool_domains.get(group.pool_id.as_str()) {
                if existing != &domains {
                    return Err(KvContractError::invalid_capabilities(format!(
                        "capacity groups sharing pool '{}' must name the same memory domains",
                        group.pool_id
                    )));
                }
            } else {
                pool_domains.insert(group.pool_id.as_str(), domains);
            }
        }
        Ok(())
    }

    /// Compute device bytes while honoring cache groups that alias one backend
    /// allocation pool (as vLLM's hybrid memory allocator does).
    pub fn bytes_for_reservations(&self, reservations: &[KvGroupReservation]) -> Option<u64> {
        self.bytes_by_domain_for_reservations(reservations)?
            .values()
            .try_fold(0u64, |total, bytes| total.checked_add(*bytes))
    }

    /// Compute physical bytes per authority domain while honoring cache groups
    /// that alias the same backend allocation pool.
    pub fn bytes_by_domain_for_reservations(
        &self,
        reservations: &[KvGroupReservation],
    ) -> Option<BTreeMap<KvMemoryDomain, u64>> {
        self.validate().ok()?;
        let groups = self
            .groups
            .iter()
            .map(|group| (group.group_id.as_str(), group))
            .collect::<BTreeMap<_, _>>();
        let mut pool_bytes = BTreeMap::<(KvMemoryDomain, &str), u64>::new();
        for reservation in reservations {
            let group = groups.get(reservation.group_id.as_str())?;
            let bytes = group.bytes_for_reservation(reservation)?;
            for domain in &group.memory_domains {
                pool_bytes
                    .entry((domain.clone(), group.pool_id.as_str()))
                    .and_modify(|current| *current = (*current).max(bytes))
                    .or_insert(bytes);
            }
        }
        let mut domain_bytes = BTreeMap::<KvMemoryDomain, u64>::new();
        for ((domain, _), bytes) in pool_bytes {
            let total = domain_bytes.entry(domain).or_default();
            *total = total.checked_add(bytes)?;
        }
        Some(domain_bytes)
    }
}
