//! Runtime policy for selecting an inference provider.

/// Environment variable controlling whether runtime provider selection follows
/// the package manifest or considers the fastest available provider.
pub const PROVIDER_POLICY_ENV: &str = "KAPSL_PROVIDER_POLICY";

/// Policy for selecting an inference provider at runtime.
///
/// Only the explicit `manifest` value opts out of automatic fastest-provider
/// selection. Missing, empty, and unrecognized values retain the historical
/// [`ProviderPolicy::Fastest`] behavior.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum ProviderPolicy {
    Manifest,
    #[default]
    Fastest,
}

impl ProviderPolicy {
    /// Parse a configured policy value without reading process state.
    pub fn from_value(value: Option<&str>) -> Self {
        match value {
            Some(value) if value.trim().eq_ignore_ascii_case("manifest") => Self::Manifest,
            _ => Self::Fastest,
        }
    }

    /// Read the provider policy from [`PROVIDER_POLICY_ENV`].
    pub fn from_env() -> Self {
        let value = std::env::var(PROVIDER_POLICY_ENV).ok();
        Self::from_value(value.as_deref())
    }

    /// Whether provider selection must honor the package manifest.
    pub const fn uses_manifest(self) -> bool {
        matches!(self, Self::Manifest)
    }

    /// Whether provider selection may consider the fastest available provider.
    pub const fn uses_fastest(self) -> bool {
        matches!(self, Self::Fastest)
    }
}

#[cfg(test)]
mod tests {
    use super::ProviderPolicy;

    #[test]
    fn missing_policy_defaults_to_fastest() {
        assert_eq!(ProviderPolicy::from_value(None), ProviderPolicy::Fastest);
        assert_eq!(ProviderPolicy::default(), ProviderPolicy::Fastest);
    }

    #[test]
    fn manifest_policy_is_trimmed_and_case_insensitive() {
        for value in ["manifest", " MANIFEST ", "\tMaNiFeSt\n"] {
            assert_eq!(
                ProviderPolicy::from_value(Some(value)),
                ProviderPolicy::Manifest
            );
        }
    }

    #[test]
    fn every_other_value_uses_fastest_policy() {
        for value in ["", "fastest", " FASTEST ", "unknown", "manifest-only"] {
            assert_eq!(
                ProviderPolicy::from_value(Some(value)),
                ProviderPolicy::Fastest
            );
        }
    }

    #[test]
    fn policy_predicates_are_typed() {
        assert!(ProviderPolicy::Manifest.uses_manifest());
        assert!(!ProviderPolicy::Manifest.uses_fastest());
        assert!(ProviderPolicy::Fastest.uses_fastest());
        assert!(!ProviderPolicy::Fastest.uses_manifest());
    }
}
