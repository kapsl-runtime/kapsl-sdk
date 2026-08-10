//! Environment-variable parsing shared across the backend crate.

/// Read a boolean-ish env var, falling back to `default` when unset or unparseable.
///
/// Accepts `1/true/yes/on` and `0/false/no/off`, case-insensitively.
pub(crate) fn read_env_flag(name: &str, default: bool) -> bool {
    std::env::var(name)
        .ok()
        .and_then(|value| match value.trim().to_ascii_lowercase().as_str() {
            "1" | "true" | "yes" | "on" => Some(true),
            "0" | "false" | "no" | "off" => Some(false),
            _ => None,
        })
        .unwrap_or(default)
}
