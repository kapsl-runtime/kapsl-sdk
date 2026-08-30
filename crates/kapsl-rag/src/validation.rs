use std::path::{Component, Path};

pub(crate) fn path_component(value: &str, label: &str) -> Result<(), String> {
    if value.is_empty() || value.trim() != value {
        return Err(format!("{label} must be a non-empty path component"));
    }
    if value.contains(['/', '\\', '\0']) || matches!(value, "." | "..") {
        return Err(format!("{label} contains an invalid path component"));
    }
    let mut components = Path::new(value).components();
    if !matches!(components.next(), Some(Component::Normal(_))) || components.next().is_some() {
        return Err(format!("{label} must be a single relative path component"));
    }
    Ok(())
}

pub(crate) fn extension_id(value: &str) -> Result<(), String> {
    path_component(value, "extension id")?;
    if !value
        .chars()
        .all(|character| character.is_ascii_alphanumeric() || matches!(character, '.' | '_' | '-'))
    {
        return Err(
            "extension id may contain only ASCII letters, numbers, dots, underscores, or hyphens"
                .to_string(),
        );
    }
    Ok(())
}

pub(crate) fn relative_path(value: &str, label: &str) -> Result<(), String> {
    if value.is_empty() || value.contains('\0') {
        return Err(format!("{label} must be a non-empty relative path"));
    }
    if value
        .split(['/', '\\'])
        .any(|component| component.is_empty() || matches!(component, "." | ".."))
    {
        return Err(format!("{label} must stay within its configured root"));
    }
    let path = Path::new(value);
    if path.is_absolute()
        || path
            .components()
            .any(|component| !matches!(component, Component::Normal(_)))
    {
        return Err(format!("{label} must stay within its configured root"));
    }
    Ok(())
}

pub(crate) fn env_key_value(key: &str, value: &str) -> Result<(), String> {
    if key.is_empty() {
        return Err("environment key cannot be empty".to_string());
    }
    if key.contains(['=', '\0']) || value.contains('\0') {
        return Err("environment key/value contains an invalid character".to_string());
    }
    Ok(())
}

pub(crate) fn guest_path(path: &str) -> Result<(), String> {
    if path.is_empty() || !path.starts_with('/') {
        return Err("preopened guest path must be absolute".to_string());
    }
    if path.contains('\0') {
        return Err("preopened guest path cannot contain NUL".to_string());
    }
    if path != "/"
        && path[1..]
            .split('/')
            .any(|component| component.is_empty() || matches!(component, "." | ".."))
    {
        return Err("preopened guest path contains an invalid component".to_string());
    }
    Ok(())
}

pub(crate) fn host_path(path: &Path) -> Result<(), String> {
    if !path.is_absolute() {
        return Err("preopened host path must be absolute".to_string());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn path_components_reject_traversal_and_separators() {
        for value in ["", ".", "..", "../escape", "nested/path", "nested\\path"] {
            assert!(path_component(value, "id").is_err(), "accepted {value:?}");
        }
        assert!(path_component("workspace-1", "id").is_ok());
    }

    #[test]
    fn relative_paths_allow_nested_documents_but_not_traversal() {
        assert!(relative_path("folder/document.txt", "document id").is_ok());
        assert!(relative_path("../document.txt", "document id").is_err());
        assert!(relative_path("/tmp/document.txt", "document id").is_err());
    }

    #[test]
    fn guest_paths_reject_parent_and_empty_components() {
        assert!(guest_path("/data").is_ok());
        assert!(guest_path("/").is_ok());
        assert!(guest_path("/../data").is_err());
        assert!(guest_path("/data//nested").is_err());
    }
}
