use kapsl_engine_api::RequestMetadata;
use pyo3::{exceptions::PyValueError, prelude::*, types::PyDict};

pub(crate) fn parse_options(
    values: Option<&Bound<'_, PyDict>>,
    api_token: Option<&str>,
) -> PyResult<Option<RequestMetadata>> {
    let mut metadata = RequestMetadata {
        auth_token: api_token.map(str::to_owned),
        ..Default::default()
    };
    let Some(values) = values else {
        return Ok(api_token.map(|_| metadata));
    };
    for (key, value) in values.iter() {
        let key: String = key.extract()?;
        macro_rules! fields {
            ($($name:ident),+ $(,)?) => {
                match key.as_str() {
                    $(stringify!($name) => metadata.$name = value.extract()?,)+
                    _ => return Err(PyValueError::new_err(format!("Unknown request option: {key}"))),
                }
            };
        }
        fields!(
            request_id,
            timeout_ms,
            priority,
            force_cpu,
            model_version,
            max_new_tokens,
            min_new_tokens,
            temperature,
            top_p,
            top_k,
            repetition_penalty,
            seed,
            stop_token_ids
        );
    }
    if metadata.timeout_ms == Some(0) {
        return Err(PyValueError::new_err("timeout_ms must be positive"));
    }
    for (name, value) in [
        ("temperature", metadata.temperature),
        ("top_p", metadata.top_p),
        ("repetition_penalty", metadata.repetition_penalty),
    ] {
        if value.is_some_and(|v| !v.is_finite()) {
            return Err(PyValueError::new_err(format!("{name} must be finite")));
        }
    }
    if metadata.temperature.is_some_and(|v| v < 0.0)
        || metadata.top_p.is_some_and(|v| !(0.0..=1.0).contains(&v))
        || metadata.repetition_penalty.is_some_and(|v| v <= 0.0)
    {
        return Err(PyValueError::new_err("Invalid generation parameter range"));
    }
    if matches!(
        (metadata.min_new_tokens, metadata.max_new_tokens),
        (Some(min), Some(max)) if min > max
    ) {
        return Err(PyValueError::new_err(
            "min_new_tokens must not exceed max_new_tokens",
        ));
    }
    Ok(Some(metadata))
}
