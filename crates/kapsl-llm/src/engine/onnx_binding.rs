//! Execution ordering for device-resident KV bindings.

use kapsl_engine_api::EngineError;
use ort::io_binding::IoBinding;
use ort::session::{Session, SessionOutputs};

pub(super) fn run_device_kv_binding<'b, 's: 'b>(
    session: &'s mut Session,
    binding: &'b IoBinding,
) -> Result<SessionOutputs<'b>, EngineError> {
    // BindInput may enqueue transfers independently of the execution stream.
    // Complete those transfers before consuming token/mask/position inputs or
    // KV from another session. Keep both the inputs and allocation scope alive
    // in the caller through the output synchronization below.
    binding.synchronize_inputs().map_err(|error| {
        EngineError::backend(format!("Device KV input synchronization failed: {error}"))
    })?;
    let outputs = session
        .run_binding(binding)
        .map_err(|error| EngineError::backend(error.to_string()))?;
    binding.synchronize_outputs().map_err(|error| {
        EngineError::backend(format!("Device KV output synchronization failed: {error}"))
    })?;
    Ok(outputs)
}
