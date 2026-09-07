mod client;
pub mod hybrid_client;
mod native_stream;
mod request_options;
pub mod shm_client;
mod shm_tensor;

use client::KapslClient;
pub use hybrid_client::KapslHybridClient;
use pyo3::prelude::*;
pub use shm_client::KapslShmClient;

/// Register the native classes exposed by the `kapsl_sdk` Python extension.
#[pymodule]
fn kapsl_sdk(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<KapslClient>()?;
    module.add_class::<KapslShmClient>()?;
    module.add_class::<KapslHybridClient>()?;
    Ok(())
}
