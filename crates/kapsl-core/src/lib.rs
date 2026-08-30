#[path = "scaling/auto_scaler.rs"]
pub mod auto_scaler;
#[path = "model/engine_kind.rs"]
pub mod engine_kind;
#[path = "package/loader.rs"]
pub mod loader;
#[path = "model/registry.rs"]
pub mod model_registry;
#[path = "provider/pack.rs"]
pub mod provider_pack;
#[path = "provider/policy.rs"]
pub mod provider_policy;
#[path = "model/requirements.rs"]
pub mod requirements;

pub use auto_scaler::{AutoScaler, ScalingPolicy};
pub use engine_kind::EngineKind;
pub use loader::{
    CronJobDef, CronOverflowPolicyDef, CronPriorityDef, CronScheduleDef, Manifest, PackageLoader,
};
pub use model_registry::{ModelInfo, ModelRegistry, ModelStatus};
pub use provider_pack::{
    accelerator_provider_pack_installed, AcceleratorProviderPack, ALLOW_UNMANAGED_PROVIDERS_ENV,
    PROVIDER_PATH_ENV,
};
pub use provider_policy::{ProviderPolicy, PROVIDER_POLICY_ENV};
pub use requirements::HardwareRequirements;
