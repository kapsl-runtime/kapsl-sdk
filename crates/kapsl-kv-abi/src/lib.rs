//! Backend-neutral control and data-plane contract for KV-cache participants.
//!
//! The contract deliberately separates inference compatibility from KV
//! integration. An OpenAI-compatible endpoint can be routed by Kapsl without
//! implementing this crate, but it remains an unmanaged endpoint. Official
//! Kapsl backends advertise at least [`KvIntegrationTier::KvConnected`].
//!
//! Opaque mode is a first-class connected mode: the backend may keep its
//! physical block layout private while Kapsl still controls capacity leases,
//! admission, lifecycle, and global budgets. [`KvIntegrationTier::SharedPool`]
//! is the deepest mode and lets backend attention consume Kapsl-owned blocks.

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use thiserror::Error;

#[path = "contract/capabilities.rs"]
mod capabilities;
#[path = "contract/capacity.rs"]
mod capacity;
#[path = "contract/control.rs"]
mod control;
#[path = "contract/error.rs"]
mod error;
#[path = "contract/leases.rs"]
mod leases;
#[path = "contract/registration.rs"]
mod registration;
#[path = "contract/resize.rs"]
mod resize;
#[path = "contract/shared_pool.rs"]
mod shared_pool;
#[path = "contract/topology.rs"]
mod topology;
#[path = "contract/traits.rs"]
mod traits;

pub use capabilities::*;
pub use capacity::*;
pub use control::*;
pub use error::*;
pub use leases::*;
pub use registration::*;
pub use resize::*;
pub use shared_pool::*;
pub use topology::*;
pub use traits::*;

#[cfg(test)]
#[path = "tests/mod.rs"]
mod tests;
