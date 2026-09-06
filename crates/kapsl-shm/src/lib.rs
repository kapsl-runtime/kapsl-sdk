//! Shared-memory transport primitives and server integration for Kapsl.
//!
//! The crate is organized by responsibility: memory mapping, tensor-pool
//! allocation, request queues, response mailboxes, and server dispatch.
//!
//! # Concurrency
//!
//! Queue metadata supports concurrent producers and consumers, and
//! [`ShmServer`] dispatches each accepted request in its own Tokio task.
//! Responses are routed through request-owned mailboxes instead of a shared
//! consumer queue. Tensor slots carry process-shared atomic leases, preventing
//! independent clients and the server from selecting overlapping live memory.
//!
//! The hybrid transport adds a socket or named-pipe control plane while using
//! these same shared tensor leases for its data plane. Direct SHM and hybrid IPC
//! therefore both support multiple concurrent clients without copying tensor
//! payloads through the control channel.

pub mod allocator;
pub mod mailbox;
pub mod memory;
pub mod protocol;
pub mod ring_buffer;
pub mod server;

pub use allocator::{
    ModelSubPoolConfig, PerModelAllocatorSnapshot, PerModelShmAllocator, SharedShmAllocator,
    SharedShmLease, ShmClassBudget, ShmPoolAllocator, SimpleShmAllocator, TieredShmAllocator,
};
pub use mailbox::{ResponseMailboxClaim, ResponseMailboxRegistry, RESPONSE_MAILBOX_COUNT};
pub use memory::ShmManager;
pub use protocol::{ShmRequest, ShmResponse, SHM_PROTOCOL_VERSION, SHM_QUEUE_CAPACITY};
pub use ring_buffer::{LockFreeRingBuffer, QueueError};
pub use server::{SchedulerLookup, SchedulerSnapshot, ShmServer};
