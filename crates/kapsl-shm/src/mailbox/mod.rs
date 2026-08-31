//! Request-specific response routing inside the shared-memory control region.

use crate::memory::ShmManager;
use crate::protocol::ShmResponse;
use std::cell::UnsafeCell;
use std::mem::MaybeUninit;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

/// Number of simultaneous direct-SHM requests supported by one region.
pub const RESPONSE_MAILBOX_COUNT: usize = 256;

const MAILBOX_FREE: u32 = 0;
const MAILBOX_CLAIMED: u32 = 1;
const MAILBOX_PROCESSING: u32 = 2;
const MAILBOX_RESPONSE_READY: u32 = 3;
const MAILBOX_RELEASING: u32 = 4;
const MAILBOX_READING: u32 = 5;
const RESPONSE_RETENTION_SECS: u64 = 30;

/// One cache-line-separated response mailbox.
///
/// Only the client that successfully changes `state` from free to claimed may
/// assign the request id. The server becomes the sole response writer after it
/// changes claimed to processing. A release/acquire transition to response-ready
/// publishes the response back to that client.
#[repr(C, align(64))]
pub(crate) struct ResponseMailbox {
    state: AtomicU32,
    _padding: u32,
    request_id: AtomicU64,
    expires_at: AtomicU64,
    response: UnsafeCell<MaybeUninit<ShmResponse>>,
}

// Access to `response` is serialized by the atomic state machine above.
unsafe impl Sync for ResponseMailbox {}

impl ResponseMailbox {
    fn initialize(&mut self) {
        self.state = AtomicU32::new(MAILBOX_FREE);
        self._padding = 0;
        self.request_id = AtomicU64::new(0);
        self.expires_at = AtomicU64::new(0);
        self.response = UnsafeCell::new(MaybeUninit::uninit());
    }
}

/// Exclusive reservation of one response mailbox for a request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResponseMailboxClaim {
    index: u32,
    request_id: u64,
}

impl ResponseMailboxClaim {
    /// Zero-based mailbox index carried by [`crate::protocol::ShmRequest`].
    pub fn index(self) -> u32 {
        self.index
    }

    /// Request id that must match before the mailbox can be accessed.
    pub fn request_id(self) -> u64 {
        self.request_id
    }
}

/// Process-shared registry of request-specific response mailboxes.
#[derive(Clone)]
pub struct ResponseMailboxRegistry {
    shm: Arc<ShmManager>,
}

impl ResponseMailboxRegistry {
    /// Initialize every mailbox in a newly created shared-memory region.
    pub(crate) fn initialize(shm: &ShmManager) {
        for index in 0..shm.response_mailbox_count() {
            // SAFETY: `ShmManager::create` calls this before publishing the
            // region to clients, and the validated mailbox region is writable.
            unsafe {
                let mailbox = &mut *mailbox_ptr(shm, index);
                mailbox.initialize();
            }
        }
    }

    /// Connect to the mailbox registry owned by `shm`.
    pub fn connect(shm: Arc<ShmManager>) -> Self {
        Self { shm }
    }

    /// Reserve a mailbox for `request_id`.
    pub fn claim(&self, request_id: u64) -> Option<ResponseMailboxClaim> {
        if request_id == 0 {
            return None;
        }
        let count = self.shm.response_mailbox_count();
        if count == 0 {
            return None;
        }
        let start = request_id as usize % count;
        for distance in 0..count {
            let index = (start + distance) % count;
            let mailbox = self.mailbox(index)?;
            let state = mailbox.state.load(Ordering::Acquire);
            let claimed = state == MAILBOX_FREE
                && mailbox
                    .state
                    .compare_exchange(
                        MAILBOX_FREE,
                        MAILBOX_CLAIMED,
                        Ordering::AcqRel,
                        Ordering::Acquire,
                    )
                    .is_ok();
            let reclaimed = state == MAILBOX_RESPONSE_READY
                && mailbox.expires_at.load(Ordering::Acquire) <= unix_seconds()
                && mailbox
                    .state
                    .compare_exchange(
                        MAILBOX_RESPONSE_READY,
                        MAILBOX_CLAIMED,
                        Ordering::AcqRel,
                        Ordering::Acquire,
                    )
                    .is_ok();
            if claimed || reclaimed {
                mailbox.request_id.store(request_id, Ordering::Release);
                mailbox.expires_at.store(
                    unix_seconds().saturating_add(RESPONSE_RETENTION_SECS),
                    Ordering::Release,
                );
                return Some(ResponseMailboxClaim {
                    index: index as u32,
                    request_id,
                });
            }
        }
        None
    }

    /// Return a claim that was never published to the request queue.
    pub fn abort(&self, claim: ResponseMailboxClaim) -> bool {
        let Some(mailbox) = self.mailbox(claim.index as usize) else {
            return false;
        };
        if mailbox.request_id.load(Ordering::Acquire) != claim.request_id {
            return false;
        }
        if mailbox
            .state
            .compare_exchange(
                MAILBOX_CLAIMED,
                MAILBOX_RELEASING,
                Ordering::AcqRel,
                Ordering::Acquire,
            )
            .is_ok()
        {
            clear_and_free(mailbox);
            true
        } else {
            false
        }
    }

    /// Transfer a claimed mailbox to the server task handling the request.
    pub fn begin_processing(&self, index: u32, request_id: u64) -> bool {
        let Some(mailbox) = self.mailbox(index as usize) else {
            return false;
        };
        if mailbox.request_id.load(Ordering::Acquire) != request_id
            || mailbox
                .state
                .compare_exchange(
                    MAILBOX_CLAIMED,
                    MAILBOX_PROCESSING,
                    Ordering::AcqRel,
                    Ordering::Acquire,
                )
                .is_err()
        {
            return false;
        }
        // Processing requests are never reclaimed by another client. The
        // server will eventually publish a fresh response retention deadline.
        mailbox.expires_at.store(0, Ordering::Release);
        true
    }

    /// Release a processing mailbox when its server task unwinds before it can
    /// publish a response. The original client will time out, but the failure
    /// cannot permanently consume region capacity.
    pub(crate) fn abandon_processing(&self, index: u32, request_id: u64) -> bool {
        let Some(mailbox) = self.mailbox(index as usize) else {
            return false;
        };
        if mailbox.request_id.load(Ordering::Acquire) != request_id {
            return false;
        }
        if mailbox
            .state
            .compare_exchange(
                MAILBOX_PROCESSING,
                MAILBOX_RELEASING,
                Ordering::AcqRel,
                Ordering::Acquire,
            )
            .is_ok()
        {
            clear_and_free(mailbox);
            true
        } else {
            false
        }
    }

    /// Publish a response to the request that owns `index`.
    pub fn publish(&self, index: u32, request_id: u64, response: ShmResponse) -> bool {
        let Some(mailbox) = self.mailbox(index as usize) else {
            return false;
        };
        if mailbox.request_id.load(Ordering::Acquire) != request_id
            || mailbox.state.load(Ordering::Acquire) != MAILBOX_PROCESSING
        {
            return false;
        }

        // SAFETY: the processing state grants the server exclusive write
        // access. The release store below publishes the initialized value.
        unsafe { (*mailbox.response.get()).write(response) };
        mailbox.expires_at.store(
            unix_seconds().saturating_add(RESPONSE_RETENTION_SECS),
            Ordering::Release,
        );
        mailbox
            .state
            .store(MAILBOX_RESPONSE_READY, Ordering::Release);
        true
    }

    /// Take a ready response without consuming another request's mailbox.
    ///
    /// A successful call must be followed by [`release`](Self::release).
    pub fn try_take(&self, claim: ResponseMailboxClaim) -> Option<ShmResponse> {
        let mailbox = self.mailbox(claim.index as usize)?;
        if mailbox.request_id.load(Ordering::Acquire) != claim.request_id {
            return None;
        }
        mailbox
            .state
            .compare_exchange(
                MAILBOX_RESPONSE_READY,
                MAILBOX_READING,
                Ordering::AcqRel,
                Ordering::Acquire,
            )
            .ok()?;

        // SAFETY: acquiring RESPONSE_READY guarantees initialization, and the
        // READING state prevents reclamation or a new server write.
        Some(unsafe { (*mailbox.response.get()).assume_init_read() })
    }

    /// Release a response mailbox after its owner copied the response metadata.
    pub fn release(&self, claim: ResponseMailboxClaim) -> bool {
        let Some(mailbox) = self.mailbox(claim.index as usize) else {
            return false;
        };
        if mailbox.request_id.load(Ordering::Acquire) != claim.request_id {
            return false;
        }
        if mailbox
            .state
            .compare_exchange(
                MAILBOX_READING,
                MAILBOX_RELEASING,
                Ordering::AcqRel,
                Ordering::Acquire,
            )
            .is_ok()
        {
            clear_and_free(mailbox);
            true
        } else {
            false
        }
    }

    fn mailbox(&self, index: usize) -> Option<&ResponseMailbox> {
        if index >= self.shm.response_mailbox_count() {
            return None;
        }
        // SAFETY: the region and index were validated when ShmManager connected.
        Some(unsafe { &*mailbox_ptr(self.shm.as_ref(), index) })
    }
}

fn clear_and_free(mailbox: &ResponseMailbox) {
    mailbox.request_id.store(0, Ordering::Relaxed);
    mailbox.expires_at.store(0, Ordering::Relaxed);
    mailbox.state.store(MAILBOX_FREE, Ordering::Release);
}

fn unix_seconds() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn mailbox_ptr(shm: &ShmManager, index: usize) -> *mut ResponseMailbox {
    // SAFETY is carried by callers: `index` must be in the advertised registry.
    unsafe {
        shm.as_ptr()
            .add(shm.response_mailbox_offset())
            .cast::<ResponseMailbox>()
            .add(index)
    }
}

pub(crate) const fn mailbox_bytes(count: usize) -> usize {
    std::mem::size_of::<ResponseMailbox>() * count
}

pub(crate) const fn mailbox_alignment() -> usize {
    std::mem::align_of::<ResponseMailbox>()
}

#[cfg(test)]
mod tests;
