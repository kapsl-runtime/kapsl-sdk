//! Backend-neutral device-allocation scopes for ONNX generation.

use std::cell::Cell;

/// Purpose assigned to allocations made while a provider scope is active.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum DeviceAllocationClass {
    PersistentWeights,
    KvCache,
    TransientWorkspace,
    BlockTable,
    RequestTransient,
    Other,
}

/// Ownership cardinality for one provider execution scope.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum DeviceAllocationScopeKind {
    Model,
    Replica,
    Request,
    RequestBatch,
}

/// Complete logical context for allocations triggered by one synchronous
/// provider operation.
///
/// `scope_id` is non-zero and unique across the configured provider's
/// unload/reload lifecycle.
/// Request and request-batch scopes contain exactly one or at least two request
/// IDs respectively. Model and replica scopes contain none. The model/replica
/// pair remains the durable charge owner; request IDs describe which operation
/// triggered an allocation, while the allocator's returned identity controls
/// its actual lifetime.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DeviceAllocationScope {
    pub kind: DeviceAllocationScopeKind,
    pub scope_id: u64,
    pub device_id: u32,
    pub model_id: u32,
    pub replica_id: u32,
    pub allocation_class: DeviceAllocationClass,
    pub request_ids: Vec<u64>,
}

impl DeviceAllocationScope {
    pub fn is_well_formed(&self) -> bool {
        self.scope_id != 0
            && match self.kind {
                DeviceAllocationScopeKind::Model | DeviceAllocationScopeKind::Replica => {
                    self.request_ids.is_empty()
                }
                DeviceAllocationScopeKind::Request => self.request_ids.len() == 1,
                DeviceAllocationScopeKind::RequestBatch => self.request_ids.len() >= 2,
            }
    }
}

/// Type-erased RAII value that keeps a provider's allocation context active.
///
/// This deliberately does not require `Send`: thread-local provider scopes
/// must be entered and dropped around one synchronous operation on the same
/// thread.
pub trait DeviceAllocationScopeGuard {}

impl<T> DeviceAllocationScopeGuard for T {}

/// Adapter-supplied bridge from the published LLM engine to its provider's
/// allocation-context mechanism.
///
/// Implementations must fail closed when the scope cannot be established. The
/// returned guard remains alive until the synchronous provider operation has
/// completed and all allocations triggered by it have been observed.
pub trait DeviceAllocationScopeProvider: Send + Sync {
    fn enter(
        &self,
        scope: &DeviceAllocationScope,
    ) -> Result<Box<dyn DeviceAllocationScopeGuard>, String>;
}

thread_local! {
    static INFERENCE_REQUEST_ID: Cell<Option<u64>> = const { Cell::new(None) };
}

pub(crate) fn with_inference_request_id<T>(request_id: u64, operation: impl FnOnce() -> T) -> T {
    struct Restore(Option<u64>);

    impl Drop for Restore {
        fn drop(&mut self) {
            INFERENCE_REQUEST_ID.with(|current| current.set(self.0));
        }
    }

    let previous = INFERENCE_REQUEST_ID.with(|current| current.replace(Some(request_id)));
    let _restore = Restore(previous);
    operation()
}

pub(crate) fn inference_request_id() -> Option<u64> {
    INFERENCE_REQUEST_ID.with(Cell::get)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_id_scope_restores_nested_context() {
        assert_eq!(inference_request_id(), None);
        with_inference_request_id(11, || {
            assert_eq!(inference_request_id(), Some(11));
            with_inference_request_id(22, || {
                assert_eq!(inference_request_id(), Some(22));
            });
            assert_eq!(inference_request_id(), Some(11));
        });
        assert_eq!(inference_request_id(), None);
    }

    #[test]
    fn request_id_scope_restores_context_after_panic() {
        let result = std::panic::catch_unwind(|| {
            with_inference_request_id(11, || panic!("scope test"));
        });
        assert!(result.is_err());
        assert_eq!(inference_request_id(), None);
    }

    #[test]
    fn ownership_cardinality_is_explicit() {
        let mut scope = DeviceAllocationScope {
            kind: DeviceAllocationScopeKind::Request,
            scope_id: 1,
            device_id: 0,
            model_id: 7,
            replica_id: 3,
            allocation_class: DeviceAllocationClass::TransientWorkspace,
            request_ids: vec![42],
        };
        assert!(scope.is_well_formed());
        scope.kind = DeviceAllocationScopeKind::RequestBatch;
        assert!(!scope.is_well_formed());
        scope.request_ids.push(43);
        assert!(scope.is_well_formed());
    }
}
