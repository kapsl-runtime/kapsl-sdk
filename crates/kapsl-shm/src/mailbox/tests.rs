use super::*;
use crate::memory::{ShmError, ShmManager};
use kapsl_transport::ResponseMetadata;
use std::collections::HashSet;

fn test_registry(label: &str) -> Option<ResponseMailboxRegistry> {
    let name = format!("/km_{}_{}", label, std::process::id());
    match ShmManager::create(&name, 1024 * 1024) {
        Ok(manager) => Some(ResponseMailboxRegistry::connect(Arc::new(manager))),
        Err(ShmError::ShmemError(shared_memory::ShmemError::MapCreateFailed(_))) => None,
        Err(error) => panic!("create test SHM: {error}"),
    }
}

#[test]
fn response_is_visible_only_through_its_claim() {
    let Some(registry) = test_registry("route") else {
        return;
    };
    let claim = registry.claim(41).expect("claim mailbox");
    let other = registry.claim(42).expect("claim other mailbox");
    assert!(registry.begin_processing(claim.index(), claim.request_id()));
    let response = ShmResponse {
        metadata: ResponseMetadata::success(41, 10),
        result_offset: 200,
        result_size: 20,
        error_offset: 0,
        payload_lease: 77,
    };
    assert!(registry.publish(claim.index(), claim.request_id(), response));

    assert_eq!(registry.try_take(claim).unwrap().metadata.request_id, 41);
    assert!(registry.try_take(other).is_none());
    assert!(registry.release(claim));
    assert!(registry.abort(other));
}

#[test]
fn concurrent_claims_receive_distinct_mailboxes() {
    let Some(registry) = test_registry("claims") else {
        return;
    };
    let registry = Arc::new(registry);
    let threads: Vec<_> = (1..=64)
        .map(|request_id| {
            let registry = registry.clone();
            std::thread::spawn(move || registry.claim(request_id).expect("mailbox available"))
        })
        .collect();
    let claims: Vec<_> = threads
        .into_iter()
        .map(|thread| thread.join().expect("claim thread"))
        .collect();
    let indices: HashSet<_> = claims.iter().map(|claim| claim.index()).collect();
    assert_eq!(indices.len(), claims.len());
    for claim in claims {
        assert!(registry.abort(claim));
    }
}

#[test]
fn expired_unread_response_mailbox_can_be_reclaimed_safely() {
    let Some(registry) = test_registry("reclaim") else {
        return;
    };
    let first = registry.claim(7).expect("first claim");
    assert!(registry.begin_processing(first.index(), first.request_id()));
    let response = ShmResponse {
        metadata: ResponseMetadata::success(7, 1),
        result_offset: 0,
        result_size: 0,
        error_offset: 0,
        payload_lease: 0,
    };
    assert!(registry.publish(first.index(), first.request_id(), response));
    registry
        .mailbox(first.index() as usize)
        .unwrap()
        .expires_at
        .store(0, Ordering::Release);

    let replacement_id = 7 + RESPONSE_MAILBOX_COUNT as u64;
    let replacement = registry.claim(replacement_id).expect("reclaim mailbox");
    assert_eq!(replacement.index(), first.index());
    assert!(!registry.release(first));
    assert!(registry.abort(replacement));
}

#[test]
fn mailbox_routes_across_independent_mappings() {
    let name = format!("/km_maps_{}", std::process::id());
    let owner = match ShmManager::create(&name, 1024 * 1024) {
        Ok(manager) => Arc::new(manager),
        Err(ShmError::ShmemError(shared_memory::ShmemError::MapCreateFailed(_))) => return,
        Err(error) => panic!("create test SHM: {error}"),
    };
    let client_mapping = Arc::new(ShmManager::connect(&name).expect("connect client mapping"));
    let server = ResponseMailboxRegistry::connect(owner);
    let client = ResponseMailboxRegistry::connect(client_mapping);
    let claim = client.claim(88).expect("client claim");

    assert!(server.begin_processing(claim.index(), claim.request_id()));
    assert!(server.publish(
        claim.index(),
        claim.request_id(),
        ShmResponse {
            metadata: ResponseMetadata::success(88, 2),
            result_offset: 0,
            result_size: 0,
            error_offset: 0,
            payload_lease: 0,
        }
    ));
    assert_eq!(client.try_take(claim).unwrap().metadata.request_id, 88);
    assert!(client.release(claim));
}

#[test]
fn abandoned_processing_mailbox_returns_to_the_pool() {
    let Some(registry) = test_registry("abandon") else {
        return;
    };
    let claim = registry.claim(99).expect("claim mailbox");
    assert!(registry.begin_processing(claim.index(), claim.request_id()));
    assert!(registry.abandon_processing(claim.index(), claim.request_id()));

    let replacement_id = claim.request_id() + RESPONSE_MAILBOX_COUNT as u64;
    let replacement = registry.claim(replacement_id).expect("reclaim mailbox");
    assert_eq!(replacement.index(), claim.index());
    assert!(registry.abort(replacement));
}
