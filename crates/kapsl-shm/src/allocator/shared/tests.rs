use super::*;
use std::collections::HashSet;
use std::sync::{Arc, Barrier};

fn test_region(label: &str) -> Option<(String, Arc<ShmManager>)> {
    let name = format!("/ks_{}_{}", label, std::process::id());
    match ShmManager::create(&name, 16 * 1024 * 1024) {
        Ok(manager) => Some((name, Arc::new(manager))),
        Err(ShmError::ShmemError(shared_memory::ShmemError::MapCreateFailed(_))) => None,
        Err(error) => panic!("create test SHM: {error}"),
    }
}

use crate::memory::ShmError;

#[test]
fn allocators_connected_to_one_region_never_overlap_live_slots() {
    let Some((name, shm)) = test_region("leases") else {
        return;
    };
    let workers = 16;
    let barrier = Arc::new(Barrier::new(workers));
    let mut threads = Vec::new();
    for _ in 0..workers {
        let name = name.clone();
        let barrier = barrier.clone();
        threads.push(std::thread::spawn(move || {
            let mapping = Arc::new(ShmManager::connect(&name).expect("connect client mapping"));
            let allocator = SharedShmAllocator::connect(mapping, Duration::from_secs(30));
            barrier.wait();
            allocator
                .try_allocate(1024)
                .expect("shared slot must be available")
        }));
    }

    let leases: Vec<_> = threads
        .into_iter()
        .map(|thread| thread.join().expect("allocation thread"))
        .collect();
    let offsets: HashSet<_> = leases.iter().map(|lease| lease.offset()).collect();
    assert_eq!(offsets.len(), workers);

    let allocator = SharedShmAllocator::connect(shm, Duration::from_secs(30));
    for lease in leases {
        assert!(allocator.release(lease));
    }
    assert_eq!(allocator.snapshot().in_use_slots, 0);
}

#[test]
fn stale_release_cannot_free_a_new_generation() {
    let Some((_name, shm)) = test_region("generation") else {
        return;
    };
    let allocator = SharedShmAllocator::connect(shm, Duration::from_secs(30));
    let first = allocator.try_allocate(1024).expect("first lease");
    assert!(allocator.release(first));
    let second = allocator.try_allocate(1024).expect("second lease");

    assert_eq!(first.offset(), second.offset());
    assert_ne!(first.token(), second.token());
    assert!(!allocator.release(first));
    assert!(allocator
        .validate(second.offset(), 1024, second.token())
        .is_some());
    assert!(allocator.release(second));
}

#[test]
fn acquiring_a_wire_lease_pins_the_exact_generation() {
    let Some((_name, shm)) = test_region("acquire") else {
        return;
    };
    let allocator = SharedShmAllocator::connect(shm, Duration::from_secs(30));
    let advertised = allocator.try_allocate(1024).expect("advertised lease");
    let acquired = allocator
        .acquire(advertised.offset(), 1024, advertised.token())
        .expect("acquire advertised lease");

    if acquired.token() != advertised.token() {
        assert!(!allocator.release(advertised));
    }
    assert!(allocator.release(acquired));
}

#[test]
fn layout_preserves_large_slots_without_sacrificing_small_tensor_concurrency() {
    let slots = shared_slot_layout(0, 128 * 1024 * 1024);
    assert!(slots
        .iter()
        .any(|slot| slot.payload_capacity >= 64 * 1024 * 1024 - LEASE_WORD_BYTES));
    assert!(
        slots
            .iter()
            .filter(|slot| slot.payload_capacity < 1024 * 1024)
            .count()
            > 16
    );
    let default_runtime_slots = shared_slot_layout(0, 256 * 1024 * 1024);
    assert!(default_runtime_slots
        .iter()
        .any(|slot| { slot.payload_capacity >= 128 * 1024 * 1024 - LEASE_WORD_BYTES }));
}
