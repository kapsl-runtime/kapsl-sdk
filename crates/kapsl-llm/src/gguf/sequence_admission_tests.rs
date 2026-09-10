use super::*;
use std::collections::HashMap;
use std::sync::{atomic::AtomicUsize, Barrier, Mutex};

struct Binding {
    slots: Arc<Mutex<HashMap<u32, u64>>>,
    slot: u32,
    request: u64,
}

impl Drop for Binding {
    fn drop(&mut self) {
        assert_eq!(
            self.slots.lock().unwrap().remove(&self.slot),
            Some(self.request)
        );
    }
}

fn admission(slots: &Arc<Mutex<HashMap<u32, u64>>>, request: u64) -> RequestAdmission {
    let slots = Arc::clone(slots);
    RequestAdmission {
        memory: None,
        sequence: Some(GgufSequenceAdmission::new(move |slot| {
            let mut assigned = slots.lock().unwrap();
            if assigned.contains_key(&slot) {
                return Err(EngineError::backend("slot belongs to another request"));
            }
            assigned.insert(slot, request);
            Ok(Binding {
                slots: Arc::clone(&slots),
                slot,
                request,
            })
        })),
        cancellation: Some(CancellationToken::new()),
    }
}

#[test]
fn concurrent_slots_keep_their_engine_request_ownership_until_retirement() {
    let slots = Arc::new(Mutex::new(HashMap::new()));
    let first = admission(&slots, 101);
    let second = admission(&slots, 202);
    let guard1 = first.acquire(0).unwrap();
    let guard2 = second.acquire(1).unwrap();
    assert_eq!(*slots.lock().unwrap(), HashMap::from([(0, 101), (1, 202)]));
    first.cancellation.as_ref().unwrap().cancel();
    assert!(guard1.is_cancelled());
    assert!(!guard2.is_cancelled());
    assert!(admission(&slots, 303).acquire(0).is_err());
    drop(guard1);
    let guard3 = admission(&slots, 303).acquire(0).unwrap();
    assert_eq!(*slots.lock().unwrap(), HashMap::from([(0, 303), (1, 202)]));
    drop(guard2);
    assert_eq!(*slots.lock().unwrap(), HashMap::from([(0, 303)]));
    drop(guard3);
    assert!(slots.lock().unwrap().is_empty());
}

#[test]
fn cloned_hook_can_bind_only_once_even_with_concurrent_acquisition() {
    let calls = Arc::new(AtomicUsize::new(0));
    let hook = GgufSequenceAdmission::new({
        let calls = Arc::clone(&calls);
        move |_| {
            calls.fetch_add(1, Ordering::AcqRel);
            Ok(())
        }
    });
    let barrier = Arc::new(Barrier::new(3));
    let workers: Vec<_> = (0..2)
        .map(|slot| {
            let hook = hook.clone();
            let barrier = Arc::clone(&barrier);
            std::thread::spawn(move || {
                barrier.wait();
                hook.acquire(slot).is_ok()
            })
        })
        .collect();
    barrier.wait();
    assert_eq!(
        workers
            .into_iter()
            .map(|t| usize::from(t.join().unwrap()))
            .sum::<usize>(),
        1
    );
    assert_eq!(calls.load(Ordering::Acquire), 1);
}

#[test]
fn queued_cancellation_cannot_acquire_memory_or_a_slot() {
    let slots = Arc::new(Mutex::new(HashMap::new()));
    let mut queued = admission(&slots, 101);
    queued.memory = Some(RequestMemoryAdmission::new(
        || -> Result<(), EngineError> {
            panic!("cancelled queued request tried to acquire memory")
        },
    ));
    queued.cancellation.as_ref().unwrap().cancel();
    assert!(matches!(
        queued.acquire(0),
        Err(EngineError::Cancelled { .. })
    ));
    assert!(slots.lock().unwrap().is_empty());
}

struct Lease(Arc<AtomicUsize>);
impl Drop for Lease {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::AcqRel);
    }
}

#[test]
fn failed_binding_returns_the_memory_lease_without_retrying_the_hook() {
    let leases = Arc::new(AtomicUsize::new(0));
    let memory = RequestMemoryAdmission::new({
        let leases = Arc::clone(&leases);
        move || {
            leases.fetch_add(1, Ordering::AcqRel);
            Ok(Lease(Arc::clone(&leases)))
        }
    });
    let sequence =
        GgufSequenceAdmission::new(|_| Err::<(), _>(EngineError::backend("admission rejected")));
    let request = RequestAdmission {
        memory: Some(memory),
        sequence: Some(sequence.clone()),
        cancellation: None,
    };
    assert!(request.acquire(0).is_err());
    assert_eq!(leases.load(Ordering::Acquire), 0);
    assert!(sequence
        .acquire(1)
        .err()
        .unwrap()
        .to_string()
        .contains("more than once"));
}

#[test]
fn cancellation_during_binding_releases_both_guards_before_compute() {
    let leases = Arc::new(AtomicUsize::new(0));
    let token = CancellationToken::new();
    let request = RequestAdmission {
        memory: Some(RequestMemoryAdmission::new({
            let leases = Arc::clone(&leases);
            move || {
                leases.fetch_add(1, Ordering::AcqRel);
                Ok(Lease(Arc::clone(&leases)))
            }
        })),
        sequence: Some(GgufSequenceAdmission::new({
            let leases = Arc::clone(&leases);
            let token = token.clone();
            move |_| {
                assert_eq!(leases.load(Ordering::Acquire), 1);
                leases.fetch_add(1, Ordering::AcqRel);
                token.cancel();
                Ok(Lease(Arc::clone(&leases)))
            }
        })),
        cancellation: Some(token),
    };
    assert!(matches!(
        request.acquire(0),
        Err(EngineError::Cancelled { .. })
    ));
    assert_eq!(leases.load(Ordering::Acquire), 0);
}
