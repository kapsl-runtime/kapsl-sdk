use super::*;
use kapsl_engine_api::CancellationToken;

struct DropEvent {
    events: Arc<Mutex<Vec<(i32, &'static str)>>>,
    slot: i32,
}

impl Drop for DropEvent {
    fn drop(&mut self) {
        self.events.lock().unwrap().push((self.slot, "drop"));
    }
}

type Reply = std_mpsc::Receiver<Result<Vec<u8>, EngineError>>;

fn prefill(
    slot: i32,
    events: &Arc<Mutex<Vec<(i32, &'static str)>>>,
) -> (PendingPrefill, CancellationToken, Reply) {
    let cancellation = CancellationToken::new();
    let admission = RequestAdmission {
        memory: None,
        sequence: Some(GgufSequenceAdmission::new({
            let events = Arc::clone(events);
            move |assigned| {
                assert_eq!(assigned, slot as u32);
                Ok(DropEvent {
                    events: Arc::clone(&events),
                    slot,
                })
            }
        })),
        cancellation: Some(cancellation.clone()),
    };
    let (tx, rx) = std_mpsc::channel();
    (
        PendingPrefill {
            seq_id: slot,
            tokens: vec![LlamaToken::new(1), LlamaToken::new(2)],
            next_token: 0,
            max_tokens: 4,
            min_tokens: 0,
            session_id: None,
            memory_guard: Some(admission.acquire(slot as u32).unwrap()),
            response: GgufResponse::Final(tx),
            copies: Vec::new(),
        },
        cancellation,
        rx,
    )
}

#[test]
fn cancelled_prefill_leader_releases_its_binding_and_preserves_followers() {
    let events = Arc::new(Mutex::new(Vec::new()));
    let (mut leader, cancel_leader, leader_reply) = prefill(0, &events);
    let (copy, cancel_copy, copy_reply) = prefill(1, &events);
    let (other, _, other_reply) = prefill(2, &events);
    let mut pending = VecDeque::from([copy, other]);
    coalesce_exact_prompt_copies(&mut leader, &mut pending);
    assert!(pending.is_empty());
    assert_eq!(leader.copies.len(), 2);
    leader.next_token = 1;
    cancel_leader.cancel();
    let mut promoted = prune_cancelled_prefill(leader, |slot| {
        events.lock().unwrap().push((slot, "release"))
    })
    .unwrap();
    assert_eq!(promoted.seq_id, 1);
    assert_eq!(promoted.next_token, 0);
    assert_eq!(promoted.copies.len(), 1);
    assert_eq!(*events.lock().unwrap(), [(0, "release"), (0, "drop")]);
    assert!(matches!(
        leader_reply.try_recv().unwrap(),
        Err(EngineError::Cancelled { .. })
    ));
    assert!(matches!(
        copy_reply.try_recv(),
        Err(std_mpsc::TryRecvError::Empty)
    ));
    assert!(matches!(
        other_reply.try_recv(),
        Err(std_mpsc::TryRecvError::Empty)
    ));

    // A second cancellation does not reuse the old leader's owner or cancel
    // the remaining follower. The exact prompt and its admission survive.
    promoted.next_token = 2;
    cancel_copy.cancel();
    let survivor = prune_cancelled_prefill(promoted, |slot| {
        events.lock().unwrap().push((slot, "release"))
    })
    .unwrap();
    assert_eq!(survivor.seq_id, 2);
    assert_eq!(survivor.next_token, 0);
    assert_eq!(survivor.tokens, [LlamaToken::new(1), LlamaToken::new(2)]);
    assert!(matches!(
        copy_reply.try_recv().unwrap(),
        Err(EngineError::Cancelled { .. })
    ));
    assert!(matches!(
        other_reply.try_recv(),
        Err(std_mpsc::TryRecvError::Empty)
    ));
    assert_eq!(
        *events.lock().unwrap(),
        [(0, "release"), (0, "drop"), (1, "release"), (1, "drop")]
    );
}

#[test]
fn cancelled_copy_does_not_restart_or_release_the_prefill_leader() {
    let events = Arc::new(Mutex::new(Vec::new()));
    let (mut leader, _, leader_reply) = prefill(0, &events);
    let (copy, cancel_copy, copy_reply) = prefill(1, &events);
    coalesce_exact_prompt_copies(&mut leader, &mut VecDeque::from([copy]));
    leader.next_token = 1;
    cancel_copy.cancel();
    let leader = prune_cancelled_prefill(leader, |slot| {
        events.lock().unwrap().push((slot, "release"))
    })
    .unwrap();
    assert_eq!(leader.seq_id, 0);
    assert_eq!(leader.next_token, 1);
    assert!(leader.copies.is_empty());
    assert_eq!(*events.lock().unwrap(), [(1, "release"), (1, "drop")]);
    assert!(matches!(
        copy_reply.try_recv().unwrap(),
        Err(EngineError::Cancelled { .. })
    ));
    assert!(matches!(
        leader_reply.try_recv(),
        Err(std_mpsc::TryRecvError::Empty)
    ));
}
