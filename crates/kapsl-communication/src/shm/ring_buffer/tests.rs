use super::{LockFreeRingBuffer, QueueError};
use std::alloc::{alloc, dealloc, Layout};

#[test]
fn test_is_empty_transitions() {
    unsafe {
        let layout = Layout::array::<u64>(16).unwrap();
        let buffer = alloc(layout) as *mut u64;

        let queue = LockFreeRingBuffer::new(buffer, 16);
        assert!(queue.is_empty());
        assert_eq!(queue.len(), 0);

        queue.push(1).unwrap();
        assert!(!queue.is_empty());
        assert_eq!(queue.len(), 1);

        assert_eq!(queue.pop(), Some(1));
        assert!(queue.is_empty());
        assert_eq!(queue.len(), 0);

        dealloc(buffer as *mut u8, layout);
    }
}

#[test]
fn test_connect_reads_existing_state() {
    unsafe {
        let layout = Layout::array::<u64>(16).unwrap();
        let buffer = alloc(layout) as *mut u64;

        let queue = LockFreeRingBuffer::new(buffer, 16);
        queue.push(10).unwrap();
        queue.push(11).unwrap();

        let connected = LockFreeRingBuffer::connect(buffer, 16);
        assert_eq!(connected.pop(), Some(10));
        assert_eq!(connected.pop(), Some(11));
        assert_eq!(connected.pop(), None);

        dealloc(buffer as *mut u8, layout);
    }
}

#[test]
fn test_queue_full_returns_error() {
    unsafe {
        // capacity=7 → 3 control slots → 4 data slots → 3 max items (1 sentinel)
        let capacity = 7;
        let layout = Layout::array::<u64>(capacity).unwrap();
        let buffer = alloc(layout) as *mut u64;

        let queue = LockFreeRingBuffer::new(buffer, capacity);
        assert_eq!(queue.capacity(), capacity - 4);

        for i in 0..(capacity - 4) {
            queue.push(i as u64).unwrap();
        }

        let err = queue.push(999).unwrap_err();
        assert_eq!(err, QueueError::Full);

        dealloc(buffer as *mut u8, layout);
    }
}

#[test]
fn test_len_tracks_pushes_and_pops() {
    unsafe {
        let layout = Layout::array::<u64>(16).unwrap();
        let buffer = alloc(layout) as *mut u64;

        let queue = LockFreeRingBuffer::new(buffer, 16);
        assert_eq!(queue.len(), 0);

        queue.push(1).unwrap();
        queue.push(2).unwrap();
        assert_eq!(queue.len(), 2);

        queue.pop();
        assert_eq!(queue.len(), 1);

        queue.pop();
        assert_eq!(queue.len(), 0);

        dealloc(buffer as *mut u8, layout);
    }
}

#[test]
fn test_capacity_reported_correctly() {
    unsafe {
        let total = 20usize;
        let layout = Layout::array::<u64>(total).unwrap();
        let buffer = alloc(layout) as *mut u64;

        let queue = LockFreeRingBuffer::new(buffer, total);
        // 3 control slots + 1 sentinel = 4 overhead
        assert_eq!(queue.capacity(), total - 4);

        dealloc(buffer as *mut u8, layout);
    }
}

/// MPSC: two threads push concurrently, one thread pops all items.
#[test]
fn test_concurrent_mpsc() {
    use std::sync::Arc;
    use std::thread;

    const TOTAL: usize = 512;
    const ITEMS_PER_PRODUCER: u64 = 256;

    unsafe {
        let layout = Layout::array::<u64>(TOTAL).unwrap();
        let raw = alloc(layout) as *mut u64;

        // Use Arc<*mut u64> via a wrapper to share the raw pointer.
        // SAFETY: LockFreeRingBuffer is Send+Sync.
        let queue = Arc::new(LockFreeRingBuffer::new(raw, TOTAL));

        let q1 = Arc::clone(&queue);
        let t1 = thread::spawn(move || {
            for i in 0..ITEMS_PER_PRODUCER {
                loop {
                    if q1.push(i).is_ok() {
                        break;
                    }
                    std::hint::spin_loop();
                }
            }
        });

        let q2 = Arc::clone(&queue);
        let t2 = thread::spawn(move || {
            for i in 0..ITEMS_PER_PRODUCER {
                loop {
                    if q2.push(ITEMS_PER_PRODUCER + i).is_ok() {
                        break;
                    }
                    std::hint::spin_loop();
                }
            }
        });

        let mut received = Vec::with_capacity((2 * ITEMS_PER_PRODUCER) as usize);
        while (received.len() as u64) < 2 * ITEMS_PER_PRODUCER {
            if let Some(v) = queue.pop() {
                received.push(v);
            }
        }

        t1.join().unwrap();
        t2.join().unwrap();

        // All items from both producers must be present (order may differ).
        received.sort_unstable();
        assert_eq!(received.len(), (2 * ITEMS_PER_PRODUCER) as usize);
        for i in 0..ITEMS_PER_PRODUCER {
            assert!(received.contains(&i));
            assert!(received.contains(&(ITEMS_PER_PRODUCER + i)));
        }

        dealloc(raw as *mut u8, layout);
    }
}

/// Multiple producers and consumers must deliver every item exactly once.
#[test]
fn test_concurrent_mpmc() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};
    use std::thread;

    const BUFFER_SLOTS: usize = 128;
    const PRODUCERS: usize = 4;
    const CONSUMERS: usize = 4;
    const ITEMS_PER_PRODUCER: usize = 200;
    const TOTAL_ITEMS: usize = PRODUCERS * ITEMS_PER_PRODUCER;

    unsafe {
        let layout = Layout::array::<u64>(BUFFER_SLOTS).unwrap();
        let raw = alloc(layout) as *mut u64;
        let queue = Arc::new(LockFreeRingBuffer::new(raw, BUFFER_SLOTS));
        let remaining = Arc::new(AtomicUsize::new(TOTAL_ITEMS));
        let received = Arc::new(Mutex::new(Vec::with_capacity(TOTAL_ITEMS)));

        let mut consumers = Vec::new();
        for _ in 0..CONSUMERS {
            let queue = Arc::clone(&queue);
            let remaining = Arc::clone(&remaining);
            let received = Arc::clone(&received);
            consumers.push(thread::spawn(move || {
                while remaining.load(Ordering::Acquire) > 0 {
                    if let Some(value) = queue.pop() {
                        received
                            .lock()
                            .unwrap_or_else(|poison| poison.into_inner())
                            .push(value);
                        remaining.fetch_sub(1, Ordering::AcqRel);
                    } else {
                        thread::yield_now();
                    }
                }
            }));
        }

        let mut producers = Vec::new();
        for producer in 0..PRODUCERS {
            let queue = Arc::clone(&queue);
            producers.push(thread::spawn(move || {
                let start = producer * ITEMS_PER_PRODUCER;
                let end = start + ITEMS_PER_PRODUCER;
                for value in start..end {
                    while queue.push(value as u64).is_err() {
                        thread::yield_now();
                    }
                }
            }));
        }

        for producer in producers {
            producer.join().unwrap();
        }
        for consumer in consumers {
            consumer.join().unwrap();
        }

        let mut values = received
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
            .clone();
        values.sort_unstable();
        assert_eq!(values, (0..TOTAL_ITEMS as u64).collect::<Vec<_>>());

        drop(queue);
        dealloc(raw as *mut u8, layout);
    }
}
