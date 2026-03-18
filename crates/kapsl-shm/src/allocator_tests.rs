#[cfg(test)]
mod tests {
    use super::super::{
        ModelSubPoolConfig, PerModelShmAllocator, ShmClassBudget, ShmPoolAllocator,
        SimpleShmAllocator, TieredShmAllocator,
    };
    use std::thread;
    use std::time::Duration;

    #[test]
    fn test_try_allocate_rejects_oversized_payload() {
        let allocator = SimpleShmAllocator::new(100, 64, 3);
        assert!(allocator.try_allocate(65).is_none());
    }

    #[test]
    fn test_try_allocate_returns_none_when_pool_is_full() {
        let allocator = SimpleShmAllocator::new(100, 64, 2);

        assert!(allocator.try_allocate(16).is_some());
        assert!(allocator.try_allocate(16).is_some());
        assert!(allocator.try_allocate(16).is_none());
    }

    #[test]
    fn test_release_makes_slot_available_immediately() {
        let allocator = SimpleShmAllocator::new(100, 64, 1);
        let first = allocator.try_allocate(32).expect("first slot");
        assert!(allocator.try_allocate(32).is_none());

        assert!(allocator.release(first));
        let second = allocator
            .try_allocate(32)
            .expect("released slot should be reusable");
        assert_eq!(first, second);
    }

    #[test]
    fn test_expired_lease_is_reused() {
        let allocator = SimpleShmAllocator::new_with_ttl(256, 32, 1, Duration::from_millis(20));
        let first = allocator.try_allocate(8).expect("slot");
        assert!(allocator.try_allocate(8).is_none());
        thread::sleep(Duration::from_millis(30));
        let second = allocator
            .try_allocate(8)
            .expect("expired lease should be reusable");
        assert_eq!(first, second);
    }

    #[test]
    fn test_release_rejects_invalid_offsets() {
        let allocator = SimpleShmAllocator::new(100, 64, 2);
        assert!(!allocator.release(99));
        assert!(!allocator.release(101));
        assert!(!allocator.release(100 + 2 * 64));
    }

    #[test]
    fn test_snapshot_reports_in_use_and_oldest_lease() {
        let allocator = SimpleShmAllocator::new_with_ttl(100, 64, 2, Duration::from_millis(100));
        let a = allocator.try_allocate(16).expect("slot a");
        thread::sleep(Duration::from_millis(10));
        let _b = allocator.try_allocate(16).expect("slot b");

        let snap = allocator.snapshot();
        assert_eq!(snap.in_use_slots, 2);
        assert!(snap.oldest_lease_ms >= 5);

        assert!(allocator.release(a));
        let snap2 = allocator.snapshot();
        assert_eq!(snap2.in_use_slots, 1);
    }

    #[test]
    fn test_tiered_allocator_chooses_smallest_fitting_class() {
        let allocator = TieredShmAllocator::new_with_class_sizes(
            1024,
            16 * 1024 * 1024,
            &[64, 256, 1024],
            Duration::from_secs(1),
        );
        let classes = allocator.classes();
        assert!(classes.len() >= 2);

        let small = allocator
            .try_allocate(32)
            .expect("small payload should fit smallest class");
        let medium = allocator
            .try_allocate(200)
            .expect("medium payload should fit medium class");

        let small_class = classes
            .iter()
            .find(|c| small >= c.base_offset && small < c.base_offset + c.bytes)
            .expect("small offset must belong to one class");
        let medium_class = classes
            .iter()
            .find(|c| medium >= c.base_offset && medium < c.base_offset + c.bytes)
            .expect("medium offset must belong to one class");

        assert!(small_class.slot_size <= medium_class.slot_size);
    }

    #[test]
    fn test_tiered_allocator_release_roundtrip() {
        let allocator = TieredShmAllocator::new_with_class_sizes(
            4096,
            8 * 1024 * 1024,
            &[128, 512],
            Duration::from_secs(1),
        );
        let classes = allocator.classes();
        let off = allocator
            .try_allocate(100)
            .expect("expected allocation in first class");
        assert!(allocator.release(off));
        let off2 = allocator
            .try_allocate(100)
            .expect("released slot should be reusable");
        let class_slot_size = |offset: usize| {
            classes
                .iter()
                .find(|c| offset >= c.base_offset && offset < c.base_offset + c.bytes)
                .map(|c| c.slot_size)
        };
        assert_eq!(class_slot_size(off), class_slot_size(off2));
    }

    #[test]
    fn test_tiered_allocator_snapshot_aggregates_classes() {
        let allocator = TieredShmAllocator::new_with_class_sizes(
            4096,
            8 * 1024 * 1024,
            &[64, 256],
            Duration::from_secs(5),
        );
        let _a = allocator.try_allocate(32).expect("slot a");
        let _b = allocator.try_allocate(200).expect("slot b");
        let snap = allocator.snapshot();
        assert_eq!(snap.in_use_slots, 2);
        assert!(snap.oldest_lease_ms < 5_000);
    }

    #[test]
    fn test_tiered_allocator_weighted_budgets_bias_capacity() {
        let budgets = [
            ShmClassBudget {
                slot_size: 256,
                weight: 1,
            },
            ShmClassBudget {
                slot_size: 4096,
                weight: 8,
            },
        ];
        let allocator = TieredShmAllocator::new_with_class_budgets(
            0,
            2 * 1024 * 1024,
            &budgets,
            Duration::from_secs(1),
        );
        let classes = allocator.classes();
        let small = classes
            .iter()
            .find(|c| c.slot_size == 256)
            .expect("small class must exist");
        let large = classes
            .iter()
            .find(|c| c.slot_size == 4096)
            .expect("large class must exist");
        assert!(large.bytes > small.bytes);
    }

    // ── PerModelShmAllocator tests ────────────────────────────────────────────

    fn make_pool(
        base: usize,
        total: usize,
        model_bytes: &[(u32, usize)],
    ) -> PerModelShmAllocator {
        let configs: Vec<ModelSubPoolConfig> = model_bytes
            .iter()
            .map(|(id, bytes)| ModelSubPoolConfig {
                model_id: *id,
                pool_bytes: *bytes,
                class_budgets: vec![
                    ShmClassBudget {
                        slot_size: 256,
                        weight: 1,
                    },
                    ShmClassBudget {
                        slot_size: 1024,
                        weight: 1,
                    },
                ],
            })
            .collect();
        PerModelShmAllocator::new(base, total, configs, Duration::from_secs(30))
    }

    #[test]
    fn test_per_model_allocator_routes_to_model_pool() {
        // 2 MiB total: 512 KiB for model 1, 512 KiB for model 2, rest shared.
        let pool = make_pool(0, 2 * 1024 * 1024, &[(1, 512 * 1024), (2, 512 * 1024)]);

        let off1 = pool.try_allocate(1, 100).expect("model 1 slot");
        let off2 = pool.try_allocate(2, 100).expect("model 2 slot");

        // The two offsets must be in different regions (model 1 starts at 0, model 2 at 512 KiB).
        assert!(off1 < 512 * 1024, "model 1 offset should be in its sub-pool");
        assert!(
            off2 >= 512 * 1024 && off2 < 1024 * 1024,
            "model 2 offset should be in its sub-pool"
        );
    }

    #[test]
    fn test_per_model_allocator_falls_back_to_shared_pool() {
        // Give model 1 a tiny pool (only one 256-byte slot possible) then exhaust it.
        let configs = vec![ModelSubPoolConfig {
            model_id: 1,
            pool_bytes: 256,
            class_budgets: vec![ShmClassBudget {
                slot_size: 256,
                weight: 1,
            }],
        }];
        let pool = PerModelShmAllocator::new(0, 4 * 1024 * 1024, configs, Duration::from_secs(30));

        // Exhaust the model pool.
        let _first = pool.try_allocate(1, 200).expect("first slot from model pool");

        // Second allocation must come from the shared pool (offset beyond model-1 range).
        let second = pool
            .try_allocate(1, 200)
            .expect("second slot should come from shared pool");
        assert!(second >= 256, "fallback offset must be in the shared pool");
    }

    #[test]
    fn test_per_model_allocator_release_by_offset() {
        // Use a single-slot model pool so release/reuse is deterministic.
        let configs = vec![ModelSubPoolConfig {
            model_id: 1,
            pool_bytes: 256,
            class_budgets: vec![ShmClassBudget {
                slot_size: 256,
                weight: 1,
            }],
        }];
        let pool = PerModelShmAllocator::new(0, 4 * 1024 * 1024, configs, Duration::from_secs(30));

        let off = pool.try_allocate(1, 200).expect("slot from model pool");
        assert!(pool.release(off), "release by offset should succeed");
        // With a single model-pool slot, the next allocation reuses the same offset.
        let off2 = pool.try_allocate(1, 200).expect("reused slot");
        assert_eq!(off, off2, "single-slot pool must reuse released offset");
    }

    #[test]
    fn test_per_model_allocator_unknown_model_uses_shared_pool() {
        let pool = make_pool(0, 4 * 1024 * 1024, &[(1, 512 * 1024)]);
        // model 99 has no dedicated pool — should still allocate from shared.
        let off = pool.try_allocate(99, 100).expect("shared pool slot for unknown model");
        assert!(off >= 512 * 1024, "unknown model must use shared pool");
    }

    #[test]
    fn test_per_model_allocator_layout_summary_contains_model_and_shared() {
        let pool = make_pool(0, 4 * 1024 * 1024, &[(3, 512 * 1024), (7, 512 * 1024)]);
        let summary = pool.layout_summary();
        assert!(summary.contains("model3:"), "summary should include model 3");
        assert!(summary.contains("model7:"), "summary should include model 7");
        assert!(summary.contains("shared:"), "summary should include shared pool");
    }

    #[test]
    fn test_per_model_allocator_aggregate_snapshot() {
        let pool = make_pool(0, 4 * 1024 * 1024, &[(1, 512 * 1024), (2, 512 * 1024)]);
        pool.try_allocate(1, 100).expect("slot in model 1");
        pool.try_allocate(2, 100).expect("slot in model 2");

        let snap = pool.snapshot();
        assert_eq!(snap.in_use_slots, 2);

        let m1 = pool.model_snapshot(1);
        assert_eq!(m1.in_use_slots, 1);
        let m2 = pool.model_snapshot(2);
        assert_eq!(m2.in_use_slots, 1);
    }
}
