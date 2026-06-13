#[cfg(test)]
mod tests {
    use super::super::{new_shared_allocator, BlockAllocator, BlockManager, PhysicalTokenBlock};
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    #[test]
    fn allocator_allocates_and_frees_blocks() {
        let mut allocator = BlockAllocator::new(2, 16, 7);
        assert_eq!(allocator.get_num_total_blocks(), 2);
        assert_eq!(allocator.get_num_free_blocks(), 2);

        let first = allocator.allocate().expect("expected first block");
        assert_eq!(
            first,
            PhysicalTokenBlock {
                block_number: 0,
                block_size: 16,
                device_id: 7,
            }
        );
        let second = allocator.allocate().expect("expected second block");
        assert_eq!(second.block_number, 1);
        assert_eq!(allocator.get_num_free_blocks(), 0);
        assert!(allocator.allocate().is_none());

        allocator.free(first);
        assert_eq!(allocator.get_num_free_blocks(), 1);
        let recycled = allocator.allocate().expect("expected recycled block");
        assert_eq!(recycled.block_number, 0);
    }

    #[test]
    fn block_manager_tracks_block_tables_and_capacity() {
        let mut manager = BlockManager::new(3, 16, 0);
        assert!(manager.can_allocate(3));

        manager.allocate(10).expect("block for seq 10");
        manager.allocate(10).expect("second block for seq 10");
        manager.allocate(11).expect("block for seq 11");

        let table_10 = manager.get_block_table(10).expect("table for seq 10");
        assert_eq!(table_10.len(), 2);
        let table_11 = manager.get_block_table(11).expect("table for seq 11");
        assert_eq!(table_11.len(), 1);
        assert!(!manager.can_allocate(1));

        manager.free(10);
        assert!(manager.get_block_table(10).is_none());
        assert!(manager.can_allocate(2));
    }

    #[test]
    fn live_cap_hard_limits_allocation_below_pool_free() {
        // Shared pool with 8 blocks, but this engine's fair-share cap is 2.
        let pool = new_shared_allocator(8, 16, 0);
        let mut manager = BlockManager::new_shared(pool, 16);
        let cap = Arc::new(AtomicUsize::new(2));
        manager.set_live_cap(cap.clone());

        // Pool has plenty free, but the quota allows only 2 blocks.
        assert!(manager.can_allocate(2));
        assert!(!manager.can_allocate(3), "quota must cap below pool free");

        assert!(manager.allocate(1).is_some());
        assert!(manager.allocate(1).is_some());
        assert_eq!(manager.held_blocks(), 2);
        // At the cap now: further allocation is refused despite free pool blocks.
        assert!(manager.allocate(1).is_none());
        assert!(manager.free_blocks() >= 6, "pool still has free blocks");

        // Freeing returns headroom under the cap.
        manager.free(1);
        assert_eq!(manager.held_blocks(), 0);
        assert!(manager.allocate(2).is_some());

        // Runtime raises the cap → more headroom without a restart.
        cap.store(5, Ordering::Relaxed);
        for _ in 0..4 {
            assert!(manager.allocate(2).is_some());
        }
        assert_eq!(manager.held_blocks(), 5);
        assert!(manager.allocate(2).is_none(), "capped at the new ceiling");
    }
}
