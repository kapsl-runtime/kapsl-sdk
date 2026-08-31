use super::*;
use async_trait::async_trait;
use kapsl_engine_api::{EngineError, EngineMetrics, EngineStream};
use kapsl_transport::RequestMetadata;
use std::sync::Mutex;

struct EchoScheduler;

#[async_trait]
impl ReplicaScheduler for EchoScheduler {
    fn get_queue_depth(&self) -> (usize, usize) {
        (0, 0)
    }

    fn is_healthy(&self) -> bool {
        true
    }

    fn get_metrics(&self) -> EngineMetrics {
        EngineMetrics::default()
    }

    async fn infer(
        &self,
        request: &InferenceRequest,
        _priority: Priority,
        _force_cpu: bool,
    ) -> Result<BinaryTensorPacket, EngineError> {
        tokio::task::yield_now().await;
        Ok(request.input.clone())
    }

    async fn infer_stream(
        &self,
        _request: InferenceRequest,
        _priority: Priority,
        _force_cpu: bool,
    ) -> Result<EngineStream, EngineError> {
        Err(EngineError::backend("streaming not supported in tests"))
    }
}

fn test_region(label: &str, size: usize) -> Option<Arc<ShmManager>> {
    let name = format!("/ksv_{}_{}", label, std::process::id());
    match ShmManager::create(&name, size) {
        Ok(manager) => Some(Arc::new(manager)),
        Err(crate::memory::ShmError::ShmemError(shared_memory::ShmemError::MapCreateFailed(_))) => {
            None
        }
        Err(error) => panic!("create test SHM: {error}"),
    }
}

fn scheduler_lookup() -> SchedulerLookup {
    let scheduler: Arc<dyn ReplicaScheduler + Send + Sync> = Arc::new(EchoScheduler);
    Arc::new(move |model_id| (model_id == 1).then(|| scheduler.clone()))
}

fn request_for(
    request_id: u64,
    mailbox: crate::mailbox::ResponseMailboxClaim,
    lease: crate::allocator::SharedShmLease,
    encoded_size: usize,
) -> ShmRequest {
    ShmRequest {
        metadata: RequestMetadata::new(request_id, 1, 0, false),
        tensor_offset: lease.offset() as u64,
        tensor_size: encoded_size as u64,
        tensor_lease: lease.token(),
        response_mailbox: mailbox.index(),
        protocol_version: SHM_PROTOCOL_VERSION,
        _padding: [0; 2],
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn concurrent_requests_keep_payloads_and_responses_isolated() {
    const REQUESTS: usize = 32;
    let Some(shm) = test_region("concurrent", 64 * 1024 * 1024) else {
        return;
    };
    let allocator = Arc::new(SharedShmAllocator::connect(
        shm.clone(),
        TENSOR_SLOT_LEASE_TTL,
    ));
    let mailboxes = ResponseMailboxRegistry::connect(shm.clone());
    let lookup = scheduler_lookup();
    let mut claims = Vec::with_capacity(REQUESTS);
    let mut tasks = Vec::with_capacity(REQUESTS);

    for value in 0..REQUESTS as u32 {
        let request_id = shm.next_request_id();
        let claim = mailboxes.claim(request_id).expect("response mailbox");
        let packet = BinaryTensorPacket {
            shape: vec![1],
            dtype: TensorDtype::Int32,
            data: value.to_ne_bytes().to_vec(),
        };
        let (lease, encoded_size) =
            stage_response_tensor(&shm, &allocator, &packet).expect("stage request tensor");
        let request = request_for(request_id, claim, lease, encoded_size);
        claims.push((claim, value));

        let shm = shm.clone();
        let allocator = allocator.clone();
        let mailboxes = mailboxes.clone();
        let lookup = lookup.clone();
        tasks.push(tokio::spawn(async move {
            handle_request(request, shm, allocator, mailboxes, lookup, None).await;
        }));
    }

    for task in tasks {
        task.await.expect("request task");
    }

    for (claim, expected) in claims {
        let response = mailboxes.try_take(claim).expect("routed response");
        assert_eq!(response.metadata.request_id, claim.request_id());
        assert!(response.metadata.is_success());
        let response_request = ShmRequest {
            metadata: RequestMetadata::new(claim.request_id(), 1, 0, false),
            tensor_offset: response.result_offset,
            tensor_size: response.result_size,
            tensor_lease: response.payload_lease,
            response_mailbox: claim.index(),
            protocol_version: SHM_PROTOCOL_VERSION,
            _padding: [0; 2],
        };
        let packet =
            take_request_tensor(&shm, &allocator, &response_request).expect("read response tensor");
        assert_eq!(packet.data, expected.to_ne_bytes());
        assert!(mailboxes.release(claim));
    }
    assert_eq!(allocator.snapshot().in_use_slots, 0);
}

#[tokio::test]
async fn invalid_input_lease_returns_an_error_to_the_owning_mailbox() {
    let Some(shm) = test_region("invalid", 4 * 1024 * 1024) else {
        return;
    };
    let allocator = Arc::new(SharedShmAllocator::connect(
        shm.clone(),
        TENSOR_SLOT_LEASE_TTL,
    ));
    let mailboxes = ResponseMailboxRegistry::connect(shm.clone());
    let request_id = shm.next_request_id();
    let claim = mailboxes.claim(request_id).expect("mailbox");
    let request = ShmRequest {
        metadata: RequestMetadata::new(request_id, 1, 0, false),
        tensor_offset: shm.tensor_pool_offset() as u64,
        tensor_size: std::mem::size_of::<TensorHeader>() as u64,
        tensor_lease: 123,
        response_mailbox: claim.index(),
        protocol_version: SHM_PROTOCOL_VERSION,
        _padding: [0; 2],
    };

    handle_request(
        request,
        shm,
        allocator.clone(),
        mailboxes.clone(),
        scheduler_lookup(),
        None,
    )
    .await;
    let response = mailboxes.try_take(claim).expect("error response");
    assert!(!response.metadata.is_success());
    assert_eq!(response.metadata.request_id, request_id);
    if response.error_offset > 0 {
        assert!(allocator.release_wire(response.error_offset as usize, response.payload_lease));
    }
    assert!(mailboxes.release(claim));
}

#[tokio::test]
async fn stale_protocol_returns_a_routed_error_and_releases_input() {
    let Some(shm) = test_region("stale", 4 * 1024 * 1024) else {
        return;
    };
    let allocator = Arc::new(SharedShmAllocator::connect(
        shm.clone(),
        TENSOR_SLOT_LEASE_TTL,
    ));
    let mailboxes = ResponseMailboxRegistry::connect(shm.clone());
    let request_id = shm.next_request_id();
    let claim = mailboxes.claim(request_id).expect("mailbox");
    let input = allocator.try_allocate(128).expect("input lease");
    let request = ShmRequest {
        metadata: RequestMetadata::new(request_id, 1, 0, false),
        tensor_offset: input.offset() as u64,
        tensor_size: 128,
        tensor_lease: input.token(),
        response_mailbox: claim.index(),
        protocol_version: SHM_PROTOCOL_VERSION - 1,
        _padding: [0; 2],
    };

    handle_request(
        request,
        shm,
        allocator.clone(),
        mailboxes.clone(),
        scheduler_lookup(),
        None,
    )
    .await;

    assert!(allocator
        .validate(input.offset(), 128, input.token())
        .is_none());
    let response = mailboxes.try_take(claim).expect("version error response");
    assert!(!response.metadata.is_success());
    if response.error_offset > 0 {
        assert!(allocator.release_wire(response.error_offset as usize, response.payload_lease));
    }
    assert!(mailboxes.release(claim));
}

#[test]
fn server_lookup_observes_hot_loaded_scheduler() {
    let schedulers = Arc::new(Mutex::new(HashMap::<
        u32,
        Arc<dyn ReplicaScheduler + Send + Sync>,
    >::new()));
    let lookup: SchedulerLookup = {
        let schedulers = schedulers.clone();
        Arc::new(move |model_id| {
            schedulers
                .lock()
                .unwrap_or_else(|poison| poison.into_inner())
                .get(&model_id)
                .cloned()
        })
    };
    let snapshot: SchedulerSnapshot = {
        let schedulers = schedulers.clone();
        Arc::new(move || {
            schedulers
                .lock()
                .unwrap_or_else(|poison| poison.into_inner())
                .clone()
        })
    };
    let server = ShmServer::new_with_lookup("/unused", 1024 * 1024, lookup, snapshot);
    assert!((server.scheduler_lookup)(7).is_none());

    schedulers
        .lock()
        .unwrap_or_else(|poison| poison.into_inner())
        .insert(7, Arc::new(EchoScheduler));
    assert!((server.scheduler_lookup)(7).is_some());
}
