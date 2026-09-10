use super::*;
use std::io::{BufReader, Cursor};

fn owner() -> InstanceIdentity {
    InstanceIdentity {
        instance_id: "test-instance-1".into(),
        model_id: 7,
        replica_id: 2,
    }
}

fn descriptor() -> Descriptor {
    Descriptor {
        schema_version: 1,
        protocol_version: 1,
        execution_mode: "managed".into(),
        backend: "fake-managed".into(),
        profile: "cpu".into(),
        pack_version: "1.0.0".into(),
        formats: vec!["onnx".into()],
        model_types: vec![],
        tasks: vec!["forward".into()],
        capabilities: Capabilities {
            memory_reporting: true,
            streaming: true,
            cancellation: true,
            batching: true,
            concurrent_inference: true,
            openai_wire: false,
            kv_participation: false,
        },
        methods: [
            Method::Describe,
            Method::Initialize,
            Method::PlannedMemory,
            Method::Load,
            Method::PlannedRequestMemory,
            Method::Infer,
            Method::InferBatch,
            Method::InferStream,
            Method::Cancel,
            Method::ActualMemory,
            Method::Metrics,
            Method::ModelInfo,
            Method::BatchingPolicy,
            Method::Health,
            Method::Unload,
            Method::Shutdown,
        ]
        .into(),
    }
}

#[test]
fn every_required_and_advertised_method_is_checked() {
    let original = descriptor();
    original.validate().unwrap();
    for method in &original.methods {
        let mut candidate = original.clone();
        candidate.methods.remove(method);
        assert!(candidate.validate().is_err(), "missing {method:?}");
    }
    let mut candidate = original.clone();
    candidate.methods.insert(Method::KvTopology);
    assert!(candidate.validate().is_err());
    let mut candidate = original.clone();
    candidate.capabilities.cancellation = false;
    candidate.methods.remove(&Method::Cancel);
    assert!(candidate
        .validate()
        .unwrap_err()
        .contains("streaming requires cancellation"));
    let mut candidate = original.clone();
    candidate.capabilities.memory_reporting = false;
    assert!(candidate.validate().is_err());
    let mut candidate = original;
    candidate.formats.push(" ONNX ".into());
    assert!(candidate.validate().is_err());
}

#[test]
fn response_identity_rejects_other_models_replicas_processes_and_calls() {
    let response = AdapterMessage::new(owner(), 42, Response::Ack);
    response.validate_for(&owner(), 42).unwrap();
    for mismatch in [
        InstanceIdentity {
            model_id: 8,
            ..owner()
        },
        InstanceIdentity {
            replica_id: 3,
            ..owner()
        },
        InstanceIdentity {
            instance_id: "restarted-instance".into(),
            ..owner()
        },
    ] {
        assert!(response.validate_for(&mismatch, 42).is_err());
    }
    assert!(response.validate_for(&owner(), 43).is_err());
}

#[test]
fn admission_cannot_cross_request_or_batch_ownership() {
    let mut grant = Admission {
        grant_id: "grant-1".into(),
        request_ids: vec![5, 9],
        limits: MemoryReport::default(),
    };
    grant.validate_for(&[9, 5]).unwrap();
    for wrong in [vec![], vec![5], vec![5, 8], vec![5, 9, 9], vec![0, 9]] {
        assert!(grant.validate_for(&wrong).is_err());
    }
    grant.request_ids.push(9);
    assert!(grant.validate_for(&[5, 9]).is_err());
    grant.request_ids.clear();
    grant.validate_for(&[]).unwrap();
    grant.grant_id.clear();
    assert!(grant.validate_for(&[]).is_err());
}

#[test]
fn cancellation_needs_explicit_distinct_ownership() {
    Command::Cancel {
        request_ids: vec![4, 5],
    }
    .validate()
    .unwrap();
    for invalid in [vec![], vec![0], vec![4, 4]] {
        assert!(Command::Cancel {
            request_ids: invalid
        }
        .validate()
        .is_err());
    }
}

#[test]
fn codec_preserves_frame_boundaries_and_escaped_newlines() {
    let first = AdapterMessage::new(
        owner(),
        1,
        Response::Health {
            healthy: false,
            details: "first\nsecond".into(),
        },
    );
    let second = AdapterMessage::new(owner(), 2, Response::Ack);
    let mut bytes = Vec::new();
    write_frame(&mut bytes, &first).unwrap();
    write_frame(&mut bytes, &second).unwrap();
    let mut reader = Cursor::new(bytes);
    let received: AdapterMessage = read_frame(&mut reader).unwrap().unwrap();
    assert!(
        matches!(received.body, Response::Health { details, .. } if details == "first\nsecond")
    );
    assert_eq!(read_frame::<Response>(&mut reader).unwrap().unwrap().id, 2);
    assert!(read_frame::<Response>(&mut reader).unwrap().is_none());
}

#[test]
fn malformed_partial_unknown_version_and_unknown_method_frames_are_rejected() {
    let message = HostMessage::new(owner(), 1, Command::Describe);
    let valid = serde_json::to_value(&message).unwrap();
    for (field, value) in [
        ("protocol_version", Value::from(2)),
        ("id", Value::from(0)),
        ("unexpected", Value::Null),
    ] {
        let mut invalid = valid.clone();
        invalid[field] = value;
        let mut bytes = serde_json::to_vec(&invalid).unwrap();
        bytes.push(b'\n');
        assert!(
            read_frame::<Command>(&mut Cursor::new(bytes)).is_err(),
            "{field}"
        );
    }
    let mut invalid = valid;
    invalid["body"]["method"] = Value::from("launch_vllm");
    let mut bytes = serde_json::to_vec(&invalid).unwrap();
    bytes.push(b'\n');
    assert!(read_frame::<Command>(&mut Cursor::new(bytes)).is_err());
    let bytes = serde_json::to_vec(&message).unwrap();
    assert!(read_frame::<Command>(&mut Cursor::new(bytes)).is_err());
    assert!(read_frame::<Command>(&mut Cursor::new(b"{bad json}\n")).is_err());
}

#[test]
fn reader_is_bounded_even_without_a_delimiter() {
    // An effectively endless producer must not cause unbounded buffering.
    let mut source = BufReader::with_capacity(1024, io::repeat(b'x'));
    assert!(read_frame::<Command>(&mut source).is_err());
    // The same source remains usable; this was a bounded read, not read-to-EOF.
    let mut next = [0; 1];
    source.read_exact(&mut next).unwrap();
    assert_eq!(next, [b'x']);
}

#[test]
fn oversized_outgoing_frames_do_not_partially_write_the_transport() {
    let message = AdapterMessage::new(
        owner(),
        1,
        Response::Health {
            healthy: false,
            details: "x".repeat(MAX_FRAME_BYTES),
        },
    );
    let mut transport = Vec::new();
    assert!(write_frame(&mut transport, &message).is_err());
    assert!(transport.is_empty());
}

fn request(id: u64) -> Request {
    Request {
        request_id: id,
        input: Input::Tensors(InferenceRequest::new(
            BinaryTensorPacket::new(vec![1], kapsl_engine_api::TensorDtype::Uint8, vec![42])
                .unwrap(),
        )),
    }
}

#[test]
fn invalid_tensor_shapes_and_public_credentials_fail_before_dispatch() {
    let mut credentials = request(1);
    let Input::Tensors(tensors) = &mut credentials.input else {
        unreachable!()
    };
    tensors.metadata = Some(kapsl_engine_api::RequestMetadata {
        auth_token: Some("synthetic-test-credential".into()),
        ..Default::default()
    });
    assert!(credentials.validate().unwrap_err().contains("credentials"));

    let mut malformed = request(1);
    let Input::Tensors(tensors) = &mut malformed.input else {
        unreachable!()
    };
    tensors.input.shape = vec![2];
    for invalid in [credentials, malformed] {
        for command in [
            Command::Infer {
                request: invalid.clone(),
                admission: admission(&[1]),
            },
            Command::InferStream {
                request: invalid.clone(),
                admission: admission(&[1]),
            },
            Command::InferBatch {
                requests: vec![invalid.clone()],
                admission: admission(&[1]),
            },
            Command::PlannedRequestMemory { request: invalid },
        ] {
            assert!(command.validate().is_err());
        }
    }

    let Output::Tensor(mut tensor) = output() else {
        unreachable!()
    };
    tensor.data.clear();
    let command = Command::Infer {
        request: request(1),
        admission: admission(&[1]),
    };
    let result = Response::Result {
        result: ResultItem {
            request_id: 1,
            output: Output::Tensor(tensor.clone()),
        },
    };
    assert!(result.validate_for(&command).is_err());
    let mut stream = StreamProgress::default();
    let command = Command::InferStream {
        request: request(1),
        admission: admission(&[1]),
    };
    assert!(stream
        .accept(
            &command,
            &Response::StreamChunk {
                request_id: 1,
                sequence: 0,
                output: Output::Tensor(tensor)
            }
        )
        .is_err());
    stream
        .accept(
            &command,
            &Response::StreamChunk {
                request_id: 1,
                sequence: 0,
                output: output(),
            },
        )
        .unwrap();
}

fn admission(ids: &[u64]) -> Admission {
    Admission {
        grant_id: "admitted".into(),
        request_ids: ids.to_vec(),
        limits: MemoryReport::default(),
    }
}

fn output() -> Output {
    Output::Tensor(
        BinaryTensorPacket::new(vec![1], kapsl_engine_api::TensorDtype::Uint8, vec![42]).unwrap(),
    )
}

#[test]
fn batch_output_cannot_change_count_order_or_request_ownership() {
    let command = Command::InferBatch {
        requests: vec![request(1), request(2)],
        admission: admission(&[1, 2]),
    };
    command.validate_capabilities(&descriptor()).unwrap();
    let result = |ids: &[u64]| Response::BatchResult {
        results: ids
            .iter()
            .map(|id| ResultItem {
                request_id: *id,
                output: output(),
            })
            .collect(),
    };
    result(&[1, 2]).validate_for(&command).unwrap();
    for invalid in [&[2, 1][..], &[1], &[1, 1], &[1, 3]] {
        assert!(result(invalid).validate_for(&command).is_err());
    }
    assert!(Response::Ack.validate_for(&command).is_err());
    let cancel = Command::Cancel {
        request_ids: vec![1, 2],
    };
    assert!(Response::Cancelled {
        request_ids: vec![1, 3]
    }
    .validate_for(&cancel)
    .is_err());
    assert!(Response::Cancelled {
        request_ids: vec![1, 2, 2]
    }
    .validate_for(&cancel)
    .is_err());
}

#[test]
fn stream_enforces_owner_order_terminal_and_cancellation_semantics() {
    let command = Command::InferStream {
        request: request(7),
        admission: admission(&[7]),
    };
    let chunk = |id, sequence| Response::StreamChunk {
        request_id: id,
        sequence,
        output: output(),
    };
    let mut stream = StreamProgress::default();
    assert!(stream.accept(&command, &chunk(8, 0)).is_err());
    assert!(stream.accept(&command, &chunk(7, 1)).is_err());
    stream.accept(&command, &chunk(7, 0)).unwrap();
    assert!(stream.accept(&command, &chunk(7, 0)).is_err());
    let failure = Failure {
        code: ErrorCode::Cancelled,
        message: "cancelled".into(),
    };
    assert!(stream
        .accept(
            &command,
            &Response::Error {
                error: failure.clone()
            }
        )
        .is_err());
    assert!(stream
        .accept(
            &command,
            &Response::StreamEnd {
                request_id: 7,
                next_sequence: 2,
                error: None
            }
        )
        .is_err());
    let terminal = Response::StreamEnd {
        request_id: 7,
        next_sequence: 1,
        error: Some(failure),
    };
    stream.accept(&command, &terminal).unwrap();
    assert!(stream.finished());
    assert!(stream.accept(&command, &terminal).is_err());
    assert!(stream.accept(&command, &chunk(7, 1)).is_err());
}

#[test]
fn malformed_memory_reports_and_zero_batch_limits_fail_before_use() {
    use kapsl_engine_api::{MemoryAllocationClass, MemoryDomain};
    let mut report = MemoryReport::single(
        "weights",
        MemoryDomain::Host,
        MemoryAllocationClass::PersistentWeights,
        64,
    );
    Response::Memory {
        report: report.clone(),
    }
    .validate_for(&Command::ActualMemory)
    .unwrap();
    report.allocations.push(report.allocations[0].clone());
    assert!(Response::Memory { report }
        .validate_for(&Command::ActualMemory)
        .is_err());
    let mut policy = BatchingPolicy::from(kapsl_engine_api::BatchingPolicy::none());
    policy.max_requests = 0;
    assert!(Response::BatchingPolicy { policy }
        .validate_for(&Command::BatchingPolicy)
        .is_err());
}
