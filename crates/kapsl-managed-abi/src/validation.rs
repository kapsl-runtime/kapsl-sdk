use super::*;

impl Request {
    pub fn validate(&self) -> Result<(), String> {
        if self.request_id == 0 {
            return Err("managed requests require a nonzero engine request ID".into());
        }
        match &self.input {
            Input::Tensors(request) => {
                if request
                    .metadata
                    .as_ref()
                    .is_some_and(|m| m.auth_token.is_some())
                {
                    return Err("public credentials must be removed before managed dispatch".into());
                }
                request.input.validate().map_err(|e| e.to_string())?;
                let mut names = BTreeSet::new();
                for input in &request.additional_inputs {
                    if input.name.trim().is_empty() || !names.insert(&input.name) {
                        return Err("managed tensor inputs require distinct nonempty names".into());
                    }
                    input.tensor.validate().map_err(|e| e.to_string())?;
                }
                Ok(())
            }
            Input::OpenaiWire(request) => {
                request.validate(MAX_FRAME_BYTES).map_err(|e| e.to_string())
            }
        }
    }
}

impl Initialize {
    pub fn validate(&self) -> Result<(), String> {
        if [&self.backend, &self.profile, &self.pack_version]
            .iter()
            .any(|s| s.trim().is_empty())
            || !self.manifest.is_object()
            || self.devices.is_empty()
        {
            return Err(
                "initialization requires signed identity, a manifest object and assigned devices"
                    .into(),
            );
        }
        let mut devices = BTreeSet::new();
        for device in &self.devices {
            if device.kind.trim().is_empty() || !devices.insert((&device.kind, device.device_id)) {
                return Err("initialization contains empty or duplicate device assignments".into());
            }
        }
        Ok(())
    }
}

impl Command {
    pub fn validate_capabilities(&self, descriptor: &Descriptor) -> Result<(), String> {
        descriptor.validate()?;
        self.validate()?;
        if !descriptor.methods.contains(&self.method()) {
            return Err("managed command is not supported by the selected descriptor".into());
        }
        let requests = match self {
            Self::Infer { request, .. }
            | Self::InferStream { request, .. }
            | Self::PlannedRequestMemory { request } => std::slice::from_ref(request),
            Self::InferBatch { requests, .. } => requests,
            _ => &[],
        };
        for request in requests {
            if let Input::OpenaiWire(wire) = &request.input {
                if !descriptor.capabilities.openai_wire {
                    return Err("managed descriptor does not support OpenAI wire requests".into());
                }
                wire.validate(MAX_FRAME_BYTES).map_err(|e| e.to_string())?;
            }
        }
        if let Self::Initialize { config } = self {
            config.validate()?;
            if config.backend != descriptor.backend
                || config.profile != descriptor.profile
                || config.pack_version != descriptor.pack_version
            {
                return Err(
                    "managed initialization does not match the selected signed identity".into(),
                );
            }
            if config.kv_connection.is_some() && !descriptor.capabilities.kv_participation {
                return Err("KV connection requires the advertised KV capability".into());
            }
        }
        Ok(())
    }
}

fn validate_result(request: &Request, result: &ResultItem) -> Result<(), String> {
    if request.request_id != result.request_id {
        return Err("managed result belongs to another request".into());
    }
    match (&request.input, &result.output) {
        (Input::Tensors(_), Output::Tensor(tensor)) => tensor.validate().map_err(|e| e.to_string()),
        (Input::OpenaiWire(_), Output::OpenaiWire(output)) => {
            output.head.validate().map_err(|e| e.to_string())
        }
        _ => Err("managed result format does not match its request".into()),
    }
}

impl Response {
    /// Validate the operation-specific shape and all embedded request IDs.
    /// Envelope ownership and lifecycle are checked independently. Streaming
    /// consumers additionally retain one `StreamProgress` per dispatch.
    pub fn validate_for(&self, command: &Command) -> Result<(), String> {
        command.validate()?;
        match (command, self) {
            (_, Self::Error { .. }) => Ok(()),
            (Command::Describe, Self::Descriptor { descriptor }) => descriptor.validate(),
            (
                Command::Initialize { .. }
                | Command::Load { .. }
                | Command::Unload
                | Command::Shutdown,
                Self::Ack,
            ) => Ok(()),
            (
                Command::PlannedMemory { .. }
                | Command::PlannedRequestMemory { .. }
                | Command::ActualMemory,
                Self::Memory { report },
            ) => validate_memory_report(report),
            (Command::Infer { request, .. }, Self::Result { result }) => {
                validate_result(request, result)
            }
            (Command::InferBatch { requests, .. }, Self::BatchResult { results }) => {
                if requests.len() != results.len() {
                    return Err("managed batch output count changed".into());
                }
                for (request, result) in requests.iter().zip(results) {
                    validate_result(request, result)?;
                }
                Ok(())
            }
            (
                Command::Cancel { request_ids },
                Self::Cancelled {
                    request_ids: acknowledged,
                },
            ) => {
                if acknowledged.iter().collect::<BTreeSet<_>>().len() != acknowledged.len()
                    || request_ids.iter().collect::<BTreeSet<_>>()
                        != acknowledged.iter().collect::<BTreeSet<_>>()
                {
                    return Err("cancellation acknowledgment changed request ownership".into());
                }
                Ok(())
            }
            (
                Command::InferStream { request, .. },
                Self::StreamChunk { request_id, .. } | Self::StreamEnd { request_id, .. },
            ) => {
                if request.request_id != *request_id {
                    return Err("managed stream belongs to another request".into());
                }
                Ok(())
            }
            (Command::Metrics, Self::Metrics { .. })
            | (Command::ModelInfo, Self::ModelInfo { .. })
            | (Command::KvCapabilities, Self::KvCapabilities { .. })
            | (Command::KvTopology, Self::KvTopology { .. })
            | (Command::Health, Self::Health { .. }) => Ok(()),
            (Command::BatchingPolicy, Self::BatchingPolicy { policy }) => {
                kapsl_engine_api::BatchingPolicy::try_from(policy.clone()).map(|_| ())
            }
            _ => Err("managed response kind does not match the dispatched method".into()),
        }
    }
}

#[derive(Debug, Default)]
pub struct StreamProgress {
    next_sequence: u64,
    finished: bool,
    head_seen: bool,
}

impl StreamProgress {
    pub fn finished(&self) -> bool {
        self.finished
    }

    /// Reject foreign IDs, repeated/missing/out-of-order chunks, format changes
    /// and output after termination. Errors do not advance the stream state.
    pub fn accept(&mut self, command: &Command, response: &Response) -> Result<(), String> {
        if self.finished {
            return Err("managed stream already terminated".into());
        }
        response.validate_for(command)?;
        let Command::InferStream { request, .. } = command else {
            return Err("stream state requires an infer_stream command".into());
        };
        match response {
            Response::StreamChunk {
                sequence, output, ..
            } => {
                if *sequence != self.next_sequence {
                    return Err(
                        "managed stream sequence is missing, duplicated or out of order".into(),
                    );
                }
                let next = self
                    .next_sequence
                    .checked_add(1)
                    .ok_or("managed stream sequence overflow")?;
                match (&request.input, output) {
                    (Input::Tensors(_), Output::Tensor(tensor)) => {
                        tensor.validate().map_err(|e| e.to_string())?;
                    }
                    (Input::OpenaiWire(_), Output::OpenaiHead(head)) if !self.head_seen => {
                        head.validate().map_err(|e| e.to_string())?;
                        self.head_seen = true;
                    }
                    (Input::OpenaiWire(_), Output::OpenaiChunk(_)) if self.head_seen => {}
                    _ => return Err("managed stream changed format or emitted an invalid OpenAI head/body order".into()),
                }
                self.next_sequence = next;
            }
            Response::StreamEnd {
                next_sequence,
                error,
                ..
            } => {
                if *next_sequence != self.next_sequence {
                    return Err("managed stream terminal sequence does not match its chunks".into());
                }
                if matches!(request.input, Input::OpenaiWire(_))
                    && !self.head_seen
                    && error.is_none()
                {
                    return Err("successful OpenAI stream omitted its response head".into());
                }
                self.finished = true;
            }
            Response::Error { .. } if self.next_sequence == 0 => self.finished = true,
            _ => {
                return Err(
                    "managed stream requires a chunk or exactly one terminal response".into(),
                )
            }
        }
        Ok(())
    }
}
