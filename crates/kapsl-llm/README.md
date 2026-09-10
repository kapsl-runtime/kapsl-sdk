# Kapsl LLM implementations

GGUF and ONNX compute implementations for integration adapters. Engines that
host signed backend packs should depend on the host contracts, not this crate.

## GGUF request ownership

With the `gguf` feature, `GgufSequenceAdmission` binds an integration's engine
request identity when the scheduler assigns a sequence slot. Pass it to
`GgufBackend::infer_with_sequence_admission` or
`GgufBackend::infer_stream_with_sequence_admission`.

The integration captures the engine-issued model, replica and request identity
in the hook. The scheduler supplies the assigned slot before any KV operation.
The hook validates the binding and returns a guard that owns it. That guard
remains alive through prefill, copies and decode, and is dropped after KV cleanup
and before the terminal response. Independent slots remain concurrently active.
The hook is one-shot even when cloned; a failed hook cannot be retried under a
different slot. A slot may be reused after retirement, but a reusable slot ID
must never substitute for a request or KV sequence identity.

Cancellation is retained through queueing, admission, prefill and decode.
Cancelled requests cannot acquire a new slot, and cancellation during admission
releases any acquired guards before compute. Cancelling a shared-prefill leader
preserves its followers and resumes their prompt under a surviving request.
Cancellation is cooperative at scheduler boundaries; an already-running device
operation must finish before its storage can be reclaimed.

This hook is an internal Rust interface inside the integration library. The
engine boundary still uses the versioned backend/KV C ABI. Existing `Engine`
methods and request-memory admission remain supported. No SDK engine API change
or cross-library Rust trait is required.

Host tests cover one-shot admission, concurrent ownership, slot reuse, failed
admission cleanup and independent cancellation of prefill leaders/followers.
They do not establish real GPU correctness or performance qualification.
