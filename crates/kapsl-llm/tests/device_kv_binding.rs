//! Host-only test of the production binding helper against the real ORT C API.
//! This executable owns its API interception; other test executables are unaffected.
#![cfg(feature = "onnx")]

#[path = "../src/engine/onnx_binding.rs"]
mod binding_execution;

use ort::memory::{AllocationDevice, AllocatorType, MemoryInfo, MemoryType};
use ort::session::Session;
use ort::sys::{OrtApi, OrtErrorCode, OrtIoBinding, OrtRunOptions, OrtSession, OrtStatusPtr};
use ort::value::Tensor;
use std::cell::RefCell;
use std::sync::OnceLock;

static ORIGINAL: OnceLock<OrtApi> = OnceLock::new();

#[derive(Clone, Copy, Debug, Default, PartialEq)]
enum Failure {
    #[default]
    None,
    Inputs,
    Run,
    Outputs,
}

#[derive(Default)]
struct Probe {
    failure: Failure,
    events: Vec<&'static str>,
    // The test owns the tensor until the binding is dropped. Only the test
    // thread uses this pointer, to model a pending host-to-device input copy.
    pending_input: Option<*mut f32>,
    released_bindings: usize,
}

thread_local! {
    static PROBE: RefCell<Probe> = RefCell::default();
}

fn injected_error() -> OrtStatusPtr {
    // SAFETY: The original runtime owns this status and ort releases it.
    unsafe {
        (ORIGINAL.get().unwrap().CreateStatus)(
            OrtErrorCode::ORT_FAIL,
            c"injected binding failure".as_ptr(),
        )
    }
}

unsafe extern "system" fn synchronize_inputs(binding: *mut OrtIoBinding) -> OrtStatusPtr {
    let fail = PROBE.with(|probe| {
        let mut probe = probe.borrow_mut();
        probe.events.push("inputs");
        if probe.failure == Failure::Inputs {
            return true;
        }
        if let Some(input) = probe.pending_input.take() {
            // SAFETY: Four writable f32 elements belong to the live input
            // tensor. No other thread or inference accesses it at this point.
            unsafe { input.copy_from_nonoverlapping([1.0, 2.0, 3.0, 4.0].as_ptr(), 4) };
        }
        false
    });
    if fail {
        return injected_error();
    }
    // SAFETY: Forward the original ORT binding without changing its ownership.
    unsafe { (ORIGINAL.get().unwrap().SynchronizeBoundInputs)(binding) }
}

unsafe extern "system" fn run(
    session: *mut OrtSession,
    options: *const OrtRunOptions,
    binding: *const OrtIoBinding,
) -> OrtStatusPtr {
    let fail = PROBE.with(|probe| {
        let mut probe = probe.borrow_mut();
        probe.events.push("run");
        probe.failure == Failure::Run
    });
    if fail {
        return injected_error();
    }
    // SAFETY: Arguments come directly from the safe ort wrapper.
    unsafe { (ORIGINAL.get().unwrap().RunWithBinding)(session, options, binding) }
}

unsafe extern "system" fn synchronize_outputs(binding: *mut OrtIoBinding) -> OrtStatusPtr {
    let fail = PROBE.with(|probe| {
        let mut probe = probe.borrow_mut();
        probe.events.push("outputs");
        probe.failure == Failure::Outputs
    });
    if fail {
        return injected_error();
    }
    // SAFETY: Forward the original ORT binding without changing its ownership.
    unsafe { (ORIGINAL.get().unwrap().SynchronizeBoundOutputs)(binding) }
}

unsafe extern "system" fn release_binding(binding: *mut OrtIoBinding) {
    PROBE.with(|probe| probe.borrow_mut().released_bindings += 1);
    // SAFETY: Called exactly once by IoBinding::drop, as with the original API.
    unsafe { (ORIGINAL.get().unwrap().ReleaseIoBinding)(binding) }
}

#[test]
fn device_kv_binding_waits_for_inputs_and_propagates_failures() {
    // This is the only test in this executable. Set the proxy before any safe
    // ort API initializes its global function table, retaining the real CPU EP.
    // SAFETY: The linked runtime exposes its immutable API table for this version.
    let mut api = unsafe {
        let base = ort::sys::OrtGetApiBase();
        assert!(!base.is_null());
        let api = ((*base).GetApi)(ort::sys::ORT_API_VERSION);
        assert!(!api.is_null());
        *api
    };
    assert!(ORIGINAL.set(api).is_ok());
    api.SynchronizeBoundInputs = synchronize_inputs;
    api.RunWithBinding = run;
    api.SynchronizeBoundOutputs = synchronize_outputs;
    api.ReleaseIoBinding = release_binding;
    assert!(ort::set_api(api));

    for failure in [
        Failure::None,
        Failure::Inputs,
        Failure::Run,
        Failure::Outputs,
    ] {
        PROBE.with(|probe| {
            *probe.borrow_mut() = Probe {
                failure,
                ..Probe::default()
            };
        });
        {
            let mut session = Session::builder()
                .unwrap()
                .with_intra_threads(1)
                .unwrap()
                .commit_from_memory(include_bytes!("fixtures/profile-matmul.onnx"))
                .unwrap();
            let mut input = Tensor::from_array(([1, 4], vec![0.0_f32; 4])).unwrap();
            let input_ptr = input
                .try_extract_tensor_mut::<f32>()
                .unwrap()
                .1
                .as_mut_ptr();
            let mut binding = session.create_binding().unwrap();
            binding
                .bind_input(session.inputs()[0].name(), &input)
                .unwrap();
            let cpu = MemoryInfo::new(
                AllocationDevice::CPU,
                0,
                AllocatorType::Device,
                MemoryType::CPUOutput,
            )
            .unwrap();
            binding
                .bind_output_to_device(session.outputs()[0].name(), &cpu)
                .unwrap();
            PROBE.with(|probe| probe.borrow_mut().pending_input = Some(input_ptr));

            if failure == Failure::None {
                // Negative control: without the input fence, actual ORT MatMul
                // consumes the stale input. Output synchronization cannot fix it.
                let outputs = session.run_binding(&binding).unwrap();
                binding.synchronize_outputs().unwrap();
                assert_eq!(outputs[0].try_extract_tensor::<f32>().unwrap().1, &[0.0; 4]);
                drop(outputs);
                PROBE.with(|probe| probe.borrow_mut().events.clear());
            }

            let result = binding_execution::run_device_kv_binding(&mut session, &binding);
            if failure == Failure::None {
                let outputs = result.unwrap();
                assert_eq!(
                    outputs[0].try_extract_tensor::<f32>().unwrap().1,
                    &[2.0, 4.0, 6.0, 8.0]
                );
            } else {
                let error = result.expect_err("injected error must reach the caller");
                assert!(error.to_string().contains("injected binding failure"));
            }
            PROBE.with(|probe| probe.borrow_mut().pending_input = None);
        }
        PROBE.with(|probe| {
            let probe = probe.borrow();
            let expected: &[&str] = match failure {
                Failure::None | Failure::Outputs => &["inputs", "run", "outputs"],
                Failure::Inputs => &["inputs"],
                Failure::Run => &["inputs", "run"],
            };
            assert_eq!(probe.events, expected, "{failure:?}");
            assert_eq!(probe.released_bindings, 1, "{failure:?}");
        });
    }
}
