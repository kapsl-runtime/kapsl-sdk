# kapsl-llama-cpp-2

Kapsl-maintained Rust bindings for the pinned `llama.cpp-kapsl` ABI. The
package includes Kapsl's shared-KV and device-memory reporting extensions and
is published under a distinct name so downstream applications can resolve the
complete runtime dependency graph from crates.io.

The Rust library name remains `llama_cpp_2` for source compatibility. This
crate is derived from `llama-cpp-2` and retains its MIT OR Apache-2.0 license;
the repository history records the upstream source and Kapsl modifications.

## Tool calling

`llama-cpp-2` exposes the raw llama.cpp OpenAI-compatible tool-calling flow, so Rust callers can pass tool definitions into chat templates and get the generated grammar back.

```rust
use llama_cpp_2::openai::OpenAIChatTemplateParams;
use serde_json::json;

let template = model.chat_template(None)?;

let tools_json = json!([
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Fetch current weather by city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": { "type": "string" }
                },
                "required": ["location"]
            }
        }
    }
])
.to_string();

let messages_json = json!([
    {
        "role": "system",
        "content": "You are a tool caller."
    },
    {
        "role": "user",
        "content": "Fetch the weather in Paris."
    }
])
.to_string();

let params = OpenAIChatTemplateParams {
    messages_json: &messages_json,
    tools_json: Some(&tools_json),
    tool_choice: Some("auto"),
    json_schema: None,
    grammar: None,
    reasoning_format: None,
    chat_template_kwargs: Some("{}"),
    add_generation_prompt: true,
    use_jinja: true,
    parallel_tool_calls: false,
    enable_thinking: false,
    add_bos: false,
    add_eos: false,
    parse_tool_calls: true,
};

let result = model.apply_chat_template_oaicompat(&template, &params)?;
```

For standalone grammar generation from a JSON schema string, use `llama_cpp_2::json_schema_to_grammar`.

## Build dependencies

This uses bindgen to build the bindings to llama.cpp. This means that you need to have clang installed on your system.

See [bindgen](https://rust-lang.github.io/rust-bindgen/requirements.html) for more information.

## Safety

This crate exposes thin wrappers over a native C/C++ API. Some operations are
unsafe, and misuse can cause undefined behavior. Report Kapsl-specific issues
in the Kapsl SDK repository.

The higher-level Kapsl backend is the recommended integration surface for
applications.
