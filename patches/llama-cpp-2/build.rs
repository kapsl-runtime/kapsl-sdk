//! Re-exports the ggml backend directory published by `llama-cpp-sys-2` as the
//! `GGML_BACKENDS_DIR` compile-time env var, so the crate can locate the
//! dynamic backend libraries at runtime.

fn main() {
    if let Ok(dir) = std::env::var("DEP_LLAMA_BACKENDS_DIR") {
        println!("cargo:rustc-env=GGML_BACKENDS_DIR={}", dir);
    }
}
