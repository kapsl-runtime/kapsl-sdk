//! With the `cute` feature, compiles the CuTe kernels in
//! `src/cuda/kernels/cute/` to PTX with nvcc and leaves them in `OUT_DIR` for
//! `src/cuda/cute.rs` to embed. Without the feature this does nothing.
//!
//! Environment:
//! - `NVCC`: path to nvcc. Default: `$CUDA_PATH|$CUDA_HOME|$CUDA_ROOT/bin/nvcc`,
//!   then `nvcc` on `PATH`.
//! - `CUTLASS_DIR`: a CUTLASS checkout (or its `include/` dir). Default: the
//!   `third_party/cutlass` submodule.
//! - `KAPSL_NVCC_FLAGS`: extra whitespace-separated nvcc flags.

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    #[cfg(feature = "cute")]
    cute::compile_kernels();
}

#[cfg(feature = "cute")]
mod cute {
    use std::env;
    use std::path::{Path, PathBuf};
    use std::process::Command;

    /// `<name>.cu` in `src/cuda/kernels/cute/` becomes `<name>.ptx` in `OUT_DIR`.
    const KERNELS: &[&str] = &["prefill_attention"];

    /// Oldest architecture the kernels run on: they use the SM80 tensor-core
    /// MMA. Newer GPUs JIT the same PTX.
    const ARCH: &str = "sm_80";

    pub fn compile_kernels() {
        for var in [
            "NVCC",
            "CUDA_PATH",
            "CUDA_HOME",
            "CUDA_ROOT",
            "CUTLASS_DIR",
            "KAPSL_NVCC_FLAGS",
        ] {
            println!("cargo:rerun-if-env-changed={var}");
        }

        let manifest_dir = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap());
        let out_dir = PathBuf::from(env::var_os("OUT_DIR").unwrap());
        let src_dir = manifest_dir.join("src/cuda/kernels/cute");
        let cutlass = cutlass_include(&manifest_dir);
        let nvcc = nvcc_path();
        let extra_flags = env::var("KAPSL_NVCC_FLAGS").unwrap_or_default();

        for name in KERNELS {
            let src = src_dir.join(format!("{name}.cu"));
            let ptx = out_dir.join(format!("{name}.ptx"));
            println!("cargo:rerun-if-changed={}", src.display());

            let mut cmd = Command::new(&nvcc);
            cmd.arg("--ptx")
                .arg("-O3")
                .arg("-std=c++17")
                .arg(format!("-arch={ARCH}"))
                .arg("-DNDEBUG")
                .arg(format!("-I{}", cutlass.display()))
                .args(extra_flags.split_whitespace())
                .arg("-o")
                .arg(&ptx)
                .arg(&src);
            host_compiler_env(&mut cmd);

            let output = cmd.output().unwrap_or_else(|e| {
                panic!(
                    "kapsl-kernels: the `cute` feature needs nvcc, but running `{}` failed: {e}\n\
                     Install the CUDA toolkit, or set NVCC or CUDA_PATH.",
                    nvcc.display()
                )
            });
            if !output.status.success() {
                panic!(
                    "kapsl-kernels: nvcc failed on {} ({})\n--- stdout\n{}\n--- stderr\n{}",
                    src.display(),
                    output.status,
                    String::from_utf8_lossy(&output.stdout),
                    String::from_utf8_lossy(&output.stderr)
                );
            }
        }
    }

    fn nvcc_path() -> PathBuf {
        if let Some(nvcc) = env::var_os("NVCC") {
            return nvcc.into();
        }
        let exe = if cfg!(windows) { "nvcc.exe" } else { "nvcc" };
        for var in ["CUDA_PATH", "CUDA_HOME", "CUDA_ROOT"] {
            if let Some(root) = env::var_os(var) {
                let candidate = Path::new(&root).join("bin").join(exe);
                if candidate.is_file() {
                    return candidate;
                }
            }
        }
        PathBuf::from("nvcc")
    }

    fn cutlass_include(manifest_dir: &Path) -> PathBuf {
        let candidates = match env::var_os("CUTLASS_DIR") {
            Some(dir) => vec![Path::new(&dir).join("include"), PathBuf::from(dir)],
            None => vec![manifest_dir.join("../../third_party/cutlass/include")],
        };
        candidates
            .into_iter()
            .find(|dir| dir.join("cute/tensor.hpp").is_file())
            .unwrap_or_else(|| {
                panic!(
                    "kapsl-kernels: the `cute` feature needs the CUTLASS headers.\n\
                     Run `git submodule update --init third_party/cutlass`, \
                     or set CUTLASS_DIR to a CUTLASS checkout."
                )
            })
    }

    /// nvcc preprocesses with the host C++ compiler. On MSVC targets that is
    /// cl.exe, which is normally not on `PATH` outside a Developer Command
    /// Prompt, so borrow the environment the `cc` crate resolves for it.
    fn host_compiler_env(cmd: &mut Command) {
        if env::var("CARGO_CFG_TARGET_ENV").as_deref() != Ok("msvc") {
            return;
        }
        let Ok(cl) = cc::Build::new()
            .cpp(true)
            .cargo_metadata(false)
            .try_get_compiler()
        else {
            return; // let nvcc report the missing compiler itself
        };

        let mut path = env::var_os("PATH").unwrap_or_default();
        for (key, value) in cl.env() {
            if key.eq_ignore_ascii_case("PATH") {
                path = value.clone();
            } else {
                cmd.env(key, value);
            }
        }
        let mut dirs: Vec<PathBuf> = cl
            .path()
            .parent()
            .map(Path::to_path_buf)
            .into_iter()
            .collect();
        dirs.extend(env::split_paths(&path));
        if let Ok(joined) = env::join_paths(dirs) {
            cmd.env("PATH", joined);
        }
    }
}
