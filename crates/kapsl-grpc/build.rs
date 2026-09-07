use prost_build::Config;
use protoc_bin_vendored::protoc_bin_path;
use tonic_prost_build::configure;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut prost = Config::new();
    prost.protoc_executable(protoc_bin_path()?);
    configure().compile_with_config(
        prost,
        &[
            "proto/open_inference_grpc.proto",
            "proto/kapsl_inference.proto",
        ],
        &["proto"],
    )?;
    println!("cargo:rerun-if-changed=proto");
    Ok(())
}
