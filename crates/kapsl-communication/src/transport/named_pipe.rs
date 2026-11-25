//! Connection admission for Windows named pipes.

use std::{ffi::OsStr, io, time::Duration};
use tokio::net::windows::named_pipe::{ClientOptions, NamedPipeClient};

/// Wait for the listener to create an available pipe instance. No request has
/// been written at this point, so this never replays an inference operation.
/// Callers may apply a shorter overall request deadline around this future.
pub async fn connect(path: impl AsRef<OsStr>) -> io::Result<NamedPipeClient> {
    const ERROR_PIPE_BUSY: i32 = 231;
    let deadline = tokio::time::Instant::now() + Duration::from_secs(5);
    loop {
        match ClientOptions::new().open(path.as_ref()) {
            Ok(client) => return Ok(client),
            Err(error) if error.raw_os_error() == Some(ERROR_PIPE_BUSY) => {
                if tokio::time::Instant::now() >= deadline {
                    return Err(io::Error::new(
                        io::ErrorKind::TimedOut,
                        "Named pipe is busy",
                    ));
                }
                tokio::time::sleep(Duration::from_millis(5)).await;
            }
            Err(error) => return Err(error),
        }
    }
}
