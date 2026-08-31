# Security policy

Report suspected vulnerabilities privately through GitHub's **Security** tab by
opening a private vulnerability report. Do not include exploit details in a
public issue.

## Dependency policy

CI audits the workspace lockfile and every independently maintained patch
lockfile. Known vulnerabilities, unsound dependencies, yanked releases, and new
maintenance warnings fail the build.

Two informational RustSec notices are explicitly tracked:

- `RUSTSEC-2025-0141`: Bincode is frozen and unmaintained. Kapsl retains
  Bincode 1.3.3 because it defines the deployed IPC wire format, and the
  advisory reports no vulnerability. Incoming frames are size-bounded before
  deserialization. Replacing it requires a versioned protocol migration.
- `RUSTSEC-2024-0436`: `paste` is an unmaintained compile-time dependency of
  Hugging Face Tokenizers. It is not linked as runtime logic, and Tokenizers has
  no release that removes it yet.

The audit script ignores only these two reviewed notices so any new warning is
still a CI failure.
