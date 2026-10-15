# Security policy

Report suspected vulnerabilities privately through GitHub's **Security** tab by
opening a private vulnerability report. Do not include exploit details in a
public issue.

## Dependency policy

CI audits the workspace lockfile and every independently maintained patch
lockfile. Known vulnerabilities, unsound dependencies, yanked releases, and new
maintenance warnings fail the build.

Three informational RustSec notices are explicitly tracked:

- `RUSTSEC-2025-0141`: Bincode is frozen and unmaintained. Kapsl retains
  Bincode 1.3.3 because it defines the deployed IPC wire format, and the
  advisory reports no vulnerability. Incoming frames are size-bounded before
  deserialization. Replacing it requires a versioned protocol migration.
- `RUSTSEC-2024-0436`: `paste` is an unmaintained compile-time dependency of
  Hugging Face Tokenizers. It is not linked as runtime logic, and Tokenizers has
  no release that removes it yet.
- `RUSTSEC-2024-0449`: `unsafe-libyaml` is unmaintained and has no patched
  version. It is pulled in through `serde_yaml` 0.9.34. `kapsl-core` exposes
  `serde_yaml` types in its public API, so replacing it is a semver-breaking
  change. Tracked in kapsl-runtime/kapsl-sdk#172.

The audit script ignores only these three reviewed notices so any new warning is
still a CI failure.
