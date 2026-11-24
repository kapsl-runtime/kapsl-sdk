#!/usr/bin/env bash
set -euo pipefail

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# These are reviewed informational notices, documented in SECURITY.md. Deny
# every other warning as well as all vulnerabilities.
# RUSTSEC-2024-0449: unsafe-libyaml via serde_yaml is unmaintained with no fix,
# tracked in kapsl-runtime/kapsl-sdk#172.
# RUSTSEC-2026-0543 (chrono, via cron) and RUSTSEC-2026-0542 (humantime, a
# dev-dependency via env_logger): unmaintained notices, no vulnerability.
cargo audit \
  --deny warnings \
  --ignore RUSTSEC-2025-0141 \
  --ignore RUSTSEC-2024-0436 \
  --ignore RUSTSEC-2024-0449 \
  --ignore RUSTSEC-2026-0543 \
  --ignore RUSTSEC-2026-0542 \
  --file "$repository_root/Cargo.lock"

for lockfile in \
  "$repository_root/patches/esaxx-rs/Cargo.lock" \
  "$repository_root/patches/llama-cpp-2/Cargo.lock" \
  "$repository_root/patches/llama-cpp-sys-2/Cargo.lock"
do
  cargo audit --no-fetch --deny warnings --file "$lockfile"
done
