#!/usr/bin/env bash
set -euo pipefail

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# These are reviewed informational notices, documented in SECURITY.md. Deny
# every other warning as well as all vulnerabilities.
cargo audit \
  --deny warnings \
  --ignore RUSTSEC-2025-0141 \
  --ignore RUSTSEC-2024-0436 \
  --file "$repository_root/Cargo.lock"

for lockfile in \
  "$repository_root/patches/esaxx-rs/Cargo.lock" \
  "$repository_root/patches/llama-cpp-2/Cargo.lock" \
  "$repository_root/patches/llama-cpp-sys-2/Cargo.lock"
do
  cargo audit --no-fetch --deny warnings --file "$lockfile"
done
