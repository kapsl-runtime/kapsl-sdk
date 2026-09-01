#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
c_compiler="${CC:-cc}"
cxx_compiler="${CXX:-c++}"

"$c_compiler" \
  -std=c11 \
  -Wall \
  -Wextra \
  -Werror \
  -pedantic \
  -I "$repo_root/crates/kapsl-backend-abi/include" \
  -fsyntax-only \
  "$repo_root/crates/kapsl-backend-abi/tests/header_smoke.c"

"$cxx_compiler" \
  -std=c++17 \
  -Wall \
  -Wextra \
  -Werror \
  -pedantic \
  -I "$repo_root/crates/kapsl-backend-abi/include" \
  -fsyntax-only \
  "$repo_root/crates/kapsl-backend-abi/tests/header_smoke.cpp"
