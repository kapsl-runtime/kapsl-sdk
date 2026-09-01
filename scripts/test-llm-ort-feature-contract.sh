#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$repo_root"

cpu_tree="$(
  cargo tree \
    --package kapsl-llm \
    --no-default-features \
    --features onnx \
    --locked \
    --edges features \
    --invert ort@2.0.0-rc.11
)"

for provider_feature in cuda tensorrt coreml openvino rocm; do
  if grep -Fq "ort feature \"$provider_feature\"" <<<"$cpu_tree"; then
    echo "kapsl-llm/onnx unexpectedly enables ORT $provider_feature" >&2
    exit 1
  fi
done

default_tree="$(
  cargo tree \
    --package kapsl-llm \
    --locked \
    --edges features \
    --invert ort@2.0.0-rc.11
)"
for provider_feature in cuda tensorrt coreml openvino; do
  if ! grep -Fq "ort feature \"$provider_feature\"" <<<"$default_tree"; then
    echo "kapsl-llm default compatibility profile lost ORT $provider_feature" >&2
    exit 1
  fi
done

cargo check \
  --package kapsl-llm \
  --no-default-features \
  --features onnx \
  --locked

echo "kapsl-llm ORT feature contract passed."
