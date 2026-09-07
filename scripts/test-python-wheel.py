#!/usr/bin/env python3
"""Install and exercise an actual wheel, including optional dependency isolation."""

import argparse
import os
from pathlib import Path
import subprocess
import sys


def run(*args, **kwargs):
    subprocess.run([sys.executable, *args], check=True, **kwargs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dist", type=Path, default=Path("dist"))
    parser.add_argument("--install-only", action="store_true")
    parser.add_argument("--without-grpc", action="store_true")
    parser.add_argument("--triton", action="store_true")
    args = parser.parse_args()
    if args.triton:
        for imports in (
            "import tritonclient.grpc; from kapsl_sdk import grpc_protocol",
            "from kapsl_sdk import grpc_protocol; import tritonclient.grpc",
        ):
            run("-c", imports + "; assert grpc_protocol.inference.ModelInferRequest(id='test').id == 'test'")
        return
    wheels = list(args.dist.glob("*.whl"))
    if len(wheels) != 1:
        raise SystemExit(f"Expected exactly one platform wheel in {args.dist}")
    wheel = str(wheels[0].resolve())
    run("-m", "pip", "install", "--force-reinstall", "--no-deps", wheel)
    if args.without_grpc:
        run("-c", """
import importlib.util
import kapsl_sdk
assert importlib.util.find_spec('grpc') is None, 'Use an environment without grpcio'
with kapsl_sdk.KapslClient() as client:
    assert not client.closed
assert callable(kapsl_sdk.list_voices)
try:
    kapsl_sdk.KapslGrpcClient()
except ImportError as error:
    assert 'kapsl-sdk[grpc]' in str(error)
else:
    raise AssertionError('Expected an optional dependency installation hint')
""")
    else:
        run("-m", "pip", "install", wheel + "[grpc]")
    if args.install_only:
        return
    if args.without_grpc:
        raise SystemExit("The integration suite requires gRPC; omit --without-grpc")
    root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    executable = "python_test_server.exe" if os.name == "nt" else "python_test_server"
    target = Path(env.get("CARGO_TARGET_DIR", root / "target"))
    env.setdefault("KAPSL_PYTHON_TEST_SERVER", str(target / "debug/examples" / executable))
    run("-m", "pytest", "tests/python", "-q", cwd=root, env=env)


if __name__ == "__main__":
    main()
