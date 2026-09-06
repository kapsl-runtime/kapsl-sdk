#!/usr/bin/env python3
"""Regenerate the bundled clients with grpcio-tools==1.71.2.

The runtime supports Python 3.9; newer grpcio-tools releases require 3.10.
Generated modules use package-relative imports and a private descriptor pool
so applications can also import Triton's overlapping protobuf definitions.
"""

import argparse
import importlib.metadata
from pathlib import Path
import subprocess
import sys
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if importlib.metadata.version("grpcio-tools") != "1.71.2":
        raise SystemExit("Install grpcio-tools==1.71.2 before generating clients")
    root = Path(__file__).resolve().parents[1]
    protos = root / "crates/kapsl-grpc/proto"
    destination = root / "crates/kapsl-pyo3/kapsl_sdk/_grpc"
    names = ("open_inference_grpc", "kapsl_inference")
    with tempfile.TemporaryDirectory(prefix="kapsl-protoc-") as output:
        subprocess.run(
            [
                sys.executable, "-m", "grpc_tools.protoc", f"-I{protos}",
                f"--python_out={output}", f"--grpc_python_out={output}",
                *(str(protos / f"{name}.proto") for name in names),
            ],
            check=True,
        )
        for source in sorted(Path(output).glob("*.py")):
            content = source.read_text()
            for name in names:
                content = content.replace(
                    f"import {name}_pb2 as ",
                    f"from . import {name}_pb2 as ",
                )
                content = content.replace(
                    f"'{name}_pb2'", f"'kapsl_sdk._grpc.{name}_pb2'"
                )
            content = content.replace(
                "from google.protobuf import descriptor_pool as _descriptor_pool",
                "from ._pool import POOL as _kapsl_descriptor_pool",
            ).replace("_descriptor_pool.Default()", "_kapsl_descriptor_pool")
            target = destination / source.name
            if args.check:
                if not target.exists() or target.read_text() != content:
                    raise SystemExit(f"Generated client is stale: {target}")
            else:
                target.write_text(content)


if __name__ == "__main__":
    main()
