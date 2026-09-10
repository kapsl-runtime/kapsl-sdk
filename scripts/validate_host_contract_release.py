#!/usr/bin/env python3
"""Validate publication of an exact, main-merged host-contract tag."""

import os
from pathlib import Path
import re
import subprocess
import tomllib

PACKAGES = frozenset(("kapsl-kv-abi", "kapsl-managed-abi"))
VERSION = r"(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)"


def validate(package, version, *, event, ref_type, ref_name, publish, on_main):
    if package not in PACKAGES or re.fullmatch(VERSION, version) is None:
        raise ValueError("host publication requires an allowed package and stable version")
    tag = f"{package}-v{version}"
    publishing = event == "push" or publish
    if ref_type == "tag" and ref_name != tag:
        raise ValueError("host contract tag does not match its package version")
    if publishing and (ref_type != "tag" or ref_name != tag or not on_main):
        raise ValueError("publication requires the exact stable tag already merged into main")
    if publishing and event not in ("push", "workflow_dispatch"):
        raise ValueError("unsupported publication event")
    return publishing


def main():
    ref_name = os.environ.get("REF_NAME", "")
    ref_type = os.environ.get("REF_TYPE", "")
    package = os.environ.get("PACKAGE_INPUT", "kapsl-managed-abi")
    if ref_type == "tag":
        candidates = [name for name in PACKAGES if ref_name.startswith(name + "-v")]
        if len(candidates) != 1:
            raise ValueError("unsupported host contract release tag")
        package = candidates[0]
    if package not in PACKAGES:
        raise ValueError("unsupported host contract package")
    manifest = Path("crates") / package / "Cargo.toml"
    with manifest.open("rb") as handle:
        version = tomllib.load(handle)["package"]["version"]
    publishing = validate(
        package, version,
        event=os.environ.get("EVENT_NAME", ""),
        ref_type=ref_type, ref_name=ref_name,
        publish=os.environ.get("REQUESTED_PUBLISH", "false").lower() == "true",
        on_main=subprocess.run(
            ["git", "merge-base", "--is-ancestor", "HEAD", "refs/remotes/origin/main"],
            check=False,
        ).returncode == 0,
    )
    with Path(os.environ["GITHUB_OUTPUT"]).open("a", encoding="utf-8") as handle:
        handle.write(f"package={package}\nversion={version}\npublish={str(publishing).lower()}\n")


if __name__ == "__main__":
    main()
