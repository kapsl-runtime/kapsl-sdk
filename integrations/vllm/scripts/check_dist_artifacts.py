#!/usr/bin/env python3
"""Fail closed unless connector release archives contain the certified files."""

from __future__ import annotations

import argparse
import configparser
import email.parser
import tarfile
import tomllib
import zipfile
from pathlib import Path, PurePosixPath


PROJECT_NAME = "kapsl-vllm-connector"
NORMALIZED_NAME = "kapsl_vllm_connector"
REQUIRED_MODULES = ("plan.py", "planning.py")
REQUIRED_ENTRY_POINTS = {
    "kapsl-vllm-flash-attn-probe": (
        "kapsl_vllm_connector.flash_attn_probe:main"
    ),
    "kapsl-vllm-plan": "kapsl_vllm_connector.plan:main",
}


class ArtifactError(RuntimeError):
    """Raised when a distribution is incomplete or inconsistent."""


def _project_version(project_root: Path) -> str:
    with (project_root / "pyproject.toml").open("rb") as handle:
        value = tomllib.load(handle)["project"]["version"]
    if not isinstance(value, str) or not value:
        raise ArtifactError("pyproject project.version must be a non-empty string")
    return value


def _check_source_licenses(project_root: Path) -> None:
    repository_root = project_root.parents[1]
    for filename in ("LICENSE", "NOTICE"):
        packaged = (project_root / filename).read_text(encoding="utf-8")
        canonical = (repository_root / filename).read_text(encoding="utf-8")
        if packaged.splitlines() != canonical.splitlines():
            raise ArtifactError(
                f"connector {filename} differs from repository-root {filename}"
            )


def _only(paths: list[Path], description: str) -> Path:
    if len(paths) != 1:
        rendered = ", ".join(path.name for path in paths) or "none"
        raise ArtifactError(f"expected exactly one {description}; found {rendered}")
    return paths[0]


def _safe_members(names: set[str], artifact: Path) -> None:
    for name in names:
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts:
            raise ArtifactError(f"{artifact.name} contains unsafe path {name!r}")


def _require(names: set[str], required: set[str], artifact: Path) -> None:
    missing = sorted(required - names)
    if missing:
        raise ArtifactError(
            f"{artifact.name} is missing required members: {', '.join(missing)}"
        )


def _check_wheel(
    wheel: Path,
    project_root: Path,
    version: str,
) -> None:
    expected_name = f"{NORMALIZED_NAME}-{version}-py3-none-any.whl"
    if wheel.name != expected_name:
        raise ArtifactError(f"expected wheel {expected_name!r}, found {wheel.name!r}")

    dist_info = f"{NORMALIZED_NAME}-{version}.dist-info"
    required = {
        f"kapsl_vllm_connector/{module}" for module in REQUIRED_MODULES
    }
    required.update(
        {
            f"{dist_info}/METADATA",
            f"{dist_info}/entry_points.txt",
            f"{dist_info}/licenses/LICENSE",
            f"{dist_info}/licenses/NOTICE",
        }
    )

    with zipfile.ZipFile(wheel) as archive:
        names = set(archive.namelist())
        _safe_members(names, wheel)
        _require(names, required, wheel)

        metadata = email.parser.BytesParser().parsebytes(
            archive.read(f"{dist_info}/METADATA")
        )
        if metadata["Name"] != PROJECT_NAME or metadata["Version"] != version:
            raise ArtifactError(
                f"{wheel.name} metadata identity is "
                f"{metadata['Name']!r} {metadata['Version']!r}"
            )

        entries = configparser.ConfigParser(interpolation=None)
        entries.read_string(
            archive.read(f"{dist_info}/entry_points.txt").decode("utf-8")
        )
        actual_entries = dict(entries.items("console_scripts"))
        if actual_entries != REQUIRED_ENTRY_POINTS:
            raise ArtifactError(
                f"{wheel.name} console scripts differ: {actual_entries!r}"
            )

        for filename in ("LICENSE", "NOTICE"):
            expected = (project_root / filename).read_bytes()
            actual = archive.read(f"{dist_info}/licenses/{filename}")
            if actual != expected:
                raise ArtifactError(
                    f"{wheel.name} contains a non-canonical {filename}"
                )


def _check_sdist(
    sdist: Path,
    project_root: Path,
    version: str,
) -> None:
    expected_name = f"{NORMALIZED_NAME}-{version}.tar.gz"
    if sdist.name != expected_name:
        raise ArtifactError(f"expected sdist {expected_name!r}, found {sdist.name!r}")

    root = f"{NORMALIZED_NAME}-{version}"
    required = {
        f"{root}/LICENSE",
        f"{root}/NOTICE",
        f"{root}/pyproject.toml",
    }
    required.update(
        f"{root}/src/kapsl_vllm_connector/{module}"
        for module in REQUIRED_MODULES
    )

    with tarfile.open(sdist, mode="r:gz") as archive:
        names = {member.name for member in archive.getmembers()}
        _safe_members(names, sdist)
        _require(names, required, sdist)

        for filename in ("LICENSE", "NOTICE"):
            member = archive.extractfile(f"{root}/{filename}")
            if member is None:
                raise ArtifactError(f"could not read {filename} from {sdist.name}")
            if member.read() != (project_root / filename).read_bytes():
                raise ArtifactError(
                    f"{sdist.name} contains a non-canonical {filename}"
                )


def check_dist(dist: Path, project_root: Path) -> None:
    if not dist.is_dir():
        raise ArtifactError(f"distribution directory does not exist: {dist}")
    version = _project_version(project_root)
    _check_source_licenses(project_root)
    files = sorted(path for path in dist.iterdir() if path.is_file())
    wheel = _only([path for path in files if path.suffix == ".whl"], "wheel")
    sdist = _only(
        [path for path in files if path.name.endswith(".tar.gz")],
        "source distribution",
    )
    unexpected = [path.name for path in files if path not in {wheel, sdist}]
    if unexpected:
        raise ArtifactError(f"unexpected distribution files: {', '.join(unexpected)}")
    _check_wheel(wheel, project_root, version)
    _check_sdist(sdist, project_root, version)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dist", type=Path)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    args = parser.parse_args()
    try:
        check_dist(args.dist, args.project_root)
    except (ArtifactError, OSError, KeyError, ValueError, tarfile.TarError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
