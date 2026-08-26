from __future__ import annotations

import importlib.util
import io
import tarfile
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import ModuleType
from typing import Callable


def _load_checker() -> ModuleType:
    path = Path(__file__).parents[1] / "scripts" / "check_dist_artifacts.py"
    spec = importlib.util.spec_from_file_location("check_dist_artifacts", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load artifact checker from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CHECKER = _load_checker()


class DistributionArtifactTests(unittest.TestCase):
    version = "0.6.0"
    normalized_name = "kapsl_vllm_connector"

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        repository = Path(self.temporary.name) / "repository"
        self.project = repository / "integrations" / "vllm"
        self.dist = repository / "dist"
        self.project.mkdir(parents=True)
        self.dist.mkdir()
        (self.project / "pyproject.toml").write_text(
            "[project]\n"
            'name = "kapsl-vllm-connector"\n'
            f'version = "{self.version}"\n',
            encoding="utf-8",
        )
        for filename, contents in (
            ("LICENSE", "canonical license\n"),
            ("NOTICE", "canonical notice\n"),
        ):
            (repository / filename).write_text(contents, encoding="utf-8")
            (self.project / filename).write_text(contents, encoding="utf-8")
        self.wheel = self.dist / (
            f"{self.normalized_name}-{self.version}-py3-none-any.whl"
        )
        self.sdist = self.dist / (
            f"{self.normalized_name}-{self.version}.tar.gz"
        )
        self._write_wheel()
        self._write_sdist()

    @property
    def dist_info(self) -> str:
        return f"{self.normalized_name}-{self.version}.dist-info"

    @property
    def sdist_root(self) -> str:
        return f"{self.normalized_name}-{self.version}"

    def _wheel_members(self) -> dict[str, bytes]:
        return {
            "kapsl_vllm_connector/plan.py": b"def main(): pass\n",
            "kapsl_vllm_connector/planning.py": b"SCHEMA_VERSION = 1\n",
            f"{self.dist_info}/METADATA": (
                "Metadata-Version: 2.4\n"
                "Name: kapsl-vllm-connector\n"
                f"Version: {self.version}\n\n"
            ).encode(),
            f"{self.dist_info}/entry_points.txt": (
                "[console_scripts]\n"
                "kapsl-vllm-flash-attn-probe = "
                "kapsl_vllm_connector.flash_attn_probe:main\n"
                "kapsl-vllm-plan = kapsl_vllm_connector.plan:main\n"
            ).encode(),
            f"{self.dist_info}/licenses/LICENSE": (
                self.project / "LICENSE"
            ).read_bytes(),
            f"{self.dist_info}/licenses/NOTICE": (
                self.project / "NOTICE"
            ).read_bytes(),
        }

    def _sdist_members(self) -> dict[str, bytes]:
        return {
            f"{self.sdist_root}/LICENSE": (self.project / "LICENSE").read_bytes(),
            f"{self.sdist_root}/NOTICE": (self.project / "NOTICE").read_bytes(),
            f"{self.sdist_root}/pyproject.toml": (
                self.project / "pyproject.toml"
            ).read_bytes(),
            f"{self.sdist_root}/src/kapsl_vllm_connector/plan.py": (
                b"def main(): pass\n"
            ),
            f"{self.sdist_root}/src/kapsl_vllm_connector/planning.py": (
                b"SCHEMA_VERSION = 1\n"
            ),
        }

    def _write_wheel(
        self,
        mutate: Callable[[dict[str, bytes]], None] | None = None,
    ) -> None:
        members = self._wheel_members()
        if mutate is not None:
            mutate(members)
        with zipfile.ZipFile(self.wheel, mode="w") as archive:
            for name, contents in members.items():
                archive.writestr(name, contents)

    def _write_sdist(
        self,
        mutate: Callable[[dict[str, bytes]], None] | None = None,
    ) -> None:
        members = self._sdist_members()
        if mutate is not None:
            mutate(members)
        with tarfile.open(self.sdist, mode="w:gz") as archive:
            for name, contents in members.items():
                info = tarfile.TarInfo(name)
                info.size = len(contents)
                archive.addfile(info, io.BytesIO(contents))

    def _check(self) -> None:
        CHECKER.check_dist(self.dist, self.project)

    def test_accepts_complete_canonical_archives(self) -> None:
        self._check()

    def test_rejects_missing_planner_module_from_wheel(self) -> None:
        self._write_wheel(
            lambda members: members.pop("kapsl_vllm_connector/planning.py")
        )
        with self.assertRaisesRegex(CHECKER.ArtifactError, "planning.py"):
            self._check()

    def test_rejects_missing_planner_module_from_sdist(self) -> None:
        missing = f"{self.sdist_root}/src/kapsl_vllm_connector/plan.py"
        self._write_sdist(lambda members: members.pop(missing))
        with self.assertRaisesRegex(CHECKER.ArtifactError, "plan.py"):
            self._check()

    def test_rejects_missing_console_entry_point(self) -> None:
        entry_points = f"{self.dist_info}/entry_points.txt"

        def remove_planner_entry(members: dict[str, bytes]) -> None:
            members[entry_points] = (
                "[console_scripts]\n"
                "kapsl-vllm-flash-attn-probe = "
                "kapsl_vllm_connector.flash_attn_probe:main\n"
            ).encode()

        self._write_wheel(remove_planner_entry)
        with self.assertRaisesRegex(CHECKER.ArtifactError, "console scripts"):
            self._check()

    def test_rejects_noncanonical_license_and_notice_in_wheel(self) -> None:
        for filename in ("LICENSE", "NOTICE"):
            with self.subTest(filename=filename):
                member = f"{self.dist_info}/licenses/{filename}"
                self._write_wheel(
                    lambda members, member=member: members.__setitem__(
                        member, b"noncanonical\n"
                    )
                )
                with self.assertRaisesRegex(CHECKER.ArtifactError, filename):
                    self._check()
                self._write_wheel()

    def test_rejects_noncanonical_license_and_notice_in_sdist(self) -> None:
        for filename in ("LICENSE", "NOTICE"):
            with self.subTest(filename=filename):
                member = f"{self.sdist_root}/{filename}"
                self._write_sdist(
                    lambda members, member=member: members.__setitem__(
                        member, b"noncanonical\n"
                    )
                )
                with self.assertRaisesRegex(CHECKER.ArtifactError, filename):
                    self._check()
                self._write_sdist()


if __name__ == "__main__":
    unittest.main()
