#!/usr/bin/env python3
"""Tests for the release planner behind .github/workflows/publish.yml."""

import json
import tempfile
import unittest
import urllib.error
from pathlib import Path

import release_plan as rp
from release_plan import Package, ReleaseError


def crate(name, version="1.0.0", deps=(), publish=None, path=None):
    """A package entry shaped like `cargo metadata --no-deps` output."""
    return {
        "id": f"path+file:///repo/{name}#{version}",
        "name": name,
        "version": version,
        "publish": publish,
        "manifest_path": f"/repo/{path or 'crates/' + name}/Cargo.toml",
        "dependencies": [
            {"name": dep, "kind": kind, "req": req, "path": f"/repo/crates/{dep}"}
            for dep, kind, req in (d if isinstance(d, tuple) else (d, None, "^1.0.0") for d in deps)
        ],
    }


def metadata(*crates):
    return {
        "workspace_root": "/repo",
        "workspace_members": [c["id"] for c in crates],
        "packages": list(crates),
    }


def package(name, kind="crate", version="1.0.0", deps=()):
    return Package(name, kind, version, f"crates/{name}", [f"crates/{name}"], [], deps=list(deps))


class VersionTests(unittest.TestCase):
    def test_only_stable_versions_parse(self):
        self.assertEqual(rp.parse_version("0.12.3"), (0, 12, 3))
        for text in ("1.0", "1.0.0-rc.1", "01.0.0", "v1.0.0"):
            self.assertIsNone(rp.parse_version(text))

    def test_newest_compares_numerically(self):
        self.assertEqual(rp.newest({"0.9.0", "0.10.0", "0.2.0"}), "0.10.0")
        self.assertIsNone(rp.newest(set()))

    def test_crates_index_paths(self):
        self.assertEqual(rp.crates_index_path("a"), "1/a")
        self.assertEqual(rp.crates_index_path("ab"), "2/ab")
        self.assertEqual(rp.crates_index_path("abc"), "3/a/abc")
        self.assertEqual(rp.crates_index_path("Kapsl-HAL"), "ka/ps/kapsl-hal")


class BaselineTests(unittest.TestCase):
    def test_newest_published_tag_wins(self):
        tags = ["kapsl-hal-v0.3.1", "kapsl-hal-v0.3.10", "kapsl-hal-v0.3.2", "kapsl-hal-v0.4.0"]
        # 0.4.0 was tagged but never reached the registry, so it is no baseline.
        self.assertEqual(rp.baseline_tag("kapsl-hal", tags, {"0.3.1", "0.3.2", "0.3.10"}), "kapsl-hal-v0.3.10")

    def test_tags_of_packages_sharing_a_prefix_are_ignored(self):
        tags = ["kapsl-sdk-v0.2.0", "kapsl-vllm-connector-v0.7.0", "kapsl-v1.0.0-rc.1", "v0.1.23"]
        self.assertIsNone(rp.baseline_tag("kapsl", tags, {"0.7.0", "0.2.0", "0.1.23"}))
        self.assertEqual(rp.baseline_tag("kapsl-sdk", tags, {"0.2.0"}), "kapsl-sdk-v0.2.0")


class DecideTests(unittest.TestCase):
    def decide(self, published, baseline="hal-v0.9.0", changed=True, requested=None, version="1.0.0"):
        return rp.decide(package("hal", version=version), published, baseline, changed, requested)

    def test_changed_package_with_new_version_is_released(self):
        decision = self.decide({"0.9.0"})
        self.assertEqual((decision.action, decision.latest, decision.new), ("release", "0.9.0", False))

    def test_never_published_package_is_a_new_release(self):
        decision = self.decide(None, baseline=None, changed=None)
        self.assertEqual((decision.action, decision.new), ("release", True))

    def test_unpublished_version_without_a_tag_is_released(self):
        self.assertEqual(self.decide({"0.9.0"}, baseline=None, changed=None).action, "release")

    def test_change_without_version_bump_is_flagged(self):
        decision = self.decide({"1.0.0"}, baseline="hal-v1.0.0")
        self.assertEqual(decision.action, "bump")
        self.assertIn("bump the version", decision.reason)

    def test_unchanged_package_is_skipped(self):
        self.assertEqual(self.decide({"1.0.0"}, baseline="hal-v1.0.0", changed=False).action, "skip")
        self.assertEqual(self.decide({"1.0.0"}, baseline=None, changed=None).action, "skip")

    def test_version_bump_without_changes_is_skipped(self):
        # A workspace-wide version bump does not touch the crate itself.
        decision = self.decide({"0.9.0"}, changed=False)
        self.assertEqual(decision.action, "skip")
        self.assertIn("list it explicitly", decision.reason)

    def test_requested_package_is_released_even_if_unchanged(self):
        self.assertEqual(self.decide({"0.9.0"}, changed=False, requested={"hal"}).action, "release")
        self.assertEqual(self.decide({"0.9.0"}, requested={"other"}).action, "skip")

    def test_requested_package_must_have_an_unpublished_version(self):
        with self.assertRaisesRegex(ReleaseError, "already on crates.io"):
            self.decide({"1.0.0"}, requested={"hal"})


class OrderTests(unittest.TestCase):
    def test_dependencies_publish_first_in_stable_order(self):
        crates = [
            package("llm", deps=["core", "abi"]),
            package("core", deps=["abi"]),
            package("abi"),
            package("zeta"),
            package("hal", deps=["unreleased"]),
        ]
        self.assertEqual([c.name for c in rp.publish_order(crates)], ["abi", "hal", "zeta", "core", "llm"])

    def test_cycles_are_reported(self):
        with self.assertRaisesRegex(ReleaseError, "cycle"):
            rp.publish_order([package("a", deps=["b"]), package("b", deps=["a"])])


class LoadTests(unittest.TestCase):
    def setUp(self):
        self.workspace = metadata(
            crate("abi"),
            crate("api", deps=["abi"]),
            crate("client", deps=["api", ("abi", "dev", "*"), ("tools", "dev", "^1.0.0")]),
            crate("tools"),
            crate("pyo3", deps=["client"], publish=[]),
        )
        self.config = {name: {"kind": "crate"} for name in ("abi", "api", "client", "tools")}

    def test_crates_get_paths_dependencies_and_default_checks(self):
        packages = rp.load_packages(self.config, self.workspace)
        client = packages["client"]
        self.assertEqual(client.path, "crates/client")
        self.assertEqual(client.watch, ["crates/client"])
        # A versionless dev-dependency is dropped on publish; a versioned one is kept.
        self.assertEqual(client.deps, ["api", "tools"])
        self.assertEqual(client.checks, ["cargo test -p client --locked"])

    def test_every_publishable_crate_must_be_declared(self):
        del self.config["tools"]
        with self.assertRaisesRegex(ReleaseError, "missing .*tools"):
            rp.load_packages(self.config, self.workspace)

    def test_unpublishable_and_unknown_crates_are_refused(self):
        for name in ("pyo3", "ghost"):
            with self.subTest(name=name), self.assertRaises(ReleaseError):
                rp.load_packages({**self.config, name: {"kind": "crate"}}, self.workspace)

    def test_python_package_watches_its_crate_and_its_workspace_dependencies(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "py").mkdir()
            (root / "pyproject.toml").write_text('[project]\nname = "sdk"\nversion = "0.2.0"\n')
            (root / "py" / "pyproject.toml").write_text('[project]\nname = "sdk"\nversion = "0.2.0"\n')
            config = {**self.config, "sdk": {
                "kind": "python", "path": ".", "build": "maturin", "crate": "pyo3",
                "watch": ["pyproject.toml"], "version-files": ["py/pyproject.toml"],
            }}
            sdk = rp.load_packages(config, self.workspace, root)["sdk"]
            self.assertEqual((sdk.version, sdk.checks), ("0.2.0", []))
            self.assertEqual(
                sdk.watch,
                ["pyproject.toml", "crates/abi", "crates/api", "crates/client", "crates/pyo3"],
            )

            (root / "py" / "pyproject.toml").write_text('[project]\nname = "sdk"\nversion = "0.1.0"\n')
            with self.assertRaisesRegex(ReleaseError, "has version '0.1.0'"):
                rp.load_packages(config, self.workspace, root)

    def test_committed_config_is_well_formed(self):
        config = rp.load_config()
        self.assertTrue(config)
        for name, settings in config.items():
            if settings["kind"] == "python":
                self.assertIn(settings.get("build"), rp.PYTHON_BUILDS, name)

    def test_unknown_settings_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "release.toml"
            path.write_text('[crate.abi]\nchecs = ["cargo test"]\n')
            with self.assertRaisesRegex(ReleaseError, "unknown settings"):
                rp.load_config(path)


class PlanTests(unittest.TestCase):
    def setUp(self):
        self.packages = {
            "abi": package("abi", version="1.1.0"),
            "llm": package("llm", version="2.0.0", deps=["abi"]),
            "hal": package("hal", version="3.0.0"),
            "sdk": package("sdk", kind="python", version="0.2.0"),
        }
        self.facts = {
            "abi": ({"1.0.0"}, "abi-v1.0.0", True),
            "llm": ({"1.0.0"}, "llm-v1.0.0", True),
            "hal": ({"3.0.0"}, "hal-v3.0.0", True),
            "sdk": (None, None, None),
        }

    def plan(self, requested=None, publish=True, ref="refs/heads/main"):
        return rp.build_plan(self.packages, requested, publish, ref, lambda p: self.facts[p.name])

    def test_changed_packages_are_planned_in_dependency_order(self):
        decisions, order, errors = self.plan()
        self.assertEqual(errors, [])
        actions = {d.package.name: d.action for d in decisions}
        self.assertEqual(actions, {"abi": "release", "hal": "bump", "llm": "release", "sdk": "release"})
        self.assertEqual([c.name for c in order], ["abi", "llm"])

        outputs = rp.plan_outputs(decisions, order, publish=True)
        self.assertEqual(outputs["crate-order"], "abi llm")
        self.assertEqual(outputs["new-crates"], "false")
        self.assertEqual(outputs["existing-crates"], "true")
        self.assertEqual(outputs["names"], '["abi", "llm", "sdk"]')
        self.assertIn('"name": "sdk"', outputs["python"])
        self.assertEqual(outputs["setuptools"], "[]")

    def test_publishing_requires_main(self):
        _, _, errors = self.plan(ref="refs/heads/feature/x")
        self.assertTrue(any("only be published from main" in e for e in errors))
        self.assertEqual(self.plan(publish=False, ref="refs/heads/feature/x")[2], [])

    def test_releases_need_stable_versions(self):
        self.packages["abi"].version = "1.1.0-rc.1"
        _, _, errors = self.plan()
        self.assertTrue(any("stable X.Y.Z" in e for e in errors))

    def test_explicit_requests_are_validated(self):
        _, _, errors = self.plan(requested={"hal", "nope"})
        self.assertEqual(len(errors), 2)
        self.assertIn("unknown packages: nope", errors[0])
        self.assertIn("hal 3.0.0 is already on crates.io", errors[1])

    def test_request_parsing(self):
        self.assertIsNone(rp.parse_request(" changed "))
        self.assertEqual(rp.parse_request("kapsl-hal, kapsl-llm  kapsl-sdk"), {"kapsl-hal", "kapsl-llm", "kapsl-sdk"})


class ChecksTests(unittest.TestCase):
    def test_each_check_runs_in_its_own_subshell(self):
        script = rp.checks_script("hal", ["cargo test -p hal --locked", "A=1 \\\n  run 'x'\n"])
        self.assertTrue(script.startswith("set -euo pipefail\n"))
        self.assertIn("echo '::group::cargo test -p hal --locked'\n(\ncargo test -p hal --locked\n)\n", script)
        self.assertIn("(\nA=1 \\\n  run 'x'\n)\n", script)
        self.assertIn("has no release checks", rp.checks_script("sdk", []))

    def test_config_checks_override_the_default(self):
        self.assertEqual(rp.package_checks("hal", {"kind": "crate"}), ["cargo test -p hal --locked"])
        self.assertEqual(rp.package_checks("hal", {"kind": "crate", "checks": []}), [])


class PublishCratesTests(unittest.TestCase):
    def test_publishes_in_order_with_the_right_token_and_skips_published(self):
        calls = []
        registry = {"abi": {"1.0.0", "1.1.0"}, "llm": {"1.0.0"}, "new": None}
        crates = [
            {"name": "abi", "version": "1.1.0", "new": False},
            {"name": "new", "version": "0.1.0", "new": True},
            {"name": "llm", "version": "2.0.0", "new": False},
        ]
        rp.publish_crates(
            crates, "oidc", "bootstrap",
            run=lambda cmd, **kw: calls.append((cmd[3], kw["env"]["CARGO_REGISTRY_TOKEN"])),
            lookup=lambda kind, name: registry[name],
        )
        self.assertEqual(calls, [("new", "bootstrap"), ("llm", "oidc")])

    def test_first_release_without_a_token_fails_before_publishing(self):
        with self.assertRaisesRegex(ReleaseError, "CRATES_IO_TOKEN"):
            rp.publish_crates(
                [{"name": "new", "version": "0.1.0", "new": True}], "oidc", "",
                run=lambda *a, **k: self.fail("must not publish"), lookup=lambda kind, name: None,
            )


class FakeGitHub:
    def __init__(self, tags=None):
        self.tags = dict(tags or {})

    def tag_commit(self, tag):
        return self.tags.get(tag)

    def create_tag(self, tag, sha, message):
        self.tags[tag] = sha


class TagTests(unittest.TestCase):
    releases = [
        {"name": "abi", "kind": "crate", "version": "1.1.0"},
        {"name": "llm", "kind": "crate", "version": "2.0.0"},
        {"name": "sdk", "kind": "python", "version": "0.2.0"},
    ]

    def test_tags_only_what_reached_the_registry(self):
        registry = {"abi": {"1.1.0"}, "llm": {"1.0.0"}, "sdk": {"0.2.0"}}
        github = FakeGitHub({"sdk-v0.2.0": "abc"})
        sleeps = []
        created = rp.tag_releases(
            self.releases, "abc", {"crate": "failure", "python": "success"}, github,
            lookup=lambda kind, name: registry[name], sleep=sleeps.append,
        )
        self.assertEqual(created, ["abi-v1.1.0"])
        self.assertNotIn("llm-v2.0.0", github.tags)
        self.assertEqual(sleeps, [])  # failed jobs are not polled

    def test_waits_for_the_registry_after_a_successful_publish(self):
        answers = iter([set(), set(), {"1.1.0"}])
        sleeps = []
        created = rp.tag_releases(
            self.releases[:1], "abc", {"crate": "success"}, FakeGitHub(),
            lookup=lambda kind, name: next(answers), sleep=sleeps.append,
        )
        self.assertEqual((created, len(sleeps)), (["abi-v1.1.0"], 2))

    def test_a_tag_on_another_commit_is_an_error(self):
        with self.assertRaisesRegex(ReleaseError, "already exists"):
            rp.tag_releases(
                self.releases[:1], "abc", {"crate": "success"}, FakeGitHub({"abi-v1.1.0": "def"}),
                lookup=lambda kind, name: {"1.1.0"}, sleep=lambda s: None,
            )


class FakeResponse:
    def __init__(self, body):
        self.body = json.dumps(body).encode()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self):
        return self.body


def http_error(code):
    return urllib.error.HTTPError("https://api.github.test", code, "error", None, None)


class GitHubClientTests(unittest.TestCase):
    def client(self, answers):
        """A client whose requests get `answers` in turn: an exception or a JSON body."""
        self.requests, self.sleeps = [], []
        answers = iter(answers)

        def urlopen(request, timeout):
            self.requests.append((request.get_method(), request.full_url.rsplit("/repos/o/r", 1)[1]))
            answer = next(answers)
            if isinstance(answer, Exception):
                raise answer
            return FakeResponse(answer)

        return rp.GitHub("o/r", "token", "https://api.github.test", attempts=4, sleep=self.sleeps.append, urlopen=urlopen)

    def test_server_errors_rate_limits_and_network_failures_are_retried(self):
        github = self.client([http_error(500), http_error(429), urllib.error.URLError("reset"), {"ok": True}])
        self.assertEqual(github.request("GET", "/x"), {"ok": True})
        self.assertEqual(self.sleeps, [5, 10, 20])

    def test_client_errors_are_not_retried(self):
        github = self.client([http_error(422)])
        with self.assertRaises(urllib.error.HTTPError) as raised:
            github.request("POST", "/git/refs", {})
        raised.exception.close()
        self.assertEqual((len(self.requests), self.sleeps), (1, []))

    def test_gives_up_after_the_last_attempt(self):
        github = self.client([http_error(500) for _ in range(4)])
        with self.assertRaises(urllib.error.HTTPError) as raised:
            github.request("GET", "/x")
        raised.exception.close()
        self.assertEqual(len(self.requests), 4)

    def test_a_missing_tag_is_none_without_retrying(self):
        self.assertIsNone(self.client([http_error(404)]).tag_commit("abi-v1.0.0"))
        self.assertEqual(self.sleeps, [])

    def test_creating_a_tag_tolerates_a_ref_made_by_an_attempt_that_seemed_to_fail(self):
        github = self.client([
            {"sha": "tagobj"},                                   # POST /git/tags
            http_error(502),                                     # POST /git/refs: created, but answered 502
            http_error(422),                                     # retry: the ref already exists
            {"object": {"type": "tag", "sha": "tagobj"}},        # GET /git/ref/tags/abi-v1.0.0
            {"object": {"type": "commit", "sha": "abc"}},        # GET /git/tags/tagobj
        ])
        github.create_tag("abi-v1.0.0", "abc", "abi 1.0.0")
        self.assertEqual([path for _, path in self.requests][-2:], ["/git/ref/tags/abi-v1.0.0", "/git/tags/tagobj"])

    def test_an_existing_ref_on_another_commit_is_still_an_error(self):
        github = self.client([
            {"sha": "tagobj"},
            http_error(422),
            {"object": {"type": "commit", "sha": "def"}},
        ])
        with self.assertRaises(urllib.error.HTTPError) as raised:
            github.create_tag("abi-v1.0.0", "abc", "abi 1.0.0")
        raised.exception.close()


if __name__ == "__main__":
    unittest.main()
