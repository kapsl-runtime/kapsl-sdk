#!/usr/bin/env python3
"""Plan, check, publish and tag SDK releases for .github/workflows/publish.yml.

Every releasable package is declared in .github/release-packages.toml. For each
one the planner compares the version in the repository with the versions on
its registry (crates.io or PyPI), and its files with its last release tag, and
decides whether to release it.

  plan            decide what to release; writes GITHUB_OUTPUT and the summary
  checks          write a bash script that runs one package's release checks
  publish-crates  publish the planned crates in dependency order
  tag             tag every planned package that is now on its registry
  validate        check the release config against the Cargo workspace

RELEASING.md describes the release process.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import time
import tomllib
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path, PurePath

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / ".github" / "release-packages.toml"
STABLE_VERSION = re.compile(r"(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)")
REGISTRY = {"crate": "crates.io", "python": "PyPI"}
PYTHON_BUILDS = ("maturin", "setuptools")
SETTINGS = {
    "crate": {"checks", "submodules"},
    "python": {"path", "build", "watch", "crate", "version-files", "checks", "r2-prefix"},
}
USER_AGENT = "kapsl-sdk-release-plan (github.com/kapsl-runtime/kapsl-sdk)"


class ReleaseError(Exception):
    """The release config or the release request breaks the release policy."""


@dataclass
class Package:
    name: str
    kind: str  # "crate" or "python"
    version: str
    path: str  # project directory, relative to the repository root
    watch: list[str]  # changes under these paths are changes to the package
    checks: list[str]
    submodules: bool = False
    build: str = ""  # python: "maturin" or "setuptools"
    r2_prefix: str = ""  # python: also upload to R2 under <prefix>/v<version>/
    deps: list[str] = field(default_factory=list)  # crate: workspace crates it needs on crates.io

    @property
    def registry(self) -> str:
        return REGISTRY[self.kind]

    @property
    def tag(self) -> str:
        return f"{self.name}-v{self.version}"


@dataclass
class Decision:
    package: Package
    action: str  # "release", "bump" (changed, but its version is published) or "skip"
    reason: str
    latest: str | None = None  # newest version on the registry
    baseline: str | None = None  # last release tag, the base for change detection
    new: bool = False  # the registry has never seen the package


# Versions -------------------------------------------------------------------


def parse_version(text: str) -> tuple[int, ...] | None:
    match = STABLE_VERSION.fullmatch(text)
    return tuple(int(part) for part in match.groups()) if match else None


def newest(versions: set[str]) -> str | None:
    stable = [version for version in versions if parse_version(version)]
    if stable:
        return max(stable, key=parse_version)
    return max(versions) if versions else None


# Configuration --------------------------------------------------------------


def load_config(path: Path = CONFIG_PATH) -> dict[str, dict]:
    """Return {name: settings}, with each package's kind under "kind"."""
    with path.open("rb") as handle:
        raw = tomllib.load(handle)
    unknown = sorted(set(raw) - set(SETTINGS))
    if unknown:
        raise ReleaseError(f"{path.name}: unknown sections {unknown}; use [crate.<name>] or [python.<name>]")
    config: dict[str, dict] = {}
    for kind, allowed in SETTINGS.items():
        for name, settings in raw.get(kind, {}).items():
            if name in config:
                raise ReleaseError(f"{name} is declared twice in {path.name}")
            extra = sorted(set(settings) - allowed)
            if extra:
                raise ReleaseError(f"{kind}.{name}: unknown settings {extra}")
            checks = settings.get("checks", [])
            if not isinstance(checks, list) or not all(isinstance(check, str) for check in checks):
                raise ReleaseError(f"{kind}.{name}: checks must be a list of strings")
            config[name] = {"kind": kind, **settings}
    return config


def default_checks(name: str, kind: str) -> list[str]:
    return [f"cargo test -p {name} --locked"] if kind == "crate" else []


def package_checks(name: str, settings: dict) -> list[str]:
    checks = settings.get("checks")
    return list(checks) if checks is not None else default_checks(name, settings["kind"])


def workspace_crates(metadata: dict) -> dict[str, dict]:
    members = set(metadata["workspace_members"])
    return {package["name"]: package for package in metadata["packages"] if package["id"] in members}


def crate_dir(crate: dict, workspace_root: PurePath) -> str:
    return PurePath(crate["manifest_path"]).parent.relative_to(workspace_root).as_posix()


def publishable(crate: dict) -> bool:
    # cargo metadata reports `publish = false` as [] and an unset field as null.
    return crate["publish"] is None or "crates-io" in crate["publish"]


def crate_dependencies(crate: dict, crates: dict[str, dict]) -> list[str]:
    """Workspace crates that must be on crates.io before `crate` is published."""
    needed = set()
    for dependency in crate["dependencies"]:
        name = dependency["name"]
        if name not in crates or name == crate["name"]:
            continue
        # Cargo drops a dev-dependency without a version from the published
        # manifest; every other workspace dependency stays in it.
        if dependency.get("kind") == "dev" and dependency.get("req", "*") == "*":
            continue
        needed.add(name)
    return sorted(needed)


def workspace_closure(name: str, crates: dict[str, dict]) -> set[str]:
    """`name` and every workspace crate it builds with (dev-dependencies aside)."""
    seen: set[str] = set()
    pending = [name]
    while pending:
        current = pending.pop()
        if current in seen:
            continue
        seen.add(current)
        pending.extend(
            dependency["name"]
            for dependency in crates[current]["dependencies"]
            if dependency["name"] in crates and dependency.get("kind") != "dev"
        )
    return seen


def read_project(path: Path) -> dict:
    try:
        with path.open("rb") as handle:
            return tomllib.load(handle)["project"]
    except FileNotFoundError:
        raise ReleaseError(f"{path} does not exist") from None
    except KeyError:
        raise ReleaseError(f"{path} has no [project] table") from None


def load_python(name: str, settings: dict, crates: dict, workspace_root: PurePath, root: Path) -> Package:
    path = settings.get("path")
    build = settings.get("build")
    if not path:
        raise ReleaseError(f"python.{name}: set path to the directory holding its pyproject.toml")
    if build not in PYTHON_BUILDS:
        raise ReleaseError(f"python.{name}: build must be one of {list(PYTHON_BUILDS)}")
    pyproject = PurePath(path, "pyproject.toml").as_posix()
    project = read_project(root / pyproject)
    if project.get("name") != name:
        raise ReleaseError(f"python.{name}: {pyproject} names the project {project.get('name')!r}")
    version = project.get("version")
    if not version:
        raise ReleaseError(f"python.{name}: {pyproject} needs a static version")
    for other in settings.get("version-files", []):
        other_version = read_project(root / other).get("version")
        if other_version != version:
            raise ReleaseError(
                f"python.{name}: {other} has version {other_version!r} but {pyproject} has {version!r}"
            )
    watch = list(settings.get("watch", [path]))
    crate = settings.get("crate")
    if crate:
        if crate not in crates:
            raise ReleaseError(f"python.{name}: crate {crate!r} is not a workspace crate")
        watch += sorted(crate_dir(crates[member], workspace_root) for member in workspace_closure(crate, crates))
    return Package(
        name=name,
        kind="python",
        version=version,
        path=PurePath(path).as_posix(),
        watch=list(dict.fromkeys(watch)),
        checks=package_checks(name, settings),
        build=build,
        r2_prefix=settings.get("r2-prefix", ""),
    )


def load_packages(config: dict[str, dict], metadata: dict, root: Path = ROOT) -> dict[str, Package]:
    crates = workspace_crates(metadata)
    workspace_root = PurePath(metadata["workspace_root"])
    packages: dict[str, Package] = {}
    for name, settings in config.items():
        if settings["kind"] == "python":
            packages[name] = load_python(name, settings, crates, workspace_root, root)
            continue
        crate = crates.get(name)
        if crate is None:
            raise ReleaseError(f"crate.{name} is not a crate in the Cargo workspace")
        if not publishable(crate):
            raise ReleaseError(f"crate.{name} cannot be published to crates.io (see `publish` in its Cargo.toml)")
        path = crate_dir(crate, workspace_root)
        packages[name] = Package(
            name=name,
            kind="crate",
            version=crate["version"],
            path=path,
            watch=[path],
            checks=package_checks(name, settings),
            submodules=bool(settings.get("submodules", False)),
            deps=crate_dependencies(crate, crates),
        )
    undeclared = sorted(name for name, crate in crates.items() if publishable(crate) and name not in config)
    if undeclared:
        raise ReleaseError(
            f"publishable workspace crates missing from {CONFIG_PATH.name}: {', '.join(undeclared)}; "
            "declare each under [crate.<name>], or set `publish = false` in its Cargo.toml"
        )
    return packages


def cargo_metadata(path: Path | None) -> dict:
    if path is not None:
        return json.loads(path.read_text(encoding="utf-8"))
    output = subprocess.run(
        ["cargo", "metadata", "--no-deps", "--format-version", "1"],
        cwd=ROOT, check=True, capture_output=True, text=True,
    ).stdout
    return json.loads(output)


# Registries -----------------------------------------------------------------


def http_get(url: str, headers: dict[str, str] | None = None, attempts: int = 3) -> str | None:
    """Return the response body, or None when the server answers 404."""
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, **(headers or {})})
    for attempt in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return response.read().decode("utf-8")
        except urllib.error.HTTPError as error:
            if error.code == 404:
                return None
            if attempt == attempts or not (error.code == 429 or error.code >= 500):
                raise
        except (urllib.error.URLError, TimeoutError):
            if attempt == attempts:
                raise
        time.sleep(2 * attempt)
    raise AssertionError("unreachable")


def crates_index_path(name: str) -> str:
    """The crate's file in the crates.io sparse index."""
    name = name.lower()
    if len(name) <= 2:
        return f"{len(name)}/{name}"
    if len(name) == 3:
        return f"3/{name[0]}/{name}"
    return f"{name[:2]}/{name[2:4]}/{name}"


def registry_versions(kind: str, name: str) -> set[str] | None:
    """Every version on the package's registry, or None if it was never published.

    Yanked crate versions count: their version numbers cannot be reused.
    """
    if kind == "crate":
        body = http_get(f"https://index.crates.io/{crates_index_path(name)}")
        if body is None:
            return None
        return {json.loads(line)["vers"] for line in body.splitlines() if line.strip()}
    project = re.sub(r"[-_.]+", "-", name).lower()
    body = http_get(
        f"https://pypi.org/simple/{project}/",
        headers={"Accept": "application/vnd.pypi.simple.v1+json"},
    )
    if body is None:
        return None
    return set(json.loads(body)["versions"])


# Git ------------------------------------------------------------------------


def git(*args: str, root: Path = ROOT) -> str:
    return subprocess.run(["git", *args], cwd=root, check=True, capture_output=True, text=True).stdout


def baseline_tag(name: str, tags: list[str], published: set[str]) -> str | None:
    """The newest `<name>-v<version>` tag whose version is on the registry."""
    prefix = f"{name}-v"
    best: tuple[tuple[int, ...], str] | None = None
    for tag in tags:
        if not tag.startswith(prefix):
            continue
        version = tag[len(prefix):]
        parsed = parse_version(version)
        if parsed and version in published and (best is None or parsed > best[0]):
            best = (parsed, tag)
    return best[1] if best else None


def changed_since(tag: str, paths: list[str], root: Path = ROOT) -> bool:
    return bool(git("diff", "--name-only", tag, "HEAD", "--", *paths, root=root).strip())


# Planning -------------------------------------------------------------------


def parse_request(text: str) -> set[str] | None:
    """None for "changed"; otherwise the package names listed."""
    text = text.strip()
    if text in ("", "changed"):
        return None
    return {name for name in re.split(r"[\s,]+", text) if name}


def decide(
    package: Package,
    published: set[str] | None,
    baseline: str | None,
    changed: bool | None,
    requested: set[str] | None,
) -> Decision:
    """Decide whether to release `package`.

    `published` is None if the registry has never seen the package, and
    `changed` is None when there is no release tag to compare with.
    """
    versions = published or set()
    on_registry = package.version in versions
    registry = package.registry

    def result(action: str, reason: str) -> Decision:
        return Decision(package, action, reason, newest(versions), baseline, published is None)

    if requested is not None:
        if package.name not in requested:
            return result("skip", "not requested")
        if on_registry:
            raise ReleaseError(
                f"{package.name} {package.version} is already on {registry}; bump its version to release it"
            )
        return result("release", "requested")
    if on_registry:
        if changed:
            return result(
                "bump",
                f"changed since {baseline}, but {package.version} is already on {registry}; "
                "bump the version to release it",
            )
        return result("skip", "up to date")
    if changed is False:
        return result(
            "skip",
            f"{package.version} is not on {registry}, but nothing it ships changed since {baseline}; "
            "list it explicitly to release it anyway",
        )
    if published is None:
        return result("release", f"first release on {registry}")
    if changed:
        return result("release", f"changed since {baseline}")
    return result("release", f"{package.version} is not on {registry} (no release tag to compare with)")


def publish_order(crates: list[Package]) -> list[Package]:
    """Order crates so each one follows the workspace crates it depends on."""
    remaining = {crate.name: crate for crate in crates}
    ordered: list[Package] = []
    while remaining:
        ready = sorted(name for name, crate in remaining.items() if not any(dep in remaining for dep in crate.deps))
        if not ready:
            raise ReleaseError(f"dependency cycle between {', '.join(sorted(remaining))}")
        ordered.extend(remaining.pop(name) for name in ready)
    return ordered


def build_plan(packages, requested, publish, ref, facts):
    """Return (decisions, crate publish order, errors).

    `facts(package)` returns (registry versions or None, baseline tag, changed).
    """
    errors: list[str] = []
    if requested is not None:
        unknown = sorted(requested - packages.keys())
        if unknown:
            errors.append(f"unknown packages: {', '.join(unknown)}; choose from: {', '.join(sorted(packages))}")
        if not requested:
            errors.append('list at least one package, or use "changed"')
    decisions: list[Decision] = []
    for name in sorted(packages):
        package = packages[name]
        published, baseline, changed = facts(package)
        try:
            decisions.append(decide(package, published, baseline, changed, requested))
        except ReleaseError as error:
            errors.append(str(error))
            decisions.append(Decision(package, "error", str(error), newest(published or set()), baseline))
    releases = [decision for decision in decisions if decision.action == "release"]
    for decision in releases:
        if parse_version(decision.package.version) is None:
            errors.append(f"{decision.package.name} {decision.package.version}: releases need a stable X.Y.Z version")
    if publish and ref != "refs/heads/main":
        errors.append(f"packages can only be published from main, not {ref or 'an unknown ref'}")
    order: list[Package] = []
    try:
        order = publish_order([decision.package for decision in releases if decision.package.kind == "crate"])
    except ReleaseError as error:
        errors.append(str(error))
    return decisions, order, errors


def plan_outputs(decisions: list[Decision], order: list[Package], publish: bool) -> dict[str, str]:
    releases = [decision for decision in decisions if decision.action == "release"]
    new = {decision.package.name for decision in releases if decision.new}
    python = [decision.package for decision in releases if decision.package.kind == "python"]

    def python_entry(package: Package) -> dict:
        return {"name": package.name, "version": package.version, "path": package.path, "r2_prefix": package.r2_prefix}

    def flag(value: bool) -> str:
        return "true" if value else "false"

    return {
        "publish": flag(publish),
        "release": flag(bool(releases)),
        "packages": json.dumps(
            [{"name": d.package.name, "kind": d.package.kind, "version": d.package.version} for d in releases]
        ),
        "names": json.dumps([decision.package.name for decision in releases]),
        "crates": json.dumps(
            [{"name": c.name, "version": c.version, "submodules": c.submodules, "new": c.name in new} for c in order]
        ),
        "crate-order": " ".join(crate.name for crate in order),
        "crate-submodules": flag(any(crate.submodules for crate in order)),
        "new-crates": flag(any(crate.name in new for crate in order)),
        "existing-crates": flag(any(crate.name not in new for crate in order)),
        "python": json.dumps([python_entry(package) for package in python]),
        "maturin": json.dumps([python_entry(p) for p in python if p.build == "maturin"]),
        "setuptools": json.dumps([python_entry(p) for p in python if p.build == "setuptools"]),
    }


def summary(decisions, order, errors, requested, publish) -> str:
    mode = "packages changed since their last release" if requested is None else ", ".join(sorted(requested))
    lines = [
        "## Release plan",
        "",
        f"- Selection: {mode}",
        f"- Publish: {'yes, once the sdk-release environment is approved' if publish else 'no (dry run)'}",
        "",
        "| Package | Registry | Version | Newest published | Last release tag | Decision |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    labels = {"release": "**release**", "bump": "⚠️ needs version bump", "skip": "skip", "error": "❌ error"}
    for decision in decisions:
        package = decision.package
        lines.append(
            f"| {package.name} | {package.registry} | {package.version} | {decision.latest or 'never'} "
            f"| {decision.baseline or '-'} | {labels[decision.action]}: {decision.reason} |"
        )
    lines.append("")
    releases = [decision for decision in decisions if decision.action == "release"]
    if order:
        lines.append(f"Crates publish in this order: {' → '.join(crate.name for crate in order)}")
    if not releases and not errors:
        lines.append("Nothing to release.")
    if errors:
        lines += ["", "### Errors", "", *(f"- {error}" for error in errors)]
    return "\n".join(lines) + "\n"


def write_github_file(variable: str, text: str) -> None:
    path = os.environ.get(variable)
    if path:
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(text)


def command_plan(args) -> int:
    requested = parse_request(args.packages)
    publish = args.publish == "true"
    packages = load_packages(load_config(), cargo_metadata(args.metadata))
    tags = git("tag", "--list").split()

    def facts(package: Package):
        published = registry_versions(package.kind, package.name)
        baseline = baseline_tag(package.name, tags, published or set())
        changed = changed_since(baseline, package.watch) if baseline else None
        return published, baseline, changed

    decisions, order, errors = build_plan(packages, requested, publish, args.ref, facts)
    report = summary(decisions, order, errors, requested, publish)
    print(report)
    write_github_file("GITHUB_STEP_SUMMARY", report)
    for decision in decisions:
        if decision.action == "bump":
            print(f"::warning title={decision.package.name} needs a version bump::{decision.reason}")
    if errors:
        for error in errors:
            print(f"::error title=Release plan::{error}")
        return 1
    outputs = plan_outputs(decisions, order, publish)
    write_github_file("GITHUB_OUTPUT", "".join(f"{key}={value}\n" for key, value in outputs.items()))
    for key, value in outputs.items():
        print(f"{key}={value}")
    return 0


# Checks ---------------------------------------------------------------------


def checks_script(name: str, checks: list[str]) -> str:
    """A bash script running each check in its own subshell, as a log group."""
    lines = ["set -euo pipefail"]
    if not checks:
        lines.append(f"echo {shlex.quote(f'{name} has no release checks')}")
    for check in checks:
        title = check.strip().splitlines()[0]
        lines += [f"echo {shlex.quote(f'::group::{title}')}", "(", check.strip("\n"), ")", "echo '::endgroup::'"]
    return "\n".join(lines) + "\n"


def command_checks(args) -> int:
    config = load_config()
    if args.package not in config:
        raise ReleaseError(f"{args.package} is not declared in {CONFIG_PATH.name}")
    script = checks_script(args.package, package_checks(args.package, config[args.package]))
    # newline="\n" keeps the script runnable by bash on Windows runners.
    with open(args.output, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(script)
    print(script)
    return 0


# Publishing -----------------------------------------------------------------


def publish_crates(crates: list[dict], oidc_token: str, bootstrap_token: str, run=subprocess.run, lookup=registry_versions):
    """Publish crates in the planned order, skipping versions already published.

    Skipping makes a re-run of a partly finished job safe. Crates were verified
    by `cargo publish --dry-run` in the same run and commit, so they are not
    rebuilt here.
    """
    for crate in crates:
        name, version = crate["name"], crate["version"]
        if version in (lookup("crate", name) or set()):
            print(f"{name} {version} is already on crates.io; skipping")
            continue
        token = bootstrap_token if crate["new"] else oidc_token
        if not token:
            reason = (
                "a first release needs the CRATES_IO_TOKEN secret of the sdk-release environment"
                if crate["new"]
                else "crates.io trusted publishing returned no token"
            )
            raise ReleaseError(f"cannot publish {name} {version}: {reason}")
        print(f"::group::Publishing {name} {version}", flush=True)
        run(
            ["cargo", "publish", "-p", name, "--locked", "--no-verify"],
            cwd=ROOT, check=True, env={**os.environ, "CARGO_REGISTRY_TOKEN": token},
        )
        print("::endgroup::", flush=True)


def command_publish_crates(args) -> int:
    publish_crates(
        json.loads(os.environ["CRATES"]),
        os.environ.get("OIDC_TOKEN", ""),
        os.environ.get("BOOTSTRAP_TOKEN", ""),
    )
    return 0


# Tagging --------------------------------------------------------------------


class GitHub:
    def __init__(
        self,
        repository: str,
        token: str,
        api: str = "https://api.github.com",
        attempts: int = 8,
        sleep=time.sleep,
        urlopen=urllib.request.urlopen,
    ):
        self.base = f"{api}/repos/{repository}"
        self.headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": USER_AGENT,
        }
        self.attempts = attempts
        self.sleep = sleep
        self.urlopen = urlopen

    def request(self, method: str, path: str, body: dict | None = None):
        """Send an API request, retrying rate limits, server errors and network failures.

        The tags are made after the packages are already on their registries,
        so a brief GitHub outage should delay the tag, not fail the release.
        The waits add up to about four minutes.
        """
        data = json.dumps(body).encode() if body is not None else None
        for attempt in range(1, self.attempts + 1):
            request = urllib.request.Request(self.base + path, data=data, method=method, headers=self.headers)
            try:
                with self.urlopen(request, timeout=30) as response:
                    return json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as error:
                if attempt == self.attempts or not (error.code == 429 or error.code >= 500):
                    raise
                error.close()
                print(f"GitHub answered {error.code} to {method} {path}; retrying")
            except (urllib.error.URLError, TimeoutError) as error:
                if attempt == self.attempts:
                    raise
                print(f"{method} {path} failed ({error}); retrying")
            self.sleep(min(5 * 2 ** (attempt - 1), 60))
        raise AssertionError("unreachable")

    def tag_commit(self, ref: str) -> str | None:
        """The commit a tag points at, or None if the tag does not exist."""
        try:
            target = self.request("GET", f"/git/ref/tags/{ref}")["object"]
        except urllib.error.HTTPError as error:
            if error.code == 404:
                error.close()
                return None
            raise
        while target["type"] == "tag":
            target = self.request("GET", f"/git/tags/{target['sha']}")["object"]
        return target["sha"]

    def create_tag(self, tag: str, sha: str, message: str) -> None:
        """Create an annotated tag, like the existing release tags."""
        tag_object = self.request("POST", "/git/tags", {"tag": tag, "message": message, "object": sha, "type": "commit"})
        try:
            self.request("POST", "/git/refs", {"ref": f"refs/tags/{tag}", "sha": tag_object["sha"]})
        except urllib.error.HTTPError as error:
            # An attempt that got a 5xx back may still have created the ref,
            # and then the retry is refused because the ref exists.
            if error.code != 422 or self.tag_commit(tag) != sha:
                raise
            error.close()


def tag_releases(releases, sha, results, github, lookup=registry_versions, sleep=time.sleep) -> list[str]:
    """Tag each planned release that reached its registry; return the tags made.

    A registry can take a little while to list a new version, so a package
    whose publish job succeeded is polled for a few minutes.
    """
    created = []
    for release in releases:
        name, kind, version = release["name"], release["kind"], release["version"]
        tag = f"{name}-v{version}"
        attempts = 10 if results.get(kind) == "success" else 1
        for attempt in range(attempts):
            if version in (lookup(kind, name) or set()):
                break
            if attempt + 1 < attempts:
                sleep(15)
        else:
            print(f"{name} {version} is not on {REGISTRY[kind]}; not tagging it")
            continue
        existing = github.tag_commit(tag)
        if existing == sha:
            print(f"{tag} already marks this commit")
            continue
        if existing is not None:
            raise ReleaseError(f"tag {tag} already exists on {existing}, not on the released commit {sha}")
        github.create_tag(tag, sha, f"{name} {version}")
        print(f"Tagged {tag}")
        created.append(tag)
    return created


def command_tag(args) -> int:
    github = GitHub(
        os.environ["GITHUB_REPOSITORY"],
        os.environ["GITHUB_TOKEN"],
        os.environ.get("GITHUB_API_URL", "https://api.github.com"),
    )
    results = {"crate": os.environ.get("CRATES_RESULT", ""), "python": os.environ.get("PYPI_RESULT", "")}
    created = tag_releases(json.loads(os.environ["RELEASES"]), os.environ["GITHUB_SHA"], results, github)
    write_github_file(
        "GITHUB_STEP_SUMMARY",
        "## Release tags\n\n" + ("".join(f"- {tag}\n" for tag in created) or "No new tags.\n"),
    )
    return 0


# Validation -----------------------------------------------------------------


def command_validate(args) -> int:
    packages = load_packages(load_config(), cargo_metadata(args.metadata))
    crates = sum(package.kind == "crate" for package in packages.values())
    print(f"{CONFIG_PATH.name} declares {crates} crates and {len(packages) - crates} Python packages")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)

    plan = commands.add_parser("plan", help="decide what to release")
    plan.add_argument("--packages", default="changed", help='"changed", or package names separated by commas')
    plan.add_argument("--publish", choices=("true", "false"), default="false")
    plan.add_argument("--ref", default=os.environ.get("GITHUB_REF", ""), help="git ref the workflow runs on")
    plan.add_argument("--metadata", type=Path, help="`cargo metadata` JSON to use instead of running cargo")
    plan.set_defaults(handler=command_plan)

    checks = commands.add_parser("checks", help="write a bash script running a package's release checks")
    checks.add_argument("--package", required=True)
    checks.add_argument("--output", required=True, type=Path)
    checks.set_defaults(handler=command_checks)

    publish = commands.add_parser("publish-crates", help="publish $CRATES in order (used by the workflow)")
    publish.set_defaults(handler=command_publish_crates)

    tag = commands.add_parser("tag", help="tag the planned packages now on their registry (used by the workflow)")
    tag.set_defaults(handler=command_tag)

    validate = commands.add_parser("validate", help="check the release config against the workspace")
    validate.add_argument("--metadata", type=Path, help="`cargo metadata` JSON to use instead of running cargo")
    validate.set_defaults(handler=command_validate)

    args = parser.parse_args(argv)
    # The summary is UTF-8; do not let a Windows console code page reject it.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    try:
        return args.handler(args)
    except ReleaseError as error:
        print(f"::error title=Release::{error}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
