# Releasing

One workflow releases every package in this repository: the workspace crates
on crates.io and the Python distributions on PyPI. It is
[`.github/workflows/publish.yml`](.github/workflows/publish.yml), shown in the
Actions tab as **Publish SDK packages**. The packages it can release, and the
checks each must pass, are listed in
[`.github/release-packages.toml`](.github/release-packages.toml).

## Release a change

1. **Bump the version** of each package you want to release, as a stable
   `X.Y.Z`:
   - a crate: `version` in its `Cargo.toml`, or `[workspace.package]` in the
     root `Cargo.toml` for crates that inherit it
   - `kapsl-sdk`: `pyproject.toml` and `crates/kapsl-pyo3/pyproject.toml`,
     which must match
2. **Merge to `main`.**
3. **Run the workflow** from Actions → Publish SDK packages → Run workflow,
   on branch `main`:
   - `packages`: keep `changed` to release everything that changed, or name
     packages, for example `kapsl-hal, kapsl-llm`
   - `publish`: tick it to publish; leave it unticked for a dry run that only
     plans, builds and checks
4. **Read the plan.** The *Plan the release* job's summary lists every
   package, its local and published versions, and whether it will be
   released. The selected packages are then built and checked.
5. **Approve.** When every build and check has passed, the run waits on the
   `sdk-release` environment. A required reviewer checks the plan and
   approves it with *Review deployments*. One approval publishes the whole
   plan. If any build or check fails, nothing is published.
6. **Tags.** Each published version is tagged `<name>-v<version>`, for
   example `kapsl-hal-v0.3.4`. The next run compares against these tags.

## How `changed` picks packages

A package is released when its version is not yet on its registry and the
files it ships changed since its last release:

| Version already published? | Changed since last release? | Result |
| --- | --- | --- |
| no | yes, or no release tag to compare with | released |
| no | no | skipped; name it explicitly to release it anyway |
| yes | yes | skipped, with a warning to bump its version |
| yes | no | up to date |

- *Last release* is the newest `<name>-v<version>` tag whose version is on
  the registry.
- *Files it ships* are a crate's directory. For a Python package they are its
  `watch` paths; for `kapsl-sdk` they also include every workspace crate
  compiled into the wheel.
- Crates published before this workflow existed may have no release tag.
  Until their first release through the workflow, they are judged on version
  alone.
- A workspace-wide version bump does not, on its own, release the crates that
  did not change.

Naming packages releases exactly those, whether or not they changed; each
must have a version that is not yet published.

Crates are published in dependency order. Before approval they are verified
together with one multi-package `cargo publish --dry-run`. That means a crate
can depend on a new version of another crate released in the same run.

## Who can publish

- Anyone with write access can start a run, including a dry run.
- Nothing is published until a required reviewer of the `sdk-release`
  environment approves. Those reviewers are the people with release
  authority.
- Publishing is refused unless the run is on `main`. The environment's branch
  rule enforces the same thing, so a workflow edited on another branch cannot
  reach the environment or its secrets.
- crates.io and PyPI trust only this workflow and this environment (trusted
  publishing through GitHub OIDC). There is no long-lived registry token,
  except the one crates.io needs for a crate's very first release.

## One-time setup

These steps need repository admin rights and owner rights on crates.io and
PyPI. Releases fail until they are done.

1. **GitHub environment.** Settings → Environments → New environment
   `sdk-release`:
   - *Required reviewers*: the release maintainers. One approval is enough.
   - *Prevent self-review*: on, if there are at least two reviewers, so nobody
     approves their own run.
   - *Deployment branches and tags*: selected branches only, `main`.
   - Secrets:
     - `CRATES_IO_TOKEN`: a crates.io API token with the `publish-new` scope,
       used only for a crate's first release. `kapsl-managed-abi` needs it for
       its first release. You can delete it between first releases.
     - `R2_ACCESS_KEY_ID` and `R2_SECRET_ACCESS_KEY`: moved from repository
       secrets. They upload `kapsl-sdk` releases to R2.
2. **crates.io trusted publishing.** For every crate in
   `release-packages.toml`, open the crate's Settings → Trusted Publishing and
   add a GitHub publisher: owner `kapsl-runtime`, repository `kapsl-sdk`,
   workflow `publish.yml`, environment `sdk-release`. Remove the entries for
   the old workflows. Do this for `kapsl-managed-abi` after its first release.
3. **PyPI trusted publishing.** For `kapsl-sdk`, open Manage → Publishing and
   add a GitHub publisher with the same four values. `kapsl-vllm-connector`
   is released from
   [`kapsl-integrations`](https://github.com/kapsl-runtime/kapsl-integrations),
   so do not add a `kapsl-sdk` publisher for it.
4. **Tag rules.** If a ruleset restricts tag creation, allow GitHub Actions to
   create tags matching `*-v*`.
5. **Clean up.** Once releases work, delete the old environments:
   `crates-io-backend-abi`, `crates-io-llm`, `pypi` and `pypi-vllm-connector`.

## Add a package

- **A crate.** Add `[crate.<name>]` to `release-packages.toml`, with any extra
  `checks`. CI fails while a publishable workspace crate is missing from the
  file; mark a crate that must never be published with `publish = false` in
  its `Cargo.toml`. A crate that crates.io has never seen is published with
  `CRATES_IO_TOKEN`; add its trusted publisher afterwards.
- **A Python package.** Add `[python.<name>]` with its `path` and `build`
  (`maturin` or `setuptools`), then add a pending publisher on PyPI.

The top of `release-packages.toml` documents every setting.

## When a run fails

- **Before approval**, nothing has been published. Fix the problem, merge it
  and run the workflow again.
- **After approval**, re-run the failed jobs. Crates are published one at a
  time; versions that already reached crates.io are skipped, and PyPI uploads
  skip files that already exist. The tag job tags whatever reached its
  registry, even after a partial failure.

## What changed from the previous process

This workflow replaces the per-package workflows `publish-backend-abi.yml`,
`publish-host-contracts.yml`, `publish-llm.yml`, `publish-vllm-connector.yml`
and `publish-python-package.yml`.

- Pushing a tag no longer publishes anything. Tags are now created by the
  workflow after publishing.
- The `kapsl-sdk` version comes from `pyproject.toml`, not from a `v*` tag.
- `kapsl-sdk` distributions are uploaded to R2 only when they are released.
  Unapproved "dev" uploads no longer happen.
