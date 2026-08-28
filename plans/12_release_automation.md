# Release automation — one manual tag, two registries

Goal: keep cutting releases **by hand in the GitHub UI** (create the release, write the notes, let it
create the tag), and have everything after that happen on its own — the version landing in
`Cargo.toml`, the crate going to crates.io, the wheels going to PyPI. No `semantic-release`, no
conventional-commit parsing, no bot deciding the version number -- a `.releaserc.json` +
`semantic-release-replace-plugin` setup is deliberately *not* the model here.

## Where this starts

- **Today's practice** (from the repository, not from memory): four tags exist — `v0.1.0`, `v0.1.1`,
  `v0.1.3`, `v0.2.0` — and each one points at a commit whose `Cargo.toml` **already carries that
  version** (`git show v0.2.0:Cargo.toml`). So the bump has been part of the release PR, and
  `cargo publish` has been run by hand.
- **crates.io** has `xdmf` with four versions, `0.2.0` published 2026-08-27. **PyPI** has nothing
  yet; the name is free.
- **What already exists in CI**: `.github/workflows/release.yml` (M6 Part 2) fires on a `v*` tag,
  refuses a tag that disagrees with the crate version, builds and tests five wheels plus an sdist,
  and publishes them to PyPI through trusted publishing. crates.io is not automated at all.

So this plan adds two things: the version bump, and the crates.io half.

## The one thing that makes this awkward

The version is *in the tree*, in three places:

1. `[workspace.package] version` in `Cargo.toml` — inherited by both crates via `version.workspace = true`,
   and by `python/pyproject.toml` via `dynamic = ["version"]`, which needs no edit ever.
2. `Cargo.lock`, which records both workspace members' versions.
3. Every XDMF file the crate writes: `<Information Name="version" Value="0.2.0"/>`, from
   `env!("CARGO_PKG_VERSION")` (`src/time_series_writer.rs:679`) — and therefore **14 hardcoded
   expected-XML strings** (13 in `tests/time_series_writer.rs`, 1 in `src/time_series_writer.rs`).

A tag cannot change the tree it points at. "The tag updates `Cargo.toml`" therefore means a bot
commits *after* the tag exists, and the tag either keeps pointing at a commit that says the previous
version or has to be force-moved. That is the whole design question below.

## Step 0 (done): stop hardcoding the version in the tests

Those 14 strings are 14 of the 15 places a release touches. Derive them instead:

- Keep the literal, with a placeholder in the one line: `<Information Name="version" Value="$VERSION"/>`.
- One helper, `fn with_version(expected: &str) -> String`, doing
  `expected.replace("$VERSION", env!("CARGO_PKG_VERSION"))`. A `replace` rather than `format!` because
  the expected XML would otherwise need every brace escaped.
- `env!` in an integration test expands to the *test* crate's version, which is the workspace
  version, i.e. the same string the writer stamps.

Two payoffs: a bump becomes a one-line change (plus the lockfile), and the tests start asserting what
they actually mean — "the writer stamps its own version" rather than "the writer stamps 0.2.0".

Mechanical, no behaviour change. Verified by bumping the version locally and confirming the suite
still passes with no test edited.

## Design A — the tag drives everything (the literal reading of the wish)

On a `v*` tag: check out the tag, rewrite `Cargo.toml` from `${GITHUB_REF_NAME#v}`, publish crates.io
and PyPI from that (dirty) tree, then commit the bump to `main` and force-move the tag onto the bump
commit so the tag matches what was published.

It works, and I would not pick it:

- **The tag's tree is not what was published**, unless you force-move the tag — and force-moving a
  tag a GitHub Release already points at is exactly the kind of thing that later makes
  `cargo install --git … --tag v0.3.0` build the *previous* version.
- **`cargo publish` needs `--allow-dirty`**, so the one command whose job is to package a known tree
  gets told to ignore that it does not know the tree.
- **The bot has to push to `main`** — a protected branch says no, and a second release racing a merge
  gets interesting.
- **A half-failed release leaves a moved tag and a bump commit to unwind**, on top of a registry entry
  you cannot delete.

## Design B — a "prepare release" button, then your manual release (recommended)

Same manual release step, one extra click before it, and none of the four problems above.

1. **Actions → "Prepare release" → Run workflow → `0.3.0`** (a `workflow_dispatch` input).
   The job runs `cargo set-version --workspace 0.3.0` (cargo-edit — it understands the
   workspace-inherited version and refreshes `Cargo.lock`; a `sed` on the single `^version = "…"` line
   plus `cargo update --workspace` is the fallback if that tool disappoints), then runs clippy and the
   full suite, then commits `chore(release): 0.3.0`.
   - Push straight to `main` with the default `GITHUB_TOKEN` (`contents: write`) if `main` takes
     direct pushes; otherwise open a PR (`peter-evans/create-pull-request`) and merge it. A
     `GITHUB_TOKEN` push triggers no other workflow, which is what we want here.
2. **You create the release in the GitHub UI** on that commit: tag `v0.3.0`, notes generated and
   edited. This is the step that stays manual, and it is the only place a human decides anything.
3. **The tag fires `release.yml`**, which verifies, builds, tests, and only then publishes to both
   registries.

Why this is the better shape: the tag points at exactly the tree that gets published, no tag is ever
moved, `cargo publish` packages a tree it knows, and a failed release is retried by cutting the next
patch version rather than by unwinding a bot commit. It is also what you already do by hand — the
button only replaces the manual `Cargo.toml` edit, and step 0 shrinks that edit to one line anyway.

## The publish workflow

`release.yml` (renamed from `wheels.yml`, since it stops being about wheels only) grows two jobs:

| Job | What it does | Why it is a separate job |
|-----|--------------|--------------------------|
| `version-check` | tag vs. `cargo metadata` version (exists today) | catches "tagged without bumping" in 20 s, before anything builds |
| `crate-dry-run` | `cargo publish -p xdmf --dry-run` on a runner with `libhdf5-dev` (done) | catches a wrong `include` list, a missing readme — the mistakes that only surface when packaging |
| `wheels` | five native wheels, each installed and pytested (exists) | the platform matrix |
| `sdist` | build the sdist, then build and import a wheel *from* it (exists, as the reusable `sdist.yml` that `rust.yml` calls too) | the include lists are load-bearing for the sdist |
| `publish-pypi` | `pypa/gh-action-pypi-publish`, environment `pypi`, `id-token: write` (exists) | irreversible, so it needs everything above green |
| `publish-crates-io` | `rust-lang/crates-io-auth-action@v1` → `cargo publish -p xdmf`, environment `crates-io`, `id-token: write` (done) | same, and the OIDC token is scoped to this job |

Notes that matter:

- **Both publishes depend on every build**, and only on them — they can run in parallel, since neither
  registry knows about the other.
- **crates.io trusted publishing** (available since mid-2025) removes the long-lived token:
  `rust-lang/crates-io-auth-action` exchanges the GitHub OIDC token for a short-lived
  `CARGO_REGISTRY_TOKEN`. The crate already exists, so this is just configuration.
- **`cargo publish` runs a verification build** with default features, i.e. it needs HDF5 — install
  `libhdf5-dev` in that job, as the other CI jobs do. Not `--no-verify`.
- **`skip-existing: true` on the PyPI upload**: six files go up one by one, so a network failure
  halfway leaves some of them published; a re-run should land the rest instead of failing on the ones
  already there. The guard against publishing the *wrong* version is `version-check`, not the upload.
- **crates.io refuses a version that already exists**, which is the guard against a double release.
  There is no unpublish on either registry; a bad release is followed by the next patch version.

## One-time setup, in three web UIs (nothing lands in the repository)

- **crates.io** → `xdmf` → Settings → Trusted Publishing → GitHub: owner `philbucher`, repo `xdmf`,
  workflow `release.yml`, environment `crates-io`.
- **PyPI** → Account → Publishing → add a **pending** publisher (the project does not exist yet) for
  `xdmf`: repo `philbucher/xdmf`, workflow `release.yml`, environment `pypi`.
- **GitHub** → Settings → Environments → create `pypi` and `crates-io`. Adding yourself as a required
  reviewer on both turns each publish into a deliberate click — worth it for the first few releases,
  since it is the only remaining place to stop a release that got this far.

No secrets, no tokens, nothing to rotate.

## Failure modes, and what catches each

| Failure | Caught by |
|---------|-----------|
| tag says 0.3.0, `Cargo.toml` says 0.2.0 | `version-check`, before any build |
| version already on crates.io | `cargo publish` refuses; the dry-run job builds the same package first |
| the crate would ship without a file it needs (`include` list) | `crate-dry-run` and the `sdist` job's build-from-sdist |
| one platform's static HDF5 build breaks | `fail-fast: false` — the others still build, and no publish runs until you decide |
| the version written into users' files disagrees with the release | step 0's tests, once they derive the version |
| PyPI upload dies halfway | `skip-existing: true` on the re-run |

## Deliberately not automated

- **The version number** — semver is a judgment call about the public API, and this crate is
  pre-1.0 and still breaking things on purpose.
- **The release notes** — GitHub's generator plus your edit.
- **A `CHANGELOG.md`** — the GitHub releases are the changelog. Add a file only if a crates.io
  consumer asks for one.
- **`python/pyproject.toml`** — it takes the version from `Cargo.toml` dynamically and must never be
  edited.

## Work items, in order

1. ~~Step 0: derive the version in the 13 expected-XML sites.~~ Done.
2. ~~`release.yml`: add `crate-dry-run` and `publish-crates-io`.~~ Done — `publish` became
   `publish-pypi`, and both publishes wait on `crate-dry-run` too, so a release is all-or-nothing
   across the two registries. `Cargo.lock` is now committed and every CI cargo command passes
   `--locked` -- a library's lockfile is ignored by its consumers, so this costs them nothing, while
   the wheels (which *are* binaries) become reproducible from the tag, and `cargo package` was
   shipping a lockfile into the crate and the sdist either way. `latest-deps.yml` runs
   `cargo update` weekly to keep the canary a lockfile otherwise removes.
3. `prepare-release.yml`: the `workflow_dispatch` bump.
4. The one-time registry/environment setup (yours to click).
5. First run, `0.2.1` or `0.3.0`: it publishes a new crates.io version *and* the first PyPI release,
   so expect to iron out the trusted-publisher configuration on that run rather than the plumbing.
