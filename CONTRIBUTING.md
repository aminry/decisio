<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Copyright contributors to the decisio project -->

# Contributing to Decisio

Thank you for considering a contribution.
This document says how changes get in, what they must pass, and what we ask of you.

## Before you start

- Open an issue first for anything beyond a small fix, so the design is agreed before the code is written.
- Check `docs/` and `EVAL_CARD.md`: many decisions here were measured, and a change that alters a measured answer needs a measurement.
- The project is maintained by one person plus automated agents (below); be patient with review, and be specific in what you ask.

## Developer Certificate of Origin

Every commit must carry a `Signed-off-by` line with your real name and an email you control:

```
git commit -s
```

This is the [Developer Certificate of Origin](https://developercertificate.org/) 1.1: by signing off you certify that you wrote the change or have the right to submit it under the project's Apache-2.0 licence.
No contributor licence agreement is required.
The DCO check runs on every pull request and blocks merge until every commit is signed.

## Setting up

```
uv sync --extra dev --frozen
uvx pre-commit install --hook-type pre-commit --hook-type commit-msg
uv run pytest
```

The unit tests, the lint and the patch checks run on any machine without a GPU.
`pytest` runs the fast tier; `pytest -m slow` runs the tests that load small models on the CPU, which take about three minutes and peak at about 20 GB of RAM.
Pull request CI runs the fast tier; the slow tier runs nightly and on demand.
`uv.lock` is authoritative; do not upgrade a dependency in a pull request that does anything else.

GPU tests (`tests/gpu`, `pytest -m gpu`) need vLLM 0.30.0, a CUDA card and a local checkpoint.
They do not run on pull requests.
A maintainer runs them on a trusted runner after review, and nightly on `main`.
If your change touches the served path, say so in the pull request so the maintainer runs them before merge.

## What a pull request must have

- One change per pull request, described in the title as a [Conventional Commit](https://www.conventionalcommits.org/) (`feat:`, `fix:`, `docs:`, `test:`, `chore:`); the changelog is generated from these, never edited by hand.
- Tests for new behaviour and for any bug fixed.
- No change to a measured answer unless the pull request includes the measurement: a run under `runs/` with its manifest, or a statement of the check that shows bit-identity (the suite, the conformance gates).
- No benchmark item text, no API keys, no addresses of machines, no personal data in any file, including test data and run records.
- Every new source file carries the SPDX header used throughout the tree.
- The DCO sign-off on every commit.

Merges are squash merges by a maintainer; pull requests are not rebased or force-pushed by the reviewer.

## Style

`ruff` decides formatting and lint; the pre-commit hook runs it.
Prose in docs follows the tree's conventions: one sentence per line in long Markdown files, plain wording, numbers with their record.

## Benchmark claims

A claim that Decisio scores X on some benchmark belongs in an issue using the "Benchmark claim" template, with the commit, the served configuration, the hardware, the harness version and a per-item results file.
Claims without a record are not merged into `EVAL_CARD.md` or `README.md`.

## AI-assisted contributions

Parts of this project were written with AI coding agents under human direction, and that continues.
Rules that apply to everyone, human or agent:

- Disclose assistance in the pull request description.
- Every pull request is reviewed and merged by a human maintainer; no change lands on `main` without that review.
- Do not open pull requests you have not read and cannot explain.
- Agents operated by the project commit under their own identity so their changes are reviewed like anyone else's.

## Reporting problems

Bugs and feature requests: GitHub issues, with the templates.
Security issues: never in a public issue; see `SECURITY.md`.
Conduct: see `CODE_OF_CONDUCT.md`.
