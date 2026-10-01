<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Copyright contributors to the decisio project -->

# Launch checklist

Everything to apply the moment the repository goes public, with the exact settings page or `gh api` call for each.
Nothing here is applied while the repository is private, and nothing is applied until the owner says the repository is going public.
This file holds no secrets; it stays public as the record of how the settings were made.

## How to read this

- **API**: can be applied with `gh` in the owner's session.
  The session's token has the scopes `repo` and `workflow`, which is all these calls need, and the owner's admin role on the repository.
- **CLICK**: needs the owner in a browser (installing an app, creating an app or token, registering a PyPI publisher, approving a deployment).
- Commands assume `gh` is logged in as the owner and use `aminry/decisio`.
- Every call has a read-back in section 4; run it after the call and compare with the expected value.

## State on 2026-09-30 (read-only probes)

The repository is private on a personal account, on the free plan, so most of the settings below cannot even be read yet:

| Probe | Answer today |
| --- | --- |
| `GET /repos/aminry/decisio/rulesets` | 403, "Upgrade to GitHub Pro or make this repository public" |
| `GET .../code-scanning/default-setup` | 403, code scanning not enabled |
| `GET .../private-vulnerability-reporting` | 404 |
| `GET .../actions/permissions/fork-pr-contributor-approval` | 422, "not allowed for private repositories" |
| `GET .../actions/permissions` | all actions allowed, `sha_pinning_required: false` |
| `GET .../actions/permissions/workflow` | default token `read`, Actions may not approve pull requests (already the target) |
| `pypi.org/pypi/decisio/json`, `test.pypi.org/pypi/decisio/json` | 404 on both: the name is free |

It follows that nothing in sections 2 to 3 could be rehearsed on this repository; the commands were checked against GitHub's API documentation and the error bodies above, not executed.
Section 4 is the rehearsal: it is run straight after each step.

## Summary

| # | Item | How | Notes |
| --- | --- | --- | --- |
| 1 | Flip the visibility | API | the owner's decision, the last thing before section 2 |
| 2.0 | The maintainer's commit signing (SSH key, git config, GitHub signing key) | local, then CLICK or API | before 2.4 |
| 2.1 | Squash-only merges, branch cleanup, web sign-off | API | applied 2026-10-01, while private |
| 2.2 | Actions: allowed actions, SHA pinning, fork approval, read-only token | API | |
| 2.3 | DCO GitHub App | CLICK | before the ruleset, or the required check never reports |
| 2.4 | Rulesets on `main` and on release tags | API | needs 2.3 |
| 2.5 | Secret scanning with push protection, private vulnerability reporting, CodeQL default setup, Dependabot alerts | API | |
| 2.6 | The release token, from a GitHub App | CLICK to create, API to store | the workflow edit is a pull request |
| 2.7 | `DECISIO_MODEL`, `DECISIO_VIEW`, `GPU_RUNNER_READY` variables | API | when a GPU runner exists |
| 2.8 | The `benchmark` label | API | |
| 3.0 | Gate: the image's first GPU start | by hand, on a GPU virtual machine | before any release tag |
| 3.1 | PyPI trusted publisher | CLICK | no API exists |
| 3.2 | The `pypi` environment with a required reviewer | API | |
| 3.3 | The publish workflow | pull request | added after 3.2 |
| 3.4 | First release | CLICK (approval) | |
| 3.5 | The container image on ghcr.io | CLICK (package visibility) | built by `docker.yml` on the release tag |

## Decisions taken (2026-09-30)

- **Bypass on `main`** (2.4): repository admins can merge a pull request past a failing required check, never push directly; every bypass is logged.
  The price of no bypass would be that a broken required check (for instance the DCO app being down) blocks every merge until the ruleset is edited.
- **The release token** (2.6): a GitHub App.
  The DCO app skips bots but not people, so release pull requests opened with a personal token would carry no `Signed-off-by` and fail the required DCO check.
- **The first version number** (3.4): `0.1.0`.
- **Required code owner review stays off**: with one code owner who opens the pull requests, turning it on would make every pull request unmergeable.

## History of `main` (rebuilt once, 2026-09-30, while private)

`main` was rebuilt as four commits before the flip: the extraction commit (the tree PR #1 merged, squashed from its 24 commits), then one commit each for #2, #4 and #5, composed the way a squash merge composes its message (the pull request title and every commit's message, sign-offs included).
The final tree is identical to the tree of the branch that held all three pull requests; no file differs.
It was done with one force-push, with a lease on the old tip; no rewrite is planned after it, and the rulesets in 2.4 forbid one.

| | Commit |
| --- | --- |
| Old `main` | `195b908d976765fcdad407b77821b07075c91dcd` (36 commits; #2 as a merge commit, without #4 and #5) |
| Old branch with all three pull requests | `826cadae8789161c68290cfb329d82db99484519` (`community-files`, deleted) |
| Old extraction branch | `75e637d72a00c7a12e4b627e52bde2e5969f3c97` (`extract-v0.1`, deleted) |
| New `main` before this record | `9185ada87b77f8aa4f39d2162c78b80a53172727` |

The commits that were replaced stay reachable by hash and through the pull request refs (`refs/pull/N/head`), which GitHub does not let an owner delete.
The rebuild changes what `main` shows, not what the repository serves.

## Flip record (2026-10-01)

The repository became public at **2026-10-01 01:26:26 UTC**, after a history scan with no leaks (below).
What was applied, and what each setting read back as afterwards:

| Step | Read back |
| --- | --- |
| Before the flip: gitleaks | `.gitleaksignore` with 27 fingerprints for `gitleaks git` (28 findings: sha256 digests in run records, none a credential) and 12 for `gitleaks dir`; on a fresh clone with all pull request refs: full history 72 commits, old pull request tips 52 commits, working tree: no leaks |
| 2.1 merge settings | squash only, `PR_TITLE` and `COMMIT_MESSAGES`, delete branch on merge, update branch, web sign-off required (applied 2026-10-01 before the flip, read back then) |
| 2.2 Actions | `allowed_actions` selected, 9 patterns, `github_owned_allowed` and `verified_allowed` true, `sha_pinning_required` true; fork approval `all_external_contributors`; default token `read`, `can_approve_pull_request_reviews` false |
| 2.3 DCO app | installed by the maintainer; reports a check named `DCO` from the app `dco` |
| 2.4 rulesets | `main` (id 24282347) and `release tags` (id 24282348), both active; effective rules on `main`: deletion, non_fast_forward, pull_request, required_linear_history, required_signatures, required_status_checks; bypass: `RepositoryRole` 5, `pull_request` mode; required checks `lint, unit tests, build` (Actions app) and `DCO`, strict; squash only |
| 2.5 security | secret scanning and push protection enabled; private vulnerability reporting enabled; CodeQL default setup configured for `actions` and `python`; Dependabot alerts and security updates enabled |
| 2.6 release token | GitHub App `decisio-release` (ID 5143933) installed on this repository only; secret `RELEASE_PLEASE_APP_PRIVATE_KEY` and variable `RELEASE_PLEASE_APP_ID` set; `release-please` ran with the app token and opened the release pull request |
| 2.7 variables | **not applied**: no GPU runner exists, so `GPU_RUNNER_READY`, `DECISIO_MODEL` and `DECISIO_VIEW` stay unset (the only variable is the app ID) |
| 2.8 label | `benchmark` created |

Behaviour, from a scratch clone: a direct push to `main` is rejected ("Changes must be made through a pull request", "2 of 2 required status checks are expected"), a force-push is rejected, deleting `main` is refused, and deleting a `v*` tag is rejected ("Cannot delete this tag").
The first scans: CodeQL ran on `main` for both languages with 0 results and 0 open code scanning alerts; Scorecard's first published score is 6.5; there are 0 secret scanning alerts and 1 Dependabot alert (`setuptools` 80.10.2 in `uv.lock`, fixed in 83.0.0, medium), dismissed as not used (see below).
The signed-commit rule was tested with two pull requests: one whose branch commit is signed with the maintainer's SSH key (accepted, squash commit verified by GitHub), and one whose branch commit is unsigned (this one); the result of the second is in the maintainer's notes, not here, because it can only be known after this merges.

Things that went differently from the sections below, and are corrected there:
- The DCO app reports on a pull request's `opened` and `synchronize` events, not on `reopened`: after installing it, push a commit to an open pull request to get the check.
- A `v*` tag can be created under the tag ruleset (only deletion and updates are forbidden), so a probe tag cannot be removed afterwards without switching the ruleset off; do not use one as a test.
- `RELEASE_PLEASE_APP_ID` must be the app's numeric ID, not its name.
- GitHub's merge-state flag can say `BLOCKED` for a pull request whose every required check is green (seen on one pull request, whose branch commit was unsigned; the merge call was accepted and every rule evaluated `pass` in the rule-suite record).
  Treat it as GitHub's merge-state lag: do not chase it, and read the rule-suite record (`gh api repos/aminry/decisio/rulesets/rule-suites`) if the result matters.
- Dependabot's `uv` jobs failed on every run, scheduled and security, on one dependency, `setuptools` (error "No files have changed!" from `Dependabot::Uv::FileUpdater`, reproduced locally with the Dependabot CLI and the same updater image).
  The cause is not this repository's configuration: vLLM 0.30.0 declares `setuptools<81.0.0,>=77.0.3`, vLLM is pinned exactly, so the Linux branch of `uv.lock` cannot take the fixed 83.0.0; `uv lock --upgrade-package setuptools` resolves 84.0.0 only for the non-Linux branch and keeps 80.10.2 for Linux, after which Dependabot's updater finds no file to change.
  The alert (the advisory concerns building sdists with `MANIFEST.in` on macOS file systems, which the locked copy is not used for) is dismissed as "not used" with that note, and `setuptools` is ignored in `.github/dependabot.yml` so the weekly job stops failing; remove both when vLLM lifts its cap.

## 0. Before the repository goes public

These are gates, not settings.

- [ ] Pull request #2 (community files and CI) is merged, and so is the pull request that carries this file.
- [ ] Keys that were shared outside a secret store are rotated, the contamination screen is shipped, and the benchmark document is final.
- [ ] The history is scanned for secrets once more, because after the flip it is public and cannot be recalled: `gitleaks git --redact -v .` in the clone (`brew install gitleaks`; expect no findings).
  A plain pattern search of all 31 commits on 2026-09-30 (cloud and model-hub key formats, private key headers) found nothing; that is a cheap first look, not a scanner run.
- [ ] Anyone else who pushes to this repository knows that after section 2.4 everything goes through a pull request with a signed-off commit (section 5).

## 1. Make it public [API]

```
gh repo edit aminry/decisio --visibility public --accept-visibility-change-consequences
```

From this moment `main` is readable by everyone and unprotected; only the owner can push.
Do section 2 immediately, in order, without a break.

## 2. Apply, in this order

### 2.0 The maintainer's commit signing [local, then CLICK or API]

What the `required_signatures` rule in 2.4 needs, as far as GitHub's documentation says: commits that land on `main` must be verified.
Every change reaches `main` as a squash merge made on github.com, and GitHub signs that commit itself, so a pull request whose branch commits are unsigned should still merge; the first pull request after 2.4 is the test (verification below).
Signing locally is still worth doing now: the maintainer's commits on branches show "Verified", and it covers any path that puts the maintainer's own commit on `main`.
Nothing is configured on the maintainer's machine yet (2026-09-30: `gpg.format`, `user.signingkey` and `commit.gpgsign` are unset), and the `gh` session's token cannot read or add signing keys or read the account's emails, so steps 3 and 4 are the maintainer's.

1. [local] Generate a dedicated signing key, or select an existing one.
   The machine has `~/.ssh/id_ed25519.pub`; a separate key keeps signing apart from logging in, and GitHub lists the two kinds separately.
   ```
   ssh-keygen -t ed25519 -C "roudaky@gmail.com" -f ~/.ssh/decisio_signing     # set a passphrase
   ssh-add --apple-use-keychain ~/.ssh/decisio_signing                        # macOS: unlock once per login
   ```
2. [local] Configure git.
   `--global` signs every repository on the machine, including work that automated agents do in the maintainer's checkout under the maintainer's identity; use `--local` inside the clone to limit it to decisio, and decide that on purpose, because a signature says the holder of the key made the commit.
   ```
   git config --global gpg.format ssh
   git config --global user.signingkey ~/.ssh/decisio_signing.pub
   git config --global commit.gpgsign true
   mkdir -p ~/.config/git
   echo "roudaky@gmail.com namespaces=\"git\" $(cat ~/.ssh/decisio_signing.pub)" >> ~/.config/git/allowed_signers
   git config --global gpg.ssh.allowedSignersFile ~/.config/git/allowed_signers
   ```
   The allowed-signers file is only for checking signatures on this machine (`git log --show-signature`); GitHub does not read it.
3. [CLICK] Confirm that `roudaky@gmail.com` is a verified email on the GitHub account (<https://github.com/settings/emails>); a signature by a key whose email is not verified on the account shows as unverified.
4. [CLICK or API] Add the public key to GitHub as a signing key.
   Settings page: <https://github.com/settings/ssh/new>, Key type "Signing Key", paste `~/.ssh/decisio_signing.pub`.
   Or from `gh`, after granting it the scope in a browser once:
   ```
   gh auth refresh -h github.com -s admin:ssh_signing_key
   gh ssh-key add ~/.ssh/decisio_signing.pub --type signing --title "decisio signing"
   ```
5. Verify, before the flip.
   Locally: `git commit --allow-empty -m "test: signing probe"` in a scratch clone, then `git log -1 --show-signature`; expect `Good "git" signature for roudaky@gmail.com with ED25519 key SHA256:...`.
   On GitHub: push that commit to a throwaway branch (not `main`) and read the verdict, then delete the branch:
   ```
   git push origin HEAD:refs/heads/signing-probe
   gh api repos/aminry/decisio/commits/signing-probe --jq .commit.verification
   #   verified true, reason "valid"
   git push origin --delete signing-probe
   ```
   After 2.4, the first pull request is the real test of the rule: a docs-only pull request with one signed commit, squash-merged; then one with an unsigned branch commit, to confirm that the squash merge is accepted.
   If an unsigned branch commit blocks the merge, keep `commit.gpgsign` on for everyone who pushes to this repository and say so in `CONTRIBUTING.md`.

### 2.1 Merge settings [API]

Squash merges only, with the pull request title as the commit title (the changelog is generated from Conventional Commit titles) and the individual commit messages in the body (this keeps each author's `Signed-off-by` line in the history).
A squash merge made on github.com is signed by GitHub, which is what the signed-commit rule in 2.4 accepts.

```
gh api -X PATCH repos/aminry/decisio --input - <<'JSON'
{
  "allow_squash_merge": true,
  "allow_merge_commit": false,
  "allow_rebase_merge": false,
  "allow_auto_merge": false,
  "allow_update_branch": true,
  "delete_branch_on_merge": true,
  "squash_merge_commit_title": "PR_TITLE",
  "squash_merge_commit_message": "COMMIT_MESSAGES",
  "web_commit_signoff_required": true,
  "has_wiki": false,
  "has_projects": false
}
JSON
```

Settings page: Settings, General, Pull Requests and Features.

**Applied on 2026-10-01, while the repository was private** (merge settings are available on a private repository on the free plan, and they stop merge commits from landing by default).
Read back with a fresh `GET` afterwards; the seven fields that changed:

| Field | Before | After |
| --- | --- | --- |
| `allow_merge_commit` | true | false |
| `allow_rebase_merge` | true | false |
| `allow_update_branch` | false | true |
| `delete_branch_on_merge` | false | true |
| `squash_merge_commit_title` | `COMMIT_OR_PR_TITLE` | `PR_TITLE` |
| `web_commit_signoff_required` | false | true |
| `has_projects` | true | false |

Already as specified and unchanged: `allow_squash_merge` true, `allow_auto_merge` false, `squash_merge_commit_message` `COMMIT_MESSAGES`, `has_wiki` false, `has_discussions` false.
The pull requests merged before this date (#1, #2, #4, #5, #8, #9) were merged with merge commits; the history of `main` was rebuilt once for the first four and #8 and #9 stay as merged.

### 2.2 Actions policy [API]

Allowed actions are GitHub's own and verified creators, plus an explicit list of the third-party actions the workflows use, so that a creator's verification status cannot break a workflow.
`sha_pinning_required` rejects any workflow that uses an action by tag; every workflow in the repository is already pinned by commit hash.

```
gh api -X PUT repos/aminry/decisio/actions/permissions --input - <<'JSON'
{ "enabled": true, "allowed_actions": "selected", "sha_pinning_required": true }
JSON

gh api -X PUT repos/aminry/decisio/actions/permissions/selected-actions --input - <<'JSON'
{
  "github_owned_allowed": true,
  "verified_allowed": true,
  "patterns_allowed": [
    "astral-sh/setup-uv@*",
    "googleapis/release-please-action@*",
    "ossf/scorecard-action@*",
    "step-security/harden-runner@*",
    "pypa/gh-action-pypi-publish@*",
    "docker/setup-buildx-action@*",
    "docker/build-push-action@*",
    "docker/login-action@*",
    "anchore/sbom-action@*"
  ]
}
JSON
```

Workflows from outside contributors need approval for every outside contributor, not only first-timers (the default), so no fork code runs on a runner without a maintainer's click.
The default token is read-only and Actions cannot approve pull requests; both are already the values today and are set again so that the policy is explicit.

```
gh api -X PUT repos/aminry/decisio/actions/permissions/fork-pr-contributor-approval \
  -f approval_policy=all_external_contributors

gh api -X PUT repos/aminry/decisio/actions/permissions/workflow \
  -f default_workflow_permissions=read -F can_approve_pull_request_reviews=false
```

Settings page: Settings, Actions, General.
If a GitHub-run workflow (code scanning, Dependabot) is rejected after this step with a pinning error, set `sha_pinning_required` back to `false`, say so in the pull request that records it, and keep every workflow of ours pinned.

### 2.3 DCO app [CLICK]

1. Open <https://github.com/apps/dco> and choose Install (or Configure if it is already installed on the account).
2. Repository access: only select repositories, `aminry/decisio`.
3. Optional, recommended for outside contributors: add `.github/dco.yml` on `main` (through a pull request) so that a contributor can fix a missing sign-off with a remediation commit instead of rewriting history:

```
allowRemediationCommits:
  individual: true
```

The check appears on pull requests with the name `DCO`, which is the name the ruleset requires; it is sent on `opened` and `synchronize`, so an open pull request needs a new push to get it.
The app's address is exactly <https://github.com/apps/dco>; a trailing full stop makes GitHub answer 404.
Bots are skipped by the app; people, including the owner, are not.

### 2.4 Rulesets [API]

Requires 2.3: a required check that nothing reports blocks every merge.

The required CI check is named after the job, `lint, unit tests, build`; the `ci / ` prefix that the pull request page shows is the workflow name and is not part of the check's name.
The entry is pinned to the GitHub Actions app (`integration_id` 15368), so no other app can report a check of that name.
`strict_required_status_checks_policy` makes a pull request pass against the current `main`, which `allow_update_branch` makes one click.

```
gh api -X POST repos/aminry/decisio/rulesets --input - <<'JSON'
{
  "name": "main",
  "target": "branch",
  "enforcement": "active",
  "conditions": { "ref_name": { "include": ["~DEFAULT_BRANCH"], "exclude": [] } },
  "bypass_actors": [
    { "actor_id": 5, "actor_type": "RepositoryRole", "bypass_mode": "pull_request" }
  ],
  "rules": [
    { "type": "deletion" },
    { "type": "non_fast_forward" },
    { "type": "required_linear_history" },
    { "type": "required_signatures" },
    {
      "type": "pull_request",
      "parameters": {
        "required_approving_review_count": 0,
        "dismiss_stale_reviews_on_push": false,
        "require_code_owner_review": false,
        "require_last_push_approval": false,
        "required_review_thread_resolution": true,
        "allowed_merge_methods": ["squash"]
      }
    },
    {
      "type": "required_status_checks",
      "parameters": {
        "strict_required_status_checks_policy": true,
        "do_not_enforce_on_create": false,
        "required_status_checks": [
          { "context": "lint, unit tests, build", "integration_id": 15368 },
          { "context": "DCO" }
        ]
      }
    }
  ]
}
JSON
```

What each rule is: `deletion` restricts deletions; `non_fast_forward` blocks force pushes; `required_linear_history` and `allowed_merge_methods: ["squash"]` make merges squash-only; `required_signatures` requires signed commits on `main`; `pull_request` makes every change a pull request (no approvals are required, because a sole maintainer cannot approve their own pull request; every pull request is still read and merged by a person); the last rule requires the two checks.
`actor_id` 5 is the repository admin role in GitHub's ruleset format, but the API's description does not list role ids: read it back (section 4) and confirm the settings page shows "Repository admin".
If it shows another role, delete the entry and add the bypass in the page (Settings, Rules, Rulesets, `main`, Bypass list, Add bypass, Repository admin, "For pull requests only").

Release tags cannot be moved or deleted once created:

```
gh api -X POST repos/aminry/decisio/rulesets --input - <<'JSON'
{
  "name": "release tags",
  "target": "tag",
  "enforcement": "active",
  "conditions": { "ref_name": { "include": ["refs/tags/v*"], "exclude": [] } },
  "bypass_actors": [],
  "rules": [ { "type": "deletion" }, { "type": "update" } ]
}
JSON
```

Settings page: Settings, Rules, Rulesets.
After the first CodeQL run has completed on `main` (2.5), tighten `main` by adding the rule `{"type": "code_scanning", "parameters": {"code_scanning_tools": [{"tool": "CodeQL", "alerts_threshold": "errors", "security_alerts_threshold": "high_or_higher"}]}}` with `gh api -X PUT repos/aminry/decisio/rulesets/<id>` (the body is the whole ruleset).
Renaming the CI job later changes the name of the required check: change the name and the ruleset together, or no pull request can merge.

### 2.5 Security features [API]

Secret scanning and push protection (both default to on for a public repository; this makes it explicit), private vulnerability reporting (the path `SECURITY.md` and the issue template point to), CodeQL default setup for Python and for the workflow files, and Dependabot alerts and security updates (version updates are already in `.github/dependabot.yml`).

```
gh api -X PATCH repos/aminry/decisio --input - <<'JSON'
{ "security_and_analysis": {
    "secret_scanning": { "status": "enabled" },
    "secret_scanning_push_protection": { "status": "enabled" } } }
JSON

gh api -X PUT repos/aminry/decisio/private-vulnerability-reporting

gh api -X PATCH repos/aminry/decisio/code-scanning/default-setup --input - <<'JSON'
{ "state": "configured", "languages": ["python", "actions"], "query_suite": "default" }
JSON

gh api -X PUT repos/aminry/decisio/vulnerability-alerts
gh api -X PUT repos/aminry/decisio/automated-security-fixes
```

Settings page: Settings, Advanced Security (Code security).
The CodeQL call returns a run id; the first scan takes a few minutes.

### 2.6 The release token, from a GitHub App [CLICK, then API]

release-please needs a token that is not the default one, which is why the workflow mints one from the app (the secret name `RELEASE_PLEASE_TOKEN` of the current workflow goes away with this edit): pull requests and tags created with the default token do not start other workflows, so the release pull request would never get its checks and the publish workflow would never run.

Its tokens last an hour, it is not tied to a person, and its commits are a bot's, which the DCO app skips.

1. [CLICK] <https://github.com/settings/apps/new>: name `decisio-release` (any free name), homepage `https://github.com/aminry/decisio`, Webhook: untick Active, repository permissions Contents: read and write, Pull requests: read and write (Metadata: read is added automatically), "Where can this GitHub App be installed": only on this account.
2. [CLICK] On the app's page, Generate a private key (a `.pem` downloads) and note the App ID.
3. [CLICK] Install App, then the account, then only select repositories, `aminry/decisio`.
4. [API] Store both, then delete the local `.pem`.
   The variable is the app's numeric **App ID** (shown on its settings page), not its name; the key goes in through standard input (`<`), not as an argument:

```
gh variable set RELEASE_PLEASE_APP_ID --repo aminry/decisio --body <app id>
gh secret set RELEASE_PLEASE_APP_PRIVATE_KEY --repo aminry/decisio < decisio-release.private-key.pem
```

5. [pull request] In `.github/workflows/release-please.yml` mint the token in a step before release-please (the job's `if:` stays as it is) and add `workflow_dispatch:` under `on:` so the workflow can be started by hand:

```
      - uses: actions/create-github-app-token@bcd2ba49218906704ab6c1aa796996da409d3eb1 # v3.2.0
        id: app-token
        with:
          app-id: ${{ vars.RELEASE_PLEASE_APP_ID }}
          private-key: ${{ secrets.RELEASE_PLEASE_APP_PRIVATE_KEY }}
      - uses: googleapis/release-please-action@45996ed1f6d02564a971a2fa1b5860e934307cf7 # v5.0.0
        with:
          token: ${{ steps.app-token.outputs.token }}
          config-file: release-please-config.json
          manifest-file: .release-please-manifest.json
```

The action pin is checked against its tag (v3.2.0 was the latest release on 2026-09-30).

The first release pull request shows whether the DCO check is skipped and whether `uv.lock` and `src/decisio/__init__.py` carry the new version; see the review notes on pull request #2 for the `uv.lock` extra-file.

### 2.7 Repository variables [API]

Only the nightly GPU job uses these (`.github/workflows/gpu.yml`), and it is skipped until a runner labelled `gpu` exists and `GPU_RUNNER_READY` is `true`.
The values are paths on the runner's disk, not secrets, so they are variables.

```
gh variable set DECISIO_MODEL --repo aminry/decisio --body /path/to/Qwen3.6-35B-A3B-FP8
gh variable set DECISIO_VIEW  --repo aminry/decisio --body /path/to/text-only-view
gh variable set GPU_RUNNER_READY --repo aminry/decisio --body true     # last, after the runner is registered
```

`DECISIO_MODEL` is the official checkpoint directory; `DECISIO_VIEW` is the text-only view built by `python -m decisio.serve.make_text_only` (see `tests/gpu/gpu_tier.py`).
Without either, a GPU test skips, so the job fails on an empty variable instead of passing without running the serving gates.

A runner on a public repository is reachable from workflow code, so it must be ephemeral (one job, then gone), hold no standing credentials, and run only what the maintainer reviewed; the workflow is triggered only by `schedule` and `workflow_dispatch`, and 2.2's "approve all outside contributors" is what keeps a pull request from pointing a workflow at it.
A just-in-time runner configuration comes from the API (not tested here, no runner exists yet):

```
gh api -X POST repos/aminry/decisio/actions/runners/generate-jitconfig \
  -f name=gpu-1 -F runner_group_id=1 -f 'labels[]=gpu' --jq .encoded_jit_config
```

and starts with `./run.sh --jitconfig <value>` on the box.

### 2.8 Labels [API]

The bug and feature issue forms use GitHub's default labels; the benchmark-claim form uses `benchmark`, which does not exist, and GitHub drops unknown labels silently.
Dependabot and release-please create their own labels; check that they did after their first pull requests.

```
gh label create benchmark --repo aminry/decisio --color 5319E7 --description "A benchmark claim with its record"
```

## 3. The first release: the image's first GPU start, PyPI, and the image

### 3.0 First GPU start of the image [gate, by hand]

A gate before the first release tag (not before the flip): the image has to start on the measured kind of card and answer correctly before a tag publishes it.
Today CI builds the image for linux/amd64 and starts it without a GPU, where it must fail with a clear message; nothing has served a request from the container, so nothing the README says about the image is measured.
Until this gate passes, the README's Docker section says so (step 11).

Before renting anything:
- **A virtual machine with one 96 GB card**, Docker Engine with the NVIDIA Container Toolkit, a driver that supports CUDA 13.0 (580 or later), `git`, `uv`, and about 100 GB of free disk (the image is about 25 GB unpacked, the checkpoint 36 GB).
  The record in `EVAL_CARD.md` was measured on an RTX PRO 6000 Blackwell; no other card is covered by this gate.
  Nothing on the machine is exposed: the compose file publishes the port on 127.0.0.1 and everything below runs on the machine itself or over `ssh`.
- **An items file for C3 and C4.** C2 needs none; C3 and C4 read a JSON list in the format documented at the top of `src/decisio/serve/systemone_conformance.py`.
  `python benchmarks/make_conformance_items.py --out conformance_items.json` builds one from two public sets (100 BoolQ validation items and 100 BANKING77 test items, about 80 s); prepare it before the machine is rented.
  The earlier records used a private 1,400-item suite.
- About two hours of machine time; the first start is dominated by the checkpoint download.

Steps (`R` is the run folder, named for the day of the run):

1. Check the machine and fetch the exact commit that will be tagged (a read-only deploy key or `gh auth login` is enough while the repository is private).
   ```
   nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
   git clone https://github.com/aminry/decisio && cd decisio && git checkout <the commit to be tagged>
   R=runs/$(date +%F)_docker-first-gpu-start && mkdir -p "$R"
   ```
2. Build the image on the machine, and check that the toolkit passes the card through.
   No image is on `ghcr.io` before the first tag (the release workflow pushes it), so the gate builds it from the same `Dockerfile`.
   ```
   uv build --wheel
   t0=$(date +%s); docker compose build 2>&1 | tee "$R/build.log"; echo "build: $(( $(date +%s) - t0 )) s" | tee "$R/time_build.txt"
   docker run --rm --gpus all --entrypoint nvidia-smi decisio:local -L | tee "$R/gpu_in_container.txt"
   ```
3. First start, and the time to healthy.
   ```
   t0=$(date +%s); docker compose up -d
   until curl -sf http://127.0.0.1:8000/health > "$R/health.json"; do sleep 10; done
   echo "time to healthy, first start (checkpoint download included): $(( $(date +%s) - t0 )) s" | tee "$R/time_first_start.txt"
   docker compose logs --no-color > "$R/server_first_start.log"
   docker compose ps --format json > "$R/compose_ps.json"
   ```
   If it is not healthy after an hour, stop: the logs are the record of a failed gate, and the fix goes through a pull request.
4. The image, the wheel and the checkpoint, by digest.
   The image of a local build has an ID, not a registry digest; the base image's digest is the `FROM` line of the `Dockerfile`.
   The checkpoint's snapshot directory is its commit revision, and each weight file links to a blob named by its sha256.
   ```
   docker image inspect decisio:local > "$R/image_inspect.json"
   sha256sum dist/decisio-*.whl | tee "$R/wheel.sha256"
   docker compose exec -T decisio id | tee "$R/container_user.txt"                  # uid 10001 (decisio), not root
   docker compose exec -T decisio find /data/hf/hub/models--Qwen--Qwen3.6-35B-A3B-FP8/snapshots -maxdepth 2 \
     -printf '%P -> %l\n' > "$R/checkpoint_files.txt"
   nvidia-smi > "$R/nvidia_smi.txt"; docker version > "$R/docker_version.txt"; nvidia-ctk --version > "$R/nvidia_ctk_version.txt"
   ```
5. The README's example request, exactly as the README prints it.
   ```
   python3 - <<'PY' > "$R/example_request.json"
   import json, re
   s = open("README.md").read()
   m = re.search(r"curl http://127\.0\.0\.1:8000/v1/systemone[^\n]*-d '(\{.*?\n\})'", s, re.S)
   print(json.dumps(json.loads(m.group(1)), indent=1))
   PY
   curl -sS -o "$R/example_response.json" -w "%{http_code}\n" http://127.0.0.1:8000/v1/systemone \
     -H 'Content-Type: application/json' -d @"$R/example_request.json" | tee "$R/example_status.txt"
   ```
   Expect `200` and answers `urgent` (noul), `category` (choice) and `score` in the shapes the README shows; read the values, they are the README's illustration and are not required to match it.
6. Conformance C2-C4 from the machine that runs the container (its host), against the container.
   ```
   uv sync --extra bench --frozen
   uv run python -m decisio.serve.systemone_conformance --url http://127.0.0.1:8000 --items conformance_items.json \
     --tokenizer Qwen/Qwen3.6-35B-A3B-FP8 --block-size 1056 --n 200 --out "$R/conformance.json" 2>&1 | tee "$R/conformance.log"
   ```
   Expect `CONFORMANCE PASS`, with C4's `max_abs_delta_p` exactly `0.0`.
   Gzip a JSON file of 100 KB or more, as the other runs do.
7. Second start, with the checkpoint already in the volume.
   ```
   docker compose down                      # keeps the decisio-data volume
   t0=$(date +%s); docker compose up -d
   until curl -sf http://127.0.0.1:8000/health > /dev/null; do sleep 5; done
   echo "time to healthy, second start: $(( $(date +%s) - t0 )) s" | tee "$R/time_second_start.txt"
   curl -sS http://127.0.0.1:8000/v1/systemone -H 'Content-Type: application/json' -d @"$R/example_request.json" \
     -o "$R/example_response_second_start.json"
   ```
   The chosen options must be the same as in step 5; the probabilities may differ in the last bits (the request history differs, `EVAL_CARD.md` section 4).
8. Write the run record in `runs/<date>_docker-first-gpu-start/`: a `manifest.json` with the fields of the earlier runs (see `runs/2026-09-30_plugin-verification/manifest.json`: `id`, `title`, `date`, `hardware`, `software` with the image ID, the base image digest, the wheel's sha256, the checkpoint revision, the driver and the Docker and toolkit versions, `code` with the commit, `servers` with the compose command, `gates` with health, the example and C2-C4, `results` with the three times, `files`), and a `files.json` with every file's sha256:
   ```
   python3 - <<'PY'
   import hashlib, json, pathlib, sys
   root = pathlib.Path(sys.argv[1])
   rows = [{"path": str(p.relative_to(root)), "bytes": p.stat().st_size, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
           for p in sorted(root.rglob("*")) if p.is_file() and p.name != "files.json"]
   json.dump(rows, open(root / "files.json", "w"), indent=1)
   PY
   ```
   (Run it as `python3 - "$R"`.)
   No benchmark item text goes into the run: the responses are to the README's own request, and the conformance records keep ids and probabilities only.
9. Tear down: `docker compose down`, copy `$R` back to the maintainer's machine, destroy the virtual machine, and add the run to the repository through a pull request.
10. The gate is passed when all of these hold: the container was healthy and ran as `decisio`; the example returned `200` with the documented shapes; C2, C3 and C4 passed, with C4's maximum difference exactly `0.0`; the second start was healthy from the cached volume; and the run is committed.
    One failure leaves the README sentence in place and no tag is made.
11. When it passes, in the same pull request as the run, replace the README's status sentence in "Run with Docker" with the run's path and the two times to healthy.
    After the first release, repeat steps 3 to 5 once with the pushed image, by digest from the release notes (set `image:` in a copy of `compose.yaml` instead of `build:`): the digest on `ghcr.io` is a different build from the one tested here, and the repeat is the check that it starts the same way.

**Result of the first run (2026-10-01): passed.**
The record is `runs/2026-10-01_docker-first-gpu-start` (manifest, files with their hashes).
One RTX PRO 6000 Blackwell Workstation Edition on a vast.ai virtual machine, `main` at `eb9ae2e`: the image built in 236 s; healthy in 651 s on the first start (the 36 GB checkpoint download at about 121 MB/s, 338 s of loading, and two CUDA graph captures of 144 s and 75 s) and in 206 s from the cached volume; the example request returned `200` and the same answer after the restart (difference 0.0); C2 passed, C3 gave identical prompts for 200 of 200 items, and C4's maximum difference was exactly `0.0`.
Two deviations, both in the manifest: the machine had Python 3.10 only (uv installed 3.12 for the wheel build), and the conformance client ran in a second container of the same image on the host network, because the machine reached PyPI at about 0.7 MB/s and a host environment with the CUDA wheels of torch (`uv sync --extra bench`) would have taken over an hour; on a machine with a normal link, use step 6 as written.
**Repeat with the pushed image (2026-10-01, the `0.1.0` image, `ghcr.io/aminry/decisio@sha256:756a13db4e03ad1855bb8d0f71ec6f471e1a857194c1341fcf5bd07cb70a02dd`): passed.**
Pulled by digest in 193 s and started with `image:` in place of `build:`: the card is visible in the container, healthy in 223 s with the checkpoint already in the volume, the example request returns `200`, and C2 to C4 pass again (C4 maximum difference exactly `0.0`); the record is `pushed_image_0.1.0` in the run.
One finding from it, in `repeat_variability`: the README's three-question example does not return identical probabilities on every request, on either image (30 repeats on one server gave two distinct vectors, 22 and 8; each question asked alone gave one value in 30 of 30; `access` ranged from 0.79 to 0.91 over the whole record; the chosen options never changed).
The locally built image and the pushed image behave alike, so this is the serving stack's batching, not a difference between the images, and the README's note on the example's spread should quote the larger figure.

### 3.1 Register the trusted publisher [CLICK]

PyPI has no API for this.
The project does not exist yet, so it is a pending publisher: the project is created by the first upload.

1. Sign in at <https://pypi.org> (two-factor authentication on), then open <https://pypi.org/manage/account/publishing/>.
2. Add a new pending publisher, GitHub tab: PyPI project name `decisio`, owner `aminry`, repository name `decisio`, workflow name `publish.yml`, environment name `pypi`.

The name `decisio` was free on PyPI and on TestPyPI on 2026-09-30; register it close to the first release, because a pending publisher does not reserve the name.
No PyPI token is ever stored in GitHub.

### 3.2 Create the `pypi` environment [API]

The environment must exist, with its reviewer, before the workflow in 3.3 is merged: a job that names a missing environment creates it, unprotected.
The reviewer is the owner (user id 6414758, from `gh api user --jq .id`); the environment accepts deployments only from release tags.

```
gh api -X PUT repos/aminry/decisio/environments/pypi --input - <<'JSON'
{
  "wait_timer": 0,
  "prevent_self_review": false,
  "reviewers": [ { "type": "User", "id": 6414758 } ],
  "deployment_branch_policy": { "protected_branches": false, "custom_branch_policies": true }
}
JSON

gh api -X POST repos/aminry/decisio/environments/pypi/deployment-branch-policies \
  -f name='v*' -f type=tag
```

Settings page: Settings, Environments, pypi.

### 3.3 The publish workflow [pull request]

Add as `.github/workflows/publish.yml`, through a pull request, after 3.2.
Three jobs: `build` has a read-only token and no publishing rights; `provenance` signs a GitHub build-provenance attestation for the files; `publish` runs only the uploader, with `id-token: write` granted on that job alone, and waits for the `pypi` environment's reviewer.
`pypa/gh-action-pypi-publish` also uploads PEP 740 attestations when it publishes through a trusted publisher.
Every action is pinned by commit hash, each checked against its release tag and each the latest release on 2026-09-30.

```
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
# Publishes a release to PyPI when release-please tags it. Trusted publishing (OIDC): no PyPI token is stored.
# build (no credentials) -> provenance (GitHub artifact attestation) -> publish (the uploader only; it waits for the
# required reviewer of the "pypi" environment). The tag is made with the release app's token, because a tag
# made with the default token would not start this workflow.
name: publish

on:
  push:
    tags: ["v[0-9]+.[0-9]+.[0-9]+"]

permissions: {}

jobs:
  build:
    name: build the distributions
    runs-on: ubuntu-latest
    timeout-minutes: 15
    permissions:
      contents: read
    steps:
      - name: Harden the runner
        uses: step-security/harden-runner@e14015d583714f6e62063499dc959a02595150a1 # v2.21.1
        with:
          egress-policy: audit

      - uses: actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1 # v7.0.1
        with:
          persist-credentials: false

      - uses: astral-sh/setup-uv@c18668ad3cf93ea998bef934396af7bb5c839dc7 # v10.2.0
        with:
          python-version: "3.12"

      - name: The tag matches the package version
        run: |
          version=$(uv run --no-project python -c "import tomllib; print(tomllib.load(open('pyproject.toml', 'rb'))['project']['version'])")
          test "v${version}" = "${GITHUB_REF_NAME}"

      - name: Build
        run: uv build

      - uses: actions/upload-artifact@043fb46d1a93c77aae656e7c1c64a875d1fc6a0a # v7.0.1
        with:
          name: dist
          path: dist/
          if-no-files-found: error
          retention-days: 7

  provenance:
    name: attest build provenance
    needs: build
    runs-on: ubuntu-latest
    timeout-minutes: 10
    permissions:
      contents: read
      id-token: write
      attestations: write
    steps:
      - uses: actions/download-artifact@3e5f45b2cfb9172054b4087a40e8e0b5a5461e7c # v8.0.1
        with:
          name: dist
          path: dist/

      - uses: actions/attest-build-provenance@4d101475d8b20a2381f78447822ac1eab6504dd8 # v4.2.2
        with:
          subject-path: dist/*

  publish:
    name: upload to PyPI
    needs: [build, provenance]
    runs-on: ubuntu-latest
    timeout-minutes: 10
    environment:
      name: pypi
      url: https://pypi.org/project/decisio/
    permissions:
      id-token: write
    steps:
      - uses: actions/download-artifact@3e5f45b2cfb9172054b4087a40e8e0b5a5461e7c # v8.0.1
        with:
          name: dist
          path: dist/

      - uses: pypa/gh-action-pypi-publish@dc37677b2e1c63e2034f94d8a5b11f265b73ba33 # v1.14.2
        with:
          packages-dir: dist/
          attestations: true
```

Lint it with `actionlint` before opening the pull request.

### 3.4 First release [CLICK]

`0.1.0` was released on 2026-10-01 without a PyPI upload: its tag was made before `publish.yml` was on `main`, and a tag's workflow runs from the file at the tagged commit, so it cannot be repeated for that tag (tags cannot be moved).
The first PyPI release is `0.1.1`, with the image and its SBOM attestation from the same tag.

Not before 3.0 has passed: the release tag publishes the image.

1. The first public release is `0.1.0`.
   `pyproject.toml` and `.release-please-manifest.json` already say `0.1.0` and no tag exists, so release-please would propose the next version: before the flip, through a pull request, add `"release-as": "0.1.0"` to the package entry in `release-please-config.json`, and remove it in the pull request after the release.
   Then review the release pull request that release-please keeps open on `main` (its `CHANGELOG.md` and version bump) and merge it (squash).
2. Merging creates the tag `v<version>` and the GitHub Release; the tag starts `publish.yml`.
3. [CLICK] The `publish` job waits for approval: Actions, the run, "Review deployments", tick `pypi`, Approve and deploy.
4. Check the release (section 4).

### 3.5 The container image on ghcr.io [CLICK]

`.github/workflows/docker.yml` builds the image on every release tag (`v0.1.0` and so on), pushes it to `ghcr.io/aminry/decisio:<version>`, attests build provenance and an SBOM, and writes the image digest into the release notes.
On pull requests it only builds (for linux/amd64, nothing pushed).
Its release jobs are skipped while the repository is private, so the first image is made by the first release tag after the flip.
It needs nothing beyond 2.2's allowed actions (the Docker and Anchore ones are listed there) and the workflow's own `GITHUB_TOKEN`; there is no registry secret.

1. The first push creates the package as private, linked to the repository through the `org.opencontainers.image.source` label.
   [CLICK] Make it public: <https://github.com/users/aminry/packages/container/decisio/settings>, Danger Zone, Change visibility, Public.
   GitHub has no API for changing a package's visibility.
2. Check that the repository has write access to the package (Package settings, Manage Actions access); a package created by the workflow gets it automatically.
3. Verify the release: pull by digest from the release notes, then verify the attestations.

```
docker pull ghcr.io/aminry/decisio@sha256:<digest in the release notes>
gh attestation verify oci://ghcr.io/aminry/decisio@sha256:<digest> --repo aminry/decisio
gh attestation verify oci://ghcr.io/aminry/decisio@sha256:<digest> --repo aminry/decisio --predicate-type https://cyclonedx.org/bom
```

The SBOM is CycloneDX because the SPDX document of this 9 GB image is 49 MB and `actions/attest-sbom` accepts 16 MiB (CycloneDX: 8.4 MB, measured with syft on the `0.1.0` image).
The `0.1.0` image has a build-provenance attestation and no SBOM attestation (the workflow's SBOM step failed on that tag, and its release notes say so); `0.1.1` is the first release with both.
The base image is pinned by digest in the `Dockerfile`; Dependabot refreshes the digest of the pinned tag and never moves the tag.
The image has not been run on a GPU; 3.0 is the gate that must pass before the tag that publishes it, and its repeat with the pushed digest comes after.

## 4. Verify

Run after each step; the expected value is in the comment.

```
gh api repos/aminry/decisio --jq '{private, allow_squash_merge, allow_merge_commit, allow_rebase_merge, delete_branch_on_merge, web_commit_signoff_required}'
#   private false, squash true, merge false, rebase false, delete_branch_on_merge true, web_commit_signoff_required true

gh api repos/aminry/decisio/actions/permissions
#   enabled true, allowed_actions "selected", sha_pinning_required true
gh api repos/aminry/decisio/actions/permissions/selected-actions
#   github_owned_allowed true, verified_allowed true, the five patterns
gh api repos/aminry/decisio/actions/permissions/fork-pr-contributor-approval
#   approval_policy "all_external_contributors"
gh api repos/aminry/decisio/actions/permissions/workflow
#   default_workflow_permissions "read", can_approve_pull_request_reviews false

gh api repos/aminry/decisio/rulesets --jq '.[] | {id, name, enforcement}'
#   "main" and "release tags", both active
gh api repos/aminry/decisio/rules/branches/main --jq '[.[].type] | sort'
#   deletion, non_fast_forward, pull_request, required_linear_history, required_signatures, required_status_checks
gh api repos/aminry/decisio/rulesets/<main id> --jq '.bypass_actors'
#   one RepositoryRole entry with bypass_mode "pull_request"; the settings page shows it as "Repository admin"

gh api repos/aminry/decisio --jq '.security_and_analysis'
#   secret_scanning enabled, secret_scanning_push_protection enabled
gh api repos/aminry/decisio/private-vulnerability-reporting --jq .enabled
#   true
gh api repos/aminry/decisio/code-scanning/default-setup --jq '{state, languages}'
#   configured, ["actions","python"]

gh variable list --repo aminry/decisio
gh secret list --repo aminry/decisio
#   the variables of 2.7 and the secrets of 2.6; values of secrets are never shown
gh api repos/aminry/decisio/environments/pypi --jq '{protection_rules, deployment_branch_policy}'
#   the owner as reviewer, custom branch policies
```

Behaviour, in a scratch clone (none of these should succeed):

```
git commit --allow-empty -s -m "test: ruleset probe" && git push origin HEAD:main   # rejected: changes must be made through a pull request
git push --force origin HEAD:main                                                    # rejected
git push origin :main                                                                # rejected: deletion restricted
```

Do not probe the tag ruleset by pushing a tag: creation is allowed and the tag then cannot be deleted without switching the ruleset off; once a real release tag exists, try to delete or move that one instead.

And in the browser:

- [ ] The first pull request shows `lint, unit tests, build` and `DCO` as required, and "Squash and merge" as the only merge button.
- [ ] A pull request from a second account's fork shows "Approve and run workflows" and runs nothing until the owner approves.
- [ ] Security, Code scanning shows a finished CodeQL run with no error in the workflow analysis.
- [ ] Security, "Report a vulnerability" is available.
- [ ] After the first release: <https://pypi.org/project/decisio/> lists the version with a verified attestation, and `gh attestation verify <file>.whl --repo aminry/decisio` succeeds on the downloaded wheel.

## 5. What changes for contributors and for the maintainer's own agents

- Nobody, the owner included, pushes to `main`; every change is a pull request with a signed-off commit and a Conventional Commit title.
  Automated agents that committed directly to a branch of this repository keep doing so, and open pull requests instead of pushing to `main`.
- The squash commit on `main` takes its title from the pull request title; write that title as the commit you want in the changelog.
- A pull request waits for CI on the current `main`; use "Update branch" when `main` has moved.
- A pull request from outside shows "Approve and run workflows"; read the diff of any change under `.github/` before approving.
- To switch the main ruleset off in an emergency: `gh api -X PUT repos/aminry/decisio/rulesets/<id> -f enforcement=disabled`; switch it back on the same way with `active`.
  Both changes appear in the ruleset's audit history.

## 6. Not covered here

- The image has been built and its start-up checked without a GPU only; a GPU start, and the measured numbers through the container, are not part of this checklist.
- A merge queue: it is available only to repositories owned by an organisation; `ci.yml` already runs on `merge_group`, so moving the repository to an organisation needs no workflow change.
- OpenSSF Scorecard starts running on its own once the repository is public (`scorecard.yml` skips while it is private); check its first result on the Actions tab.
- Second maintainer: when there is one, turn on one required approval and required code owner review in the main ruleset, and review `GOVERNANCE.md`.
