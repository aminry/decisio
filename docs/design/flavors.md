<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Copyright contributors to the decisio project -->

# Design: flavors, named configurations of the same weights

Status: decided by Amin on 2026-10-23 (names, overrides, one flavor per server, the 31B served in FP8 only); the mechanism is being built.
Nothing here changes a server that names no flavor.

The same weights are asked to do three different jobs, and each job wants different settings.
A flavor is a named bundle of those settings with its own served name, so that a client, a certificate and a board row can say which one they mean.

## The three flavors

| Flavor | Used for | Temperatures | Thinking | Commit rule | Read twice | Layout |
| --- | --- | --- | --- | --- | --- | --- |
| `calibrated` | The honest one | Calibrated for every type, yes/no included, with option-count bands for choice | Off | Off | Off | Header |
| `thinking-commit` | JevBench | As the board rewards (today's yes/no temperature) | When unsure | On | Off | Header |
| `read-enhanced` | The Decision Index | The calibrated ones | Off | Off | On, with the threshold and the question cap | Header |

Every flavor uses the header layout ("questions first across the board").
A tuned adapter joins a flavor only after its training shows where it helps; those names are decided then.
A flavor changes how the weights are read and what is reported, never the weights.

## Served names

The served name is the model name a client sends and a response reports: `<base>-<flavor>`.

| Base | Precision | Served names |
| --- | --- | --- |
| Gemma 4 31B | FP8 on load (the only one served) | `Decisio-31b-calibrated`, `Decisio-31b-thinking-commit`, `Decisio-31b-read-enhanced` |
| Gemma 4 12B | bf16 | `Decisio-12b-calibrated`, `Decisio-12b-thinking-commit`, `Decisio-12b-read-enhanced` |
| Gemma 4 12B | FP8 | `Decisio-12b-fp8-calibrated`, `Decisio-12b-fp8-thinking-commit`, `Decisio-12b-fp8-read-enhanced` |

- **Only the 12B has two precisions**, so only the 12B names carry one: `Decisio-12b-*` is the bf16 base and `Decisio-12b-fp8-*` the FP8 one. The 31B is served FP8 and has no bf16 flavor, so its names carry none.
- A server started with no flavor keeps today's name and behaviour (`decisio-gemma-4-31b-it-letters` and its siblings).
- The Qwen base has no flavors; `--flavor` is refused with it.

**Capitals.** The names are written with a capital D and matched case-insensitively:
- A request's `model` that equals the served name in any case (`decisio-31b-calibrated`, `DECISIO-31B-CALIBRATED`) is accepted.
- `GET /v1/models`, every response's `model`, the task fingerprint and the abstention tasks' `model` hold the capitalised name, so a record never depends on how a client spelt it.
- A request that names *another flavor of the same base* (`Decisio-31b-calibrated` sent to a `thinking-commit` server) is refused with the served name in the message, since answering it would be a lie. A `model` that names no flavor is accepted as before and the response reports the served name.
- Where capitals cannot go: a **Docker image repository** name must be lowercase (a tag may have capitals); **Ollama** model names are lowercase in the existing listing (`aminroudaki/decisio`) and I have not tested capitals there, so the rule is lowercase for every artifact name. **Hugging Face** repository names keep their case. A flavor is a runtime setting, not an artifact: there is no flavor-specific image, repository or tag in this design, so nothing needs a capital that cannot take one. If one is ever made, its artifact name is the lowercase form (`decisio-31b-calibrated`) and the capitalised name stays the served name.
- The `model` string never appears in a URL path in this API (`POST /v1/systemone` takes it in the body), so case in a path is not a concern.

## What a flavor sets, and where it lives

| Setting | Today's flag | Today's other home |
| --- | --- | --- |
| Temperatures per type | `--temperature`, `--temperature-<type>` | The base's profile (`Family`: global and choice; yes/no, score and bands since #303); a tuned repository's `defaults.temperatures`; `--temperatures-file` |
| Choice bands | `--temperatures-file`, repository `defaults.choice_bands` | The base's profile (#303) |
| Commit rule | `--noul-commit` | Repository `defaults.noul_commit` |
| Thinking | `--think`, `--think-factor`, `--think-budget`, `--think-temperature-<type>` | Repository `defaults.think`, `defaults.think_temperatures` |
| Read twice | `--read-twice-under N`, `--read-twice-max-questions R` | None |
| Layout | `--layout` | Repository `defaults.layout`; the base's `Family.layout` |

A flavor is a block of exactly these settings, kept in a table per base next to its profile (`decisio.flavors`).
`--flavor NAME` (the short flavor name, `calibrated`, `thinking-commit` or `read-enhanced`) applies it.
The precedence is: an explicit flag, then the flavor, then the base's profile.

## Overrides are allowed, and visible

An explicit flag wins over the flavor, on any flavor, and nothing is refused.
What keeps it honest is that the override is recorded everywhere the flavor is:
- **`/health`** carries `systemone.flavor`: the name, the served name, the `overrides` (each flag that differs from the flavor, with the flavor's value and the one served) and `pending`.
- **`x-decisio-flavor`** on every response is the short name, `calibrated`, or `calibrated+overrides` when any flag differs, and `+pending` when a value is still a placeholder.
- **The start line** prints `FLAVOR Decisio-31b-calibrated` with each override and each pending value on a line of its own.
- **The serving-stack fingerprint** holds the flavor, its overrides and its pending marks, only when a flavor is set. A certificate made under one flavor, or under the same flavor with other overrides, is `serving_stack_changed` under the other, and an unflavored server's fingerprint is what it was.
- **Tasks:** the served name is already in the task fingerprint, so a task registered under one flavor is not applied under another with no new code. Settings that change the readout (layout, read twice) are also in the prompt format's facts.

An override changes the values but not the name: the server is still `Decisio-31b-calibrated`, marked `+overrides`.
A client or a board row that needs to say what it ran has the flavor and the overrides in `/health`.

## What is measured and what is not

- **`read-enhanced`:** read twice is merged (#296): N = 2,000 counted input tokens is the measured choice. R (`--read-twice-max-questions`) waits for Lab 4's timing cell, and the confirming run on a card held at 600 W is not done. Its temperatures are the calibrated ones, which do not exist yet.
- **`thinking-commit`:** thinking, the commit rule and the think temperatures are merged (#249, #278). What is not measured is thinking on the *untuned* bases at the default factor and budget; the measurements so far are of the tuned models. "As the board rewards" is a claim: the yes/no temperature that is best for the board's calibration axis with the commit rule on has not been fitted (with the rule on, the yes/no count no longer depends on the temperature).
- **`calibrated`:** does not exist yet. The check of 2026-10-23 (RLCD `f74e6ad9`) found that every fit set asks for a yes/no temperature of 12 to 21 on the 31B under the header (5.252 today) and 6 to 9 on the 12B, which improves log loss, Brier and ECE on held-out sets and moves 20 to 69% of yes/no answers into the 0.2 to 0.8 band. The values need the 1,400-item suite read under the header (Lab 2, not yet scheduled) and the per-type and band profile (#303).

### Flavors with a value still pending

Proposal:
- **A flavor whose defining value is missing refuses to start** (`calibrated`: its whole meaning is its temperatures; the message says they are not fitted yet for that base). No name that promises honesty serves numbers that are not calibrated.
- **A flavor with a missing secondary value serves, marked `pending`**: `read-enhanced` (its temperatures: today's are served until the calibrated ones exist) and `thinking-commit` (the untuned-base thinking measurement). The start line, `/health` and `x-decisio-flavor` (`+pending`) say so, and the fingerprint holds the mark, so a certificate made on placeholder values is suspended when the real ones land.
- The alternative, shipping none of them until every value exists, delays read-enhanced and thinking-commit, which are usable now and are the ones the Decision Index and JevBench runs want.

## Decisions

| Decision | Choice |
| --- | --- |
| How a flavor is expressed | A flavor block per base in `decisio.flavors`, selected by `--flavor`. Tuned repositories do not carry flavors yet; `--flavor` with one is refused until their names are decided |
| Contradicting flags | Allowed and recorded as overrides (above) |
| Served name | `Decisio-<base>-<flavor>`, case-insensitive on input, capitalised on output |
| Which flavor when none is named | None: today's behaviour and today's name. A later decision, announced a release ahead, could make `calibrated` the default; it would change every served probability and suspend every certificate |
| Tasks and certificates | Separate per flavor, through the served name and the stack fingerprint |
| Flavors per server | One. Several on one server, chosen by the request's `model`, is real work (the stack fingerprint is taken once; thinking changes the readout path; batching would mix flavors) and is not built |
| Temperatures of `read-enhanced` | The calibrated ones |

What the DI kit and the JevBench harness send as `model` is not known to this note; it matters only if one server ever serves several flavors.

## Release and versioning

- **Additive, so a minor release and not breaking:** no route, field or schema changes; no default changes; an unflavored server is bit for bit what it was. The commit is `feat(serve):`.
- It follows the per-type and band profile (#303). The held 0.27.0 (#297) has read twice and the think temperatures; `--flavor` would be 0.28.0, or join 0.27.0 if the release is still held when it is ready.
- A repository file is not changed: flavors are in the base's table in the code, so no older repository is refused.
- **The card:** `EVAL_CARD.md` states the flavor of every figure it prints, as it states the layout. A board row names its flavor.
- **Documentation:** a `running.md` section ("Flavors"), `cli.md` and the generated `flags.md`, an `upgrading.md` entry (who is affected: nobody, with no flavor named), and this note's "As built" section.

## Build plan

| Step | Working hours |
| --- | ---: |
| `decisio.flavors`: the block, the validator, the table per base, the naming and the case-insensitive match | 4 |
| `--flavor`: resolve at start (precedence, overrides, pending, the refusals), the served name, `/health`, `x-decisio-flavor`, the start line, the flavor in the serving facts only when set, the model check on requests | 5 |
| Tests: each flavor on each base resolves to the intended settings; an override is recorded and an unflavored server is unchanged; case-insensitive names; another flavor's name refused; tasks and certificates separate per flavor; pending served or refused as above | 4 |
| Docs and the regenerated reference | 2 |
| **Mechanism, total** | **15** |
| The values: `calibrated` and `read-enhanced`'s temperatures wait for the suite's header reads | outside this estimate |

## As built

Not built yet.
