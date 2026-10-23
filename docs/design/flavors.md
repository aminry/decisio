<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Copyright contributors to the decisio project -->

# Design: flavors, named configurations of the same weights

Status: proposed.
Nothing here is built, and nothing ships until it is chosen.

The same weights are asked to do three different jobs, and each job wants different settings.
A flavor is a named bundle of those settings, with its own served-name suffix, so that a client, a certificate and a board row can say which one they mean.

## What a flavor is

Direction from Amin (2026-10-23), three flavors of one set of weights:

| Flavor | Temperatures | Thinking | Commit rule | Read twice |
| --- | --- | --- | --- | --- |
| `calibrated` (the honest one) | Calibrated for every type, yes/no included, with option-count bands for choice | Off | Off | Off |
| `jevbench` | As the board rewards (the current yes/no temperature) | When unsure | On | Off |
| `decision-index` | Not yet decided (question 3 below) | Off | Off | On, with the threshold and the question cap (`--read-twice-under`, `--read-twice-max-questions`) |

A tuned adapter is added to whichever flavors its gates support, after its training.
A flavor changes how the weights are read and what is reported, never the weights.

What each setting is, today, and where it lives:

| Setting | Today's flag | Today's other home |
| --- | --- | --- |
| Temperatures per type | `--temperature`, `--temperature-<type>` | The base's profile has two numbers only (`Family.temperature`, `choice_temperature`); a tuned repository's `defaults.temperatures` and `choice_bands`; `--temperatures-file` |
| Choice bands | `--temperatures-file`, repository `defaults.choice_bands` | Not in the profile |
| Commit rule | `--noul-commit` | Repository `defaults.noul_commit` |
| Thinking | `--think`, `--think-factor`, `--think-budget`, `--think-temperature-<type>` | Repository `defaults.think`, `defaults.think_temperatures` |
| Read twice | `--read-twice-under N`, `--read-twice-max-questions R` | None |
| Layout | `--layout` | Repository `defaults.layout`; the base's `Family.layout` |

Two things in the table are being built on their own and a flavor depends on them: per-type temperatures and option-count bands in the base's profile (Amin's decision of 2026-10-23), and the fit of the calibrated values on the suite read under the header (Lab 2, not yet scheduled).

## What is measured and what is not

- **`decision-index`:** the pieces exist and are merged (read twice, #296). N = 2,000 is the measured choice; R is waiting for Lab 4's timing cell; a confirming run on a card held at 600 W is not done.
- **`jevbench`:** the pieces exist (thinking, the commit rule, think temperatures, #278). "As the board rewards" is a claim, not a number: the yes/no temperature that is best for the board's calibration axis with the commit rule on has not been fitted. With the rule on, the yes/no count no longer depends on the temperature (the rule puts an in-band answer at the edge of its side), so the temperature there only moves the calibration axis.
- **`calibrated`:** does not exist yet. The check of 2026-10-23 (`experiments/2026-10-23_serve_profile_temperatures`, RLCD) found that every fit set asks for a yes/no temperature of 12 to 21 on the 31B under the header (5.252 today) and 6 to 9 on the 12B in FP8 (3.592), and that this improves log loss, Brier and ECE on held-out sets while moving 20 to 69% of yes/no answers into the 0.2 to 0.8 band. That is what the flavor is for: it is honest about calibration and pays the board's penalty for it.
  The values need the suite's header readouts to be called fitted by decisio's method.

A flavor can be listed before its values are final, but it must not be served under its name before they are, so the first release carries only the flavors whose values are measured.

## Decisions asked for

### 1. How a flavor is expressed

| Option | Behaviour | For | Against |
| --- | --- | --- | --- |
| A. `--flavor NAME` on the server, values in decisio | A table per base in the code (like `Family`): the flavor sets the flags it names | One word to start a deployment. The values are reviewed, tested and versioned with the code. The base's own measurements live next to its profile | A tuned repository cannot add a flavor without a decisio release |
| B. A repository `defaults` block | `defaults.flavors: {name: {...}}` in `decision_config.json`, chosen by `--flavor` | The tuned model ships with the flavors its gates support. No release for a new tuned model | The untuned bases have no repository, so they would still need A; two formats |
| **C. One block format, two homes** (recommended) | The same flavor block (temperatures, choice bands, commit, think and its temperatures, read twice, layout) is read from the base's table in the code and from a tuned repository's `defaults.flavors`; `--flavor NAME` selects | One schema to validate and document. A tuned repository lists exactly the flavors it was gated for, and a flavor the repository lacks is refused, so a tuned model cannot be served under a name it was not measured for | A schema, a validator and two readers (about the cost of `defaults.layout`) |
| D. Only the flags, documented as recipes | Three command lines in `running.md` | Nothing to build | No served name, no certificate boundary, no way for a board row to name what it ran. Rejected |

**Recommendation: C.**
A repository file that records `defaults.flavors` needs the release that adds it, as `defaults.layout` and `defaults.think_temperatures` did.
Precedence is the existing one with the flavor inserted: the flavor, then the repository's `defaults`, then the base's profile; a flag that contradicts the flavor is refused (below), and a flag that only adds to it (a temperatures file for a type the flavor does not set) is applied.

**An explicit flag that contradicts the flavor is refused**, not overridden: `--flavor calibrated --noul-commit` stops with "calibrated serves without the commit rule; start without --flavor for a custom setup".
The settings that define a flavor are temperatures, bands, thinking, the commit rule, read twice and the layout.
Infrastructure flags (ports, memory, concurrency, context length) are not touched by a flavor.
The reason is the name: a server called `...-calibrated` that quietly runs the commit rule is the one failure this design exists to prevent.

### 2. The served-name suffix

Today the served name is `decisio-gemma-4-31b-it-letters` (a base's `served_name`), and it appears in the response's `model`, in the task fingerprint and in the `model` of abstention tasks.

| Option | Example | For | Against |
| --- | --- | --- | --- |
| **A. Hyphen suffix with the flavor's name** (recommended) | `decisio-gemma-4-31b-it-letters-calibrated`, `...-jevbench`, `...-decision-index` | Readable. Valid in every place a model name goes. Stays one token for a client that keys on the name. Sorts next to the unflavored name | `decision-index` has a hyphen, so the suffix is found by matching the known flavor names from the end, not by splitting |
| B. `@flavor` or `:flavor` | `...-letters@calibrated` | Visibly a variant | `:` is a tag separator in Ollama and Docker names; `@` is a digest separator. Both read as something else |
| C. Short codes | `-cal`, `-jb`, `-di` | Short | Unreadable in a board row or a log; collisions with future flavors |

**Recommendation: A**, with the flavor names exactly `calibrated`, `jevbench`, `decision-index`.
A tuned adapter's served name keeps its own stem and takes the same suffix (`decisio-n1f-12b-letters-calibrated`).
The flavor is also reported in `/health` (`systemone.flavor`) and in a response header (`x-decisio-flavor`), so a client that sends its own `model` string still sees which flavor answered.

### 3. Which flavor is served when none is named

| Option | Behaviour | For | Against |
| --- | --- | --- | --- |
| **A. No flavor: today's behaviour, today's name** (recommended for the first release) | `--flavor` is opt-in. A server started as before serves what it served, under the same name | Nothing changes for any existing deployment, task, certificate or board row. The release is additive | The unflavored server is none of the three: its yes/no temperature is the board's, its commit rule is off, there is no thinking. It is a legacy setting with no name |
| B. `calibrated` is the default | A bare server answers as the calibrated flavor | The honest one is the default, which is where decisio's intent points | Changes the served probabilities of every deployment on upgrade, suspends every certificate, and **lowers the JevBench yes/no count without the commit rule** (measured: hard390 0.589 to 0.233 at the fitted temperature). A board row measured on the old default stops being comparable. It would be the second answer-changing default in two releases |
| C. `jevbench` is the default | The commit rule on and thinking on by default | Matches what the board rewards | Changes yes/no probabilities to the band's edges for every user and turns on thinking, which costs latency. Against the project's stated rule that the commit rule is off unless asked |

**Recommendation: A now, and a separate decision later.**
When the suite is read and `calibrated` exists, the question is whether a bare server becomes `calibrated`, announced a release ahead, with the upgrade note and the `--flavor` lines that keep each old behaviour (the way the header default was done in 0.26.0).
A "legacy" flavor name for today's unflavored behaviour would make that move possible without losing the old setting; it can be added at that time.

### 4. How fingerprints, tasks and certificates separate

The suffix does most of the work, because the served name is already part of what a registration is kept under:
- **The task fingerprint** holds the served name (`decisio.names.task_fingerprint`). A calibration task, an intent head or a prior registered under one flavor is not applied under another, with no new code. Tasks registered with no flavor keep their fingerprint and apply to an unflavored server only.
- **Abstention tasks** carry `model` (the served name) and apply only under it.
- **Read twice and the layout** are already in the prompt format's facts (0.26.0 and #296), so their tasks separate by prompt format as well.
- **Certificates** are checked against the serving-stack fingerprint (`SystemOne.stack_facts`), which holds the temperatures, the commit rule and the think settings. Two flavors differ in those, so a certificate made under one is `serving_stack_changed` under the other. The flavor name is added to the serving facts **only when set**, so that the fingerprint of an unflavored server is what it was.
- **What does not separate on its own:** two flavors that differed only in the name. There are none; each flavor sets at least one fingerprinted setting.
- **A shared task store** (`--task-store`) already keeps tasks apart by fingerprint, so two flavors of one deployment can share a store without seeing each other's tasks. Whether a task should be usable across flavors that differ only in post-processing (for example, a calibration task registered under `calibrated` read by a server in `jevbench`) is a convenience that this design does not build: a task is kept under the flavor it was fitted under, the same rule as the layout.

### 5. One flavor per server, or several

| Option | Behaviour | For | Against |
| --- | --- | --- | --- |
| **A. One flavor per server process** (recommended) | `--flavor` at start. Several flavors on one card need several servers | Simple. Every fingerprint, `/health` and certificate is for one flavor. Nothing to route | The 31B does not fit on a card twice, so one card serves one flavor at a time |
| B. Several flavors on one server, chosen by the request's `model` | The client names the flavor; one engine, three configurations | One card, one set of weights in memory, all three flavors | The stack fingerprint is taken once per server; it would become per request. Thinking changes the engine's readout path (`columns_off`), read twice and the layout change the prompt, and batching mixes requests of different flavors. Real work and real risk |

**Recommendation: A for the first release, B only if a deployment needs it.**
What the DI kit and the JevBench harness send as `model` is not known to this note; if they send a fixed or arbitrary string, B would not be selectable by them anyway.

## Release and versioning

- **Additive, so a minor release and not breaking**, if question 3 is answered A: no route, field or schema changes; no default changes; an unflavored server is bit for bit what it was. The commit is `feat(serve):`, not `feat(serve)!:`.
- **It rides with the work it depends on:** the per-type and band temperatures in the profile (a separate PR), the confirmed values of each flavor, and (if wanted) the suite's header readouts. The held 0.27.0 (#297) already has read twice and the think temperatures; `--flavor` would be 0.28.0, or join 0.27.0 if the release is still held when it is ready.
- **A repository that records `defaults.flavors` is refused by an older decisio** (an unknown key of `defaults`), as `defaults.layout` was. The release notes say so, and the hub profile is not changed.
- **The card:** `EVAL_CARD.md` states the flavor of every figure it prints, as it now states the layout. A board row names its flavor.
- **Documentation:** a `running.md` section ("Flavors") with the table of what each sets, `cli.md` and the generated `flags.md`, an `upgrading.md` entry (who is affected: nobody, with no flavor named), and this note's "As built" section.

## Build plan (after the choices)

| Step | Working hours |
| --- | ---: |
| The flavor block: schema and validator, the base's table in `families.py` or a `flavors.py`, `defaults.flavors` in the repository file (`hub tuned-config --flavor`) | 4 |
| `--flavor`: resolve at start (refuse contradicting flags and unknown or unsupported names), the suffix on the served name, `/health`, `x-decisio-flavor`, the flavor in the serving facts only when set | 4 |
| Tests: each flavor on each base resolves to the intended settings; contradictions refused; an unflavored server's fingerprint and name unchanged; tasks and certificates separate per flavor; a repository without the flavor refuses it | 4 |
| Docs and the regenerated reference | 3 |
| **Mechanism, total** | **15** |
| The values: `jevbench` and `decision-index` need only their measured settings (already merged); `calibrated` waits for the suite's header reads and the profile's per-type temperatures and bands | outside this estimate |

## Questions for Amin

1. **Expression:** C (one block format, the base's table and a tuned repository's `defaults.flavors`, selected by `--flavor`), with a contradicting flag refused.
2. **Suffix:** hyphen suffix with the flavor's name.
3. **Which temperatures does `decision-index` use?** The Decision Index scores calibration columns, so the calibrated temperatures would help it; if it uses the board's, it is the `jevbench` temperatures without thinking or the commit rule. Recommendation: the calibrated ones, once they exist.
4. **Default:** unflavored stays as it is; revisit once `calibrated` exists.
5. **One flavor per server** in the first release.
