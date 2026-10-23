<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Copyright contributors to the decisio project -->

# Design: the header layout as the default

Status: accepted and built in 0.26.0. All five recommendations were accepted (every request gets the header; `--layout plain` for state reuse, a per-request field only if asked; the layout held at plain where plain-fitted tasks, certificates or adapters are loaded; adapters serve their recorded layout, plain if none; 0.26.0 as a breaking `feat`). What was built is under "As built" at the end; the user documentation is `running.md` ("The header layout") and `upgrading.md`.
`--layout header` (0.25.0 plus #290) lists a request's questions before its state. This note is about making it the layout decisio serves when nothing else is said.

## The evidence, and what it leaves open

On the Gemma 4 31B (FP8 on load, one RTX PRO 6000, decisio c0b1c08d), the header against the served layout, same items, same card:

| Set | Plain | Header | Difference (95%) |
| --- | ---: | ---: | --- |
| Decision Index sample, the kit's own scoring (2,000 rows) | 55.41 | 60.44 | **+5.03** [2.87, 6.94] |
| H776 (single question per request) | 80.67 | 85.05 | **+4.38** [2.32, 6.44] |
| hard390 | 52.31 | 59.74 | **+7.44** [3.08, 11.79] |
| dev500 | 67.40 | 67.00 | -0.40 [-3.0, 2.2] (choice -0.6, yes/no +1.9, score -2.3) |

On the 12B in FP8 the gain on H776 was +0.90 (not distinguishable from 0), and on the Decision Index it was concentrated in requests with many questions about one state (equal family weighting +1.75, interval crossing 0). So the header is measured to help the 31B on three of four sets and not to hurt on the fourth; it is not measured to help every base, every item type or the tuned adapters.

Costs, measured: 3 to 10% more input tokens as counted on single-question items (1.32 times the computed tokens on the multi-question Decision Index requests, which read the header on every row); latency median +1 to +10%, Decision Index mean 303 to 542 ms.

**The cost this note is about** is not in those figures: a client that asks a series of different questions about one state loses the cache. Server time for a question about a state the server has already read, against one it has not (the 31B, `running.md`, "First-question latency"):

| State, tokens | Second question (plain, cached state) | First question (what every request becomes under the header) |
| ---: | ---: | ---: |
| 1,000 | 37 ms | 171 ms |
| 3,000 | 41 ms | 485 ms |
| 6,000 | 54 ms | 1,034 ms |
| 16,000 | 94 ms | 3,370 ms |

Agents, the games and the teach-it demos ask exactly this. Under the header their later questions cost 5 to 36 times as much, with identical answers otherwise.

## What changes when the default changes

Everything that is keyed by the prompt format moves, because the layout is in `PromptFormat.facts()`:
- the task fingerprint: a task registered under the plain layout is kept and **not applied** (its question is answered untaught, and under the header);
- the serving-stack fingerprint: a certificate made under the plain layout is `serving_stack_changed`;
- a tuned repository or an adapter, trained on the plain layout and measured only on it, would be served on a layout it has never seen.

None of that is a crash; all of it is a silent change of answers on upgrade unless the design says otherwise. That is what decisions 3 and 4 are for.

## Decisions asked for

### 1. What "all its questions at once" means for the server

| Option | Behaviour | For | Against |
| --- | --- | --- | --- |
| **A. Every request** (recommended) | The header lists the request's questions, one or many | It is where the measured gain is: H776 and hard390 are single-question requests. One rule, no heuristics. A request's answer stays a function of the request alone | Clients that reuse a state across requests pay for it (table above) unless they opt out (decision 2) |
| B. Multi-question requests only | Single-question requests keep the plain layout | Leaves single-question clients as they are | **Forfeits the measured gain**: JevBench items and H776, hard390 are single-question. The 12B's gain was larger on multi-question requests, but the 31B's was not limited to them |
| C. A request field or a client hint decides | `layout` in the body, or a header | The client knows whether it reuses a state | Needs a default for everyone who says nothing: it is a mechanism, not an answer |

**Recommendation: A, with C as the way out (decision 2).** "Brings all its questions at once" then means: a request lists its questions, and the header lists those. A client that sends one state with a series of requests is the client that opts out.

### 2. How clients that rely on state reuse keep it

| Option | Behaviour | For | Against |
| --- | --- | --- | --- |
| **`--layout plain`** (recommended, first release) | The server serves the plain layout | Exists today (#290). A deployment for agents or games sets it once. No new code | Per server, not per client: a server shared by a series client and a board client cannot serve both layouts |
| **A per-request `layout` field** (recommended, second step) | `"layout": "plain"` in the body, like `think` and `over_limit` | One server serves both. The request says what it needs | Real work: tasks and certificates must carry the layout they were fitted under and a request applies one only if the layouts agree (task keys, the stack facts per task, abstention tasks), and registration must say which layout it fits. About 8 working hours beyond the rest |
| Automatic detection of repeated states | The server notices a state it has seen and uses plain | Needs nothing from the client | **Rejected.** A request's answer would depend on what came before it. decisio's rule is that no history decides anything (the same request gives the same bytes, `running.md` "Concurrency", "Thinking when unsure"); this breaks certificates and every comparison between two runs |

**Recommendation:** ship `--layout plain` as the documented way out with the default change, and the request field as the next change, if clients ask for it. A client sees the cost in `x-decisio-server-ms`; the build adds the cached share of the prompt (the engine already counts it, `cached_tokens_mean`) to the response's timing headers so a client can see why; the release note says what to look for.

### 3. Migration of registered tasks and certificates

Options:

| Option | Behaviour | For | Against |
| --- | --- | --- | --- |
| Loud upgrade note only | Plain-fitted tasks are orphaned; the start prints the existing "not applied" warning | Simple. The precedent: the prompt tail's default changed on 2026-10-03 this way | **Silently changes answers** for every taught deployment on the day of the upgrade |
| Auto-refit | Re-register the tasks under the header | Tasks keep applying | **Impossible**: a task stores its fit and digests of its examples, not the examples. Refitting needs the customer's labelled data |
| **Hold the layout where something needs it** (recommended) | The default is `header` for a base with nothing loaded that was fitted or trained on the plain layout. If the server loads tasks (a directory, a store, a file) or an abstention task fitted under plain, or an adapter or tuned repository with no recorded layout, the default stays `plain`, the start prints `LAYOUT plain (held: <what>); start with --layout header to switch, after registering the tasks again`, and `/health` says the same (`systemone.layout: {"value", "why"}`) | A taught deployment's answers do not change on upgrade. A new deployment, a board run or a bare base gets the header. One explicit flag switches, once the tasks are registered again. The decision is made at start from the configuration and the stores, never from requests | The default depends on what is loaded: deleting the last plain task flips a bare base to the header on the next start (the start line and `/health` say so, and `--layout plain` pins it) |
| Refuse to start | The start stops when the default would orphan something, unless `--layout` is given | Cannot be missed | A restart policy turns an upgrade into a crash loop, on a change the operator did not ask for |

**Recommendation: hold the layout where something needs it, loudly.** Tasks and certificates keep their meaning; the change reaches the deployments that have nothing to lose, and the others move on their own command. `--layout header` on a server holding plain-fitted tasks keeps today's behaviour for those tasks (not applied, said at the start), so the switch is always explicit.

### 4. Tuned adapters and repositories

A fine-tuned adapter was trained on one layout and measured on it. The rule:
- A tuned repository (`decisio.decision_config/2`) and an adapter's file (`decisio_adapter.json`) may record the layout they were trained and measured on (`defaults.layout`, `layout`). A repository's file is already a place for exactly that kind of fact (its temperatures, commit rule, `think`).
- **With no recorded layout they are served plain**, whatever the base's default is: it is what they were trained on. Every tuned repository and adapter in existence is in this case, so none changes.
- A recorded `"header"` serves the header. Lab 1 and Lab 2 record it for an adapter only when it was measured so (by training in it, or by an A/B on that adapter).
- `--layout` given explicitly overrides the record (a measurement run needs to).
- With adapters named per request, an adapter without a record and a base with the header cannot share a server's one layout: the held layout of decision 3 covers it (an adapter without a record loaded keeps the server on `plain`).

### 5. Versioning and the release notes

This changes the answers of every bare base on upgrade. It is breaking for answers, not for the API (no route, field or schema changes).
- **0.26.0, not 1.0.** decisio is below 1.0 and the project's rule is that minor releases may change behaviour (`bump-minor-pre-major`); the commit is `feat(serve)!:` with a `BREAKING CHANGE:` paragraph, so the changelog marks it. 1.0 is a statement about stability, not a place to put one change.
- The release notes (and `upgrading.md`, first entry) must say, in this order:
  1. **The default layout of the Gemma bases is now the header; answers change.** With the measured effect (the 31B: +5.0 on a Decision Index sample, +4.4 H776, +7.4 hard390, -0.4 dev500; the 12B: smaller, concentrated in multi-question requests) and its cost (tokens +3 to 10% as counted, latency, the cache).
  2. **Who is not affected:** a server holding tasks, certificates, adapters or tuned repositories fitted on the plain layout keeps serving plain, and the start says so.
  3. **How to keep the old behaviour** (`--layout plain`) and **how to move** (register again under `--layout header`).
  4. **What to look for:** clients that ask a series of questions about one state will see later questions cost like first questions (the table above).
  5. `usage.input_tokens` counts the header.
- A board row measured under the old default is not comparable with one under the new; the card (`EVAL_CARD.md`) states the layout of every figure it prints.

## Build plan (after the choices)

| Step | Working hours |
| --- | ---: |
| The held default: resolve the layout at start from the base, `--layout`, the repository's and adapters' records, the loaded tasks and abstention tasks; the start line and `/health` | 3 |
| `defaults.layout` in the repository file (schema `/2`, validated, an older reader refuses it) and `layout` in the adapter file; `hub tuned-config` writes it | 2 |
| Tests: every combination of the table, the default rows unchanged where held, tasks and certificates, adapters | 3 |
| Docs: `running.md`, `cli.md`, `upgrading.md` (first entry, the five points), `EVAL_CARD.md` layout statements, the README; the cached share in the timing headers | 2.5 |
| **First release, total** | **10.5** |
| Later, if clients ask: the per-request `layout` field (task and certificate keys per layout, registration's layout, abstention tasks) | 8 |

## Not covered here

- Read twice: removed in #290 on the 12B's evidence; Lab 4 proposes testing it on the 31B before writing it off. It would be its own decision.
- A new Decision Index or JevBench row under the header is the owner's call, not a decisio change.
- The temperatures under the header: `--temperatures-file` exists and the refit is the open item (Lab 4 found the natural-split refit harmful on yes/no as fitted); nothing here changes the served temperatures.

## As built (0.26.0)

- `decisio.serve.layout.decide_layout` decides once at start: `--layout`, else the layout a tuned repository (`defaults.layout`) or an adapter (`layout` in its file) recorded (plain when none; things recorded with different layouts cannot share a server), else the base's default (`Family.layout`: header for the Gemma bases), held at plain while stored tasks fitted under plain apply. The stores are read before the task fingerprint is built. The start prints `LAYOUT ...`; `/health` carries `systemone.layout`.
- The hub profile is unchanged (it describes the plain-layout base), so every repository file exported before 0.26.0 still agrees. An older decisio refuses a repository that records `defaults.layout`.
- Abstention tasks carry `layout` when it is not plain and apply only under it. Responses carry `x-decisio-prompt: layout=...; tokens=...; cached=...`.
- Not built: the per-request `layout` field (about 8 working hours, if clients ask).
