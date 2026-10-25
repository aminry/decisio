<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Copyright contributors to the decisio project -->

# Upgrading

What changes for a running deployment from one release to the next, newest first.
The changelog (`CHANGELOG.md`) lists every change; this page is for the ones that change what a deployment serves.

**After any upgrade, certify certified tasks again.** A certificate covers one task on one serving setup, and the setup it is checked against includes the decisio version, the engine and its version, the card and the checkpoint (`docs/tasks.md`, "Certified answers"): after an upgrade the old certificate is suspended (`serving_stack_changed`) until `POST /v1/tasks/{id}/certify` makes it again.

## 0.31.0: a tuned repository can be a flavor

### What changed

- **A tuned repository's `decision_config.json` (`/2`) can carry a `flavor` block and read-twice keys in `defaults`** ([running.md](running.md#shipping-a-tuned-model)): a repository of a tuned model is then one of the flavors `calibrated`, `commit` or `read-enhanced`, named by its `served_name`.
  The server it starts is named by the file, reads twice where the defaults say so, and reports the flavor and any flag that differs from it, as a base's flavors do; `hub tuned-config --flavor ... --served-name ... --read-twice-under N` writes the file and `hub check` refuses one that disagrees with its flavor.
  A repository with no `flavor` block is served exactly as before, and its digest is unchanged.

### Who is affected

| You run | What happens | What to do |
| --- | --- | --- |
| A tuned repository with no `flavor` block | Nothing changes | Nothing |
| A tuned repository with a `flavor` block (new) | Served under its flavor's name; an older decisio serves it as a tuned model with no flavor, and refuses one with read-twice keys | Use this release or later |

## 0.30.0: the 31B in bf16

### What changed

- **`--base gemma-4-31b-bf16`** serves Google's 31B checkpoint as published, in bf16, with no quantization, beside the default 31B (FP8) ([running.md](running.md#the-31b-in-bf16)).
  Until now the 31B could only be served in FP8 (stored, or quantized on load).
  Its profile, temperatures and layout are the 31B's; its key, served name and fingerprints are its own, so tasks and certificates made on the FP8 31B are not applied to it.
  It needs a card with room for 57.91 GiB of weights and a KV cache; the context ceiling is 65,536, not measured at bf16.

### Who is affected

| You run | What happens | What to do |
| --- | --- | --- |
| `--base gemma-4-31b`, or a 31B checkpoint without `--base` | Nothing changes (FP8) | Nothing |
| `--base gemma-4-31b-bf16` (new) | The bf16 31B, vLLM only | Check `/health` for the context it chose; compare before relying on FP8 records |

## 0.29.0: a header that does not outgrow the context

### What changed

- **Under the header layout a request whose rows would be over the context is answered, not refused.**
  The header lists the instructions of every question on every row, so a request with many long instructions could render a row several times longer than the plain layout's and past the context (a 422).
  The server now lists as many of the questions as fit, in the request's order, and none when no line fits (the plain layout's row).
  The response says so (`header_limited`, `x-decisio-layout`, `/health`), `usage.input_tokens` counts the header that was read, and a certified task's answer in such a request is not certified (`header_limited`).
  A request whose rows fit is built as it always was ([running.md](running.md#a-header-over-the-context---header-limit)).
- **`--header-limit auto|off|TOKENS`.** `off` restores the 422; a number also caps the header's tokens.
- **The 422 for a row nothing can bring under the context says where its tokens are** (header, state, longest question, template) and what to do.

### Who is affected

| You run | What happens | What to do |
| --- | --- | --- |
| The header layout (the Gemma default) and requests whose rows fit | Nothing changes | Nothing |
| The header layout and a request that was refused for its length | It is answered, with a shorter header, and the response says so | Compare against `--layout plain` where the answer matters; or `--header-limit off` for the 422 |
| Certified tasks | An answer in a limited request has `certified: false`, reason `header_limited` | Nothing |
| `--layout plain` or the Qwen base | Nothing changes | Nothing |

## 0.28.1: the 12B's calibrated yes/no temperatures per precision, and a warning from the adapter fit tool

### What changed

- **The 12B's calibrated yes/no temperature is now fitted per precision, on items that share no text with a public benchmark's test set:** 4.84 on bf16 (was 4.83) and 4.72 on FP8 (was 4.83); score stays at 7.85 and choice at the base's 3.592 ([running.md](running.md#flavors)).
  The change is inside the fit's noise (bootstrap 95% intervals of about 4.1 to 5.4); no answer changes, only the confidences move, by at most about 0.01 of a probability.
- **`adapter_temperatures fit` and `fit-split` warn when `--layout` is not given.** The file then records no layout, and a server that loads the adapter treats it as plain-trained and holds a header-default base at plain; the warning says so.
  Give `--layout header` (or `plain`) to record the layout the reads were taken under.

### Who is affected

| You run | What happens | What to do |
| --- | --- | --- |
| `--flavor calibrated`, `commit` or `read-enhanced` on the 12B | Yes/no confidences move slightly (FP8 by the larger step) | Certify tasks again under the flavor |
| Anything else | Nothing changes | Nothing |
| A script that calls the fit tool without `--layout` | Same output, and a warning on stderr | Add `--layout header` or `--layout plain` |

## 0.28.0: a default adapter, the commit flavor, and the 12B's calibrated temperatures

### What changed

- **`--default-adapter NAME`** answers a request that names no adapter with a loaded adapter ([running.md](running.md#a-default-adapter)), so a client that sends no `adapter` field (a benchmark kit, a plain caller) gets the tuned model. `"adapter": "none"` still asks for the base, and a request naming another loaded adapter gets that one. The serving-stack fingerprint holds the default only when one is set. A server started without it is what it was.
- **The `commit` flavor replaces `thinking-commit`.** The flavors are `calibrated`, `commit` and `read-enhanced`; `commit` is the header layout with the commit rule on, the calibrated temperatures and no thinking, served as `Decisio-31b-commit`, `Decisio-12b-commit` or `Decisio-12b-fp8-commit`. Thinking is an ordinary option (`--think`), not a flavor.
- **On the 12B (both precisions) `calibrated` and `commit` start,** with yes/no at 4.83 and score at 7.85 (choice stays at the base's 3.592), fitted by minimum log loss on readouts under the header layout and judged on calibration ([running.md](running.md#flavors)); `read-enhanced` uses the same temperatures. On the 31B `calibrated` and `commit` still refuse to start until its temperatures are fitted.

### Who is affected

| You run | What happens | What to do |
| --- | --- | --- |
| Without `--default-adapter` and without `--flavor` | Nothing changes | Nothing |
| `--default-adapter NAME` (new) | Requests with no `adapter` field are answered by that adapter; certificates made under another default are suspended; a plain request matching a registered task is refused (tasks read the base) | Ask a task's questions with `"adapter": "none"`; certify again under the new setup |
| `--flavor thinking-commit` | Refused: it is not a flavor any more | Use `--flavor commit`, and `--think` if you want thinking |
| `--flavor calibrated` or `commit` on the 12B | Served at the calibrated temperatures | Certify tasks again under the flavor |
| `--flavor calibrated` or `commit` on the 31B | Refused until its temperatures are fitted | Nothing yet |

## 0.27.0: read twice, think-mode temperatures, per-type profile temperatures and flavors (all opt-in), and a truncation fix

### What changed

Read twice:

- **`--read-twice-under N`** writes a request that read once is at most N counted input tokens (the header, the state and each question's text) twice, the second copy after `Read the same input again before answering:`; **`--read-twice-max-questions R`** reads a request with more than R questions once, whatever its size ([running.md](running.md#read-twice---read-twice-under-n---read-twice-max-questions-r)). Both are **off by default and read twice is never the default**: with them off every prompt, fingerprint and task key is what it was.
  On the 31B with the header: N = 2,000 is the measured choice on our own proxies (+3.2 pooled, the smallest within 0.25 points of the best) and +1.8 on a public multi-domain sample with p80 latency about 866 ms; a threshold of 3,000 counted on rows gave +1.9 with p80 at or over a 1,000 ms target. It roughly doubles the input tokens of the requests it applies to (so it is off by default), and `usage.input_tokens` counts both copies. On the 12B the same step added nothing within noise.
- Tasks registered, and certificates made, under one setting are not applied under another.

Think-mode temperatures:

- **A separate temperature for the answers that thought.** A tuned repository's `defaults.think_temperatures`, an adapter file's `think_temperatures`, and `--think-temperature-choice|noul|score` give a temperature per question type for the mix of the plain read and the read after the reasoning ([running.md](running.md#thinking-when-unsure)).
  The single-pass temperatures and bands are unchanged, and so is which questions think: the unsure test still reads the single-pass temperature.
  A type with no think temperature keeps its single-pass temperature, so a repository or a server that does not set them answers exactly as before.
- **`python -m decisio.serve.adapter_temperatures fit-think`** adds them to an adapter file from labelled mixes of questions that thought.
  `python -m decisio.hub tuned-config` carries them into a repository's `defaults` and the think fit's method (never its data).
- **The fingerprint** of a server that thinks holds its think temperatures when it has any (`serving.think.temperatures`); a task certified under other ones is `think_mode_changed`.
  A server that does not think, or has none, keeps the fingerprint it had.

Flavors:

- **`--flavor calibrated|thinking-commit|read-enhanced`** serves a named configuration of a Gemma base on the header layout, named `Decisio-31b-<flavor>`, `Decisio-12b-<flavor>` or `Decisio-12b-fp8-<flavor>` ([running.md](running.md#flavors)). A flag you give still wins and is recorded as an override. `calibrated` refuses to start until its temperatures are fitted; the other two serve with their pending values marked. **A server with no `--flavor` is what it was.**

A base's profile can carry per-type temperatures and bands:

- **A base's profile can hold a temperature for yes/no and for score, and choice temperatures by band of option count**, beside the global and choice ones ([running.md](running.md#refitting-the-bases-temperatures-for-a-layout)). It is the last word of the precedence (a flag, `--temperatures-file`, a tuned repository's defaults, then the profile). **No base sets them in this release**, so every served temperature is what it was; the exported hub profile is unchanged, so repository files exported before this still agree.

Fix:

- **`--over-limit truncate` works under the header layout.** In 0.26.0 the room the cut left for the state did not count the header's list of the request's questions, so a truncated request whose list was longer than the 32-token margin came out over the ceiling and the engine refused it with a 422 (27 of 60 items at a 85,000-token target on a 65,536 ceiling, 12B). The cut now counts the list: the state keeps exactly what the rows leave, and every such request is answered. Nothing changes on the plain layout, and nothing for a server that refuses over-limit states (the default).

### Who is affected

| You run | What happens | What to do |
| --- | --- | --- |
| Without `--read-twice-under` | Nothing changes | Nothing |
| A Gemma base with `--over-limit truncate` (or a request's `over_limit: truncate`) on the header layout | Requests whose question list was longer than 32 tokens got a 422 in 0.26.0 and are answered now, from a state cut by the list's length | Nothing |
| `--read-twice-under N` (new) | Short requests are read twice; tasks and certificates of other settings are not applied | Choose N and R from a run of your own; register and certify again under them |
| Any server that does not use `--think` or a repository with `defaults.think` | Nothing changes | Nothing |
| A thinking server and a repository or file without think temperatures | Nothing changes | Nothing |
| A repository with `defaults.think_temperatures` | Needs this release or later; an older decisio refuses the file (an unknown key of `defaults`) | Run this release or later |
| A server given think temperatures | Thinking answers are served at them; certified tasks made without them are suspended (`think_mode_changed`) | Certify the tasks again |

## 0.26.0: the Gemma bases are served with the header layout (answers change)

### What changed

- **The Gemma bases (the 31B, the 12B and the 12B in FP8) now put the request's questions before the state** (`Questions that will be asked about the context:`, a `- <question>` line each), then the state and the question as before. **Answers change.** Measured on the 31B against the plain layout: +5.0 points on a Decision Index sample, +4.4 on H776 and +7.4 on hard390 (95% intervals [2.9, 6.9], [2.3, 6.4], [3.1, 11.8]), -0.4 on dev500 (within noise); on the 12B the gain was smaller (+3.8 on the Decision Index sample, +3.6 on dev500, +0.9 on H776, which is within noise) and concentrated in requests with many questions about one state ([running.md](running.md#the-header-layout---layout)). The Qwen base is unchanged.
- **Who is not affected: a server that loads something fitted or trained on the plain layout keeps serving plain, and says so.** Stored tasks (a directory, a shared store or a file) that apply to the server, abstention tasks written before this release, and adapters or tuned repositories with no recorded layout hold the layout at plain; the start prints `LAYOUT plain (held at plain; this base's default is header): ...` and `/health` carries `systemone.layout`. Nothing that exists changes on upgrade.
- **How to keep the old behaviour everywhere: `--layout plain`.** How to move a taught deployment to the header: register its tasks again under `--layout header` (a task is kept under the layout it was fitted under; the records of the old ones are not applied under the new layout) and measure.
- **The cost, for clients that ask a series of questions about one state** (agents, games, teach-it loops): a state cached for one set of questions is not reused for another, so each later question costs like a first question (the 31B, 3,000 tokens: 485 ms against 41; 16,000 tokens: 3,370 ms against 94). They want `--layout plain`. Every response carries `x-decisio-prompt: layout=...; tokens=...; cached=...` to see it.
- **Tuned repositories and adapters can record their layout:** `defaults.layout` in a repository's `decision_config.json` (`hub tuned-config --layout`; an older decisio refuses a repository that has the key), `layout` in an adapter's `decisio_adapter.json` (`adapter_temperatures fit --layout`). A model is served on the layout it records, plain when it records none; the fine-tuned adapters have not been measured under the header.
- `usage.input_tokens` counts the header (3 to 10% more on single-question items).
- Tasks registered, and certificates made, under one layout are not applied under another (the same rule as `--prompt-tail`).

### Who is affected

| You run | What happens | What to do |
| --- | --- | --- |
| A Gemma base with nothing stored and no adapter (a new deployment, a board run) | Answers change: the header layout | Nothing, or `--layout plain` to keep the old answers |
| A Gemma base with stored tasks, abstention tasks, or an adapter or tuned repository without a recorded layout | Nothing changes: the server holds `plain` and says why | Register again under `--layout header` when you want to move |
| Clients that ask a series of different questions about one state | Later questions cost like first ones under the header | `--layout plain` |
| The Qwen base | Nothing changes | Nothing |
| A tuned repository that records `defaults.layout` | An older decisio refuses it | Run this release or later |

## 0.25.0: taught tasks kept across restarts, and shared between replicas

### What changed

- **`--tasks-dir PATH`** keeps every taught task (calibration and head tasks, and abstention tasks) in a directory, written through atomically before the response and loaded at start ([tasks.md](tasks.md#saved-to-disk---tasks-dir)).
  A write that fails fails the request (500) and nothing is registered.
  **`--require-tasks N`** refuses to start with fewer than N usable tasks, so a lost store cannot silently become untaught answers.
- **Abstention tasks** can be imported (`POST /v1/abstention/tasks/import`) and removed (`DELETE /v1/abstention/tasks/{id}`), and `GET /v1/abstention/tasks` is their export.
  An abstention task now applies only on the served name it was fitted for; a task fitted under another served name is kept and not applied.
- `GET /v1/tasks`, `GET /v1/abstention/tasks` and `/health` say where tasks are saved and how many are loaded and apply.
- **`--task-store URL`** shares taught tasks between replicas through object storage (S3 and S3-compatible services, `gs://`, `az://`; `decisio[task-store]`, in the image): a task taught on one replica reaches the others within `--task-store-poll-seconds` (5) as the exact fitted record. A write names the version it replaces and answers 409 if the store holds another (`?replace=1` overrides); `--task-store-readonly` makes a replica follow without teaching; an unreachable store stops teaching (503), not serving ([tasks.md](tasks.md#replicas-a-shared-store---task-store)).

### Who is affected

| You run | What happens | What to do |
| --- | --- | --- |
| No `--tasks-dir` | Nothing changes, except that an abstention task fitted under another served name is no longer applied | Nothing; re-register such a task under the served name |
| `--tasks-dir` (new) | Tasks survive a restart; one directory belongs to one server | Put it on a volume that outlives the container; add `--require-tasks N` |
| Several replicas | Each is its own server; `--tasks-dir` does not share tasks between them | Give each its own directory, or start them all with `--task-store` |
| `--task-store` (new) | Tasks are shared; a second registration of one task is a 409, and a certificate is suspended on a replica of another stack | Run one stack if you certify; mark all but one replica `--task-store-readonly` if fitting should stay on one |

## 0.24.0: the 12B in FP8, batching that batches, thinking that reasons

### What changed

- **The Gemma 4 12B in FP8, as a second precision** (`--base gemma-4-12b-fp8`, [running.md](running.md#the-12b-in-two-precisions)).
  The bf16 12B stays the 12B's default and the base every 12B record was measured on.
  FP8 is its own function (on 200 requests the top answer was the same as bf16's on 96%, the largest difference 0.438), with its own served name, task fingerprint and certificates, and it takes the full 65,536-token context on a 32 GB card where the bf16 12B starts only at 4,096.
  A stored FP8 checkpoint is loaded as stored on either Gemma base; `--base gemma-4-12b` with an FP8 checkpoint is refused, naming the FP8 base.
- **`--concurrency batched` no longer stalls.** In 0.23.0 it was slower than serial on requests about new states, because vLLM's tokenizer pool deep-copied the whole tokenizer for each thread tokenizing at once.
  The engine's tokenizer is now used one call at a time.
  Measured on one RTX 5090 with the FP8 12B and new states of about 1,000 tokens: 8.9 and 9.2 requests a second at 8 and 16 clients against 7.2 serial.
  Other bases, client counts and traffic were not re-measured; measure on your own.
  Serial servers are unaffected.
- **Thinking now reasons on a default server.** Think mode (0.21.0 to 0.23.x) did not reason on a default server: decisio's hidden-readout class writes the hidden state into reserved logit columns on every row, a read allows only its labels and never saw them, but the greedy writing of reasoning may sample any token, took those columns, and wrote reserved token ids for the whole budget.
  Every thinking question ran to `--think-budget` tokens and the read after it followed junk; answers that thought in those releases are not what thinking is meant to give.
  The writing call now runs with those columns switched off.
  Measured on one RTX 5090 (vLLM 0.31.0, decisio 0.24.0, the 12B in FP8, 20 questions one at a time, 2026-10-21): real reasoning on all 20, 15 stopped on their own at the end of the turn (median 402 tokens), 5 hard items ran to the 512-token budget, no request failed.
  Ten ids of the readout range appeared in 6,348 written tokens; each is an ordinary word in coherent text (the range holds real vocabulary tokens as well as the hidden-state columns), not a leak (`runs/2026-10-21_5090-think24/`).
  That shows the mechanism works on one card and base, not how much thinking helps.
  A question whose reasoning comes back unusable (mostly reserved ids) is answered at its single pass, marked `thinking: unavailable` with the reason in the response, logged, and counted in `/health` under `systemone.think.unavailable`; the request does not fail.
  Thinking with decisio's hidden-readout class needs the engine in the server's process (`--engine-process in`, the default); with `--engine-process separate`, `--think` is refused at start with the reason (serve a text-only class with a second engine for the head to think there).

### Who is affected

| You run | What happens | What to do |
| --- | --- | --- |
| `--base gemma-4-12b-fp8` (new) | A second 12B, in FP8; its answers differ slightly from bf16's | Choose it for memory and speed; refit temperatures on FP8 reads before relying on them |
| `--concurrency batched` | Throughput on new states is now above serial where measured; answers are unchanged | Measure on your traffic |
| A server with `--think`, a request's `"think": true`, or a repository whose `defaults.think` is on | Thinking answers change: the reasoning is text now, usually shorter than the budget, so they are also faster | Check your factor against measured answers; certify tasks again if you certify under thinking |
| `--think` with `--engine-process separate` | Refused at start, with the reason | Use the default engine process, or a text-only class with a second engine |
| Everything else | Nothing changes | Nothing |

## 0.23.0: batching concurrent requests (opt-in)

### What changed

- **`--concurrency batched`** scores the requests waiting for the engine together, in one engine call, up to `--batch-max-requests` (16) ([running.md](running.md#concurrency)).
  The default stays `--concurrency serial`, one request at a time, unchanged.
  On a batched server a certified task's questions and every task registration and certification read still take the serial path; the other answers can move slightly with the traffic they arrived with.
- **New fields:** `concurrency` in `/health` (`{"mode": "serial"}` on a serial server; the calls and requests per call on a batched one), the header `x-decisio-batch` and `timing.batch` on `/v1/answer` when a request shared a call, and `shared_skipped` in a task monitor's facts when the monitor skipped answers scored in a shared call.
- **A new certificate reason, `concurrency_changed`:** on a batched server the mode is part of the serving-stack fingerprint, so a task certified on a serial server is not served as certified by a batched one, and the other way round.
  The response schema (`docs/reference/openapi.json`) now lists every `certified_reason` the server gives; `state_truncated` and `think_mode_changed` were missing from it.
- **The engine's lock is first come, first served** on the vLLM engine and the CPU stand-in, on every server: under load a request waits for the requests that arrived before it, in order.

### Who is affected

| You run | What happens | What to do |
| --- | --- | --- |
| A serial server (the default) | Nothing changes: the same answers and serving-stack fingerprint; `/health` has one more field | Nothing |
| `--concurrency batched` | Uncertified answers can depend on the traffic; certified tasks made on a serial server are suspended (`concurrency_changed`) | Certify the tasks again on the batched server, or keep serial |
| A client that reads `certified_reason` | A new value, `concurrency_changed`, only from a batched server | Treat it as the other suspension reasons |

## 0.22.0: a tuned 12B repository

### What changed

- **The Gemma 4 12B can be shipped as a tuned repository** (`decisio.decision_config/2`), stored in bf16 as the 12B is served ([running.md](running.md#shipping-a-tuned-model)).
  `python -m decisio.hub_bf16 merge` adds the adapter in fp32 and rounds once to bf16, stochastically and seeded (the seed is in the report and the repository's file); every other tensor is the base's, byte for byte. `python -m decisio.hub tuned-config --base gemma-4-12b` writes the file.
  A tuned 12B is served at its repository's defaults exactly as a tuned 31B is: temperatures and choice bands, the commit setting, `over_limit`, and `think`.
- **A tuned repository's file no longer says anything about the data.** `tuned-config` keeps the method of the temperature fit and drops the fit tools' notes on data, counts and details; `defaults.fit.data` is no longer required (files that have it still load).

### Who is affected

| You run | What happens | What to do |
| --- | --- | --- |
| The bases and `/1` repositories | Nothing changes | Nothing |
| A tuned 31B repository | Nothing changes; a new `tuned-config` writes a `fit` without `data` | Nothing |
| A tuned 12B repository | Needs 0.22.0 or later; an older decisio refuses it by its schema | Run 0.22.0 or later |

## 0.21.0: think when unsure

### What changed

- **A server can read an unsure question again after the served model has written its own reasoning** ([running.md](running.md#thinking-when-unsure)).
  It is off unless asked: `--think`, a request's `"think": true`, or a tuned repository's `defaults.think` (its factor and budget).
  A question below its type's threshold times `--think-factor` (choice 0.75, yes/no 0.92, score 0.70) with at most 16 options is sent again with a fixed prefix, the same model writes up to `--think-budget` (512) tokens, a fixed suffix follows and the options are read again; the two reads are mixed in log space and go through the temperature a single read gets.
- **New fields when it is on:** `thinking` in the response, `usage.thinking_tokens` (and `usage.output_tokens` counts the reasoning tokens), `x-decisio-think-ms` and `x-decisio-think-tokens`, and `systemone.think` in `/health`.
- **A new certificate reason, `think_mode_changed`:** the server's think setting is part of the serving-stack fingerprint when it is on, so a task certified one-pass is not served as certified by a server that thinks, and the other way round.
  Questions of a registered task are always answered at one read.
- **One more kind of vLLM call:** thinking is the only call that generates text; its length is a server setting, never a request field ([SECURITY.md](../SECURITY.md)).

### Who is affected

| You run | What happens | What to do |
| --- | --- | --- |
| A server without `--think` and a request without `think` | Nothing changes | Nothing |
| A tuned repository whose `defaults` carry `think` | Needs 0.21.0 or later; 0.20.0 refuses the file | Run 0.21.0 or later |
| Certified tasks on a server you start with `--think` | Suspended (`think_mode_changed`) | Run without `--think`, or certify the task again |
| The MLX build | `--think` is refused at start | Nothing |

## 0.20.0: tuned repositories, and states over the prompt limit

### What changed

- **A tuned model has its own repository format, `decisio.decision_config/2`.**
  It keeps the base's profile exactly and adds `tuned` (the base, the adapter with its digests, the merge report, every weight file by sha256) and `defaults`: the temperatures per question type, the choice bands, the commit setting, and optionally `over_limit`.
  A plain request to such a repository gets those defaults with no flag and no adapter field; a flag beats the repository, and `--temperature-choice` drops its bands.
  An older decisio reads `/1` only and refuses a `/2` repository by its schema, instead of serving the model at the base's temperatures.
  `python -m decisio.hub_fp8 merge` writes the merged FP8 weights (an adapter added in fp32 and quantized once), and `python -m decisio.hub tuned-config` and `check` write and verify the file ([running.md](running.md#shipping-a-tuned-model)).
- **A state over the engine's prompt ceiling can be answered from its start.**
  `--over-limit truncate`, a request's `over_limit` field, or a tuned repository's `defaults.over_limit` cuts the state (never a question) once per request; the response carries `state_truncated` and the header `x-decisio-state-truncated`, and no answer of a truncated state is certified (`certified_reason: "state_truncated"`).
  The default is `refuse`, as before ([running.md](running.md#states-over-the-limit)).
- **New flags:** `--over-limit`, `--over-limit-tokens`, `--noul-commit` / `--no-noul-commit` (the pair; a tuned repository states its own commit setting).

### Who is affected

| You run | What happens | What to do |
| --- | --- | --- |
| The default base or a `/1` repository, no new flag | Nothing changes: the same temperatures, the same refusals | Nothing |
| A script that reads `/health` | It has new fields (`systemone.over_limit`, and for a tuned repository `systemone.repository_defaults` and `choice_bands`) | Nothing, unless it rejects unknown fields |
| A tuned (`/2`) repository | Served at its defaults; an older decisio refuses it | Run 0.20.0 or later |
| Certified tasks | Suspended by the new version as after any upgrade | Certify them again |

## 0.19.0: decisio's name only on the wire, in new records and in the served name

### What changed

- **Only `x-decisio-debug` and `x-decisio-route` are read as request headers.**
  The earlier spellings of those two headers, with the project's former name in place of `decisio`, are ignored, not refused: a request that still carries one is served as if it did not, so a client that sent the debug header gets no `decisio_debug` field and one that sent the route header gets the route the server picks.
  Responses never carried the earlier spellings.
- **New task records and the Qwen base's served name use decisio's name.**
  A task is written as `decisio-task/1`, with `decisio-task-prior/1`, `decisio-intent-head/1`, `decisio-certificate/1` and `decisio-abstention/1` inside it, and the Qwen base's served name is `decisio-qwen3.6-35b-a3b-letters`; the server writes no record, header or served name that carries the former name.
- **Records and requests that use the earlier names are still accepted.**
  A task record in the earlier format ids loads and applies as written (it is not rewritten), a task fitted under the Qwen base's earlier default served name applies on a server with the current one, and a request's `model` is not checked against the served name, so the earlier served name is accepted and echoed like any other.

### Who is affected

| You run | What happens | What to do |
| --- | --- | --- |
| A client or script that sends the earlier debug or route header | The header is ignored: no debug readout, no route choice | Send `x-decisio-debug` / `x-decisio-route` |
| Stored tasks in the earlier record formats | They load and apply as before | Nothing |
| A client that names the Qwen base's earlier served name in `model` | It is accepted; the response echoes it | Nothing; use the name `GET /v1/models` lists |
| Anything else | Nothing changes | Nothing |

## 0.18.0: adapter temperatures, more than two images, shares that fit one card

### What changed

- **An adapter can carry its own temperatures.**
  A LoRA adapter changes the scale of the readout, and the base's temperatures were fitted on the base.
  `decisio_adapter.json` beside the adapter's weights, or `--adapter-temperatures name=file`, gives the adapter a temperature per question type (`choice`, `noul`, `score`) and, for choice, optionally one per band of option count (`choice_bands`: 2-10, 11-26, 27-255 or your own).
  A request that names the adapter is served at them.
  A request that does not, and an adapter with no file, are served at the base's temperatures as before.
  `python -m decisio.serve.adapter_temperatures fit` and `fit-split` write the file ([running.md](running.md#an-adapters-own-temperatures)).
- **An adapter's identity and temperatures are in the serving-stack fingerprint.**
  `/health` shows them in `adapter_identity` and `adapter_temperatures`, so a different adapter, a retrained one or a changed temperature is a different stack.
  A server with no adapter has none of these keys.
- **The image count of a request is limited by the context, not by a cap of 2.**
  `--max-images` (default 128) is the most images one request may carry on a server started with `--image-model`.
  Lowering it saves multimodal profiling memory at start-up.
- **`--image-model` starts with shares that fit one card.**
  The text engine defaults to 0.47 of the card and the image engine to 0.49, the pair that was measured.
  Before, the defaults (0.90 and 0.50) could not fit together and the start failed in the image engine after the text engine had loaded.
  A pair with a default in it that totals over 0.96, or any pair over 1.0, is now refused at start, before either engine is built.
  A pair you give in full is yours up to 1.0.
  The start prints `ENGINE SHARES`.
- **A Marlin W4A16 checkpoint works with the default in-process engine.**
  The 0.16.0 limit that it needed `--engine-process separate` is gone: every thread that calls the engine now runs on the stream the engine was warmed up on.
- **The OpenAPI file and the flags table are generated from the code.**
  `docs/reference/openapi.json` and `docs/reference/flags.md` are rewritten at release time, and a test fails if they disagree with the code.

### Who is affected

| You run | What happens | What to do |
| --- | --- | --- |
| `--adapter` without a temperatures file | Nothing changes: the base's temperatures | Fit and ship a file if the adapter's answers are over- or under-confident |
| `--image-model` with no memory flag | It starts now; before, the image engine failed to load | Nothing |
| `--image-model` with only one of the two memory flags | The other takes the measured pair's value (0.47 text, 0.49 image), where before it took 0.90 or 0.50 | Give both, or check that the pair totals 0.96 or less |
| A client that sends more than 2 images in a request | Accepted up to `--max-images` and the context | Nothing |
| A Marlin W4A16 checkpoint with `--engine-process separate` | Still works | You may drop the flag |

## 0.17.0: System One's `assets` are read, and too many options is a 422

### What changed

- **A choice with more than 255 options now returns 422 instead of 400.** It is System One's rejection, a `too_long` validation error at `body.questions.<name>.choice.criteria` (`Dictionary should have at most 255 items after validation, not 256`); before, the engine refused it with a 400 and a plain `letter readout supports 1 to 255 options` message. The limit itself, 255, is unchanged.
- **System One's `assets` are now read.** A request that carried images as `assets` (placeholder names mapped to images, the placeholders written in the state or a question's text) was accepted before and answered **without its image**: the placeholder stayed text and nothing said so. Now every placeholder becomes the marker `[image N]` and its image goes to the image engine, in JSON (`data_base64`) and in multipart (`part`); a problem with an asset is a 400 in System One's `api_usage_error` shape. Details and the remaining differences from System One: [`api.md`](api.md#where-this-route-differs-from-system-one).
- **A conformance gate, C5,** compares the request and response fields with System One's published OpenAPI document (`python -m decisio.serve.systemone_conformance c5 --snapshot benchmarks/system_one_wire_0.2.0.json`).

### Who is affected

| You run | What happens | What to do |
| --- | --- | --- |
| A client that checks for status 400 on a choice with more than 255 options | It now gets 422 | Accept 422 (System One's own code) |
| A client that sent `assets` to decisio | Its answers before 0.17.0 were made without the images | Send them again on 0.17.0, on a server started with `--image-model` (the Qwen base); a text-only server answers with "no image engine" |
| A client that sends no `assets` and stays under 255 options | Nothing changes | Nothing |

## 0.16.0: certified tasks, drift protection and the unknown-model refusal

### What changed

- **A certified task is watched for drift by its confidences, and stops certifying while the alert is on.** The traffic monitor gained a second signal beside the answer mix: a CUSUM on how often an answer's top probability falls below the certification examples' 5th percentile. When it alerts, every answer of the task is `certified: false` with the reason `traffic_changed` until it clears (`"suspend_on_alert": false` in the certify block only reports). Measured on the stored FP8 31B on one 96 GB card, replaying 5,363 CLINC150 messages (a task with 10 examples per intent and no out-of-scope option, out-of-scope messages rising from message 1,000 to 30%): under 0.14.0 the answer mix first alerted after message 5,200, with 1,027 wrong certified answers served before it and certification left on; under 0.16.0 the first answer was withheld at message 1,570, with 40 wrong certified answers served between message 1,000 and it, and certification stopped. Registration also warns (`certificate.warnings`) when a choice task has no catch-all option. Details: [`tasks.md`](tasks.md#the-monitor).
- **A task certified before 0.16.0 has no confidence statistic, so it gets no drift protection until it is certified again.** Its monitor baseline was stored without one, and the statistic is built when the certificate is made. It keeps working as before (the answer mix alone, which reports and never suspends). **To get the protection:** certify it again on the running 0.16.0 server with `POST /v1/tasks/{id}/certify`, body `{"examples": [{"request": ..., "answer": ...}, ...]}` (labelled like the task's own; optionally `alpha`, `delta`, `min_examples`, `suspend_on_alert`), on examples that share no input with the task's fit, such as the held-out examples it was certified on. The fit is kept; the call returns the new certificate, and `suspend_on_alert` there is true unless you name it false, whatever the old certificate said. Check it took: `GET /v1/tasks/{id}` shows `status.monitor.confidence` (the statistic and its limit) and the certificate's `warnings`. Importing a task exported by an older version keeps its old certificate until you do this.
- **A certified task's questions are scored by the function that certified them.** A question a certified task matches is scored alone, in request order, by the function the certificate was made with, not batched with the request's other questions, and a task with an intent head is certified on the hidden-state read alone (registration used to read each example plain first, which on vLLM moved the confidences slightly: one measurement certified 153 answers where serving put 154 above the threshold). Registration records `served_check` (a sample of the held-out examples re-read through the serving path). A certificate made before 0.16.0 keeps its gap until it is made again.
- **A `--model` of a type no base serves is refused** instead of being given the Qwen profile; `--base <key>` with the same `--model` is the explicit way to serve it (the CPU stand-in `--backend hf` is unchanged).
- **The MLX backend downloads `--model` at `--revision`**, so `--model tachara-ai/decisio-gemma-4-31b-mlx-4bit --revision v0.15.0` pins the build.
- **Known limit: a Marlin W4A16 checkpoint (compressed-tensors, such as `google/gemma-4-31B-it-qat-w4a16-ct`) needs `--engine-process separate` on 0.16.0**, because the default in-process engine refuses its first request with a `marlin_locks` error; the in-process fix (#215) shipped in 0.18.0, and `--engine-process separate` is no longer needed.

### Who is affected

| You run | What happens | What to do |
| --- | --- | --- |
| A certified task made before 0.16.0 | Works as before; no confidence alert, so no suspension on drift | Certify it again on 0.16.0 (above) |
| A certified task made on 0.16.0 | The confidence alert suspends certification while it is on | Nothing; `"suspend_on_alert": false` to only report |
| A certified task with an intent head | Its next certificate is made on the hidden-state read alone | Nothing |
| `--model` of a checkpoint no base serves, on vLLM or MLX | Refused at start with the supported bases named | Add `--base <key>` if it is that base's profile |
| An MLX server with `--revision` | The conversion is downloaded at that revision (before, it was the repository's main) | Nothing |
| A certified task, exported and imported | The export carries the new baseline and `suspend_on_alert` | Nothing |

## 0.14.0: the default context follows the card

### What changed

- A vLLM server started without `--max-model-len` on a Gemma base now serves **the largest context that fits the card**, up to the base's ceiling (31B 98,304; 12B 65,536) and not below 32,768, instead of a fixed 32,768. The Qwen base stays at 32,768.
  The server asks vLLM for the longest single request the card's KV cache holds and serves the largest multiple of 8,192 of that; `/health` reports the length (`max_model_len`) and why (`max_model_len_source`, for example "auto: fits 81,920 of 98,304 on this card").
- Why: a fixed longer context can fail to start on a smaller card, and a failed start loses the whole run, while a fixed 32,768 leaves out the prompts of 50,000 to 98,000 tokens that a bigger card holds.
  The default now never fails where 32,768 would have started, and never goes above the ceiling.
  How long that is depends on the card and on `--gpu-memory-utilization`, strongly and not linearly: on an H100 PCIe 80 GB the 31B served 98,304 at the default share, 49,152 at 0.85, and did not start at 0.70 (`running.md` has the table); a card where 32,768 does not fit fails to start as it did before.
- Details: [`running.md`](running.md#the-context-length).

### Who is affected

| You run | What happens | What to do |
| --- | --- | --- |
| The 31B or the 12B on vLLM with no `--max-model-len` | A prompt between 32,768 tokens and the chosen length is answered instead of refused with 422; it takes longer the longer it is (the first question about a new 32,000-token state took 8.9 s on the 31B at 600 W) | Nothing, or give `--max-model-len 32768` to keep the old limit |
| The same with `--max-model-len N`, or a `max_model_len` in `--engine` | The same length as before | Nothing |
| The Qwen base, `--image-model`, `--head-engine`, `--engine-process separate`, `--backend mlx` or `hf` | 32,768, as before | Nothing |
| Certified answers | The context length served is part of the serving-stack fingerprint: a certificate made at one length is suspended at another, so one made on 0.13.0 (32,768) is suspended on 0.14.0 on the 31B or the 12B, as it already is by the new version, and a card that fits a different length suspends one made on another card | Register and certify again on the server that will serve; give `--max-model-len` to pin the length across cards |
| A client that relied on 422 for prompts over 32,768 tokens | Those prompts are answered where the card allows it | Check the length on your side, or pin `--max-model-len 32768` |

## 0.12.0: a POST that is not JSON is refused, and vLLM is 0.31.0

### What changed

- **Every POST must be JSON.** A POST whose `Content-Type` is not `application/json` (parameters such as `; charset=utf-8` are fine), or has none, is refused with 415 and `accept-post: application/json` before its body is read, on every route. Before 0.12.0 such a POST was carried out: a web page could make the server do the work (and register or replace tasks) without being able to read the answer. The one other form is `multipart/form-data` on `/v1/systemone` (the image clients), which is accepted when the request carries an `X-Decisio-Client` header with any value. `SECURITY.md`, "Cross-site requests from a web page", says why.
- **A non-finite probability is a 500, not a 400.** If an engine returns NaN or infinity, the model's arithmetic on that device is at fault, not the request: the route answers 500 with a `detail` that names the model and the engine.
- **vLLM is 0.31.0** (the lock file and the image pin it). Rebuild the environment with `uv sync` or pull the new image.
- **New and optional:** certified answers and a traffic monitor in `/v1/tasks` (`docs/tasks.md`), an abstention task that matches by option keys, and `/health`'s `state_boundary` reporting `running` while a registration is being sent.

### Who is affected

| You run | What happens | What to do |
| --- | --- | --- |
| A client that posts form-encoded or `text/plain` bodies, or sends no `Content-Type` | It gets 415 | Send JSON with `Content-Type: application/json` |
| A client that sends `multipart/form-data` to `/v1/systemone` | It gets 415 without the header | Add `X-Decisio-Client: <any value>` |
| A client that treats every 4xx as the request's fault | A model whose arithmetic fails now answers 500 | Treat a 500 with "not finite" as a server fault (`docs/api.md`) |
| An environment built for 0.11.x | It has vLLM 0.30.0 | Rebuild it; measurements made on 0.30.0 are labelled as such in `EVAL_CARD.md` |

## 0.11.0: the model repositories moved to tachara-ai

### What changed

- The three model repositories are now under the Hugging Face organisation `tachara-ai` (Tachara AI Lab), with the same names: `tachara-ai/decisio-gemma-4-31b`, `tachara-ai/decisio-gemma-4-12b` and `tachara-ai/decisio-qwen3.6-35b-a3b`.
  The code, the weights and the settings are the same as in 0.10.0.
- Each repository has a new revision (a new card, and a `decision_config.json` that names 0.11.0), tagged `v0.11.0`; 0.11.0 pins these.
  The 0.10.0 revisions are still in each repository, tagged `v0.10.0`, and 0.10.0 still serves them.

### The old names keep working

- Hugging Face redirects `aminry/decisio-gemma-4-31b`, `aminry/decisio-gemma-4-12b` and `aminry/decisio-qwen3.6-35b-a3b` to the new names.
  A fresh install of 0.10.0 opens all three through the old names at its pinned revisions, with no token (checked on 2026-10-08, after the move).
- 0.11.0 treats an old name as the new one: `--base aminry/decisio-gemma-4-31b` serves `tachara-ai/decisio-gemma-4-31b` at 0.11.0's pin, and `/health` reports the new name.
  `--base aminry/decisio-gemma-4-31b@<commit>` pins that commit as before.

### Who is affected

| You run | What happens | What to do |
| --- | --- | --- |
| A command, script or image that names `aminry/decisio-...` | It works through the redirect, on 0.10.0 and on 0.11.0 | Change the name when convenient |
| A deployment with the models already in its Hugging Face cache | It downloads the new revision of a repository it serves by default (about 31 GB for the 31B) | Pin `@<commit>` to keep the 0.10.0 revision |
| `--base gemma-4-12b` or `--base qwen3.6-35b-a3b` | Nothing: those serve the source checkpoints, as before | Nothing |

## 0.10.0: the default base is Gemma 4 31B

### What changed

- A vLLM server started with neither `--base` nor `--model` serves `gemma-4-31b`.
  Before, it refused to start.
- The container, `docker compose` and the README's Docker path served the Qwen base before, because the image set `DECISIO_MODEL=Qwen/Qwen3.6-35B-A3B-FP8`.
  The image no longer sets it: the entrypoint starts the server with `--base "${DECISIO_BASE:-gemma-4-31b}"`.
- Gemma 4 31B is served from its FP8 repository, `aminry/decisio-gemma-4-31b` (moved to `tachara-ai/decisio-gemma-4-31b` in 0.11.0, above), so the first start downloads about 31 GB and the server does not quantize Google's weights at every start.
  `--model google/gemma-4-31B-it` serves Google's weights at the pinned revision instead, quantized to FP8 each time they load, as before.
- Nothing else changed for the Qwen and Gemma 4 12B bases: their checkpoints, settings, served names and answers are the same as in 0.9.0.
  They are still chosen with `--base qwen3.6-35b-a3b` and `--base gemma-4-12b`.

### Who is affected

| You run | What happens | What to do |
| --- | --- | --- |
| `docker compose up`, or the image, with no `DECISIO_MODEL` set | The container now serves Gemma 4 31B, which is measured on a 96 GB card (an 80 GB H100 also runs it, `docs/running.md`) | To keep the Qwen base, set `DECISIO_BASE=qwen3.6-35b-a3b` (in `compose.yaml`'s `environment`, or `-e` on `docker run`) |
| The image with `DECISIO_MODEL` set | Nothing: a checkpoint alone names its own base, so the container serves what it served | Nothing |
| A derived image that relied on `ENV DECISIO_MODEL` coming from this image | It no longer comes from it, so the default base is served | Set `DECISIO_MODEL` or `DECISIO_BASE` in your image |
| `uv run python -m decisio.serve.vllm_engine --base ...` or `--model ...` | Nothing | Nothing |
| A Mac with `--backend mlx`, or Ollama | Nothing: they always named a base or a conversion (`--backend mlx` needs `--model`), so the default base changes nothing for them | Nothing |
| A client that reads the served name from `GET /v1/models`, or matches `decisio-qwen3.6-35b-a3b-letters` | A server on the default base lists `decisio-gemma-4-31b-it-letters` | Read the name from `/v1/models`, or start the server with `--served-name` |
| Registered tasks (`POST /v1/tasks`, or a `--tasks-file`) fitted on the Qwen base | A server on another base does not apply them: it prints `WARNING: tasks fitted under another model or rendering are not applied` at start, and `POST /v1/tasks/import` refuses them | Register them again from their examples (`docs/tasks.md`), or keep serving the Qwen base |

### What differs in what you get from the default base

- The wire format and the routes are the same.
- The probabilities are another model's, calibrated with its own temperatures and prompt (`docs/cli.md`), so some choices differ from the Qwen base's.
  `EVAL_CARD.md` section 7 has the 31B's measurements and section 3 the Qwen base's; the README's "Choosing a base" says when each is the better choice.
- It reads a new state more slowly: 1.6, 3.2 and 5.1 times the Qwen base's time at 300, 1,000 and 3,000 tokens (80.7, 160.1 and 438.1 ms against 49.9, 50.0 and 85.2, one card at 585 W; `runs/2026-10-06_latency-585w/`).
- It is measured on a 96 GB card on vLLM (an 80 GB H100 also runs it, `docs/running.md`); on a Mac it runs from its 6-bit MLX conversion for states up to 16,383 tokens (`docs/running.md`).
- Its probabilities repeat exactly within a session, not across sessions: near-tied answers moved between two sessions in our measurements (`EVAL_CARD.md` 7.5).

### Keeping the previous behaviour

```bash
uv run python -m decisio.serve.vllm_engine --base qwen3.6-35b-a3b    # the Qwen base, as the container served it
DECISIO_BASE=qwen3.6-35b-a3b docker compose up                       # the same in the container
```
