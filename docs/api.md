<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Copyright contributors to the decisio project -->

# The HTTP API

The server speaks TypeSafe's System One wire format on `POST /v1/systemone`, so existing clients work unchanged, and adds routes for task registration, abstention and health.
It listens on `http://127.0.0.1:8000` unless `--host` and `--port` say otherwise.
A malformed request is answered with status 422 and a `detail` naming what is wrong; a request the engine cannot answer, with 400.

| Route | What it does |
| --- | --- |
| `POST /v1/systemone` | answers typed questions about a state |
| `GET /v1/models` | lists the served model's name |
| `POST /v1/tasks`, `GET /v1/tasks`, `DELETE /v1/tasks/{id}` | registers, lists and removes per-task calibration and intent heads |
| `POST /v1/tasks/import` | loads tasks exported with `GET /v1/tasks?full=1` |
| `GET /v1/tasks/{id}`, `POST /v1/tasks/{id}/certify` | one task with its status; makes a certified task's certificate again on new labelled examples |
| `GET /metrics` | Prometheus text: per certified task its state, answers, certified answers and answer-mix distance |
| `POST /v1/abstention/tasks`, `GET /v1/abstention/tasks` | registers and lists per-task abstention thresholds |
| `GET /health` | what the server is serving |
| `POST /v1/answer` | the engine's own letters route, for the conformance gates and benchmarks |

## `POST /v1/systemone`

### Request

| Field | Type | Meaning |
| --- | --- | --- |
| `state` | string, object or array | the text the questions are about; an object is rendered as `key: value` lines |
| `questions` | object, at least one entry | the questions, by name; the names are the keys of the response's `answers` |
| `model` | string, optional | TypeSafe's API requires it; decisio accepts a request without it and names the served model in the response |
| `images` | array of data URLs, optional | images for the image route (a server started with `--image-model`) |

A prompt, the state with one question, can hold up to 32,768 tokens (`EVAL_CARD.md` section 2).
A prompt over that is answered with 422 and a `detail` that says the rendered prompt is longer than the context, with the lengths vLLM reports (shorten the state or the question and send it again); the server stays up (before 0.11.0 it was a bare 500).
A NaN or infinity in the engine's probabilities (a model whose arithmetic does not survive the device's dtype or matmul precision, or a checkpoint served under a profile that is not its own) is answered with 500 and a `detail` that says the values are not finite, that the request is not at fault, and which served model and engine produced them; before 0.12.0 a NaN in a choice question surfaced as a 400 "min() iterable argument is empty".
The same holds for `POST /v1/answer` and for the examples of `POST /v1/tasks` and `POST /v1/abstention/tasks`.
Fields the server does not know are ignored, so extensions (such as `orders`, for two-order averaging) and fields added to the wire format later do not break a request.

Each question:

| Field | Type | Meaning |
| --- | --- | --- |
| `type` | `"noul"`, `"choice"` or `"score"` | yes/no, a choice among named options, or a level on an ordered scale |
| `instructions` | string, object or array, optional | the question; an object or array is sent as compact JSON; absent or blank, a fixed default per type |
| `criteria` | depends on `type` | `noul`: optional, `{"true": ..., "false": ...}` describing each side; `choice`: required, an object of option keys to descriptions (or `null`), up to 255 options; `score`: required, a list of level descriptions, lowest first |

How a question is shown to the model (the layout, the yes/no rendering, which keys are shown) is set by the server's flags, listed in `docs/cli.md`; the answer keys are always the request's own.

Images: up to 2 per request, PNG, JPEG or WebP, up to 20 MB each.
They can be sent as data URLs in `images`, as `data:image/...` URIs anywhere inside the state, or as a `multipart/form-data` request whose `request` field holds the JSON body and whose `image`, `images` or `image[]` fields hold the files; a multipart request must carry an `X-Decisio-Client` header (any value, for example `X-Decisio-Client: my-app`), and is refused with 415 without one.
A request with images is served by the image engine; the text route is unchanged.

What an image costs and where it is cut down (the image route runs on the Qwen base, `--image-model Qwen/Qwen3.6-35B-A3B-FP8`; the numbers are that model's processor settings and `smart_resize` from the pinned transformers, computed on 2026-10-14):
- An image over 20 MiB, or over 20 megapixels (width times height), or in another format, is refused with 422 (`bad image: ...`) before it is read; a third image is refused with 422 as well.
- The processor then resizes the image, keeping its proportions and rounding both sides to multiples of 32, so that it has between 65,536 pixels (256 by 256) and 16,777,216 (4,096 by 4,096). An image of 200 by 200 is read as 256 by 256; one of 4,000 by 6,000 as 3,328 by 4,992.
- Each image costs one token for every 32 by 32 pixels of the resized image, and the tokens count in `usage.input_tokens` and against the 32,768-token context: 64 tokens at the least, 16,384 at the most (1,024 by 768: 768 tokens; 1,920 by 1,080, read as 1,920 by 1,088: 2,040; 4,000 by 6,000: 16,224).
- The Ollama listings are not this route. `ollama show` (2026-10-14) lists the Qwen listing `aminroudaki/decisio` with the capabilities `decision` and `vision`, and the Gemma listing `aminroudaki/decisio-gemma` (every tag) with `decision` only. Whether the Qwen listing's image answers match this route's was not measured.

Every POST must have `Content-Type: application/json` (parameters such as `; charset=utf-8` are fine): any other type, or none, is refused with 415 and `accept-post: application/json` before the body is read (`SECURITY.md`, "Cross-site requests from a web page"). The one exception is `multipart/form-data` on this route, with an `X-Decisio-Client` header (any value): a web page cannot add a custom header to a cross-site request without a preflight, which the server refuses.

Request headers:
- `x-decisio-route: text` or `image` chooses the route explicitly.
- `x-decisio-debug: readout` or `hidden` adds the raw readout to the response, on a server started with `--debug-readout` only (verification, never production).

The earlier `x-rlcd-route` and `x-rlcd-debug` headers are still honoured.

### Response

```json
{"model": "decisio-gemma-4-31b-it-letters",
 "answers": {
   "urgent":   {"type": "noul", "noul": ...},
   "category": {"type": "choice", "choice": "access", "confidence": ..., "probabilities": {"billing": ..., "access": ..., ...}},
   "impact":   {"type": "score", "score": ..., "confidence": ..., "legend": {"0": "...", ...}, "probabilities": {"0": ..., ...}}},
 "usage": {"input_tokens": ..., "output_tokens": ...}}
```

| Answer field | Meaning |
| --- | --- |
| `noul` | the probability of yes |
| `choice` | the most probable option's key; among exactly tied options, the key that sorts first in Unicode code-point order, so the answer never depends on the order the keys arrive in |
| `score` | the probability-weighted mean level |
| `probabilities` | the distribution over the options (choice) or the levels by index (score) |
| `confidence` | System One's confidence: for a choice, the top probability rescaled from uniform; for a score, the concentration at the mode |
| `legend` | each level's index and description |
| `unknown_probability`, `abstained` | with imajev's abstention extension (`--abstain-option`): the probability of the offered "can't tell" option and whether the server abstained |

`usage.input_tokens` counts the state and each question's own text; `usage.output_tokens` is one per question.

Response headers:
- `x-decisio-server-ms`: the server's time for the request.
- `x-decisio-route`: the route that served it (`text` or `image`).
- `x-decisio-tasks`: the registered tasks applied, when any were.
- `x-decisio-stages`: where the engine's time went, for example `prepare=...;warm=...;questions=...;readout=...;engine=...;wait=...`. `engine` is the engine call itself; `wait` is the time before it, from the request reaching the engine until it took the engine's lock, so `engine + wait` is the server time within the request's own overhead (a few ms). The wait is 0 on an idle server.
- `x-decisio-wait-behind`: `registration` when a boundary registration (`--register-boundary after`) was running when the request arrived, so the `wait` was behind it; absent otherwise. On `POST /v1/answer` the same facts are `timing.waited_ms` and `timing.waited_behind_registration`.

## `GET /v1/models`

The served model in System One's format; the name is the base's served name unless `--served-name` sets another.

## What one request can hold

Decisio sets no limit of its own on the size of a request body, on the number of questions in one request, or on the number of concurrent connections; uvicorn's defaults apply (no body limit, a backlog of 2,048). What bounds a request:
- **The context.** The state with one question, as rendered, can hold up to 32,768 tokens; over it the answer is 422 (above).
- **Options per question.** A choice has up to 255 options (the wire format's cap); more is refused with 400 and the detail `letter readout supports 1 to 255 options, got N` (checked on the stand-in with 256).
- **Questions per request.** No cap: a request with no questions is 422, and any other number is answered. Each question costs its own time: with the default `--multi-question sequential` about 20 ms per further question on a cached state (`EVAL_CARD.md` section 4); with `--multi-question warm` the questions are scored in one batch, 5.4 ms per question at 100 questions and 2.7 ms at 1,000 on an 8,000-token state on the Qwen base (`EVAL_CARD.md` section 3, `runs/2026-09-30_plugin-verification/bench_patched_on.json`; the Gemma 31B's default is `warm`). The response and the batch grow with the question count, so for thousands of questions send several requests about the same state: the second and later read the state from the prefix cache.
- **Concurrency.** The server runs one engine and its calls are serialized by a lock, so concurrent requests queue behind it and are answered in turn; there is no queue limit. It was measured with 32 and 64 concurrent clients sending first and follow-up questions about new 1,000-token states (`docs/running.md`, "A later question about the same state"); `x-decisio-stages` reports each request's wait (`wait=`), and `x-decisio-wait-behind: registration` says when it waited behind a background boundary registration.

## Calling the server from a browser

The server sets no CORS headers and has no authentication on any route.
A page on another origin therefore cannot read its answers, and cannot make it do work either:
- A request with a JSON body makes the browser send a preflight `OPTIONS` first; the server answers 405 (`allow: POST`), so the browser never sends the `POST`. Checked on the CPU stand-in on 2026-10-14 with `curl -X OPTIONS -H 'Origin: http://example.com' -H 'Access-Control-Request-Method: POST'`: `405 Method Not Allowed`, no `access-control-*` header.
- A request a browser sends **without** a preflight (a "simple" one: `text/plain`, `application/x-www-form-urlencoded` or `multipart/form-data`) is refused with `415` and `accept-post: application/json` before its body is read, on every route, from 0.12.0 (`SECURITY.md`, "Cross-site requests from a web page"). **Before 0.12.0 such a POST was carried out:** a `text/plain` POST to `/v1/systemone` was answered 200 on 0.11.1 (checked on the stand-in the same day), the page could not read the answer, but the server did the work, and a page could register or replace tasks. The one exception is `multipart/form-data` on `/v1/systemone` (imajev's image clients), which is accepted when the request carries an `X-Decisio-Client` header (any value): a page cannot add a custom header without a preflight, which the server refuses.

Do not expose the port to browsers, and do not rely on CORS to protect it.
To call it from a web page, call it from your own backend, or put a reverse proxy in front that authenticates the caller and adds the CORS headers your page needs (`Access-Control-Allow-Origin`, `Access-Control-Allow-Headers: content-type`, and an answer to `OPTIONS`).
A client of this server sends `Content-Type: application/json`; every client the integration checks cover does (`docs/integrations.md`).

## What to watch

- **`GET /health`.** The HTTP status is the contract: 200 while the server is serving, 503 with `{"ok": false, "engine": "dead", ...}` from the moment the engine is declared dead. Alert on anything but 200. The other fields describe the running server (listed below) and can gain fields between releases; read them for what is being served, not as an alert signal.
- **The process exit code.** 70 means the engine died and the server ended itself; a restart policy acts on the exit (`docs/running.md`, "When the engine dies"; `docs/operating.md`).
- **Per request**, in the response headers: `x-decisio-server-ms` (the server's time), `x-decisio-stages` (`prepare`, `warm`, `engine`, `wait`) and `x-decisio-wait-behind` when the request waited behind a registration. A `wait` that grows is requests queueing behind the engine.
- **Metrics only for certified tasks, and no structured log.** `GET /metrics` serves Prometheus text for certified tasks (state, answers, certified answers and the answer-mix distance: [`tasks.md`](tasks.md), "Certified answers"); the server exposes no Prometheus metrics for the engine, and its output is the plain text of its start-up lines and uvicorn's warnings. Take latency and queueing from the headers above, in your client or your proxy's log.

## Tasks

`POST /v1/tasks` registers one recurring question from labelled examples:

```json
{"id": "ticket-routing",
 "examples": [{"request": <a /v1/systemone request with exactly one question, text only>, "answer": <the option key>}, ...]}
```

Every example must ask the same question with the same options.
The server fits a per-task calibration and, for a question with 10 or more options, an intent head on the model's hidden state, keeps each only when cross-validation on the examples shows a gain, and answers with the task's record: what was fitted, what is applied, and why anything was not.
Every later question with the same option list is answered with the task applied.

- `GET /v1/tasks` lists the registered tasks, whether tasks are applied (`--tasks`, on by default), whether the server can serve intent heads (`head_engine`), and the server's fingerprint (the model and rendering tasks are fitted under).
- `GET /v1/tasks?full=1` includes the heads' arrays: the form `--tasks-file` loads at start and `POST /v1/tasks/import` takes.
- `POST /v1/tasks/import` with `{"tasks": [...]}` loads tasks without fitting them again; tasks fitted under another model or rendering are refused and must be registered again from their examples.
- `DELETE /v1/tasks/{id}` removes a task (404 when there is none of that id).

**Certified answers.** `POST /v1/tasks` also takes an optional `certify` block, `{"alpha": 0.05, "delta": 0.1, "min_examples": 59, "holdout": 0.5, "examples": [...]}` (every key optional; `examples` are labelled like the task's own and are never fitted on, and without them the server holds out `holdout` of the task's examples, fits on the rest and says so in the stored `certificate.split`).
It refuses with 422, and registers nothing, when fewer held-out examples than the procedure can use at all are given: 59 at the defaults, whatever the model does.
The response carries the `certificate` (`threshold`, `alpha`, `delta`, `n`, `certified_share`, the `split`, the serving-stack fingerprint, a `summary` in words) and a `status` (`ok`, `traffic_changed` or `suspended`, with a `text`).
With no threshold that holds, `certificate.threshold` is `null` and the task works as before with nothing certified.
Every later answer of a certified task carries `certified` (true or false), `certificate` (its id) and, when false, `certified_reason`: `below_threshold`, `nothing_certified` or `serving_stack_changed`.
A change of the serving stack suspends the certificate (every answer certified false, status `suspended: serving stack changed`, the differing parts listed in `status.changed`); `POST /v1/tasks/{id}/certify` with `{"examples": [...]}` (and optionally `alpha`, `delta`, `min_examples`) makes it again under the current stack, on examples that share no input with the fit.
`docs/tasks.md`, "Certified answers", says what the promise is and where it stops.

`docs/tasks.md` walks through registration and states what each number was measured on; `docs/handoffs/tasks.md` is the reference for its rules.

## Abstention

`POST /v1/abstention/tasks` registers a threshold on an abstain option's probability, fitted from labelled examples that include unanswerable ones:

```json
{"id": "...",
 "option": {"key": "<the request's own abstain key>"} or {"append": "<text of an option the server adds>"},
 "match": "option_set", "option_keys" or "imajev_extension",
 "examples": [{"request": <a /v1/systemone request, images allowed>, "answer": <the option key, or null when unanswerable>}, ...]}
```

`option_set` is for one question with one option list: the task applies to a question whose options (keys and descriptions, in any order) are those of the examples.
`option_keys` is for a family of questions that share their option keys and not their texts, such as a multiple-choice exam whose items each have their own four answers behind the keys A to D and an `idk` key: the task applies to a choice question whose set of keys is exactly the examples' (any descriptions, any instructions), and the threshold is fitted on the pooled probabilities of the abstain key across the items.
It is deliberately the whole key set and not the abstain key alone: the probability of an abstain option moves with the number of candidates, so a threshold fitted on four candidates plus `idk` is not applied to ten plus `idk` (register one task per key set), and a question that merely has an option of the same name is not caught.
When several tasks match a question, the exact `option_set` one wins over an `option_keys` one.
The server then abstains when that option's probability exceeds the threshold, and otherwise answers the best of the other options; the probabilities themselves are reported unchanged.
The threshold is kept only when the examples justify it.
`GET /v1/abstention/tasks` lists the tasks and whether thresholds are applied (`--abstention`, on by default).

## `GET /health`

What the server is serving, in one object a run record can quote:
- `ok`, and the engine's facts: the vLLM version, the GPU, the mode, the padding (`pad_unit`, `pad_where`), the quantization and the cache data types.
- `block_size` and `match_unit`, the KV cache's block and the unit a prefix match is counted in; `cache_hit_unit`, the step prefix-cache hits come in, and `hash_unit`, the step prefixes are hashed at (on the Gemma 4 12B base, whose cache keeps groups of 16- and 64-token blocks, hits come in 64-token steps; 32 on the 31B; 1,056 on the Qwen base); `registers_state_boundary`, whether a single question registers its state's boundary in the cache for the next question, and `register_boundary`, when (`after` the response or `before` the question; `docs/running.md`); with `after`, `state_boundary` counts the registrations pending, deferred, registered, dropped and failed.
- `base` and `profile`: the base, the checkpoint and its revision, where vLLM's engine runs (`engine_process`), the temperature for each question type, and the prompt (layout, answer position, label forms, system turn, yes/no and option rendering, multi-question scoring, padding).
- `prompt_format`: the prompt's layout settings.
- `systemone`: the System One route's settings (rendering rules, `noul_commit`, abstention and its tasks, `orders`, tasks and which are applied, the temperatures).
- `head_engine`, the intent head's hidden-state reader (the serving engine by default, a second engine with `--head-engine`), and `image_engine` when one is running.

Once the engine has died, `/health` answers 503 with `{"ok": false, "engine": "dead", "reason": "...", "exit_code": 70}`, every other route answers 503 with the reason, and the server exits with code 70 (`docs/running.md`, "When the engine dies").

## `POST /v1/answer`

The engine's own route, below System One: a state and a list of questions in the engine's form (`kind`, `instructions`, `options`), answered with each question's options and probabilities and the engine's timing.
The conformance gates and the benchmark scripts use it; clients should use `/v1/systemone`.
