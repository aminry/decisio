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
Fields the server does not know are ignored, so extensions (such as `orders`, for two-order averaging) and fields added to the wire format later do not break a request.

Each question:

| Field | Type | Meaning |
| --- | --- | --- |
| `type` | `"noul"`, `"choice"` or `"score"` | yes/no, a choice among named options, or a level on an ordered scale |
| `instructions` | string, object or array, optional | the question; an object or array is sent as compact JSON; absent or blank, a fixed default per type |
| `criteria` | depends on `type` | `noul`: optional, `{"true": ..., "false": ...}` describing each side; `choice`: required, an object of option keys to descriptions (or `null`), up to 255 options; `score`: required, a list of level descriptions, lowest first |

How a question is shown to the model (the layout, the yes/no rendering, which keys are shown) is set by the server's flags, listed in `docs/cli.md`; the answer keys are always the request's own.

Images: up to 2 per request, PNG, JPEG or WebP, up to 20 MB each.
They can be sent as data URLs in `images`, as `data:image/...` URIs anywhere inside the state, or as a `multipart/form-data` request whose `request` field holds the JSON body and whose `image`, `images` or `image[]` fields hold the files.
A request with images is served by the image engine; the text route is unchanged.

Request headers:
- `x-decisio-route: text` or `image` chooses the route explicitly.
- `x-decisio-debug: readout` or `hidden` adds the raw readout to the response, on a server started with `--debug-readout` only (verification, never production).

The earlier `x-rlcd-route` and `x-rlcd-debug` headers are still honoured.

### Response

```json
{"model": "decisio-qwen3.6-35b-a3b-letters",
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
- `x-decisio-stages`: where the engine's time went, for example `prepare=...;warm=...;questions=...;readout=...;engine=...`.

## `GET /v1/models`

The served model in System One's format; the name is the base's served name unless `--served-name` sets another.

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

`docs/tasks.md` walks through registration and states what each number was measured on; `docs/handoffs/tasks.md` is the reference for its rules.

## Abstention

`POST /v1/abstention/tasks` registers a threshold on an abstain option's probability, fitted from labelled examples that include unanswerable ones:

```json
{"id": "...",
 "option": {"key": "<the request's own abstain key>"} or {"append": "<text of an option the server adds>"},
 "match": "option_set" or "imajev_extension",
 "examples": [{"request": <a /v1/systemone request, images allowed>, "answer": <the option key, or null when unanswerable>}, ...]}
```

The server then abstains when that option's probability exceeds the threshold, and otherwise answers the best of the other options; the probabilities themselves are reported unchanged.
The threshold is kept only when the examples justify it.
`GET /v1/abstention/tasks` lists the tasks and whether thresholds are applied (`--abstention`, on by default).

## `GET /health`

What the server is serving, in one object a run record can quote:
- `ok`, and the engine's facts: the vLLM version, the GPU, the mode, the padding (`pad_unit`, `pad_where`), the quantization and the cache data types.
- `block_size` and `match_unit`, the KV cache's block and the unit a prefix match is counted in; `cache_hit_unit`, the step prefix-cache hits come in, and `hash_unit`, the step prefixes are hashed at (on the Gemma base, whose cache keeps groups of 16- and 64-token blocks, hits come in 64-token steps).
- `base` and `profile`: the base, the checkpoint and its revision, the temperature for each question type, and the prompt (layout, answer position, label forms, system turn, yes/no and option rendering, multi-question scoring, padding).
- `prompt_format`: the prompt's layout settings.
- `systemone`: the System One route's settings (rendering rules, `noul_commit`, abstention and its tasks, `orders`, tasks and which are applied, the temperatures).
- `head_engine`, the intent head's hidden-state reader (the serving engine by default, a second engine with `--head-engine`), and `image_engine` when one is running.

Once the engine has died, `/health` answers 503 with `{"ok": false, "engine": "dead", "reason": "...", "exit_code": 70}`, every other route answers 503 with the reason, and the server exits with code 70 (`docs/running.md`, "When the engine dies").

## `POST /v1/answer`

The engine's own route, below System One: a state and a list of questions in the engine's form (`kind`, `instructions`, `options`), answered with each question's options and probabilities and the engine's timing.
The conformance gates and the benchmark scripts use it; clients should use `/v1/systemone`.
