<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Copyright contributors to the decisio project -->

# The HTTP API

The server speaks TypeSafe's System One wire format on `POST /v1/systemone`, so existing clients work unchanged, and adds routes for task registration, abstention and health.
It listens on `http://127.0.0.1:8000` unless `--host` and `--port` say otherwise.
The machine-readable form of this page is [`reference/openapi.json`](reference/openapi.json) (OpenAPI 3.1: every served route, its parameters, request and response models and headers), generated from the server's code and written at release time; the flags have the same kind of table, [`reference/flags.md`](reference/flags.md).
A malformed request is answered with status 422 and a `detail` naming what is wrong; a request the engine cannot answer, with 400.

| Route | What it does |
| --- | --- |
| `POST /v1/systemone` | answers typed questions about a state |
| `GET /v1/models` | lists the served model's name |
| `POST /v1/tasks`, `GET /v1/tasks`, `DELETE /v1/tasks/{id}` | registers, lists and removes per-task calibration and intent heads |
| `POST /v1/tasks/import` | loads tasks exported with `GET /v1/tasks?full=1` |
| `GET /v1/tasks/{id}`, `POST /v1/tasks/{id}/certify` | one task with its status; makes a certified task's certificate again on new labelled examples |
| `GET /metrics` | Prometheus text: per certified task its state, answers, certified answers and answer-mix distance |
| `POST /v1/abstention/tasks`, `GET /v1/abstention/tasks`, `DELETE /v1/abstention/tasks/{id}` | registers, lists (the export form) and removes per-task abstention thresholds |
| `POST /v1/abstention/tasks/import` | loads abstention tasks exported with `GET /v1/abstention/tasks` |
| `GET /health` | what the server is serving |
| `POST /v1/answer` | the engine's own letters route, for the conformance gates and benchmarks |

## `POST /v1/systemone`

This route follows TypeSafe's System One, so a System One client works unchanged.
vLLM 0.32 also serves a route called `/v1/systemone` ([vLLM #59299](https://github.com/vllm-project/vllm/pull/59299)); it is not this wire.
It defines `confidence` as the chosen option's probability over the whole vocabulary, where System One and decisio use the top probability rescaled from uniform (choice) and the concentration at the mode (score), and it takes up to 26 options per choice where System One and decisio take 255.
Check which server a client is pointed at before comparing confidences or sending more than 26 options.

### Request

| Field | Type | Meaning |
| --- | --- | --- |
| `state` | string, object or array | the text the questions are about; an object is rendered as `key: value` lines |
| `questions` | object, at least one entry | the questions, by name; the names are the keys of the response's `answers` |
| `model` | string, optional | TypeSafe's API requires it; decisio accepts a request without it. The response's `model` is the name the request sent, or the served model's name when it sent none |
| `images` | array of data URLs, optional | images for the image route (a server started with `--image-model`) |
| `assets` | object, optional | System One's images: placeholder names mapped to images, each placeholder found in the state or a question's text is replaced by its image (below) |

A prompt, the state with one question, can hold 32,768 tokens on the Qwen base. On a Gemma base it can hold the largest length that fits the card (at least 32,768, at most 98,304 on the 31B and 65,536 on the 12B), which the server chooses at start-up and `/health` reports as `max_model_len` with its reason, unless `--max-model-len` fixed it; with `--image-model` it is 32,768 ([`running.md`](running.md#the-context-length)). Every measured record in `EVAL_CARD.md` is at 32,768.
A prompt over that is answered with 422 and a `detail` that says the rendered prompt is longer than the context, with the lengths vLLM reports (shorten the state or the question and send it again); the server stays up.
A NaN or infinity in the engine's probabilities (a model whose arithmetic does not survive the device's dtype or matmul precision, or a checkpoint served under a profile that is not its own) is answered with 500 and a `detail` that says the values are not finite, that the request is not at fault, and which served model and engine produced them.
The same holds for `POST /v1/answer` and for the examples of `POST /v1/tasks` and `POST /v1/abstention/tasks`.
Fields the server does not know are ignored, so extensions (such as `orders`, for two-order averaging) and fields added to the wire format later do not break a request.

**`adapter` (optional, decisio's extension).** The name of a LoRA adapter the server loaded with `--adapter name=path`; the question is then answered by the base with that adapter (text route only). Without the field, or with `null` or an empty string, the base answers. An unknown name is answered 404 with the name and the adapters that are loaded; a value that is not a string is 422. The response carries `"adapter": "<name>"` (and the header `x-decisio-adapter`) when an adapter answered. The registered tasks (calibration, intent head, certificates, abstention thresholds) were fitted on the base's readout, so a question that one of them matches is answered 400 with the task named, not with a correction that does not fit. The temperature is the adapter's own for the question's type when the adapter has one (`--adapter-temperatures`, [running.md](running.md#lora-adapters)), and otherwise the base's. The task registration and certification routes refuse an `adapter` field instead of ignoring it. `POST /v1/answer` takes `adapter` too and reports it.

**`think` (optional, decisio's extension).** `true` or `false`: whether the questions of this request that are unsure are read again after the served model has written its reasoning ([running.md](running.md#thinking-when-unsure)). Absent or `null`, the server's setting (`--think`, or a tuned repository's `defaults.think`; off unless one of them says otherwise). A value that is not a boolean is 422, and so is `true` on a server whose engine cannot write reasoning (the vLLM engine can). When thinking is on for the request the response carries `"thinking": {"questions": n, "of": m, "tokens": t, "ms": ms, "factor": f, "budget": b, "per_question": {name: {"thought": true, "tokens": n} or {"thought": false, "why": ...}}}`, `usage.thinking_tokens` and the headers `x-decisio-think-ms` and `x-decisio-think-tokens`; `usage.output_tokens` is then one per question plus the reasoning tokens. A question of a registered task, one with more than 16 options, and every question of a request that asks for two orders, an abstention option or the image route stays at one read and the response says why.

**`layout` (optional, decisio's extension).** `"plain"` or `"header"`: the layout of the user turn this request is read under ([running.md](running.md#the-header-layout---layout)).
`"header"` lists the request's questions before the state; `"plain"` is the state, a blank line, then the question.
Absent or `null`, the server's layout (`--layout`, or the one the server decided at start: `/health`, `systemone.layout`); any other value is 422, naming the two.
A request of either layout is built token for token as a server started with that `--layout` builds it, and it is looked up among the registered tasks, and certified, as that server would do it:
- A registered task applies only to the requests of the layout it was fitted under (`POST /v1/tasks` and `POST /v1/abstention/tasks` take `layout` too, [tasks.md](tasks.md#a-task-under-a-layout)).
  A question of a certified task fitted under the other layout is answered with decisio's own read, with `certified: false` and `certified_reason: "layout_mismatch"`; a fast table never answers it, and a task that learns from traffic does not log it.
- The response says the layout it was read under in `x-decisio-prompt` (`layout=plain; ...`).
- An adapter, a tuned repository, the temperatures file (`--temperatures-file`) or a flavor that recorded another layout than the request's still answers, as asked, and the response carries `"layout_note"`, for example `"the adapter n1 was trained, and its temperatures fitted, under the header layout"`.
  It is absent when nothing differs.
- A request with images has no layout (the image route builds its own prompt): the field is 422 there.
  A server in `--mode packed` reads the plain layout only, so `"header"` is 422 there.
- The two layouts share no cached prefix for a state: a client that alternates them reads the state again on each change ([running.md](running.md#the-header-layout---layout)).

`POST /v1/answer` takes `layout` too, and reports the layout its rows were built under in `timing.layout`.

**A header over the context.** Under the header layout a request whose rows would be over the context because the header lists many long instructions is answered with a header that lists only the first questions that fit (none: the plain layout's row), unless the server was started with `--header-limit off` (then it is a 422 that says where the tokens are). The response then carries `"header_limited": {"layout": "header" | "plain", "listed": K, "of": N, "header_tokens": T, "of_header_tokens": W, "row_limit": L, "allowed_tokens": A, "reason": "context" | "cap"}` and the header `x-decisio-layout`; both are absent when the whole header was listed. A question of a certified task in such a request carries `certified: false` with `certified_reason: "header_limited"` ([running.md](running.md#a-header-over-the-context---header-limit)).

**`over_limit` (optional, decisio's extension).** What to do with a state that makes a row longer than the engine's prompt ceiling (the served context less the one generated token): `"refuse"` or `"truncate"`; absent or `null`, the server's setting (`--over-limit`, or a tuned repository's `defaults.over_limit`; refuse unless one of them says otherwise). A refused request is answered 422, as the engine says. With `"truncate"` the state, never a question, is cut once to its start, so that the longest question of the request still fits, and the request is answered from what was kept. The response then carries `"state_truncated": {"kept_tokens": N, "of_tokens": M, "part": "start", "limit": L}` (the state's tokens kept and in all, and the ceiling) and the header `x-decisio-state-truncated`; both are absent when the whole state was read. **No answer of a truncated state is certified**: a question of a certified task carries `certified: false` with `certified_reason: "state_truncated"`, and the traffic monitor does not count it. A request whose questions alone are over the ceiling is answered 422 whatever the setting; a value other than the two is 422. A server whose engine has no ceiling to read (the MLX build) takes `--over-limit-tokens` to say where to cut. Image requests are not cut.

**`task` (optional, decisio's extension).** The id of a registered task (`POST /v1/tasks`) that this request is for.
It answers the request's questions that ask its question (the same type and options; for a task whose options change with every request, the same instructions), whatever task they would find by themselves, and its context, if it has one, is read before the state ([tasks.md](tasks.md#task-context)).
Without the field each question finds its task by its question, as always; when several tasks ask the same question, each with its own context, a request that names none gets the one without a context, and is answered 422 when there is none without one.
The field is 422 when no task has that id, when the task asks none of the request's questions, when the task was fitted under the other layout than the request's (the message names the layout to send), and on the image route.
`x-decisio-tasks` says which task answered.
`POST /v1/answer` applies no registered task, so it answers the field with 422, and an example of a registration that names a task is 422 too.

**A task's context.** A question of a task registered with a `context` is read with the context before the state, so the request sends only the part that changes ([tasks.md](tasks.md#task-context)).
The questions of one request share one read of the state, so they must share one context: a request whose questions' tasks have different contexts, or that mixes a question of a task with a context and a question read without one, is answered 422 (send them as separate requests).
A request read under the other layout than its question's context task is answered 422 naming the task and the layout to send, whether or not it names the task: it is never read without the context.
A row over the context window is answered 422 with where its tokens are (the header, the task's context, the state, the question), and `over_limit: truncate` cuts the state only, never the context.
A request whose state starts with the task's context (leading whitespace aside) is served as sent, so the context is read twice, and the response carries the header `x-decisio-warning: state starts with the task's context; send only the varying part`.

Each question:

| Field | Type | Meaning |
| --- | --- | --- |
| `type` | `"noul"`, `"choice"` or `"score"` | yes/no, a choice among named options, or a level on an ordered scale |
| `instructions` | string, object or array, optional | the question; an object or array is sent as compact JSON; absent or blank, a fixed default per type |
| `criteria` | depends on `type` | `noul`: optional, `{"true": ..., "false": ...}` describing each side; `choice`: required, an object of option keys to descriptions (or `null`), up to 255 options (more is 422); `score`: required, a list of level descriptions, lowest first |

How a question is shown to the model (the layout, the yes/no rendering, which keys are shown) is set by the server's flags, listed in `docs/cli.md`, and a request may choose the layout (`layout`, above); the answer keys are always the request's own.

Images: as many per request as the context holds (the image engine takes up to `--max-images`, 128 unless set), PNG, JPEG or WebP, up to 20 MB each; each costs tokens that count against the context (a 512 by 512 image 256 tokens on the Qwen base).
They can be sent as data URLs in `images`, as `data:image/...` URIs anywhere inside the state, or as a `multipart/form-data` request whose `request` field holds the JSON body and whose `image`, `images` or `image[]` fields hold the files; a multipart request must carry an `X-Decisio-Client` header (any value, for example `X-Decisio-Client: my-app`), and is refused with 415 without one.
A request with images is served by the image engine; the text route is unchanged.
Starting the server with `--image-model` makes vLLM's engine run in a process of its own (`--engine-process separate`), which turns off the 31B's in-process `--multi-question warm` default and keeps the context at 32,768 tokens (`docs/cli.md`); the image route is measured with the Qwen base only.

**Images the System One way (`assets`).**
`assets` maps a placeholder name to an image: `{"type": "image", "format": "png"|"jpg"|"jpeg", "data_base64": "..."}`, or, in a multipart request, `{"type": "image", "part": "<the name of an image part>"}` (a part wins over `data_base64`).
Every place a placeholder appears, as a string or inside one, in the state or in a question's instructions or criteria descriptions (option keys are not searched; the longest name wins when one contains another), becomes the marker `[image N]` and the asset's bytes become image N, in reading order; a placeholder used twice is two images.
That is the same path as a `data:image` URI in the state: the image engine gets the same state text, the same questions and the same images, and gives the same probabilities as the other encodings (test I7), with System One's answer fields only (no `unknown_probability` or `abstained`).
A placeholder name that matches no asset is plain text.
A multipart request puts the `request` part first and every image in a part of its own, with a unique, non-empty name that exactly one asset uses; it needs the `X-Decisio-Client` header like any multipart request here.
A problem with an asset (a name no text uses, an empty `assets`, a format that is not png, jpg or jpeg, bad base64, bytes that are not an image, an image part nothing uses or two assets share, `assets` together with `images`, more images than the engine's `--max-images`) is a 400 with `{"detail": {"error_type": "api_usage_error", "message": "..."}}`, System One's shape for image errors.
System One's `TypeSafe-Danger-Zone-Images` header is not needed and is ignored.
A calibration or head task's examples (`POST /v1/tasks`) are text only and refuse `assets`; the examples of `POST /v1/abstention/tasks` may carry them.

What an image costs and where it is cut down (the image route runs on the Qwen base, `--image-model Qwen/Qwen3.6-35B-A3B-FP8`; the numbers are that model's processor settings and `smart_resize` from the pinned transformers, computed on 2026-10-14):
- An image over 20 MiB, or over 20 megapixels (width times height), or in another format, is refused with 422 (`bad image: ...`) before it is read; a request with more images than `--max-images` (128 unless set) is a 400 `api_usage_error`, as the paragraph above says.
- The processor then resizes the image, keeping its proportions and rounding both sides to multiples of 32, so that it has between 65,536 pixels (256 by 256) and 16,777,216 (4,096 by 4,096). An image of 200 by 200 is read as 256 by 256; one of 4,000 by 6,000 as 3,328 by 4,992.
- Each image costs one token for every 32 by 32 pixels of the resized image, and the tokens count in `usage.input_tokens` and against the 32,768-token context that a server with `--image-model` serves: 64 tokens at the least, 16,384 at the most (1,024 by 768: 768 tokens; 1,920 by 1,080, read as 1,920 by 1,088: 2,040; 4,000 by 6,000: 16,224).
- The Ollama listings are not this route. `ollama show` (2026-10-14) lists the Qwen listing `aminroudaki/decisio` with the capabilities `decision` and `vision`, and the Gemma listing `aminroudaki/decisio-gemma` (every tag) with `decision` only. Whether the Qwen listing's image answers match this route's was not measured.

Every POST must have `Content-Type: application/json` (parameters such as `; charset=utf-8` are fine): any other type, or none, is refused with 415 and `accept-post: application/json` before the body is read (`SECURITY.md`, "Cross-site requests from a web page"). The one exception is `multipart/form-data` on this route, with an `X-Decisio-Client` header (any value): a web page cannot add a custom header to a cross-site request without a preflight, which the server refuses.

Request headers:
- `x-decisio-route: text` or `image` chooses the route explicitly.
- `x-decisio-debug: readout` or `hidden` adds the raw readout to the response, on a server started with `--debug-readout` only (verification, never production).


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
| `unknown_probability`, `abstained` | with the abstention extension of imajev (an open-source System One client and test harness, github.com/mohit67890/imajev; `--abstain-option`): the probability of the offered "can't tell" option and whether the server abstained |

`model` echoes the name the request sent, so a client left on the SDK's default model name gets that name back unchanged; a request without one gets the served model's name (`--served-name`, or the base's).
`usage.input_tokens` counts the state and each question's own text, and what else is read with them: the header under the header layout, a task's context, both copies under read twice; `usage.output_tokens` is one per question.

Response headers:
- `x-decisio-server-ms`: the server's time for the request, from the route receiving it to the response: reading and validating the body, loading its images, and every read, including a certified task's questions and the intent head's.
- `x-decisio-route`: the route that served it (`text` or `image`).
- `x-decisio-tasks`: the registered tasks applied, when any were.
- `x-decisio-warning`: something in the request is likely a mistake, though it was served as sent: `state starts with the task's context; send only the varying part` ([tasks.md](tasks.md#task-context)); absent otherwise.
- `x-decisio-adapter`: the adapter that answered, when one did.
- `x-decisio-think-ms`, `x-decisio-think-tokens`: the milliseconds spent on reasoning and the second reads, and the reasoning tokens generated, when thinking was on for the request.
- `x-decisio-state-truncated`: `kept=N; of=M; part=start`, when the state was cut (`over_limit`, above); absent otherwise.
- `x-decisio-layout`: `header; listed=K; of=N; reason=context` (or `plain; ...`), when the header listed only some of the request's questions (`--header-limit`, above); absent otherwise.
- `x-decisio-prompt`: `layout=header; tokens=1523; cached=1440`: the layout of the user turn the request was read under (its `layout`, else the server's), the mean prompt tokens the engine read per question, and the mean of them the prefix cache served. Under the header layout a state cached for another set of questions shows as a low `cached`. Absent when the engine does not report prompt sizes.
- `x-decisio-stages`: where the engine's time went, for example `prepare=...;warm=...;questions=...;readout=...;engine=...;wait=...`. `engine` is the request's own engine call; `wait` is the time before it, from the request reaching the engine until it took the engine's lock. A request read with a task's context has `context`: the context's warm-up sent in this request (vLLM) or the context's part of the prefix (MLX), summed over the reads that read the context, so within `engine`, `task` or `head` ([tasks.md](tasks.md#the-context-in-the-prefix-cache)). A certified task's questions, read one at a time by the certificate's function or answered by its fast table, are `task`; the intent head's read is `head`; the reasoning pass is `think`. So `engine + wait + task + head + think` is the server time within the request's own overhead (a few ms). The wait is 0 on an idle server. With `--concurrency batched`, a request that shared an engine call (`x-decisio-batch`) reports the whole call's `questions` and `engine` times.
- `x-decisio-batch`: `requests=N; rows=M` when the server batches concurrent requests (`--concurrency batched`) and this request's questions were scored in one engine call with other requests' (N requests, M rows in all); absent when it was scored alone. On `POST /v1/answer` the same facts are `timing.batch`.
- `x-decisio-wait-behind`: `registration` when a boundary registration (`--register-boundary after`) was running when the request arrived, so the `wait` was behind it; absent otherwise. On `POST /v1/answer` the same facts are `timing.waited_ms` and `timing.waited_behind_registration`.

## `GET /v1/models`

The served model in System One's format; the name is the base's served name unless `--served-name` sets another.

## Where this route differs from System One

The conformance check compares the request and response fields with System One's published OpenAPI document (version 0.2.0) and fails on any difference that is not in this list:
- **`model` is optional.** System One requires it and refuses a request without it with 422; here a request without it is answered and the response names the served model.
- **A score may have more than 10 levels.** System One refuses more than 10 with 422 (`too_long`); decisio accepts more (up to the engine's 255 labels) and does not cap them.
- **`box` questions are not supported.** A request with `"type": "box"` is refused with 422.
- **Images (`assets`) are read, with these differences.** Images per request are limited by the context, as on System One (about 110 images of 320 by 320 pixels in System One's 32k tokens, more with its long-context header), up to the engine's `--max-images` (128 by default); no size window (System One takes sides of 280 to 1,568 pixels and refuses others with "Invalid image dimensions"; here the image is resized, up to 20 megapixels and 20 MB); WebP bytes are accepted besides PNG and JPEG (System One refuses them); a placeholder is replaced longest name first (System One replaces naively, so `PHOTO_1` inside `PHOTO_10` breaks it); an image costs the Qwen image engine's tokens, not System One's; a multipart request needs `X-Decisio-Client`; and the image route runs on a server started with `--image-model`, on the Qwen base.
- **Unknown fields are accepted.** System One refuses a request with an unknown top-level field (400 `Invalid request.`), so its `images`-style extensions cannot be sent to it; decisio ignores unknown fields (imajev's `images` and `orders` are such extensions).
- **`usage.output_tokens` is 1 per question**, not the tokens a generating model produced.
- Probabilities and confidences are returned at full precision (System One rounds them to two decimals), the response has no `x-typesafe-request-id` header (see the `x-decisio-*` headers), and `model` is the name the request sent (the served model's name when it sent none), not the name of the model that answered.

## What one request can hold

Decisio sets no limit of its own on the size of a request body, on the number of questions in one request, or on the number of concurrent connections; uvicorn's defaults apply (no body limit, a backlog of 2,048). What bounds a request:
- **The context.** The state with one question, as rendered, can hold up to the server's `max_model_len` tokens (at least 32,768; `/health` reports it and why, and `running.md` says how it is chosen); over it the answer is 422 (above).
- **Options per question.** A choice has up to 255 options (the wire format's cap); more is refused with 422 and a `too_long` detail, as System One does (`Dictionary should have at most 255 items after validation, not 256`). A score has no cap on its levels here (System One's is 10).
- **Questions per request.** No cap: a request with no questions is 422, and any other number is answered. Each question costs its own time: with `--multi-question sequential` (the default of the Qwen and 12B bases, and of the 31B when it has a second engine) about 20 ms per further question on a cached state (`EVAL_CARD.md` section 4); with `--multi-question warm` (the default of the 31B, the default base, when its engine runs in the server's process) the questions are scored in one batch, 5.4 ms per question at 100 questions and 2.7 ms at 1,000 on an 8,000-token state on the Qwen base (`EVAL_CARD.md` section 3, `runs/2026-09-30_plugin-verification/bench_patched_on.json`). The response and the batch grow with the question count, so for thousands of questions send several requests about the same state: the second and later read the state from the prefix cache.
- **Concurrency.** The server runs one engine and, by default, its calls are serialized by a lock, so concurrent requests queue behind it and are answered in turn; there is no queue limit. With `--concurrency batched` the requests waiting for the engine are scored together in one call instead (`running.md`, "Concurrency"; `x-decisio-batch` says when a request shared a call). It was measured with 32 and 64 concurrent clients sending first and follow-up questions about new 1,000-token states (`docs/running.md`, "A later question about the same state"); `x-decisio-stages` reports each request's wait (`wait=`), and `x-decisio-wait-behind: registration` says when it waited behind a background boundary registration.

## Calling the server from a browser

The server sets no CORS headers and has no authentication on any route.
A page on another origin therefore cannot read its answers, and cannot make it do work either:
- A request with a JSON body makes the browser send a preflight `OPTIONS` first; the server answers 405 (`allow: POST`), so the browser never sends the `POST`. Checked on the CPU stand-in on 2026-10-14 with `curl -X OPTIONS -H 'Origin: http://example.com' -H 'Access-Control-Request-Method: POST'`: `405 Method Not Allowed`, no `access-control-*` header.
- A request a browser sends **without** a preflight (a "simple" one: `text/plain`, `application/x-www-form-urlencoded` or `multipart/form-data`) is refused with `415` and `accept-post: application/json` before its body is read, on every route (`SECURITY.md`, "Cross-site requests from a web page"), because a simple request needs no preflight, so a page could otherwise make the server do the work, register or replace tasks, even though it cannot read the answer. The one exception is `multipart/form-data` on `/v1/systemone` (imajev's image clients), which is accepted when the request carries an `X-Decisio-Client` header (any value): a page cannot add a custom header without a preflight, which the server refuses.

Do not expose the port to browsers, and do not rely on CORS to protect it.
To call it from a web page, call it from your own backend, or put a reverse proxy in front that authenticates the caller and adds the CORS headers your page needs (`Access-Control-Allow-Origin`, `Access-Control-Allow-Headers: content-type`, and an answer to `OPTIONS`).
A client of this server sends `Content-Type: application/json`; every client the integration checks cover does.

## What to watch

- **`GET /health`.** The HTTP status is the contract: 200 while the server is serving, 503 with `{"ok": false, "engine": "dead", ...}` from the moment the engine is declared dead. Alert on anything but 200. The other fields describe the running server (listed below) and can gain fields between releases; read them for what is being served, not as an alert signal.
- **The process exit code.** 70 means the engine died and the server ended itself; a restart policy acts on the exit (`docs/running.md`, "When the engine dies"; `docs/operating.md`).
- **Per request**, in the response headers: `x-decisio-server-ms` (the server's time), `x-decisio-stages` (`prepare`, `warm`, `engine`, `wait`, and `context`, `task`, `head` and `think` when the request used them) and `x-decisio-wait-behind` when the request waited behind a registration. A `wait` that grows is requests queueing behind the engine.
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
An optional `"layout": "plain" | "header"` fits the task under that layout (default: the server's): its examples are read under it, and it applies to the requests of that layout only (`layout` above, [tasks.md](tasks.md#a-task-under-a-layout)); an example whose request names another layout is 422.
The other registration routes (`"options": "dynamic"`, a task that learns from traffic, `POST /v1/abstention/tasks`) take it too; `POST /v1/tasks/{id}/certify` certifies a task under the layout it was fitted under.

An optional `"context"` is a long, fixed text read before every state of the task, the examples' included: a string as given, an object or array as compact JSON.
Requests then send only the part that changes, and after the first request the context is read from the prefix cache ([tasks.md](tasks.md#the-context-in-the-prefix-cache)).
On a server that pads (the Qwen base on vLLM) the pad is sized so that the context ends on the cache block, so such a request is not always the same tokens as one that sends the context in its state.
The labelled, dynamic-option and learning registrations take it; `POST /v1/abstention/tasks` does not (abstention tasks take no context).
Its sha256 is part of the task's identity: the same question registered with another context is another task, kept beside the first, and a request picks between them with `task` (above).
An empty context, or one of another type, is 422, and so is a context that with the question and the template is over the context window before any state.
The task's record, its listing and `/health` give the context's `sha256`, `chars` and `tokens`, never its text; the export (`GET /v1/tasks?full=1`) holds the text, which an import needs.

- `GET /v1/tasks` lists the registered tasks, whether tasks are applied (`--tasks`, on by default), whether the server can serve intent heads (`head_engine`), and the server's fingerprint (the model and rendering tasks are fitted under).
- `GET /v1/tasks?full=1` includes the heads' arrays: the form `--tasks-file` loads at start and `POST /v1/tasks/import` takes.
- With `--task-store` a task's listing and registration response carry `store_version`; a write names the version it replaces (by default the one the replica serves; `x-decisio-replaces: N` names another) and answers **409** `{message, kind, task, store_version, your_version}` when the store holds another (`?replace=1` replaces it), 409 on a replica started `--task-store-readonly`, and **503** when the store cannot be reached; the change is not made in any of them (`docs/tasks.md`, "Replicas").
- A write that fails under `--tasks-dir` (disk full, read-only) answers 500 on registration, certification, import and removal, and the change is not made (`docs/tasks.md`, "Saved to disk").
- `GET /v1/tasks` and `GET /v1/abstention/tasks` carry a `store` object: where tasks are saved, how many are loaded and apply, what the start loaded from where.
- `POST /v1/tasks/import` with `{"tasks": [...]}` loads tasks without fitting them again; tasks fitted under another model or rendering are refused and must be registered again from their examples.
  A task of either layout loads, and applies to the requests of its layout (so does `POST /v1/abstention/tasks/import`).
- `DELETE /v1/tasks/{id}` removes a task (404 when there is none of that id).

**Certified answers.** `POST /v1/tasks` also takes an optional `certify` block, `{"alpha": 0.05, "delta": 0.1, "min_examples": 59, "holdout": 0.5, "suspend_on_alert": true, "examples": [...]}` (every key optional; `min_examples` defaults to what `alpha` and `delta` need, 59 at the defaults, and can only raise that; `examples` are labelled like the task's own and are never fitted on, and without them the server holds out `holdout` of the task's examples, fits on the rest and says so in the stored `certificate.split`).
It refuses with 422, and registers nothing, when fewer held-out examples than the procedure can use at all are given: 59 at the defaults, whatever the model does.
The response carries the `certificate` (`threshold`, `alpha`, `delta`, `n`, `certified_share`, the `split`, the serving-stack fingerprint, `served_check` (the confidences re-read through the serving path on a sample of the held-out examples: `n`, `max_shift`, `p99_shift`, `crossed_threshold`, `top_answer_changed`), a `summary` in words) and a `status` (`ok`, `traffic_changed` or `suspended`, with a `text`).
With no threshold that holds, `certificate.threshold` is `null`, nothing is certified, and the task is otherwise served normally.
Every later answer of a certified task carries `certified` (true or false), `certificate` (its id) and, when false, `certified_reason`: `below_threshold`, `nothing_certified`, `serving_stack_changed` (the serving stack differs from the certificate's; also a request whose `orders` is not the server's `--orders`, which nothing changed on the server: [tasks.md](tasks.md#certified-answers)), `traffic_changed` (the traffic monitor's confidence or unseen-options alert is on and the certificate was made with `suspend_on_alert`, the default: [tasks.md](tasks.md#the-monitor)), `state_truncated` (the request's state was cut, `over_limit` above), `header_limited` (the header listed only some of the request's questions, above), `think_mode_changed` (the server thinks and the certificate was made one-pass, or the other way round), `concurrency_changed` (the server batches concurrent requests and the certificate was made one request at a time, or the other way round: `running.md`, "Concurrency") or `layout_mismatch` (the task was fitted under the other layout than the request's, `layout` above).
A change of the serving stack suspends the certificate (every answer certified false, status `suspended: serving stack changed`, the differing parts listed in `status.changed`); `POST /v1/tasks/{id}/certify` with `{"examples": [...]}` (and optionally `alpha`, `delta`, `min_examples`) makes it again under the current stack, on examples that share no input with the fit. It keeps the old `alpha`, `delta` and `suspend_on_alert` unless the call names them, and a `min_examples` only if the original call asked for more than its `alpha` and `delta` need; otherwise the minimum is derived from the new `alpha` and `delta` (29 at 10% and 10%), so a looser target needs fewer examples than the 59 of the defaults.
**Options that change with every request.** `POST /v1/tasks` also takes `"options": "dynamic"` (with `"match": "instructions"`, the only value, and optionally `"none_option"`, the key of the requests' "none of these" option, `"none"` by default) for a recurring question whose options are different in every request, such as which brand a product title names or which of a request's tools to call.
The examples are choice questions that share one set of instructions; each may offer its own options, and its `answer` is one of its option keys, or `null` for its none option.
A `certify` block is required: a dynamic task is certified, on decisio's own top probability whatever the options.
The task is then matched on the question's instructions, and every later choice question with those instructions is answered through it (unless a fixed-option task registered for exactly that set of options matches it first).
The response's `dynamic` object says what was fitted (`applied`, `reason`, the cross-validated `cv` figures, the `temperature`, the 13 `weights`, and in the listing the size of the `prior`; `GET /v1/tasks?full=1` holds the table of hashed option texts) and what is stored; its `calibration` and `head` are present and not applied.
The task's `status.monitor` has no answer mix; its signals are `confidence` and, where options recur in the examples, `unseen_options` (the share of requests whose options the examples never contained; `status.monitor.unseen_options.armed` says whether it is watched), and either one suspends certification while it alerts.
`docs/tasks.md`, "Tasks whose options change with every request", has the measurements and the limits.

`docs/tasks.md`, "Certified answers", says what the promise is and where it stops.

**A fast table.** `POST /v1/tasks` also takes `"fast_table": true` (with a `certify` block; choice questions; `pip install 'decisio[table]'`): a sentence encoder and a classifier on the host CPU answer the requests they are sure of without the model, kept only if the registration times them at least 25% faster. The response's `fast_table_report` says whether it was kept and why, an answer of such a task carries `answered_by` (`table` or `decisio`), and `status.fast_table` says what this replica does with it. `docs/tasks.md`, "A fast table in front of decisio", has the scope, the numbers and the multi-replica rules.

`docs/tasks.md` walks through registration and states what each number was measured on.

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
  `systemone.readout_tasks` has one entry per task the server holds, keyed by the task's id, whatever its layout, model or rendering (whether it applies is in `GET /v1/tasks/{id}`):
  - `calibration`: true when its per-task calibration is applied;
  - `head`: true when its intent head is applied;
  - `context`, only for a task with a context: `sha256`, `chars` and `tokens`, never its text.

  It holds nothing else: the certificate, the monitor's status, the fast table and the layout are in `GET /v1/tasks/{id}` and `GET /v1/tasks`.
- `head_engine`, the intent head's hidden-state reader (the serving engine by default, a second engine with `--head-engine`), and `image_engine` when one is running.

Once the engine has died, `/health` answers 503 with `{"ok": false, "engine": "dead", "reason": "...", "exit_code": 70}`, every other route answers 503 with the reason, and the server exits with code 70 (`docs/running.md`, "When the engine dies").

## `POST /v1/answer`

The engine's own route, below System One: a state and a list of questions in the engine's form (`kind`, `instructions`, `options`), answered with each question's options and probabilities and the engine's timing.
The conformance gates and the benchmark scripts use it; clients should use `/v1/systemone`.
