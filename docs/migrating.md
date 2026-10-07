<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Copyright contributors to the decisio project -->

# Moving a System One client to decisio

decisio serves `POST /v1/systemone` and `GET /v1/models` in TypeSafe's System One wire format, so an existing client needs only a new base URL and a model name.
This page shows how to point a client at a decisio server and lists the differences that matter once it is there.
The SDK steps below were run with TypeSafe's Python SDK 0.7.2 against a decisio server on 2026-10-05.
The SDK is TypeSafe's, published under the MIT licence ([PyPI](https://pypi.org/project/typesafe-sdk/), [repository](https://github.com/typesafe-ai/typesafe-sdk-python)); decisio does not include it, and a client installs it on its own.
decisio is an independent project, not affiliated with or endorsed by TypeSafe; it implements TypeSafe's published System One wire format.

## Point the client at the server

With TypeSafe's Python SDK, pass the server's address and its served model name:

```python
from typesafe_sdk import TypeSafeClient

client = TypeSafeClient(
    base_url="http://127.0.0.1:8000",
    api_key="unused",                          # the SDK requires one; decisio ignores it
    model="decisio-gemma-4-31b-it-letters",    # the name GET /v1/models lists
)
answer = client.system_one(
    state="Since this morning none of our staff can log in to the dashboard.",
    questions={"category": {"type": "choice", "instructions": "Which team should handle this ticket?",
                            "criteria": {"billing": None, "access": None, "bug": None}}},
)
```

Or leave the code as it is and set the SDK's environment variables:

```bash
export TYPESAFE_BASE_URL=http://127.0.0.1:8000
export TYPESAFE_API_KEY=unused
export TYPESAFE_DEFAULT_MODEL=decisio-gemma-4-31b-it-letters
```

Any other client sends the same JSON body to `http://<host>:8000/v1/systemone`.
`client.models.list()`, or `curl http://127.0.0.1:8000/v1/models`, returns the served name: `decisio-gemma-4-31b-it-letters` for the default base, `decisio-qwen3.6-35b-a3b-letters` or `decisio-gemma-4-12b-it-letters` for the others, unless the server was started with `--served-name`.

## The differences that matter

**The `model` field does not choose the model.**
A decisio server serves one base, chosen when it starts (`--base`), whatever the request names.
The response echoes the name the request sent, so a client left on the SDK's default model name gets that name back unchanged.
Set the client's model to the served name, and read `GET /health` (its `base` and `profile`) when you need to know what answered.

**There is no authentication.**
The `Authorization` header is accepted and ignored, and so is any API key.
Keep the server on 127.0.0.1, or behind a reverse proxy that authenticates (`SECURITY.md`).

**Answers are read, not generated.**
Each question's distribution comes from the option letters' logits at one position, in one forward pass, tempered by the base's fitted temperatures.
`confidence` follows System One's rules, `usage.output_tokens` is one per question, and `usage.input_tokens` counts tokens in the base's own tokenizer, so token counts differ from another provider's.

**Unknown request fields are ignored, not rejected.**
A field decisio does not know is accepted and has no effect, so check that a field your client relies on is in `docs/api.md`.
The extensions decisio reads are `images` and `orders`.

**Limits.**
Up to 255 options per question (a request with more gets 400), a prompt of up to 32,768 tokens, and up to 2 images per request, which need a server started with `--image-model`.

**Errors.**
A request that fails validation gets 422 and one the engine cannot answer gets 400, each with a `detail` naming the problem.
TypeSafe's SDK raises its own error types for them (`TypeSafeUnprocessableEntityError`, `TypeSafeBadRequestError`).
decisio sets no rate limit, so the SDK's retries on 429 never fire; it still retries 408 and 5xx.

**Timeouts.**
The SDK's default timeout is 10 s.
A long state seen for the first time takes longer on the larger bases, and the first request after a start warms the engine; raise `timeout` for those.

**Repeatability is stated per base.**
On the Qwen base a question returns the same probabilities every time, alone or among other questions; the Gemma 4 12B base returns the same choice with probabilities that can move slightly; `EVAL_CARD.md` sections 6.5 and 7.5 have what was measured on each Gemma base (and the README's "Choosing a base", `docs/cli.md` "Determinism").
Among exactly tied options, the answer is the key that sorts first, so it never depends on the order the keys arrive in.

**What decisio adds.**
- Per-task calibration and intent heads, learned from your own labelled examples (`POST /v1/tasks`, `docs/tasks.md`).
- Per-task abstention thresholds (`POST /v1/abstention/tasks`).
- Response headers naming the server's time, the route and the tasks applied (`x-decisio-server-ms`, `x-decisio-route`, `x-decisio-tasks`, `x-decisio-stages`).
- `GET /health`, naming the base, its checkpoint and revision, the temperatures and the prompt.

`docs/api.md` has every field, and `docs/running.md` every way to start a server.
