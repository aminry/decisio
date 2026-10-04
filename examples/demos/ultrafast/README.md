# Ultrafast browser agent

A browser agent with an indexed action space.
Each step reads the page into a table of elements, then makes one System One request that picks an operation and, in the same request, a target for each operation that applies.
Code executes the choice.
A small text model writes text only when the chosen operation is `TYPE_TEXT`.
The demo is a modified copy of jev-ultrafast (see `ATTRIBUTION.md`), changed to talk to any System One server.

## One step, one request

The request carries the page as `state` and these questions:

| question | options | asked when |
| --- | --- | --- |
| `operation` | the supported operations: click, type text, select, scroll up, scroll down, wait, done, blocked | always |
| `click_target` | the clickable elements | there are some |
| `type_text_target` | the text fields | there are some |
| `select_target` | the dropdowns | there are some |

The number of options per question is the number of elements of that kind on the page.
Every decision records it (`option_counts`), and `../tools/measure_ultrafast.py` reports the maximum and the mean per question.
On the fixture pages the largest set is 9 options.
On real pages it grows with the page, which is why the server's option limit matters (below).

## Option limits by server

| server | most options per question |
| --- | --- |
| Decisio | 255 |
| Cygnet decision server | 255 (grouped passes) |
| decider | 255 |
| Winnow | 64 |
| Jev-Omni head | 256 slots; quality established up to 20 |
| Nimble, System One route | 26 (its scorer takes 255, so it needs the shim) |
| Ollama listing | 26, and at least 2 per question |

A model is excluded from a task where a step asks over more options than its limit.
The Ollama listing also refuses a question with a single option, which the agent asks whenever a page has exactly one text field or one dropdown, and it needs option descriptions as strings.
Two opt-ins cover both without changing the agent's choices: `SYSTEMONE_STRING_DESCRIPTIONS=1` sends each description as a compact JSON string, and `SYSTEMONE_SKIP_SINGLE_OPTION=1` leaves a one-option head out of the request and answers it with its only element at probability 1 (the record keeps the real option count and lists the skipped heads).
With both set, the agent runs end to end on the Ollama listing on the fixture pages, whose largest step asks over 9 options.
Real pages can ask over more than 26, and the listing is then excluded for that task.

## Run it

```sh
cp .env.example .env     # SYSTEMONE_BASE_URL, TEXT_MODEL_BASE_URL, TEXT_MODEL
uv run ultrafast-demo    # the inspector, on 127.0.0.1:$DEMO_PORT
```

The text helper is a small local model behind an OpenAI-compatible endpoint (for example Qwen3-0.6B on Ollama or vLLM).
It only writes the text of a typed field, so it is not what is being compared.
Set `TEXT_MODEL_EXTRA_JSON='{"chat_template_kwargs": {"enable_thinking": false}}'` where the model has a thinking switch.

`browser-harness` sends telemetry and checks for updates by default; `.env.example` turns both off (`BH_TELEMETRY=0`, `BH_UPDATE_CHECK=0`).

## Measure it

```sh
python ../tools/measure_ultrafast.py --label Decisio --runs 10
```

It serves the two fixture tasks on a loopback port, drives a headless Chrome over CDP, checks each outcome independently of the agent's own DONE, and writes a run record under `runs/`.
The numbers are decisions per second, p50 and p95 of the per-decision latency (the System One request), and task completion time with a bootstrap interval.
Each run also writes a trajectory (a screenshot per step with the chosen element and the decision), and `../tools/render_ultrafast.py` draws a clip from it without loading the page again (`../README.md`).
