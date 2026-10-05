<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Copyright contributors to the decisio project -->

# The command line

```bash
uv run python -m decisio.serve.vllm_engine [flags]
```

`--base` chooses the base model and brings its checkpoint at a pinned revision and its value for every setting marked "the base's" below; a flag given explicitly overrides it.
Without `--base`, the base is detected from `--model`'s `config.json`; with neither, the server refuses to start.
The served defaults need no other flag: every default below is what was measured, base by base (`EVAL_CARD.md` sections 1, 6.1 and 7.1).
`gemma-4-31b` is quantized to FP8 when it loads (vLLM 0.30.0's FP8 on load), a setting of its profile; `--engine '{"quantization": null}'` loads it at bf16, which does not leave room for a 32,768-token context on a 96 GB card (`EVAL_CARD.md` section 7).
`--help` prints the same list with each flag's description.

## Every flag

Where the columns differ, the value is the base's.

### Model and engine

| Flag | `qwen3.6-35b-a3b` | `gemma-4-12b` | `gemma-4-31b` | What it does |
| --- | --- | --- | --- | --- |
| `--base` | | | the base; without it, detected from `--model` |
| `--model` | `Qwen/Qwen3.6-35B-A3B-FP8` | `google/gemma-4-12B-it` | `google/gemma-4-31B-it` | the checkpoint, a local directory or a Hugging Face repository id |
| `--revision` | `95a723d08a9490559dae23d0cff1d9466213d989` | `707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7` | `842da3794eaa0b77d5f08bae87a17459d91ff475` | the checkpoint's revision on the Hub; the pin applies whenever `--model` is the base's own repository |
| `--backend` | `vllm` | `vllm` | `vllm` | `vllm`; `mlx` for Apple silicon (the Qwen base and `gemma-4-12b`, `docs/running.md`; not `gemma-4-31b`); `hf` for the CPU stand-in, not for measurement |
| `--model-class` | `hidden-readout` | `hidden-readout` | `hidden-readout` | `hidden-readout`: decisio's text class that also returns the hidden state at the answer position, for the intent head; `text-only`: the same without it (the default with `--head-engine`); `view`: a directory built by `decisio.serve.make_text_only`, loaded as it is, needing no plugin |
| `--mode` | `separate` | `separate` | `separate` | `separate`: one prompt per question, the state shared through the prefix cache; `packed`: questions packed into one pooling request (the compact layout only) |
| `--pack` | 16 | 16 | 16 | questions per pack in packed mode |
| `--engine` | `{"compilation_config": {"max_cudagraph_capture_size": 4096}}` | the same | the same | extra `LLM(...)` keyword arguments, as JSON |
| `--adapter` | none | none | none | `name=path` of a vLLM-format LoRA adapter; repeatable |
| `--gpu-memory-utilization` | 0.9 | 0.9 | 0.9 | the text engine's share of the card |
| `--allow-deep-gemm` | off | off | off | start even when `VLLM_USE_DEEP_GEMM` is set to something other than 0 |
| `--served-name` | `decisio-qwen3.6-35b-a3b-letters` | `decisio-gemma-4-12b-it-letters` | `decisio-gemma-4-31b-it-letters` | the name `GET /v1/models` lists |
| `--host`, `--port` | `127.0.0.1`, 8000 | the same | the same | the listen address; put a reverse proxy in front to expose it |

### The prompt

| Flag | `qwen3.6-35b-a3b` | `gemma-4-12b` | `gemma-4-31b` | What it does |
| --- | --- | --- | --- | --- |
| `--prompt-tail` | `spaced` | `spaced` | `spaced` | `spaced`: a blank line before and after the lettered options, then one line asking for the chosen option's letter alone; `compact`: the earlier layout, for tasks registered under it |
| `--answer-slot` | `prefill` | `template` | `template` | where the label is read: after "Answer:" prefilled in the assistant turn, or at the chat template's own first assistant position |
| `--label-variants` | `single` | `summed` | `summed` | the tokens read per label: the one form the slot reads, or every single-token form summed |
| `--system-prompt` | off | on | on | a system turn before each question |
| `--noul-rendering` | `letters-keys` | `letters` | `letters` | how a yes/no question is asked: `letters-keys`, a two-option letter choice, the false side first, the sides named; `letters`, the same with each side shown as its description; `words`, the earlier rendering, read from the yes and no tokens |
| `--describe-options` | on | on | on | show an option that has a description as its description alone; `--no-describe-options` shows `key: description` |
| `--hide-index-keys` | on | on | on | never show enumerated keys (`option_0`, `option_1`, ...) |
| `--desnake-labels` | on | on | on | show bare snake_case labels as words |

### Temperatures and outputs

| Flag | `qwen3.6-35b-a3b` | `gemma-4-12b` | `gemma-4-31b` | What it does |
| --- | --- | --- | --- | --- |
| `--temperature` | 1.506 | 3.592 | 5.252 | the temperature on the text route's plain readout, `softmax(log p / T)`; 1 switches it off; never changes the most probable option |
| `--temperature-choice` | 1.370 | the global one | 4.672 | the temperature for choice questions |
| `--temperature-noul` | the global one | the global one | the global one | the temperature for yes/no questions |
| `--temperature-score` | the global one | the global one | the global one | the temperature for score questions |
| `--noul-commit` | off | off | off | report an uncommitted yes/no probability at the band's edge (below) |
| `--orders` | 1 | 1 | 1 | 2: two-order averaging, each question also read in a second option order |
| `--branch-log` | none | none | none | a JSONL file for the two orders' disagreement per question |

### Several questions and padding

| Flag | `qwen3.6-35b-a3b` | `gemma-4-12b` | `gemma-4-31b` | What it does |
| --- | --- | --- | --- | --- |
| `--multi-question` | `sequential` | `sequential` | `sequential` | how a request's questions are scored (below) |
| `--engine-process` | `separate` | `separate` | `separate` | where vLLM's engine runs: `separate`, a process of its own (vLLM's arrangement); `in`, the server's process (below) |
| `--pad-policy` | `always` | `always` | `always` | `always`: pad every state; `shared`: pad only requests with more than one question; `row`: pad a single-question request's whole row (below); `none`: never pad (the default with `--backend mlx`) |
| `--pad-to` | `block` | `none` | `none` | what a state is padded to: the KV cache block, a token count, or nothing |
| `--pad-where` | `front` | `front` | `front` | where the padding goes: `front`, before the chat template; `user`, at the start of the user turn; `between`, between the state and the question |

### Tasks and abstention

| Flag | `qwen3.6-35b-a3b` | `gemma-4-12b` | `gemma-4-31b` | What it does |
| --- | --- | --- | --- | --- |
| `--tasks` | on | on | on | apply registered per-task calibration and intent heads (`POST /v1/tasks`) |
| `--tasks-file` | none | none | none | a JSON list of tasks to load at start, as `GET /v1/tasks?full=1` returns them |
| `--head-engine` | off | off | off | read the intent head's hidden state from a second copy of the model in vLLM's pooling mode instead of from the serving engine: faster head questions, at a second weight copy; not with `--image-model` |
| `--head-gpu-memory-utilization` | 0.47 | 0.47 | 0.47 | the second engine's share of the card |
| `--abstention` | on | on | on | apply registered abstention thresholds (`POST /v1/abstention/tasks`) |
| `--abstention-tasks` | none | none | none | a JSON list of abstention tasks to load at start |
| `--abstain-option` | none | none | none | offer this extra option (for example "can't tell") on requests that use imajev's extension, reported as `unknown_probability` and `abstained`; untrained |

### Images

| Flag | `qwen3.6-35b-a3b` | `gemma-4-12b` | `gemma-4-31b` | What it does |
| --- | --- | --- | --- | --- |
| `--image-model` | none | none | none | the full multimodal checkpoint, as a second engine serving only requests that carry images; measured with the Qwen base only |
| `--image-gpu-memory-utilization` | 0.5 | 0.5 | 0.5 | the image engine's share of the card |
| `--one-engine` | off | off | off | load only the image engine and serve text requests on it too, for a card that cannot hold both; changes the text route's class |

### The Mac (`--backend mlx`)

| Flag | Default | What it does |
| --- | --- | --- |
| `--prefix-cache-mb` | 2048 | the cross-request prefix cache's budget in MB; 0 turns it off |
| `--tokenizer` | `Qwen/Qwen3.6-35B-A3B-FP8` | the tokenizer the prompts are built with, so they are the vLLM path's byte for byte |

### Verification

| Flag | Default | What it does |
| --- | --- | --- |
| `--debug-readout` | off | honour the `x-decisio-debug` request header (the raw readout in the response); never in production |

## What the behaviour flags do

### Several questions in one request (`--multi-question`)

With `sequential`, the served default, a request with several questions first prefills the state once, then scores each question in its own engine call, reading the state from the prefix cache, so every answer equals the same question sent alone.
With `warm`, for bulk scoring, the questions are scored in one batch after the same prefill: faster (four questions on a new state take about half the time of `sequential` on the Qwen base and about two thirds on the Gemma bases), but a batched answer differs from the same question sent alone, and when two options are close the choice can change (`EVAL_CARD.md` section 4).
With the engine in its own process (`--engine-process separate`, the default) a warm answer can also move between repeats, because that process does not always put a request's questions in one engine step (vllm-project/vllm#59764); with `--engine-process in` an identical request repeats exactly.
`--engine-process in` is refused with a second engine (`--image-model`, `--head-engine`, `--one-engine`), and `/health` reports the arrangement as `engine_process`.
With `batch`, every question goes in one engine call that prefills the state itself, with no warm-up.
Measurements: `EVAL_CARD.md` section 4, `runs/2026-10-02_multi-question-and-rendering/`, and the README's example request in `runs/2026-10-01_docker-first-gpu-start/repeat_variability/`.

### Row padding (`--pad-policy row`)

By default every state is padded to end on the KV cache block, so a single question on a state the server has not seen is prefilled in two engine steps, the state and then the question.
With `row`, a single-question request is padded so that its whole row, state and question, ends on the block boundary, and vLLM prefills it in one engine step; requests with several questions are padded as by default.
It is faster on a new state, and accuracy and calibration stayed within noise on the suite, JevBench, the four Decision Index benchmarks and the intent heads.
Its cost is repeatability: a repeated identical request can move slightly on long prompts, where the default padding returns the same probabilities every time, so it is off by default.
The demos ran with it (`docs/demos/README.md`).
Measurements: `runs/2026-10-02_pad-policy-row/`; not measured on the Gemma base.

### Determinism

On a running server with the Qwen base, a question returns the same probabilities every time, alone or among other questions: each question is scored in its own engine call.
Across server restarts, single-question answers have matched an earlier record on the whole suite in most starts measured, and moved slightly without changing a choice in the others.
The Gemma base does not repeat bit for bit: its logits come out in bf16 after its soft cap, and a state read for the first time and read again from the prefix cache can land a step apart; several questions in one request stay within the bound in the README's settings table.
Any change of rounding moves this model's answers, the batch, the padding or where a long row is split, so every bit-identity claim holds for a fixed configuration and request form.
Measurements: `EVAL_CARD.md` sections 4 and 6.5.

### Committed yes/no answers (`--noul-commit`)

For scorers that treat a yes/no probability strictly between 0.20 and 0.80 as no answer, as JevBench v1.5 does, such an answer is reported at the band's edge on its own side: 0.80 above 0.5, and 0.20 at or below it.
The answer never changes; its probability does, and calibration pays for it: log loss and ECE rise, while JevBench v1.5's yes/no competence rises.
Measurements: `runs/2026-10-03_noul-commit/` and `docs/handoffs/tasks.md`.

### The earlier layout (`--prompt-tail compact`)

The compact layout has no blank lines around the options and ends with "Answer with the letter only.".
Tasks registered under one layout are not applied under the other, so a task registered before the spaced layout became the default needs `--prompt-tail compact` or registering again.
Packed mode reads the compact layout only, and the image route keeps it.
