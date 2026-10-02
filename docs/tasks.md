<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Copyright contributors to the decisio project -->

# Teaching the server your question: task registration

This guide is for someone who has never used Decisio's tasks.
It explains what a task is, when registering one helps, and how to do it end to end.
The exact rules (formats, fields, the head's two modes) are in the reference, `docs/handoffs/tasks.md`.
A runnable walk-through, which works on a laptop without a GPU, is in `examples/tasks/`.

## What a task is

A task is one question you ask again and again, with the same list of options every time.
"Which support queue should handle this ticket?" with the same twelve queues on every ticket is a task.
"Which of these four answers is right?" on exam questions, where the four answers change with each question, is not.

## Why register one

The model behind Decisio is a frozen generalist: it reads your options' names and descriptions and answers from what it already knows.
It does not know that in your company "access" means login problems and not building badges, or that it tends to pick one queue too often for your tickets.
Registering a task teaches the server both from a handful of labelled examples of your own question, without training or changing the model:

- **Calibration** fits one correction per option to the model's answer probabilities: it removes a steady bias toward some options and makes the probabilities honest.
- **The intent head**, for questions with 10 or more options, fits a small linear layer on the model's internal representation of the text: it learns what your labels mean in your data.

Both are fitted in seconds to minutes when you register, stored with the task, and applied automatically to every later question with the same option list.
Each is kept only if cross-validation on your own examples shows that it helps; otherwise the question is served exactly as before, and the server tells you why.

## What you need

- **Examples of the same question.** Each example is one real input (a ticket, an email, a record) with the option you would want chosen.
  Every example must use the same question with the same options, in the same order, as the questions you will ask later.
- **Enough of them.**
  - For calibration: at least 10 examples, below which the server declines to fit. The calibration results under "What you get" were measured with 20 examples per task; fewer were not measured here.
  - For the head: a question with 10 or more options, and at least 5 examples of every option. The measured gains below used 10 per option, which is what we recommend.
- **Examples from your own traffic.** Past inputs your team has already routed or labelled are ideal. They should look like what the server will see: the same kind of text, the same mix of options where you can.
- **A few more to check the result.** Keep some labelled inputs out of registration; `examples/tasks/evaluate.py` measures accuracy on them before and after.

## Walk-through

The commands below use the example task in `examples/tasks/`, a synthetic ticket-routing question with 12 queues, and run from the repository root.
Install the package and start a server in one terminal:

```
uv sync --extra dev --frozen
uv run python -m decisio.serve.vllm_engine --backend hf --model Qwen/Qwen3-0.6B-Base
```

That is the CPU stand-in, which runs anywhere; on a GPU, install with `uv sync --extra serve --frozen` and start `uv run python -m decisio.serve.vllm_engine --model Qwen/Qwen3.6-35B-A3B-FP8`.
The server is ready when `curl -s http://127.0.0.1:8000/health` answers; run the steps below in a second terminal.

### 1. Build the examples file

Registration is one `POST /v1/tasks` whose body names the task and lists the examples.
Each example is an ordinary `/v1/systemone` request with exactly one question, plus the answer you want:

```
{"id": "ticket-routing",
 "examples": [
   {"request": {"state": "My card keeps getting declined when I renew.",
                "questions": {"route": {"type": "choice",
                                        "instructions": "Which support queue should handle this ticket?",
                                        "criteria": {"billing": "Invoices, charges on the account, plan prices",
                                                     "payment_failed": "A card or bank payment was declined or failed",
                                                     "...": "..."}}}},
    "answer": "payment_failed"},
   ...]}
```

`examples/tasks/register.py` builds this body from a CSV with `text` and `label` columns and the question in `question.json`; `examples/tasks/examples.json` is the result for the example's 120 tickets.

### 2. Register

```
uv run python examples/tasks/register.py --id ticket-routing --data examples/tasks/train.csv
```

or, with the prepared body:

```
curl -s http://127.0.0.1:8000/v1/tasks -H 'Content-Type: application/json' -d @examples/tasks/examples.json
```

The server scores every example exactly as it would serve it, fits both corrections, and answers with the stored task.
Registering the same `id` again replaces the task.

### 3. Read the response

The response has one record per correction, `calibration` and `head`, each with:

- **`applied`**: whether the server will use it. `true` means later questions with this option list are answered with the correction.
- **`reason`**: why, in words. "cross-validated gain" when it was kept; otherwise, for example, that there were fewer than 10 examples, that some option had fewer than 5 examples (head), that the question has fewer than 10 options (head), or that cross-validation on your examples prefers the plain readout.
- **The cross-validation evidence**: the server split your examples into five parts and scored each part with a correction fitted on the other four.
  - For calibration: `cv_logloss_plain` and `cv_logloss_fitted` (lower is better), `cv_acc_plain` and `cv_acc_fitted`, and `cv_t`, the test statistic of the change in log loss (negative means the correction lowered it). Calibration is kept only if log loss falls by at least 0.005 per example, `cv_t` is at most -1.645 (a one-sided 95% test), and accuracy does not fall.
  - For the head: `cv_logloss`, the cross-validated log loss for each penalty strength tried and for no head (`"none"`); the head is kept with the best penalty (`lambda`) if that beats no head.

Where the head is kept, calibration is not stacked on top of it: the head already includes what calibration would correct.

**When a correction is declined**, nothing is lost: the question is answered as it was before registration.
Look at the reason:

- Too few examples: add examples, up to 20 for calibration or 10 per option for the head, and register again.
- Cross-validation prefers the plain readout: the model already answers this question as well as your examples can show. That is common for a clear yes/no question.
- Check that the labels are consistent. Two people labelling the same kind of input differently look like noise, and the server will decline.

### 4. Ask a question and see the task applied

Ask the question exactly as registered:

```
uv run python examples/tasks/ask.py "My card keeps getting declined when I renew."
```

The response header `x-decisio-tasks: ticket-routing` says the question matched the registered task, so the task's kept corrections answered it.
No header means the question matched no task: check that its question and options are exactly those you registered.
If the task kept neither correction, the header still names it and the answer is the plain one; `GET /v1/tasks` shows what each task kept.

### 5. List, export, reload, delete

```
curl -s http://127.0.0.1:8000/v1/tasks                        # registered tasks and each fit's summary
curl -s 'http://127.0.0.1:8000/v1/tasks?full=1' > tasks.json  # export, head parameters included
curl -s http://127.0.0.1:8000/v1/tasks/import -H 'Content-Type: application/json' -d @tasks.json
curl -s -X DELETE http://127.0.0.1:8000/v1/tasks/ticket-routing
```

**Registration is per server instance, in memory.** A restart, or a second server behind a load balancer, does not see a task registered on another.
To persist a task, export it (`GET /v1/tasks?full=1`), then either start each server with `--tasks-file tasks.json` or post the export to `POST /v1/tasks/import`.
A task is valid only for the model and rendering it was fitted under (the served name, the checkpoint, padding and the rendering flags): a task fitted on another configuration is not applied, and the server says which.
The description rule (`--describe-options`, on by default) is part of the task's key instead, and only for questions it changes: a question whose options have descriptions has another key with the rule on than off, so a task fitted under the other setting is simply not matched; a question it leaves alone (index keys, bare labels, yes/no, score) keeps its tasks either way.

**Abstention tasks come after the readout task.** An abstention threshold (`POST /v1/abstention/tasks`, a threshold on a "can't tell" option) is fitted on the probabilities the question is actually served with, which include a registered task's correction.
Register the readout task first and the abstention task second; if you register the readout task again, register its abstention task again too.

## What you get

Every number here is on the served model (Qwen3.6-35B-A3B-FP8 on one RTX PRO 6000 Blackwell, `EVAL_CARD.md`) and comes from the record named beside it.
In every intent measurement the test items are held out: none of them is among the registered examples.

| Measure | Without a task | With a task | Measured on | Record |
| --- | ---: | ---: | --- | --- |
| BANKING77 intent accuracy (77 options) | 0.740 | **0.847**; +10.7 points [+5.8, +15.8] | 150 test items; 10 labelled training examples per intent registered; mean of three draws; the default single-engine mode | `runs/2026-09-30_plugin-verification/intent_heads/`, `derived/intent_head_gains.json` |
| CLINC150 intent accuracy (150 options) | 0.820 | **0.893**; +7.3 points [+2.3, +13.0] | 100 test items; as above | same |
| The same two, second-engine mode | 0.740, 0.820 | 0.849 (+10.9 [+6.0, +16.0]); 0.890 (+7.0 [+1.7, +12.7]) | the same items and draws, the hidden state from a second engine | `runs/2026-09-29_tasks-endpoint/manifest.json` |
| Decision Index BANKING77, macro-F1 | 0.729 | **0.841** | its 3,080 requests; heads registered from 10 training texts per intent; second-engine mode; the same server with and without | `runs/2026-09-29_tasks-endpoint/decision_index/*/di_report.json` |
| Decision Index CLINC150+OOS, macro-F1 | 0.812 | **0.916** | its 5,500 requests; as above, plus 10 out-of-scope training texts | same |
| Per-task calibration, harm | | never hurts on six held-out tasks | 20 labelled examples, 100 draws; "never hurts": mean accuracy change at least -0.5 points, mean log-loss change at most +0.005, at most 5% of draws losing 2 points; measured change -0.14 to +0.28 points | `runs/2026-09-29_tasks-endpoint/evaluation.json` (`r10`) |
| Per-task calibration, gain | pooled ECE 0.058 | pooled ECE **0.037**; SciFact +6.6 points | six other held-out tasks, 20 labelled examples, 100 draws, 900 test items | same (`r9`) |
| Registration time | | about 230 s for 770 examples, 455 s for 1,500 (0.30 s per example) | the default single-engine mode | `runs/2026-09-30_plugin-verification/intent_heads/*.json.gz` (`register_s`) |
| | | about 160 s for 770 examples, 330 s for 1,500 (0.21 s per example) | second-engine mode | `runs/2026-09-29_tasks-endpoint/intent_heads/*.json.gz` |
| Server time per intent question | 46.5 ms | 126.8 ms | the default single-engine mode (three extra engine requests per head question) | `runs/2026-09-30_plugin-verification/manifest.json` (`latency.json.gz`) |
| | 82 ms | 82 ms on first sight; 42 ms when the identical question is asked again | second-engine mode, on that run's card: the median of each round, a question's first sight (82.35 ms with and without a task) and two repeat rounds (41.6, 41.8 ms), where the pooling engine reuses more of an identical prompt | `runs/2026-09-29_tasks-endpoint/evaluation.json` (`latency.head.server_ms_round_medians`) |

The intervals are 95% paired bootstrap intervals over test items (the gain averaged over the three draws of examples).
The single-engine intervals are computed from the record's per-item rows by `benchmarks/intent_head_gain.py`, with the plain answer taken from the same forward pass as the head's, 10,000 paired resamples of the test items and numpy's `default_rng(0)`; the output is stored as `runs/2026-09-30_plugin-verification/derived/intent_head_gains.json`, and another seed moves the interval ends by a few tenths of a point.
The second-engine intervals are the ones that run recorded.
The Decision Index heads were fitted on the datasets' training texts and scored on the benchmark's test requests.
These numbers use labelled examples, so they compare the registered server with the unregistered one, not with systems that see no labels.

## When it helps, and when it does not

- **Large option lists gain most.** On the two intent sets above, 77 and 150 options, heads from 10 examples per intent added 7 to 11 points of accuracy and 10 to 11 points of macro-F1 on the Decision Index.
  A long list of fine-grained labels is exactly where a generalist's reading of the label names is weakest.
- **A clear, well-calibrated question gains nothing, and the server says so.** On ToxicChat, a yes/no question the model already answers well, calibration was switched on in 0 of 100 draws; on SciFact, where the plain probabilities were off, it was switched on in 88 and gained 6.6 points (`runs/2026-09-29_tasks-endpoint/evaluation.json`, `r9`).
  Registering such a question costs nothing but the registration time: the declined correction leaves answers unchanged.
- **Heads are only fitted for 10 or more options.** Fitted at 5 examples per option on questions with few options, heads broke the never-hurts criterion on four tasks where calibration alone held (among them CUAD clause, -0.73 points on average, and FiQA, -0.62; `evaluation.json`, `head_any`), so the server fits a head only on option lists of 10 or more.

## What it costs

- **Registration**: every example is one scored request, about 0.30 s each in the default mode (0.21 s in second-engine mode), plus the fit; 120 examples take well under a minute on the served model.
- **Each later question**: in the default single-engine mode a question answered by a head takes about 80 ms more server time (126.8 against 46.5 ms), because the hidden state is read in three extra requests.
  Calibration and questions without a task cost nothing extra.
- **Second-engine mode** (`--head-engine`) answers head questions without the extra requests (82 ms on a question's first sight, as a plain question took on that run; 42 ms when the same question repeats) but holds a second copy of the model's weights, leaves no room for the image engine on the card, and its pairing with the default text-only class has not been timed yet (`docs/handoffs/tasks.md`).

## Limits

- **Option list identity.** A task is found by its question's type and its options, keys and descriptions, in order (plus the instructions for yes/no and score questions). Renaming, adding, removing or reordering an option makes a different task, which needs its own registration.
- **Order bound.** The head learns which answer position to trust, so it is bound to the option order it was fitted on, and it is not used with two-order averaging.
- **Options that change per input** cannot be registered: exam-style questions whose answers differ every time have no fixed list to learn, and registration refuses them.
- **One task per option list per server.** Registration is in memory per server instance; persist it with an export and `--tasks-file`, as above.
- **Single-engine or second-engine mode.** The default reads the hidden state from the serving engine at the latency cost above; `--head-engine` trades a second weight copy for speed. Both declare the same choice on every stored evaluation item (`tests/unit/test_head_modes.py`), and a task registered in one mode loads in the other.
- **The measured gains are on intent classification.** Your question's gain depends on how far the model's plain reading is from your labels; `examples/tasks/evaluate.py` measures it on your own held-out examples.
