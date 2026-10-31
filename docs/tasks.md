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
  - For the head: a question with 10 or more options, and at least 2 examples of every option. The measured gains below used 10 per option, which is what we recommend; see "A few options with few examples" under "When it helps, and when it does not" for what the floor of 2 gives.
- **Examples from your own traffic.** Past inputs your team has already routed or labelled are ideal. They should look like what the server will see: the same kind of text, the same mix of options where you can.
- **A few more to check the result.** Keep some labelled inputs out of registration; `examples/tasks/evaluate.py` measures accuracy on them before and after.

## Walk-through

The commands below use the example task in `examples/tasks/`, a synthetic ticket-routing question with 12 queues, and run from the repository root.
Install the package and start a server in one terminal:

```bash
uv sync --extra dev --frozen
uv run python -m decisio.serve.vllm_engine --backend hf --model Qwen/Qwen3-0.6B-Base
```

That is the CPU stand-in, which runs anywhere; on a GPU, install with `uv sync --extra serve --frozen` and start `uv run python -m decisio.serve.vllm_engine --model Qwen/Qwen3.6-35B-A3B-FP8`.
The server is ready when `curl -s http://127.0.0.1:8000/health` answers; run the steps below in a second terminal.

### 1. Build the examples file

Registration is one `POST /v1/tasks` whose body names the task and lists the examples.
Each example is an ordinary `/v1/systemone` request with exactly one question, plus the answer you want:

```json
{
  "id": "ticket-routing",
  "examples": [
    {
      "request": {
        "state": "My card keeps getting declined when I renew.",
        "questions": {
          "route": {
            "type": "choice",
            "instructions": "Which support queue should handle this ticket?",
            "criteria": {
              "billing": "Invoices, charges on the account, plan prices",
              "payment_failed": "A card or bank payment was declined or failed",
              "...": "..."
            }
          }
        }
      },
      "answer": "payment_failed"
    },
    ...
  ]
}
```

`examples/tasks/register.py` builds this body from a CSV with `text` and `label` columns and the question in `question.json`; `examples/tasks/examples.json` is the result for the example's 120 tickets.

### 2. Register

```bash
uv run python examples/tasks/register.py --id ticket-routing --data examples/tasks/train.csv
```

or, with the prepared body:

```bash
curl -s http://127.0.0.1:8000/v1/tasks -H 'Content-Type: application/json' -d @examples/tasks/examples.json
```

The server scores every example exactly as it would serve it, fits both corrections, and answers with the stored task.
Registering the same `id` again replaces the task.

### 3. Read the response

The response has one record per correction, `calibration` and `head`, each with:

- **`applied`**: whether the server will use it. `true` means later questions with this option list are answered with the correction.
- **`reason`**: why, in words. "cross-validated gain" when it was kept; otherwise, for example, that there were fewer than 10 examples, that some option had fewer than 2 examples (head), that the question has fewer than 10 options (head), or that cross-validation on your examples prefers the plain readout.
- **`short_options`** (head, when it was declined for short options): the options that have too few examples, in option order. The `reason` gives each with its count, so you can see which few more labels would let the head be fitted.
- **The cross-validation evidence**: the server split your examples into five parts and scored each part with a correction fitted on the other four.
  - For calibration: `cv_logloss_plain` and `cv_logloss_fitted` (lower is better), `cv_acc_plain` and `cv_acc_fitted`, and `cv_t`, the test statistic of the change in log loss (negative means the correction lowered it). Calibration is kept only if log loss falls by at least 0.005 per example, `cv_t` is at most -1.645 (a one-sided 95% test), and accuracy does not fall.
  - For the head: `cv_logloss`, the cross-validated log loss for each penalty strength tried and for no head (`"none"`); the head is kept with the best penalty (`lambda`) if that beats no head.
  - For a kept head, its **temperature**: `temperature`, and `temperature_cv`, the head's log loss (`log_loss`) and expected calibration error (`ece`, ten bins of equal size on the top probability) on the same out-of-fold examples, `without` the temperature and `with` it, and `n`, how many examples they cover. The head fits its examples tightly and is more confident than it is right; the temperature divides its scores before the softmax to correct that. It is fitted on the examples the head did not see (each of the five folds refitted at the chosen penalty), never on held-out examples. It does not change which option is the top one, so accuracy is the same; it changes the probabilities, and so which answers clear a certified threshold. A temperature near 1 says the head was already calibrated; above 1, it was too sure. What it did to the calibration of one task on both Gemma bases is in [What you get](#what-you-get).

- **`inputs_sha256`**, on both records: a digest of what registration read from the engine for your examples (the calibration's label log-scores; for a head, also the hidden states). Registering the same examples twice gives the same digest exactly when the server read them identically; if two registrations have different digests, a difference between their heads or thresholds comes from the reading, not from the fit. The reading of a Gemma base depends on what the server's cache held and, on some boxes, differs between server starts (`EVAL_CARD.md` section 6.5). A task you want to reproduce exactly is registered once and kept: export it (`GET /v1/tasks?full=1`) and import it (`POST /v1/tasks/import`); an imported task is exactly the stored one.

Where the head is kept, calibration is not stacked on top of it: the head already includes what calibration would correct.

**When a correction is declined**, nothing is lost: the question is answered as it was before registration.
Look at the reason:

- Too few examples: add examples, up to 20 for calibration or 10 per option for the head, and register again.
- Cross-validation prefers the plain readout: the model already answers this question as well as your examples can show. That is common for a clear yes/no question.
- Check that the labels are consistent. Two people labelling the same kind of input differently look like noise, and the server will decline.

### 4. Ask a question and see the task applied

Ask the question exactly as registered:

```bash
uv run python examples/tasks/ask.py "My card keeps getting declined when I renew."
```

The response header `x-decisio-tasks: ticket-routing` says the question matched the registered task, so the task's kept corrections answered it.
No header means the question matched no task: check that its question and options are exactly those you registered.
If the task kept neither correction, the header still names it and the answer is the plain one; `GET /v1/tasks` shows what each task kept.

### 5. List, export, reload, delete

```bash
curl -s http://127.0.0.1:8000/v1/tasks                        # registered tasks and each fit's summary
curl -s 'http://127.0.0.1:8000/v1/tasks?full=1' > tasks.json  # export, head parameters included
curl -s http://127.0.0.1:8000/v1/tasks/import -H 'Content-Type: application/json' -d @tasks.json
curl -s -X DELETE http://127.0.0.1:8000/v1/tasks/ticket-routing
```

**Without `--tasks-dir`, registration is per server instance, in memory.** A restart, or a second server behind a load balancer, does not see a task registered on another.
To keep a task, either start the server with `--tasks-dir` (below), or export it (`GET /v1/tasks?full=1`) and start each server with `--tasks-file tasks.json` or post the export to `POST /v1/tasks/import`.
### Saved to disk (`--tasks-dir`)

```bash
python -m decisio.serve.vllm_engine --tasks-dir /data/tasks --require-tasks 3 ...
```

With `--tasks-dir PATH` the server keeps every taught task in that directory, one JSON file per task, and loads all of them at start.
This covers both kinds: the calibration and head tasks (`POST /v1/tasks`) and the abstention tasks (`POST /v1/abstention/tasks`), which until now could not be loaded back except by hand-copying the listing.

- **Written through, before the response.** A registration, a certification or re-certification, an import and a removal are saved first (to a temporary file in the same directory, flushed and `fsync`ed, then renamed over the task's file, then the directory is `fsync`ed), and only then stored in memory and reported. A crash leaves the previous file or the new one, never half of one.
- **A write that fails fails the request.** The response is a 500 saying the task was not saved and the change was not made; the task is not registered (a replaced task keeps its previous version, a certificate keeps the previous certificate, a removal leaves the task). A multi-task import is not atomic: tasks saved before the failure stay.
- **One server owns a directory.** The server takes an exclusive lock on `<dir>/.lock`; a second server on the same directory does not start. Replicas behind a load balancer each need their own directory (or, later, a shared store); do not mount one directory into several.
- **A file that cannot be read stops the start**, naming it. Move the file away to start without it, or restore it; a damaged store must not turn into untaught answers silently.
- **`--require-tasks N`** refuses to start unless at least N tasks apply to this server: calibration tasks of this server's stack (model and rendering) and abstention tasks of its served name. Use it where an empty or lost store would be a mistake: a store volume that failed to mount gives a server that starts, answers, and has forgotten everything.
- **What the server says.** The start prints a `TASKS` line (the directory, how many tasks came from it and from the files, how many apply). `GET /v1/tasks`, `GET /v1/abstention/tasks` and `/health` (`systemone.task_store`) carry the same: `saved_to`, per kind `loaded`, `applied` and `other_stack` (or `other_model`), and `at_start`.
- **Another stack.** A task saved under another model or rendering loads and is kept, and is not applied (the same rule as for a task loaded from a file); an abstention task is applied only on the served name it was fitted for. A certificate made under another serving stack (engine, version, card, settings) is suspended (`serving_stack_changed`), exactly as after an import. Nothing is deleted from the directory because the stack changed, so a return to the old stack finds its tasks.
- **With `--tasks-file` or `--abstention-tasks`.** The directory is the store of record. A task of the file whose id the directory does not hold is saved into it at start (the file seeds the directory); a task whose id the directory holds stays as the directory has it, and the start says so (`file_tasks_left_as_saved`), because a restart must not undo a task taught since. To replace a task from a file, post it to `POST /v1/tasks/import` (or `POST /v1/abstention/tasks/import`) or delete it first.
- **Not saved:** the answer-mix monitors' running counts (they restart from the certificate's baseline at every start, and are per server anyway) and anything about requests. The monitor's baseline is part of the task.
- **Abstention tasks** now also have `POST /v1/abstention/tasks/import` and `DELETE /v1/abstention/tasks/{id}`; `GET /v1/abstention/tasks` is their export. They carry the served name they were fitted for.

### Replicas: a shared store (`--task-store`)

```bash
pip install 'decisio[task-store]'        # the image already has it
python -m decisio.serve.vllm_engine --task-store s3://my-bucket/decisio/prod --require-tasks 1 ...
python -m decisio.serve.vllm_engine --task-store s3://my-bucket/decisio/prod --task-store-readonly ...   # a replica that does not teach
```

Replicas behind a load balancer are separate servers; with `--task-store` they share their taught tasks through object storage (S3, and S3-compatible services with `--task-store-endpoint`; `gs://` and `az://` through the same client, `obstore`). Credentials come from the provider's usual chain or environment (`AWS_*`, instance or workload identity, `GOOGLE_APPLICATION_CREDENTIALS`), never from decisio's flags, and are not logged. It is an alternative to `--tasks-dir`, not an addition: a replica keeps its tasks in one of them.

- **What happens.** A task taught on any replica is fitted there exactly as on one server, then published as the exact fitted record (no re-fit anywhere). Every other replica checks the store's manifest every `--task-store-poll-seconds` (5 by default; an unchanged manifest costs one empty conditional GET), fetches what is newer, checks it against its digest, and applies it as an import does: nothing is scored again, so every replica serves identical numbers.
- **Each replica checks its own stack.** A task fitted under another model or rendering is kept and not applied; a certificate made under another serving stack (engine, version, card, settings) is suspended on that replica (`serving_stack_changed`). A fleet that wants certified answers everywhere runs one stack.
- **A write names the version it replaces.** Every task has a `store_version` (in its listing and in the registration response). A registration, certification, import or removal is checked against the store's version of that task, before the fit and again when it is published: by default against the version this replica serves, or the one the caller was shown (`x-decisio-replaces: N`). If the store holds another, the answer is **409** with `store_version` and `your_version`, and nothing changes: someone taught the task since. Read it again and send it again, or send `?replace=1` to replace whatever the store holds. Two replicas registering the same new task at once: one succeeds, the other gets 409. Different tasks never conflict.
- **Removals propagate** as tombstones (kept 7 days, so a replica that was away learns of them; one away longer drops tasks that are no longer in the store).
- **`--task-store-readonly`** makes a replica apply the store's tasks and refuse to teach: registration, certification, import and removal answer 409 before any example is read. Use it on every replica but one to keep the fitting load on a designated replica, and route the task-writing paths (`/v1/tasks*`, `/v1/abstention/tasks*` with POST and DELETE) to it at the load balancer. Without it any replica may teach; the version check is what keeps them from overwriting each other.
- **Start and failure.** A replica refuses to start if it cannot read the store, or if the store does not enforce conditional writes (it writes, rewrites and removes a probe object under `probe/`; a replica with `--task-store-readonly` skips that, as its credentials may be read-only). Use `--require-tasks N` as for a directory. If the store goes away later the replica keeps serving what it has applied: registration and removal answer 503 and the change is not made, `/health` stays 200 and `systemone.task_store` says `ok: false` with the last error and the age of the last sync, and `/metrics` carries `decisio_task_store_ok` and `decisio_task_store_last_sync_age_seconds`.
- **Reads are eventually consistent.** After a registration the other replicas serve the new version within the poll interval plus the fetch, and the old one until then; a client that must read its own write asks the replica that took it. Replicas that start later load everything at once.
- **Monitors stay per replica.** The answer-mix counts of a certified task are not shared; `/metrics` is per replica, so sum `decisio_task_*` across replicas in Prometheus. There is no fleet monitoring.
- **Housekeeping.** Each fitted version of a task is an immutable object under `objects/`; the manifest points at the current ones. Superseded objects are not deleted automatically (a lifecycle rule by age would delete tasks nobody has changed): `python -m decisio.serve.task_store_shared gc s3://bucket/prefix --older-than-days 1 [--dry-run]` removes the ones no entry points to.
- **Checked on** obstore's in-memory store (which enforces the same preconditions), and against a running RustFS and SeaweedFS (S3-compatible services) for the start-up probe, with a two-replica run on RustFS (`DECISIO_TEST_S3_ENDPOINT`, `tests/unit/test_task_store_shared.py`). Not run against AWS S3, R2, MinIO, GCS or Azure (MinIO's and R2's own images and endpoints were not available to the build); the start-up probe is what tells you whether yours enforces conditional writes.

A task is valid only for the model and rendering it was fitted under (the served name, the checkpoint, padding and the rendering flags): a task fitted on another configuration is not applied, and the server says which.
The description rule (`--describe-options`, on by default) is part of the task's key instead, and only for questions it changes: a question whose options have descriptions has another key with the rule on than off, so a task fitted under the other setting is simply not matched; a question it leaves alone (index keys, bare labels, yes/no, score) keeps its tasks either way.

**Abstention tasks come after the readout task.** An abstention threshold (`POST /v1/abstention/tasks`, a threshold on a "can't tell" option) is fitted on the probabilities the question is actually served with, which include a registered task's correction.
Register the readout task first and the abstention task second; if you register the readout task again, register its abstention task again too.

## What you get

Every number here is on the served model (Qwen3.6-35B-A3B-FP8 on one RTX PRO 6000 Blackwell, `EVAL_CARD.md`) and comes from the record named beside it.
In every intent measurement the test items are held out: none of them is among the registered examples.

| Measure | Without a task | With a task | Measured on | Record |
| --- | ---: | ---: | --- | --- |
| BANKING77 intent accuracy (77 options) | 0.747 | **0.840**; +9.3 points [+5.1, +14.0] | 150 test items; 10 labelled training examples per intent registered; mean of six draws; the default single-engine mode, the served prompt of 2026-10-03 (under the compact layout: 0.740 to 0.847, +10.7 [+5.8, +15.8], three draws) | `runs/2026-10-03_qiv-default/` (`heads6.json`, `intents_qiv.json.gz`) |
| CLINC150 intent accuracy (150 options) | 0.820 | **0.912**; +9.2 points [+3.5, +15.2] | 100 test items; as above (compact layout: 0.820 to 0.893, +7.3 [+2.3, +13.0]) | same |
| The same two, second-engine mode | 0.740, 0.820 | 0.849 (+10.9 [+6.0, +16.0]); 0.890 (+7.0 [+1.7, +12.7]) | the same items and three draws, the hidden state from a second engine; under the compact layout | `runs/2026-09-29_tasks-endpoint/manifest.json` |
| Decision Index BANKING77, macro-F1 | 0.729 | **0.841** | its 3,080 requests; heads registered from 10 training texts per intent; second-engine mode; the same server with and without; under the compact layout | `runs/2026-09-29_tasks-endpoint/decision_index/*/di_report.json` |
| Decision Index CLINC150+OOS, macro-F1 | 0.812 | **0.916** | its 5,500 requests; as above, plus 10 out-of-scope training texts | same |
| Per-task calibration, harm | | never hurts on six held-out tasks | 20 labelled examples, 100 draws; "never hurts": mean accuracy change at least -0.5 points, mean log-loss change at most +0.005, at most 5% of draws losing 2 points; measured change -0.14 to +0.28 points | `runs/2026-09-29_tasks-endpoint/evaluation.json` (`r10`) |
| Per-task calibration, gain | pooled ECE 0.058 | pooled ECE **0.037**; SciFact +6.6 points | six other held-out tasks, 20 labelled examples, 100 draws, 900 test items | same (`r9`) |
| Registration time | | about 230 s for 770 examples, 455 s for 1,500 (0.30 s per example) | the default single-engine mode | `runs/2026-09-30_plugin-verification/intent_heads/*.json.gz` (`register_s`) |
| | | about 160 s for 770 examples, 330 s for 1,500 (0.21 s per example) | second-engine mode | `runs/2026-09-29_tasks-endpoint/intent_heads/*.json.gz` |
| Server time per intent question | 46.5 ms | 126.8 ms | the default single-engine mode (three extra engine requests per head question) | `runs/2026-09-30_plugin-verification/manifest.json` (`latency.json.gz`) |
| | 82 ms | 82 ms on first sight; 42 ms when the identical question is asked again | second-engine mode, on that run's card: the median of each round, a question's first sight (82.35 ms with and without a task) and two repeat rounds (41.6, 41.8 ms), where the pooling engine reuses more of an identical prompt | `runs/2026-09-29_tasks-endpoint/evaluation.json` (`latency.head.server_ms_round_medians`) |

The intervals are 95% paired bootstrap intervals over test items (the gain averaged over the draws of examples).
Under the served prompt of 2026-10-03 they are computed the same way from that record's per-item rows (`heads6.json`), with the plain answer from the same server without the task.
The single-engine intervals are computed from the record's per-item rows by `benchmarks/intent_head_gain.py`, with the plain answer taken from the same forward pass as the head's, 10,000 paired resamples of the test items and numpy's `default_rng(0)`; the output is stored as `runs/2026-09-30_plugin-verification/derived/intent_head_gains.json`, and another seed moves the interval ends by a few tenths of a point.
The second-engine intervals are the ones that run recorded.
The Decision Index heads were fitted on the datasets' training texts and scored on the benchmark's test requests.
These numbers use labelled examples, so they compare the registered server with the unregistered one, not with systems that see no labels.

**The head's temperature, live, on the Gemma bases (decisio 0.12.0, 2026-10-15).**
On a 16-option task (which of 16 team rules covers a shell command; a synthetic set that is not published; 300 test items; one draw of the examples), a task registered with 200 or 1,000 labelled examples got a head served at a per-task temperature of 3.4 to 3.8 on Gemma 4 31B and 2.5 to 2.7 on Gemma 4 12B.
The tie-robust expected calibration error of its answers on the 300 held-out items was 0.040 to 0.061, against 0.143 to 0.168 for the same head served without the temperature, and the paired 95% interval of the difference excluded zero in all four cells (31B and 12B, 200 and 1,000 examples); accuracy changed by one point or less.
The record is `runs/2026-10-15_live-head-temperature/`.
Its caveats: one task, one draw and 300 test items, so the intervals cover the test items only and nothing here says the result holds for other tasks; "without" is an offline replica of the head decisio 0.11.1 served, close to it and not the same function (the live answers differ from the offline head at the server's own temperature by a median of 0.011 to 0.023 per item, with 2% to 7% of the top answers differing); and the same registrations certified little (the 31B nothing, the 12B at 1,000 examples and alpha 0.05 only), so a registration can honestly come back "not certified".

## When it helps, and when it does not

- **Large option lists gain most.** On the two intent sets above, 77 and 150 options, heads from 10 examples per intent added 7 to 11 points of accuracy and 10 to 11 points of macro-F1 on the Decision Index.
  A long list of fine-grained labels is exactly where a generalist's reading of the label names is weakest.
- **A clear, well-calibrated question gains nothing, and the server says so.** On ToxicChat, a yes/no question the model already answers well, calibration was switched on in 0 of 100 draws; on SciFact, where the plain probabilities were off, it was switched on in 88 and gained 6.6 points (`runs/2026-09-29_tasks-endpoint/evaluation.json`, `r9`).
  Registering such a question costs nothing but the registration time: the declined correction leaves answers unchanged.
- **A few options with few examples.** A head needs at least 2 examples of every option. With a higher floor, two or three rare options out of 77 switched the head off for the whole task.
  Measured offline, with the server's own fitting code, on the stored Gemma 4 readouts of ten decision sets (seven with 10 or more options, so 42 cells that can have a head: two model sizes, 12B and 31B, at 50, 200 and 1,000 labelled examples), the floor of 2 against a floor of 5 (`runs/2026-10-30_intent-head-floor`):
  - macro-F1 rose by 1.08 points on average; 10 cells rose, 30 did not change, 2 fell, by 0.39 point at most;
  - BANKING77 at 1,000 examples, macro-F1 73.4 to 81.2 on the 31B and 69.9 to 80.7 on the 12B (accuracy 73.8 to 81.5 and 71.0 to 81.2), and the share of answers that certify at `alpha` 0.05 went from 0 to 62.3% on the 31B (12B: 0 to 0);
  - a 27-intent customer-support set at 200 examples, macro-F1 78.2 to 89.7 on the 31B and 77.0 to 90.5 on the 12B;
  - in the other 38 cells the changes ran from -0.4 to +0.9 point, and no certificate was violated where a floor of 5 had kept its promise.
  The gain comes from a few cells, mostly two sets; a task with no rare options is unchanged. The head is still kept only if its cross-validation prefers it, and a task with an option that has no example still serves calibration.
- **Heads are only fitted for 10 or more options.** Fitted at 5 examples per option on questions with few options, heads broke the never-hurts criterion on four tasks where calibration alone held (among them CUAD clause, -0.73 points on average, and FiQA, -0.62; `evaluation.json`, `head_any`), so the server fits a head only on option lists of 10 or more.

## What it costs

- **Registration**: every example is one scored request, about 0.30 s each in the default mode (0.21 s in second-engine mode), plus the fit; 120 examples take well under a minute on the served model.
- **Each later question**: in the default single-engine mode a question answered by a head takes about 80 ms more server time (126.8 against 46.5 ms), because the hidden state is read in three extra requests.
  Calibration and questions without a task cost nothing extra.
- **Second-engine mode** (`--head-engine`) answers head questions without the extra requests (82 ms on a question's first sight, as a plain question took on that run; 42 ms when the same question repeats) but holds a second copy of the model's weights, leaves no room for the image engine on the card, and its pairing with the default text-only class has not been timed yet (`docs/handoffs/tasks.md`).

## Certified answers

A calibrated probability says how sure the model is. It does not say how often the model is right when it is that sure, on your data.
A certified answer does: it is an answer the server has checked against your own labels, at a confidence level where, on examples it set aside, it was almost always right.

### What certified means

When you register a task with a `certify` block, the server does four things:

1. It **sets aside some of your labelled examples** (half by default) and fits the task's calibration and intent head on the rest. The held-out examples are never used to fit anything.
2. It reads those held-out examples through the task as the server will serve them (with the head's temperature, which is fitted first, on the fit part), and looks at how often the answer is wrong at each confidence level.
3. It picks the **lowest confidence threshold** at which it can say, with a stated confidence, that the wrong answers above it are at most `alpha` of them.
4. It stores the threshold, the numbers behind it and the split, and **from then on every answer for that task says `certified: true` or `certified: false`**: true when the answer's top probability is at or above the threshold, false otherwise, with the reason.

The promise, in full: **with probability at least 1 - `delta` over the draw of your examples (90% by default), at most `alpha` of the answers marked certified are wrong (5% by default), for traffic like your held-out examples, read through the serving stack the certificate was made on.**
Three things it is not.
It is a rate over many answers: a single certified answer can still be wrong.
It is not a statement about other traffic, another task or another server.
And it is not free: the guarantee is paid for in the share of answers that get certified, and with too few examples that share is zero.

If no threshold holds, the server says so plainly, `certificate.threshold` is `null`, and the task works exactly as before, with every answer marked `certified: false` and `certified_reason: "nothing_certified"`.

### A worked example

The run behind this example is `runs/2026-10-14_certified-answers/results/mac_mlx/`: the Gemma 4 12B (6-bit, MLX) on a Mac, on a three-queue ticket question (billing, shipping, access) with some tickets mentioning a second queue after the first (the label is the first), with 400 labelled tickets the script writes itself.
Register the task and ask for a certificate at `alpha` 0.15 and `delta` 0.1:

```json
POST /v1/tasks
{"id": "ticket-queue",
 "examples": [{"request": <a /v1/systemone request with one ticket and the three queues>, "answer": "billing"}, ... 400 of them],
 "certify": {"alpha": 0.15, "delta": 0.1}}
```

The server holds out half of the tickets, fits on the rest, reads the held-out ones as it would serve them, and answers with the certificate:

```json
{
 "certificate": {
  "id": "ticket-queue#c2e4cc6e431b",
  "alpha": 0.15,
  "delta": 0.1,
  "n": 200,
  "accuracy": 0.84,
  "threshold": 0.9863,
  "certified": true,
  "certified_share": 0.845,
  "certified_n": 169,
  "errors_at_threshold": 17,
  "split": {
   "method": "holdout",
   "holdout": 0.5,
   "fit_examples": 200,
   "certify_examples": 200,
   "disjoint": true
  },
  "summary": "Answers whose top probability is at least 0.9863 are certified: on the 200 held-out examples that is 84% of the answers, 17 of 169 wrong. With probability at least 90% over the draw of such examples, at most 15% of the certified answers are wrong, for traffic like them and on this serving stack."
 },
 "status": {
  "state": "ok",
  "text": "ok"
 }
}
```

On those 200 held-out tickets the threshold 0.986 keeps 169 of them (84%) with 17 wrong.
The promise is that, with probability at least 90%, at most 15% of the answers above the threshold are wrong.
From then on each answer for the task says whether it is certified:

```json
[
 {
  "ticket": "Urgent: the refund has not reached my bank account. Can you check toda...",
  "choice": "billing",
  "top_probability": 1.0,
  "certified": true,
  "certificate": "ticket-queue#c2e4cc6e431b"
 },
 {
  "ticket": "Hey, I need a copy of the invoice for my subscription. Also: the deliv...",
  "choice": "shipping",
  "top_probability": 0.9855,
  "certified": false,
  "certificate": "ticket-queue#c2e4cc6e431b",
  "certified_reason": "below_threshold"
 }
]
```

The script then asked 200 further tickets (none of them among the 400): 169 of the answers were certified and 19 of those were wrong (11.2%); 31 were not certified.
That is one draw of tickets, not a test of the promise: the promise is about a rate over many answers.


### How many examples it takes

The server refuses a certification, and registers nothing, below the fewest held-out examples at which the procedure could certify anything at all, which depends only on `alpha` and `delta`:

| `alpha` | `delta` 0.1 (the default) | `delta` 0.05 |
| ---: | ---: | ---: |
| 0.02 | 149 | 183 |
| 0.05 | 59 | 72 |
| 0.1 | 29 | 36 |
| 0.2 | 14 | 17 |

That is the floor, for a task whose every answer is right. The number you need is larger, and depends on how accurate the model already is on your question and on how well its confidence tells its right answers from its wrong ones.
The table below is decisio's own certification run on the recorded answers of the Gemma 4 31B on the 1,400-item suite, task by task, drawing the certification examples 1,000 times (`runs/2026-10-14_certified-answers/`; `delta` 0.1).
"Promise broken" is the share of draws in which the threshold's true error among that task's items was above `alpha`; "no bound" is the most lenient threshold whose error on the examples themselves is at most `alpha`, which is what a certificate without the statistical bound would give:

| Task (recorded 31B answers) | Items | Accuracy | Certification examples | `alpha` | Median share certified | Draws with nothing certified | Promise broken, with the bound | Promise broken, no bound |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| boolq | 150 | 0.887 | 75 | 0.05 | 0.00 | 100% | 0.0% | 47.0% |
| boolq | 150 | 0.887 | 75 | 0.1 | 0.00 | 97% | 0.1% | 41.5% |
| banking77 | 150 | 0.787 | 75 | 0.05 | 0.00 | 100% | 0.0% | 56.8% |
| banking77 | 150 | 0.787 | 75 | 0.1 | 0.00 | 100% | 0.0% | 48.8% |
| toxicchat | 150 | 0.973 | 75 | 0.05 | 0.00 | 94% | 0.0% | 0.0% |
| toxicchat | 150 | 0.973 | 75 | 0.1 | 0.99 | 26% | 0.0% | 0.0% |
| mmlu | 150 | 0.887 | 75 | 0.05 | 0.00 | 100% | 0.0% | 36.3% |
| mmlu | 150 | 0.887 | 75 | 0.1 | 0.00 | 97% | 0.0% | 35.1% |
| mmlu_pro_text_x350 | 350 | 0.700 | 75 | 0.05 | 0.00 | 100% | 0.0% | 78.8% |
| mmlu_pro_text_x350 | 350 | 0.700 | 75 | 0.1 | 0.00 | 100% | 0.0% | 59.0% |
| mmlu_pro_text_x350 | 350 | 0.700 | 175 | 0.05 | 0.00 | 100% | 0.0% | 69.0% |
| mmlu_pro_text_x350 | 350 | 0.700 | 175 | 0.1 | 0.00 | 98% | 0.8% | 49.4% |

Read it as a warning about small example sets.
With 75 examples the bounded procedure certified nothing on four of the five tasks at `alpha` 0.05 or 0.1, and on ToxicChat (answered 97% right) it certified nearly everything at 0.1 in three draws out of four, and nothing at 0.05 in 94% of the draws.
The rule without the bound would have certified more and broken its promise on 35 to 79% of the draws on every task but ToxicChat; the bounded procedure broke it at most once in a hundred draws, well inside `delta`.
With 175 examples the task answered 70% right still certified nothing at either `alpha`.
How many examples your task needs depends on how accurate the model is on it and on how well its confidence tells its right answers from its wrong ones; the stored `certified_share` tells you what yours gave, and more examples certify more.
Set `certify.min_examples` above the floor if you want the server to refuse a certification that thin.

### Where it stops

- **Per task.** The certificate belongs to one task, its option list and its question. A threshold from one task means nothing on another.
- **Per serving stack.** The server stores a fingerprint of the stack it certified on: the decisio version, the base and checkpoint revision, the engine and its version, the card, the quantisation, the KV cache dtype, the context length served (`max_model_len`; on the 31B and the 12B it follows the card unless `--max-model-len` is given, [`running.md`](running.md#the-context-length)), the attention backend, a hash of the token rows the server builds for a fixed probe question (the tokenizer, chat template and rendering), the temperatures, the serving flags (orders, multi-question mode, padding, `--noul-commit`) and the task's own fit. **If any of it differs when an answer is served, every answer of the task is `certified: false` with the reason `serving_stack_changed`, and the task's status reads `suspended: serving stack changed`**, listing what differs. The calibration still applies; only the certification stops.
  Re-certify with an explicit call, `POST /v1/tasks/{id}/certify`, on new labelled examples that share no input with the fit.
  The server cannot know whether a changed stack kept the promise: a new version may leave a certificate as good as it was, and the only way to know is labels.
- **Same stack is not always the same answers.** A fingerprint cannot see the hardware's own variation. Two fresh servers on the same stack can give slightly different probabilities, which moves answers near the threshold across it; the Gemma 4 12B in particular does not repeat from one fresh server to the next (`EVAL_CARD.md` section 6.5). The stored Gemma 4 31B did not repeat in one measurement either: on a 776-item read every item moved between its two fresh starts (a median of 0.0098 in the largest option probability) and 24 changed their top answer, where in an earlier session the same pair was identical, and about 4 percent of the items near a 0.90 threshold were certified on one start and not the other, with no certified answer changing its option (`runs/2026-10-17_s18-f-restart/`, one pair on one box, cause not known). The rate promise is about the examples as read through one server. `runs/2026-10-14_certified-answers/` has a restart check: on the CPU stand-in and on a Mac, a fresh process certifies the same threshold and marks every answer the same way; the 12B on a card restarted with the same threshold, certificate id and stack digest and no certified flag changed, but it certified nothing at alpha 0.05 (no threshold), so no certified answer was exercised (`runs/2026-10-14_s11-session/K/`); the measurement of 2026-10-15 (the restart check) ran the same check on the 12B at `alpha` 0.15, where a threshold exists, and it held under its pass rule: the threshold, certificate id and stack digest were equal across the restarts and the re-imports, every import came back `ok`, and no certified answer differed (threshold 0.9912, 166 of 200 questions certified, 10.2% of the certified answers wrong; the threshold, certificate id and stack digest were equal in all five comparisons and no choice or flag differed; `runs/2026-10-15_s12-session/K2/`).
- **A repeated request can differ from its first read.**
  A request about a state the server has read before is computed on top of the blocks the prefix cache kept, and that arithmetic differs slightly from reading the same request cold.
  On the Gemma 4 12B in FP8, a second read of 300 hard choice questions in the same start (three or more options, a median prompt of 460 tokens) matched the first read bit for bit on 296; four differed, by up to 0.06 in a probability, and one top answer changed in one of two such second reads.
  A check of 500 questions on other sets, also on the 12B, found a largest difference of 0.088 and one top answer in 500.
  Only requests whose state is long enough to reach into the cache differ (about 100 to 130 tokens and more), consistent with the cache's 64-token blocks; a short request is bit-exact.
  It is a different effect from two fresh servers giving different probabilities, and it does not need a restart: it appears within one server, for a state the server has already read.
  It matters for a certificate in the same way: a certificate is made on first reads of new states, and a later request about a state already in the cache can come back slightly different from the confidence it was certified on, which moves answers near the threshold across it.
  It was measured on the 12B in FP8 on one card and one session; it was not measured on the 31B (a check on the 31B on a fresh server, with an empty cache, read bit for bit the same on 500 of 500 requests, which says nothing about cached reads).
  On the vLLM server, where the prefix cache is always on, no setting removes it.
- **Scored as it was certified.** A certificate is made on confidences scored one question at a time, and a question of a certified task is served the same way: by the function that scored the examples, alone, in request order, never batched with the other questions of its request, whatever the load (on a server with `--concurrency batched` too: below). So on one stack the confidence served for an example is the one the certificate was made on. This costs the request's certified questions the shared-prefix saving (each is a read of its own), and uncertified questions of the same request are scored as before. For a task with an intent head the confidence is the hidden-state read alone, at registration and in serving: registration used to read each example plain first and then hidden, and on vLLM that second read differs slightly from the same read made without the first (one measurement certified 153 answers where serving put 154 above the threshold, the same whether read one at a time or 16 in flight). At registration the server re-reads a tenth of the held-out examples (20 to 200) through the serving path, one request each, and the certificate records `served_check`: the largest and 99th-percentile difference in confidence, how many examples changed sides of the threshold and how many changed their top answer. On one stack it reads zero when the re-read finds the state as the registration did; a state the engine's prefix cache still holds is read warm, and on vLLM a warm read can differ slightly from a cold one, so a small nonzero value does not by itself mean that new traffic is served differently. A nonzero value is said in the certificate's summary. It is a check on the sample, not a proof for every request, and it does not cover a fresh start of the same stack (the previous point).
- **Thinking.** A question of a registered task is always answered at one read, whatever the server's or the request's `think`. The server's think setting is part of the serving-stack fingerprint (only when it is on): a certificate made one-pass is `certified: false` with the reason `think_mode_changed` on a server that thinks, and the other way round, until the task is certified again.
- **Concurrency.** On a server with `--concurrency batched` a certified task's questions, and its registration and certification reads, still take the serial path: one request at a time, never sharing an engine call with other traffic, so the promise above holds there too. The mode is part of the serving-stack fingerprint when it is `batched`: a certificate made on a serial server is `certified: false` with the reason `concurrency_changed` on a batching server, and the other way round, until the task is certified again ([running.md](running.md#concurrency)).
- **Whole inputs.** A request answered from a truncated state (`over_limit: "truncate"`, [api.md](api.md)) is never certified: every answer of a certified task in it is `certified: false` with the reason `state_truncated`, whatever its confidence, and the monitor below does not count it. Registration and certification never truncate: an example over the ceiling is refused.
- **Traffic like the examples.** The held-out examples must look like what the server will see: the same kind of text and the same mix of options. The monitor below says when the traffic moves away.
- **What is certified.** Choice and yes/no questions, on text requests, in the single-order form the certificate was made on (a request that asks for two orders is not certified). Score questions are not certified: an ordinal answer has no single wrong answer. A question that matches no task, or an image request, carries no certification fields.

### The monitor

A certified task is watched by three checks that need no labels.

- **The serving-stack fingerprint**, above: any change suspends the certificate. `GET /v1/tasks/{id}` shows the status and what changed.
- **The confidences.** After every certified answer the server updates a CUSUM on how often the answer's top probability falls below the 5th percentile of the certification examples' (a count and a number, never the question, the state or the option names). It treats up to 1.3 times the examples' share as normal, a share of 2.5 times as worth an alarm, and its limit is the smallest at which a stream at the normal share alarms with probability at most 1 percent in 5,000 answers, simulated at certification with a fixed seed. When it crosses the limit the task's status reads `traffic_changed` with `confidence` among its signals, and it clears when the statistic has fallen to half the limit (about 300 normal answers after the traffic returns). **By default this suspends certification:** every answer of the task is `certified: false` with the reason `traffic_changed` while the alert is on, because the promise is for traffic like the examples and the monitor says it is not; the answers themselves do not change. A certify block with `"suspend_on_alert": false` only reports. A task certified before this signal existed has no confidence statistic and runs the answer mix alone until it is certified again; the certify call that makes the statistic takes the default, `suspend_on_alert` true, when the old certificate names none.
  On the replay of a CLINC150 task (the stored 31B, 10 examples per intent; 5,363 messages, out-of-scope messages rising from message 1,000 to 30%, which the task has no option for), on the card, under 0.14.0 the answer mix first alerted after message 5,200 with 1,027 wrong certified answers served before it, and under 0.16.0 the confidence signal first withheld an answer at message 1,570 after 40 wrong certified answers (the offline replay of the recorded reads, in `runs/2026-10-17_drift-monitor/`, alerted at message 1,697 after 55). On the 4,500 in-scope messages alone, and on the test split of three BANKING77 draws read through their tasks, it did not alert (`runs/2026-10-17_drift-monitor/`). The thresholds were chosen on the CLINC150 stream, so the first test on traffic it was not chosen on is the card cell that replays a second stream.
- **The answer mix.** The server counts, for each certified task, how often each option was chosen in the last 500 answers, as counts only, and compares the mix with the certification examples' every 100 answers once the window is full. When the distance is above a line fixed at certification, the task's status reads `traffic_changed` with `answer_mix` among its signals. This says the traffic is no longer like the examples (a new channel, more out-of-scope messages, a launch); **by itself it does not suspend the certificate or change any answer**. It is weak on a task with many options, where a shift spread over them barely moves the mix (the CLINC150 replay's distance sat at 0.19 to 0.28 against a line of 0.28 until message 5,200), and it alerted on normal traffic from another split at its first check on all three BANKING77 draws, whose examples' mix is the training split's and whose traffic was the test split's.
  The state is in `GET /v1/tasks` and `GET /v1/tasks/{id}` (`status`) and in `GET /metrics`, Prometheus text: `decisio_task_state{task,state}`, answers and certified answers, `decisio_task_mix_distance` and `decisio_task_mix_line`, `decisio_task_confidence_statistic`, `_limit` and `_alerts_total`. The counters are in memory and start again with the server.

**A task with no option for a message that fits none of its labels** (no "other", "none of the above" or "out of scope") gets a wrong answer for every such message, and that answer can be certified. Registration says so in the certificate (`warnings`, and its summary) when a choice task has no option whose key says it. Adding one is the remedy; the monitor only tells you after the fact.

The monitor cannot see a wrong answer, because it has no labels. The only direct check that the promise holds on live traffic is to label some of it and certify again.

## Tasks whose options change with every request

A fixed-option task is one question with one list of options. Some recurring questions are not: which brand a product title names, which entity a mention refers to, which of this request's tools to call, which catalogue product matches a search. Their options are different in every request, so there is no option list to calibrate and no intent head to fit.
What carries over from one request to the next is the question's instructions, and how the right option relates to the text and to what decisio already thinks of it.
Register such a task with `"options": "dynamic"`:

```json
{"id": "brand",
 "options": "dynamic",
 "examples": [{"request": {"state": "Apple iPhone 13 Pro 256GB graphite",
                           "questions": {"q": {"type": "choice",
                                                "instructions": "Which of these is the brand of the product in this listing?",
                                                "criteria": {"1": "Apple", "2": "Samsung", "3": "Google", "none": "none of these"}}}},
               "answer": "1"}, ...],
 "certify": {}}
```

- **Matched on the instructions.** Every later choice question with the same instructions is answered through the task, whatever its options, unless a fixed-option task registered for exactly that set of options matches it first.
- **Each example carries its own options.** Its `answer` is one of its option keys, or `null` for the request's none option (the option with the key `none`, or the key you name in `"none_option"`).
- **A certificate is part of it.** A dynamic task is always certified, with a `certify` block ([Certified answers](#certified-answers), 59 held-out examples at the defaults): the certificate is made on decisio's top probability, whatever the options in the examples were. Without it a dynamic task does nothing.
- **A re-ranker, where it helps.** The server also fits one weight vector over 13 features of an option, computed from the request itself: decisio's probability for it, whether it is decisio's top, how much of the option occurs in the text and where, its length, how often it was the answer in your examples, and whether it appeared in them at all. It is kept only when its 5-fold cross-validation beats decisio's own distribution, the way the intent head is (in the measurement below it was kept in 14 of 15 draws on the 12B and 10 of 15 on the 31B); the certificate is then made on its top probability. It needs no encoder and adds well under a millisecond. The registration response says whether it was kept (`dynamic.applied`) and why.
- **What is stored.** The 13 weights and the table the re-ranker uses of your examples' options: for each option text a hash and two counts. No request text and no option text is stored, and the registration response says so.
- **The monitor.** The traffic monitor of such a task has no answer mix, since the options differ. It watches the confidences as before. It also watches the share of requests none of whose options your examples ever contained, but only where options recur: when more than one in five of the examples' own requests has only never-seen options (a task whose options are new almost every time, such as entity linking or product matching), a rise cannot be told from normal and that signal is not armed. `status.monitor.unseen_options` says which: `armed: true` with its statistic and limit, or `armed: false` with the share and why. Where it is armed and moves, what the re-ranker learned from your options stops meaning what it meant, and certification is suspended while the alert lasts (`suspend_on_alert`, as for the confidences).

**What was measured.** Offline, with the server's own code, on five recurring questions (brand from a product title, entity linking, tool choice, product matching, intent routing; 1,200 teaching, 800 certification and 1,000 test requests each, 200 labels used for the fit, two to eight options and a none option, `alpha` 0.05, `delta` 0.1), on the Gemma 4 12B and 31B (`runs/2026-10-30_dynamic-options`):

| | 12B | 31B |
| --- | ---: | ---: |
| decisio alone: accuracy | 80.3 | 83.2 |
| decisio alone: certified share / error among them | 53.3 / 3.3 | 54.4 / 3.3 |
| with the re-ranker as shipped: accuracy | 84.1 | 85.0 |
| with the re-ranker as shipped: certified share / error among them | 55.2 / 3.9 | 54.9 / 3.2 |

All in percent, the mean over the five questions of the median over three draws of 200 labels. Where decisio is already strong (brand, tool and intent routing at 91 to 96% accuracy) most requests are certified with no teaching at all; where it is weaker (entity linking at 57 to 67%, product matching at 63 to 67%) the re-ranker adds 3 to 10 points of accuracy and nothing is certified, because 5% error is out of reach there. Using 1,000 labels instead of 200 added little accuracy.

**Limits.**
- Choice questions only; a yes/no or a score question is not a dynamic task.
- The text route, one order per request (`orders` 1); a request of two orders is answered as before and not certified.
- The features read the option text as the request shows it, so options that differ only in a description (a tool's name and its description) are read as written.
- The features are string overlaps with the request's text, so the re-ranker helps where the right option can be seen in the text, or has been the answer before; it does not help where neither is true, and the server then declines it.
- The gate is itself slightly optimistic: the features are standardised on all the labelled examples, and the cross-validation that picks the regularisation is the one the gate reads. It declines a re-ranker whose gain is below 0.005 in log loss per example, which on one question (intent routing on the 31B) gave up a small real gain.
- The gain rests on five question types and two model sizes; a task of another kind may gain more or nothing, and the cross-validation says which.


## A fast table in front of decisio (opt-in)

For a task of short texts that arrive in volume, a small classifier on the host CPU can answer most requests without asking the model.
It is off unless the registration asks for it, and it keeps itself only where it pays.

**What it is.**
A sentence encoder (`BAAI/bge-base-en-v1.5`, a pinned revision, run by ONNX Runtime), a multinomial logistic regression fitted on the task's own labelled examples, and a confidence threshold: a small learned classifier on the server's CPU, not a lookup of earlier answers.
A request whose table confidence reaches the threshold is answered by the table and the model is not asked.
Every other request goes to decisio as it would without the table.

**Where it helps, and where it does not.**
It helps on short, command-style texts with about a thousand labelled examples: an assistant's commands, spam, a fixed set of intents.
It does not help on long texts.
The encoder runs in front of every request, on the host CPU.
With the torch encoder of 0.37.0 that was 21 to 30 ms for a short text and 134 to 183 ms for a text of 600 to 1,150 tokens (the table below), which is more than decisio's own read.
ONNX Runtime (0.38.0) gives the same embeddings to about 1.5e-6, and is 2.5 times faster on one short text (4.6 ms against 11.6 ms for the whole table path, median, on an Apple M5 Pro at 4 threads); the encoder is about 99% of that path.
A table made on 0.37.0 serves the same answers and confidences (to about 2e-6).
A table registered on 0.38.0 is fitted on embeddings that differ by about 1e-6; on the eight sets measured its threshold moved by at most 0.0015 and its certified share by at most 0.3 points, with accuracy unchanged.
That ratio was not measured on the x86 server CPUs, and the registration times the encoder it will serve on the host it runs on.
It does not help where decisio is weak enough that nothing is certified, and it does not help on a task whose options differ from request to request.
The registration times both paths on your own examples and drops the table when it does not pay (below), so asking for it on the wrong task costs the registration, not the traffic.

**How to ask for it.**

```bash
pip install 'decisio[table]'     # onnxruntime and scikit-learn; the image does not carry them
```

```json
{"id": "assistant-commands", "examples": [...], "certify": {"alpha": 0.05, "delta": 0.1}, "fast_table": true}
```

- A `certify` block is required: the table is part of the certificate.
- Choice questions only, with at least 100 fit examples.
- Without the extra the registration is refused with 422 and says what to install.
- The encoder's two files (the model's published ONNX export, 436 MB, and its tokenizer) are fetched from the Hugging Face Hub at the pinned revision on first use, and the export is checked against its pinned sha256. A server with no route to the Hub needs them fetched first (`hf download BAAI/bge-base-en-v1.5 onnx/model.onnx tokenizer.json --revision a5beb1e3e68b9ab74eb54cfd186867f64f240e1a`), or the registration is refused naming that command.

**What the registration does.**

1. It fits the table on the fit examples, and embeds the held-out examples one at a time.
2. It reads the held-out examples through decisio, one at a time, as the certificate does.
3. It makes the cascade's thresholds (below).
4. It times the two paths on the held-out examples, on this host, one request at a time, with decisio's reads of texts it has not seen: the expected mean time per request is the table's time on every request plus decisio's time on the share the table defers.
5. It keeps the table only if the expected time falls by at least 25% and a table threshold holds.

The response says what it did.
`fast_table_report` has `kept`, the `reason` when it is false (for example "it would save 11% of the time per request, under the 25% required"), the `guard` (the table's and decisio's mean milliseconds, the table's share, the saving) and the cascade's thresholds.
A table that is not kept leaves a task that is served, and certified, exactly as it would be without one.

**The certificate is the cascade's.**
The table's threshold is the lowest at which at most `alpha` of its answers are wrong, with probability at least 1 - `delta`/2, on the held-out examples.
Decisio's threshold is made the same way on the requests the table defers.
An answer of either stage that reaches its threshold is certified, and the promise is the cascade's: at most `alpha` of the certified answers are wrong, for traffic like the examples and on this serving stack.
The table's stage keeps that promise by the procedure's proof, with probability at least 1 - `delta`/2.
Decisio's stage is certified on the requests the table's chosen threshold defers, from the same held-out examples, so which requests those are depends on the same labels that set the table's threshold.
Its promise is therefore supported by simulation and not proved: on simulated populations, including one where the table fails exactly where decisio finds requests easy, it broke its allowance in at most 1.3% of draws against the 5% allowed.
A stricter mode that certifies the two stages on disjoint examples is possible later.
The certificate lists both stages (`table_threshold`, `threshold` for decisio's, and per stage the answers, the errors and an upper bound), and `decisio_only` holds the certificate decisio alone would have had, which is what serves when the table is off.
Each answer says which stage answered it, `answered_by: "table"` or `"decisio"`.
The cascade as a whole broke its promise in about 2 to 3 of every 100 simulated draws of the held-out examples, against the 10 allowed (`tests/unit/test_fast_table.py`), because each stage is held to `delta`/2.
That costs examples: the held-out set needs to be larger than for decisio alone (about 600 or more, at the defaults).

**What was measured.**
Offline, with the server's own code, on public sets at 1,000 labels (the 31B and the 12B; the table on a server CPU with four threads; decisio on one card; medians of two draws of the labelled examples):

**Across the ten sets tried, the table passed the 25% guard on 4: CLINC150, MASSIVE, Snips and SMS spam.**
**Without the three assistant-command sets (CLINC150, MASSIVE, Snips) it passed on 1 of 7: SMS spam.**
Six sets were dropped: CFPB and arXiv (long texts, where the table doubles the time per request), Civil comments (the table answers 70% but its time does not pay), SciCite (nothing certified), and the two sets whose options change per request (brand, entity linking; nothing certified).
MASSIVE passes only because decisio's own read there goes through the intent head's hidden-state read and is slow; timed as a plain read it saves 21% on the 31B, under the 25%.

| Set | Table answers | Of those, wrong | Time per request, decisio alone / with the table (31B) | Kept |
| --- | ---: | ---: | ---: | --- |
| CLINC150 (150 intents) | 92.3% | 3.1% | 384 / 71 ms (82%) | yes |
| MASSIVE (55 intents) | 45.2% | 2.2% | 310 / 190 ms (39%) | yes |
| CFPB complaints (long) | 0.0% | - | 154 / 315 ms (-104%) | no |
| arXiv abstracts (long) | 0.0% | - | 184 / 370 ms (-101%) | no |
| Civil comments (noisy labels) | 69.8% | 3.7% | 65 / 74 ms (-13%) | no |
| SMS spam | 99.8% | 1.4% | 62 / 32 ms (49%) | yes |
| Snips (7 intents) | 100.0% | 2.1% | 62 / 24 ms (62%) | yes |
| SciCite (citation intent) | 0.0% | - | 65 / 113 ms (-74%) | no |
| Brand from a product title (options change per request) | 0.0% | - | 63 / 95 ms (-53%) | no |
| Entity linking (options change per request) | 0.0% | - | 73 / 162 ms (-120%) | no |

The saving is largest where decisio's own read is expensive (CLINC150 and MASSIVE use the intent head, whose hidden-state read costs 190 to 380 ms) and where the table answers nearly everything.
On short texts with a plain read it needs the table to answer roughly 60 to 75% of requests to clear 25%.
These are offline figures on one card and one CPU; your hosts decide your own, which is why the registration measures them.

**What the table is, and what a request costs.**
The table is a small learned classifier: a sentence encoder followed by a logistic regression, both run on the server's CPU, in front of the model.
It is not a lookup of earlier answers, and it uses no GPU.
It has to run on every request to find out whether it is sure, so its time is paid by every request, and the model's time is paid on top of it by the requests it hands on.
A request the table answers costs the table's own time; a request handed to the model costs the table's time plus the model's.
Per request, on a server CPU with four threads (an AMD EPYC 9755; 200 timed texts per set; milliseconds; the model's time is its mean per request on the same texts, 31B, one card):

| Set | Table's own time: median | mean | p95 | Model's time alone (mean) | A request handed to the model: table plus model |
| --- | ---: | ---: | ---: | ---: | ---: |
| CLINC150 (150 intents) | 20.7 | 21.1 | 26.0 | 384 | 405 |
| MASSIVE (55 intents) | 20.6 | 20.5 | 26.6 | 310 | 330 |
| CFPB complaints (long) | 134.1 | 160.3 | 356.7 | 154 | 314 |
| arXiv abstracts (long) | 182.5 | 185.5 | 291.7 | 184 | 370 |
| Civil comments (noisy labels) | 48.6 | 59.3 | 131.9 | 65 | 124 |
| SMS spam | 29.7 | 31.6 | 48.7 | 62 | 94 |
| Snips (7 intents) | 23.0 | 23.6 | 29.6 | 62 | 86 |
| SciCite (citation intent) | 44.4 | 47.8 | 72.5 | 65 | 113 |
| Brand from a product title (options change per request) | 31.5 | 32.9 | 50.2 | 63 | 96 |
| Entity linking (options change per request) | 91.2 | 88.2 | 103.6 | 73 | 161 |

On CLINC150 a request the table answers takes **21.1 ms** (median 20.7 ms), and a request handed to the model takes about **405 ms**: 21.1 for the table and 384 for the model's read (the intent head's hidden-state read).
The 71 ms in the table above is the mean over the 200 timed requests, of which 26 (13%) were handed to the model: 21.1 + 0.13 x 384 = 71.
Those 200 requests happened to hand on more than the whole test set did: on the 1,000 test items the table answered 92.3% (the median of two draws of the labelled examples; 90.4% in the draw that was timed), so at the whole set's share the same arithmetic gives about 51 to 58 ms.
The mean time per request is the number that decides whether the table pays; the registration computes it on your own held-out examples, request by request.

**Each replica decides for itself whether to use it.**
The timing is true for the stack it was made on: the host CPU, the engine and its card, the encoder with its revision, the ONNX file's hash, ONNX Runtime's version and its 4 threads (`fast_table.timing.stack` in the record).
`status.fast_table` on `GET /v1/tasks/{id}` says what this replica is doing:

| `state` | What it does |
| --- | --- |
| `active` | The table answers the requests it is sure of. |
| `timing` | The table is not used and decisio answers everything, under the `decisio_only` certificate. The replica is on another stack than the one the table was timed on, or the table stopped beating decisio under this replica's load. Every tenth request also times the table, after decisio has answered, and when 30 such pairs show the saving the table is taken up again (`retimed`). |
| `suspended` | The monitor's alarm fired here or on another replica that shares the task store (below). Decisio answers everything, under the `decisio_only` certificate. |
| `unavailable` | This server cannot run the table: it has another encoder revision than the one the table was fitted with (register the task again), or the `table` extra is not installed. |

**Load.**
The table's time at registration cannot see contention on the host.
While the table answers, the server keeps the median of its last 50 table-path times (judged from 20).
If the expected time per request, with that median on every request and decisio's registered time on the share the table defers, is no longer at least 25% under decisio's, the replica falls back to decisio (`timing`) and the status says why and gives the numbers.
The encoder runs one text at a time on a replica (it is one model on the CPU), so the rate it can sustain is the reciprocal of its time: with the torch encoder's 21 to 30 ms about 35 to 45 table requests per second; above that, requests queue behind it, its median rises and the replica falls back.
Size the CPU for the table, or leave it off.

**The monitor.**
The monitor of a table task watches the stream of answers the task serves, table and decisio together, with the same checks as any certified task (the answer mix and the confidences, [above](#the-monitor)).
Every 50th table answer is also read by decisio, to compare: decisio's answer is not served, and that request takes one more model read.
A CUSUM on how often decisio would have answered otherwise, against the held-out examples' rate, is the agreement alarm (`status.fast_table.agreement`; it is armed only when at least 20 held-out examples were answered by the table and their disagreement is under 20%).
When the confidence alarm or the agreement alarm fires and the task suspends on alerts (`suspend_on_alert`, the default), the table is suspended.

**Suspension reaches the fleet through a shared store.**
With `--task-store` the replica that sees the alarm writes the task again with the suspension in it (a new version of the task, `fast_table.suspended`, with the reason and who wrote it), and every other replica stops using the table when it next follows the store, within `--task-store-poll-seconds`.
A replica applying the new version starts that task's monitor afresh.
With `--tasks-dir`, with no store, on a `--task-store-readonly` replica, or when the write is refused, the suspension holds on the replica that saw the alarm and `suspended.shared` in the record is false.
Re-certify the task (`POST /v1/tasks/{id}/certify` with new labelled examples) to lift the suspension: the table is kept as fitted, its thresholds and its timing are made again, and the table is dropped if it no longer pays.

**What is stored.**
The encoder name and revision, the classifier (a matrix of options by 768 floats and their intercepts), the cascade's thresholds and the timing record.
No request text, no embedding of a request and no example is stored.

**Limits.**
- Choice questions on the text route, one order; a yes/no or score question, an image request, an adapter and a task of options that change per request have no table.
- The table reads the state as text (a string as it is, an object as its JSON); two requests the table cannot tell apart get the same answer.
- The encoder revision is fixed in the release; a task fitted under another revision is `unavailable` until it is registered again.
- The table cannot answer with an option none of its examples carried; those requests go to decisio.
- The saving is a mean over requests: a request the table defers takes the table's time and then decisio's.

## Limits

- **Option list identity.** A task is found by its question's type and its options, keys and descriptions, in order (plus the instructions for yes/no and score questions). Renaming, adding, removing or reordering an option makes a different task, which needs its own registration.
- **Order bound.** The head learns which answer position to trust, so it is bound to the option order it was fitted on, and it is not used with two-order averaging.
- **Options that change per input** cannot be registered: exam-style questions whose answers differ every time have no fixed list to learn, and registration refuses them.
- **The prompt it was fitted under.** A task is fitted on the server's prompt and is applied only by a server with the same prompt and yes/no rendering. The served default changed on 2026-10-03 (the spaced layout, `--prompt-tail spaced`, with yes/no as named letters, `docs/handoffs/tasks.md`): a task registered or exported before then is not applied by the current default, and the server says so at start-up (`WARNING: tasks fitted under another model or rendering are not applied`). Register it again from the same examples; or, to keep using it unchanged, start the server with the compact layout (`--prompt-tail compact --noul-rendering words`).
- **One task per option list per server.** Registration is per server instance; keep it with `--tasks-dir` (or an export and `--tasks-file`), as above.
- **Single-engine or second-engine mode.** The default reads the hidden state from the serving engine at the latency cost above; `--head-engine` trades a second weight copy for speed. Both declare the same choice on every stored evaluation item (`tests/unit/test_head_modes.py`), and a task registered in one mode loads in the other.
- **The measured gains are on intent classification.** Your question's gain depends on how far the model's plain reading is from your labels; `examples/tasks/evaluate.py` measures it on your own held-out examples.
