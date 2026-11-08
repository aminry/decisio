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
- **A fast table that learns from traffic learns across the replicas.** Every replica that writes to the store logs its rows there, one replica trains under a lease, and every replica serves the published versions after timing them itself ([below](#a-fast-table-that-learns-from-traffic)).
- **Monitors stay per replica.** The answer-mix counts of a certified task are not shared; `/metrics` is per replica, so sum `decisio_task_*` across replicas in Prometheus. There is no fleet monitoring.
- **Housekeeping.** Each fitted version of a task is an immutable object under `objects/`; the manifest points at the current ones. Superseded objects are not deleted automatically (a lifecycle rule by age would delete tasks nobody has changed): `python -m decisio.serve.task_store_shared gc s3://bucket/prefix --older-than-days 1 [--dry-run]` removes the ones no entry points to, and the learned rows and versions of removed registrations.
- **Checked on** obstore's in-memory store (which enforces the same preconditions), and against a running RustFS and SeaweedFS (S3-compatible services) for the start-up probe, with two-replica runs of the task store and of a learning fast table on RustFS and, in CI on every pull request, on MinIO (Chainguard's build, `cgr.dev/chainguard/minio`) (`DECISIO_TEST_S3_ENDPOINT`, `tests/unit/test_task_store_shared.py`, `tests/unit/test_fast_table_replicas.py`). Not run against AWS S3, R2, GCS or Azure; the start-up probe is what tells you whether yours enforces conditional writes.

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
- **Per serving stack.** The server stores a fingerprint of the stack it certified on: the decisio version, the base and checkpoint revision, the engine and its version, the card, the quantisation, the KV cache dtype, the context length served (`max_model_len`; on the 31B and the 12B it follows the card unless `--max-model-len` is given, [`running.md`](running.md#the-context-length)), the attention backend, a hash of the token rows the server builds for a fixed probe question (the tokenizer, chat template and rendering), the temperatures, the serving flags (multi-question mode, padding, `--noul-rendering`, `--noul-commit`), the number of orders certification reads (always one, whatever the server's `--orders`: "One order" below) and the task's own fit. **If any of it differs when an answer is served, every answer of the task is `certified: false` with the reason `serving_stack_changed`, and the task's status reads `suspended: serving stack changed`**, listing what differs. The calibration still applies; only the certification stops.
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
  On the vLLM server the prefix cache is always on.
  vLLM's batch-invariant mode ([running.md](running.md#batch-invariance---batch-invariant)), `--batch-invariant on` (or a repository's `defaults.batch_invariant`; the default for the Gemma 4 12B from 0.39.0 to 0.44.0, off again since 0.45.0), removes it: on the merged tuned 12B and the plain 12B in bf16, the 12B in FP8 and an adapter applied at run time (vLLM 0.31.0, one RTX PRO 5000), requests read with the cache on gave the same bits as with the cache off (622 and 979 requests, and 214 with 69% of their tokens from the cache), where without the mode two reads of the 622 with the cache on differed on 145 by more than 0.01 and 4 top answers.
  It costs time (1.05 to 1.33 times a read on bf16, about 1.03 on FP8), and it changes every answer's last digits once.
  Without the mode (the default since 0.45.0) the difference stays.
- **Scored as it was certified.** A certificate is made on confidences scored one question at a time, and a question of a certified task is served the same way: by the function that scored the examples, alone, in request order, never batched with the other questions of its request, whatever the load (on a server with `--concurrency batched` too: below). So on one stack the confidence served for an example is the one the certificate was made on. This costs the request's certified questions the shared-prefix saving (each is a read of its own), and uncertified questions of the same request are scored as before. For a task with an intent head the confidence is the hidden-state read alone, at registration and in serving: registration used to read each example plain first and then hidden, and on vLLM that second read differs slightly from the same read made without the first (one measurement certified 153 answers where serving put 154 above the threshold, the same whether read one at a time or 16 in flight). At registration the server re-reads a tenth of the held-out examples (20 to 200) through the serving path, one request each, and the certificate records `served_check`: the largest and 99th-percentile difference in confidence, how many examples changed sides of the threshold and how many changed their top answer. On one stack it reads zero when the re-read finds the state as the registration did; a state the engine's prefix cache still holds is read warm, and on vLLM a warm read can differ slightly from a cold one, so a small nonzero value does not by itself mean that new traffic is served differently. A nonzero value is said in the certificate's summary. It is a check on the sample, not a proof for every request, and it does not cover a fresh start of the same stack (the previous point).
- **Thinking.** A question of a registered task is always answered at one read, whatever the server's or the request's `think`. The server's think setting is part of the serving-stack fingerprint (only when it is on): a certificate made one-pass is `certified: false` with the reason `think_mode_changed` on a server that thinks, and the other way round, until the task is certified again.
- **Concurrency.** On a server with `--concurrency batched` a certified task's questions, and its registration and certification reads, still take the serial path: one request at a time, never sharing an engine call with other traffic, so the promise above holds there too. The mode is part of the serving-stack fingerprint when it is `batched`: a certificate made on a serial server is `certified: false` with the reason `concurrency_changed` on a batching server, and the other way round, until the task is certified again ([running.md](running.md#concurrency)).
- **Whole inputs.** A request answered from a truncated state (`over_limit: "truncate"`, [api.md](api.md)) is never certified: every answer of a certified task in it is `certified: false` with the reason `state_truncated`, whatever its confidence, and the monitor below does not count it. Registration and certification never truncate: an example over the ceiling is refused.
- **One order.** A certification reads every example in one option order, on any server, so the certificate's serving stack records one order whatever the server's `--orders`.
  A request read in another number of orders (`"orders": 2`, or a request without `orders` on a server started with `--orders 2`) is answered as read, the two-order average included, with `certified: false` and the reason `orders_changed`, whatever its confidence.
  The monitor below does not count it, and it is a fact of that request only: the certificate is not suspended, and the task's other requests are certified as before.
  On a server started with `--orders 2`, send `"orders": 1` on the requests that should carry the promise.
  A certificate made by an earlier release on a server started with `--orders 2` recorded the server's 2, though its reads were one order; it is read as one, and the stored record is not rewritten.
- **Traffic like the examples.** The held-out examples must look like what the server will see: the same kind of text and the same mix of options. The monitor below says when the traffic moves away.
- **What is certified.** Choice and yes/no questions ([below](#yesno-questions)), on text requests, in the single-order form the certificate was made on (a request read in two orders is `certified: false` with `orders_changed`, "One order" above).
  Score questions are not certified: an ordinal answer has no single wrong answer.
  A question that matches no task, or an image request, carries no certification fields.

### The monitor

A certified task is watched by three checks that need no labels.

- **The serving-stack fingerprint**, above: any change suspends the certificate. `GET /v1/tasks/{id}` shows the status and what changed.
- **The confidences.** After every answer of the task it counts (its scope, below) the server updates a CUSUM on how often the answer's top probability falls below the 5th percentile of the certification examples' (a count and a number, never the question, the state or the option names). It treats up to 1.3 times the examples' share as normal, a share of 2.5 times as worth an alarm, and its limit is the smallest at which a stream at the normal share alarms with probability at most 1 percent in 5,000 answers, simulated at certification with a fixed seed. When it crosses the limit the task's status reads `traffic_changed` with `confidence` among its signals, and it clears when the statistic has fallen to half the limit (about 300 normal answers after the traffic returns). **By default this suspends certification:** every answer of the task is `certified: false` with the reason `traffic_changed` while the alert is on, because the promise is for traffic like the examples and the monitor says it is not; the answers themselves do not change. A certify block with `"suspend_on_alert": false` only reports. A task certified before this signal existed has no confidence statistic and runs the answer mix alone until it is certified again; the certify call that makes the statistic takes the default, `suspend_on_alert` true, when the old certificate names none.
  On the replay of a CLINC150 task (the stored 31B, 10 examples per intent; 5,363 messages, out-of-scope messages rising from message 1,000 to 30%, which the task has no option for), on the card, under 0.14.0 the answer mix first alerted after message 5,200 with 1,027 wrong certified answers served before it, and under 0.16.0 the confidence signal first withheld an answer at message 1,570 after 40 wrong certified answers (the offline replay of the recorded reads, in `runs/2026-10-17_drift-monitor/`, alerted at message 1,697 after 55). On the 4,500 in-scope messages alone, and on the test split of three BANKING77 draws read through their tasks, it did not alert (`runs/2026-10-17_drift-monitor/`). The thresholds were chosen on the CLINC150 stream, so the first test on traffic it was not chosen on is the card cell that replays a second stream.
- **The answer mix.** The server counts, for each certified task, how often each option was chosen in the last 500 answers, as counts only, and compares the mix with the certification examples' every 100 answers once the window is full. When the distance is above a line fixed at certification, the task's status reads `traffic_changed` with `answer_mix` among its signals. This says the traffic is no longer like the examples (a new channel, more out-of-scope messages, a launch); **by itself it does not suspend the certificate or change any answer**. It is weak on a task with many options, where a shift spread over them barely moves the mix (the CLINC150 replay's distance sat at 0.19 to 0.28 against a line of 0.28 until message 5,200), and it alerted on normal traffic from another split at its first check on all three BANKING77 draws, whose examples' mix is the training split's and whose traffic was the test split's.
  The state is in `GET /v1/tasks` and `GET /v1/tasks/{id}` (`status`) and in `GET /metrics`, Prometheus text: `decisio_task_state{task,state}`, answers and certified answers, `decisio_task_mix_distance` and `decisio_task_mix_line`, `decisio_task_confidence_statistic`, `_limit` and `_alerts_total`. The counters are in memory and start again with the server.

**Its scope.** A task's `traffic_changed` is this replica's, about the answers this replica served:
- **Per task and per replica.** Each replica counts its own answers in memory; nothing of the monitor's state is written to `--tasks-dir`, a shared task store or an export, so replicas behind a load balancer each alert on their own traffic.
  The baseline and the limits are stored with the task, so every replica watches against the same ones.
- **What starts it again.** A restart, and any new version of the task in this replica's memory: registering it again, certifying it again, an import, or a version another replica wrote to a shared store (a fast table's suspension is one).
  The counts and the statistics start at zero, and an alert on at that moment is gone.
- **What it counts.** Every answer of the task from decisio's read or its fast table, certified or not; not a registration's re-read of its sample, not an answer scored in an engine call shared with other requests (`--concurrency batched`, counted as `shared_skipped`), not a truncated request or one whose header was limited, and not decisio's answers on requests a fast table was sure of and left to decisio to explore, or that a learning table deferred.
  A task with a fast table has two monitors, one for the combined answers and one for decisio's answers alone when the table is off; the status shows the one in use.
- **Which signals suspend.** The confidence signal, and a dynamic task's unseen-options signal, make every answer of the task `certified: false` with `traffic_changed` while they are on, unless the certify block said `"suspend_on_alert": false`.
  The answer mix only reports.
  The confidence alarm (and a table's agreement alarm) also switches a fast table off, and that is written to the task's record and shared ([A fast table](#a-fast-table-in-front-of-decisio-opt-in), "Suspension reaches the fleet through a shared store"); unlike the alert, it does not clear by itself.
- **What clears it.** The confidence and unseen-options alerts clear when their statistic falls under half its limit; the answer-mix alert clears at the next check (every 100 answers) whose distance is at or under its line.
  The state is `ok` when none is on.

**A task with no option for a message that fits none of its labels** (no "other", "none of the above" or "out of scope") gets a wrong answer for every such message, and that answer can be certified. Registration says so in the certificate (`warnings`, and its summary) when a choice task has no option whose key says it. Adding one is the remedy; the monitor only tells you after the fact.

The monitor cannot see a wrong answer, because it has no labels. The only direct check that the promise holds on live traffic is to label some of it and certify again.

### Yes/no questions

A yes/no task (`"type": "noul"`) is certified as a choice task with two options, `yes` and `no`.
Give each example's `answer` as `true` or `false` (or `"yes"` or `"no"`).
An answer reports `noul`, the probability of yes; the confidence the certificate is about is its top probability, the larger of `noul` and `1 - noul`, and the answer is `certified: true` when that is at or above `certificate.threshold`.

- **The commit rule.** With `--noul-commit` (or a tuned repository's `defaults.noul_commit`), an answer whose probability of yes is between 0.20 and 0.80 is reported at the band's edge on its own side: 0.80 for yes, 0.20 for no.
  The answer itself never changes, and the task's calibration or the temperatures apply first, as for every yes/no answer.
  **The certificate is made on the committed probabilities**, the ones the answers report, and each answer's flag is read on the probability it reports, after the rule.
  So the promise is about what the client receives: every answer reported at 0.80 is certified, or none is, and on the held-out examples they counted the same way.
  The price is resolution: the unsure answers all report 0.80, so no threshold falls inside the band, and the band is certified as a whole or not at all (equal confidences are served together).
  A certificate on the probabilities before the rule could certify some answers reported at 0.80 and not others, and the flag could no longer be read off the probability the answer shows.
  The rule is part of the serving-stack fingerprint: switching it on or off suspends the certificate (`serving_stack_changed`) until the task is certified again.
- **The monitor.** The answer mix counts yes and no.
  Under the commit rule the confidence signal reads each answer's top probability before the rule, at certification and in serving: the committed ones are never below 0.80, so the tail it watches would sit at 0.80 with nothing below it.
  The task's monitor record and its status say so (`"reads": "before_noul_commit"` in the confidence block).
  A yes/no task certified under the rule by an earlier release has no such key and reads the reported probabilities, which can stay quiet when unsure answers rise; certify it again to arm the signal.
- **The rendering.** `--noul-rendering` (`words`, `letters` or `letters-keys`) is part of the task's key and of the serving stack.
  A task registered under one rendering is not applied on a server with another: its answers carry `certified: false` with `serving_stack_changed`, and its status reads `suspended: serving stack changed` with `serving.noul_rendering` among the changes.
- **The fallback.** An answer that is not certified is still answered.
  Below the threshold, with nothing certified (`nothing_certified`) or on a changed stack, the task's calibration still applies; where the task is not applied at all (a request of the other layout, `layout_mismatch`; a task fitted under another model or rendering, `serving_stack_changed`) the answer is decisio's own read.
  Either way the commit rule, when on, applies last, and the answer carries `certified: false` and the reason.
- **Status, export and storage** are those of a choice task: `GET /v1/tasks/{id}` gives the state (`ok`, `traffic_changed` or `suspended`) and the monitor, and export, import, `--tasks-dir` and a shared store keep the certificate and the monitor's record.
- **A fast table.** A yes/no task can have one, labelled or learned from traffic: a two-class table that reports its own calibrated probability of yes ([A yes/no table](#a-yesno-table)).

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
- The text route, one order per request (`orders` 1); a request of two orders is answered as before and not certified (`orders_changed`).
- The features read the option text as the request shows it, so options that differ only in a description (a tool's name and its description) are read as written.
- The features are string overlaps with the request's text, so the re-ranker helps where the right option can be seen in the text, or has been the answer before; it does not help where neither is true, and the server then declines it.
- The gate is itself slightly optimistic: the features are standardised on all the labelled examples, and the cross-validation that picks the regularisation is the one the gate reads. It declines a re-ranker whose gain is below 0.005 in log loss per example, which on one question (intent routing on the 31B) gave up a small real gain.
- The gain rests on five question types and two model sizes; a task of another kind may gain more or nothing, and the cross-validation says which.


## A fast table in front of decisio (opt-in)

For a task of short texts that arrive in volume, a small classifier on the host CPU can answer most requests without asking the model.
It is off unless the registration asks for it, and it keeps itself only where it pays.
A task with no labels can learn one from its own traffic ([below](#a-fast-table-that-learns-from-traffic)).

**What it is.**
An embedding of the request's text, a multinomial logistic regression fitted on the task's own labelled examples, and a confidence threshold: a small learned classifier on the server's CPU, not a lookup of earlier answers.
A request whose table confidence reaches the threshold is answered by the table and the model is not asked.
Every other request goes to decisio as it would without the table.
The embedding is the table's student, and since 0.41.0 a task chooses it.

**Choose a kind.**
`"fast_table": true` asks for the default; an object chooses:

| `fast_table` | Student | Use it for |
| --- | --- | --- |
| `true` or `{"kind": "static"}` (the default) | `minishlab/potion-base-8M`: a static embedding, the mean of the text's token vectors (Model2Vec), computed by decisio itself | most tasks: it costs so little that a table of it never makes a request slower |
| `{"kind": "static", "static_model": "multilingual"}` | `minishlab/potion-multilingual-128M` | traffic that is not English |
| `{"kind": "encoder"}` | `BAAI/bge-base-en-v1.5` through ONNX Runtime (the only student before 0.41.0) | many-class English tasks where its larger share pays for its time on your host |
| `{"kind": "auto"}` | fits the static student (and `static_model`) and the encoder, and keeps the one that saves more on this host | when you do not know |
| `{"kind": "hashed"}` | reserved; refused | not available in this version |

The static models are MIT licensed, as is bge-base.
Each student's files are pinned to a revision and checked against pinned sha256 hashes when they are fetched.

**What a student costs.**
The table path (the student and the classifier) for one request, p50 milliseconds, on an Apple M5 Pro:

| Student | A short text (median 22 tokens) | A long text (median 1,416 tokens) |
| --- | ---: | ---: |
| static, `base` | 0.05 | 1.0 |
| static, `multilingual` | 0.05 | 1.9 |
| encoder | 6.3 | 85 |

On an x86 server CPU the static students cost about the same (0.04 to 0.06 ms a short text, 1 to 4 ms at 1,400 tokens), while bge-base at one thread costs about 25 ms a short text and up to about 450 ms a long one.
A table runs on every request to find out whether it is sure, so its time is paid by every request; with a static student that time is close to nothing, so a request the table hands on costs about what it would without a table.

**Where it helps, and where it does not.**
It helps on texts with about a thousand labelled examples where decisio is reliable: an assistant's commands, spam, a fixed set of intents.
The static students read the whole text; the encoder reads at most 512 tokens (for a longer text, its first 256 and last 254 tokens).
It does not help where decisio is weak enough that nothing is certified, and it does not help on a task whose options differ from request to request.
On long or hard tasks a table usually certifies nothing; the registration then keeps no table, and the task is served exactly as without one.
The registration times every student it fits on your own examples and drops the table when it does not pay (below), so asking for it on the wrong task costs the registration, not the traffic.

**How to ask for it.**

```bash
pip install 'decisio[table]'     # scikit-learn (the fit) and onnxruntime (the encoder); the image does not carry them
```

```json
{"id": "assistant-commands", "examples": [...], "certify": {"alpha": 0.05, "delta": 0.1}, "fast_table": {"kind": "auto"}}
```

- A `certify` block is required: the table is part of the certificate.
- Choice and yes/no questions ([A yes/no table](#a-yesno-table)), with at least 100 fit examples.
- `min_saving` sets the share of the time per request the table must save ([The saving bar](#the-saving-bar-min_saving)); the default is 0.10.
- Without the extra the registration is refused with 422 and says what to install.
- `static_model` goes with the `static` and `auto` kinds only; with `encoder` it is refused.
- Each student's files are fetched from the Hugging Face Hub at the pinned revision on first use. The fast-table image (`ghcr.io/aminry/decisio:<version>-table`, [running.md](running.md#the-fast-table-image)) has both static students built in, so it needs no Hub access for them. Any other server with no route to the Hub needs them fetched first, or the registration is refused naming the command:

```bash
hf download minishlab/potion-base-8M model.safetensors tokenizer.json --revision bf8b056651a2c21b8d2565580b8569da283cab23            # static, base (31 MB)
hf download minishlab/potion-multilingual-128M model.safetensors tokenizer.json --revision 73908c3438cf03b6a01bcb9611d62b23d0726f08  # static, multilingual (531 MB)
hf download BAAI/bge-base-en-v1.5 onnx/model.onnx tokenizer.json --revision a5beb1e3e68b9ab74eb54cfd186867f64f240e1a                # encoder (436 MB)
```

**What the registration does.**

1. It fits each student it is asked for (both under `auto`) on the fit examples, and embeds the held-out examples one at a time.
2. It reads the held-out examples through decisio, one at a time, as the certificate does.
3. It makes the cascade's thresholds (below) for each student.
4. It times the paths on the held-out examples, on this host, one request at a time, with decisio's reads of texts it has not seen: the expected mean time per request is the table's time on every request plus decisio's time on the share the table defers.
5. It keeps a table only if the expected time falls by at least the task's `min_saving` (10% by default) and a table threshold holds. Under `auto` it keeps the student with the larger saving; when the two savings are within 1 point it keeps the static one; when neither qualifies there is no table.

The response says what it did.
`fast_table_report` has `kept`, the `reason` when it is false (for example "it would save 7% of the time per request, under the 10% required"), `required` (the task's `min_saving`), the `guard` (the table's and decisio's mean milliseconds, the table's share, the saving) and the cascade's thresholds.
`fast_table_report.kinds` lists every student tried, with its table share, its error at the threshold and the upper bound, its expected saving and its mean table time, which one was kept and why; a student that could not be loaded is listed under `unavailable`.
A table that is not kept leaves a task that is served, and certified, exactly as it would be without one.

**The table records its feature function.**
`fast_table.features` in the task's record names the student exactly: its kind, model, revision, the files' sha256 and the long-state rule (`whole` for a static student, `head256+tail254` for the encoder, `trunc512` for an encoder table registered before 0.41.0, read as its first 510 tokens).
A table is served only with that function.
A server that cannot compute it (another model, revision or rule) serves the task through decisio and says why in `status.fast_table` (`unavailable`, "register the task again"); the rest of the task (its calibration, its certificate for decisio alone) still applies.
A table registered before 0.41.0 has no `features` and is served, exported, imported, certified again and timed exactly as before.
A replica on a version before 0.41.0 checks only the record's `encoder` and `revision`: a new encoder table names its encoder with its rule (`BAAI/bge-base-en-v1.5#head256+tail254`), so such a replica finds an encoder it does not have and serves the task through decisio rather than read the table's features another way.

**The certificate is the cascade's.**
The table's threshold is the lowest at which at most `alpha` of its answers are wrong, with probability at least 1 - `delta`/2, on the held-out examples.
Decisio's threshold is made the same way on the requests the table defers.
An answer of either stage that reaches its threshold is certified, and the promise is the cascade's: at most `alpha` of the certified answers are wrong, for traffic like the examples and on this serving stack.
The table's stage keeps that promise by the procedure's proof, with probability at least 1 - `delta`/2.
That holds because the table's score is fixed before the held-out labels are read: a choice table's probabilities come from the fit examples alone, and a yes/no table's temperature ([below](#a-yesno-table)) is fitted on the fit examples by cross-fitting, never on the held-out ones.
Decisio's stage is certified on the requests the table's chosen threshold defers, from the same held-out examples, so which requests those are depends on the same labels that set the table's threshold.
Its promise is therefore supported by simulation and not proved: on simulated populations, including one where the table fails exactly where decisio finds requests easy, it broke its allowance in at most 1.3% of draws against the 5% allowed.
A stricter mode that certifies the two stages on disjoint examples is possible later.
The certificate lists both stages (`table_threshold`, `threshold` for decisio's, and per stage the answers, the errors and an upper bound), and `decisio_only` holds the certificate decisio alone would have had, which is what serves when the table is off.
Each answer says which stage answered it, `answered_by: "table"` or `"decisio"`.
The cascade as a whole broke its promise in about 2 to 3 of every 100 simulated draws of the held-out examples, against the 10 allowed (`tests/unit/test_fast_table.py`), because each stage is held to `delta`/2.
That costs examples: the held-out set needs to be larger than for decisio alone (about 600 or more, at the defaults).

**What was measured with the encoder.**
These figures are for the `encoder` kind (bge-base), measured before the static student existed.
With the torch encoder of 0.37.0 the table path was 21 to 30 ms for a short text and 134 to 183 ms for a text of 600 to 1,150 tokens (the table below), which is more than decisio's own read.
ONNX Runtime (0.38.0) gives the same embeddings to about 1.5e-6, and is 2.5 times faster on one short text (4.6 ms against 11.6 ms for the whole table path, median, on an Apple M5 Pro at 4 threads); the encoder is about 99% of that path.
A table made on 0.37.0 serves the same answers and confidences (to about 2e-6).
A table registered on 0.38.0 is fitted on embeddings that differ by about 1e-6; on the eight sets measured its threshold moved by at most 0.0015 and its certified share by at most 0.3 points, with accuracy unchanged.
That ratio was not measured on the x86 server CPUs, and the registration times the encoder it will serve on the host it runs on.
Offline, with the server's own code, on public sets at 1,000 labels (the 31B and the 12B; the table on a server CPU with four threads; decisio on one card; medians of two draws of the labelled examples):

**Across the ten sets tried, the table passed the 25% guard on 4: CLINC150, MASSIVE, Snips and SMS spam.**
These figures were read at the fixed 25% guard of the time.
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

**What a request costs with the encoder.**
The table is a small learned classifier: an embedding of the text followed by a logistic regression, both run on the server's CPU, in front of the model.
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
The timing is true for the stack it was made on: the host CPU, the engine and its card, and the student: for `static` its model and revision, for `encoder` the encoder with its revision, the ONNX file's hash, ONNX Runtime's version, its threads and its sessions (`fast_table.timing.stack` in the record).
`status.fast_table` on `GET /v1/tasks/{id}` says what this replica is doing:

| `state` | What it does |
| --- | --- |
| `active` | The table answers the requests it is sure of. |
| `timing` | The table is not used and decisio answers everything, under the `decisio_only` certificate. The replica is on another stack than the one the table was timed on, or the table stopped beating decisio under this replica's load. Every tenth request also times the table, after decisio has answered, and when 30 such pairs show the saving the table is taken up again (`retimed`). |
| `suspended` | The monitor's alarm fired here or on another replica that shares the task store (below). Decisio answers everything, under the `decisio_only` certificate. |
| `unavailable` | This server cannot compute the table's features exactly: another model, revision, file or long-state rule than the record's `fast_table.features` (register the task again), or the `table` extra is not installed; or a yes/no table's threshold was made under another `--noul-commit` setting than this server's (certify the task again). |

**Load.**
The table's time at registration cannot see contention on the host.
While the table answers, the server keeps the median of its last 50 table-path times (judged from 20).
If the expected time per request, with that median on every request and decisio's registered time on the share the table defers, is no longer at least the task's `min_saving` under decisio's, the replica falls back to decisio (`timing`) and the status says why and gives the numbers; the 30 paired timings that take the table up again are held to the same bar.
A static student costs a fraction of a millisecond, so it is not what limits a replica's rate.
The encoder runs as two sessions of four threads each: at 16 requests in flight it sustained about 90 table requests a second on an Apple M5 Pro, against 68 when it ran one encode at a time (0.39.2 and earlier); above its rate, requests queue behind it, its median rises and the replica falls back.
Size the CPU for the encoder, or choose a static student.

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
The student's feature function (`features`), the classifier (a matrix of options by the student's dimension, 256 floats for the static students and 768 for the encoder, with their intercepts), the cascade's thresholds and the timing record.
No request text, no embedding of a request and no example is stored.

**Limits.**
- Choice and yes/no questions on the text route, one order; a score question, an image request, an adapter and a task of options that change per request have no table.
- The table reads the state as text (a string as it is, an object as its JSON); two requests the table cannot tell apart get the same answer.
- The static students read the whole text; the encoder reads at most 512 tokens of it (the first 256 and the last 254 of a longer text).
- Each student's revision is fixed in the release; a task fitted under another revision or rule is `unavailable` until it is registered again.
- The table cannot answer with an option none of its examples carried; those requests go to decisio.
- The saving is a mean over requests: a request the table defers takes the table's time and then decisio's.

**Measure it on your own host.**
`python -m decisio.bench.table_latency --url http://127.0.0.1:8000 --examples train.jsonl --probe probe.jsonl --out table_latency.json` times a certified task with and without its table, end to end, through your running server.
It registers the task twice, one arm after the other: with `"fast_table": true` (arm `table`), then without it (arm `model`), each timed on its own held-out requests and deleted.
`--kinds static,encoder,auto` times a table arm for each student kind instead (`static:multilingual` names the static model), each registered on the same examples, and the cells name their kind; the probe file then needs requests for every arm.
The cells are each arm at two state lengths (`short`, the requests as they are; `long`, padded to `--long-chars`) and two concurrencies (`--concurrency 1,16`).
Each line of its output is a cell: the wall time's p50 and p95, the server's time (`x-decisio-server-ms`), the `task` stage (the certified question's read: the table's, or decisio's when the table deferred), the share the table answered and the errors.
Compare the `table` arm's wall time with the `model` arm's in the same cell: that difference is what the table saves your callers, under your load.

### The saving bar: `min_saving`

`"fast_table": {"min_saving": 0.2}` sets the share of the expected time per request a table must save, from 0.01 to 0.90 inclusive; a value outside that range, or not a number, is refused with 422.
The default is 0.10.
The same value decides at registration whether the table is kept, and at run time whether a replica keeps using it under load (**Load.**, above); every message that names the required saving names the task's own.
It is stored in the task's record (`fast_table.min_saving`, and for a task that learns from traffic in `learn.request.min_saving`), so export, import, `--tasks-dir` and a shared store keep it, and `GET /v1/tasks/{id}` shows it under `status.fast_table.min_saving`.
A table registered before the setting existed has no `min_saving` in its record and was kept under a fixed 25%: it is held to 25% at run time, and a re-certification keeps that bar (and then records it).
Register the task again to take the new default.
The saving counts the table's own time on every request, as before: a static student's fraction of a millisecond, an encoder's tens of milliseconds.
On a measured suite of ten sets the bar changed no kept table at 25, 15, 10 or 5%, because a table of the static student there either certified nothing or certified 30% of the requests and more.
The lower default matters for a task whose table certifies about 10 to 25% of its requests.

### A yes/no table

A yes/no task (`"type": "noul"`) gets a two-class table, under the same rules as a choice table: the certificate, the bar, the timing, the monitor and its alarms, replicas and suspension, and learning from traffic.
What differs is the probability it reports.
- **Its own P(yes), calibrated.** The table's answer carries `noul`, its own probability of yes, never decisio's, and `answered_by: "table"`.
  A logistic regression's probability is often too sure or not sure enough, so the registration calibrates it: one temperature `T` divides the table's logit for yes, `P(yes) = sigmoid(logit / T)`.
  `T` is fitted on the fit examples by cross-fitting: they are split into 5 deterministic folds (fewer when one answer has fewer than 5 examples), a classifier of the table's strength is fitted on four folds and gives its logit on the fifth, and `T` is the one that minimises the log loss of all those out-of-fold logits against their gold answers (with Platt's targets, `(N+ + 1) / (N+ + 2)` for a yes and `1 / (N- + 2)` for a no, which keep `T` finite when the logits separate the answers perfectly).
  The table that serves is still the one fitted on all the fit examples.
  The certification examples are never used, and neither is any traffic, so the table's score is fixed before a single certification label is read, and the table stage's guarantee holds as proved (above): changing only the certification examples' labels changes neither `T` nor any score of the table.
  A re-certification therefore keeps `T`, and serving never moves it.
  `fast_table.calibration` in the record (and `fast_table_report.calibration`) holds `T`, the method (`"cross_fit"`), the folds, the examples it was fitted on and the out-of-fold log loss before and after.
- **Its answer.** Yes when its calibrated P(yes) is above 0.5, no otherwise: exactly 0.5 is no, decisio's own tie-break.
  The temperature has one parameter and no bias, so it changes neither the table's answer nor the order of its confidences: the certificate certifies the same answers it would have certified on the table's uncalibrated probabilities, and only the number reported changes.
- **Its confidence and its certificate.** The confidence the table's threshold is made on is the one decisio's certified yes/no answers are read on: the top probability, the larger of `noul` and `1 - noul`.
- **The commit rule.** On a server with `--noul-commit`, a table answer is moved as decisio's answers are: a P(yes) between 0.20 and 0.80 is reported at the band's edge on its own side, and the answer never changes.
  The table's threshold, like decisio's, is made on the committed confidences, and the table answers a request when its committed confidence reaches it, so the band is answered by the table as a whole or not at all.
  The record says which way it was made (`fast_table.commit_band`: `[0.2, 0.8]`, or `null` without the rule).
  A server whose rule differs does not use the table: under the rule a P(yes) of 0.55 would score 0.80 against a threshold made without it, and the other way round.
  It serves the task through decisio, `status.fast_table` reads `unavailable` with the reason, and `POST /v1/tasks/{id}/certify` makes the table's threshold again under this server's rule.
  A learned version is bound the same way, through the serving stack it was certified on, which includes the rule.
- **The monitor** reads each answer's confidence before the rule (`"reads": "before_noul_commit"`, [Yes/no questions](#yesno-questions)): a table answer contributes its calibrated P(yes) before the rule, and decisio's answers (deferred, explored, with the table off, or logged while the task learns) their own before the rule.
  The baselines of the cascade's monitor and of the decisio-only monitor are made the same way, at registration and at every re-certification.
- **A table learned from traffic** calibrates each refit's P(yes) the same way, by cross-fitting on that version's fit rows, never on the held-out rows its certificate is made on: against decisio's answers, so its P(yes) is the chance that decisio says yes (and, for a labelled task that learns, the labelled examples' gold answers among them).
  Its threshold and its sureness read its confidence after the commit rule when the rule is on.
  Its low-confidence guard reads the table's confidence before the rule: an output counts as low when it is inside the band or under the threshold, against that share on the certificate's rows.
  The committed confidences are never under 0.80, so with a threshold at the band's edge a guard on them could never fire, and traffic that moved to P(yes) 0.4 to 0.6 would stay answered and certified.

### A fast table that learns from traffic

A task with no labelled examples can still get a fast table: decisio's own answers to the task's requests are the table's training data.

**How to ask for it.**
Register the task's question with `"learn": true` in its `fast_table`:

```bash
curl -s localhost:8000/v1/tasks -H 'content-type: application/json' -d '{
  "id": "queue",
  "question": {"type": "choice", "instructions": "Which team should handle this message?",
               "criteria": {"billing": "...", "tech": "...", "sales": "..."}},
  "fast_table": {"learn": true, "kind": "static"}
}'
```

The question can be a yes/no one too (`{"type": "noul", "instructions": "..."}`); its table reports its own P(yes), calibrated against decisio's answers ([A yes/no table](#a-yesno-table)).
`kind` is `static` (the default), `encoder` or `auto`, as for a labelled table, and `static_model` names the static model.
`explore` (default 0.02) and `max_rows` (default 200,000) are the two learning settings, and `min_saving` is the bar, as for a labelled table; `retain_text` is refused, because no request text is kept.
`"certify": {"alpha": 0.05, "delta": 0.1}` beside `fast_table` sets the certificate's bounds (these are the defaults).
Until a table certifies, decisio answers every request of the task, exactly as with no table.

**What is logged.**
Each request decisio reads for the task is logged as one row: the students' features (float16), decisio's distribution (float16) and its confidence, the row's kind, the serving-stack fingerprint and the time.
No request text is kept.
The kinds are `no_table` (no table was live, or it was suspended), `deferred` (the live table was not sure), `explored` (see below) and `spot_check` (a table answer decisio also read).
With `--tasks-dir` the rows are append-only segment files under `<tasks-dir>/learn/`, and they survive a restart.
Without it they are kept in memory, are lost at restart, and `status.fast_table.rows.stored` says so.
The oldest rows are dropped past `max_rows`.

**Refits.**
One background thread at low priority refits a task every 2,000 new rows or every hour, once it has 100 rows, and never two tasks at once.
Rows are split once, by a keyed digest of their text: 30% of the digests are held out of every fit, for certificates.
A refit fits the classifier on the other rows logged on this serving stack.
A request the live table would have answered is logged only when it was explored or spot-checked, so each such row is weighted by one over its chance of being logged, and the fit stands for the traffic.

**The certificate is an agreement certificate.**
It is made only on held-out rows drawn independently of the table's confidence: the `no_table` rows and the `explored` rows.
Deferred rows are the hard requests the table chose to pass on, and spot checks are drawn only from the requests it answered, so neither stands for the traffic, and neither enters a certificate.
The table's threshold is the lowest at which its answers differ from decisio's on at most `alpha` (default 0.05) of the held-out rows it answers, with probability 1 - `delta` (default 0.1).
That bounds how often the table disagrees with decisio, not how often it is wrong: a learned table is as right as decisio is, at best.
Answers from a learned table carry `"certificate_kind": "agreement"`; a table certified on labelled examples carries `"gold"`.

**Promotion.**
A refit's version goes live only if its certificate holds and its expected saving, timed on this host from the logged times, is at least the task's `min_saving` (10% by default; 25% for a task registered before the setting) and more than the live version's.
A refit that fails leaves the live version live.
The last 5 versions are kept, and `POST /v1/tasks/{id}/fast_table/rollback` with `{"version": n}` serves one of them again.

**Exploration, and what it costs.**
While a table is live, each request it reads is explored with chance `explore`: decisio answers it, whatever the table's confidence, and the row is logged as `explored`.
These rows are a uniform sample of the traffic, so later certificates keep rows drawn like the traffic, and not only the hard ones the table defers.
An explored request the table would have answered costs one model read, at decisio's latency.
With a table answering a share `c` of requests, exploration adds about `explore` × `c` of the traffic to the model's load: at the default 0.02 and a table answering 80%, 1.6% of requests.
The spot checks add another 2% of the table's answers (every 50th), and a spot-checked request waits for decisio's read too.
`status.fast_table.explore.cost` gives the share for the live version.
`"explore": 0` turns exploration off; then the only rows drawn like the traffic are those logged before the first table, and later certificates rest on them.

**Two drift guards.**
Either alarm suspends the live table to decisio-only answers (`state: suspended`); logging and refits go on, and the next version that certifies goes live.
- The low-confidence share: over the last 200 requests the table read, the share under its threshold rising more than 15 points above the share at certification.
- A CUSUM on disagreement, fed by the spot checks and by the explored requests the table would have answered, against the certificate's `alpha`.

**A labelled task can learn too.**
`"learn": true` on a registration with `examples` adds the examples to every refit's fit, at twice a traffic row's weight.
Its certificate stays gold, made on its held-out labelled examples, as long as there are enough of them with the student's features; otherwise it is an agreement certificate.

**What the status says.**
`GET /v1/tasks/{id}` reports the learning under `status.fast_table`:

| Field | What it holds |
| --- | --- |
| `mode`, `state` | `learning` (a task with no labels) or `both`; `learning` (no version yet), `active`, `suspended` or `off` (this server cannot serve the live version: another feature function or serving stack), with the `reason` |
| `kind` | the kind asked for and the one chosen |
| `rows` | kept, since the last refit, bytes, logged by kind, `max_rows`, and where they are stored |
| `version` | the live version's number and the kept versions |
| `coverage`, `saving`, `certificate` | the live version's certified and observed table share, its expected saving, and its certificate (with `certificate_kind` and the rows it drew on) |
| `explore` | the fraction, the requests explored, and the cost |
| `guards` | both guards' statistics and whether they fired |
| `refit` | the last refit's outcome, the cadence, and the refits and promotions so far |

`/health` gives a short summary of every learning task under `systemone.fast_table_learning`.

**Limits of learning.**
- A choice or yes/no question on the text route, as for a labelled table.
- The table learns decisio's answers, including decisio's mistakes.
- The certificate's rows are weighted as they were logged: before the first table every request is a `no_table` row, and afterwards only the explored share is, so early traffic weighs more in a certificate until `max_rows` drops it. The drift guards watch the live traffic.
- A server on another serving stack does not serve a version certified elsewhere: decisio's answers there are another teacher's.

**On a shared store (`--task-store`), the replicas learn one table together.**
Every replica logs its rows to the store, as segment files of its own (`tasks/<digest>/log/<replica>/`), written from the task store's poll thread: a request never waits on the store.
One replica trains: the holder of a lease in the store (`learner/LEASE.json`, 120 s, renewed every 40 s), and when it stops another takes the lease.
It fits on every replica's rows and certifies on every replica's `no_table` and `explored` rows, in both cases only rows logged on its own serving stack (the engine, its version, the card and the settings); the certificate names the replicas and how many rows each gave.
Run one serving stack across the fleet if every replica's rows should count.
A published version carries the lease's generation, and a trainer that lost the lease while it fitted publishes nothing.
Every replica picks up a new version at its next poll and times it on its own requests before its table answers there: until then its status is `timing` and decisio answers (every tenth request is timed both ways, and the table answers once 30 pairs show the saving).
One replica's drift alarm suspends the table on every replica, and a rollback on any replica reaches all of them.
A replica that restarts, or a whole fleet, serves the promoted versions after its own timing; nothing is learned again.
While the store cannot be reached, serving goes on with what each replica has, rows wait in memory up to 256 MB per server and are then dropped and counted (`/health`, `systemone.fast_table_learning.shared`), and no version is promoted.
A `--task-store-readonly` replica serves the versions and writes nothing: it logs no rows, never trains, its alarm suspends only itself, and a rollback there answers 409; its traffic is not in the certificate.
`GET /v1/tasks/{id}` on any replica shows this replica's part (`replica`: its id, whether it trains, its timing) and the fleet's (`cluster`, from the trainer's last aggregate of the replicas' stats, every 5 minutes).
Lease expiry is judged on each replica's clock, so keep the clocks synchronised (NTP); a holder whose clock runs behind still cannot publish over its successor.

## A task under a layout

A server serves requests of both layouts of the user turn, the plain one and the header ([running.md](running.md#a-layout-per-request-layout)), and a task is fitted under one of them.

**Registering.** Every registration route takes an optional `"layout": "plain" | "header"`: `POST /v1/tasks` (with examples, `"options": "dynamic"`, or a task that learns from traffic) and `POST /v1/abstention/tasks`.
Absent, the task is fitted under the server's layout (`/health`, `systemone.layout`).
The examples are read under that layout, the task's fingerprint names it (a header task's carries `"layout": "header"` in its prompt format; a plain one's is what it always was), and an abstention task's record carries `"layout"` when it is not plain, as before.
An example whose request names another layout than the task's is refused with 422; one that names the same, or none, is read as the task's.
An abstention task with image examples takes no `layout`: the image route has none.
`POST /v1/tasks/{id}/certify` certifies a task again under the layout it was fitted under.

```bash
curl -s http://127.0.0.1:8000/v1/tasks -H 'Content-Type: application/json' \
  -d '{"id": "ticket-routing", "layout": "plain", "examples": [...], "certify": {}}'
```

**Serving.** A task applies only to the requests of its layout: a request that names that layout, or names none on a server whose layout it is.
- A question of a task fitted under the other layout is answered as if the task did not exist: decisio's own read, under the request's layout.
  If the task is certified, the answer carries `certified: false`, the task's `certificate` and `certified_reason: "layout_mismatch"` (the pattern of `serving_stack_changed`); the traffic monitor does not count it.
- A fast table, labelled or learned from traffic, never answers a request of the other layout, and a task that learns from traffic logs no row from one: such a row would be decisio's answer under another layout than the one its table is certified against.
  A learning task's rows, versions and certificates are on its own layout's serving stack.
- A certificate is checked against the serving stack of its task's layout, so a header task certified on a header server stays certified for header requests on a plain server whose stack is otherwise the same.

**At start, import and export.** Tasks of both layouts load from `--tasks-dir`, `--task-store` and `--tasks-file`, and both apply, each to its own layout's requests (`store.readout.applied` counts both).
`POST /v1/tasks/import` and `POST /v1/abstention/tasks/import` load a task of either layout; only a task of another model or rendering is refused.
An export (`GET /v1/tasks?full=1`) carries each task's fingerprint, and so its layout.

**One question, one task per layout.** A server holds one task per question and layout (and per context, [Task context](#task-context)): the same question registered under each layout is two tasks, each serving its own layout's requests, and neither replaces the other, whichever was registered or loaded first.
Registering the question again under the same layout replaces the task the server serves for it, as for any task (Limits, below).
Removing one leaves the other; a request of the removed task's layout then gets `layout_mismatch` if the other is certified.

## Task context

A task can carry a context: a long, fixed text that the server reads before the state of every request of the task.
Requests then send only the part that changes, and the server puts the context in front of it.

**When to use it.** When every request of a task is read against the same text that does not change between requests: a policy, a rubric, a catalogue, a set of definitions, a long instruction.
- The requests get smaller: the context is sent once, at registration.
- Every request of the task starts with the context, so after the first request the context is read from the prefix cache and only the changing part is computed ([below](#the-context-in-the-prefix-cache)).
- A fast table encodes the state alone, so it learns from the part that changes, not from a text every request shares.

Keep in the context only what is the same on every request; what changes belongs in the state.

**Registering.** `POST /v1/tasks` takes an optional `"context"`: a string, or an object or array, which is read as compact JSON (`json.dumps(value, separators=(",", ":"), ensure_ascii=False)`).
The text's leading and trailing whitespace is removed, and the text without it is what is read, stored and hashed: some chat templates trim the start of the content, and a context that started with whitespace would then never match its own tokens in the prompt.
The labelled registration, the dynamic-option one (`"options": "dynamic"`) and a task that learns from traffic take it.
The examples' states are the changing part only: the server reads each example with the context, so the task is fitted, certified and timed on exactly what it serves.

```bash
curl -s http://127.0.0.1:8000/v1/tasks -H 'Content-Type: application/json' \
  -d '{"id": "refund-eligible", "context": "Refund policy: ...", "examples": [...], "certify": {}}'
```

An empty context, or one that is not a string, an object or an array, is refused with 422.
Abstention tasks (`POST /v1/abstention/tasks`) take no context: that route does not read the field (Limits, below).

**The prompt.** The context is joined to the state by a blank line, as the state is joined to the question:
- plain layout: the context, a blank line, the state, a blank line, the question and its options;
- header layout: the header (`Questions that will be asked about the context:` and its lines), the context, a blank line, the state, a blank line, the question and its options.

A state that is an object is rendered as `key: value` lines after the context, as it is without one.
The text is that of the same request whose state is the context, a blank line and the rendered state, built by the row builder every text engine shares (vLLM, and the HF and MLX engines, which inherit it).
The tokens can differ in one place, the padding: a server that pads (the Qwen base on vLLM) sizes the pad so that the context's end, not the state's, falls on the cache block (below).
So a request of a task with a context is not always the same prompt as the request that sends the context in its state, and their answers can differ slightly; on a server that does not pad (the Gemma bases, and MLX by default) the two are the same tokens.
On MLX the same tokens are read in two steps, the context and then the rest of the prefix ([below](#the-context-in-the-prefix-cache)), and that arithmetic can make answers differ slightly from reading the same tokens in one step.
On the Gemma 4 12B on a Mac, probabilities moved by a median of 6e-5 and at most 0.03, and no answer changed.

### The context in the prefix cache

Every request of a task starts with the same tokens: the template's opening, the header under the header layout, the context and its blank line.
The server finds where they end in each request's rows (the context boundary) and keeps everything up to there in the prefix cache once per context, so a request computes only its state and its questions.
- **Padding (vLLM, a base that pads).** The pad before the prompt is sized from the context's end, so the context ends on the block (1,056 tokens on the Qwen base) and the pad is the same on every request of the task.
  The state's own end is then not on the block, since the state is the part that changes.
  This holds under `--pad-policy always` and also under `shared` and `row`, whose single questions are otherwise unpadded or padded to their whole row; `none` never pads.
  `--pad-where between` pads after the state, so the pad is sized as without a context and the context stays where it is.
- **Registration (vLLM).** vLLM keeps a finished request's latest checkpoint only, which lies inside its state or question, not at the context's end.
  So the server registers the context's boundary with a warm-up of the context's tokens and one more, once per context and adapter, as it registers a state's boundary ([running.md](running.md#a-later-question-about-the-same-state---register-boundary)), on the Gemma bases and on the Qwen base.
  `--register-boundary` sets when: `before`, ahead of the first request's questions, in an engine call of its own; `after`, once the first response is out; `off`, never.
  A request that finds the boundary evicted makes the next one register it again.
  The check reads the request's first read from the context's end: the state's warm-up when one goes ahead of the questions (several questions, or a new state under `before`), else its first question.
  The question is not enough there: it reads the context through the warm-up's own checkpoint whether the context was in the cache or not.
  A state's own boundary is registered as for a request without a context.
- **The intent head.** A task with an intent head reads its hidden state from the same rows, and on vLLM's default head mode, where the serving engine reads it, registers the context's boundary the same way, in the same registry, so the letters path and the head find each other's registration.
  With `--head-engine` the second engine has no prefix cache, and every head question reads its whole prompt, the context included.
  On MLX the head reads the letters' own forward, context entry included.
- **MLX.** The context's tokens are their own entry in the prefix cache (`--prefix-cache-mb`), and each request's state continues from a copy of it.
  The prefix is always computed in these two steps, whether the context came from the cache or not, so an answer does not depend on it.
- **What a request computes once the context is cached:** its state, from the context's end (on the Gemma bases from the last 64-token step before it on the 12B, 32-token on the 31B), and its questions.
  With several questions on the Qwen base each question also reads the state's last partial block again (under 1,056 tokens), because the state's end is no longer on a block.
- **What the response says.** `x-decisio-stages` has `context=` on every request read with a context: the context's warm-up sent in this request (vLLM; 0 when the boundary was already registered) or the context's part of the prefix, read or computed (MLX).
  It is summed over every read of the request that read the context: its own engine call, a certified task's questions (`task`) and the intent head's read (`head`), so it lies within those stages.
  On vLLM, `x-decisio-prompt` counts the context in `cached` once it came from the cache.
  On MLX, `/health` counts the context entry's lookups in `prefix_cache.context_hits` and `context_misses`.
- **A state that starts with whitespace** can merge with the context's blank line in the tokenizer, so its tokens do not start with the context's own.
  Such a request is built as it would be without the context boundary (the pad sized from the state's end, as without a context) and reads the context again; the server logs it once per context, and the response's `context_boundary` counts it as `unaligned`.
  The same holds, with a log line that names the chat template, when the template changes the start of the content so that the prompt does not start with the context's own tokens.

**Identity.** The context's sha256 is part of the task's identity.
- The same question registered with another context is another task, kept beside the first (one per context, per layout).
- Registering an id again with another context moves that task to the new context, with a new certificate.
- The context enters the task's fit digest, which the certificate's serving stack names (`calibration.task_fit`): a certificate made under one context does not hold under another, and a record whose context was changed has its certificate suspended (`serving_stack_changed`).
- A task without a context is stored, found and fingerprinted exactly as before.

**Which task a request gets.** A question finds its task by its question (its type and options), as always.
When several tasks ask the same question, each with its own context, a request names the one it is for with `"task": "<id>"` on `/v1/systemone` ([api.md](api.md)).
A request that names none gets the one without a context, which is what such a request got before contexts existed; when every one has a context, it is answered 422 and asked to name one.
`"task"` is refused with 422 for an unknown id, for a task that asks none of the request's questions, for a task fitted under the other layout than the request's (the message names the layout to send), and on the image route; `x-decisio-tasks` says which task answered.
The questions of one request share one read of the state, so they share one context: a request whose questions' tasks have different contexts, or that mixes a question of a task with a context and one read without, is refused with 422; send them as separate requests.
A request that names no task, whose question has a task with a context under the other layout only, is refused with 422 naming the task and the layout to send, as a request naming that task is: the context is the task's instruction, so a request is never read without it.
A request sent after the task was deleted finds no task, and its state is read as sent.

**Limits.**
- A registration whose context, with the question and the template, is over the context window before any state is refused with 422.
- **Abstention reads no context.** An abstention task's threshold is fitted on the state read alone, so a question answered through one is always read without a context: it gets only a task without a context of its question, and a `"task"` that names one with a context is refused with 422.
  A request that mixes such a question with a question of a task with a context is refused as questions with different contexts are.
  Registering an abstention task for a question that a task with a context asks is refused with 422 naming those tasks; register it for a question no such task asks, or delete them first.
  A task with a context registered after an abstention task of its question does not change how the abstention question is read.
- A row over the window is refused with 422 and where its tokens are: the header, the task's context, the state, the question.
  Under the header layout the header limit lists fewer questions first, as it does without a context (the context counts in the row).
- `"over_limit": "truncate"` cuts the state only, never the context; a request whose context and questions alone are over the window is refused.

**The fast table.** A labelled table and one that learns from traffic, of every student kind, encode the request's state alone, never the context.
A learning task logs the features of the state alone.
The certificates, gold and agreement, are made on decisio's own reads, which have the context in the model's prompt, as served.

**Usage.** `usage.input_tokens` counts the context too, with the blank line after it: it is read.

**Storage.** A task's record holds the context's text, so `--tasks-dir` and `--task-store` keep it across restarts and replicas, as the export does.
An import needs the text: a record taken from the listing (no text) is refused, and so is one whose context's sha256 is not its text's.

**Status.** `GET /v1/tasks/{id}`, `GET /v1/tasks` and `/health` (`systemone.readout_tasks.<id>.context`) give the context's `sha256`, its length in characters (`chars`) and in tokens (`tokens`), never its text.
The export (`GET /v1/tasks?full=1`) holds the text, since an import needs it.

**A client that still sends the context in its state** gets it twice: the server cannot tell the context from a state that quotes it, so it reads both, which costs tokens and is not the prompt the task was fitted on.
A state that starts with the task's context (after the same serialisation, leading whitespace aside) is served as sent, with the response header `x-decisio-warning: state starts with the task's context; send only the varying part`, and the server logs it at warning level once per task.
A context repeated elsewhere in the state is not detected.
When you move a client to a task with a context, register the task first, then drop the context from the state the client sends.

## Limits

- **Option list identity.** A task is found by its question's type and its options, keys and descriptions, in order (plus the instructions for yes/no and score questions). Renaming, adding, removing or reordering an option makes a different task, which needs its own registration.
- **Order bound.** The head learns which answer position to trust, so it is bound to the option order it was fitted on, and it is not used with two-order averaging.
- **Options that change per input** cannot be registered: exam-style questions whose answers differ every time have no fixed list to learn, and registration refuses them.
- **The prompt it was fitted under.** A task is fitted on the server's prompt and is applied only by a server with the same prompt and yes/no rendering. The served default changed on 2026-10-03 (the spaced layout, `--prompt-tail spaced`, with yes/no as named letters, `docs/handoffs/tasks.md`): a task registered or exported before then is not applied by the current default, and the server says so at start-up (`WARNING: tasks fitted under another model or rendering are not applied`). Register it again from the same examples; or, to keep using it unchanged, start the server with the compact layout (`--prompt-tail compact --noul-rendering words`).
- **One task per option list per server.** Registration is per server instance; keep it with `--tasks-dir` (or an export and `--tasks-file`), as above.
  Tasks of one question with different contexts are kept side by side, and a request names the one it is for ([Task context](#task-context)).
  It holds across layouts: a task applies to the requests of the layout it was fitted under ([A task under a layout](#a-task-under-a-layout)).
- **Single-engine or second-engine mode.** The default reads the hidden state from the serving engine at the latency cost above; `--head-engine` trades a second weight copy for speed. Both declare the same choice on every stored evaluation item (`tests/unit/test_head_modes.py`), and a task registered in one mode loads in the other.
- **The measured gains are on intent classification.** Your question's gain depends on how far the model's plain reading is from your labels; `examples/tasks/evaluate.py` measures it on your own held-out examples.
