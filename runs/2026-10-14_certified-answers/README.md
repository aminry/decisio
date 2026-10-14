# Certified answers: what they give on recorded answers, and whether a certificate survives a restart

Decisio's `POST /v1/tasks` with a `certify` block (`decisio.readout.certify`, `docs/tasks.md`, "Certified answers").
Run on 2026-10-14. Everything here is read from the files beside this one.

## What certification gives on recorded answers (`certify_public_records.py`)

Decisio's own certification procedure on the 31B suite capture of `runs/2026-10-04_gemma-4-31b/capture_g31.json.gz`, task by task, 1,000 seeded draws of the certification examples (75 of each task's 150 items; 75 and 175 for the 350-item task), `delta` 0.1, `alpha` 0.05 and 0.1.
`results/certification_on_the_31b_suite.md` and `.json` have every row; `results/minimum_examples.json` the fewest held-out examples at which any threshold can be certified (59 at `alpha` 0.05 and `delta` 0.1).
The true risk of a threshold is the error among all of the task's items at or above it (the task's items taken as the population).
The probabilities are the capture's plain readouts at the served temperatures of the same run.

What it shows: with 75 examples the bounded procedure certifies nothing on four of the five tasks shown in the docs at either `alpha`; ToxicChat (answered 97% right) certifies nearly everything at `alpha` 0.1 in three draws of four.
The procedure with no statistical bound (the most lenient threshold whose error on the examples themselves is at most `alpha`) breaks its promise in 35 to 79% of the draws on every task but ToxicChat; the bounded procedure breaks it in at most 0.8% of draws.

    uv run python runs/2026-10-14_certified-answers/certify_public_records.py --out runs/2026-10-14_certified-answers/results

## Does a certificate survive a restart? (`restart_check.py`)

A fresh server process each time, on one backend and model.
The task is a three-queue support-ticket question on tickets the script writes itself (400 labelled examples for registration, half held out of the fit by the server; 200 further tickets to ask), `delta` 0.1.
(A) Registering the task with certification again after each restart, or (B) loading the task exported from the first start (`GET /v1/tasks?full=1`, `POST /v1/tasks/import`); then the same 200 questions.
Compared with the first start: the certificate's threshold, certificate id and serving-stack digest, and per question the choice, the top probability and whether the answer was certified.

| Backend and model | `alpha` | Threshold | Certified on the held-out | Restarts compared | Differences |
| --- | ---: | ---: | --- | --- | --- |
| CPU stand-in, `Qwen/Qwen3-0.6B-Base` (`hf`; the numbers mean nothing about served models) | 0.15 | 0.4268 | 126 of 200 (63%), 12 wrong | A: 2 restarts, B: 3 | none: 0 certified flags, 0 choices, largest top-probability difference 0.0, the same threshold, id and stack digest |
| Mac, `mlx-community/gemma-4-12B-it-6bit` (`mlx`, 12B base, half the tickets also mention a second queue) | 0.15 | 0.9863 | 169 of 200 (84.5%), 17 wrong | A: 1 restart, B: 2 | none, as above |

On the 200 further tickets the first start certified 131 (CPU, 65.5%; 6 wrong) and 169 (Mac, 84.5%; 19 wrong, 11.2%, under `alpha`).
That is one draw, not a test of the promise.

**What this does not say.** Both backends repeat exactly from one fresh start to the next, so a certificate cannot be moved by restart noise here. The Gemma 4 12B on a card does not repeat across fresh starts (`EVAL_CARD.md` section 6.5); whether that moves certified answers is not measured; the case is in the next card runbook.

    uv run python runs/2026-10-14_certified-answers/restart_check.py --backend hf --model Qwen/Qwen3-0.6B-Base --out runs/2026-10-14_certified-answers/results/cpu_stand_in --restarts 3 --alpha 0.15
    uv run --extra mlx python runs/2026-10-14_certified-answers/restart_check.py --backend mlx --base gemma-4-12b --model mlx-community/gemma-4-12B-it-6bit --out runs/2026-10-14_certified-answers/results/mac_mlx --restarts 2 --alpha 0.15 --mixed 0.5

Each result file records the decisio commit it ran on (`decisio_commit`) and whether `src` had changed since.
