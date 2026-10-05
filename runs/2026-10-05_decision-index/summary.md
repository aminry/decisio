# Decision Index 0.2.1, the full suite, self-run (2026-10-04 to 2026-10-05)

Three runs of the kit's whole 0.2.1 suite (150,759 requests, 150,317 scoreable) on one RTX PRO 6000 Blackwell Workstation Edition, each base served by its tag with `--base` alone, through the kit's `http` engine at `87d4650`, one request at a time, `--compact`.
Self-run and submitted in [apolinario/decision-index#62](https://github.com/apolinario/decision-index/pull/62), pending the maintainers' validation; not a board number until they add it.

| Base | Tag | Decision Index 0.2.1 | Raw | Breadth | Scoreable ok | Unsupported / errors | Request median / p95 ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| qwen3.6-35b-a3b | v0.4.0 | 48.06 | 60.45 | 46.55 | 150,317 | 0 / 0 | 70.5 / 261.2 |
| gemma-4-12b | v0.4.0 | 49.43 | 61.47 | 47.57 | 150,317 | 0 / 0 | 37.8 / 275.9 |
| gemma-4-31b | v0.6.0 | 57.58 | 67.33 | 56.5 | 150,317 | 0 / 0 | 56.0 / 439.3 |

| Base | Knowledge | Language | Retrieval | Tools | Arts | GPQA Diamond acc. | MMLU-Pro acc. | BANKING77 macro-F1 | CLINC150+OOS macro-F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| qwen3.6-35b-a3b | 32.45 | 51.39 | 53.56 | 68.43 | 31.57 | 0.5102 | 0.6126 | 0.7461 | 0.8223 |
| gemma-4-12b | 31.37 | 51.66 | 58.0 | 71.62 | 32.59 | 0.3878 | 0.5497 | 0.7294 | 0.8716 |
| gemma-4-31b | 45.57 | 61.02 | 62.75 | 75.04 | 37.43 | 0.5204 | 0.6939 | 0.7845 | 0.9015 |

Request latency is the kit's HTTP wall time per request on this card, one at a time, not the maintainers' measurement.
The four benchmarks run earlier on each base (sections 3, 6 and 7 of `EVAL_CARD.md`) reproduce here: exactly on the Qwen and 31B bases, within 0.001 on the 12B except GPQA Diamond (two of 196 items), as its bf16 logits allow.
The 31B run was resumed once after a container restart, with an identical `/health`; only unanswered requests were sent (`manifest.json`, deviations).

The public board's Jev row (`board_jev.json`: [data/index-v0.2.1.json](https://huggingface.co/spaces/multimodalart/jev-decision-index/blob/cdbd1cab3b6eb1811ebd82d2686ad563a4d0fc0d/data/index-v0.2.1.json), generated 2026-09-28): GPQA Diamond 0.7857 and MMLU-Pro 0.827 accuracy, above all three bases on both.

Files: per run, `scores.json` (the kit's) and `environment.json` (the kit's), and `health.json`, the server's `/health` before the first request (and the 31B's `health_resume.json`, after the restart); `checkpoints_v040.json`, `checkpoint_v060_31b.json`, the commits downloaded; `board_jev.json`; `manifest.json`; `files.json`.
