# An adapter's own temperatures, checked offline against one card session's reads of a tuned 31B (2026-10-17)

**Status: offline check; nothing was served with these files yet.**
The server change (an adapter carries temperatures per question type, and its identity and temperatures are in the serving-stack fingerprint) is checked here against reads that were already made: Gemma 4 31B on R2 (a stored FP8 build of Google's QAT-trained weights) with MeTask's published adapter (`MeTask-Rain/metask-jev-rain-31B-trained` at `2feff076`), read through decisio 0.14.0 at the base's temperatures (choice 4.672, yes/no and score 5.252), on a card session of 2026-10-16.
The arithmetic is the server's: `apply_temperature(p, T_adapter / T_served)` on a readout that was served at `T_served`, which equals `apply_temperature` of the raw readout at `T_adapter`.

## The temperatures

Fitted by `python -m decisio.serve.adapter_temperatures fit` on a 500-item development set of public-source questions (choice 173, yes/no 155, score 172 items; the score items were sent as choice questions), no benchmark item in it, from the raw readout of the first start (by log loss the second start gives the same within 1%):

| Type | Temperature (log loss) | Second start | Score by expected-position error |
| --- | ---: | ---: | ---: |
| choice | 2.141 | 2.155 | |
| yes/no | 3.075 | 3.086 | |
| score | 4.185 | 4.157 | 2.864 (second start 3.160) |

Files: `decisio_adapter.metask.log_loss.json` and `decisio_adapter.metask.expected_position.json`.
Cross-fitted in halves of the development set, calibration moves from the base's temperatures to the adapter's own as follows (expected calibration error, tie-robust): choice 0.344 to 0.036, yes/no 0.093 to 0.058, score 0.103 to 0.077.

## JevBench 231 through decisio's v1.5 reading (the board's public items; start 1)

| Served as | Choice CC | Yes/no CC | Yes/no between 0.2 and 0.8 | Score CC |
| --- | ---: | ---: | ---: | ---: |
| The base's temperatures (what 0.14.0 does) | 85.18 | -45.11 | 70.3% | 58.59 |
| The adapter's own, by log loss | 85.18 | 19.55 | 35.1% | 63.12 |
| The adapter's own, score by expected-position error | 85.18 | 19.55 | 35.1% | 70.09 |
| MeTask's own temperatures (1.297 for choice and yes/no, 1.189 for score, fitted on these 231 items) | 85.18 | 70.55 | 9.5% | 75.60 |
| MeTask's own service as the board ran it | 84.04 | 88.60 | 0.0% | 73.21 |
| The stored 31B (no adapter) | 87.46 | 85.59 | 1.4% | 70.91 |

- **Top answers are the same.** Yes/no: 70 of 74 right for the stored 31B, the adapter and MeTask's service. The yes/no loss is hedging only: the honest probability of an adapter that is 83% accurate on public-source questions sits nearer 0.5 than the 0.2/0.8 band the reading counts as an answer.
  MeTask's service answers every yes/no with one of two constants (0.898 and 0.102), which never falls in the band. decisio does not do that: its yes/no output stays a probability (`--noul-commit` is the operator's choice).
- **Score: the gap to MeTask's service is the temperature, not the readout.** MeTask reads the next-token logits of the level digits, takes the log-sum-exp over each digit's forms, and applies a temperature of 1.189 fitted on the 18 public score items that this reading scores. decisio reads the level letters in one forward pass. At MeTask's own temperature the adapter scores 75.60 on the 18 items, above their service's 73.21.
  The score CC reads the expected position of the returned distribution, so a flat (calibrated) distribution is pulled to the middle level and a sharp one is not. At the log-loss temperature the adapter scores 63.12, 10.1 below their service; at the expected-position temperature fitted on the development set, 70.09, 3.1 below.

## H (776 multiple-choice items, the same session)

| Served as | Accuracy | Log loss | ECE |
| --- | ---: | ---: | ---: |
| Stored 31B | 0.8028 | 0.578 | 0.048 |
| R2, no adapter named | 0.8028 | 0.577 | 0.089 |
| Adapter, base temperatures, start 1 / 2 | 0.8376 / 0.8363 | 0.862 / 0.860 | 0.377 / 0.376 |
| Adapter, its own temperatures, start 1 / 2 | 0.8376 / 0.8363 | 0.567 / 0.565 | 0.174 / 0.173 |
| MeTask's service, start 1 / 2 | 0.8389 / 0.8415 | 0.440 / 0.441 | 0.051 / 0.043 |

Accuracy does not change with a temperature.
Fitted on harder, public-source questions, the adapter's temperatures leave it underconfident on this easier synthetic set (ECE 0.17): a temperature is a statement about the items it was fitted on.

## What this does not show

One adapter, one box, one pair of starts; the development set is the one the temperatures were fitted on, so its figures are cross-fitted; JevBench's 18 score items and 74 yes/no items are small samples; the score temperature by expected-position error is a design choice (the error is flat between 0.3 and 2.5 and falls beyond 3), and it is not the default.
