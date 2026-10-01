<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Copyright contributors to the decisio project -->

# Three demos on a System One server

Three open-source projects that were built against a hosted decision API with a System One request format, changed to talk to any `/v1/systemone` server with no key: a decisio server, or another model served by its own recipe.
Each directory keeps its original licence and an `ATTRIBUTION.md` with the upstream commit and the list of changes.

| demo | upstream | licence | one decision is |
| --- | --- | --- | --- |
| `pong/` | ably-labs/jev-pong | Apache-2.0 | one request with the ball state and one 3-option question; one answer moves the ball one step |
| `fsd/` | BrendanH18/jev_fsd | MIT | one request with the car and the road as JSON and up to three questions, at most 16 options |
| `ultrafast/` | browser-use/jev-ultrafast | MIT | one request per browser step: an operation question and a target question per kind of element on the page |

The clients keep their behaviour.
The only changes are the endpoint (`SYSTEMONE_BASE_URL`, no key), the places that named a vendor, a few counters so a measurement can read decisions and latencies, and the opt-ins described in each README.

## Measure and record

`tools/` holds the scripts that run a demo against a server and keep a run record under `runs/` (a manifest, a `files.json` with every file's sha256, the per-run rows and the tables).
`pip install -r tools/requirements.txt`, then:

```sh
python tools/measure_pong.py --lanes-file lanes.json --runs 5 --seconds 45 --video
python tools/measure_pong.py --lanes-file lanes.json --only sys1 --runs 5   # one model at a time on a single card
python tools/compose_pong.py --replay runs/a/replay_run1.json --replay runs/b/replay_run1.json --out runs/c/media
python tools/measure_fsd.py --base-url http://127.0.0.1:8100 --label Decisio --runs 3 --count 6
python tools/measure_ultrafast.py --label Decisio --runs 10
python tools/record_fsd.py --base-url http://127.0.0.1:8100 --label Decisio --out runs/<run>/media
python tools/record_ultrafast.py --label Decisio --out runs/<run>/media
```

What is measured, per model and demo:

- decisions per second, and p50 and p95 of the per-decision latency (the System One request, timed in the client);
- the demo's own score: Pong's rallies (returns of the ball by the model's paddle), the driving rules score (the share of drives that arrive with no collision, red light, rolled stop sign, failure to yield or a second off the road), the browser agent's task completion and completion time;
- 95% bootstrap intervals over the repeated runs (Pong, driving, browser agent each repeat with different seeds or runs); latencies are pooled over every decision.

Pong's lanes run at the same instant when their servers can be up together; on a single card they are recorded one model at a time with the same seed and merged onto one page by `compose_pong.py`.
The client and the server run on the same machine, so the latency on screen and in the record is the server's, with a loopback round trip.
Videos are recorded in headless Chrome driven by Playwright (`tools/demo_recorder.py`): a WebM, an MP4 with a caption strip under the picture (model, card, median latency) and a GIF under 3 MB.
Every manifest says which card ran the server.
A number taken on a CPU stand-in or on the Ollama listing is a check that the demo runs, not a measurement of the model, and is labelled so.

## How many options each demo asks over

A model can serve a demo only if its option limit covers the largest question the demo asks.

| model | most options per question | Pong (3) | driving (16) | browser agent (9 on the fixtures, more on real pages) |
| --- | --- | --- | --- | --- |
| Decisio | 255 | yes | yes | yes |
| decider-4b v2 | 255 | yes | yes | yes |
| Cygnet decision server | 255 (grouped passes) | yes | yes | yes |
| Winnow-12B | 64 | yes | yes | up to 64 |
| Jev-Omni head | 256 slots, quality established to 20 | yes | yes | up to 20 |
| Nimble 9B | 26 on its System One route, 255 on its scorer | yes | yes | up to 26 (255 with the scorer) |
| Ollama listing of Decisio | 26, and at least 2 | yes | yes | up to 26, with two opt-ins |

The limits come from each model's published serving path and are checked on the card before a run; a model that fails the check is excluded from that demo, not tuned.
Jev-Omni and Nimble publish an in-process library only, so a thin System One server in front of the library is needed to run the clients against them.

## What one pass cannot do

A decision here is one forward pass over the state and one choice among the listed options: the model reads, it does not search, and it does not look ahead.
That suits the demos above, where the state already shows what the next step needs and the options are a short list.
It does not suit a task whose answer comes from searching or planning.
In private tests on two small games, a maze chase against two chasers (100 seeded mazes) and Hangman (200 held-out words), Decisio won 0% of the mazes from the grid alone and 13.0% [7.0, 20.0] when the state also listed the next two cells and the distances; decider-4b v2 won 0% and 9.0% [4.0, 15.0]; a breadth-first-search player won 63.0% [53.0, 72.0].
In Hangman, which is won by searching a dictionary for the letter that splits the remaining words best, Decisio won 13.0% [8.5, 18.0] of the words and decider-4b v2 2.5% [0.5, 5.0], against 91.5% [87.5, 95.0] for an entropy solver.
Registering labelled examples through `POST /v1/tasks` moved the letter probabilities toward the solver's but did not raise the win rate beyond its interval (-4.5 points [-10.0, +1.0] and +1.5 [-2.5, +5.5] in the two Hangman states).
A decision model is a fast, calibrated chooser for closed questions; a search or a plan belongs in code around it.

