<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Copyright contributors to the decisio project -->

# Four demos, judged decision by decision

Decisio served the demos in `examples/demos/` (Pong, a driving simulator, a browser agent, and a support-ticket triage feed) on each of its three bases, and every decision it made was checked against an oracle.
Cygnet and four hosted models played the same games on the same seeds for comparison.
This page has the results, the error analysis behind them, the renderings that were tested on held-out data, and what teaching (task registration) did.

Each base ran from tag v0.6.0 (0ae97ca) as `--base <key> --pad-policy row`: `qwen3.6-35b-a3b`, `gemma-4-12b` and `gemma-4-31b`, in sequence in one session on one NVIDIA RTX PRO 6000 Blackwell Workstation Edition (600 W) on 2026-10-05, with the demo client on the same machine and the same seeds, oracles and renderings for each.
`--pad-policy row` pads the state to the base's block: the Qwen base has one (1,056 tokens), and the Gemma bases have none, so on them the flag has no effect.
It is stated under every clip and table.
Cygnet and the hosted models were measured on 2026-10-03, Cygnet on a card of the same type (500 W).

The Qwen base was also the first round's player on 2026-10-03, served from main at 62dfe1f with the same prompt layout.
Its counts in Pong, lockstep driving, the browser agent and triage are identical to that round's, and its latencies moved by a few ms; realtime driving differs, because there the latency decides which moments are asked about.

## Watch it play

Every clip is rendered from a recorded trajectory (`examples/demos/tools/trajectory.py`), at the speed the run happened.
Nothing in a clip is re-simulated or re-driven; the caption under it names the model, the card, the median latency of that run and the padding.
Every clip ends on a scoreboard card with that demo's totals for every player, the clip's own player highlighted.

| demo | Qwen3.6-35B-A3B base | Gemma 4 12B base | Gemma 4 31B base |
| --- | --- | --- | --- |
| Pong, the three bases on the same serve (seed 20260918) | [MP4](media/pong_three_bases.mp4) · [GIF](media/pong_three_bases.gif) | the same clip | the same clip |
| Driving, one drive in real time (scenario s1-1) | [MP4](media/driving_qwen3.6-35b-a3b.mp4) | [MP4](media/driving_gemma-4-12b.mp4) | [MP4](media/driving_gemma-4-31b.mp4) · [GIF](media/driving_gemma-4-31b.gif) (an excerpt) |
| Browser agent, the travel task | [MP4](media/browser_travel_qwen3.6-35b-a3b.mp4) · [GIF](media/browser_travel_qwen3.6-35b-a3b.gif) | [MP4](media/browser_travel_gemma-4-12b.mp4) · [GIF](media/browser_travel_gemma-4-12b.gif) | [MP4](media/browser_travel_gemma-4-31b.mp4) · [GIF](media/browser_travel_gemma-4-31b.gif) |
| Browser agent, the reading room | [MP4](media/browser_reading_qwen3.6-35b-a3b.mp4) | [MP4](media/browser_reading_gemma-4-12b.mp4) | [MP4](media/browser_reading_gemma-4-31b.mp4) |
| Triage, the same tickets before and after registering 200 labelled tickets | [MP4](media/triage_qwen3.6-35b-a3b.mp4) | [MP4](media/triage_gemma-4-12b.mp4) · [GIF](media/triage_gemma-4-12b.gif) (an excerpt) | [MP4](media/triage_gemma-4-31b.mp4) |

| demo | Cygnet beside the Qwen base |
| --- | --- |
| Driving, the same drive side by side | [MP4](media/driving_qwen3.6-35b-a3b_beside_cygnet.mp4) |
| Browser agent, the travel task side by side | [MP4](media/browser_travel_qwen3.6-35b-a3b_beside_cygnet.mp4) |

The hosted players' clips are rendered from the same final runs as their table rows, on the same seeds as Decisio's clips, at the speed the run happened, so their latency shows.
They show each player's moves only: every option is drawn as chosen or not, with no probability.
Jev 1.13 returns a probability for every option; its clips show its moves and not its probabilities.
Each caption carries the exact model ID, the date, the route and serving provider, the run's median latency and how many requests had no answer in time, and says "model output" and "AI-generated", rendered by the decisio project.

| demo | Jev 1.13 through its public API | anthropic/claude-haiku-4.5 | openai/gpt-6-luna |
| --- | --- | --- | --- |
| Pong (seed 20260918) | [MP4](media/pong_jev.mp4) | [MP4](media/pong_haiku.mp4) | [MP4](media/pong_luna.mp4) |
| Driving, one drive in real time (scenario s1-1) | [MP4](media/driving_jev.mp4) | [MP4](media/driving_haiku.mp4) | [MP4](media/driving_luna.mp4) |
| Browser agent, the travel task | [MP4](media/browser_travel_jev.mp4) | [MP4](media/browser_travel_haiku.mp4) | [MP4](media/browser_travel_luna.mp4) |

There is no clip of google/gemini-3.8-flash: Google's Gemini API terms forbid publicly displaying the content it returns, so Gemini appears in the tables and on the scoreboard cards only.
The browser clips hold each step for at least 0.8 s so it can be read; their captions say so.
The driving clips draw map data from OpenStreetMap (© OpenStreetMap contributors, ODbL).

## How each decision is judged

| demo | the oracle | what it is not |
| --- | --- | --- |
| Pong | a perfect-information paddle: the true ball's crossing of the paddle's plane after wall bounces, and a move toward it, staying inside half a step (6 units) | while the ball moves away there is nothing to judge; those decisions are counted apart |
| driving | the demo's own rules driver on the same snapshot and the same candidates: the motion (drive or stop) and the manoeuvre | a good driver, not a perfect one; the drive's own score (arrive with no collision, red light, rolled stop sign, failure to yield or a second off the road) is reported beside it |
| browser agent | the set of actions a careful user could take at each step of two fixture tasks; 272 labelled steps from three places | a label for the operation and the target on two small fixtures |
| triage | the ticket's true queue | synthetic, templated tickets |

A disagreement is classed, in this order:

1. **Latency effect**: the right decision arrived too late (driving only: the answer equals the oracle on the snapshot it was asked about but not when it was applied, or the demo's 1.5 s timeout replaced it).
2. **Prompt or state effect**: a written detector per demo, defined before any fix was tested: the model followed the question's own wording where it differs from the oracle (Pong's 5-unit band against the game's 6; the driving state's "at" a light up to 6 m before the line, where the rules driver stops within 1.5 m, counted "as worded" and not as an error), or the state lacked what the decision needs (the browser agent: DONE while a typed search is still unsubmitted).
3. **Near-tie**: the chosen option's probability minus the oracle option's is below 0.10.
4. **Judgement error**: anything else; reported, not tuned away.

The hosted models answer with one option, not a distribution, so their misses cannot be split into near-tie and judgement.

## Results

Decisio's three bases and Cygnet ran on the card; the hosted models ran from a Mac on the same seeds and renderings.
Driving compares every player on the same three drives (suite seed 1, scenarios 1 to 3); the card players also drove more, given below each table.
Intervals are 95% bootstrap intervals over games or drives; latency is the client's round trip, p50 / p95, in ms.

### The players

| row | model ID | date | route | serving provider |
| --- | --- | --- | --- | --- |
| Decisio, Qwen3.6-35B-A3B base | Qwen/Qwen3.6-35B-A3B-FP8 @95a723d0 served by decisio v0.6.0 (`--base qwen3.6-35b-a3b --pad-policy row`) | 2026-10-05 | local server on the card | |
| Decisio, Gemma 4 12B base | google/gemma-4-12B-it @707f0a3b served by decisio v0.6.0 (`--base gemma-4-12b --pad-policy row`) | 2026-10-05 | local server on the card | |
| Decisio, Gemma 4 31B base | google/gemma-4-31B-it @842da379, FP8 on load, served by decisio v0.6.0 (`--base gemma-4-31b --pad-policy row`) | 2026-10-05 | local server on the card | |
| Cygnet | google/gemma-4-12B-it @707f0a3b with the cygnet-recipe decision server @3cf591c6 (T 3.4) | 2026-10-03 | local server on the card | |
| Jev 1.13 through its public API | jev-1.13.0 | 2026-10-03 | TypeSafe public API | TypeSafe |
| google/gemini-3.8-flash | google/gemini-3.8-flash | 2026-10-03 | OpenRouter | Google AI Studio |
| anthropic/claude-haiku-4.5 | anthropic/claude-haiku-4.5 | 2026-10-03 | OpenRouter | Anthropic |
| openai/gpt-6-luna | openai/gpt-6-luna | 2026-10-03 | OpenRouter | OpenAI |

The hosted models' latencies include the network from the Mac; OpenRouter's own added time was measured at 130 to 155 ms per call.

### Browser agent (10 runs of each task)

| player | travel completed | travel decisions right | reading room completed | reading room decisions right | latency p50 / p95 |
| --- | --- | --- | --- | --- | --- |
| Decisio, Qwen3.6-35B-A3B base | 10/10 | 50/70 | 10/10 | 80/140 | 141 / 174 |
| Decisio, Gemma 4 12B base | 10/10 | 60/60 | 10/10 | 20/20 | 141 / 275 |
| Decisio, Gemma 4 31B base | 10/10 | 40/50 | 10/10 | 20/20 | 184 / 417 |
| Cygnet | 10/10 | 60/60 | 10/10 | 20/20 | 58 / 253 |
| Jev 1.13 through its public API | 10/10 | 60/60 | 10/10 | 20/20 | 130 / 185 |
| google/gemini-3.8-flash | 10/10 | 60/60 | 10/10 | 20/20 | 1308 / 2030 |
| anthropic/claude-haiku-4.5 | 10/10 | 40/50 | 10/10 | 20/20 | 1248 / 1767 |
| openai/gpt-6-luna | 10/10 | 60/68 | 10/10 | 20/20 | 2878 / 4464 |

On the 272 labelled steps (the operation question alone), the answer is inside the acceptable set on 243 (89%) for the Qwen base, 269 (99%) for Gemma 4 12B, 248 (91%) for Gemma 4 31B and 268 (99%) for Cygnet.

### Driving

| player | lockstep: drives passed | lockstep: motion errors / motion answers | realtime: drives passed | realtime: motion errors / latency effects | decisions per second (lockstep, realtime) | latency p50 / p95 (lockstep) |
| --- | --- | --- | --- | --- | --- | --- |
| Decisio, Qwen3.6-35B-A3B base | 3 of 3 | 8 / 545 | 3 of 3 | 5 / 2 | 7.74, 3.12 | 102 / 158 |
| Decisio, Gemma 4 12B base | 3 of 3 | 25 / 792 | 3 of 3 | 6 / 4 | 4.01, 2.99 | 218 / 343 |
| Decisio, Gemma 4 31B base | 3 of 3 | 5 / 568 | 3 of 3 | 0 / 3 | 2.70, 2.37 | 341 / 569 |
| Cygnet | 3 of 3 | 0 / 574 | 3 of 3 | 0 / 7 | 2.49, 2.30 | 371 / 693 |
| Jev 1.13 through its public API | 3 of 3 | 0 / 711 | 3 of 3 | 0 / 2 | 6.58, 3.03 | 126 / 178 |
| google/gemini-3.8-flash | 3 of 3 | 0 / 564 | 1 of 3 | 1 / 98 | 0.59, 0.40 | 1319 / 3853 |
| anthropic/claude-haiku-4.5 | 3 of 3 | 0 / 609 | 3 of 3 | 0 / 12 | 0.89, 0.96 | 1026 / 1574 |
| openai/gpt-6-luna | 3 of 3 | 0 / 848 | 1 of 3 | 0 / 109 | 0.30, 0.08 | 3195 / 5911 |

Lockstep waits for every decision, so it measures the decisions alone; realtime keeps the car moving while a request is in flight, as the demo does, so a slow answer arrives too late.
Gemini and Luna lose drives in realtime to latency, not to judgement: about a second per decision, and the demo's 1.5 s timeout hands most of Luna's decisions to the rules driver.
Over all of their drives, in lockstep and in realtime:

- the Qwen base passed 16 of 18 and 9 of 9 (in lockstep one collision in s2-4 and one drive off the road in s2-5; the rules driver alone passes both);
- Gemma 4 12B passed 18 of 18 and 9 of 9;
- Gemma 4 31B passed 18 of 18 and 8 of 9 (in realtime a collision in s2-2);
- Cygnet passed 6 of 6 and 9 of 9.

### Pong (5 games of up to 45 s, seeds 20260917 to 20260921)

| player | agrees with the oracle while the ball approaches | decisions on approach | decisions per second | latency p50 / p95 | returns per game |
| --- | --- | --- | --- | --- | --- |
| Decisio, Qwen3.6-35B-A3B base | 60% [53, 65] | 531 | 22.9 | 44 / 47 | 22.8 [12.4, 32.8] |
| Decisio, Gemma 4 12B base | 62% [56, 66] | 539 | 22.8 | 48 / 50 | 23.2 [21.2, 24.8] |
| Decisio, Gemma 4 31B base | 85% [82, 88] | 1679 | 15.0 | 68 / 69 | 84.2 [83.0, 86.6] |
| Cygnet | 47% [41, 53] | 459 | 22.9 | 41 / 69 | 19.2 [12.8, 26.4] |
| Jev 1.13 through its public API | 96% [94, 98] | 913 | 8.2 | 116 / 166 | 45.8 [45.4, 46.0] |
| google/gemini-3.8-flash | 98% [94, 100] | 95 | 0.87 | 1131 / 1570 | 5.0 |
| anthropic/claude-haiku-4.5 | 69% [55, 86] | 139 | 1.29 | 745 / 1000 | 6.8 [6.4, 7.0] |
| openai/gpt-6-luna | 98% [95, 100] | 58 | 0.62 | 1460 / 2724 | 3.4 [3.0, 3.8] |

A game ends when the model's paddle misses five times; the ball advances one step per decision, so returns per game reward fast answers as much as right ones.
Gemma 4 31B played all five games to the 45 s limit, which is why it was asked three times as often as the other bases.

### Triage (400 held-out tickets over 20 queues)

| base | plain question | after registering 200 labelled tickets | paired difference | macro-F1, plain / registered | latency p50, plain / registered |
| --- | --- | --- | --- | --- | --- |
| Qwen3.6-35B-A3B | 91.0% [88.2, 93.8] | 92.5% [89.8, 95.0] | +1.5 points [+0.2, +3.0]; 7 answers fixed, 1 broken | 0.912 / 0.926 | 42 / 120 |
| Gemma 4 12B | 93.8% [91.2, 96.0] | 95.2% [93.0, 97.2] | +1.5 points [+0.0, +3.0]; 8 fixed, 2 broken | 0.938 / 0.952 | 50 / 145 |
| Gemma 4 31B | 97.0% [95.2, 98.5] | 97.2% [95.5, 98.8] | +0.2 points [-0.5, +1.2]; 2 fixed, 1 broken | 0.970 / 0.973 | 71 / 262 |

Registration was repeated on each base and took 33 s, 35 s and 69 s.
On the Qwen base it kept both the intent head and the calibration; on the Gemma bases it kept the head only (each is kept only when cross-validation on the examples shows a gain).
The head costs latency: a registered answer reads the model's hidden state as well as its letters.
The plain question is already right on 91% to 97% of these synthetic tickets, so the lift is small, and smallest where the plain answer is best.

## Decisio's errors

| demo | base | decisions judged | agree | latency | prompt or state | near-tie | judgement | what the errors are |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Pong, ball approaching | Qwen3.6-35B-A3B | 531 | 330 | 0 by construction | 12 | 63 | 126 | mostly "stay" where the perfect paddle moves down (85 of 201); 21 of the misses turned a return into a lost point |
| | Gemma 4 12B | 539 | 332 | 0 by construction | 4 | 13 | 190 | mostly "down" where the perfect paddle moves up (97 of 207); 24 turned a return into a lost point |
| | Gemma 4 31B | 1,679 | 1,421 | 0 by construction | 13 | 6 | 239 | mostly "stay" where the perfect paddle moves down (210 of 258); across its misses the ball crosses a median 7 units from the paddle, just outside the 6-unit stay band; none cost a point |
| driving lockstep, motion (18 drives) | Qwen3.6-35B-A3B | 4,173 | 2,934 + 1,207 as worded | 0 | | 13 | 19 | 30 of 32 a stop where the rules driver goes |
| | Gemma 4 12B | 5,061 | 3,927 + 1,108 as worded | 0 | | 1 | 25 | all 26 a stop where the rules driver goes |
| | Gemma 4 31B | 4,253 | 3,055 + 1,182 as worded | 0 | | 0 | 16 | 14 of 16 a stop where the rules driver goes |
| driving realtime, motion (9 drives) | Qwen3.6-35B-A3B | 1,697 | 1,077 + 607 as worded | 7 | | 1 | 5 | all 6 a stop where the rules driver goes |
| | Gemma 4 12B | 2,087 | 1,471 + 598 as worded | 12 | | 0 | 6 | all 6 a stop where the rules driver goes |
| | Gemma 4 31B | 1,554 | 836 + 702 as worded | 12 | | 1 | 3 | all 4 a stop where the rules driver goes |
| browser, labelled steps | Qwen3.6-35B-A3B | 272 | 243 | 0 by construction | 5 | 4 | 20 | DONE before the search is applied; a click on the wrong control |
| | Gemma 4 12B | 272 | 269 | 0 by construction | 0 | 1 | 2 | |
| | Gemma 4 31B | 272 | 248 | 0 by construction | 17 | 0 | 7 | mostly DONE while a typed search is still unsubmitted |

In the live browser runs every base completes both tasks every time, but not always by the shortest path.
On the travel task the Qwen base re-selects the category once and opens Casa Flora before submitting the typed destination, and Gemma 4 31B opens Casa Flora without submitting it; the fixture's own check does not test the search, so the runs count as completed and the oracle counts those steps as misses.
In the reading room the Qwen base opens the article and then clicks back to the list about six times (its probability for DONE on the article is 0.27 to 0.42) before DONE wins.
Gemma 4 12B takes the shortest path in both tasks in every run, and Gemma 4 31B in the reading room.

## Renderings tested on held-out data

These tests ran on the Qwen base in the first round, before the bases session; they were not repeated on the Gemma bases.
A change to a demo's state or question is kept only when its gain on held-out seeds or places is clear of zero (the paired 95% interval excludes it) and it adds observations only: no verdict, no recommendation, no score per option.

| demo | change | held-out data | result | kept |
| --- | --- | --- | --- | --- |
| Pong | the arrival in words ("the ball will arrive 14.2 units above the centre of your paddle") instead of coordinates, and the moves defined ("move the paddle 12 units toward the top of the screen") | seeds 40000 to 40009 | -32.3 points [-40.8, -23.9]: Decisio answers "stay" almost always | no |
| Pong | `offset`, the intercept minus the paddle's position, added to the state | the same | +4.9 points [-4.9, +15.3] | no |
| Pong | the stay band worded as the game's 6 units | the same | -1.9 points [-11.4, +9.8] | no |
| driving | after a completed stop, once the car is past the line, the state says "the car is already in the junction, N m past the stop line" instead of telling it to hold for cross traffic | suite seeds 201 to 203, 18 drives, lockstep | motion errors 101 to 59, -2.33 per drive [-5.17, -0.44]; drives passed 14 of 18 both | yes |
| browser agent | a typed field marked submitted or not (from the page's own submit events), filters marked applied, and a shorter action history | the 272 labelled steps of three places, and 10 + 10 live runs | all steps -3.3 points [-7.4, +0.7] (Casa Flora +0.8, the Glasshouse -8.3 [-15.0, -1.7]); the reading room then loops to the step limit in 10 of 10 runs | no |
| browser agent | DONE's description adds that an unsubmitted search is not applied | the labelled steps of the two places it was not designed on (it came from a Casa Flora step) | the Glasshouse and Serra Lodge 88% to 86% and 91% to 88%; Casa Flora itself +5.0 points [+0.8, +10.0]; all 272 steps +0.7 [-2.9, +4.0] | no |

The driving change fixes a sentence that contradicted the state it sits in: the default text tells the car to hold for cross traffic even when the car is already in the junction, where the rules driver goes on.
The final runs above use it; Pong and the browser agent use their default renderings.

## Pong's limit

With the default prompt the Pong question asks the model to compare two numbers, the intercept and the paddle's position, and choose a direction.
The Qwen base agrees with the perfect paddle on 60% of decisions while the ball approaches, on the evaluated seeds and on held-out seeds alike; given the signed distance it reaches 66% and reads the sign backwards on most of what remains, and given the relationship in words it stops moving.
Gemma 4 12B is at 62%, and Gemma 4 31B reaches 85%, its misses mostly a "stay" close to the edge of the stay band.
This is recorded as a limit of the models on numeric comparison, not of the prompt, and the larger model lifts it.
The same question is answered at 96% by Jev 1.13 through its public API in about 120 ms per decision, and at 98% by Gemini 3.8 Flash and GPT 6 Luna in 1.1 to 1.5 s.

## Teaching

In the games, on the Qwen base in the first round, registering the questions with fixed option lists (Pong's move, the driving motion, the browser operation) from oracle-labelled examples on seeds disjoint from the evaluated ones gave no gain.
The server declined the Pong and driving calibrations because cross-validation on their examples showed none, so those questions were answered exactly as before.
It kept the browser operation's calibration, which moved probability toward the options the examples used most and scored -1.8 points [-4.4, +0.4] on the labelled steps, with the live runs unchanged; seven per-option biases cannot make a correction that depends on the page.
The driving manoeuvre and the browser's element target cannot be registered: their options change every step.
On triage, a fixed 20-queue question, registration from 10 labelled tickets per queue lifted held-out accuracy by 1.5 points on the Qwen base ([+0.2, +3.0], from 91.0%) and on Gemma 4 12B ([+0.0, +3.0], from 93.8%), and by 0.2 points [-0.5, +1.2] on Gemma 4 31B, from 97.0%.

## Reproducing

`examples/demos/README.md` lists the commands: each `tools/measure_*.py` writes a run record with trajectories, and each `tools/render_*.py` draws a clip from them.
The hosted models were reached through `examples/demos/tools/systemone_gateway.py`, which renders a System One request as a chat request with a structured answer; the OpenRouter calls pinned the first-party provider, disallowed fallbacks and denied data collection.
