# Pong lanes

Pong where the ball moves one step per model decision.
Each lane asks a different model the same question about the same serve, so a slow model means a slow ball.
The demo is a modified copy of jev-pong (see `ATTRIBUTION.md`), changed to talk to any System One server.

## What one decision is

The client sends the ball, the paddle and the intercept point as a JSON state, plus one choice question named `move` with the options `up`, `down` and `stay`.
The server answers with the choice.
The latency on screen is the round trip of that one request, timed in the recorder.
Run the client and the server on the same machine, so that the number is the server's.

## Lanes

Lanes are listed in `lanes.example.json` (copy it, then set `PONG_LANES_FILE`, or put the JSON in `PONG_LANES`).
There are two kinds.

- `systemone`: `POST <baseUrl>/v1/systemone`, no key. The three options fit every System One server, including the Ollama listing (2 to 26 options).
- `chat`: any OpenAI-compatible server, asked for the same move as a structured object.
  This is the original demo's chat-model lane, with its hosted gateway replaced by a local text model.
  Use `extraBody` to switch thinking off where the model has a switch: `{"reasoningEffort": "none"}` for Ollama, `{"chat_template_kwargs": {"enable_thinking": false}}` for vLLM.

The lane with id `sys1` is drawn in the accent colour.

## Run it

```sh
pnpm install
PONG_LANES_FILE=lanes.json RECORD_MS=45000 pnpm record   # writes public/replay.json and public/replay-stats.json
pnpm dev -p 3100                                         # then open http://localhost:3100/?clean=1
```

`?clean=1` shows the lanes and nothing else, for recording.
`pnpm test`, `pnpm typecheck` and `pnpm lint` check the code.
`../tools/demo_recorder.py` records the page to WebM, MP4 and a GIF under 3 MB.

## What the numbers mean

`replay-stats.json` has, per lane, decisions per second, mean and p95 latency of the decision call, returns (the model's paddle returned the ball) and misses.
The model's score is its returns over the recorded window.
Repeat the recording with different seeds (`SEED=...`) to get an interval.
