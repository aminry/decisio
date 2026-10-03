# FSD drive lab: a driving simulator that a System One server drives

Vendored from [jev_fsd](https://github.com/BrendanH18/jev_fsd) (MIT, Brendan Hallas); see `ATTRIBUTION.md` for what changed.
A 3D simulator on real OpenStreetMap streets (eight Canadian neighbourhoods are bundled).
Each decision the pilot makes is one `/v1/systemone` request: the state of the car and the road as JSON, and up to three choice questions (the manoeuvre among the candidates, the motion, a route).
The decision panel shows the situation, the candidate manoeuvres, the model's probabilities and the latency.
The upstream README, `UPSTREAM_README.md`, describes the simulator, the drive coach and the scoring.

## Run it

Needs Python 3.10 or newer and a desktop browser with WebGL; the server uses only the standard library.

```
python3 examples/demos/fsd/server.py              # http://127.0.0.1:8322
SYSTEMONE_BASE_URL=http://127.0.0.1:8100 python3 examples/demos/fsd/server.py
```

`SYSTEMONE_BASE_URL` is the root of any server that speaks the System One wire format (a Decisio server, for one), with no key.
`SYSTEMONE_MODEL` is sent with each request when the server needs a model name.
With no server reachable the pilot falls back to the Rules driver for that decision, as upstream does without a key.

Open the page, choose **Explore cities**, pick a drive, and switch the autopilot to the model (button `1`).
`?map=kitsilano&traffic=16&seed=42` picks the map, the traffic and the world seed.
The benchmark (`/bench`) runs seeded scenarios with a chosen driver and saves a score.

## Recording on a machine without a display GPU

The 3D view needs a browser with hardware WebGL.
A container with a compute GPU and no display has software WebGL only, where one frame takes about a second and the simulation crawls.
`?norender=1` skips drawing the 3D view while the simulation, the minimap, the decision card and the inspector run at full speed, and `../tools/record_fsd.py --no-3d` records that.
The clip then shows the decision panels and the latency, not the street.
The benchmark page (`/bench`) never draws anything, so the measurement is unaffected.

## What a decision asks over

The question sets are built in `static/js/brain/state.js`. The manoeuvre question offers the candidate speeds and lane moves the simulator generated for that tick, at most 16 (the validator in `jev/decide.py` allows 32); the motion question has two options and the route question one option per route. A server must accept the largest of these: Ollama's decision route takes 2 to 26 options, Winnow-12B up to 64, Cygnet's decision server, decider-4b and Decisio far more. `option_counts` in every response's `meta` records the options each question asked over.

## Checks

```
python3 -m unittest discover -s tests              # the client, the validator, the benchmark logic
for f in scripts/test_*.mjs; do node $f; done      # the simulator's unit tests
python3 scripts/verify_systemone.py                # 20 saved decision snapshots against the live server
```
