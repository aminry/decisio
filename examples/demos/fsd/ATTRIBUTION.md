# Attribution

This directory is a vendored and modified copy of **jev_fsd** by Brendan Hallas, <https://github.com/BrendanH18/jev_fsd>, at commit `bbc9012331dfb175a555c76c50487467ce93fb40` (2026-10-01), under the MIT licence (`LICENSE`, copyright (c) 2026 Brendan Hallas).
The upstream project's own README is kept as `UPSTREAM_README.md`.
The upstream project is independent of this one; neither endorses the other.

## What changed

- `jev/client.py`: the hosted-API client (the `typesafe-sdk`, a key, a spend budget) is replaced by a small stdlib client that posts the same request body to `<SYSTEMONE_BASE_URL>/v1/systemone` with no key and no vendor SDK; it reads `x-decisio-server-ms` when the server sends it and records how many options each question asked over.
- `jev/config.py`, `server.py`: `SYSTEMONE_BASE_URL` (default `http://127.0.0.1:8100`) and `SYSTEMONE_MODEL` replace the TypeSafe variables; `/api/status` also reports what the server says about itself.
- `static/`: the places that named the vendor (the brain button, the page title, the bench option, the copy-as-curl snippet, the messages about a key) say "model" and "System One server" instead; the brain id in the code is unchanged.
- `scripts/verify_jev.py` is `scripts/verify_systemone.py`, and its test follows; `tests/test_client.py` is new.
- `pyproject.toml`: no dependency; `uv.lock` is removed with the SDK it pinned.
- `static/js/bench/runner.js`, `static/js/bench/metrics.js`: each drive's result also carries every model decision's latency (`latencies_ms`) and the server's own time (`server_ms`), so a measurement can report percentiles and decisions per second.
- `static/js/main.js`, `static/js/ui/hud.js`, `static/js/common.js`: the autopilot badge, the cockpit label and the decision card's source tag say "model" for the brain whose id is `jev`.
- `server.py`: with `DEMO_DECISION_LOG=<path>` set, each model decision's request body and full answer are appended to that file (`tests/test_decision_log.py`).
- `.env.example`, `jev/envfile.py`, `jev/osm/fetch.py`, `docs/realism-validation.md`: wording that named the vendor or its key.
- The decisions and measurements the upstream documents quote were taken from a hosted service and are not part of this directory; `UPSTREAM_README.md` is kept for how the simulator works.
- Removed to keep the repository small: the screenshots in `docs/` (the Markdown stays).

Everything else (the simulator, the renderer, the map packs, the scoring, the benchmark, the rules driver) is unchanged.

## Third-party material inside it

- Three.js (MIT) under `static/vendor/three`, as upstream bundles it.
- Map packs under `data/maps` are derived from OpenStreetMap, (c) OpenStreetMap contributors, ODbL 1.0 (`docs/maps.md` has the coverage and refresh notes).

Changes made in this directory are under the licence of the file they change.
