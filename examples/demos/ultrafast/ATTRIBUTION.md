# Attribution

This directory is a modified copy of **jev-ultrafast** by Browser Use.

- Upstream: https://github.com/browser-use/jev-ultrafast
- Commit: 1231850a0bf1a0c0341fe408ef1668dbbfdfac46
- Licence: MIT (the unmodified text is in `LICENSE`)

## What was changed

- `jev_ultrafast/model.py`: the hosted-API call is replaced by `POST <SYSTEMONE_BASE_URL>/v1/systemone` with no key and the model name sent only when `SYSTEMONE_MODEL` is set.
  Each decision now records how many options each question asked over (`option_counts`).
  An opt-in `SYSTEMONE_STRING_DESCRIPTIONS=1` sends each option's description as a JSON string, for servers that only accept text descriptions, and an opt-in `SYSTEMONE_SKIP_SINGLE_OPTION=1` leaves a one-option head out of the request and answers it with its only element, for servers that refuse a question with a single option.
- The text helper (typed content) reads `TEXT_MODEL_BASE_URL`, `TEXT_MODEL`, an optional key and `TEXT_MODEL_EXTRA_JSON`, so any small local OpenAI-compatible model can serve it.
- `jev_ultrafast/demo.py`, `static/`: the port variable is `DEMO_PORT`, and the vendor branding is removed.
- `.env.example`, `pyproject.toml`, `tests/test_agent.py` follow the above; the tests no longer need a key and cover the new behaviour.
- The package directory keeps the name `jev_ultrafast` so that imports in the code and tests are unchanged.
- Removed: the upstream README, its docs and published measurements, the agent notes, the live Google Flights example and its scenario in the inspector, and the scripts that render promotional media or measure live sites, and the lock file.
- Added: `../tools/measure_ultrafast.py`, which measures the agent on local fixture pages.

Nothing in this directory contains measurements taken from any hosted service.

Changes made in this directory are under the licence of the file they change.
