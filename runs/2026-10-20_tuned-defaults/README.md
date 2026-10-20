# The shipping path of a tuned model, offline

What this shows: the whole path from an adapter and a bf16 base to a repository that serves the adapter's temperatures and commit setting to a plain request, run on a tiny synthetic checkpoint with the 31B's structure on a CPU. It checks the machinery (`decisio.hub_fp8 merge`, `decisio.hub tuned-config` and `check`, `resolve_base`, the served temperature per question type and choice band); it measures no model, and its adapter and its temperatures are random numbers.

```bash
uv run python runs/2026-10-20_tuned-defaults/pipeline.py > output.txt
```

`output.txt` is the output of that command on the commit recorded in `manifest.json` (the temporary directory in one line is replaced by a relative name).

What it finds, in order: a null merge equals `convert` byte for byte; the merge applies every pair of the adapter (text-only layout mapped to the base's multimodal names) and the written and loaded fingerprints are identical; the fp32 sum differs from the weight in every entry where a bf16 round trip differs in none; `tuned-config` writes a `decisio.decision_config/2` file that `check` accepts together with the file digests; `resolve_base` applies the repository's temperatures, bands and commit setting, and `--temperature-choice` and `--no-noul-commit` override them (the bands go with the former); a question is served at the temperature of its band (2 to 10, 11 to 26, 27 to 255 options).
