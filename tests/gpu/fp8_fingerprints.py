# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""Run on a card by the identity gate of the Gemma 4 31B FP8 repository (RLCD
experiments/2026-10-07_sd_hf_model_repos/PLAN.md, G1): the FP8 weights the engine holds after loading, as fingerprints.

The server is built as the server builds it (uvicorn left out), so the engine is the served one. Run it once per arm:

    python tests/gpu/fp8_fingerprints.py online.json --base gemma-4-31b       # quantizes the bf16 checkpoint on load
    python tests/gpu/fp8_fingerprints.py stored.json --base <the converted directory or repository>

then `python -m decisio.hub_fp8 compare online.json stored.json`. `python -m decisio.hub_fp8 expected` writes what each
arm should hold, computed on a CPU from the same files, so the engine can be checked against the prediction as well as
the arms against each other.
"""

import argparse
import json
import sys

import uvicorn

from decisio.serve import vllm_engine as sv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--base", required=True, help="a base's key, a decisio repository or a converted directory")
    ap.add_argument("--model", help="the checkpoint, when --base is a key (default: the base's own)")
    a = ap.parse_args()
    got = {}
    make_app = sv.make_app
    sv.make_app = lambda engine, so=None, health=None: got.update(engine=engine) or make_app(engine, so, health)
    uvicorn.run = lambda app, **kw: None
    sys.argv = ["decisio", "--base", a.base, *(["--model", a.model] if a.model else [])]
    sv.main()
    eng = got["engine"]
    prints = eng.llm.collective_rpc("decisio_fp8_fingerprints")[0]
    if not prints:
        sys.exit("FP8: the engine holds no FP8 linear layer of the text model (or this vLLM cannot say)")
    with open(a.out, "w") as f:
        json.dump(prints, f, indent=1, sort_keys=True)
    quant = eng.facts().get("quantization")
    print(f"FP8 FINGERPRINTS {len(prints)} modules ({a.base}, quantization {quant}) -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
