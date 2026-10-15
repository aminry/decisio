"""S12 part F read from Lab 2's files (RLCD experiments/2026-10-15_lab2_s12_session/results/box/F/): what differs between the
stored FP8 repository (A) and Google's bf16 quantized on load (B) on vLLM 0.31.0, tensor by tensor and answer by answer.

    python3 -I f_analysis.py <the F directory> > f_analysis.json
"""

import collections
import json
import statistics
import sys
from pathlib import Path

F = Path(sys.argv[1])
fa, fb = json.loads((F / "f_a.fp.json").read_text()), json.loads((F / "f_b.fp.json").read_text())
tensors = [k for k in fa if k != "ALL"]
kinds = collections.Counter()
for k in tensors:
    x, y = fa[k], fb[k]
    same_bytes = x["sha256"] == y["sha256"] and x["dtype"] == y["dtype"]
    same_shape = x["shape"] == y["shape"]
    name = k.split(".")[-1]
    if k.endswith("weight_scale"):
        name = "weight_scale"
    kinds[(name if (not same_bytes or not same_shape) else "equal", same_bytes, same_shape, tuple(x["shape"]), tuple(y["shape"]))] += 1
out = {
    "tensors": len(tensors),
    "bytes_differ": sum(fa[k]["sha256"] != fb[k]["sha256"] or fa[k]["dtype"] != fb[k]["dtype"] for k in tensors),
    "fp8_weight_tensors": sum(fa[k]["dtype"].startswith("torch.float8") for k in tensors),
    "fp8_weight_tensors_byte_equal": sum(fa[k]["dtype"].startswith("torch.float8") and fa[k]["sha256"] == fb[k]["sha256"] for k in tensors),
    "differ_only_in_shape": [
        {"kind": n, "count": c, "A_shape": list(sa), "B_shape": list(sb)} for (n, sb_, sh, sa, sb), c in sorted(kinds.items()) if n != "equal" for sb_ in [sb_]
    ],
}
A, B = json.loads((F / "f_a.json").read_text())["answers"], json.loads((F / "f_b.json").read_text())["answers"]
by = collections.defaultdict(list)
flips = collections.Counter()
exact = 0
for k in A:
    a, b = A[k], B[k]
    if a["type"] == "noul":
        dp = abs(a["noul"] - b["noul"])
        flip = (a["noul"] > 0.5) != (b["noul"] > 0.5)
    else:
        dp = max(abs(a["probabilities"][o] - b["probabilities"][o]) for o in a["probabilities"])
        flip = a["choice"] != b["choice"]
    exact += dp == 0
    by[k.split(":")[0]].append(dp)
    flips[k.split(":")[0]] += flip


def q(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p * len(xs)))]


allv = [v for xs in by.values() for v in xs]
out["answers"] = {
    "n": len(allv),
    "exactly_equal": exact,
    "median_dp": statistics.median(allv),
    "p90_dp": q(allv, 0.9),
    "max_dp": max(allv),
    "top_answer_changes": sum(flips.values()),
    "by_task": {t: {"n": len(v), "median_dp": statistics.median(v), "max_dp": max(v), "top_answer_changes": flips[t]} for t, v in sorted(by.items())},
}
print(json.dumps(out, indent=1))
