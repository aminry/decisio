# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
# ruff: noqa: E501  (a script that prints sentences)
"""Are an adapter's `v_proj` pairs on the K=V layers copies of the layers' `k_proj` pairs? The merge's rule, on a
real adapter.

    uv run python runs/2026-10-20_tuned-defaults/k_eq_v_copies.py BASE_INDEX.json ADAPTER_DIR   > k_eq_v_copies.txt

BASE_INDEX.json: the 31B base's `model.safetensors.index.json` (only its tensor names are read). Uses the merge's own
loader and comparison (`decisio.hub_fp8.load_adapter`, `k_copy_difference`).
"""

import json
import sys
from pathlib import Path

from decisio import hub_fp8

names = set(json.loads(Path(sys.argv[1]).read_text())["weight_map"])
deltas, scale, sha, layout, orphans = hub_fp8.load_adapter(Path(sys.argv[2]), names)
print(
    f"adapter {sha[:12]}: {len(deltas)} pairs land on a base tensor ({layout} layout, scale {scale}); {len(orphans)} have none"
)
for name in sorted(orphans, key=lambda n: int(n.split("layers.")[1].split(".")[0])):
    d = hub_fp8.k_copy_difference(name, orphans[name], deltas)
    print(f"  {name}: {'bit for bit the layer k_proj pair' if d is None else d}")
