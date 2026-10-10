<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Copyright contributors to the decisio project -->

# Patches to third-party code

Applied at image build or by `patches/apply.sh` to a pip-installed vLLM, never by a plugin at run time.
See `docs/design/vllm-plugin.md` for why this change is a patch series and not a plugin.

| Series | Base | What it is | Needed for correct answers? |
| --- | --- | --- | --- |
| `vllm-0.31.0/suffix-staging` (2 commits) | vLLM tag `v0.31.0`, commit `db9527a46873454610df6dbedf79a36d6bf1a7f6` | Stage only the uncached prompt suffix of prefix-cache hits in Model Runner V2; inert unless `VLLM_SUFFIX_STAGING=1` | No: a latency optimisation at long states; outputs are the same with it on and off |

The series is the vLLM 0.30.0 series ported to 0.31.0 (it applies; it has not been run on a card on 0.31.0).
Measured on vLLM 0.30.0, against stock, on one RTX PRO 6000 Blackwell Max-Q (`runs/2026-09-30_plugin-verification`): 3 to 8% less time per question at 8,000-token states, nothing measurable on a 100-question throughput cell.
An earlier measurement on another card and bench had reported 24%; it did not reproduce.

Each series has the full `git format-patch` files (with upstream's tests) and `pkg/`, the same commits restricted to paths under `vllm/`, for `patch -p1` inside an installed package.
`tests/unit/test_patches.py` checks both forms against the tagged source tree (`scripts/fetch_vllm_source.sh`).
The series is to be proposed upstream; when it lands, it is dropped at the next version bump.
