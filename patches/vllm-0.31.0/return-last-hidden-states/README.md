<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Copyright contributors to the decisio project -->

# vLLM v0.31.0 patch: return the last hidden states in generation

## Base

- Upstream tag: `v0.31.0` = `db9527a46873454610df6dbedf79a36d6bf1a7f6` (vllm-project/vllm).
- Upstream proposal: [vllm-project/vllm#59543](https://github.com/vllm-project/vllm/pull/59543) (RFC [#59542](https://github.com/vllm-project/vllm/issues/59542)), a Model Runner V2 change that adds `SamplingParams.return_last_hidden_states`.
  This series is that commit carried onto `v0.31.0` (a cherry-pick; the conflicts were in the output processor and its tests, where `main` had moved), branch `rfc/return-last-hidden-states-v0.31.0` of `aminry/vllm`.
- Files here:
  - `0001-*.patch`: `git format-patch v0.31.0..HEAD` (includes upstream's tests, the docs page and the example).
  - `pkg/0001-*.patch`: the same commit restricted to `vllm/`, for `patch -p1` inside site-packages.

## What it does

A generation request that sets `return_last_hidden_states=True` gets, with its final output, the tensor `compute_logits` received for each generated token: `CompletionOutput.last_hidden_states`, `[len(token_ids), hidden_size]`, on the CPU, in the model's dtype.
The engine must be started with `enable_return_last_hidden_states=True` (`--enable-return-last-hidden-states`); a request that asks without it is rejected at validation.
With the engine flag off (the default) the model runner holds no state and does no work, and outputs are the same as stock vLLM's.

This is what decisio's intent head reads today through d + 1 reserved logit columns in three requests (`docs/design/hidden-state-readout.md`): one request that also returns the state replaces them, once this is in the engine decisio runs.

## Not applied by default

The Dockerfile applies only the `suffix-staging` series, and only with `APPLY_PATCHES=1`.
This series is carried and tested against the tagged tree (`tests/unit/test_patches.py`), and decisio's readout does not use it: the reserved-column readout is unchanged.
Dropping that workaround is a change to the served path and needs a card run (the returned rows against the reserved columns, and latency); until that run, nothing in decisio calls this.

## Verified

- The upstream CPU tests on this tag (config, sampling parameters, scheduler, output processor): 304 passed and 7 failed.
  The 7 are `test_stop_token`, with the ungated `unsloth/Llama-3.2-1B` standing in for the gated tokenizer (no Hugging Face token for it here); the same 7 fail on the `main`-based branch of #59543, where the PR description records that they fail without the change too.
  The unpatched `v0.31.0` baseline was not run.
- Applies to the tagged tree in both forms (`tests/unit/test_patches.py`).
- Not run on a card on `v0.31.0`: the GPU test (`tests/v1/sample/test_last_hidden_states.py`) passed on an RTX 4090 for the `main` version of this change, and the deployment numbers in #59543 are from a v0.30.0 backport.
