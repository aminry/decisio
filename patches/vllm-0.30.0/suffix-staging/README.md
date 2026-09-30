<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Copyright contributors to the decisio project -->

# vLLM v0.30.0 patch: suffix-only prompt staging

## Base

- Upstream tag: `v0.30.0` = `ced6857afa0ea7b2e3f0846a62e1394e90f15607` (vllm-project/vllm).
- Files here:
  - `0001-*.patch`, `0002-*.patch`: `git format-patch v0.30.0..HEAD` (includes the test file).
  - `pkg/0001-*.patch`, `pkg/0002-*.patch`: the same commits restricted to `vllm/`, for `patch -p1` inside site-packages.

## Commits

1. `[Perf][MRv2] Stage only the uncached prompt suffix of prefix-cache hits` (the main change, plus `tests/v1/worker/test_gpu_suffix_staging.py`).
2. `[Perf][MRv2] Batch per-request slot resets in MambaHybridModelState` (small, independent, same env var).

Both are inert unless `VLLM_SUFFIX_STAGING=1`.

## Problem

In Model Runner V2 (`vllm/v1/worker/gpu/`), `GPUModelRunner.add_requests` calls `RequestState.add_request` for every new request.
That stages the request's whole `prefill_token_ids` (the prompt, plus earlier outputs for a resumed request) into the persistent row `RequestState.all_token_ids[slot]` with `StagedWriteTensor.stage_write(slot, 0, all_token_ids)`.
`stage_write` extends a Python list with every token, `apply_write` converts that list to numpy inside `UvaBufferPool.copy_to_uva` (`dst[:n] = x`) and launches a Triton kernel to write the row, and `clear_staged_writes` clears the list again.
For M-RoPE models, `DefaultModelState.add_request` also stages `num_dims * prefill_len` positions into `RopeState.prefill_positions` in the same way.
`num_computed_tokens` (the prefix-cache hit length, including externally loaded KV) is already known at this point: the scheduler puts `request.num_computed_tokens` into `NewRequestData.num_computed_tokens` (`Scheduler.schedule`, `vllm/v1/core/sched/scheduler.py`, the `use_v2_model_runner` branch that builds `NewRequestData.from_request(req, ..., req._all_token_ids)`).
Only positions `>= num_computed_tokens` are ever run for the request, but all positions are staged.
With an 8,000-token cached prefix and a 28-528 token suffix, over 93% of the staged elements are never read.
This matches the profile: `copy_to_uva` 42.5%, `clear_staged_writes` 8.3% and `stage_write` 3.6% of the engine-core thread.

## What the patch does

With `VLLM_SUFFIX_STAGING=1`, a new request with `num_computed_tokens > 0` for which the predicate below holds stages only `prefill_token_ids[S:]`, written at column offset `S` of its row, where `S = min(num_computed_tokens, prefill_len)`.
Every column `>= S` therefore holds exactly the value it holds with full staging, so no reader needs any offset translation.
Columns `[0, S)` keep stale values from an earlier occupant of the slot.
`prompt_len`, `prefill_len`, `total_len`, `num_computed_tokens`, block tables and all sampling state are staged exactly as before.
For M-RoPE models the positions are still computed from the full prefill (they can depend on earlier images), but only columns `[S:]` are staged.
The runner logs `VLLM_SUFFIX_STAGING=1: staging only the uncached prompt suffix ...` at load time, or a warning that lists why it is disabled.

Commit 2 replaces the two single-element `fill_` calls per new request in `MambaHybridModelState.add_request` (`num_accepted_tokens_gpu[slot] = 1` and, in `mamba_cache_mode=align`, `_mamba_state_idx_gpu[slot] = (num_computed_tokens - 1) // block_size`) with one `index_fill_` and one `index_copy_` in `apply_staged_writes()`.
The runner calls `model_state.apply_staged_writes()` right after the `add_request` loop and nothing reads those buffers in between, so the GPU state seen by `preprocess_state` and `prepare_attn` is identical.

## Predicate

Suffix-only staging is used for a request only if all of these hold, otherwise the full prefill is staged exactly as before.

Runner-wide (checked once in `GPUModelRunner._init_suffix_staging`, at the end of `load_model`):

- `VLLM_SUFFIX_STAGING=1` (read once in `GPUModelRunner.__init__`).
- The model state is exactly `DefaultModelState` or `MambaHybridModelState`, not a subclass or a model-provided state.
- No EVS multimodal pruner (`model_state.mm_pruner is None`).
- No watermarking (`watermark_config is None`).
- No speculative decoding, not a pooling model, no custom logits processors (conservative).

Per request (`vllm/v1/worker/gpu/suffix_staging.py`, `request_needs_full_prompt` and `get_prompt_stage_start`):

- `num_computed_tokens > 0`.
- `sampling_params is not None` (not a pooling request).
- No `prompt_logprobs`.
- No penalties: `repetition_penalty == 1.0`, `frequency_penalty == 0.0`, `presence_penalty == 0.0`.
- No bad words, no structured outputs, no `thinking_token_budget`.

The target workload (temperature 0, `max_tokens=1`, `logprobs=K`, `allowed_token_ids`, `processed_logprobs`, no penalties, no prompt logprobs, no spec decode, `Qwen3_5MoeForCausalLM` which resolves to `MambaHybridModelState`) satisfies it.
`logprobs`, `logprob_token_ids`, `allowed_token_ids`, `logit_bias`, `min_tokens`, seeds and LoRA do not read prompt tokens and do not disable it.

## Correctness argument

Claim: for a slot staged with start `S`, no allowed reader ever reads a column `< S` of `all_token_ids[slot]` or of the slot's M-RoPE `prefill_positions` rows while the request occupies the slot.

1. The prefill gather (`_prepare_prefill_inputs_kernel`, `input_batch.py`) reads columns `[nct, nct + query_len)` plus lookahead columns `[nct + query_len, nct + query_len + num_lookahead)` masked to `< prefill_len`, where `nct` is the slot's GPU `RequestState.num_computed_tokens`.
2. The M-RoPE gather (`_prepare_rope_positions_kernel`, `mm/rope.py`) reads `prefill_positions` columns from the same GPU `nct` onward, and decode positions use `nct + delta` without reading the rows.
3. The GPU `num_computed_tokens[slot]` is written in exactly three places: `RequestState.add_request` sets it to `num_computed_tokens >= S`, `_post_update_kernel` adds `query_len - num_rejected` with `num_rejected <= num_logits <= query_len` (asserted in `prepare_inputs`, and both 0 for a chunked prefill), and `_post_update_num_computed_tokens_kernel` adds `query_len >= 0`.
   So it never decreases below `S` while the request holds the slot, and both gathers only read columns `>= S`.
4. Chunked prefill across steps continues from the grown `nct`, so later chunks also read `>= S`.
5. Preemption resets `num_computed_tokens` to 0 in the scheduler, but MRv2 never rewinds a live slot: the scheduler reports the id in `preempted_req_ids`, `finish_requests` removes the slot, and on resume `Scheduler.schedule` moves the request into `scheduled_new_reqs` with `prefill_token_ids=req._all_token_ids` (full prompt plus outputs) and the new `num_computed_tokens`, so the request is re-added and the predicate and `S` are re-evaluated.
   A streaming-input update also removes and re-adds the request.
   No lazy re-staging or CPU copy of the prompt is needed.
6. Async scheduling does not change any of this: the staged writes are applied on the main stream in `add_requests`, before `prepare_inputs` of the same step, as before.
7. A tripwire in `update_requests` logs a warning once if the scheduler ever reports a live request's `num_computed_tokens` below its first staged column.
   In v0.30.0 that cannot happen through the paths above (a sync KV-load failure with the recompute policy lowers it in the scheduler, but MRv2 does not rewind its GPU `nct` in that case either, before or after this patch).

### Every reader of the staged prompt ids or M-RoPE prefill positions

| Reader | Columns read | Status |
| --- | --- | --- |
| `_prepare_prefill_inputs_kernel` (`input_batch.py`) | `[nct, prefill_len)` incl. lookahead | Safe, `nct >= S` |
| `_prepare_rope_positions_kernel` (`mm/rope.py`) | `prefill_positions[nct:]` | Safe, `nct >= S` |
| `_post_update_kernel` (`input_batch.py`) | writes sampled ids at `total_len >= prefill_len` | Safe, no prompt read |
| `PenaltiesState.apply_staged_writes` -> `bincount` (`sample/penalties.py`) | `[0, prefill_len)` | Excluded per request |
| `PromptLogprobsWorker.compute_prompt_logprobs` (`sample/prompt_logprob.py`) | prompt targets | Excluded per request |
| `ThinkingBudgetState.apply` (`sample/thinking_budget.py`) | scans from column 0 | Excluded per request |
| `BadWordsState.apply_bad_words` (`sample/bad_words.py`) | output columns `[prompt_len, total_len)` | Safe, excluded anyway |
| `GPUWatermarkSampler` (`vllm/v1/watermarking/gpu_sampler.py`) | context history | Excluded runner-wide |
| `MMPruner.recompute` (`model_states/mm_pruning.py`) | `all_token_ids[:prefill_len]`, `read_prefill_positions(0..)` | Excluded runner-wide |
| Model-provided states: `DeepseekV41ModelState` (lookback kernel) and `Qwen4ExpModelState` nvidia/amd (n-gram context) | `all_token_ids` columns before `nct` | Excluded runner-wide (exact-type check) |
| Other model-provided states (`LongcatNgramModelState`, diffusion Gemma, encoder-only, encoder-decoder) | not audited in depth | Excluded runner-wide (exact-type check) |
| Structured outputs (`structured_outputs.py`) | none, bitmask comes from the scheduler | Excluded anyway |
| Speculators and rejection sampler (`spec_decode/`) | none found, they use the gathered `input_ids` and lookahead | Excluded anyway |
| `PoolingRunner` (`pool/pooling_runner.py`) | uses `new_req_data.prompt_token_ids` on CPU | Excluded anyway |
| LoRA, KV/EC connectors, PP handler, PCP/DCP, ubatching, trace replay, logit bias, logprob token ids | none | Allowed |
| Detokenization, stop strings, repetition detection, `Request._all_token_ids` | scheduler and frontend copies, not the runner row | Unaffected |

The search covered every `all_token_ids` and `req_states` use under `vllm/` (including `vllm/models/`, `vllm/model_executor/models/` and `vllm/v1/watermarking/`) and every `prefill_positions` use.

## How to apply

Into an installed vLLM 0.30.0 (the `.py` files must match the tag; the dry run checks that):

```bash
bash patches/apply.sh "$(which python)"
```

To revert, run `patch -p1 -R` with the patches in reverse order inside the site-packages directory.
Into a git checkout of `v0.30.0`, use `git am patches/vllm-0.30.0/suffix-staging/0*.patch`.

## How to verify

1. Import check: `python -c "import vllm.v1.worker.gpu.suffix_staging"`.
2. Unit tests on the GPU box (the last group needs CUDA and runs the real `RequestState` staging and prefill gather kernel against full staging, with stale data in the reused slot):

   ```bash
   mkdir -p /tmp/suffix-staging-tests && cd /tmp/suffix-staging-tests
   git apply --include='tests/*' "$DECISIO/patches/vllm-0.30.0/suffix-staging/0001-"*.patch
   python -m pytest -q tests/v1/worker/test_gpu_suffix_staging.py
   ```

3. Start the server with `VLLM_SUFFIX_STAGING=1` and check for the `staging only the uncached prompt suffix` log line (no `ignored` warning, no `rewound` warning during the run).
4. Output equivalence: send the same requests one at a time (so batch composition is identical) with the flag at 0 and at 1 after warming the prefix, and compare the returned top-K logprobs, which should match exactly.
   Under concurrent load, batch composition differs between runs, so compare label argmax agreement and the logprob differences against the run-to-run noise of the flag-0 baseline.
5. Performance: repeat the py-spy profile and the throughput benchmark with the flag at 0 and at 1 on the same installation.

## Measured effect

On one RTX PRO 6000 Blackwell Max-Q, against stock vLLM 0.30.0 in its own environment, at 8,000-token states with front padding to 1,056-token blocks (`runs/2026-09-30_plugin-verification`, `bench_*.json`): 3 to 8% less time per question with the flag on, nothing measurable on a 100-question throughput cell.
With the flag off the patched build matched stock within about 2%, which is within the variation between repeats (3 repeats, no separate noise measurement).
The patch's own CUDA test passed on that card.

## Expected effect

Per new request, the staged token elements drop from `prefill_len` (8,028-8,528 at an 8,000-token prefix) to the suffix length (28-528), a 15x to 300x reduction.
`copy_to_uva`, `clear_staged_writes` and `stage_write` (54% of the engine-core thread in the profile) should shrink roughly by the same factor, and the Triton row-write kernel moves correspondingly fewer bytes into the UVA-resident row.
Commit 2 removes two small kernel launches and two tensor-view creations per new request.
The throughput gain depends on how engine-thread-bound the steady state is.
With `max_tokens=1` and short suffixes, the GPU step is short, so a large part of the removed host time should turn into throughput, but this must be measured.
Still proportional to the full prompt per request: the block table staging (needed, since attention reads the cached prefix through it), M-RoPE position computation when serving the VL class, and, with a multiprocess executor (TP > 1), serialization of `prefill_token_ids` in `SchedulerOutput`.

## Risks and limits

- The correctness relies on the GPU `num_computed_tokens` of a slot never going below its add-time value, and on the reader list above being complete for v0.30.0; a future reader of prompt columns, an out-of-tree plugin that reads `req_states.all_token_ids`, or a new rewind path would break it silently (the tripwire only covers scheduler rewinds).
- Model-provided model states, EVS, watermarking and spec decode fall back to full staging, so they gain nothing.
- Verified on one card only (the RTX PRO 6000 Blackwell Max-Q above): the CUDA test, output equality with the flag on and off, and the latency effect.
