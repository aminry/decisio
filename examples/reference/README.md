<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Copyright contributors to the decisio project -->
# The plain-transformers reference

`decisio_reference.py` makes one decisio decision per call with Hugging Face transformers alone.
It reads `decision_config.json` from a model repository, builds the prompt the server builds, runs one forward pass, reads the option letters at the last position, applies the repository's temperatures and returns the System One answer.
It needs torch and transformers (5.10.4 or later) and imports nothing from decisio, so it can check a repository independently of the code the repository is served with.

## Use

```bash
python decisio_reference.py tachara-ai/decisio-gemma-4-12b \
  --state "The parcel is late." \
  --question '{"type": "noul", "instructions": "Is the customer unhappy?"}'
```

```bash
python decisio_reference.py ./my-repository --request request.json --dtype bfloat16
```

`--request` takes a `/v1/systemone` body, a state and named questions, and answers each question as its own call.
The output is the answer, or `{"answers": {...}}` for a request, in the wire format of the server.

In Python:

```python
from decisio_reference import Reference

ref = Reference.load("tachara-ai/decisio-gemma-4-12b")
ref.decide("The parcel is late.", {"type": "choice", "criteria": {"delivery": "Where is it?", "billing": None}})
ref.prepare(state, question)  # the token ids and label ids the model is given, for inspection
```

## What it reproduces

- The question as the server renders it: instructions, option descriptions, hidden index keys, de-snaked labels, yes/no asked as letters or in words as the file says.
- The chat prompt of the file's format (tail, answer slot, label variants, system prompt), the state padding the profile asks for, and the labels' token ids.
- The readout at the last position (the transformers model applies a softcap itself, and the file's softcap must match the model's), the choice temperature and the global one, the confidence and the exact-tie rule.

It refuses a file of another schema, a value it does not know, a softcap the model does not report, and a state that carries an image.

## What it is not

It is not a server: no prefix cache, no batching, no registered tasks, no intent head, no abstention, no two-order averaging.
A request to a server that uses a registered task is answered by the task, and this reference answers the plain readout the task replaces.

`Qwen` pads the state to the vLLM engine's 1,056-token block (`--block-size` changes it), and the Gemma bases do not pad.
Run it in the dtype the checkpoint was served in: a float32 forward pass differs from a bfloat16 or FP8 engine in the last digits.

## Attention on CUDA

On a CUDA device the reference reads with eager attention by default (`--attention safe`); on the CPU it uses transformers' own path.
Reason: on an RTX PRO 6000 Blackwell (sm_120, torch 2.13.0, CUDA 13.0), transformers' default forward of Gemma 4 12B gave probabilities 0.37 to 0.51 away from the server's on every prompt of 129, 385 or 449 tokens (a length of 1 modulo 64), five of 200 items and all five of that length.
The same reference on the CPU agreed with the server on them to 0.0013.
Lab 2's sweep over every length from 100 to 399 tokens (2026-10-13, transformers 5.17.0) shows the rule is wider than 1 modulo 64: the default path differs from eager by more than 2 logits at every length that is 1 modulo 32 (9 of 9, by 9 to 24) and at 13 other lengths (by 2 to 19), while eager and `sdpa-math` stay within 0.35 of the CPU on the rows checked.
Until the defect is pinned down, do not use `--attention default` on CUDA to check a server; `--attention sdpa-math` (SDPA with only its math backend) and `--attention eager` are the paths to compare.

## Measured agreement with the server

Gemma 4 12B (bfloat16, decisio 0.11.1, the repository at `a5a1ac8f`), the reference on the CPU against the server on an RTX PRO 6000 Blackwell, 200 conformance items over six tasks (Lab 2's session of 2026-10-10; the token rows and the answer structure are identical):

- the mean over items of the largest probability difference is 0.0068, about 0.007, and the median item is 0.0027;
- the largest difference is 0.061, on the long items with many options (banking77, mmlu_pro_text); on boolq it is 0.006;
- the top answer is the same on all 200 items.

The tolerance set before the run was 0.04 for the largest difference and 0.005 for the mean item maximum, with the check failing above 0.12.
The agreement does not meet it: the largest difference is 0.061 and the mean 0.0068.
It was not loosened afterwards, and the check's verdict is a numeric finding: between the tolerance and the failure line, with no change of answer.
Two bfloat16 engines (vLLM's Triton attention and the CPU) differ by about this much; a cached read against a fresh one in vLLM differs by a similar size (`EVAL_CARD.md` section 6.5).
## Gemma 4 31B (FP8)

The reference runs the 31B's stored FP8 weights with bfloat16 activations, and the server quantizes activations to FP8 at run time, so the two cannot meet the 12B's tolerance; the 31B has its own rule.
Lab 2's 2026-10-13 run, the 31B on decisio's vLLM 0.31.0 server against the reference on the same 200 requests: with `--attention eager` the largest difference is 0.478, the mean item maximum 0.0235 and 3 choice flips have a server top-two gap of at least 0.10; with `sdpa-math` 0.431, 0.0190 and 2; with `default` 0.342, 0.0183 and 1.
The frozen v0.11.1 server (vLLM 0.30.0) against the reference on the 2026-10-10 card: 0.594, 0.0206 and 5.
**The rule:** pass when the mean item maximum is at most 0.026 (the frozen server's plus 25 percent), the largest difference at most 0.594, at most 5 flips with a gap of at least 0.10, and the token and label rows are the same on both sides.
It asks whether the reference agrees with a server as well as it did with the frozen one, and does not claim the two agree to the 12B's tolerance.
All three paths pass it on the 0.31.0 server; the reasons for each bound are in decisio's pull request #123.

The Qwen base (FP8) has not been compared with its server: no run of the reference against the Qwen server exists yet, so nothing is claimed for it here.

## Tests

`tests/unit/test_reference.py` runs the reference and the server's own code path (`SystemOne` on the CPU stand-in engine) on the same stand-in model, for four prompt profiles and 14 questions each.
The token ids and label ids are equal, and the answers equal within 1e-9 (the largest difference seen was 3.3e-16).
The comparison against the real server on each base is a separate run.
