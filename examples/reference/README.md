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

## Tests

`tests/unit/test_reference.py` runs the reference and the server's own code path (`SystemOne` on the CPU stand-in engine) on the same stand-in model, for four prompt profiles and 14 questions each.
The token ids and label ids are equal, and the answers equal within 1e-9 (the largest difference seen was 3.3e-16).
The comparison against the real server on each base is a separate run.
