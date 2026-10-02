# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 Nood Co (github.com/blockbrain-ai) and contributors
"""The prompt text of Cygnet's readout (github.com/blockbrain-ai/cygnet-recipe at 3cf591c, `shim/cygnet_shim.py`,
MIT): its system prompt and its trailing instruction, verbatim. decisio uses them under `--system-prompt cygnet` and
`--prompt-tail cygnet` (decisio.readout.letters.PromptFormat); the layout around them is decisio's own code.
"""

SYSTEM = (
    "You are a calibration engine. You never answer in prose. You are given a state, a question and "
    "a numbered set of options, and you choose exactly one option. You reply with that option's "
    "LETTER and nothing else \u2014 a single character, no words, no punctuation, no explanation."
)

TAIL = "Answer with the letter of exactly one option, and nothing else:"
