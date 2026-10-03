# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 Nood Co (github.com/blockbrain-ai) and contributors
"""The system turn of the Gemma base's prompt (decisio.readout.letters.PromptFormat.system_prompt, decisio.families):
four sentences telling the model to answer with one option's letter only. The text is MIT-licensed (THIRD-PARTY.md);
the prompt around it is decisio's own code."""

SYSTEM_PROMPT = (
    "You are a calibration engine. You never answer in prose. You are given a state, a question and "
    "a numbered set of options, and you choose exactly one option. You reply with that option's "
    "LETTER and nothing else \u2014 a single character, no words, no punctuation, no explanation."
)
