# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""A state, question or option that quotes a thinking tag is not an open thinking block.

`check_thinking_closed` refuses a prompt whose chat template left a thinking block open. It counted the markers over the
whole rendered prompt, so an arXiv abstract that quotes "<think>" (one opener, no closer) made the request a 400
("thinking left open in chat mode") while its neighbours were answered. Only the template's own markers count.

K1  a state quoting an unbalanced "<think>" (or "<|channel>") is answered: chat_wrap and chat_turn, default and header
K2  a template that really leaves a block open is still refused, with and without such a quote in the state
K3  a balanced quote, one in the question or in an option, and the template's own closing marker quoted are answered
K4  the template trims the user turn: a state with surrounding whitespace is found and cut out all the same
K5  where the user turn is not found in the rendered text (a template that rewrites it), the whole text is checked

  uv run pytest -q tests/unit/test_think_marker_in_state.py
"""

import pytest

from decisio.readout import letters as L


class Tok:
    """A chat template that closes its thinking block (Gemma 4: an empty thought channel; Qwen: <think></think>)."""

    def __init__(self, style="gemma", leave_open=False, rewrite=None):
        self.style, self.leave_open, self.rewrite = style, leave_open, rewrite

    def apply_chat_template(self, messages, add_generation_prompt=False, tokenize=False, enable_thinking=None):
        out = ""
        for m in messages:
            text = m["content"].strip()  # Gemma's template trims the content
            if self.rewrite:
                text = self.rewrite(text)
            out += f"<|turn>{m['role']}\n{text}<turn|>\n"
        if add_generation_prompt:
            if self.style == "gemma":
                out += "<|turn>model\n" + (
                    "<|channel>thought\n" if self.leave_open else "<|channel>thought\n<channel|>"
                )
            else:
                out += "<|im_start|>assistant\n" + ("<think>\n" if self.leave_open else "<think>\n\n</think>\n\n")
        return out


STATES = [
    "Training only the initial root token (e.g., whether to invoke the <think> tag) is enough.",
    "A sample: <|channel>thought and then the answer.",
    "Closes with </think> only.",
    "An opener <think> and a closer </think> and one more <think>",
]
HEADER = L.PromptFormat(layout="header")


@pytest.mark.parametrize("style", ["gemma", "qwen"])
@pytest.mark.parametrize("state", STATES)
def test_k1_a_quoted_unbalanced_marker_in_the_state_is_answered(style, state):
    tok = Tok(style)
    body = f"{state}\n\nWhich topic?\nOptions:\nA. one\nB. two"
    out = L.chat_wrap(tok, body + "\nAnswer:", "chat")
    assert state.strip() in out and out.endswith("Answer:")
    assert L.chat_turn(tok, body).endswith("Answer:")
    assert L.chat_turn(tok, body, HEADER).endswith("Answer:")


@pytest.mark.parametrize("style", ["gemma", "qwen"])
def test_k2_a_template_that_leaves_a_block_open_is_still_refused(style):
    tok = Tok(style, leave_open=True)
    for state in ("plain text", STATES[0], "has <think> and </think> balanced"):
        body = f"{state}\n\nWhich?"
        with pytest.raises(AssertionError, match="thinking left open in chat mode"):
            L.chat_wrap(tok, body + "\nAnswer:", "chat")
        with pytest.raises(AssertionError, match="thinking left open in chat mode"):
            L.chat_turn(tok, body)
        with pytest.raises(AssertionError, match="thinking left open in chat mode"):
            L.chat_turn(tok, body, HEADER)


def test_k3_quotes_in_the_question_an_option_or_balanced_are_answered():
    tok = Tok()
    for body in (
        "A balanced <think>x</think> quote.\n\nWhich?",
        "Text.\n\nIs the <think> tag used?\nOptions:\nA. <think> only\nB. <|channel> only",
        "Quotes the template's closer <channel|> alone.\n\nWhich?",
    ):
        assert L.chat_wrap(tok, body + "\nAnswer:", "chat").endswith("Answer:")
        assert L.chat_turn(tok, body, HEADER).endswith("Answer:")


def test_k4_a_trimmed_user_turn_is_found_and_cut_out():
    tok = Tok()
    body = "   \n  <think> opened and never closed  \n\nWhich?  "
    assert L.chat_wrap(tok, body + "\nAnswer:", "chat").endswith("Answer:")
    assert L.chat_turn(tok, body).endswith("Answer:")


def test_k5_when_the_user_turn_is_not_found_the_whole_text_is_checked():
    tok = Tok(
        rewrite=lambda t: t.replace("<think>", "[think]") + " <think>"
    )  # the template rewrites and adds an opener
    with pytest.raises(AssertionError, match="thinking left open"):
        L.chat_turn(tok, "Some state with <think> in it.\n\nWhich?")
    # and the helper itself: no content given means the whole text, the old behaviour
    with pytest.raises(AssertionError):
        L.check_thinking_closed("x <think> y")
    L.check_thinking_closed("x <think> y", "x <think> y")
    L.check_thinking_closed("<think>\n\n</think>")
