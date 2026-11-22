# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""The tokens of a prompt that carry the request's state, for a mean of the hidden states over them (the head's
state-span reads, decisio.serve.hidden_engine.TapHidden).

`state_span(tok, ids, state_text, within=P, occurrence="last")`: [start, end), the smallest token range of the row
whose decoded text contains the rendered state, at its last occurrence inside the row's shared prefix (the header and
the state, before the question). `state_text_of(state)`: the request's own state as the prompt renders it, without a
task's context."""

from __future__ import annotations

from decisio.readout.letters import CONTEXT_SEP, ContextState, fmt_state


class StateSpanError(ValueError):
    """The state's text is not where a state span is looked for."""


def _lcp(a, b):
    n = min(len(a), len(b))
    if a[:n] == b[:n]:
        return n
    lo, hi = 0, n  # a[:lo] == b[:lo], a[:hi] != b[:hi]
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if a[:mid] == b[:mid]:
            lo = mid
        else:
            hi = mid
    return lo


def state_span(tok, ids, state_text, allow_repeat=False, within=None, occurrence=None):
    """[start, end): the smallest token range of `ids` whose decoded text contains `state_text`, located at the
    state's (first) occurrence in the decoded prompt. A token that overlaps the state's characters at either edge is
    included (a BPE token that merges the template's newline with the state's first word, or the state's last period
    with the blank line after it), and so are byte tokens that carry part of the state's first or last character.
    Raises StateSpanError when the text is absent, or present more than once (unless `allow_repeat`, which takes the
    first occurrence).

    `within`: only occurrences inside the text of ids[:within] count (the row builder's shared prefix, the header and
    the state, which ends before the question and its options); `occurrence`: "only" (the default; more than one is
    an error), "first" or "last" (the state is the last occurrence in the shared prefix: the header before it lists
    instructions, and an option equal to the state text comes after the prefix).

    How. A prefix ids[:i] is "clean" when its last token decodes alone without a replacement character (it does not
    end inside a multi-byte character): a clean prefix decodes to a prefix of the decoded prompt, so its length
    conf(i) is non-decreasing over clean prefixes. (Unclean prefixes are not: a byte-fallback decoder turns a whole
    run of byte tokens into replacement characters when the run ends mid-character, so conf can drop inside a run.)
    Binary search over clean prefixes finds the last clean boundary p before the state's first character and the
    first clean boundary r after its last (about 2 log2(n) prefix decodes); the exact edges inside [p, r] are then
    found by decoding windows: end = the smallest e with the state in decode(ids[p:e]), start = the largest s with
    the state in decode(ids[s:end])."""
    ids = list(ids)
    dec = lambda x: tok.decode(x, skip_special_tokens=False, clean_up_tokenization_spaces=False)  # noqa: E731
    full = dec(ids)
    if not state_text:
        raise StateSpanError("empty state text")
    occurrence = occurrence or ("first" if allow_repeat else "only")
    if occurrence not in ("only", "first", "last"):
        raise ValueError(f"occurrence must be only, first or last, not {occurrence!r}")
    limit = len(full) if within is None else _lcp(dec(ids[:within]), full)
    cs = full.rfind(state_text, 0, limit) if occurrence == "last" else full.find(state_text, 0, limit)
    where = "the decoded prompt" if within is None else f"the first {within} tokens"
    if cs < 0:
        raise StateSpanError(f"state text not found in {where} ({len(ids)} tokens): {state_text[:80]!r}")
    if occurrence == "only" and full.find(state_text, cs + 1, limit) >= 0:
        raise StateSpanError(f"state text occurs more than once in {where}: {state_text[:80]!r}")
    ce = cs + len(state_text)
    n = len(ids)
    partial, conf_cache = {}, {}

    def clean_from(i):  # the first clean boundary at or after i
        while 0 < i < n:
            t = ids[i - 1]
            if t not in partial:
                partial[t] = "�" in dec([t])
            if not partial[t]:
                break
            i += 1
        return i

    def conf(i):
        if i not in conf_cache:
            conf_cache[i] = _lcp(dec(ids[:i]), full) if i else 0
        return conf_cache[i]

    def first(pred, lo, hi):  # smallest i in [lo, hi] with pred(i); pred monotone, pred(hi) true
        while lo < hi:
            mid = (lo + hi) // 2
            if pred(mid):
                hi = mid
            else:
                lo = mid + 1
        return lo

    i = first(lambda i: conf(clean_from(i)) > cs, 0, n)  # >= 1: conf(0) = 0 <= cs
    p, q = clean_from(i - 1), clean_from(i)  # the clean boundaries around the state's first character
    j = first(lambda j: conf(clean_from(j)) >= ce, 0, n)
    r_lo, r = clean_from(j - 1) if j else 0, clean_from(j)  # ... and around its last
    win = lambda s, e: state_text in dec(ids[s:e])  # noqa: E731
    end = next((e for e in range(max(r_lo, p) + 1, r + 1) if win(p, e)), None)
    if end is None:
        raise StateSpanError(f"no window from token {p} contains the state (decoding is not prefix-stable?)")
    start = next(s for s in range(min(q, end) - 1, p - 1, -1) if win(s, end))
    return start, end


def state_text_of(state):
    """The item's own text as the prompt renders it: the rendered state, without a task context if it has one."""
    if isinstance(state, ContextState):
        return str(state)[len(state.context) + len(CONTEXT_SEP) :]
    return fmt_state(state)
