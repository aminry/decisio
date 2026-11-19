# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""The tagger's network: a Hugging Face encoder with five small heads, its training targets and its decoding.

Imported only when a tagger model is loaded or trained: it needs `torch` and `transformers`.

Heads, over the encoder's last hidden states:

  tokens     BIO labels over the roles, with the kind of line folded into the role (`fee_price`, `credit_label`,
             `adj_label`, ...), so a span says both what it is and which kind of line it belongs to
  link       for two spans of the same kind of line: do they belong to the same line (mean-pooled span vectors,
             an MLP over [a, b, a * b])
  target     for a percent span and an item label span: is that the item the adjustment applies to
  attrs      per span: after tax or not (fee, credit and day-credit amounts), each entry or the total (rounding
             steps), add or deduct (percent spans), the last day counted or not (date range ends)
  globals    over the whole text: tax charged or withheld, and whether a block should be shown at all

Decoding groups the spans of a kind of line by the link head (pairs at 0.5 or more joined, transitively); every
grouping decision, kept or not, enters `Tags.confidences` with the probability of the outcome that was used, so a
transitive join the model did not believe falls below any threshold.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch
from torch import nn

from decisio.calculator.cascade import assembler as A
from decisio.calculator.cascade.tagger import decode_bio, trim
from decisio.calculator.cascade.tags import Tags, TagSpan

FORMAT = 1
KINDED = (
    "item_label",
    "fee_label",
    "credit_label",
    "adj_label",
    "quantity",
    "price",
    "fee_count",
    "fee_price",
    "credit_amount",
    "rate",
    "duration",
    "day_count",
    "range_start",
    "range_end",
    "factor_from",
    "factor_to",
    "rounding_step",
    "percent",
    "tax_rate",
    "tax2",
    "stated_total",
    "distractor",
    "unit",
    "per_unit",
)
LABELS = ("O",) + tuple(f"{t}-{r}" for r in KINDED for t in "BI")
ATTRS = {
    "after_tax": ("before", "after"),
    "round_mode": ("each", "total"),
    "direction": ("add", "deduct"),
    "last_day": ("counts", "excluded"),
}
GLOBALS = {"tax_kind": ("charged", "withheld"), "outcome": ("shown", "abstain")}
_UNKINDED = {
    "fee_label": ("item_label", "f"),
    "credit_label": ("item_label", "c"),
    "adj_label": ("item_label", "a"),
    "fee_count": ("quantity", "f"),
    "fee_price": ("price", "f"),
    "credit_amount": ("price", "c"),
    "percent": ("percent", "a"),
    "tax_rate": ("tax_rate", "tax"),
    "tax2": ("tax_rate", "tax2"),
    "stated_total": ("stated_total", None),
    "distractor": ("distractor", None),
}
GROUPED = ("i", "f", "c", "a")
SEP = "\n\n"


def kinded(role: str, item: str | None) -> str:
    pre = (item or "")[:1]
    if role == "item_label":
        return {"f": "fee_label", "c": "credit_label", "a": "adj_label"}.get(pre, "item_label")
    if role == "price":
        return {"f": "fee_price", "c": "credit_amount"}.get(pre, "price")
    if role == "quantity" and pre == "f":
        return "fee_count"
    if role == "tax_rate" and item == "tax2":
        return "tax2"
    return role


def unkinded(label: str) -> tuple[str, str | None]:
    """(role, family): the family is the line id's prefix (i f c a), tax, tax2, or None for a free role."""
    return _UNKINDED.get(label, (label, "i"))


def join(state: str, context: str | None) -> tuple[str, int]:
    """The text the encoder reads (the context, a blank line, the state) and where the state starts in it."""
    if not context:
        return state, 0
    return context + SEP + state, len(context) + len(SEP)


# --- the network -------------------------------------------------------------------------------------------------


class CascadeTagger(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        h = encoder.config.hidden_size
        self.encoder = encoder
        self.tokens = nn.Linear(h, len(LABELS))
        self.span = nn.Sequential(nn.Linear(h, h), nn.GELU())
        self.link = nn.Sequential(nn.Linear(3 * h, h), nn.GELU(), nn.Linear(h, 1))
        self.target = nn.Sequential(nn.Linear(3 * h, h), nn.GELU(), nn.Linear(h, 1))
        self.attrs = nn.ModuleDict({k: nn.Linear(h, len(v)) for k, v in ATTRS.items()})
        self.globals = nn.ModuleDict({k: nn.Linear(h, len(v)) for k, v in GLOBALS.items()})

    def hidden(self, input_ids, attention_mask):
        return self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[0]

    def spans(self, hidden, ranges):
        if not ranges:
            return hidden.new_zeros((0, hidden.shape[-1]))
        return self.span(torch.stack([hidden[a:b].mean(0) for a, b in ranges]))

    @staticmethod
    def pair(head, a, b):
        """Logits for every (row of a, row of b)."""
        n, m = a.shape[0], b.shape[0]
        x = a[:, None, :].expand(n, m, -1)
        y = b[None, :, :].expand(n, m, -1)
        return head(torch.cat([x, y, x * y], -1)).squeeze(-1)


def build(encoder) -> CascadeTagger:
    return CascadeTagger(encoder)


# --- training targets --------------------------------------------------------------------------------------------


def _id_labels(row) -> dict[str, str]:
    out: dict[str, str] = {}
    for s in sorted(row["spans"], key=lambda s: s["start"]):
        if s["role"] == "item_label" and s.get("item") and s["item"] not in out:
            out[s["item"]] = A.norm_label(s["text"])
    return out


def _flag_by_label(flag, label: str | None):
    if not isinstance(flag, dict) or label is None:
        return None
    for k, v in flag.items():
        if A.norm_label(str(k)) == label:
            return v
    return None


def encode_row(row: dict, tokenizer, max_length: int) -> dict | None:
    """A training example from a row in the tagger's training format, or None when it does not fit the window or a
    span covers no token."""
    state, context = row["state"], row.get("context")
    text, shift = join(state, context)
    enc = tokenizer(text, return_offsets_mapping=True, add_special_tokens=True)
    if len(enc["input_ids"]) > max_length:
        return None
    offsets = [tuple(o) for o in enc["offset_mapping"]]
    spans = []
    for s in row["spans"]:
        a = s["start"] + (0 if s.get("in", "state") == "context" else shift)
        spans.append(dict(s, a=a, b=a + (s["end"] - s["start"]), k=kinded(s["role"], s.get("item"))))
    spans.sort(key=lambda s: s["a"])
    owner = [None] * len(text)
    for i, s in enumerate(spans):
        for c in range(s["a"], s["b"]):
            if owner[c] is None and not text[c].isspace():
                owner[c] = i
    labels = [-100 if a == b else 0 for a, b in offsets]
    ranges: list[list[int]] = [[] for _ in spans]
    for t, (a, b) in enumerate(offsets):
        own = next((owner[c] for c in range(a, b) if owner[c] is not None), None)
        if own is None:
            continue
        first = not ranges[own]
        ranges[own].append(t)
        labels[t] = LABELS.index(("B-" if first else "I-") + spans[own]["k"])
    if any(not r for r in ranges):
        return None
    ranges = [(r[0], r[-1] + 1) for r in ranges]

    spec = row.get("spec") or row.get("nearest_spec") or {}
    flags = row.get("flags") or {}
    ids = _id_labels(row)
    item_kinds = {f"i{n}": it.get("kind") for n, it in enumerate(spec.get("items") or [], 1)}
    directions = {f"a{n}": a.get("direction") for n, a in enumerate(spec.get("adjustments") or [], 1)}
    after = {A.norm_label(str(x)) for x in flags.get("tax_after") or []}

    fam = [unkinded(s["k"])[1] for s in spans]
    pairs, pair_y = [], []
    for i in range(len(spans)):
        for j in range(i + 1, len(spans)):
            # pairs across two families too (always two lines in the labels), so the link head is trained on the
            # pairs the group vote reads at decode (a phrase whose tokens flipped family)
            if fam[i] in GROUPED and fam[j] in GROUPED:
                pairs.append((i, j))
                pair_y.append(float(spans[i].get("item") == spans[j].get("item")))
    label_idx = [i for i, s in enumerate(spans) if s["k"] == "item_label"]
    targets = []
    for i, s in enumerate(spans):
        if s["k"] == "percent" and s.get("target"):
            pos = [n for n, j in enumerate(label_idx) if spans[j].get("item") == s["target"]]
            if pos:
                targets.append((i, pos))
    attrs = {k: [] for k in ATTRS}
    for i, s in enumerate(spans):
        item, label = s.get("item"), ids.get(s.get("item") or "")
        if s["k"] in ("fee_price", "credit_amount") or (s["k"] == "price" and item_kinds.get(item) == "daycredit"):
            attrs["after_tax"].append((i, int(label in after)))
        elif s["k"] == "rounding_step":
            r = _flag_by_label(flags.get("time_rounding"), label) or _flag_by_label(flags.get("length_rounding"), label)
            if isinstance(r, dict) and r.get("mode") in ATTRS["round_mode"]:
                attrs["round_mode"].append((i, ATTRS["round_mode"].index(r["mode"])))
        elif s["k"] == "percent":
            d = directions.get(item) or _flag_by_label(flags.get("adjustment_direction"), label)
            if d in ATTRS["direction"]:
                attrs["direction"].append((i, ATTRS["direction"].index(d)))
        elif s["k"] in ("range_start", "range_end") and flags.get("range_last_day") in ATTRS["last_day"]:
            attrs["last_day"].append((i, ATTRS["last_day"].index(flags["range_last_day"])))
    glob = {"outcome": GLOBALS["outcome"].index(row.get("outcome", "shown"))}
    if flags.get("tax_kind") in GLOBALS["tax_kind"]:
        glob["tax_kind"] = GLOBALS["tax_kind"].index(flags["tax_kind"])
    return {
        "input_ids": enc["input_ids"],
        "labels": labels,
        "ranges": ranges,
        "pairs": pairs,
        "pair_y": pair_y,
        "label_idx": label_idx,
        "targets": targets,
        "attrs": attrs,
        "globals": glob,
    }


def loss(model: CascadeTagger, ex: dict, device: str = "cpu") -> tuple[torch.Tensor, dict]:
    ids = torch.tensor([ex["input_ids"]], device=device)
    return head_loss(model, ex, model.hidden(ids, torch.ones_like(ids)), device)


def batch_loss(model: CascadeTagger, batch: list[dict], pad_id: int, device: str = "cpu") -> tuple[torch.Tensor, dict]:
    """One encoder pass over a padded batch, then each example's heads on its own tokens; the mean loss."""
    n = max(len(ex["input_ids"]) for ex in batch)
    ids = torch.full((len(batch), n), pad_id, dtype=torch.long, device=device)
    mask = torch.zeros((len(batch), n), dtype=torch.long, device=device)
    for i, ex in enumerate(batch):
        ids[i, : len(ex["input_ids"])] = torch.tensor(ex["input_ids"], device=device)
        mask[i, : len(ex["input_ids"])] = 1
    hidden = model.encoder(input_ids=ids, attention_mask=mask).last_hidden_state
    total, parts = 0, {}
    for i, ex in enumerate(batch):
        t, p = head_loss(model, ex, hidden[i, : len(ex["input_ids"])], device)
        total = total + t
        for k, v in p.items():
            parts[k] = parts.get(k, 0.0) + v / len(batch)
    return total / len(batch), parts


def head_loss(model: CascadeTagger, ex: dict, hidden, device: str = "cpu") -> tuple[torch.Tensor, dict]:
    """The heads' losses for one example, from its encoder states (tokens x hidden)."""
    ce = nn.functional.cross_entropy
    parts = {"tokens": ce(model.tokens(hidden), torch.tensor(ex["labels"], device=device), ignore_index=-100)}
    spans = model.spans(hidden, ex["ranges"])
    if ex["pairs"]:
        a = spans[[i for i, _ in ex["pairs"]]]
        b = spans[[j for _, j in ex["pairs"]]]
        logit = model.link(torch.cat([a, b, a * b], -1)).squeeze(-1)
        y = torch.tensor(ex["pair_y"], device=device)
        parts["link"] = nn.functional.binary_cross_entropy_with_logits(logit, y)
    if ex["targets"]:
        labels = spans[ex["label_idx"]]
        tl = []
        for i, pos in ex["targets"]:
            logit = model.pair(model.target, spans[i : i + 1], labels)[0]
            tl.append(torch.logsumexp(logit, 0) - torch.logsumexp(logit[pos], 0))
        parts["target"] = torch.stack(tl).mean()
    for k, items in ex["attrs"].items():
        if items:
            logit = model.attrs[k](spans[[i for i, _ in items]])
            parts[f"attr_{k}"] = ce(logit, torch.tensor([y for _, y in items], device=device))
    pooled = hidden.mean(0, keepdim=True)
    for k, y in ex["globals"].items():
        parts[f"global_{k}"] = ce(model.globals[k](pooled), torch.tensor([y], device=device))
    return sum(parts.values()), {k: float(v.detach()) for k, v in parts.items()}


# --- decoding ----------------------------------------------------------------------------------------------------


def _groups(n: int, prob) -> tuple[list[int], dict[tuple[int, int], float]]:
    """Union-find over pairs with probability >= 0.5; the confidence of every pair's final decision."""
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(n):
        for j in range(i + 1, n):
            if prob[i][j] >= 0.5:
                parent[find(i)] = find(j)
    roots = [find(i) for i in range(n)]
    conf = {}
    for i in range(n):
        for j in range(i + 1, n):
            p = prob[i][j]
            conf[(i, j)] = p if roots[i] == roots[j] else 1.0 - p
    return roots, conf


MAX_WINDOWS = 8  # a longer text is refused as too long


def windows(n: int, size: int) -> list[tuple[int, int]]:
    """Overlapping [a, b) windows over the n inner tokens (the special tokens at both ends excluded), each at most
    `size` tokens, a quarter of a window apart from full overlap; one window when the text fits."""
    if n <= size:
        return [(0, n)]
    step = max(1, size - size // 4 * 2)  # half a window of overlap: a quarter on each side of every seam
    out, a = [], 0
    while True:
        b = min(a + size, n)
        out.append((b - size, b) if b == n else (a, b))
        if b == n:
            return out
        a += step


def windowed_hidden(model: CascadeTagger, ids: list[int], max_length: int, device: str = "cpu"):
    """The encoder's hidden states for every token of `ids` (special tokens at both ends). A text longer than
    `max_length` is read in overlapping windows, each with the two special tokens, and every token takes its state
    from the window in which it lies farthest from an edge; None when it needs more than `MAX_WINDOWS` windows."""
    if len(ids) <= max_length:
        t = torch.tensor([ids], device=device)
        return model.hidden(t, torch.ones_like(t))
    first, inner, last = ids[0], ids[1:-1], ids[-1]
    spans = windows(len(inner), max_length - 2)
    if len(spans) > MAX_WINDOWS:
        return None
    best: list[tuple[int, int, int]] = [(-1, 0, 0)] * len(inner)  # (distance to an edge, window, position in it)
    states = []
    for w, (a, b) in enumerate(spans):
        t = torch.tensor([[first] + inner[a:b] + [last]], device=device)
        states.append(model.hidden(t, torch.ones_like(t)))
        for i in range(a, b):
            d = min(i - a, b - 1 - i)
            if d > best[i][0]:
                best[i] = (d, w, i - a + 1)
    rows = [states[0][0]] + [states[w][j] for _, w, j in best] + [states[-1][-1]]
    return torch.stack(rows)


# roles whose span holds one value the normalizer reads (not words, not a date range, not a distractor)
_SNAPPED = frozenset(A._ACCEPTS) - {"range_start", "range_end"}


def value_bounds(state: str, context: str | None, shift: int) -> list[tuple[int, int]]:
    """Where the normalizer's values lie in the joined text (each part read with the decimal convention of both, as the
    assembler reads them)."""
    from decisio.calculator import normalize as N

    doc = N.document(((context + "\n") if context else "") + state)
    out = [(a + shift, b + shift) for a, b in (v.span for v in N.normalize(state, doc).values)]
    if context:
        out += [v.span for v in N.normalize(context, doc).values]
    return sorted(out)


def snap_values(raw, values, text: str, shift: int):
    """Decoded spans with a value role, widened to the one value they overlap when they cut it ("$22" of "$22.66",
    "h 41 min" of "1 h 41 min"); spans of one role that land on the same value become one (the lowest probability).
    A span over two values, or over none, is left as decoded (the assembler refuses it)."""
    out: list[tuple[int, int, str, float]] = []
    seen: dict[tuple[int, int, str], int] = {}
    for s, e, k, p in raw:
        s, e = trim(text, s, e)
        if s >= e or (shift and s < shift <= e):
            continue
        if unkinded(k)[0] in _SNAPPED:
            hit = [(a, b) for a, b in values if a < e and b > s]
            if len(hit) == 1:
                s, e = min(s, hit[0][0]), max(e, hit[0][1])
        key = (s, e, k)
        if key in seen:
            i = seen[key]
            out[i] = (s, e, k, min(out[i][3], p))
            continue
        seen[key] = len(out)
        out.append((s, e, k, p))
    return out


# the group-level family vote (vote_groups) is measured before it becomes the default
GROUP_VOTE = os.environ.get("DECISIO_CASCADE_GROUP_VOTE") == "1"
CROSS_LINK = float(os.environ.get("DECISIO_CASCADE_CROSS_LINK", "0.95"))
_BASE = {k: unkinded(k)[0] for k in KINDED}


def vote_spans(probs, offsets, text):
    """Each token's best label, then one line family per span: contiguous tokens of one base role (a begin and its
    insides, or pieces of one word) take the family with the most probability summed over them, so a phrase or a value
    no longer splits across families token by token ("Break|down allowance", "$" and "88.56"). A token's probability
    becomes at most the span's mean probability of that family. Returns (top_p, top) as lists."""
    top_p, top = probs.max(-1)
    top_p, top = top_p.tolist(), top.tolist()
    runs, cur, prev_end = [], None, None
    for t, (a, b) in enumerate(offsets):
        name = LABELS[top[t]]
        if a == b or name == "O":
            cur, prev_end = None, None
            continue
        tag, k = name.split("-", 1)
        same_word = prev_end == a and not text[a : a + 1].isspace()
        if cur is not None and _BASE[k] == cur[0] and (tag == "I" or same_word):
            cur[1].append(t)
        else:
            cur = [_BASE[k], [t]]
            runs.append(cur)
        prev_end = b
    for role, toks in runs:
        fams = [k for k in KINDED if _BASE[k] == role]
        if len(toks) < 2 or len(fams) < 2:
            continue
        score = {
            k: float(sum(probs[t, LABELS.index("B-" + k)] + probs[t, LABELS.index("I-" + k)] for t in toks))
            for k in fams
        }
        best = max(fams, key=lambda k: score[k])
        for n, t in enumerate(toks):
            top[t] = LABELS.index(("B-" if n == 0 else "I-") + best)
            top_p[t] = min(top_p[t], score[best] / len(toks))
    return top_p, top


def likelier_per_value(spans):
    """Spans that snapped onto the same value (the same offsets): the likelier one stays."""
    best = {}
    for x in spans:
        if (x[0], x[1]) not in best or x[3] > best[(x[0], x[1])][3]:
            best[(x[0], x[1])] = x
    return [x for x in spans if best[(x[0], x[1])] is x]


def _family_probs(probs, f) -> dict:
    """A line span's probability of each family it can take (the mean over its tokens of its role's label in that
    family); a role of one family only (a percent, a duration) gives that family alone."""
    role = _BASE[f[2]]
    a, b = f[4]
    out = {}
    for fam in GROUPED:
        k = kinded(role, fam + "1")
        if unkinded(k)[1] != fam:
            continue
        out[fam] = float((probs[a:b, LABELS.index("B-" + k)] + probs[a:b, LABELS.index("I-" + k)]).mean())
    return out


def vote_groups(model, probs, vec, found):
    """One family per line: the line spans are linked together across families (the link head reads the two spans
    alone, never their family), and each linked group takes the family with the most probability summed over its spans;
    a span whose role cannot take that family keeps its own."""
    idx = [i for i, f in enumerate(found) if unkinded(f[2])[1] in GROUPED]
    if len(idx) < 2:
        return found
    sub = vec[idx]
    p = model.pair(model.link, sub, sub).sigmoid()
    p = ((p + p.T) / 2).tolist()
    # across two families a link counts only when it is near certain and one of the two spans could take either
    # family (a label, a price or a quantity): the link head was trained on pairs of one family
    fam_of = [unkinded(found[i][2])[1] for i in idx]
    ambiguous = [len(_family_probs(probs, found[i])) > 1 for i in idx]
    for a in range(len(idx)):
        for b in range(len(idx)):
            if fam_of[a] != fam_of[b] and not (p[a][b] >= CROSS_LINK and (ambiguous[a] or ambiguous[b])):
                p[a][b] = 0.0
    roots, _ = _groups(len(idx), p)
    out = list(found)
    for r in set(roots):
        members = [idx[n] for n, x in enumerate(roots) if x == r]
        score: dict = {}
        for i in members:
            for fam, v in _family_probs(probs, found[i]).items():
                score[fam] = score.get(fam, 0.0) + v
        if not score:
            continue
        best = max(GROUPED, key=lambda fam: score.get(fam, -1.0))
        for i in members:
            s, e, k, pr, toks = found[i]
            role = _BASE[k]
            nk = kinded(role, best + "1")
            if unkinded(nk)[1] == best and nk != k:
                out[i] = (s, e, nk, pr, toks)
    return out


@torch.inference_mode()
def predict(model: CascadeTagger, tokenizer, meta: dict, state: str, context: str | None, device: str = "cpu") -> Tags:
    text, shift = join(state, context)
    enc = tokenizer(text, return_offsets_mapping=True, add_special_tokens=True)
    hidden = windowed_hidden(model, enc["input_ids"], meta["max_length"], device)
    if hidden is None:
        return Tags(spans=(), note="too-long")
    probs = model.tokens(hidden).float().softmax(-1)
    offsets = [tuple(o) for o in enc["offset_mapping"]]
    top_p, top = vote_spans(probs, offsets, text)
    raw = decode_bio(offsets, top, top_p, LABELS, text)

    found = []  # (start, end, kinded, p, token range)
    values = value_bounds(state, context, shift)
    for s, e, k, p in likelier_per_value(snap_values(raw, values, text, shift)):
        toks = [t for t, (a, b) in enumerate(offsets) if a < e and b > s and a != b]
        found.append((s, e, k, p, (toks[0], toks[-1] + 1)))
    vec = model.spans(hidden, [f[4] for f in found])
    if GROUP_VOTE and found:
        found = vote_groups(model, probs, vec, found)
    fams = [unkinded(f[2])[1] for f in found]
    item = [None] * len(found)
    conf: dict[str, float] = {}

    def key(i: int) -> str:
        s, e = found[i][0], found[i][1]
        return f"state:{s - shift}-{e - shift}" if s >= shift else f"context:{s}-{e}"

    for fam in GROUPED:
        idx = [i for i, f in enumerate(fams) if f == fam]
        if not idx:
            continue
        sub = vec[idx]
        p = model.pair(model.link, sub, sub).sigmoid()
        p = ((p + p.T) / 2).tolist()
        roots, pc = _groups(len(idx), p)
        order = []
        for n, r in enumerate(roots):
            if r not in order:
                order.append(r)
        for n, r in enumerate(roots):
            item[idx[n]] = f"{fam}{order.index(r) + 1}"
        for (a, b), c in pc.items():
            conf[f"link:{key(idx[a])}|{key(idx[b])}"] = c
    for i, f in enumerate(fams):
        if f in ("tax", "tax2"):
            item[i] = f

    labels: dict[str, str] = {}
    for i, f in enumerate(found):
        if unkinded(f[2])[0] == "item_label" and item[i] and item[i] not in labels:
            labels[item[i]] = text[f[0] : f[1]]
    for gid in list(labels):
        texts = [text[f[0] : f[1]] for i, f in enumerate(found) if item[i] == gid and unkinded(f[2])[0] == "item_label"]
        try:
            labels[gid] = A.pick_label(texts)
        except A.Refusal:
            pass  # the assembler refuses the conflicting labels itself

    targets: dict[int, str] = {}
    label_idx = [i for i, f in enumerate(found) if f[2] == "item_label"]
    for i, f in enumerate(found):
        if f[2] != "percent":
            continue
        if not label_idx:
            conf[f"link:target:{key(i)}"] = 0.0
            continue
        logit = model.pair(model.target, vec[i : i + 1], vec[label_idx])[0].softmax(0).tolist()
        by: dict[str, float] = {}
        for n, j in enumerate(label_idx):
            by[item[j]] = by.get(item[j], 0.0) + logit[n]
        best = max(sorted(by), key=lambda g: by[g])
        targets[i] = best
        conf[f"link:target:{key(i)}"] = by[best]

    flags: dict = {}
    gp = {k: model.globals[k](hidden.mean(0, keepdim=True)).softmax(-1)[0].tolist() for k in GLOBALS}
    tk = max(range(2), key=lambda n: gp["tax_kind"][n])
    flags["tax_kind"] = GLOBALS["tax_kind"][tk]
    conf["flag:tax_kind"] = gp["tax_kind"][tk]
    abstain_p = gp["outcome"][1]

    def attr(name: str, members: list[int]) -> tuple[int, float]:
        p = model.attrs[name](vec[members]).softmax(-1).mean(0).tolist()
        n = max(range(len(p)), key=lambda x: p[x])
        return n, p[n]

    def members(gid: str, kinds: tuple[str, ...]) -> list[int]:
        return [i for i, f in enumerate(found) if item[i] == gid and f[2] in kinds]

    gids = sorted({g for g in item if g and g[0] in GROUPED}, key=lambda g: (g[0], int(g[1:])))
    flags["tax_after"] = []
    for gid in gids:
        roles = {found[i][2] for i in range(len(found)) if item[i] == gid}
        label = labels.get(gid, gid)
        daycredit = gid[0] == "i" and "price" in roles and "quantity" not in roles
        daycredit &= bool(roles & {"day_count", "range_start", "range_end"})
        money = members(gid, ("fee_price", "credit_amount")) or (members(gid, ("price",)) if daycredit else [])
        if money:
            n, c = attr("after_tax", money)
            conf[f"flag:after_tax:{gid}"] = c
            if n == 1:
                flags["tax_after"].append(label)
            if gid[0] == "c":
                flags.setdefault("credit_position", {})[label] = "after_tax" if n == 1 else "before_tax"
        steps = members(gid, ("rounding_step",))
        if steps:
            n, c = attr("round_mode", steps)
            conf[f"flag:round_mode:{gid}"] = c
            name = "time_rounding" if "duration" in roles else "length_rounding"
            flags.setdefault(name, {})[label] = {"mode": ATTRS["round_mode"][n]}
        pct = members(gid, ("percent",))
        if pct:
            n, c = attr("direction", pct)
            conf[f"flag:direction:{gid}"] = c
            flags.setdefault("adjustment_direction", {})[label] = ATTRS["direction"][n]
    ends = [i for i, f in enumerate(found) if f[2] in ("range_start", "range_end")]
    if ends:
        n, c = attr("last_day", ends)
        conf["flag:last_day"] = c
        flags["range_last_day"] = ATTRS["last_day"][n]

    spans = []
    for i, (s, e, k, p, _) in enumerate(found):
        role, fam = unkinded(k)
        where, a = ("state", s - shift) if s >= shift else ("context", s)
        src = state if where == "state" else context
        spans.append(
            TagSpan(
                start=a,
                end=a + (e - s),
                text=src[a : a + (e - s)],
                role=role,
                item=item[i] if fam is not None else None,
                target=targets.get(i),
                where=where,
                p=p,
            )
        )
    return Tags(spans=tuple(spans), flags=flags, confidences=conf, abstain_p=abstain_p)


# --- files -------------------------------------------------------------------------------------------------------


def save(model: CascadeTagger, tokenizer, directory, meta: dict) -> None:
    """A tagger directory: `encoder/config.json`, the tokenizer, `model.safetensors`, `cascade_tagger.json` (the
    label set and the window) and `SHA256SUMS` over every file."""
    import hashlib

    from safetensors.torch import save_file

    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    model.encoder.config.save_pretrained(d / "encoder")
    tokenizer.save_pretrained(d)
    state = {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}
    save_file(state, d / "model.safetensors")
    meta = dict(meta, format=FORMAT, labels=list(LABELS), attrs=ATTRS, globals=GLOBALS)
    (d / "cascade_tagger.json").write_text(json.dumps(meta, indent=1, sort_keys=True))
    lines = []
    for p in sorted(x for x in d.rglob("*") if x.is_file() and x.name != "SHA256SUMS"):
        lines.append(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(d).as_posix()}")
    (d / "SHA256SUMS").write_text("\n".join(lines) + "\n")


def load(directory, device: str = "cpu", dtype: str = "float32"):
    from safetensors.torch import load_file
    from transformers import AutoConfig, AutoModel, AutoTokenizer

    d = Path(directory)
    meta = json.loads((d / "cascade_tagger.json").read_text())
    if meta.get("format") != FORMAT or tuple(meta.get("labels", ())) != LABELS:
        raise ValueError("tagger directory: an unknown format or label set")
    config = AutoConfig.from_pretrained(d / "encoder")
    encoder = AutoModel.from_config(config, attn_implementation="sdpa")
    model = CascadeTagger(encoder)
    model.load_state_dict(load_file(d / "model.safetensors", device="cpu"))
    # float32 by default: the thresholds are fitted on float32 tags; float16 or bfloat16 only where the determinism
    # check has measured the drift against the margin band
    if dtype not in ("float32", "float16", "bfloat16"):
        raise ValueError(f"tagger dtype must be float32, float16 or bfloat16, not {dtype!r}")
    model.to(device=device, dtype=getattr(torch, dtype)).eval()
    tokenizer = AutoTokenizer.from_pretrained(d)
    return model, tokenizer, meta
