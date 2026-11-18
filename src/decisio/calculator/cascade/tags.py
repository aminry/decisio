# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""What a role tagger says about a text: spans with roles and line groups, flags, and how sure it is of each decision.

The format is the tagger's training format (one JSON object per text):

  {"spans": [{"start": 80, "end": 91, "text": "Attachments", "role": "item_label", "item": "i2"},
             {"start": 98, "end": 99, "text": "4", "role": "quantity", "item": "i2"},
             {"start": 120, "end": 127, "text": "$299.46", "role": "price", "item": "i2"},
             {"start": 7, "end": 14, "text": "INV-204", "role": "distractor", "item": null}],
   "flags": {"tax_kind": "charged", "tax_after": ["Handling fee"], "credit_position": {"Claims credit": "after_tax"},
             "time_rounding": {"Operator time": {"step_min": 15, "mode": "each"}},
             "length_rounding": {"Cut pieces": {"step": "0.25", "mode": "total"}},
             "adjustment_direction": {"Volume discount": "deduct"}, "range_last_day": "counts"}}

- `start` and `end` are character offsets into the state, or into the task's context with `"in": "context"`; `text`
  must equal the source text at those offsets.
- `item` groups the spans of one line: `i1`, `i2`, ... priced items (a day credit is an item too), `f1`... fees,
  `c1`... credits, `a1`... percentage adjustments, `tax` the tax rate (`tax2` a second, different rate); a
  `distractor` (a number that enters no line) and a `stated_total` (a figure the text itself states) have none.
- a `percent` span names the item it applies to with `target`.

Every span carries `p`, the probability of its role; `confidences` holds the probability of every other decision the
tagger made (`link:...` for the grouping and the targets, `flag:...` for the flags), and `abstain_p` the probability
that the text is one where no block should be shown. Labelled data has no probabilities: every decision is 1.0.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping

ROLES = (
    "item_label",
    "quantity",
    "unit",
    "price",
    "per_unit",
    "duration",
    "rate",
    "day_count",
    "range_start",
    "range_end",
    "factor_from",
    "factor_to",
    "percent",
    "tax_rate",
    "rounding_step",
    "stated_total",
    "distractor",
)
# roles whose span carries no number (a label, a unit word); every other role's span holds exactly one value
WORD_ROLES = frozenset({"item_label", "unit", "per_unit"})
# roles that belong to no line
FREE_ROLES = frozenset({"stated_total", "distractor"})
SOURCES = ("state", "context")


@dataclass(frozen=True)
class TagSpan:
    start: int
    end: int
    text: str
    role: str
    item: str | None = None
    target: str | None = None  # a percent span: the item it applies to
    where: str = "state"  # "state" or "context"
    p: float = 1.0  # the probability of the role (for a predicted span, the lowest over its tokens)

    @property
    def key(self) -> str:
        return f"{self.where}:{self.start}-{self.end}"


def _frozen(x):
    if isinstance(x, Mapping):
        return MappingProxyType({k: _frozen(v) for k, v in x.items()})
    if isinstance(x, (list, tuple)):
        return tuple(_frozen(v) for v in x)
    return x


@dataclass(frozen=True)
class Tags:
    spans: tuple[TagSpan, ...]
    flags: Mapping = field(default_factory=lambda: MappingProxyType({}))
    confidences: Mapping[str, float] = field(default_factory=lambda: MappingProxyType({}))
    abstain_p: float = 0.0  # the probability that no block should be shown for this text
    note: str = ""  # why a tagger gave no spans (`too-long`), for the report

    def __post_init__(self):
        object.__setattr__(self, "spans", tuple(self.spans))
        object.__setattr__(self, "flags", _frozen(self.flags))
        object.__setattr__(self, "confidences", MappingProxyType(dict(self.confidences)))

    @classmethod
    def from_row(cls, row: Mapping) -> Tags:
        """Labelled tags (the training format above); every decision has probability 1.0."""
        spans = []
        for s in row.get("spans") or ():
            where = s.get("in", "state") or "state"
            if where not in SOURCES:
                raise ValueError(f"span 'in' must be one of {SOURCES}, not {where!r}")
            spans.append(
                TagSpan(
                    start=int(s["start"]),
                    end=int(s["end"]),
                    text=s["text"],
                    role=s["role"],
                    item=s.get("item"),
                    target=s.get("target"),
                    where=where,
                    p=float(s.get("p", 1.0)),
                )
            )
        abstain = float(row.get("abstain_p", 1.0 if row.get("outcome") == "abstain" else 0.0))
        return cls(spans=tuple(spans), flags=dict(row.get("flags") or {}), abstain_p=abstain)


def thaw(x):
    """A plain dict or list copy of a frozen flag value."""
    if isinstance(x, Mapping):
        return {k: thaw(v) for k, v in x.items()}
    if isinstance(x, tuple):
        return [thaw(v) for v in x]
    return x
