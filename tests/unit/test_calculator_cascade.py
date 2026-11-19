# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""The cascade reader (`decisio.calculator.cascade`): assembler, gates, tagger interface and `read`.

K1  the assembler on correct spans gives exactly `from_input`'s block, byte for byte;
K2  every inconsistency is a typed refusal, never a guess;
K3  each gate passes on a correct case and fails, with its named code, on a broken one;
K4  the threshold rule's margin band abstains; thresholds are validated;
K5  the same tags give the same result every time;
K6  `read` through a stub tagger takes each path (rules_agree, stated_figures, threshold, rules_only) or refuses;
K7  the model wrapper (when torch and transformers are installed) loads a tiny randomly initialised tagger from a
    local directory and reads with it; no test downloads a model.

  uv run pytest -q tests/unit/test_calculator_cascade.py
"""

import dataclasses
import hashlib
from fractions import Fraction

import pytest

from decisio.calculator import Abstain, Annotation, annotate, from_input
from decisio.calculator.cascade import (
    Refused,
    Shown,
    StubTagger,
    Tags,
    TagSpan,
    Thresholds,
    assemble,
    gates,
    read,
    tagger_for,
)
from decisio.calculator.cascade.assembler import Refusal
from decisio.calculator.cascade.tagger import check_sums, decode_bio, resolve

STATE = """Work order WO-5531, page 1 of 2.
Crate packing: 4 crates at $12.50 each.
Forklift time: entries of 50 and 75 minutes, at $48.00 per hour; the total is rounded once to the nearest 15 minutes.
Cable: 1605 lb, converted with 1 lb = 0.4536 kg, priced at $0.20 per kg.
Rush supplement: 10% of Crate packing, added.
Handling fee: $7.25, added after tax.
Loyalty credit: $5.00, taken off before tax.
Tax of 8% is charged on the bill."""

EXPECTED = {
    "currency": "$",
    "items": [
        {"kind": "count", "label": "Crate packing", "quantity": "4", "price": "12.50"},
        {
            "kind": "time",
            "label": "Forklift time",
            "minutes": [50, 75],
            "rate": "48.00",
            "per": "h",
            "round": {"step": 15, "mode": "total"},
        },
        {
            "kind": "convert",
            "label": "Cable",
            "quantity": "1605",
            "unit": "lb",
            "to_unit": "kg",
            "factor": "0.4536",
            "price": "0.20",
        },
    ],
    "adjustments": [{"label": "Rush supplement", "percent": "10", "of": "Crate packing", "direction": "add"}],
    "fees": [{"label": "Handling fee", "price": "7.25"}],
    "credits": [{"label": "Loyalty credit", "amount": "5.00"}],
    "tax": {"rate": "8", "kind": "charged", "outside": ["Handling fee"]},
}

SPANS = [  # (text, occurrence, role, item, target)
    ("WO-5531", 0, "distractor", None, None),
    ("page 1 of 2", 0, "distractor", None, None),
    ("Crate packing", 0, "item_label", "i1", None),
    ("4 crates", 0, "quantity", "i1", None),
    ("$12.50", 0, "price", "i1", None),
    ("Forklift time", 0, "item_label", "i2", None),
    ("50", 1, "duration", "i2", None),  # the first "50" is inside $12.50
    ("75 minutes", 0, "duration", "i2", None),
    ("$48.00", 0, "rate", "i2", None),
    ("15 minutes", 0, "rounding_step", "i2", None),
    ("Cable", 0, "item_label", "i3", None),
    ("1605 lb", 0, "quantity", "i3", None),
    ("1 lb", 0, "factor_from", "i3", None),
    ("0.4536 kg", 0, "factor_to", "i3", None),
    ("$0.20", 0, "price", "i3", None),
    ("Rush supplement", 0, "item_label", "a1", None),
    ("10%", 0, "percent", "a1", "i1"),
    ("Handling fee", 0, "item_label", "f1", None),
    ("$7.25", 0, "price", "f1", None),
    ("Loyalty credit", 0, "item_label", "c1", None),
    ("$5.00", 0, "price", "c1", None),
    ("8%", 0, "tax_rate", "tax", None),
]

FLAGS = {
    "tax_kind": "charged",
    "tax_after": ["Handling fee"],
    "credit_position": {"Loyalty credit": "before_tax"},
    "time_rounding": {"Forklift time": {"step_min": 15, "mode": "total"}},
    "adjustment_direction": {"Rush supplement": "add"},
}


def _find(state: str, text: str, n: int) -> int:
    i = -1
    for _ in range(n + 1):
        i = state.index(text, i + 1)
    return i


def span(state, text, role, item=None, target=None, n=0, p=1.0):
    a = _find(state, text, n)
    return TagSpan(a, a + len(text), text, role, item, target, "state", p)


def tags_for(state=STATE, spans=SPANS, flags=FLAGS, **kw) -> Tags:
    return Tags(spans=[span(state, t, r, i, g, n) for t, n, r, i, g in spans], flags=flags, **kw)


def with_spans(drop=(), add=(), state=STATE, flags=FLAGS) -> Tags:
    kept = [s for s in SPANS if s[0] not in drop]
    return tags_for(state, kept + list(add), flags)


def refusal(tags, state=STATE) -> str:
    with pytest.raises(Refusal) as e:
        assemble(tags, state)
    return e.value.code


# --- K1 ----------------------------------------------------------------------------------------------------------


def test_assembler_gives_from_inputs_block():
    asm = assemble(tags_for(), STATE)
    assert asm.block().text == from_input(EXPECTED).text
    assert asm.input["items"][1]["per"] == "h"
    assert asm.input["items"][1]["round"] == {"step": "15", "mode": "total"}
    assert asm.input["tax"]["outside"] == ["Handling fee"]


def test_labelled_rows_round_trip():
    rows = [{"state": STATE, "spans": [dataclasses.asdict(s) for s in tags_for().spans], "flags": FLAGS}]
    for r in rows[0]["spans"]:
        r.pop("p"), r.pop("where")
    tags = Tags.from_row(rows[0])
    assert assemble(tags, STATE).block().text == from_input(EXPECTED).text
    assert Tags.from_row(dict(rows[0], outcome="abstain")).abstain_p == 1.0


def test_line_order_follows_the_text_and_mentions_merge():
    # a second mention of the same price in the same line is one value, not two
    state = STATE + "\nCrate packing is $12.50 a crate."
    tags = with_spans(add=[("$12.50", 1, "price", "i1", None)], state=state)
    assert assemble(tags, state).block().text == from_input(EXPECTED).text


# --- K2 ----------------------------------------------------------------------------------------------------------


def test_refusals_on_inconsistent_tags():
    assert refusal(with_spans(drop={"Crate packing"})) == "no-label"
    assert refusal(with_spans(drop={"$12.50"})) == "item-roles"
    second = STATE + "\nCrate packing is $13.00 a crate."
    two_prices = with_spans(add=[("$13.00", 0, "price", "i1", None)], state=second)
    assert refusal(two_prices, second) == "conflicting-values"  # two prices for one item
    bad_target = [s if s[0] != "10%" else ("10%", 0, "percent", "a1", "i9") for s in SPANS]
    assert refusal(tags_for(spans=bad_target)) == "target"
    fee_target = [s if s[0] != "10%" else ("10%", 0, "percent", "a1", "f1") for s in SPANS]
    assert refusal(tags_for(spans=fee_target)) == "target"
    no_item = [s if s[0] != "$7.25" else ("$7.25", 0, "price", None, None) for s in SPANS]
    assert refusal(tags_for(spans=no_item)) == "role-without-item"
    odd_item = [s if s[0] != "$7.25" else ("$7.25", 0, "price", "x1", None) for s in SPANS]
    assert refusal(tags_for(spans=odd_item)) == "unknown-item"
    labelled_number = [s if s[0] != "$7.25" else ("$7.25", 0, "percent", "f1", None) for s in SPANS]
    assert refusal(tags_for(spans=labelled_number)) == "span-value"
    assert refusal(with_spans(drop={"8%"})) == "no-tax"


def test_refusals_on_spans_and_text():
    good = tags_for()
    shifted = dataclasses.replace(good.spans[3], start=good.spans[3].start + 1)
    assert refusal(Tags(spans=(shifted,) + good.spans[4:], flags=FLAGS)) == "span-text"
    overlap = span(STATE, "4 crates at", "distractor")
    assert refusal(Tags(spans=good.spans + (overlap,), flags=FLAGS)) == "span-overlap"
    cut = [s if s[0] != "$12.50" else ("$12.5", 0, "price", "i1", None) for s in SPANS]
    assert refusal(tags_for(spans=cut)) == "span-value"  # a value crosses the span's end
    unknown = Tags(spans=good.spans + (span(STATE, "each", "colour", "i1"),), flags=FLAGS)
    assert refusal(unknown) == "unknown-role"


def test_refusals_on_flags():
    assert refusal(tags_for(flags={k: v for k, v in FLAGS.items() if k != "tax_kind"})) == "missing-flag"
    assert refusal(tags_for(flags={k: v for k, v in FLAGS.items() if k != "tax_after"})) == "missing-flag"
    assert refusal(tags_for(flags={k: v for k, v in FLAGS.items() if k != "adjustment_direction"})) == "missing-flag"
    assert refusal(tags_for(flags={k: v for k, v in FLAGS.items() if k != "time_rounding"})) == "missing-flag"
    conflict = dict(FLAGS, credit_position={"Loyalty credit": "after_tax"})
    assert refusal(tags_for(flags=conflict)) == "flag-conflict"
    assert refusal(tags_for(flags=dict(FLAGS, tax_after=["Cable"]))) == "flag-conflict"
    wrong_step = dict(FLAGS, time_rounding={"Forklift time": {"step_min": 10, "mode": "total"}})
    assert refusal(tags_for(flags=wrong_step)) == "flag-conflict"


def test_refusals_on_rates_currencies_and_entries():
    two = STATE + "\nA second rate of 5% applies."
    assert refusal(with_spans(add=[("5%", 0, "tax_rate", "tax", None)], state=two), two) == "two-tax-rates"
    assert refusal(with_spans(add=[("5%", 0, "tax_rate", "tax2", None)], state=two), two) == "two-tax-rates"
    euro = STATE + "\nRef charge €3.00 noted."
    assert refusal(with_spans(add=[("€3.00", 0, "distractor", None, None)], state=euro), euro) == "two-currencies"

    small = "Visits: 30 and 30 minutes at $60.00 per hour. Tax 5% charged."
    spans = [
        ("Visits", 0, "item_label", "i1", None),
        ("30", 0, "duration", "i1", None),
        ("30 minutes", 0, "duration", "i1", None),
        ("$60.00", 0, "rate", "i1", None),
        ("5%", 0, "tax_rate", "tax", None),
    ]
    flags = {"tax_kind": "charged", "tax_after": []}
    assert refusal(tags_for(small, spans, flags), small) == "repeated-entry"
    no_unit = "Visits: 45 minutes at $60.00. Tax 5% charged."
    spans = [
        ("Visits", 0, "item_label", "i1", None),
        ("45 minutes", 0, "duration", "i1", None),
        ("$60.00", 0, "rate", "i1", None),
        ("5%", 0, "tax_rate", "tax", None),
    ]
    assert refusal(tags_for(no_unit, spans, flags), no_unit) == "unit"


# --- K3 ----------------------------------------------------------------------------------------------------------


def test_provenance_gate():
    asm = assemble(tags_for(), STATE)
    assert gates.provenance(asm).ok
    ops = list(asm.operands)
    ops[1] = dataclasses.replace(ops[1], value=ops[1].value + 1)
    bad = gates.provenance(dataclasses.replace(asm, operands=tuple(ops)))
    assert (bad.ok, bad.code) == (False, "value")
    ops = list(asm.operands)
    ops[1] = dataclasses.replace(ops[1], spans=())
    assert gates.provenance(dataclasses.replace(asm, operands=tuple(ops))).code == "no-span"


def test_numbers_gate():
    assert gates.numbers(tags_for(), STATE).ok
    missing = with_spans(drop={"page 1 of 2"})
    check = gates.numbers(missing, STATE)
    assert (check.ok, check.code) == (False, "untagged-digits")
    words = STATE + "\nTwo copies were sent."
    check = gates.numbers(tags_for(), words)
    assert (check.ok, check.code) == (False, "untagged-value")  # a number in words counts too


def test_units_gate():
    assert gates.units(assemble(tags_for(), STATE)).ok
    swapped = STATE.replace("1 lb = 0.4536 kg", "1 lb = 4.5360 kg")
    spans = [s if s[0] != "0.4536 kg" else ("4.5360 kg", 0, "factor_to", "i3", None) for s in SPANS]
    check = gates.units(assemble(tags_for(swapped, spans), swapped))
    assert (check.ok, check.code) == (False, "factor-size")
    hours = STATE.replace("entries of 50 and 75 minutes", "entries of 50 and 75 minutes (hours: 3)")
    tags = tags_for(hours)
    a = hours.index("hours: 3") + len("hours: ")
    tags = Tags(spans=tags.spans + (TagSpan(a, a + 1, "3", "duration", "i2"),), flags=FLAGS)
    check = gates.units(assemble(tags, hours))
    assert (check.ok, check.code) == (False, "bare-minutes")


def _figures(asm):
    return {name: x for name, x, _ in gates.figures(asm)}


def test_stated_figures_gate():
    asm = assemble(tags_for(), STATE)
    assert gates.stated_figures(asm).code == "no-stated-figure"
    f = _figures(asm)
    base, total = round(f["taxed-base"], 2), round(f["total"], 2)
    stated = STATE + f"\nTaxed base: ${base:.2f}. Total due: ${total:.2f}."
    extra = [(f"${base:.2f}", 0, "stated_total", None, None), (f"${total:.2f}", 0, "stated_total", None, None)]
    asm = assemble(with_spans(add=extra, state=stated), stated)
    assert gates.stated_figures(asm).ok
    partial = assemble(
        with_spans(add=extra[:1], state=stated[: stated.index(" Total")]), stated[: stated.index(" Total")]
    )
    assert gates.stated_figures(partial).code == "partial"  # the fee after tax is not covered by the base
    wrong = STATE + "\nTaxed base: $999.99."
    asm = assemble(with_spans(add=[("$999.99", 0, "stated_total", None, None)], state=wrong), wrong)
    assert gates.stated_figures(asm).code == "stated-mismatch"


def test_agreement_gate():
    text = from_input(EXPECTED).text
    assert gates.agreement(text, text).ok
    assert gates.agreement(text, text.replace("8", "9")).code == "disagree"
    assert gates.agreement(text, None).code == "no-rules-block"


# --- K4 ----------------------------------------------------------------------------------------------------------


def test_threshold_and_margin_band():
    th = Thresholds(role=0.9, link=0.9, flag=0.9, band=0.02)
    assert gates.confidence(tags_for(), Thresholds()).code == "no-thresholds"
    assert gates.confidence(tags_for(), th).ok
    sure = tags_for(confidences={"link:a|b": 0.95, "flag:tax_kind": 0.99})
    assert gates.confidence(sure, th).ok
    assert gates.confidence(tags_for(confidences={"link:a|b": 0.91}), th).code == "margin-band"
    assert gates.confidence(tags_for(confidences={"link:a|b": 0.89}), th).code == "margin-band"
    assert gates.confidence(tags_for(confidences={"link:a|b": 0.87}), th).code == "below-threshold"
    assert gates.confidence(tags_for(abstain_p=0.09), th).code == "margin-band"
    low_role = tags_for()
    low_role = Tags(spans=(dataclasses.replace(low_role.spans[0], p=0.5),) + low_role.spans[1:], flags=FLAGS)
    assert gates.confidence(low_role, th).code == "below-threshold"
    assert gates.confidence(tags_for(confidences={"other:x": 1.0}), th).code == "unknown-decision"
    for bad in (dict(role=0.4, link=0.9, flag=0.9), dict(role=0.99, link=0.9, flag=0.9, band=0.02), dict(band=0.6)):
        with pytest.raises(ValueError):
            Thresholds(**bad)


# --- K5 ----------------------------------------------------------------------------------------------------------


def test_determinism():
    a, b = assemble(tags_for(), STATE), assemble(tags_for(), STATE)
    assert a.input == b.input and a.operands == b.operands
    th = Thresholds(role=0.9, link=0.9, flag=0.9)
    r1 = read(STATE, tagger=StubTagger({STATE: tags_for()}), thresholds=th)
    r2 = read(STATE, tagger=StubTagger({STATE: tags_for()}), thresholds=th)
    assert r1 == r2


# --- K6 ----------------------------------------------------------------------------------------------------------

TH = Thresholds(role=0.9, link=0.9, flag=0.9, band=0.02)


def test_read_threshold_path():
    out = read(STATE, tagger=StubTagger({STATE: tags_for()}), thresholds=TH)
    assert isinstance(out, Shown) and isinstance(out, Annotation)
    assert (out.path, out.text) == ("threshold", from_input(EXPECTED).text)
    off = read(STATE, tagger=StubTagger({STATE: tags_for()}))
    assert isinstance(off, Refused) and isinstance(off, Abstain)
    assert (off.reason, off.gate) == ("cascade-no-thresholds", "accept")


def test_read_stated_figures_path_and_veto():
    f = _figures(assemble(tags_for(), STATE))
    stated = STATE + f"\nTotal due: ${round(f['total'], 2):.2f}."
    tags = with_spans(add=[(f"${round(f['total'], 2):.2f}", 0, "stated_total", None, None)], state=stated)
    out = read(stated, tagger=StubTagger({stated: tags}))
    assert isinstance(out, Shown) and out.path == "stated_figures"
    wrong = STATE + "\nTotal due: $1.00."
    tags = with_spans(add=[("$1.00", 0, "stated_total", None, None)], state=wrong)
    out = read(wrong, tagger=StubTagger({wrong: tags}), thresholds=TH)
    assert isinstance(out, Refused) and out.reason == "cascade-stated-mismatch"


def test_read_with_the_rule_reader():
    block = from_input(EXPECTED)
    out = read(STATE, rules_case=block, tagger=StubTagger({STATE: tags_for()}))
    assert isinstance(out, Shown) and out.path == "rules_agree"
    other = Annotation(block.version, block.text.replace("Tax charged", "Tax withheld"), block.lines)
    out = read(STATE, rules_case=other, tagger=StubTagger({STATE: tags_for()}), thresholds=TH)
    assert isinstance(out, Refused) and out.reason == "cascade-readers-disagree"
    broken = with_spans(drop={"$12.50"})
    out = read(STATE, rules_case=block, tagger=StubTagger({STATE: broken}))
    assert isinstance(out, Shown) and out.path == "rules_only"  # never less than the rule reader
    out = read(STATE, rules_case=Abstain("unparsed-rule"), tagger=StubTagger({STATE: broken}), thresholds=TH)
    assert isinstance(out, Refused) and (out.gate, out.reason) == ("assemble", "cascade-item-roles")


def test_read_tagger_abstains_and_failures():
    out = read(STATE, tagger=StubTagger({STATE: tags_for(abstain_p=0.8)}), thresholds=TH)
    assert isinstance(out, Refused) and out.reason == "cascade-tagger-abstain"

    def boom(state, context):
        raise RuntimeError("no model")

    out = read(STATE, tagger=StubTagger(boom), thresholds=TH)
    assert isinstance(out, Refused) and out.reason == "cascade-tagger-error"
    out = read("Unknown text with $5.00.", tagger=StubTagger({}), thresholds=TH)
    assert isinstance(out, Refused) and out.reason == "cascade-unknown-text"
    missing = with_spans(drop={"WO-5531"})
    out = read(STATE, tagger=StubTagger({STATE: missing}), thresholds=TH)
    assert isinstance(out, Refused) and (out.gate, out.reason) == ("numbers", "cascade-untagged-digits")


def test_annotate_reads_through_the_cascade_on_request():
    block = from_input(EXPECTED)
    th = {"role": 0.9, "link": 0.9, "flag": 0.9, "band": 0.02}
    out = annotate(STATE, reader="cascade", tagger=StubTagger({STATE: tags_for()}), thresholds=th)
    assert type(out) is Annotation and out == block and out.text == block.text  # the path takes no part in equality
    assert out.path == "threshold" and block.path == ""
    off = annotate(STATE, reader="cascade", tagger=StubTagger({STATE: tags_for()}))
    assert type(off) is Abstain and (off.reason, off.path) == ("cascade-no-thresholds", "abstain")
    assert off == Abstain("cascade-no-thresholds")
    default = annotate(STATE)  # the rule reader, unchanged: this wording is beyond it
    assert type(default) is Abstain and default.path == "" and default.reason != "cascade-no-thresholds"
    with pytest.raises(ValueError):
        annotate(STATE, reader="cascade")
    with pytest.raises(ValueError):
        annotate(STATE, reader="model")
    with pytest.raises(TypeError):
        annotate(STATE, reader="cascade", tagger=object())


def test_context_spans():
    context = "Tax of 8% is charged on every bill."
    state = STATE.replace("\nTax of 8% is charged on the bill.", "")
    spans = [span(state, t, r, i, g, n) for t, n, r, i, g in SPANS if t != "8%"]
    a = context.index("8%")
    spans.append(TagSpan(a, a + 2, "8%", "tax_rate", "tax", None, "context"))
    out = read(state, context, tagger=StubTagger({state: Tags(spans=spans, flags=FLAGS)}), thresholds=TH)
    assert isinstance(out, Shown) and out.text == from_input(EXPECTED).text


# --- tagger helpers ----------------------------------------------------------------------------------------------


def test_decode_bio():
    labels = ["O", "B-price", "I-price", "B-quantity", "I-quantity"]
    offsets = [(0, 0), (0, 1), (1, 3), (4, 6), (6, 8), (9, 10), (0, 0)]
    ids = [0, 1, 2, 0, 4, 3, 0]
    probs = [1, 0.9, 0.8, 1, 0.7, 0.95, 1]
    assert decode_bio(offsets, ids, probs, labels) == [
        (0, 3, "price", 0.8),
        (6, 8, "quantity", 0.7),
        (9, 10, "quantity", 0.95),
    ]


def test_decode_bio_joins_one_word():
    text = "at $45.44 each, 4 crates"
    labels = ["O", "B-price", "I-price", "B-quantity"]
    offsets = [(0, 2), (2, 6), (6, 7), (7, 9), (9, 10), (15, 17)]  # "at" " $45" "." "44" "," " 4" (leading spaces)
    ids = [0, 1, 1, 1, 0, 3]
    probs = [1, 0.9, 0.6, 0.8, 1, 0.7]
    assert decode_bio(offsets, ids, probs, labels, text) == [(2, 9, "price", 0.6), (15, 17, "quantity", 0.7)]
    assert len(decode_bio(offsets, ids, probs, labels)) == 4  # without the text, each B- starts a span
    two = "4 5"
    assert decode_bio([(0, 1), (1, 3)], [3, 3], [1, 1], labels, two) == [(0, 1, "quantity", 1), (1, 3, "quantity", 1)]


def test_weights_are_pinned(tmp_path):
    with pytest.raises(ValueError):
        resolve("someone/tagger")  # a Hub repository without a commit hash
    with pytest.raises(ValueError):
        resolve("someone/tagger", revision="main")
    (tmp_path / "w.bin").write_bytes(b"weights")
    (tmp_path / "SHA256SUMS").write_text(f"{hashlib.sha256(b'weights').hexdigest()}  w.bin\n")
    check_sums(tmp_path)
    (tmp_path / "w.bin").write_bytes(b"changed")
    with pytest.raises(ValueError):
        check_sums(tmp_path)


# --- K7 ----------------------------------------------------------------------------------------------------------


def _tiny_tagger(directory):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    tokenizers = pytest.importorskip("tokenizers")
    from decisio.calculator.cascade import model as M

    tok = tokenizers.Tokenizer(tokenizers.models.WordPiece(unk_token="[UNK]"))
    tok.normalizer = tokenizers.normalizers.Lowercase()
    tok.pre_tokenizer = tokenizers.pre_tokenizers.Sequence(
        [tokenizers.pre_tokenizers.Whitespace(), tokenizers.pre_tokenizers.Digits(individual_digits=True)]
    )
    special = ["[PAD]", "[UNK]", "[CLS]", "[SEP]"]
    tok.train_from_iterator([STATE] * 4, tokenizers.trainers.WordPieceTrainer(vocab_size=300, special_tokens=special))
    tok.post_processor = tokenizers.processors.TemplateProcessing(
        single="[CLS] $A [SEP]", special_tokens=[("[CLS]", 2), ("[SEP]", 3)]
    )
    fast = transformers.PreTrainedTokenizerFast(
        tokenizer_object=tok, pad_token="[PAD]", unk_token="[UNK]", cls_token="[CLS]", sep_token="[SEP]"
    )
    config = transformers.BertConfig(
        vocab_size=fast.vocab_size, hidden_size=32, num_hidden_layers=1, num_attention_heads=2, intermediate_size=64
    )
    torch.manual_seed(0)
    model = M.build(transformers.AutoModel.from_config(config))
    row = {"state": STATE, "spans": [dataclasses.asdict(s) for s in tags_for().spans], "flags": FLAGS}
    ex = M.encode_row(row, fast, 512)
    assert ex is not None and ex["pairs"] and ex["targets"] and ex["attrs"]["after_tax"]
    total, parts = M.loss(model, ex)
    total.backward()
    assert {"tokens", "link", "target", "attr_after_tax", "global_tax_kind"} <= set(parts)
    M.save(model, fast, directory, {"max_length": 512})


@pytest.mark.slow  # imports torch and transformers (about 10 s); the model itself is tiny
def test_model_wrapper_reads_with_a_tiny_local_model(tmp_path):
    from decisio.calculator.cascade.tagger import HFTagger

    _tiny_tagger(tmp_path)
    tagger = HFTagger.load(str(tmp_path))
    first, second = tagger.predict(STATE), tagger.predict(STATE)
    assert first == second  # batch size 1, eval mode, fixed weights
    assert all(s.text == STATE[s.start : s.end] for s in first.spans)
    out = read(STATE, tagger=tagger, thresholds=TH)
    assert isinstance(out, (Shown, Refused))  # an untrained model: anything but a crash
    assert tagger_for(str(tmp_path)) is tagger_for(str(tmp_path))  # a model directory is loaded once
    via = annotate(STATE, reader="cascade", tagger=str(tmp_path), thresholds=TH)
    assert isinstance(via, (Annotation, Abstain)) and via.path
    long = HFTagger(tagger.model, tagger.tokenizer, dict(tagger.meta, max_length=8))
    assert long.predict(STATE).note == "too-long"  # more than MAX_WINDOWS windows
    n = len(tagger.tokenizer(STATE)["input_ids"])
    windowed = HFTagger(tagger.model, tagger.tokenizer, dict(tagger.meta, max_length=n // 3 + 2))
    got = windowed.predict(STATE)
    assert not got.note and got == windowed.predict(STATE)
    assert all(s.text == STATE[s.start : s.end] for s in got.spans)


@pytest.mark.parametrize(("n", "size"), [(5, 10), (10, 10), (11, 10), (40, 10), (1000, 510)])
def test_windows_cover_every_token_with_overlap(n, size):
    from decisio.calculator.cascade.model import windows

    ws = windows(n, size)
    assert ws[0][0] == 0 and ws[-1][1] == n and all(b - a <= size for a, b in ws)
    assert all(a2 < b1 for (_, b1), (a2, _) in zip(ws, ws[1:]))  # neighbours overlap: no seam without context
    if n > size:
        assert all(b1 - a2 >= size // 4 for (_, b1), (a2, _) in zip(ws, ws[1:]))


# ---- rule forms the calculator cannot express veto the block --------------------------------------------------------


@pytest.mark.parametrize(
    ("sentence", "code"),
    [
        ("The total time is rounded once, UP to the next 60-minute step.", "rounding-direction"),
        ("Time is rounded down to the nearest quarter hour.", "rounding-direction"),
        ("Any part hour counts as a full hour.", "rounding-direction"),
        ("A minimum charge of $50 applies.", "limit-rule"),
        ("The fee is $20 or 5% of the order, whichever is greater.", "limit-rule"),
        ("Round only the final money figure to the cent, half a cent going up.", ""),
        ("Each entry is rounded to the nearest 6 minutes, then they are added.", ""),
        ("Tax of 10% is added to everything except the handling fees.", ""),
        ("Each entry is rounded to a step of 5 minutes, then the rounded entries are added up.", ""),
        ("Each length is rounded on its own to 0.5 m, and the rounded lengths are summed up at the rate.", ""),
        ("Each entry is rounded to 15 minutes; together these make up the time charged.", ""),
        ("Each entry is rounded up to 15 minutes, then the entries are added up.", "rounding-direction"),
        ("The total is rounded down after the entries are added up.", "rounding-direction"),
    ],
)
def test_rule_forms_gate(sentence, code):
    from decisio.calculator.cascade import gates

    check = gates.rule_forms(f"Order 12 for a client.\n{sentence}\n- Goods: 2 x $3.00")
    assert check.code == code and check.ok == (code == "")
    assert gates.rule_forms("Goods: 2 x $3.00", context=sentence).code == code  # the context is read too


# ---- the text's own words agree with the case (`wording`) -----------------------------------------------------------


def _wording(state, spans=SPANS, flags=FLAGS):
    from decisio.calculator.cascade import gates

    return gates.wording(assemble(tags_for(state, spans, flags), state))


def test_wording_passes_the_reference_case():
    assert _wording(STATE).ok


def test_wording_refuses_a_label_cut_out_of_a_phrase():
    spans = [("packing", 0, "item_label", "i1", None) if s[0] == "Crate packing" else s for s in SPANS]
    check = _wording(STATE, spans)
    assert (check.ok, check.code) == (False, "label-cut")


def test_wording_does_not_read_the_word_after_a_label():
    state = STATE.replace("Crate packing: 4 crates", "The Crate packing line holds 4 crates")
    assert _wording(state).ok


def test_wording_refuses_a_line_taken_off_with_no_word_that_says_so():
    state = STATE.replace("Loyalty credit: $5.00, taken off before tax.", "Loyalty bonus: $5.00, before tax.")
    spans = [("Loyalty bonus",) + s[1:] if s[0] == "Loyalty credit" else s for s in SPANS]
    flags = dict(FLAGS, credit_position={"Loyalty bonus": "before_tax"})
    check = _wording(state, spans, flags)
    assert (check.ok, check.code, check.detail) == (False, "no-subtract-word", "c1")


@pytest.mark.parametrize(
    "sentence",
    [
        "Loyalty credit: $5.00, taken off once the tax has been worked out.",
        "Loyalty credit: $5.00, taken off; it is not part of the amount the tax is charged on.",
        "Loyalty credit: $5.00, taken off after tax.",
        "Loyalty credit: $5.00 is taken off. It comes off once the tax has been worked out.",
        "Loyalty credit: $5.00, taken off. The credit is not part of the amount the tax is charged on.",
    ],
)
def test_wording_refuses_a_credit_on_the_other_side_of_the_tax(sentence):
    state = STATE.replace("Loyalty credit: $5.00, taken off before tax.", sentence)
    check = _wording(state)
    assert (check.ok, check.code, check.detail) == (False, "tax-side", "c1")
    flags = dict(FLAGS, tax_after=["Handling fee", "Loyalty credit"], credit_position={"Loyalty credit": "after_tax"})
    assert _wording(state, SPANS, flags).ok  # read on the side the text says


def test_wording_refuses_a_fee_the_text_puts_before_tax():
    state = STATE.replace("Handling fee: $7.25, added after tax.", "Handling fee: $7.25, inside the taxed amount.")
    check = _wording(state)
    assert (check.ok, check.code, check.detail) == (False, "tax-side", "f1")


def test_decoded_spans_snap_to_the_value_they_cut():
    pytest.importorskip("torch")
    from decisio.calculator.cascade.model import snap_values, value_bounds

    text = "Design hours: 1 h 41 min at $22.66 per hour, 3 x $5.00."
    values = value_bounds(text, None, 0)
    at = lambda t: (text.index(t), text.index(t) + len(t))  # noqa: E731
    raw = [
        (*at("h 41 min"), "duration", 0.9),  # cut out of "1 h 41 min"
        (*at("$22"), "rate", 0.8),  # "$22" and ".66" decoded apart: one span, the lower p
        (text.index(".66"), text.index(".66") + 3, "rate", 0.7),
        (*at("Design hours"), "item_label", 0.99),  # words are never snapped
        (*at("3 x $5.00"), "price", 0.6),  # over two values: left for the assembler to refuse
    ]
    got = snap_values(raw, values, text, 0)
    assert [(text[s:e], k, p) for s, e, k, p in got] == [
        ("1 h 41 min", "duration", 0.9),
        ("$22.66", "rate", 0.7),
        ("Design hours", "item_label", 0.99),
        ("3 x $5.00", "price", 0.6),
    ]


def test_wording_reads_on_only_within_the_line_and_after_a_pronoun():
    # the next line's "It" is about something else: not read as the credit's side
    old = "Loyalty credit: $5.00, taken off before tax."
    state = STATE.replace(old, old + "\nIt is all due after tax.")
    assert _wording(state).ok
    # a sentence that does not go on about the line (no pronoun) is not read either
    state = STATE.replace(old, old + " Payment is due after tax day.")
    assert _wording(state).ok


@pytest.mark.parametrize(
    "sentence",
    [
        "Crate packing: 4 crates at $12.50 each, a credit that comes off the bill.",
        "Crate packing: 4 crates at $12.50 each. It is a credit of $20.00 to the client.",
    ],
)
def test_wording_refuses_an_added_line_the_text_takes_off_the_bill(sentence):
    state = STATE.replace("Crate packing: 4 crates at $12.50 each.", sentence)
    check = _wording(state)
    assert (check.ok, check.code, check.detail) == (False, "comes-off-word", "i1")


def test_wording_does_not_read_a_percentage_taken_from_a_charge_as_the_line_coming_off():
    state = STATE.replace(
        "Rush supplement: 10% of Crate packing, added.", "Rush supplement: 10% of Crate packing, added."
    )
    assert _wording(state).ok
    state = STATE.replace(
        "Crate packing: 4 crates at $12.50 each.", "Crate packing: 4 crates at $12.50 each, nothing taken off."
    )
    assert _wording(state).ok


@pytest.mark.parametrize(
    ("line", "label"),
    [
        ("Crate packing units: 4 crates at $12.50 each.", "Crate packing"),
        ('The "Crate packing units" line: 4 crates at $12.50 each.', "Crate packing"),
        ("| Crate packing units | 4 crates | $12.50 each |", "Crate packing"),
    ],
)
def test_wording_refuses_a_label_that_stops_short_of_its_heading(line, label):
    state = STATE.replace("Crate packing: 4 crates at $12.50 each.", line)
    spans = [(label, 0, "item_label", "i1", None) if s[0] == "Crate packing" else s for s in SPANS]
    check = _wording(state, spans)
    assert (check.ok, check.code) == (False, "label-cut")


def test_wording_keeps_a_label_followed_by_words_of_its_sentence():
    state = STATE.replace(
        "Crate packing: 4 crates at $12.50 each.", "The Crate packing line holds 4 crates at $12.50 each."
    )
    assert _wording(state).ok


def test_wording_refuses_an_adjustment_the_text_turns_the_other_way():
    state = STATE.replace(
        "Rush supplement: 10% of Crate packing, added.", "Rush supplement: 10% of Crate packing, deducted."
    )
    check = _wording(state)
    assert (check.ok, check.code, check.detail) == (False, "adjustment-direction", "a1")
    flags = dict(FLAGS, adjustment_direction={"Rush supplement": "deduct"})
    assert _wording(state, SPANS, flags).ok  # read the way the text says
    state = STATE.replace(
        "Rush supplement: 10% of Crate packing, added.", "Rush supplement: 10% of Crate packing, is ADDED to the bill."
    )
    assert _wording(state, SPANS, flags).code in ("adjustment-direction", "no-subtract-word")  # refused either way


def test_decode_votes_one_family_per_span_and_keeps_the_likelier_span_per_value():
    torch = pytest.importorskip("torch")
    from decisio.calculator.cascade.model import LABELS, likelier_per_value, vote_spans

    text = "Breakdown allowance $88.56"
    offsets = [(0, 5), (5, 9), (10, 19), (20, 21), (21, 26)]  # "Break" "down" "allowance" "$" "88.56"
    probs = torch.full((5, len(LABELS)), 1e-4)

    def put(t, label, p):
        probs[t, LABELS.index(label)] = p

    put(0, "B-credit_label", 0.6), put(0, "B-item_label", 0.3)
    put(1, "I-adj_label", 0.5), put(1, "I-credit_label", 0.45)  # one word split across families token by token
    put(2, "I-credit_label", 0.9)
    put(3, "B-price", 0.5), put(3, "B-credit_amount", 0.4)
    put(4, "I-credit_amount", 0.9)
    top_p, top = vote_spans(probs, offsets, text)
    assert [LABELS[i] for i in top] == [
        "B-credit_label",
        "I-credit_label",
        "I-credit_label",
        "B-credit_amount",
        "I-credit_amount",
    ]
    assert all(p <= 0.9 for p in top_p)
    spans = [(20, 26, "credit_amount", 0.8), (20, 26, "price", 0.6), (0, 19, "credit_label", 0.7)]
    assert likelier_per_value(spans) == [(20, 26, "credit_amount", 0.8), (0, 19, "credit_label", 0.7)]


def test_a_number_in_words_restated_in_digits_is_one_value():
    state = STATE.replace("Crate packing: 4 crates at $12.50 each.", "Crate packing: four (4) crates at $12.50 each.")
    spans = [("four (4) crates", 0, "quantity", "i1", None) if s[0] == "4 crates" else s for s in SPANS]
    asm = assemble(tags_for(state, spans), state)
    assert asm.input["items"][0]["quantity"] == "4"
    from decisio.calculator.cascade import gates

    assert gates.provenance(asm).ok
    bad = state.replace("four (4) crates", "four (5) crates")  # the two disagree: two values, refused
    spans_bad = [("four (5) crates", 0, "quantity", "i1", None) if s[0] == "4 crates" else s for s in SPANS]
    assert refusal(tags_for(bad, spans_bad), bad) == "span-value"


def test_a_signed_credit_amount_is_read_by_its_magnitude():
    state = STATE.replace(
        "Loyalty credit: $5.00, taken off before tax.",
        "Loyalty credit: -$5.00, a credit of $5.00 taken off before tax.",
    )
    spans = [s for s in SPANS if s[:4] != ("$5.00", 0, "price", "c1")] + [
        ("-$5.00", 0, "price", "c1", None),
        ("$5.00", 1, "price", "c1", None),
    ]
    asm = assemble(tags_for(state, spans), state)
    assert Fraction(asm.input["credits"][0]["amount"]) == 5
    from decisio.calculator.cascade import gates

    assert gates.provenance(asm).ok
    other = state.replace("a credit of $5.00", "a credit of $6.00")
    spans2 = [s for s in spans if s[0] != "$5.00"] + [("$6.00", 0, "price", "c1", None)]
    assert refusal(tags_for(other, spans2), other) == "conflicting-values"


def test_a_lone_range_end_on_its_own_line_prices_nothing():
    state = "Date issued: 14 March 2025\n" + STATE
    spans = SPANS + [("14 March 2025", 0, "range_end", "i1", None)]
    asm = assemble(tags_for(state, spans), state)
    got, want = asm.input["items"][0], EXPECTED["items"][0]
    assert (
        got["label"] == want["label"] and Fraction(got["quantity"]) == 4 and Fraction(got["price"]) == Fraction("12.50")
    )
    # on the line of the priced item, it is not left out
    state2 = STATE.replace(
        "Crate packing: 4 crates at $12.50 each.", "Crate packing: 4 crates at $12.50 each, until 14 March 2025."
    )
    assert refusal(tags_for(state2, spans), state2) in ("range", "item-roles", "span-value")


def test_a_generic_continuation_before_the_heading_colon_is_not_a_cut():
    state = STATE.replace("Crate packing: 4 crates at $12.50 each.", "Crate packing entries: 4 crates at $12.50 each.")
    assert _wording(state).ok


def test_wording_refuses_a_rounding_step_whose_sentence_does_not_round():
    state = STATE.replace(
        "the total is rounded once to the nearest 15 minutes.", "the total is converted at 15 minutes per quarter hour."
    )
    flags = dict(FLAGS)
    check = _wording(state, SPANS, flags)
    assert (check.ok, check.code) == (False, "rounding-without-rule")


def test_wording_reads_an_imperative_add_as_the_adjustment_direction():
    state = STATE.replace("Rush supplement: 10% of Crate packing, added.", "Rush supplement: add 10% of Crate packing.")
    flags = dict(FLAGS, adjustment_direction={"Rush supplement": "deduct"})
    assert _wording(state, SPANS, flags).code in ("adjustment-direction", "no-subtract-word")
    assert _wording(state).ok  # read as added: right


def test_wording_refuses_a_label_short_of_its_heading_line():
    state = STATE.replace(
        "Crate packing: 4 crates at $12.50 each.", "CHARGE: Crate packing units\n4 crates at $12.50 each."
    )
    spans = [("Crate packing", 0, "item_label", "i1", None) if s[0] == "Crate packing" else s for s in SPANS]
    assert _wording(state, spans).code == "label-cut"


def test_wording_reads_the_sentence_that_introduces_an_enumeration():
    state = STATE.replace(
        "Loyalty credit: $5.00, taken off before tax.",
        "Two further items are applied after the tax. First, nothing. Second, the Loyalty credit is $5.00, taken off.",
    )
    check = _wording(state)  # the credit read before the tax, against the introducing sentence
    assert (check.ok, check.code) == (False, "tax-side")


def test_wording_refuses_a_label_cut_inside_a_word():
    spans = [("packing", 0, "item_label", "i1", None) if s[0] == "Crate packing" else s for s in SPANS]
    state = STATE.replace("Crate packing: 4 crates", "Cratepacking: 4 crates")
    assert _wording(state, spans).code == "label-cut"


def test_wording_heading_line_cut_needs_a_heading_keyword():
    state = STATE.replace("Crate packing: 4 crates at $12.50 each.", "Crate packing units\n4 crates at $12.50 each.")
    spans = [("Crate packing", 0, "item_label", "i1", None) if s[0] == "Crate packing" else s for s in SPANS]
    assert _wording(state, spans).ok  # a line with no heading keyword (a title) is not read


def test_wording_rounding_words_are_read_over_a_line_wrap():
    state = STATE.replace(
        "the total is rounded once to the nearest 15 minutes.", "the total is rounded once to the\n nearest 15 minutes."
    )
    assert _wording(state).ok
