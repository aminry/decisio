"""Scores, not just answers, for the published FP8 31B (the stored repository tachara-ai/decisio-gemma-4-31b at 6673a7f4), against Google's bf16 quantized on load.

    python3 score_compare.py            (from the S12 folder; reads results/box and the committed record copied under analysis/)

Part 1, JevBench's 231 public items: S12 part J's run of the stored FP8 (decisio v0.12.0, vLLM 0.31.0) against the committed on-load record `runs/2026-10-04_gemma-4-31b/jevbench_g31/` (decisio v0.12.0's tree):
  the harness's own summaries (accuracy, ECE and Brier per tier; `summary.json` of each file) and decisio's v1.5 reading (`v15.json`: per type and tier), the scorers Lab 1's launch facts use
  (experiments/2026-10-09_lab1_launch_facts: the published tiers 1.000 / 1.000 / 0.838 and the ECE 0.035 and 0.091 on the standard and hard tiers are these summaries' fields).
  The committed on-load record holds summaries only, no item files, so the number of top answers that changed cannot be counted against it; the stored run is also compared item by item with the stored run of
  Lab 1's re-run session (2026-10-13, an older stack), to show how much the stored model moves between sessions on its own.
Part 2, F's 200 answers: the stored repository against Google's bf16 on load, same box, same stack, same 200 requests: accuracy overall and per task against the suite's gold, ECE (10 equal-width bins on the top probability,
  as the harness's summaries), the mean top probability, and the top answers that changed.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
S12 = HERE.parent
RLCD = S12.parents[1]
T3 = RLCD / "experiments/2026-10-05_t3_engine_death_gates/code"
sys.path.insert(0, str(T3))
import gates as g  # noqa: E402

J = S12 / "results/box/lab1/jevbench231_31b"
ON = HERE / "committed_onload_2026-10-04_g31"
OLDER = RLCD / "experiments/2026-10-13_lab2_tuned_rerun_and_s9_session/results/box/lab1runs/frozen/jevbench"
F = S12 / "results/box/F"
SUITE = RLCD / "experiments/2026-09-30_t1_step4_private_rerun/code/suite_items.json"
out = {}


def summ(root, f):
    d = json.loads((root / f / "summary.json").read_text())
    return {"accuracy": d["accuracy"], "n_correct": d["n_correct"], "n": d["n_scorable"], "ece": d["ece"]["ece"], "brier": d["brier_mean"]}


def v15(root):
    d = json.loads((root / "v15.json").read_text())
    return {t: {"cc": v["cc"], "tiers": v["tiers"], **({"abstention_rate": v["abstention_rate"]} if "abstention_rate" in v else {})} for t, v in d["per_type"].items()}, d["I_open_A"], d["I_open_B"]


# ---- part 1
tiers = ("easy", "original", "hard")
st = {f: summ(J, f) for f in tiers}
on = {f: summ(ON, f) for f in tiers}
out["jevbench"] = {"stored_fp8_now": st, "onload_committed_2026-10-04": on,
                   "correct_total": {"stored": sum(st[f]["n_correct"] for f in tiers), "onload": sum(on[f]["n_correct"] for f in tiers)}}
pt_s, ia_s, ib_s = v15(J)
pt_o, ia_o, ib_o = v15(ON)
out["jevbench"]["v15_per_type"] = {"stored": pt_s, "onload": pt_o, "I_open_A": [ia_s, ia_o], "I_open_B": [ib_s, ib_o]}


def items(root, f):
    r = {}
    for line in open(root / f / "results.jsonl"):
        x = json.loads(line)
        r[x["task_id"]] = x
    return r


if OLDER.exists():
    ch, tot, ident = {}, 0, 0
    for f in tiers:
        A, B = items(J, f), items(OLDER, f)
        ids = sorted(set(A) & set(B))
        tot += len(ids)
        ch[f] = {"items": len(ids), "predicted_answers_differ": sum(A[i]["predicted"] != B[i]["predicted"] for i in ids),
                 "identical_probabilities": sum(A[i]["probs"] == B[i]["probs"] for i in ids),
                 "largest_probability_difference": max(max(abs(A[i]["probs"].get(k, 0) - B[i]["probs"].get(k, 0)) for k in set(A[i]["probs"]) | set(B[i]["probs"])) for i in ids)}
    out["jevbench"]["stored_now_against_stored_2026-10-13"] = ch
    out["jevbench"]["stored_2026-10-13_summaries"] = {f: summ(OLDER, f) for f in tiers}

# ---- part 2
suite = {f"{x['task']}:{x['i']}": x for x in json.loads(SUITE.read_text())}
A = json.loads((F / "f_a.json").read_text())["answers"]
B = json.loads((F / "f_b.json").read_text())["answers"]


def hit(a, gold):
    keys, c = list(g.probs(a)), g.choice(a)
    gold = {"true": "yes", "false": "no"}.get(str(gold).lower(), gold)
    return c == str(gold).lower() or (str(gold).isdigit() and int(gold) < len(keys) and c == keys[int(gold)])


def stats(ans):
    rows = []
    for i, a in ans.items():
        gold = suite[i]["label"]
        p = g.probs(a)
        rows.append((i.split(":")[0], hit(a, gold), max(p.values())))
    def ece(rs):
        bins = [[] for _ in range(10)]
        for _t, h, c in rs:
            bins[min(int(c * 10), 9)].append((h, c))
        n = len(rs)
        return sum(len(b) / n * abs(sum(h for h, _ in b) / len(b) - sum(c for _, c in b) / len(b)) for b in bins if b)
    res = {"all": {"n": len(rows), "accuracy": sum(h for _, h, _ in rows) / len(rows), "ece": ece(rows), "mean_top_probability": sum(c for *_, c in rows) / len(rows)}}
    for t in sorted({r[0] for r in rows}):
        rr = [r for r in rows if r[0] == t]
        res[t] = {"n": len(rr), "accuracy": sum(h for _, h, _ in rr) / len(rr), "ece": ece(rr), "mean_top_probability": sum(c for *_, c in rr) / len(rr)}
    return res


sa, sb = stats(A), stats(B)
changed = [i for i in A if g.choice(A[i]) != g.choice(B[i])]
out["F200"] = {"stored": sa, "onload": sb, "top_answers_changed": len(changed), "which": changed,
               "correct_stored_wrong_onload": sum(hit(A[i], suite[i]["label"]) and not hit(B[i], suite[i]["label"]) for i in A),
               "wrong_stored_correct_onload": sum(not hit(A[i], suite[i]["label"]) and hit(B[i], suite[i]["label"]) for i in A)}
(HERE / "score_compare.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
