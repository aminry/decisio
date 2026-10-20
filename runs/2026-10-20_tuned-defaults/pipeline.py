# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
# ruff: noqa: E501  (a script that prints sentences)
"""The shipping path of a tuned model, end to end on a tiny checkpoint with the 31B's structure (CPU, no model):
merge an adapter into the bf16 base, write and check the repository's `/2` file, start from it, and read the temperature
a plain request is served at for each question type and choice band.

    uv run python runs/2026-10-20_tuned-defaults/pipeline.py [--out DIR]   > output.txt

The checkpoint and the adapter are synthetic (random tensors); what is checked is the machinery: the merge keeps the
adapter (the fp32 merge differs from the base in far more entries than a bf16 round trip), every pair lands, a null merge
equals `convert` byte for byte, the file is validated and its digests hold, `resolve_base` applies the defaults with the
flags above them, and the served probability equals softmax(log p / T) at the band's temperature to 1e-9.
"""

import argparse
import hashlib
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from safetensors.torch import save_file

from decisio import hub, hub_fp8
from decisio.serve.systemone import SystemOne, commit_noul
from decisio.serve.tasks import TaskStore
from decisio.serve.temperature import apply_temperature
from decisio.serve.vllm_engine import resolve_base

L = "model.language_model.layers"
R = 2


def bf16(*shape, scale=0.05, seed=0):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(*shape, generator=g) * scale).to(torch.bfloat16)


def checkpoint(root: Path):
    t = {}
    for i, has_v in enumerate((True, False)):  # layer 1 is full attention with no v_proj (attention_k_eq_v)
        t[f"{L}.{i}.self_attn.q_proj.weight"] = bf16(8, 6, seed=10 * i + 1)
        t[f"{L}.{i}.self_attn.k_proj.weight"] = bf16(4, 6, seed=10 * i + 2)
        if has_v:
            t[f"{L}.{i}.self_attn.v_proj.weight"] = bf16(4, 6, seed=10 * i + 3)
        t[f"{L}.{i}.self_attn.o_proj.weight"] = bf16(6, 8, seed=10 * i + 4)
        t[f"{L}.{i}.mlp.gate_proj.weight"] = bf16(10, 6, seed=10 * i + 5)
        t[f"{L}.{i}.mlp.up_proj.weight"] = bf16(10, 6, seed=10 * i + 6)
        t[f"{L}.{i}.mlp.down_proj.weight"] = bf16(6, 10, seed=10 * i + 7)
    t["model.language_model.embed_tokens.weight"] = bf16(20, 6, seed=90)
    d = root / "bf16"
    d.mkdir()
    save_file(t, str(d / "model-00001-of-00001.safetensors"), metadata={"format": "pt"})
    (d / "config.json").write_text(json.dumps({"model_type": "gemma4", "text_config": {"attention_k_eq_v": True}}))
    return d, t


def adapter(root: Path, t):
    g = torch.Generator().manual_seed(3)
    d = root / "adapter"
    d.mkdir()
    tensors = {}
    for name, w in t.items():
        if f"{L}." in name and name.endswith("_proj.weight"):
            module = name[: -len(".weight")].replace(
                "model.language_model.layers.", "model.layers.", 1
            )  # text-only layout
            tensors[f"base_model.model.{module}.lora_A.weight"] = torch.randn(R, w.shape[1], generator=g) * 0.02
            tensors[f"base_model.model.{module}.lora_B.weight"] = torch.randn(w.shape[0], R, generator=g) * 0.02
    save_file(tensors, str(d / "adapter_model.safetensors"))
    (d / "adapter_config.json").write_text(json.dumps({"r": R, "lora_alpha": 4}))
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", help="keep the files here")
    a = ap.parse_args()
    root = Path(a.out) if a.out else Path(tempfile.mkdtemp())
    root.mkdir(parents=True, exist_ok=True)
    base, t = checkpoint(root)
    ad = adapter(root, t)

    hub_fp8.convert(base, root / "converted")
    null = hub_fp8.merge(base, ad, root / "null", null=True)
    same = all(
        torch.equal(
            x.view(torch.uint8) if x.dtype == hub_fp8.FP8 else x, y.view(torch.uint8) if y.dtype == hub_fp8.FP8 else y
        )
        for x, y in (
            (hub_fp8.Source(root / "converted").get(n), hub_fp8.Source(root / "null").get(n))
            for n in hub_fp8.Source(root / "converted").map
        )
    )
    print(f"null merge equals convert byte for byte: {same}; loader changed {null['loader_changed_elements']} weights")
    merged_dir = root / "merged"
    meta = hub_fp8.merge(base, ad, merged_dir)
    print(
        f"merge: {meta['adapted_tensors']} adapter pairs applied ({meta['adapter_layout']} layout), "
        f"{meta['dropped_k_eq_v_pairs']} dropped, {meta['groups']} groups, written against loaded: "
        f"{'identical' if not meta['written_vs_loaded_differ'] else meta['written_vs_loaded_differ']}"
    )
    w = t[f"{L}.0.mlp.down_proj.weight"]
    g = torch.Generator().manual_seed(7)
    delta = torch.randn(w.shape, generator=g) * 0.0005 * float(w.float().abs().mean())
    print(
        f"adapter kept: fp32 sum differs from the weight in {int(((w.float() + delta) != w.float()).sum())} of {w.numel()} "
        f"entries, a bf16 round trip in {int(((w.float() + delta).to(torch.bfloat16) != w).sum())} (a change of 0.05 percent)"
    )

    temps = {
        "schema": "decisio.adapter/1",
        "temperatures": {"choice": 1.6, "noul": 4.2, "score": 6.2},
        "choice_bands": {"2-10": 2.4, "11-26": 1.7, "27-255": 1.3},
        "fit": {"data": "synthetic: no data", "n": {"choice": 0}},
    }
    (root / "temperatures.json").write_text(json.dumps(temps))
    basis = {"note": "synthetic: no evaluation behind this setting"}
    rc = hub.main(
        [
            "tuned-config", "--base", "gemma-4-31b", "--temperatures", str(root / "temperatures.json"), "--noul-commit",
            "--noul-commit-basis", json.dumps(basis), "--adapter-dir", str(ad), "--adapter-repository", "owner/adapter",
            "--adapter-revision", "0" * 40, "--merge-report", str(merged_dir / "MERGE_FP8.json"),
            "--weights-dir", str(merged_dir), "--out", str(merged_dir / hub.FILE),
        ]
    )  # fmt: skip
    config = json.loads((merged_dir / hub.FILE).read_text())
    print(f"tuned-config exit {rc}; schema {config['schema']}; defaults sha256 {hub.defaults_sha256(config)[:16]}...")
    print(f"check: exit {hub.main(['check', str(merged_dir / hub.FILE), '--weights-dir', str(merged_dir)])}")

    (merged_dir / "config.json").write_text(json.dumps({"model_type": "gemma4"}))
    args = SimpleNamespace(
        backend="vllm", base=str(merged_dir), model=None, revision=None, prompt_tail=None, answer_slot=None,
        label_variants=None, system_prompt=None, pad_to=None, served_name=None, noul_rendering=None, temperature=None,
        temperature_choice=None,
    )  # fmt: skip
    resolve_base(args)
    print(
        f"resolve_base: choice {args.temperature_choice}, yes/no {args.temperature_noul}, score {args.temperature_score}, "
        f"bands {args.choice_bands}, noul_commit {args.noul_commit}, task identity {args.model_identity}"
    )
    flagged = SimpleNamespace(**{**vars(args), "temperature_choice": 3.0, "noul_commit": False})
    for k in ("choice_bands", "repository_defaults"):
        flagged.__dict__.pop(k, None)
    resolve_base(flagged)
    print(
        f"with --temperature-choice 3.0 --no-noul-commit: choice {flagged.temperature_choice}, bands {flagged.choice_bands}, noul_commit {flagged.noul_commit}"
    )

    so = SystemOne(
        SimpleNamespace(tok=None), "decisio-tuned", task_store=TaskStore("fp"), noul_commit=args.noul_commit,
        temperature=1.0, choice_bands=args.choice_bands,
        temperatures={"choice": args.temperature_choice, "noul": args.temperature_noul, "score": args.temperature_score},
    )  # fmt: skip
    worst = 0.0
    for k in (3, 10, 11, 26, 27, 255):
        q = np.full(k, 0.1 / (k - 1)) if k > 1 else np.ones(1)
        q[0] = 0.9
        T = so.temperature_of("choice", None, k)
        got = apply_temperature(q, T)
        want = np.exp(np.log(q) / T) / np.exp(np.log(q) / T).sum()
        worst = max(worst, float(np.abs(got - want).max()))
        print(f"choice, {k} options: T = {T}")
    yes = apply_temperature(np.array([0.7, 0.3]), so.temperature_of("noul"))
    print(f"yes/no: T = {so.temperature_of('noul')}; P(yes) {yes[0]:.4f} -> committed {commit_noul(yes)[0]:.2f}")
    print(f"score: T = {so.temperature_of('score')}")
    print(f"largest difference of the served choice probabilities from softmax(log p / T): {worst:.1e}")
    print(
        "files:",
        ", ".join(sorted(hub.file_digests(merged_dir))[:3]),
        "...",
        len(hub.file_digests(merged_dir)),
        "listed by sha256",
    )
    print("sha256 of pipeline.py:", hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16])


if __name__ == "__main__":
    main()
