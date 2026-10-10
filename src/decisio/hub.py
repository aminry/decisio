# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""Hugging Face model repositories, one per base, and the `decision_config.json` each carries.

A decisio repository holds a base's weights, its licence and notice files, a card and `decision_config.json`: the
base's profile (decisio.families) as the decisio release it was exported from serves it. The profile in code stays the
source of truth. The file is generated from it (`export_decision_config`, `python -m decisio.hub export`), a test
regenerates it and fails on any difference from the copy kept in `hub/`, and a server started on a repository
(`--base owner/name[@revision]`) refuses to start when the file differs from the installed profile, so a profile
change cannot reach a repository without a new revision of it.

    python -m decisio.hub export --base gemma-4-31b                   # print the file
    python -m decisio.hub export --all --out hub                      # write hub/decision_config.<base>.json, each base
    python -m decisio.hub check hub/decision_config.gemma-4-31b.json  # compare a file with the installed profile

The schema, `decisio.decision_config/1`:

    schema      "decisio.decision_config/1"
    decisio     the decisio version the file was exported for (informative; not compared)
    base        the base's key (decisio.families)
    source      repository, revision and licence of the checkpoint the weights come from (not compared: a fact of the
                repository)
    weights     what the repository's tensors are: modified (false for a byte-for-byte copy), dtype, quantization (null,
                or the scheme), method (how a modified checkpoint was made), quantized_on_load (whether the server
                quantizes them again at start-up; false for a stored quantized checkpoint)
    model_types, moe, classes                                 how the checkpoint is recognised and which classes load it
    prompt      format (tail, slot, variants, system_prompt), label_form, noul_rendering, fingerprint (the task store's)
    readout     softcap, head_rows, head_label_logprobs, hidden_start
    temperatures  global and choice
    serving     served_name, pad_to, multi_question, register_state_boundary, register_boundary
    engine      the vLLM version the plugin supports

`source` and `weights` describe the repository; everything from `model_types` down is the profile, compared with the
installed one.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
from pathlib import Path

from decisio.families import BASES, Family
from decisio.names import task_fingerprint
from decisio.readout.letters import PromptFormat

SCHEMA = "decisio.decision_config/1"
FILE = "decision_config.json"
LICENSE = "apache-2.0"  # Qwen3.6 and Gemma 4 (RLCD experiments/2026-10-07_sd_hf_model_repos/PLAN.md)
QWEN = "qwen3.6-35b-a3b"

# the sections of the file that are the profile; the others describe the repository
PROFILE_KEYS = ("model_types", "moe", "classes", "prompt", "readout", "temperatures", "serving", "engine")

# the repository of each base. The 31B's weights are modified: its FP8 checkpoint is what the server would otherwise
# make on load. Its stored format is the one the identity gate keeps (compressed-tensors first, vLLM's own serialized
# FP8 if that fails; `python -m decisio.hub export --quantization`)
REPOSITORIES = {
    "gemma-4-31b": "tachara-ai/decisio-gemma-4-31b",
    "gemma-4-12b": "tachara-ai/decisio-gemma-4-12b",
    QWEN: "tachara-ai/decisio-qwen3.6-35b-a3b",
}
# the repositories were made under aminry/ for 0.10.0 and moved to tachara-ai/ (Tachara AI Lab) for 0.11.0; the Hub
# redirects the old names, and a server given one serves the repository under its new name, at the new pin
RENAMED = {f"aminry/{name.split('/')[1]}": name for name in REPOSITORIES.values()}
# the revision of each repository that a decisio release serves when none is given (--base owner/name@revision wins)
PINNED_REVISIONS: dict[str, str] = {
    "tachara-ai/decisio-gemma-4-31b": "931a9dadeb92a3533781606b3355799779c62cf5",
    "tachara-ai/decisio-gemma-4-12b": "a5a1ac8f2398147d4e5738cac9ab6dc138b13b82",
    "tachara-ai/decisio-qwen3.6-35b-a3b": "e691e3ae1d257720ecc919e9b988667dd0bd214a",
}
# the bases whose own key is served from a decisio repository rather than from the source's weights, once that
# repository's revision is pinned: the 31B's FP8 checkpoint saves quantizing Google's bf16 weights at every start
# (Amin, 2026-10-07; the others are byte-for-byte copies and keep their sources). `--model google/gemma-4-31B-it`
# serves Google's weights, quantized on load, as before.
DEFAULT_SOURCES = {"gemma-4-31b": REPOSITORIES["gemma-4-31b"]}


def default_source(base: str) -> str | None:
    """The repository `--base <base>` is served from when no checkpoint is named, or None (the source's own weights)."""
    name = DEFAULT_SOURCES.get(base)
    return f"{name}@{PINNED_REVISIONS[name]}" if name in PINNED_REVISIONS else None


def weights_of(base: str, quantization: str | None = None) -> dict:
    if base == "gemma-4-31b":
        return {
            "modified": True,
            "dtype": "fp8_e4m3",
            "quantization": quantization or "compressed-tensors",
            "method": (
                "quantized to FP8 exactly as vLLM 0.30.0 does on load (online per-tensor weight scales, activations "
                "quantized per token at run time); every other tensor, the vision tower included, is the source's; "
                "nothing was trained"
            ),
            "quantized_on_load": False,
        }
    if base == QWEN:
        return {
            "modified": False,
            "dtype": "fp8_e4m3",
            "quantization": "fp8",
            "method": None,
            "quantized_on_load": False,
        }
    return {"modified": False, "dtype": "bfloat16", "quantization": None, "method": None, "quantized_on_load": False}


def profile_of(fam: Family) -> dict:
    """The profile sections of the file, from a base's profile in code."""
    fmt = PromptFormat(
        tail=fam.prompt_tail, slot=fam.answer_slot, variants=fam.label_variants, system_prompt=fam.system_prompt
    )
    from decisio.vllm_plugin import SUPPORTED_VLLM

    return {
        "model_types": list(fam.model_types),
        "moe": fam.moe,
        "classes": dict(fam.classes),
        "prompt": {
            "format": {
                "tail": fmt.tail,
                "slot": fmt.slot,
                "variants": fmt.variants,
                "system_prompt": bool(fmt.system_prompt),
            },
            "label_form": fmt.label_form(),
            "noul_rendering": fam.noul_rendering,
            # the fingerprint a server on this base's defaults gives its task store (decisio.names.task_fingerprint)
            "fingerprint": json.loads(
                task_fingerprint(
                    served_name=fam.served_name,
                    model=os.path.basename(os.path.normpath(fam.model)),
                    pad_to=fam.pad_to,
                    pad_where="front",
                    hide_index_keys=True,
                    desnake_labels=True,
                    prompt_format=None if fmt.is_default() else fmt.facts(),
                    base=None if fam.key == QWEN else fam.key,
                )
            ),
        },
        "readout": {
            "softcap": fam.softcap,
            "head_rows": list(fam.head_rows),
            "head_label_logprobs": fam.head_label_logprobs,
            "hidden_start": fam.hidden_start,
        },
        "temperatures": {"global": fam.temperature, "choice": fam.choice_temperature},
        "serving": {
            "served_name": fam.served_name,
            "pad_to": fam.pad_to,
            "multi_question": fam.multi_question,
            "register_state_boundary": fam.register_state_boundary,
            "register_boundary": fam.register_boundary if fam.register_state_boundary else None,
        },
        "engine": {"vllm": SUPPORTED_VLLM},
    }


def export_decision_config(fam: Family, decisio_version: str | None = None, quantization: str | None = None) -> dict:
    """The file for a base: the repository's facts (`source`, `weights`) and the installed profile."""
    if decisio_version is None:
        from importlib.metadata import version

        decisio_version = version("decisio")
    return {
        "schema": SCHEMA,
        "decisio": decisio_version,
        "base": fam.key,
        "source": {"repository": fam.model, "revision": fam.revision, "license": LICENSE},
        "weights": weights_of(fam.key, quantization),
        **profile_of(fam),
    }


def differences(a, b, path="") -> list[str]:
    """The dotted paths at which two JSON values differ."""
    if isinstance(a, dict) and isinstance(b, dict):
        out = []
        for k in sorted(set(a) | set(b)):
            sub = f"{path}.{k}" if path else k
            if k not in a or k not in b:
                out.append(sub)
            else:
                out += differences(a[k], b[k], sub)
        return out
    return [] if a == b else [path]


def check_profile(config: dict, fam: Family) -> list[str]:
    """The profile keys at which a repository's file differs from the installed profile (empty: they agree)."""
    mine = profile_of(fam)
    return [d for k in PROFILE_KEYS for d in differences(config.get(k), mine[k], k)]


@dataclasses.dataclass(frozen=True)
class Repository:
    """A repository `--base` named: where its weights are, at which revision, and its file."""

    name: str  # owner/name, or the directory
    model: str  # what the engine loads: the repository id or the directory
    revision: str | None
    config: dict

    @property
    def base(self) -> str:
        return self.config["base"]

    def facts(self) -> dict:
        return {
            "name": self.name,
            "revision": self.revision,
            "exported_for_decisio": self.config.get("decisio"),
            "source": self.config.get("source"),
            "weights": self.config.get("weights"),
        }


def is_repository(value: str) -> bool:
    """Whether a `--base` value names a repository (owner/name[@revision], or a directory) rather than a base's key."""
    return value not in BASES and ("/" in value or os.path.isdir(value))


def open_repository(spec: str, revision: str | None = None) -> Repository:
    """Read a repository's file: `spec` is owner/name[@revision] or a directory; `revision` is --revision."""
    if os.path.isdir(spec):
        name, rev, path = spec, None, Path(spec) / FILE
    else:
        name, _, at = spec.partition("@")
        name = RENAMED.get(name, name)
        rev = at or revision or PINNED_REVISIONS.get(name)
        if not rev:
            from huggingface_hub import HfApi

            rev = HfApi().model_info(name).sha
            print(f"REPOSITORY {name}: no revision given, serving its current commit {rev}", flush=True)
        from huggingface_hub import hf_hub_download

        path = Path(hf_hub_download(name, FILE, revision=rev))
    try:
        config = json.loads(path.read_text())
    except (OSError, ValueError) as e:
        raise ValueError(f"{name} has no readable {FILE} ({e})") from e
    if not isinstance(config, dict) or config.get("schema") != SCHEMA:
        got = config.get("schema") if isinstance(config, dict) else None
        raise ValueError(f"{name}: {FILE} is schema {got!r}; this decisio reads {SCHEMA}")
    if config.get("base") not in BASES:
        raise ValueError(f"{name}: {FILE} names the base {config.get('base')!r}; this decisio serves {sorted(BASES)}")
    return Repository(name=name, model=name, revision=rev, config=config)


def serve_family(fam: Family, repo: Repository) -> Family:
    """The profile a repository is served under: the base's, except that a stored quantized checkpoint is not quantized
    again on load."""
    if fam.quantization and not repo.config.get("weights", {}).get("quantized_on_load", False):
        return dataclasses.replace(fam, quantization=None)
    return fam


def verify(repo: Repository, fam: Family) -> None:
    """Refuse a repository whose file differs from the installed profile, naming the keys and the release it is for."""
    from importlib.metadata import version

    diff = check_profile(repo.config, fam)
    if diff:
        raise ValueError(
            f"{repo.name} was exported for decisio {repo.config.get('decisio')}, and its {FILE} differs from the "
            f"{fam.key} profile of decisio {version('decisio')} at: {', '.join(diff)}. Serve a revision of the "
            "repository made for this decisio, or install the decisio it was made for"
        )


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m decisio.hub", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    ex = sub.add_parser("export", help="print or write the file generated from a base's profile")
    ex.add_argument("--base", choices=sorted(BASES))
    ex.add_argument("--all", action="store_true", help="every base (with --out)")
    ex.add_argument("--out", help="a directory: writes decision_config.<base>.json there")
    ex.add_argument("--decisio-version", help="the version to record (default: the installed one)")
    ex.add_argument("--quantization", help="the 31B's stored scheme, if not compressed-tensors (the identity gate)")
    ck = sub.add_parser("check", help="compare a file with the installed profile")
    ck.add_argument("file")
    a = ap.parse_args(argv)
    if a.cmd == "check":
        config = json.loads(Path(a.file).read_text())
        diff = check_profile(config, BASES[config["base"]])
        print("agrees with the installed profile" if not diff else "differs at: " + ", ".join(diff))
        return 1 if diff else 0
    bases = sorted(BASES) if a.all else [a.base]
    if not a.all and not a.base:
        ap.error("give --base or --all")
    for key in bases:
        text = json.dumps(export_decision_config(BASES[key], a.decisio_version, a.quantization), indent=2) + "\n"
        if a.out:
            Path(a.out).mkdir(parents=True, exist_ok=True)
            (Path(a.out) / f"decision_config.{key}.json").write_text(text)
        else:
            sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
