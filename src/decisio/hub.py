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

A repository of a **tuned** model (an adapter merged into the base: `python -m decisio.hub_fp8 merge` for the 31B and
gemma-4-12b-fp8, `python -m decisio.hub_bf16 merge` for gemma-4-12b) has schema
`decisio.decision_config/2`: the same sections, a `weights` section that says the tensors were modified by a merge, and
two more.

    tuned       where the weights come from, as facts: base (repository, revision), adapter (revision, weights_sha256,
                config_sha256, rank, alpha, and its repository when the file names one), merge (precision "fp32",
                output, tool, the adapter's pairs applied and the report's own fields), files (name -> sha256 of every
                file of the repository's weights) and, for a repository whose weights are an MLX conversion of the
                merged model, conversion (tool, tool_version, format "mlx", bits, group_size; `--backend mlx` serves it)
    defaults    what a plain request to this model gets, with no flag and no adapter field: temperatures (per
                question type), choice_bands (choice temperature by number of options, optional), noul_commit (true or
                false, always stated), fit (how the temperatures were fitted: the method, never the data) and
                noul_commit_basis (why the commit setting is what it is, with the figures) and over_limit
                ("refuse" or "truncate", optional: what a state over the engine's prompt ceiling gets) and
                over_limit_tokens (optional: the ceiling a truncating server cuts to) and think
                ({"factor", "budget", "types"}, optional: the model thinks when unsure, on the question types named,
                decisio.serve.think),
                think_temperatures (per question type, optional: the temperature of an answer that thought, which
                replaces the type's temperature for those answers only) and layout ("plain" or "header", optional: the
                prompt layout it was trained and measured on, which a server serves; none recorded means plain),
                read_twice_under and read_twice_max_questions (optional: a request read once in at most that many
                counted input tokens, and with at most that many questions, is read twice; decisio.readout.letters)
                and batch_invariant (optional, true only: a server of this repository runs vLLM's batch-invariant
                kernels, decisio.serve.batch_invariant)
    flavor      optional, with `defaults`: which flavor of the model the repository is (decisio.flavors): name
                ("calibrated", "commit", "read-enhanced" or, for a tuned model, "commit-think") and served_name
                (what the server is called, e.g. "Decisio-12b-tuned-commit"). The flavor must agree with `defaults`
                (calibrated: no commit rule and no read twice; commit: the commit rule; read-enhanced: read twice;
                commit-think: the commit rule and `defaults.think`), uses the header layout and, but for commit-think,
                does not think; the file has one flavor, and repositories of the same weights differ in this file only

A `/2` file may also have no `tuned`: the base's own weights with settings of their own (`tuned-config
--defaults-only`). Its `weights` must then be the base's section exactly, `source.files` (optional) lists the weight
files by sha256, and `defaults` (and a `flavor`) are as for a tuned model.

The profile sections of a `/2` file must still equal the installed base's profile (a repository cannot change how the
base is read); `defaults` is what the repository changes. A decisio that does not know `/2` refuses the file by its
schema (`open_repository` raises "is schema ...; this decisio reads ...") instead of serving the model at the base's
temperatures without saying. The temperature values are checked as an adapter file's are
(`decisio.serve.adapter_temperatures`).
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import re
import sys
from pathlib import Path

from decisio.families import BASES, Family
from decisio.names import task_fingerprint
from decisio.readout.letters import PromptFormat

SCHEMA = "decisio.decision_config/1"  # a base's repository: the profile, unchanged
SCHEMA_TUNED = "decisio.decision_config/2"  # a tuned model's: the profile plus `tuned` and `defaults`
SCHEMAS = (SCHEMA, SCHEMA_TUNED)
FILE = "decision_config.json"
# the bases a merge exists for: the 31B is stored in FP8 (`hub_fp8 merge`), the 12B in bf16 (`hub_bf16 merge`) as
# gemma-4-12b or in FP8 (`hub_fp8 merge`) as gemma-4-12b-fp8
TUNED_BASES = ("gemma-4-31b", "gemma-4-12b", "gemma-4-12b-fp8")
OVER_LIMIT_TOKENS_MIN, OVER_LIMIT_TOKENS_MAX = (
    1024,
    1_000_000,
)  # defaults.over_limit_tokens: the ceiling a state is cut to
LICENSE = "apache-2.0"  # Qwen3.6 and Gemma 4
QWEN = "qwen3.6-35b-a3b"

# the sections of the file that are the profile; the others describe the repository
PROFILE_KEYS = ("model_types", "moe", "classes", "prompt", "readout", "temperatures", "serving", "engine")

# the repository of each base. The 31B's weights are modified: its FP8 checkpoint is what the server would otherwise
# make on load. Its stored format is the one the identity gate keeps (compressed-tensors first, vLLM's own serialized
# FP8 if that fails; `python -m decisio.hub export --quantization`)
REPOSITORIES = {
    "gemma-4-31b": "tachara-ai/decisio-gemma-4-31b",
    "gemma-4-12b": "tachara-ai/decisio-gemma-4-12b",
    # the 12B's FP8 conversion (`python -m decisio.hub_fp8 convert`), a second precision beside the bf16 copy; served by
    # `--base tachara-ai/decisio-gemma-4-12b-fp8` once its revision is pinned (the bare key quantizes on load)
    "gemma-4-12b-fp8": "tachara-ai/decisio-gemma-4-12b-fp8",
    QWEN: "tachara-ai/decisio-qwen3.6-35b-a3b",
}
# the repositories were made under aminry/ for 0.10.0 and moved to tachara-ai/ (Tachara AI Lab) for 0.11.0; the Hub
# redirects the old names, and a server given one serves the repository under its new name, at the new pin
RENAMED = {f"aminry/{REPOSITORIES[k].split('/')[1]}": REPOSITORIES[k] for k in ("gemma-4-31b", "gemma-4-12b", QWEN)}
# the revision of each repository that a decisio release serves when none is given (--base owner/name@revision wins)
PINNED_REVISIONS: dict[str, str] = {
    "tachara-ai/decisio-gemma-4-31b": "eaf1e4fa30a351ab87d07803ca60f1fa5606f882",
    "tachara-ai/decisio-gemma-4-12b": "e857da2281e90719750a0a6f21707f7c07bbaa2d",
    "tachara-ai/decisio-qwen3.6-35b-a3b": "9187ed0012979321d25a054c897e8fe48e626dfc",
    # uploaded private 2026-10-21 after the card checks (identity of the engine's FP8 tensors and a served read, on an
    # RTX 5090), every file read back by sha256; made for 0.24.0
    "tachara-ai/decisio-gemma-4-12b-fp8": "eed25b6fab0d4903392536230c8837b6a0dcfa77",
}
# the bases whose own key is served from a decisio repository rather than from the source's weights, once that
# repository's revision is pinned: the 31B's FP8 checkpoint saves quantizing Google's bf16 weights at every start
# (2026-10-07; the others are byte-for-byte copies and keep their sources). `--model google/gemma-4-31B-it`
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
                "quantized to FP8 by decisio's export (per-tensor scales stored beside the weights), the same bytes as "
                "the tensors vLLM 0.30.0 and 0.31.0 hold after quantizing these weights on load (compared 2026-10-15); "
                "on 0.31.0 the answers of the two differ by restart-sized amounts through the compiled path and are "
                "identical with eager execution (see the card); activations are quantized per token at run "
                "time; every other tensor, the vision tower included, is the source's; nothing was trained"
            ),
            "quantized_on_load": False,
        }
    if base == "gemma-4-12b-fp8":
        return {
            "modified": True,
            "dtype": "fp8_e4m3",
            "quantization": quantization or "compressed-tensors",
            "method": (
                "the 12B's bf16 weights quantized to FP8 by decisio's export (per-tensor scales stored beside the "
                "weights), predicted on a CPU to be the same bytes as the tensors vLLM 0.31.0 holds after quantizing "
                "them on load (192 fused groups, the loader changing 0 weights); activations are quantized per token "
                "at run time; every other tensor, the embeddings and the vision embedder included, is the source's; "
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


MLX_DTYPE = "mlx_affine"
MLX_BITS = (4, 6, 8)
MLX_GROUP_SIZES = (32, 64, 128)
MLX_BASES = ("gemma-4-31b", "gemma-4-12b")  # the bases an MLX conversion of a tuned model is made for


def check_conversion(conversion, base_key: str | None) -> dict:
    """`tuned.conversion`: the MLX conversion a repository's weights are (`tuned-config --mlx-conversion`), made from
    the tuned model's merged weights by another tool. Raises ValueError naming what is wrong."""
    _need(conversion, "tuned.conversion")
    unknown = sorted(set(conversion) - {"tool", "tool_version", "format", "bits", "group_size"})
    if unknown:
        raise ValueError(f"tuned.conversion has unknown keys {unknown}")
    if base_key is not None and base_key not in MLX_BASES:
        raise ValueError(f"an MLX conversion of a tuned model is made for {', '.join(MLX_BASES)}, not {base_key}")
    for key in ("tool", "tool_version"):
        if not isinstance(conversion.get(key), str) or not NAME_RE.match(conversion[key]):
            raise ValueError(
                f"tuned.conversion.{key} must be a name or version (mlx-lm, 0.32.0), got {conversion.get(key)!r}"
            )
    if conversion.get("format") != "mlx":
        raise ValueError(f"tuned.conversion.format must be 'mlx', got {conversion.get('format')!r}")
    if conversion.get("bits") not in MLX_BITS:
        raise ValueError(
            f"tuned.conversion.bits must be one of {', '.join(map(str, MLX_BITS))}, got {conversion.get('bits')!r}"
        )
    if conversion.get("group_size") not in MLX_GROUP_SIZES:
        sizes = ", ".join(map(str, MLX_GROUP_SIZES))
        raise ValueError(f"tuned.conversion.group_size must be one of {sizes}, got {conversion.get('group_size')!r}")
    return conversion


def weights_of_tuned(
    base: str, quantization: str | None = None, external: dict | None = None, conversion: dict | None = None
) -> dict:
    """The `weights` section of a tuned model's repository: the base's checkpoint with an adapter merged in (FP8 for
    the 31B and for gemma-4-12b-fp8, bf16 for gemma-4-12b). `external`: the `tuned.merge` record of a checkpoint made by
    another tool (`external_facts`), whose method is then described from that record and not as decisio's merge."""
    if base not in TUNED_BASES:
        raise ValueError(f"a tuned repository is made for {', '.join(TUNED_BASES)}, not {base}")
    if conversion:
        return {
            "modified": True,
            "dtype": MLX_DTYPE,
            "quantization": f"mlx-{conversion['bits']}bit",
            "bits": conversion["bits"],
            "group_size": conversion["group_size"],
            "method": (
                f"the tuned model's merged weights converted to MLX ({conversion['bits']}-bit affine quantization, "
                f"group size {conversion['group_size']}) by {conversion['tool']} {conversion['tool_version']}, not by "
                "decisio; "
                "decisio's `--backend mlx` serves it; the adapter, its revision, the merge and the converted files are "
                "listed in `tuned`"
            ),
            "quantized_on_load": False,
        }
    if external:
        return {
            "modified": True,
            "dtype": "fp8_e4m3",
            "quantization": quantization or "compressed-tensors",
            "method": (
                f"a LoRA adapter merged into the base (the merged weights were held in {external['precision']} when "
                f"quantized) and the checkpoint quantized to FP8 "
                f"({external['scheme']}: weights {external['weights']}, activations {external['activations']}) by "
                f"{external['tool']} {external['tool_version']}, not by decisio's merge; "
                + ("the modules left in bf16 are listed in `tuned.merge`; " if external.get("bf16_modules") else "")
                + "the adapter, its revision and the merged files are listed in `tuned`"
            ),
            "quantized_on_load": False,
        }
    if base == "gemma-4-12b":
        return {
            "modified": True,
            "dtype": "bfloat16",
            "quantization": None,
            "method": (
                "a LoRA adapter merged into the bf16 base in fp32 and rounded once to bf16, stochastically and seeded, "
                "by decisio's merge (`python -m decisio.hub_bf16 merge`); every other tensor is the base's, byte for "
                "byte; the adapter, its revision, the rounding seed and the merged files are listed in `tuned`"
            ),
            "quantized_on_load": False,
        }
    if base == "gemma-4-12b-fp8":
        return {
            "modified": True,
            "dtype": "fp8_e4m3",
            "quantization": quantization or "compressed-tensors",
            "method": (
                "a LoRA adapter merged into the 12B's bf16 base in fp32 and quantized once to FP8 by decisio's merge "
                "(`python -m decisio.hub_fp8 merge`), as the 31B's; every other tensor is the 12B's FP8 export; the "
                "adapter, its revision and the merged files are listed in `tuned`"
            ),
            "quantized_on_load": False,
        }
    return {
        "modified": True,
        "dtype": "fp8_e4m3",
        "quantization": quantization or "compressed-tensors",
        "method": (
            "a LoRA adapter merged into the bf16 base in fp32 and quantized once to FP8 by decisio's merge "
            "(`python -m decisio.hub_fp8 merge`); every other tensor is the base's FP8 export; the adapter, its "
            "revision and the merged files are listed in `tuned`"
        ),
        "quantized_on_load": False,
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

    @property
    def tuned(self) -> bool:
        """Whether this is a tuned model's repository (schema `/2`: it carries `tuned` and `defaults`)."""
        return self.config.get("schema") == SCHEMA_TUNED

    @property
    def mlx(self) -> bool:
        """Whether the repository's weights are an MLX conversion (`--backend mlx` serves them, vLLM does not)."""
        return (self.config.get("weights") or {}).get("dtype") == MLX_DTYPE

    @property
    def defaults(self) -> dict | None:
        """What a plain request to the model gets (`defaults` of a `/2` file), else None."""
        return self.config.get("defaults") if self.tuned else None

    @property
    def flavor(self) -> dict | None:
        """Which flavor of the model this is (`flavor` of a `/2` file: name and served_name), else None."""
        return self.config.get("flavor") if self.tuned else None

    @property
    def defaults_sha256(self) -> str | None:
        return defaults_sha256(self.config) if self.tuned else None

    def facts(self) -> dict:
        out = {
            "name": self.name,
            "revision": self.revision,
            "schema": self.config.get("schema"),
            "exported_for_decisio": self.config.get("decisio"),
            "source": self.config.get("source"),
            "weights": self.config.get("weights"),
        }
        if self.tuned:
            out |= {
                **({"tuned": self.config["tuned"]} if "tuned" in self.config else {}),
                "defaults": self.config["defaults"],
                "defaults_sha256": self.defaults_sha256,
                **({"flavor": self.flavor} if self.flavor else {}),
            }
        return out


HEX64 = re.compile(r"^[0-9a-f]{64}$")


def defaults_sha256(config: dict) -> str:
    """The digest of what a tuned repository changes: its `tuned` and `defaults` sections, in canonical JSON. It is in
    /health and in the serving-stack fingerprint, so another adapter, another temperature or another commit setting is
    another stack."""
    parts = {"tuned": config.get("tuned"), "defaults": config.get("defaults")}
    if config.get("flavor") is not None:  # only a flavor repository: the digest of every other one is what it was
        parts["flavor"] = config["flavor"]
    body = json.dumps(parts, sort_keys=True)
    return hashlib.sha256(body.encode()).hexdigest()


def check_sections(config: dict) -> None:
    """Refuse a file whose sections do not fit its schema: a `/1` file carries neither `tuned` nor `defaults`, a `/2`
    file carries both, well formed. Raises ValueError naming the path."""
    tuned_schema = config.get("schema") == SCHEMA_TUNED
    if not tuned_schema:
        extra = [k for k in ("tuned", "defaults", "flavor") if k in config]
        if extra:
            raise ValueError(
                f"{FILE} is schema {config.get('schema')!r} but has {' and '.join(extra)}: that is {SCHEMA_TUNED}"
            )
        return
    if config.get("base") not in TUNED_BASES:
        raise ValueError(
            f"{FILE} (tuned) names the base {config.get('base')!r}; "
            f"tuned repositories are made for {', '.join(TUNED_BASES)}"
        )
    if "tuned" not in config:
        _check_defaults_only(config)
        return
    if not isinstance(config.get("weights"), dict) or config["weights"].get("modified") is not True:
        raise ValueError("weights.modified must be true in a tuned repository")
    conversion = (config.get("tuned") or {}).get("conversion") if isinstance(config.get("tuned"), dict) else None
    if conversion is not None:
        check_conversion(conversion, config["base"])
    expected = weights_of_tuned(config["base"], config["weights"].get("quantization") or None, None, conversion)
    if config["weights"] != expected and conversion is not None:
        raise ValueError(
            "weights must be the MLX conversion's section (`tuned-config --mlx-conversion` writes it): "
            f"differs at {', '.join(differences(config['weights'], expected, 'weights'))}"
        )
    if config["weights"].get("dtype") != expected["dtype"]:
        raise ValueError(
            f"weights.dtype is {config['weights'].get('dtype')!r}; "
            f"a tuned {config['base']} is stored as {expected['dtype']}"
        )
    check_tuned(config.get("tuned"), config["base"])
    check_defaults(config.get("defaults"))
    if "flavor" in config:
        check_flavor(config["flavor"], config["defaults"])


def _check_defaults_only(config: dict) -> None:
    """A `/2` file without `tuned`: the base's own weights, served with settings of their own (`defaults`, and
    optionally a `flavor`). Nothing was merged, so `weights` must be the base's own section, exactly as a `/1` file of
    that base carries it; `source.files` (optional) lists the weight files by sha256 for `check --weights-dir`."""
    expected = weights_of(
        config["base"], config["weights"].get("quantization") if isinstance(config.get("weights"), dict) else None
    )
    if config.get("weights") != expected:
        raise ValueError(
            "a /2 file without `tuned` serves the base's own weights: weights must be the base's section, differs at "
            + ", ".join(differences(config.get("weights"), expected, "weights"))
        )
    source = _need(config.get("source"), "source")
    for k in ("repository", "revision"):
        _need(source.get(k), f"source.{k}", str)
    if "files" in source:
        files = _need(source["files"], "source.files")
        if not files:
            raise ValueError("source.files must list the weight files, or be left out")
        for name, sha in files.items():
            _sha(sha, f"source.files[{name!r}]")
    check_defaults(config.get("defaults"))
    if "flavor" in config:
        check_flavor(config["flavor"], config["defaults"])


def _need(obj, path: str, kind=dict):
    if not isinstance(obj, kind):
        raise ValueError(
            f"{path} must be {'an object' if kind is dict else 'a string' if kind is str else kind.__name__}"
        )
    return obj


def _sha(value, path: str) -> str:
    if not isinstance(value, str) or not HEX64.match(value):
        raise ValueError(f"{path} must be a sha256 (64 lowercase hex digits), got {value!r}")
    return value


def _check_external(merge: dict, adapter: dict, base_key: str | None) -> None:
    """`tuned.merge` of a checkpoint made by another tool (`external_facts`): the fields it states, none of decisio's
    own (the fp32 merge, its counts), whose checks are about decisio's tools."""
    if base_key not in (None, "gemma-4-31b", "gemma-4-12b-fp8"):
        raise ValueError(f"tuned.merge is external, but {base_key} is not an FP8 base")
    rebuilt = external_facts(
        f"{merge.get('tool')}@{merge.get('tool_version')}",
        {
            k: merge.get(k)
            for k in ("adapter_sha256", "adapter_scale", "scheme", "weights", "activations", "bf16_modules")
            if merge.get(k) is not None
        }
        | {"merge_precision": merge.get("precision")},
        base_key or "gemma-4-31b",
    )
    if rebuilt != merge:
        raise ValueError("tuned.merge (external) carries fields beyond the ones an external form records")
    if merge["adapter_sha256"] != adapter["weights_sha256"]:
        raise ValueError(
            "tuned.merge.adapter_sha256 is not tuned.adapter.weights_sha256: the merge report is of another adapter"
        )


def check_tuned(tuned, base_key: str | None = None) -> None:
    _need(tuned, "tuned")
    base = _need(tuned.get("base"), "tuned.base")
    for k in ("repository", "revision"):
        _need(base.get(k), f"tuned.base.{k}", str)
    adapter = _need(tuned.get("adapter"), "tuned.adapter")
    _need(adapter.get("revision"), "tuned.adapter.revision", str)
    if "repository" in adapter:  # optional: a repository that does not name where the adapter lives keeps it out
        if not _need(adapter["repository"], "tuned.adapter.repository", str):
            raise ValueError("tuned.adapter.repository must be a name, or left out")
    _sha(adapter.get("weights_sha256"), "tuned.adapter.weights_sha256")
    if adapter.get("config_sha256") is not None:
        _sha(adapter["config_sha256"], "tuned.adapter.config_sha256")
    merge = _need(tuned.get("merge"), "tuned.merge")
    if merge.get("external") is True:
        _check_external(merge, adapter, base_key)
        files = _need(tuned.get("files"), "tuned.files")
        if not files:
            raise ValueError("tuned.files must list the repository's weight files")
        for name, sha in files.items():
            _sha(sha, f"tuned.files[{name!r}]")
        return
    if merge.get("precision") != "fp32":
        raise ValueError(
            "tuned.merge.precision must be 'fp32' (the merge is made in fp32 and quantized once), "
            f"got {merge.get('precision')!r}"
        )
    output = str(merge.get("output", ""))
    if base_key == "gemma-4-12b":
        if output.startswith("fp8"):
            raise ValueError(
                f"tuned.merge.output is {merge.get('output')!r}, an FP8 merge (`hub_fp8 merge`): a tuned 12B in FP8 is "
                "the base gemma-4-12b-fp8, not gemma-4-12b (bf16)"
            )
        if merge.get("output") != "bf16" or merge.get("rounding") != "stochastic":
            raise ValueError(
                "tuned.merge of a 12B must say output 'bf16' and rounding 'stochastic' (the merge is rounded once, "
                f"unbiased), got {merge.get('output')!r} and {merge.get('rounding')!r}"
            )
        if not isinstance(merge.get("seed"), int) or isinstance(merge["seed"], bool):
            raise ValueError("tuned.merge.seed must be the rounding's seed (a whole number)")
    elif base_key == "gemma-4-12b-fp8":
        if not output.startswith("fp8") or merge.get("rounding") is not None or "seed" in merge:
            raise ValueError(
                "tuned.merge of gemma-4-12b-fp8 must say output 'fp8' (`hub_fp8 merge`: quantized once, no stochastic "
                f"rounding or seed), got output {merge.get('output')!r} and rounding {merge.get('rounding')!r}: a bf16 "
                "merge of the 12B (`hub_bf16 merge`) is the base gemma-4-12b"
            )
    elif base_key == "gemma-4-31b" and not output.startswith("fp8"):
        raise ValueError(f"tuned.merge.output of a 31B must be fp8, got {merge.get('output')!r}")
    if not isinstance(merge.get("adapted_tensors"), int) or merge["adapted_tensors"] < 1:
        raise ValueError(
            "tuned.merge.adapted_tensors must be the number of adapter tensors the merge applied (at least 1)"
        )
    dropped = merge.get("dropped_k_eq_v_pairs", 0)
    if not isinstance(dropped, int) or isinstance(dropped, bool) or dropped < 0:
        raise ValueError(
            "tuned.merge.dropped_k_eq_v_pairs must be how many adapter pairs the merge left out (0 or more)"
        )
    if merge.get("adapter_sha256") not in (None, adapter["weights_sha256"]):
        raise ValueError(
            "tuned.merge.adapter_sha256 is not tuned.adapter.weights_sha256: the merge report is of another adapter"
        )
    files = _need(tuned.get("files"), "tuned.files")
    if not files:
        raise ValueError("tuned.files must list the repository's weight files")
    for name, sha in files.items():
        _sha(sha, f"tuned.files[{name!r}]")


def check_defaults(defaults) -> None:
    from decisio.serve.adapter_temperatures import validate_temperatures

    _need(defaults, "defaults")
    allowed = {
        "temperatures",
        "choice_bands",
        "noul_commit",
        "fit",
        "noul_commit_basis",
        "over_limit",
        "over_limit_tokens",
        "think",
        "layout",
        "think_temperatures",
        "read_twice_under",
        "read_twice_max_questions",
        "batch_invariant",
    }
    unknown = sorted(set(defaults) - allowed)
    if unknown:
        raise ValueError(f"defaults has unknown keys {unknown}")
    validate_temperatures(defaults.get("temperatures"), defaults.get("choice_bands"), "defaults")
    if not isinstance(defaults.get("noul_commit"), bool):
        raise ValueError("defaults.noul_commit must be true or false, stated (a tuned repository always says which)")
    _need(defaults.get("fit"), "defaults.fit")
    _need(
        defaults["fit"].get("method"), "defaults.fit.method", str
    )  # how the temperatures were fitted; nothing about data
    if "think_temperatures" in defaults:
        from decisio.serve.adapter_temperatures import validate_think_temperatures

        if not defaults["think_temperatures"]:
            raise ValueError("defaults.think_temperatures must be non-empty when given (leave it out for none)")
        validate_think_temperatures(defaults["think_temperatures"], "defaults")
    if "think" in defaults:
        from decisio.serve.think import check_settings

        check_settings(defaults["think"], "defaults.think")
    if "layout" in defaults and defaults["layout"] not in ("plain", "header"):
        raise ValueError('defaults.layout must be "plain" or "header"')
    if "over_limit" in defaults and defaults["over_limit"] not in ("refuse", "truncate"):
        raise ValueError('defaults.over_limit must be "refuse" or "truncate"')
    if "over_limit_tokens" in defaults:
        n = defaults["over_limit_tokens"]
        if isinstance(n, bool) or not isinstance(n, int) or not OVER_LIMIT_TOKENS_MIN <= n <= OVER_LIMIT_TOKENS_MAX:
            raise ValueError(
                f"defaults.over_limit_tokens must be a whole number of tokens from {OVER_LIMIT_TOKENS_MIN:,} to "
                f"{OVER_LIMIT_TOKENS_MAX:,}, got {n!r}"
            )
    if defaults["noul_commit"] and not isinstance(defaults.get("noul_commit_basis"), (dict, str)):
        raise ValueError("defaults.noul_commit_basis must say why the commit setting is on (with the figures)")
    if "batch_invariant" in defaults and defaults["batch_invariant"] is not True:
        raise ValueError("defaults.batch_invariant must be true (leave it out for the server's own setting)")
    for key in ("read_twice_under", "read_twice_max_questions"):
        v = defaults.get(key)
        if key in defaults and (isinstance(v, bool) or not isinstance(v, int) or v < 0):
            raise ValueError(f"defaults.{key} must be 0 or a number")
    if defaults.get("read_twice_max_questions") and not defaults.get("read_twice_under"):
        raise ValueError("defaults.read_twice_max_questions limits defaults.read_twice_under, which is off")


def check_flavor(flavor, defaults: dict) -> None:
    """A tuned repository's `flavor` block: its name and served name, and that `defaults` are that flavor's."""
    from decisio.flavors import TUNED_FLAVOR_NAMES as FLAVOR_NAMES

    _need(flavor, "flavor")
    unknown = sorted(set(flavor) - {"name", "served_name"})
    if unknown:
        raise ValueError(f"flavor has unknown keys {unknown}")
    name, served = flavor.get("name"), flavor.get("served_name")
    if name not in FLAVOR_NAMES:
        raise ValueError(f"flavor.name must be one of {', '.join(FLAVOR_NAMES)}, got {name!r}")
    if (
        not isinstance(served, str)
        or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", served)
        or not served.endswith(f"-{name}")
    ):
        raise ValueError(f"flavor.served_name must be a name ending in -{name}, got {served!r}")
    if defaults.get("layout") != "header":
        raise ValueError('a flavor uses the header layout: defaults.layout must be "header"')
    if name == "commit-think":
        if not defaults.get("think"):
            raise ValueError("the commit-think flavor thinks: defaults.think (factor, budget, types) must be given")
    elif defaults.get("think"):
        raise ValueError("a flavor does not think: leave defaults.think out")
    commit, twice = bool(defaults["noul_commit"]), bool(defaults.get("read_twice_under"))
    want = {
        "calibrated": (False, False),
        "commit": (True, False),
        "read-enhanced": (False, True),
        "commit-think": (True, False),
    }[name]
    if (commit, twice) != want:
        raise ValueError(
            f"flavor {name}: defaults.noul_commit is {commit} and read twice is {'on' if twice else 'off'}; "
            f"it needs {want[0]} and {'on' if want[1] else 'off'}"
        )


def file_digests(directory: Path) -> dict:
    """name -> sha256 of the weight files, the config and the tokenizer files of a repository's directory."""
    out = {}
    for f in sorted(Path(directory).iterdir()):
        if f.is_file() and (f.suffix in (".safetensors", ".json", ".model", ".jinja") and f.name != FILE):
            h = hashlib.sha256()
            with open(f, "rb") as fh:
                for chunk in iter(lambda: fh.read(16 << 20), b""):
                    h.update(chunk)
            out[f.name] = h.hexdigest()
    return out


def check_weights_precision(base: str, directory: Path) -> None:
    """Refuse a weights directory whose config.json declares another precision than the base's, where one checkpoint
    is served in two (the 12B): a checkpoint that declares FP8 weights is a gemma-4-12b-fp8 one, any other a
    gemma-4-12b one. Other bases have one precision, and their checkpoints are not checked here."""
    from decisio.families import declares_fp8

    fam = BASES.get(base)
    if fam is None or fam.precision is None:
        return
    path = Path(directory) / "config.json"
    fp8 = declares_fp8(json.loads(path.read_text())) if path.exists() else False
    if fp8 == (fam.precision == "fp8"):
        return
    other = next(f for f in BASES.values() if f.model == fam.model and f.precision == ("fp8" if fp8 else "bf16"))
    raise ValueError(
        f"{directory}: its config.json declares {'FP8' if fp8 else 'no FP8'} weights, so it is a {other.key} "
        f"checkpoint, not {base}"
    )


def mlx_conversion_of(form: str, bits: int | None, group_size: int | None, directory: Path, base: str) -> dict:
    """The `tuned.conversion` record of an MLX conversion (`tuned-config --mlx-conversion TOOL@VERSION`), checked
    against the conversion's own config.json: it must declare the same affine quantization."""
    tool, sep, version = str(form).partition("@")
    if not sep:
        raise ValueError(f"--mlx-conversion {form!r}: expected TOOL@VERSION, for example mlx-lm@0.32.0")
    if bits is None or group_size is None:
        raise ValueError("--mlx-conversion needs --mlx-bits and --mlx-group-size")
    conversion = check_conversion(
        {"tool": tool, "tool_version": version, "format": "mlx", "bits": bits, "group_size": group_size}, base
    )
    path = Path(directory) / "config.json"
    declared = (json.loads(path.read_text()).get("quantization") or {}) if path.exists() else {}
    if (declared.get("bits"), declared.get("group_size"), declared.get("mode", "affine")) != (
        bits,
        group_size,
        "affine",
    ):
        raise ValueError(
            f"{path}: its quantization is {declared or 'absent'}, not {bits}-bit affine with group size {group_size}"
        )
    return conversion


def merge_facts(report: dict) -> dict:
    """The `tuned.merge` section from a merge's report (`MERGE_FP8.json` of `hub_fp8 merge`, `MERGE_BF16.json` of
    `hub_bf16 merge`)."""
    if report.get("null"):
        raise ValueError("this merge report is of a null merge (an adapter of scale 0), not of a shipped model")
    if report.get("rounding") == "stochastic":
        return {
            "precision": "fp32",
            "output": "bf16",
            "rounding": "stochastic",
            "seed": report.get("seed"),
            "tool": "decisio.hub_bf16 merge",
            "adapter_sha256": report.get("adapter_sha256"),
            "adapter_scale": report.get("adapter_scale"),
            "adapted_tensors": report.get("adapted_tensors"),
            "dropped_k_eq_v_pairs": report.get("dropped_k_eq_v_pairs", 0),
            "weight_space_effect_kept": report.get("weight_space_effect_kept"),
        }
    return {
        "precision": "fp32",
        "output": "fp8 compressed-tensors",
        "tool": "decisio.hub_fp8 merge",
        "adapter_sha256": report.get("adapter_sha256"),
        "adapter_scale": report.get("adapter_scale"),
        "adapted_tensors": report.get("adapted_tensors"),
        "dropped_k_eq_v_pairs": report.get("dropped_k_eq_v_pairs", 0),
        "groups": report.get("groups"),
        # projections left in bf16 (`hub_fp8 merge --keep-bf16`): which, and the rounding of those tensors
        **({"kept_bf16": report["kept_bf16"]} if report.get("kept_bf16") else {}),
        # error-compensating rounding (`hub_fp8 merge --gptq`): the method and its settings and the calibration file's
        # digest; not the request or token counts, which are details of the calibration data
        **(
            {
                "gptq": {
                    k: report["gptq"][k] for k in ("method", "damp", "blocksize", "max_tokens", "calibration_sha256")
                }
            }
            if report.get("gptq")
            else {}
        ),
    }


EXTERNAL_KEYS = {
    "adapter_sha256",
    "adapter_scale",
    "merge_precision",
    "scheme",
    "weights",
    "activations",
    "bf16_modules",
}
EXTERNAL_WEIGHTS = ("per-channel", "per-tensor", "per-group")
EXTERNAL_ACTIVATIONS = ("per-token dynamic", "per-tensor dynamic", "per-tensor static")
EXTERNAL_PRECISIONS = ("fp32", "bf16", "fp16")
NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def external_facts(form: str, report: dict, base: str) -> dict:
    """The `tuned.merge` section for a checkpoint made by another tool (`tuned-config --external-form TOOL@VERSION`,
    for example llm-compressor@0.8.1). `report` is a hand-written JSON:
    adapter_sha256 and adapter_scale (the adapter's);
    merge_precision: the precision the merged weights held when the tool received them, that is the stored intermediate
    it quantized (bf16 for a merge written to bf16 first), not the arithmetic of the merge; fp32, bf16 or fp16;
    scheme: the tool's name for the quantization, for example FP8_DYNAMIC;
    weights (per-channel, per-tensor or per-group) and activations (per-token dynamic, per-tensor dynamic or per-tensor
    static); and, optionally, bf16_modules (the modules left unquantized).
    Nothing else is accepted: the record says how the checkpoint was made, never what it was trained or calibrated on.
    Raises ValueError naming what is wrong."""
    if base not in ("gemma-4-31b", "gemma-4-12b-fp8"):
        raise ValueError(f"--external-form records an FP8 checkpoint made by another tool; {base} is not an FP8 base")
    tool, sep, version = str(form).partition("@")
    if not sep or not NAME_RE.match(tool) or not NAME_RE.match(version):
        raise ValueError(f"--external-form {form!r}: expected TOOL@VERSION, for example llm-compressor@0.8.1")
    if tool.startswith("decisio"):
        raise ValueError(
            "--external-form is for another tool's checkpoints; decisio's own are recorded from its reports"
        )
    extra = sorted(set(report) - EXTERNAL_KEYS)
    if extra:
        raise ValueError(f"the external merge report has unknown keys {extra}; it takes {sorted(EXTERNAL_KEYS)}")
    _sha(report.get("adapter_sha256"), "report.adapter_sha256")
    scale = report.get("adapter_scale")
    if isinstance(scale, bool) or not isinstance(scale, (int, float)) or not scale > 0:
        raise ValueError("report.adapter_scale must be a positive number (a merge of scale 0 is not a shipped model)")
    for key, allowed in (
        ("merge_precision", EXTERNAL_PRECISIONS),
        ("weights", EXTERNAL_WEIGHTS),
        ("activations", EXTERNAL_ACTIVATIONS),
    ):
        if report.get(key) not in allowed:
            raise ValueError(f"report.{key} must be one of {', '.join(allowed)}, got {report.get(key)!r}")
    scheme = report.get("scheme")
    if not isinstance(scheme, str) or not NAME_RE.match(scheme):
        raise ValueError(f"report.scheme must be the tool's name for the scheme (FP8_DYNAMIC), got {scheme!r}")
    modules = report.get("bf16_modules")
    if modules is not None and (not isinstance(modules, list) or not all(isinstance(m, str) and m for m in modules)):
        raise ValueError("report.bf16_modules must be a list of module names or patterns")
    return {
        "precision": report["merge_precision"],
        "output": "fp8 compressed-tensors",
        "tool": tool,
        "tool_version": version,
        "external": True,
        "scheme": scheme,
        "weights": report["weights"],
        "activations": report["activations"],
        "adapter_sha256": report["adapter_sha256"],
        "adapter_scale": float(scale),
        **({"bf16_modules": modules} if modules else {}),
    }


def adapter_facts(directory: Path, repository: str | None, revision: str) -> dict:
    """The `tuned.adapter` section from a PEFT adapter directory (`repository` None: the name is left out)."""
    d = Path(directory)
    cfg = json.loads((d / "adapter_config.json").read_text())
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()  # noqa: E731
    return {
        **({"repository": repository} if repository else {}),
        "revision": revision,
        "weights_sha256": sha(d / "adapter_model.safetensors"),
        "config_sha256": sha(d / "adapter_config.json"),
        "rank": cfg.get("r"),
        "alpha": cfg.get("lora_alpha"),
    }


# what `defaults.fit` may say: how the temperatures were fitted. Nothing about the data (no source, mix or size): a
# tuned repository's card carries one generic data statement, so the fit tools' `data`, `n` and `detail` stay out
FIT_KEYS = ("method", "objective", "read_at", "kappa_on_raw_readout", "temperature_file_sha256", "think")
THINK_FIT_KEYS = ("objective", "read_at", "kappa_on_raw_readout")


def _defaults_section(
    *,
    temperatures: dict,
    choice_bands: dict | None,
    noul_commit: bool,
    noul_commit_basis: dict | str | None,
    fit: dict,
    over_limit: str | None,
    over_limit_tokens: int | None,
    layout: str | None,
    think: dict | None,
    think_temperatures: dict | None,
    read_twice_under: int | None,
    read_twice_max_questions: int | None,
    batch_invariant: bool,
) -> dict:
    """The `defaults` section of a `/2` file: what a plain request to the model gets."""
    fit = {k: v for k, v in fit.items() if k in FIT_KEYS}
    if isinstance(fit.get("think"), dict):  # the think fit's method and readout scale; its counts and data stay out
        fit["think"] = {k: v for k, v in fit["think"].items() if k in THINK_FIT_KEYS}
    return {
        "temperatures": temperatures,
        **({"choice_bands": choice_bands} if choice_bands else {}),
        "noul_commit": noul_commit,
        "fit": fit,
        **({"noul_commit_basis": noul_commit_basis} if noul_commit_basis is not None else {}),
        **({"over_limit": over_limit} if over_limit is not None else {}),
        **({"over_limit_tokens": over_limit_tokens} if over_limit_tokens is not None else {}),
        **({"layout": layout} if layout is not None else {}),
        **({"think": think} if think is not None else {}),
        **({"think_temperatures": think_temperatures} if think_temperatures else {}),
        **({"read_twice_under": read_twice_under} if read_twice_under else {}),
        **({"read_twice_max_questions": read_twice_max_questions} if read_twice_max_questions else {}),
        **({"batch_invariant": True} if batch_invariant else {}),
    }


def defaults_config(
    fam: Family,
    *,
    files: dict | None = None,
    decisio_version: str | None = None,
    quantization: str | None = None,
    flavor: dict | None = None,
    **settings,
) -> dict:
    """The `/2` file of a repository of the base's own weights with settings of their own: the base's `/1` file
    (profile, source, weights) plus `defaults` (and a `flavor`), no `tuned`. `files`: the weight files' sha256, kept
    in `source.files`. `settings` are those of `tuned_config` (temperatures, choice_bands, noul_commit, ...). Validated
    before it is returned."""
    if fam.key not in TUNED_BASES:
        raise ValueError(f"a repository with settings is made for {', '.join(TUNED_BASES)}, not {fam.key}")
    config = export_decision_config(fam, decisio_version, quantization)
    config["schema"] = SCHEMA_TUNED
    if files:
        config["source"] = {**config["source"], "files": files}
    config["defaults"] = _defaults_section(**_settings(settings))
    if flavor is not None:
        config["flavor"] = flavor
    check_sections(config)
    return config


def _settings(given: dict) -> dict:
    """The keyword arguments of `_defaults_section`, with the optional ones defaulted."""
    out = {
        "choice_bands": None,
        "noul_commit_basis": None,
        "over_limit": None,
        "over_limit_tokens": None,
        "layout": None,
        "think": None,
        "think_temperatures": None,
        "read_twice_under": None,
        "read_twice_max_questions": None,
        "batch_invariant": False,
    }
    unknown = sorted(set(given) - set(out) - {"temperatures", "noul_commit", "fit"})
    if unknown:
        raise TypeError(f"defaults_config got unexpected settings {unknown}")
    out.update(given)
    return out


def tuned_config(
    fam: Family,
    *,
    temperatures: dict,
    choice_bands: dict | None,
    noul_commit: bool,
    noul_commit_basis: dict | str | None,
    fit: dict,
    adapter: dict,
    merge: dict,
    files: dict,
    decisio_version: str | None = None,
    quantization: str | None = None,
    over_limit: str | None = None,
    over_limit_tokens: int | None = None,
    layout: str | None = None,
    think: dict | None = None,
    think_temperatures: dict | None = None,
    flavor: dict | None = None,
    read_twice_under: int | None = None,
    read_twice_max_questions: int | None = None,
    batch_invariant: bool = False,
    conversion: dict | None = None,
) -> dict:
    """The `/2` file of a tuned model: the base's profile, the weights section of a merge, `tuned` and `defaults`.
    Validated before it is returned."""
    config = export_decision_config(fam, decisio_version, quantization)
    config["schema"] = SCHEMA_TUNED
    config["weights"] = weights_of_tuned(fam.key, quantization, merge if merge.get("external") else None, conversion)
    config["tuned"] = {
        "base": {"repository": fam.model, "revision": fam.revision},
        "adapter": adapter,
        "merge": merge,
        "files": files,
        **({"conversion": conversion} if conversion else {}),
    }
    config["defaults"] = _defaults_section(
        temperatures=temperatures,
        choice_bands=choice_bands,
        noul_commit=noul_commit,
        noul_commit_basis=noul_commit_basis,
        fit=fit,
        over_limit=over_limit,
        over_limit_tokens=over_limit_tokens,
        layout=layout,
        think=think,
        think_temperatures=think_temperatures,
        read_twice_under=read_twice_under,
        read_twice_max_questions=read_twice_max_questions,
        batch_invariant=batch_invariant,
    )
    if flavor is not None:
        config["flavor"] = flavor
    check_sections(config)
    return config


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
    if not isinstance(config, dict) or config.get("schema") not in SCHEMAS:
        got = config.get("schema") if isinstance(config, dict) else None
        raise ValueError(f"{name}: {FILE} is schema {got!r}; this decisio reads {' and '.join(SCHEMAS)}")
    if config.get("base") not in BASES:
        raise ValueError(f"{name}: {FILE} names the base {config.get('base')!r}; this decisio serves {sorted(BASES)}")
    try:
        check_sections(config)
    except ValueError as e:
        raise ValueError(f"{name}: {e}") from e
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
    ex.add_argument("--base", choices=sorted(REPOSITORIES))
    ex.add_argument("--all", action="store_true", help="every base (with --out)")
    ex.add_argument("--out", help="a directory: writes decision_config.<base>.json there")
    ex.add_argument("--decisio-version", help="the version to record (default: the installed one)")
    ex.add_argument("--quantization", help="the 31B's stored scheme, if not compressed-tensors (the identity gate)")
    ck = sub.add_parser("check", help="compare a file with the installed profile (a /2 file: and check its sections)")
    ck.add_argument("file")
    ck.add_argument("--weights-dir", help="a /2 file: a directory whose files must have the digests in tuned.files")
    tc = sub.add_parser("tuned-config", help="write the /2 file of a tuned model (an adapter merged into the base)")
    tc.add_argument("--base", required=True, choices=list(TUNED_BASES))
    tc.add_argument("--temperatures", required=True, help="a decisio.adapter/1 file, as the split-fit tool writes it")
    tc.add_argument(
        "--noul-commit", action=argparse.BooleanOptionalAction, default=None, help="the commit setting (required)"
    )
    tc.add_argument("--noul-commit-basis", help="a JSON file, or text: why that setting, with the figures")
    tc.add_argument(
        "--think-factor",
        type=float,
        help="think when unsure for this model, with this factor on the thresholds (the one its evaluation chose)",
    )
    tc.add_argument("--think-budget", type=int, help="the most tokens of reasoning (default 512)")
    tc.add_argument(
        "--think-temperature",
        action="append",
        default=[],
        metavar="TYPE=T",
        help="the temperature of the answers that thought for a question type (choice, noul or score), e.g. "
        "score=4.0: written to defaults.think_temperatures, over the --temperatures file's own (repeatable)",
    )
    tc.add_argument(
        "--over-limit",
        choices=("refuse", "truncate"),
        help="what a server of this repository does with a state over the engine's prompt ceiling (default: not "
        "stated, so the server's own, refuse)",
    )
    tc.add_argument(
        "--over-limit-tokens",
        type=int,
        help="the prompt ceiling a truncating server of this repository cuts a state to, when it is lower than the "
        "context it serves (defaults.over_limit_tokens; the server's --over-limit-tokens beats it)",
    )
    tc.add_argument(
        "--think-types",
        help="the question types that think, comma separated (choice, noul, score), written to defaults.think.types; "
        "needs --think-factor (default: every type)",
    )
    tc.add_argument(
        "--layout",
        choices=("plain", "header"),
        help="the prompt layout the model was trained and measured on, recorded in the file; a server serves it "
        "(none recorded: plain, what every tuned model before this was trained on)",
    )
    tc.add_argument(
        "--flavor",
        choices=("calibrated", "commit", "read-enhanced", "commit-think"),
        help="which flavor of the model this repository is (decisio.flavors): written as the file's `flavor` block "
        "with --served-name; the file's defaults must be that flavor's (calibrated: --no-noul-commit; commit: "
        "--noul-commit with its basis; read-enhanced: --read-twice-under; commit-think: the commit rule and "
        "--think-factor) and the header layout",
    )
    tc.add_argument(
        "--served-name", help="the flavor's served name, ending in -<flavor>, e.g. Decisio-12b-tuned-commit"
    )
    tc.add_argument(
        "--read-twice-under",
        type=int,
        help="a request read once in at most this many counted input tokens is read twice",
    )
    tc.add_argument("--read-twice-max-questions", type=int, help="and with at most this many questions (0: no cap)")
    tc.add_argument(
        "--batch-invariant",
        action="store_true",
        help="a server of this repository runs vLLM's batch-invariant kernels (defaults.batch_invariant); an older "
        "decisio refuses such a repository",
    )
    tc.add_argument(
        "--external-form",
        metavar="TOOL@VERSION",
        help="the checkpoint was made by another tool (for example llm-compressor@0.8.1) and --merge-report is a "
        "hand-written JSON: adapter_sha256, adapter_scale, merge_precision, scheme (FP8_DYNAMIC), weights "
        "(per-channel), activations (per-token dynamic), optionally bf16_modules; the record and the weights text say "
        "how it was made, not that decisio made it. FP8 bases only",
    )
    tc.add_argument(
        "--mlx-conversion",
        metavar="TOOL@VERSION",
        help="the weights directory is an MLX conversion of the merged model, made by that tool (for example "
        "mlx-lm@0.32.0): written as tuned.conversion and the MLX weights section; --merge-report is then the merge "
        "report of the bf16 model it was converted from; a server of this repository runs with --backend mlx",
    )
    tc.add_argument("--mlx-bits", type=int, choices=MLX_BITS, help="the conversion's bits per weight")
    tc.add_argument(
        "--mlx-group-size", type=int, choices=MLX_GROUP_SIZES, help="the conversion's quantization group size"
    )
    tc.add_argument(
        "--defaults-only",
        action="store_true",
        help="the repository is the base's own weights with settings of their own: no adapter, no merge; writes a /2 "
        "file without `tuned` (the weight files are listed in source.files). --adapter-dir, --adapter-revision, "
        "--merge-report and the options about a merge are then not given",
    )
    tc.add_argument("--adapter-dir", help="the PEFT adapter directory the model was merged from")
    tc.add_argument(
        "--adapter-repository",
        help="the adapter's repository, recorded in `tuned.adapter`; leave it out to keep the name out of the file "
        "(the revision and the sha256s stay)",
    )
    tc.add_argument("--adapter-revision")
    tc.add_argument("--merge-report", help="the merge's report: MERGE_FP8.json (hub_fp8), MERGE_BF16.json (hub_bf16)")
    tc.add_argument("--weights-dir", required=True, help="the merged model's directory: its files are listed by sha256")
    tc.add_argument("--out", required=True, help="where to write decision_config.json")
    tc.add_argument("--decisio-version")
    tc.add_argument("--quantization")
    a = ap.parse_args(argv)
    if a.cmd == "check":
        config = json.loads(Path(a.file).read_text())
        try:
            check_sections(config)
            if config.get("schema") not in SCHEMAS:
                raise ValueError(f"schema {config.get('schema')!r}")
            if a.weights_dir:
                check_weights_precision(config.get("base"), Path(a.weights_dir))
                listed = config.get("tuned") or config.get("source") or {}
                have, want = file_digests(Path(a.weights_dir)), listed.get("files") or {}
                bad = sorted(n for n in set(have) | set(want) if have.get(n) != want.get(n))
                if bad:
                    raise ValueError(
                        f"files differ from {'tuned' if 'tuned' in config else 'source'}.files: " + ", ".join(bad[:10])
                    )
        except ValueError as e:
            print(f"refused: {e}")
            return 1
        diff = check_profile(config, BASES[config["base"]])
        print("agrees with the installed profile" if not diff else "differs at: " + ", ".join(diff))
        return 1 if diff else 0
    if a.cmd == "tuned-config":
        if a.noul_commit is None:
            ap.error("give --noul-commit or --no-noul-commit: a tuned repository always states its commit setting")
        if a.think_types and a.think_factor is None:
            ap.error("--think-types needs --think-factor: it limits the question types that think")
        merge_args = {
            "--adapter-dir": a.adapter_dir,
            "--adapter-revision": a.adapter_revision,
            "--merge-report": a.merge_report,
            "--adapter-repository": a.adapter_repository,
            "--external-form": a.external_form,
            "--mlx-conversion": a.mlx_conversion,
        }
        merged = [f for f, v in merge_args.items() if v]
        if a.defaults_only and merged:
            ap.error(f"--defaults-only is the base's own weights: {', '.join(merged)} describe a merge")
        if not a.defaults_only:
            missing = [f for f in ("--adapter-dir", "--adapter-revision", "--merge-report") if not merge_args[f]]
            if missing:
                ap.error(f"{', '.join(missing)} required (or --defaults-only for the base's own weights)")
        from decisio.serve.adapter_temperatures import read as read_temperatures

        fam = BASES[a.base]
        t = read_temperatures(a.temperatures)
        fit = {
            **(t["fit"] or {}),
            "temperature_file_sha256": hashlib.sha256(Path(a.temperatures).read_bytes()).hexdigest(),
        }
        fit.setdefault(
            "method", "decisio.adapter/1 file: minimum log loss per question type and choice band, on the plain readout"
        )
        from decisio.serve.adapter_temperatures import AdapterFileError, validate_think_temperatures

        try:
            given = {}
            for item in a.think_temperature:
                kind, sep, value = item.partition("=")
                if not sep:
                    raise AdapterFileError(
                        f"--think-temperature {item!r} must be TYPE=TEMPERATURE, for example score=4.0"
                    )
                try:
                    given[kind] = float(value)
                except ValueError:
                    raise AdapterFileError(f"--think-temperature {item!r}: {value!r} is not a number") from None
            think_temperatures = {
                **t["think_temperatures"],
                **validate_think_temperatures(given or None, "--think-temperature"),
            }
        except AdapterFileError as e:
            print(f"refused: {e}")
            return 1
        basis = a.noul_commit_basis
        if basis and os.path.isfile(basis):
            basis = json.loads(Path(basis).read_text())
        try:
            check_weights_precision(a.base, Path(a.weights_dir))
            common = dict(
                temperatures=t["temperatures"],
                choice_bands=t["choice_bands"] or None,
                noul_commit=a.noul_commit,
                noul_commit_basis=basis,
                fit=fit,
                decisio_version=a.decisio_version,
                quantization=a.quantization,
                over_limit=a.over_limit,
                over_limit_tokens=a.over_limit_tokens,
                layout=a.layout or t.get("layout"),
                think_temperatures=think_temperatures or None,
                flavor=({"name": a.flavor, "served_name": a.served_name} if a.flavor or a.served_name else None),
                read_twice_under=a.read_twice_under,
                read_twice_max_questions=a.read_twice_max_questions,
                batch_invariant=a.batch_invariant,
                think=(
                    {
                        "factor": a.think_factor,
                        **({"budget": a.think_budget} if a.think_budget else {}),
                        **({"types": [x.strip() for x in a.think_types.split(",")]} if a.think_types else {}),
                    }
                    if a.think_factor is not None
                    else None
                ),
            )
            if a.defaults_only:
                config = defaults_config(fam, files=file_digests(Path(a.weights_dir)), **common)
            else:
                config = tuned_config(
                    fam,
                    adapter=adapter_facts(Path(a.adapter_dir), a.adapter_repository, a.adapter_revision),
                    merge=(
                        external_facts(a.external_form, json.loads(Path(a.merge_report).read_text()), a.base)
                        if a.external_form
                        else merge_facts(json.loads(Path(a.merge_report).read_text()))
                    ),
                    files=file_digests(Path(a.weights_dir)),
                    conversion=(
                        mlx_conversion_of(a.mlx_conversion, a.mlx_bits, a.mlx_group_size, Path(a.weights_dir), a.base)
                        if a.mlx_conversion
                        else None
                    ),
                    **common,
                )
        except ValueError as e:
            print(f"refused: {e}")
            return 1
        Path(a.out).write_text(json.dumps(config, indent=2) + "\n")
        print(f"wrote {a.out}: defaults sha256 {defaults_sha256(config)}")
        return 0
    bases = sorted(REPOSITORIES) if a.all else [a.base]
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
