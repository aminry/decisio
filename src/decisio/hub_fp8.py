# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""The FP8 checkpoint of the Gemma 4 31B that decisio's server would otherwise make when it loads the bf16 one
(`quantization="fp8"` in decisio.families), made on a CPU and predicted tensor by tensor
(the identity gate).

What vLLM 0.30.0 does on load (online per-tensor FP8, `Fp8PerTensorOnlineLinearMethod`): every linear layer of the text
model, after its shards are fused (q, k and v into `qkv_proj`; gate and up into `gate_up_proj`), gets ONE scale,
`float32(amax) / 448` with `amax = max|w|` over the fused bf16 weight, and its weight becomes
`fp8_e4m3fn(clamp(w * (1 / scale), -448, 448))` (the static kernel multiplies by the fp32 reciprocal; the conversion is
round to nearest even, saturating). Activations are quantized per token at run time, so no activation scale is stored.
On the layers with `attention_k_eq_v` (full-attention layers with no `v_proj`) vLLM loads K into the V slot too, so the
fused weight is [q; k; k].

This module writes that as a checkpoint: each source tensor `X.weight` becomes an FP8 tensor with the scale of its fused
group stored beside it as `X.weight_scale` (float32, shape [1]), under the source's own names, so the file is laid out
like the source (a layer with no `v_proj` has none here either). Every other tensor, the vision and audio towers
included, is copied unchanged.

Loading such a checkpoint is NOT the identity on the weights: vLLM 0.30.0 requantizes the shards of a fused layer with
the largest of their scales, as `float16(w) * scale` and then back to FP8 (`requantize_with_max_scale`,
`per_tensor_dequantize`). With equal scales that is exact unless the layer's scale is so small that `float16(w) * scale`
falls among float16's subnormals (a layer whose amax is below about 0.02, measured over every FP8 value). So the
conversion also runs that step and reports, per layer, how many weights it would change: zero everywhere means the
loaded tensors equal the online ones bit for bit, and the engine's own tensors (`decisio_fp8_fingerprints`, compared
with `compare`) say whether that prediction holds on a card.

    python -m decisio.hub_fp8 convert --source <bf16 dir> --out <dir> [--scheme compressed-tensors|fp8]
    python -m decisio.hub_fp8 expected --source <bf16 dir> --out online.json     # what quantizing on load holds
    python -m decisio.hub_fp8 expected --checkpoint <dir> --out loaded.json     # what loading the checkpoint holds
    python -m decisio.hub_fp8 compare a.json b.json                            # tensors that differ
    python -m decisio.hub_fp8 merge --source <bf16 dir> --adapter <peft dir> --out <dir> [--device cuda] [--null]
    ... convert|merge --keep-bf16 layers:45-47 --keep-bf16 proj:down_proj,o_proj   # layers kept in bf16 (below)

`merge` makes the FP8 checkpoint of a tuned model: a LoRA adapter merged into the bf16 base in fp32 and quantized ONCE
(`W + scale * (B @ A)`, no bf16 anywhere between), with the adapter's change inside the amax of each fused group and
inside the rounding. The base weights are already on the bf16 grid and the adapter's change is 0.04 to 0.09 percent of a
weight, so `(W + delta).to(bf16)` rounds back to W for 81 to 91 percent of the entries and a merge through bf16 keeps
about half of the adapter; the fp32 merge keeps 99 percent of its effect (measured on a tuned 31B,
2026-10-13). Every adapter pair must land on a tensor of a quantized group: a pair that finds no tensor is an error,
never a merge of nothing. `--null` merges an adapter of scale 0, which must equal `convert` of the base byte for byte
(the pipeline alone). Writes `MERGE_FP8.json` (what `decisio.hub tuned-config` reads), `expected_written.json` and
`expected_loaded.json` beside the checkpoint; exits 1 if the loader would change a weight or the two fingerprints
differ.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path

import torch

FP8 = torch.float8_e4m3fn
FP8_MAX = 448.0
INDEX = "model.safetensors.index.json"
WEIGHTS = re.compile(r".*\.safetensors$")
# a quantized projection of the text model: its prefix up to the layer, the module and the projection
PROJ = re.compile(
    r"^(?P<layer>.*language_model\.layers\.(?P<idx>\d+))\.(?P<mod>self_attn|mlp)\."
    r"(?P<proj>q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)\.weight$"
)
FUSED = {"self_attn": ("q_proj", "k_proj", "v_proj"), "mlp": ("gate_proj", "up_proj")}
FUSED_NAME = {"self_attn": "qkv_proj", "mlp": "gate_up_proj"}
SINGLE = ("o_proj", "down_proj")
IGNORED = ["lm_head", "re:.*vision_tower.*", "re:.*audio_tower.*", "re:.*embed_vision.*", "re:.*embed_audio.*"]
# the 12B's (gemma4_unified) vision embedder has a linear layer of its own, `patch_dense`, which vLLM builds with the
# checkpoint's quantization config: left out of the ignore list, loading the stored checkpoint would look for an FP8
# weight and scale the conversion never wrote (it converts the text model's layers only). Only for that model type, so
# a 31B conversion writes the ignore list its published repository holds
IGNORED_BY_TYPE = {"gemma4_unified": ["re:.*vision_embedder.*"]}


def ignored_for(config: dict) -> list[str]:
    """The modules a conversion of this checkpoint leaves unquantized, by its config.json's model type."""
    return IGNORED + IGNORED_BY_TYPE.get(config.get("model_type"), [])


@dataclass
class Group:
    """The source tensors vLLM fuses into one linear layer, and quantizes with one scale."""

    key: str  # the engine's module, from the layer on: layers.N.self_attn.qkv_proj
    names: list[str]  # the source tensors present, in fused order
    fused: list[str]  # the fused order: a missing v_proj repeats k_proj (k_eq_v)


def find_groups(names, k_eq_v: bool = False) -> list[Group]:
    """The quantized groups of a checkpoint's tensor names (the text model's projections)."""
    layers: dict[tuple[str, str], dict[str, str]] = {}
    for n in names:
        m = PROJ.match(n)
        if m:
            layers.setdefault((m["layer"], m["mod"]), {})[m["proj"]] = n
    groups = []

    def order_of(item):
        (layer, mod), _ = item
        return int(re.search(r"(\d+)$", layer)[1]), mod != "self_attn"  # a layer's attention, then its MLP

    for (layer, mod), projs in sorted(layers.items(), key=order_of):
        idx = re.search(r"layers\.(\d+)$", layer)[1]
        order = FUSED[mod]
        present = [p for p in order if p in projs]
        if mod == "self_attn" and "v_proj" not in projs:
            if not k_eq_v or not {"q_proj", "k_proj"} <= set(projs):
                raise ValueError(f"{layer}.self_attn has no v_proj and the config does not say attention_k_eq_v")
            fused = [projs["q_proj"], projs["k_proj"], projs["k_proj"]]
        else:
            if present != list(order):
                raise ValueError(f"{layer}.{mod}: expected {list(order)}, found {present}")
            fused = [projs[p] for p in order]
        groups.append(Group(f"layers.{idx}.{mod}.{FUSED_NAME[mod]}", [projs[p] for p in present], fused))
        for single in (s for s in SINGLE if s in projs):
            groups.append(Group(f"layers.{idx}.{mod}.{single}", [projs[single]], [projs[single]]))
    return groups


GROUP_KEY = re.compile(
    r"^layers\.(?P<idx>\d+)\.(?P<mod>self_attn|mlp)\.(?P<name>qkv_proj|gate_up_proj|o_proj|down_proj)$"
)
MEMBERS = {
    "qkv_proj": ("q_proj", "k_proj", "v_proj"),
    "gate_up_proj": ("gate_proj", "up_proj"),
    "o_proj": ("o_proj",),
    "down_proj": ("down_proj",),
}


@dataclass(frozen=True)
class Gptq:
    """`--gptq CALIBRATION`: round the merged weights with error compensation instead of to nearest (decisio.fp8_gptq),
    from the Hessians of the calibration requests' activations (decisio.fp8_gptq_pass). The grid and the scale of each
    fused group do not change; only some codes move one step."""

    calibration: Path
    damp: float = 0.01
    blocksize: int = 128
    max_tokens: int = 6000
    limit: int | None = None


@dataclass(frozen=True)
class Keep:
    """The projections left in bf16 instead of FP8 (`--keep-bf16`): whole layers (`layers:45-47`), a projection in every
    layer (`proj:down_proj,o_proj`) or one group (`group:layers.3.mlp.down_proj`). A vLLM fused layer (qkv_proj,
    gate_up_proj) has one scheme, so naming one of its shards keeps the whole group; `extended` says where that
    happened."""

    layers: frozenset = frozenset()
    projs: frozenset = frozenset()
    groups: frozenset = frozenset()
    specs: tuple = ()
    extended: list = field(default_factory=list, compare=False)

    def __bool__(self) -> bool:
        return bool(self.layers or self.projs or self.groups)

    def keeps(self, key: str) -> bool:
        """Whether the group with this key (layers.N.self_attn.qkv_proj, ...) stays bf16."""
        m = GROUP_KEY.match(key)
        if m is None:
            raise ValueError(f"not a group key: {key!r}")
        if key in self.groups or int(m["idx"]) in self.layers:
            return True
        members = MEMBERS[m["name"]]
        hit = [p for p in members if p in self.projs or m["name"] in self.projs]
        if hit and len(members) > 1 and not set(members) <= self.projs and m["name"] not in self.projs:
            note = f"{key}: {', '.join(hit)} named, so the whole fused group is kept"
            if note not in self.extended:
                self.extended.append(note)
        return bool(hit)


def parse_keep(specs) -> Keep:
    """`--keep-bf16` specs: `layers:45-47` or `layers:3,5`, `proj:down_proj,o_proj` (any of q/k/v/o/gate/up/down_proj,
    or the fused names qkv_proj and gate_up_proj), `group:layers.3.mlp.down_proj`. Raises ValueError naming the spec."""
    layers, projs, groups = set(), set(), set()
    known = {p for ps in MEMBERS.values() for p in ps} | set(MEMBERS)
    for spec in specs or ():
        kind, sep, value = str(spec).partition(":")
        if not sep or not value:
            raise ValueError(f"--keep-bf16 {spec!r}: expected layers:45-47, proj:down_proj,o_proj or group:<key>")
        if kind == "layers":
            for part in value.split(","):
                lo, dash, hi = part.partition("-")
                if not lo.isdigit() or (dash and not hi.isdigit()) or (dash and int(hi) < int(lo)):
                    raise ValueError(f"--keep-bf16 {spec!r}: layers are numbers or ranges like 45-47")
                layers.update(range(int(lo), int(hi or lo) + 1))
        elif kind == "proj":
            bad = [p for p in value.split(",") if p not in known]
            if bad:
                raise ValueError(f"--keep-bf16 {spec!r}: {bad[0]!r} is not a projection ({', '.join(sorted(known))})")
            projs.update(value.split(","))
        elif kind == "group":
            if not GROUP_KEY.match(value):
                raise ValueError(f"--keep-bf16 {spec!r}: {value!r} is not a group key like layers.3.mlp.down_proj")
            groups.add(value)
        else:
            raise ValueError(f"--keep-bf16 {spec!r}: expected layers:, proj: or group:")
    return Keep(frozenset(layers), frozenset(projs), frozenset(groups), tuple(specs or ()))


def kept_groups(groups, keep: Keep) -> list:
    """The groups of `groups` that `keep` leaves in bf16, refused when it names a layer or group that does not exist."""
    out = [g for g in groups if keep.keeps(g.key)]
    have_layers = {int(GROUP_KEY.match(g.key)["idx"]) for g in groups}
    missing = sorted(keep.layers - have_layers)
    if missing:
        raise ValueError(
            f"--keep-bf16 layers {missing} are not in this checkpoint (layers {min(have_layers)}-{max(have_layers)})"
        )
    unknown = sorted(keep.groups - {g.key for g in groups})
    if unknown:
        raise ValueError(f"--keep-bf16 group {unknown[0]!r} is not in this checkpoint")
    if keep and not out:
        raise ValueError("--keep-bf16 names nothing in this checkpoint")
    return out


def ignored_modules(kept) -> list[str]:
    """The entries to add to the quantization config's ignore list for the kept groups: a pattern per shard of a fused
    group, and for v_proj where the layer has none (attention_k_eq_v), so the fused layer is all-or-none for vLLM.
    Patterns (`re:`), not full names, so that the engine's own module prefix does not matter (as the base list does)."""
    out = []
    for g in kept:
        m = GROUP_KEY.match(g.key)
        out += [f"re:.*layers\\.{m['idx']}\\.{m['mod']}\\.{p}$" for p in MEMBERS[m["name"]]]
    return out


def kept_report(keep: Keep, kept, seed: int | None = None) -> dict | None:
    """What the report says of the bf16 exceptions: the specs, the groups kept (and where a spec was extended to a whole
    fused group), and for a merge the rounding of the kept tensors; None when nothing is kept."""
    if not kept:
        return None
    return {
        "specs": list(keep.specs),
        "groups": [g.key for g in kept],
        "extended": list(keep.extended),
        **({"rounding": "stochastic", "seed": seed} if seed is not None else {}),
    }


def check_output_folder(out, overwrite: bool = False) -> Path:
    """The folder a conversion or merge writes into. A folder that already holds files is refused unless `overwrite`:
    vLLM keys its compiled-graph cache by the checkpoint's path, so a checkpoint rewritten in place with another layout
    (other kept projections, another merge) reuses the old compiled graph, which can crash in the first forward pass or,
    if the argument counts agree, run the wrong graph. A new folder per checkpoint avoids it; after an overwrite, clear
    the compile cache (`VLLM_CACHE_ROOT`'s torch_compile_cache and /tmp/torchinductor_*) before serving it."""
    out = Path(out)
    if not overwrite and out.is_dir() and any(out.iterdir()):
        raise ValueError(
            f"{out} is not empty: write a checkpoint into a folder of its own, because vLLM's compiled-graph cache is "
            "keyed by the folder's path and a checkpoint rewritten in place can crash or run a stale graph; or give "
            "--overwrite, and clear vLLM's compile cache before serving it"
        )
    return out


def scale_of(amax: torch.Tensor) -> torch.Tensor:
    """float32(amax) / 448, a true division as vLLM's `_fp8_scale` does."""
    return amax.to(torch.float32).reshape(1) / torch.tensor(FP8_MAX, dtype=torch.float32)


def quantize(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """The static FP8 kernel: multiply by the fp32 reciprocal of the scale, clamp, round to nearest even."""
    inv = torch.reciprocal(scale.to(torch.float32))
    return (weight.to(torch.float32) * inv).clamp(-FP8_MAX, FP8_MAX).to(FP8)


def loader_requantize(q: torch.Tensor, own_scale: torch.Tensor, max_scale: torch.Tensor) -> torch.Tensor:
    """`requantize_with_max_scale` for one shard: float16(q) * its scale, then the static kernel with the max scale."""
    dq = q.to(torch.float16) * own_scale.reshape(())  # a 0-d float32 tensor does not promote a float16 one
    return quantize(dq.to(torch.float32), max_scale)


def amax_of(tensors) -> torch.Tensor:
    """max|w| over bf16 tensors, as `weight_amax` of the fused weight (an exact bf16 value)."""
    best = torch.zeros((), dtype=torch.bfloat16)
    for t in tensors:
        lo, hi = t.aminmax()
        best = torch.maximum(best, torch.maximum(lo.abs(), hi.abs()))
    return best


def digest(q: torch.Tensor) -> str:
    """sha256 of an FP8 tensor's bytes, row-major."""
    h = hashlib.sha256()
    flat = q.contiguous().view(torch.uint8).reshape(-1).numpy()
    for i in range(0, flat.size, 1 << 28):
        h.update(flat[i : i + (1 << 28)].tobytes())
    return h.hexdigest()


def fingerprint(fused_q: torch.Tensor, scale: torch.Tensor) -> dict:
    return {"shape": list(fused_q.shape), "scale": float(scale.reshape(())), "sha256": digest(fused_q)}


def compare(a: dict, b: dict) -> list[str]:
    """The modules at which two fingerprint sets differ (a missing one, another shape, scale or tensor)."""
    return sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))


# ---- reading and writing safetensors -------------------------------------------------------------------------------


class Source:
    """A checkpoint directory's tensors, read lazily by name."""

    def __init__(self, directory: Path):
        from safetensors import safe_open

        self.dir = Path(directory)
        index = self.dir / INDEX
        if index.exists():
            self.map = json.loads(index.read_text())["weight_map"]
        else:
            files = sorted(self.dir.glob("*.safetensors"))
            if len(files) != 1:
                raise ValueError(f"{directory}: no {INDEX} and {len(files)} safetensors files")
            with safe_open(files[0], "pt") as f:
                self.map = {k: files[0].name for k in f.keys()}  # noqa: SIM118
        self._open = safe_open
        self._handles: dict[str, object] = {}

    def get(self, name: str) -> torch.Tensor:
        file = self.map[name]
        if file not in self._handles:
            self._handles[file] = self._open(self.dir / file, "pt").__enter__()
        return self._handles[file].get_tensor(name)

    def config(self) -> dict:
        return json.loads((self.dir / "config.json").read_text())

    def k_eq_v(self) -> bool:
        cfg = self.config()
        return bool((cfg.get("text_config") or cfg).get("attention_k_eq_v"))


class Writer:
    """Tensors into safetensors files of about `limit` bytes, named model-00001-of-0000N once the count is known."""

    def __init__(self, out: Path, limit: int):
        self.out, self.limit, self.pending, self.size, self.files, self.map = Path(out), limit, {}, 0, [], {}

    def add(self, name: str, tensor: torch.Tensor) -> None:
        self.pending[name] = tensor.contiguous()
        self.size += tensor.numel() * tensor.element_size()
        if self.size >= self.limit:
            self.flush()

    def flush(self) -> None:
        from safetensors.torch import save_file

        if not self.pending:
            return
        tmp = f"part-{len(self.files):05d}.safetensors"
        save_file(self.pending, str(self.out / tmp), metadata={"format": "pt"})
        self.files.append((tmp, list(self.pending)))
        self.pending, self.size = {}, 0

    def finish(self) -> dict:
        self.flush()
        n = len(self.files)
        for i, (tmp, names) in enumerate(self.files, start=1):
            final = f"model-{i:05d}-of-{n:05d}.safetensors"
            (self.out / tmp).rename(self.out / final)
            self.map.update({name: final for name in names})
        return self.map


def quantization_config(scheme: str, ignored=None) -> dict:
    """The `quantization_config` that tells vLLM the checkpoint is FP8 per tensor with dynamic activations; `ignored`
    the modules left unquantized (ignored_for the checkpoint's config; the base list when not given)."""
    ignored = IGNORED if ignored is None else ignored
    if scheme == "fp8":
        return {"quant_method": "fp8", "activation_scheme": "dynamic", "fmt": "e4m3", "modules_to_not_convert": ignored}
    if scheme == "compressed-tensors":
        return {
            "quant_method": "compressed-tensors",
            "format": "float-quantized",
            "quantization_status": "compressed",
            "global_compression_ratio": None,
            "kv_cache_scheme": None,
            "ignore": ignored,
            "config_groups": {
                "group_0": {
                    "targets": ["Linear"],
                    "weights": {
                        "num_bits": 8,
                        "type": "float",
                        "strategy": "tensor",
                        "symmetric": True,
                        "dynamic": False,
                        "observer": "minmax",
                    },
                    "input_activations": {
                        "num_bits": 8,
                        "type": "float",
                        "strategy": "token",
                        "symmetric": True,
                        "dynamic": True,
                    },
                    "output_activations": None,
                }
            },
        }
    raise ValueError(f"--scheme must be compressed-tensors or fp8, not {scheme!r}")


def convert(
    source: Path,
    out: Path,
    scheme: str = "compressed-tensors",
    shard_bytes: int = 4 << 30,
    progress=None,
    keep: Keep | None = None,
    overwrite: bool = False,
) -> dict:
    """Write the FP8 checkpoint of `source` (a bf16 Gemma 4 directory) to `out`; the report of what was done and what
    loading it would change. `keep`: groups left in bf16 as the source has them (`--keep-bf16`). `out` must be a new or
    empty folder unless `overwrite` (check_output_folder)."""
    check_output_folder(out, overwrite)
    src = Source(source)
    every = find_groups(src.map, src.k_eq_v())
    if not every:
        raise ValueError(f"{source}: no text-model projections found")
    keep = keep or Keep()
    kept = kept_groups(every, keep)
    kept_keys = {g.key for g in kept}
    groups = [g for g in every if g.key not in kept_keys]
    if not groups:
        raise ValueError("--keep-bf16 keeps every projection: nothing is left to quantize")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    quantized = {n for g in groups for n in g.names}
    writer = Writer(out, shard_bytes)
    report = {"scheme": scheme, "groups": {}, "loader_changed_elements": 0, "smallest_amax": None}
    for i, g in enumerate(groups):
        tensors = {n: src.get(n) for n in dict.fromkeys(g.fused)}
        bad = [n for n, t in tensors.items() if t.dtype != torch.bfloat16 or t.dim() != 2]
        if bad:
            raise ValueError(f"{bad[0]}: expected a 2-D bfloat16 weight, found {tensors[bad[0]].dtype}")
        amax = amax_of(tensors.values())
        scale = scale_of(amax)
        q = {n: quantize(t, scale) for n, t in tensors.items()}
        fused_q = torch.cat([q[n] for n in g.fused])
        # what vLLM's loader makes of the shards as stored: each requantized with the largest of the scales (all equal)
        loaded = torch.cat([loader_requantize(q[n], scale, scale.max()) for n in g.fused])
        changed = int((loaded.view(torch.uint8) != fused_q.view(torch.uint8)).sum())
        for n in tensors:
            writer.add(n, q[n])
            writer.add(n[: -len(".weight")] + ".weight_scale", scale.clone())
        report["groups"][g.key] = {
            "amax": float(amax),
            "scale": float(scale),
            "shape": list(fused_q.shape),
            "loader_changed": changed,
            "sha256_online": digest(fused_q),
            "sha256_loaded": digest(loaded),
        }
        report["loader_changed_elements"] += changed
        a = float(amax)
        report["smallest_amax"] = a if report["smallest_amax"] is None else min(report["smallest_amax"], a)
        if progress and (i + 1) % 20 == 0:
            progress(f"{i + 1}/{len(groups)} groups")
    for name in sorted(set(src.map) - quantized):
        writer.add(name, src.get(name))
    weight_map = writer.finish()
    (out / INDEX).write_text(json.dumps({"metadata": {}, "weight_map": weight_map}, indent=2, sort_keys=True))
    cfg = src.config()
    cfg["quantization_config"] = quantization_config(scheme, ignored_for(cfg) + ignored_modules(kept))
    (out / "config.json").write_text(json.dumps(cfg, indent=2))
    for f in src.dir.iterdir():  # tokenizer, chat template, generation and processor configs: as they are
        if f.is_file() and not WEIGHTS.match(f.name) and f.name not in (INDEX, "config.json"):
            shutil.copy2(f, out / f.name)
    report["tensors"] = len(weight_map)
    if kept:
        report["kept_bf16"] = kept_report(keep, kept)
    (out / "fp8_conversion.json").write_text(json.dumps(report, indent=1))
    return report


LORA_KEY = re.compile(r"^base_model\.model\.(?P<name>.*)\.lora_(?P<half>[AB])\.weight$")


def base_name(adapter_name: str, names) -> str:
    """The base checkpoint's tensor for an adapter module name. A PEFT adapter of the text-only layout names the text
    model `model.layers.N...`; the base checkpoint of a multimodal Gemma 4 names it `model.language_model.layers.N...`
    (decisio.serve.adapters serves both layouts). The weight is whichever of the two spellings the base has."""
    for candidate in (adapter_name, adapter_name.replace("model.layers.", "model.language_model.layers.", 1)):
        if f"{candidate}.weight" in names:
            return f"{candidate}.weight"
    raise ValueError(f"{adapter_name}: no such tensor in the base (neither as it is nor in the multimodal layout)")


K_EQ_V_V = re.compile(r"^(?P<layer>.*layers\.\d+)\.self_attn\.v_proj$")


def load_adapter(adapter_dir: Path, names) -> tuple[dict, float, str, str, dict]:
    """({base tensor name: (A fp32 [r, in], B fp32 [out, r])}, scale = alpha / r, the weights file's sha256, the layout
    the adapter's names are in ("text" or "multimodal"), and the pairs with no base tensor that are `v_proj` of a
    full-attention layer which ships no `v_proj` (attention_k_eq_v: V is K), {module name: (A, B)}). Any other pair
    with no base tensor is an error."""
    from safetensors import safe_open

    d = Path(adapter_dir)
    cfg = json.loads((d / "adapter_config.json").read_text())
    if cfg.get("use_rslora") or cfg.get("use_dora"):
        raise ValueError("this merge handles plain LoRA only (use_rslora and use_dora are off in the trained adapters)")
    if cfg.get("rank_pattern") or cfg.get("alpha_pattern"):
        raise ValueError("this merge handles one rank and one alpha for every module (rank_pattern / alpha_pattern)")
    scale = cfg["lora_alpha"] / cfg["r"]
    parts: dict[str, dict[str, torch.Tensor]] = {}
    orphans: dict[str, dict[str, torch.Tensor]] = {}
    layout = set()
    f = d / "adapter_model.safetensors"
    with safe_open(str(f), "pt") as g:
        for key in g.keys():  # noqa: SIM118
            m = LORA_KEY.match(key)
            if not m:
                raise ValueError(f"unexpected adapter tensor {key}")
            layout.add("multimodal" if ".language_model.layers." in m["name"] else "text")
            tensor = g.get_tensor(key).to(torch.float32)
            try:
                parts.setdefault(base_name(m["name"], names), {})[m["half"]] = tensor
            except ValueError:
                v = K_EQ_V_V.match(m["name"])
                k_name = v and (v["layer"] + ".self_attn.k_proj")
                if not (
                    v
                    and any(
                        f"{c}.weight" in names
                        for c in (k_name, k_name.replace("model.layers.", "model.language_model.layers.", 1))
                    )
                ):
                    raise
                orphans.setdefault(m["name"], {})[m["half"]] = tensor
    bad = [n for n, p in {**parts, **orphans}.items() if set(p) != {"A", "B"}]
    if bad:
        raise ValueError(f"{bad[0]}: a LoRA pair is missing one half")
    if not parts:
        raise ValueError(f"{f}: no LoRA pairs")
    if len(layout) != 1:
        raise ValueError(f"{f}: the adapter mixes the text-only and the multimodal layout")
    sha = hashlib.sha256(f.read_bytes()).hexdigest()
    return (
        {n: (p["A"], p["B"]) for n, p in parts.items()},
        scale,
        sha,
        layout.pop(),
        {n: (p["A"], p["B"]) for n, p in orphans.items()},
    )


def k_copy_difference(name: str, ab, deltas: dict) -> str | None:
    """None when the `v_proj` pair `ab` of module `name` is bit for bit its layer's `k_proj` pair in `deltas`, else why
    not (the largest absolute difference of each half, or that the layer has no `k_proj` pair)."""
    k = K_EQ_V_V.match(name)["layer"] + ".self_attn.k_proj"
    for cand in (k, k.replace("model.layers.", "model.language_model.layers.", 1)):
        if f"{cand}.weight" in deltas:
            ka = deltas[f"{cand}.weight"]
            if all(torch.equal(x, y) for x, y in zip(ab, ka, strict=True)):
                return None
            return "A differs by up to {:.3g}, B by up to {:.3g}".format(
                *((x - y).abs().max().item() for x, y in zip(ab, ka, strict=True))
            )
    return "the layer's k_proj has no pair in the adapter"


def merged32(w: torch.Tensor, ab, scale: float, device: str) -> torch.Tensor:
    """W + scale * (B @ A) in fp32 on `device` (W alone if the weight has no adapter); no TF32, no bf16 anywhere."""
    w32 = w.to(device=device, dtype=torch.float32)
    if ab is None or scale == 0.0:
        return w32
    a, b = ab[0].to(device), ab[1].to(device)
    return w32 + scale * (b @ a)


def run_gptq(src, deltas, scale, device, keep, spec: Gptq, progress=None):
    """The compensated codes of every quantized group (decisio.fp8_gptq_pass) and the report of what was done:
    {group key: {"codes", "scale", "info"}}, and the `gptq` section of MERGE_FP8.json."""
    import transformers

    from decisio import fp8_gptq_pass
    from decisio.families import family_of

    tokenizer = transformers.AutoTokenizer.from_pretrained(str(src.dir))
    family = family_of(str(src.dir))
    sequences = fp8_gptq_pass.render_calibration(spec.calibration, tokenizer, family, spec.limit, spec.max_tokens)
    if not sequences:
        raise ValueError(f"{spec.calibration}: no calibration request renders to at most {spec.max_tokens} tokens")
    results = fp8_gptq_pass.gptq_pass(
        src,
        lambda name: merged32(src.get(name), deltas.get(name), scale, device),
        sequences,
        device=device,
        keep=keep,
        blocksize=spec.blocksize,
        damp=spec.damp,
        progress=progress,
    )
    rtn = sum(r["info"]["error_rtn"] for r in results.values())
    comp = sum(r["info"]["error_gptq"] for r in results.values())
    report = {
        "method": "error-compensating rounding to the group's fixed FP8 scale (GPTQ), layer by layer",
        "calibration_sha256": hashlib.sha256(Path(spec.calibration).read_bytes()).hexdigest(),
        "requests": len(sequences),
        "tokens": sum(len(s) for s in sequences),
        "damp": spec.damp,
        "blocksize": spec.blocksize,
        "max_tokens": spec.max_tokens,
        "output_error_round_to_nearest": rtn,
        "output_error_compensated": comp,
    }
    return results, report


def merge(
    source: Path,
    adapter: Path,
    out: Path,
    device: str = "cpu",
    null: bool = False,
    shard_bytes: int = 4 << 30,
    progress=None,
    keep: Keep | None = None,
    seed: int = 0,
    overwrite: bool = False,
    gptq: Gptq | None = None,
) -> dict:
    """Write the FP8 checkpoint of `source` (a bf16 Gemma 4 directory) with the LoRA adapter in `adapter` merged in
    fp32 and quantized once; the report (`MERGE_FP8.json`). Raises ValueError if an adapter pair finds no quantized
    tensor. `keep`: groups left in bf16 (`--keep-bf16`); their tensors are the same fp32 merge, rounded once to bf16
    stochastically with `seed` as `decisio.hub_bf16 merge` does (round to nearest would give back most of the
    adapter's change in them). `out` must be a new or empty folder unless `overwrite` (check_output_folder). `gptq`: the
    quantized groups are rounded with error compensation from the calibration requests' activations, layer by layer
    (`Gptq`); not with `null`.

    A full-attention layer with `attention_k_eq_v` ships no `v_proj` (V is K), so an adapter's `v_proj` pair there has
    no tensor to merge into, and a checkpoint of this layout cannot hold a V that differs from K. Where each such pair
    is bit for bit the layer's `k_proj` pair (the trainer's export copies it), merging the `k_proj` delta already gives
    V that delta, so the copy is left out and the report counts it (`dropped_k_eq_v_pairs`): nothing of the adapter is
    lost. A pair that differs from its layer's `k_proj` pair is an error: it asks for a V that differs from K."""
    check_output_folder(out, overwrite)
    src = Source(source)
    deltas, scale, sha, layout, orphans = load_adapter(adapter, src.map)
    differ = {n: d for n, ab in orphans.items() if (d := k_copy_difference(n, ab, deltas)) is not None}
    if differ:
        raise ValueError(
            f"{len(differ)} adapter pairs are v_proj of full-attention layers that ship no v_proj (attention_k_eq_v: "
            "V is K) and are not copies of the layer's k_proj pair, so they ask for a V that differs from K, which "
            "this checkpoint layout cannot hold: " + "; ".join(f"{n}: {d}" for n, d in sorted(differ.items())[:5])
        )
    scale = 0.0 if null else scale
    groups = find_groups(src.map, src.k_eq_v())
    if not groups:
        raise ValueError(f"{source}: no text-model projections found")
    keep = keep or Keep()
    kept = kept_groups(groups, keep)
    kept_keys = {g.key for g in kept}
    if len(kept) == len(groups):
        raise ValueError("--keep-bf16 keeps every projection: nothing is left to quantize")
    quantized = {n for g in groups for n in g.names}  # where an adapter pair may land: kept groups included
    uncovered = [n for n in deltas if n not in quantized]
    if uncovered:
        raise ValueError(f"{len(uncovered)} adapter weights are not in a quantized group, e.g. {uncovered[0]}")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    writer = Writer(out, shard_bytes)
    report = {
        "scheme": "compressed-tensors",
        "merged": True,
        "groups": {},
        "loader_changed_elements": 0,
        "smallest_amax": None,
    }
    expected: dict = {}
    applied = 0
    compensated: dict = {}
    gptq_report = None
    if gptq is not None:
        if null:
            raise ValueError(
                "--gptq does not combine with --null: a null merge is the byte-for-byte test of the pipeline"
            )
        compensated, gptq_report = run_gptq(src, deltas, scale, device, keep, gptq, progress)
    for i, g in enumerate(groups):
        t32 = {n: merged32(src.get(n), deltas.get(n), scale, device) for n in dict.fromkeys(g.fused)}
        applied += sum(1 for n in g.names if n in deltas)
        bad = [n for n, t in t32.items() if t.dim() != 2]
        if bad:
            raise ValueError(f"{bad[0]}: expected a 2-D weight")
        if g.key in kept_keys:  # left in bf16: the fp32 merge, rounded once
            from decisio.hub_bf16 import stochastic_bf16

            for n in g.names:
                writer.add(n, stochastic_bf16(t32[n].cpu(), seed, n))
            continue
        amax = torch.stack([t.abs().max() for t in t32.values()]).max().cpu()
        sc = scale_of(amax)
        # the FP8 cast on the CPU: every device's float8 support differs, the bytes do not
        if g.key in compensated:
            if not torch.equal(compensated[g.key]["scale"], sc):
                raise ValueError(f"{g.key}: the calibration pass and the merge disagree on the group's scale")
            q = {n: compensated[g.key]["codes"][n] for n in t32}
            for (
                n,
                c,
            ) in q.items():  # the codes must be grid points of this scale: rounding what they mean gives them back
                if not torch.equal(quantize(c.to(torch.float32) * sc, sc).view(torch.uint8), c.view(torch.uint8)):
                    raise ValueError(f"{n}: compensated codes are not on the FP8 grid of the group's scale")
        else:
            q = {n: quantize(t.cpu(), sc) for n, t in t32.items()}
        fused_q = torch.cat([q[n] for n in g.fused])
        loaded = torch.cat([loader_requantize(q[n], sc, sc.max()) for n in g.fused])
        changed = int((loaded.view(torch.uint8) != fused_q.view(torch.uint8)).sum())
        for n in t32:
            writer.add(n, q[n])
            writer.add(n[: -len(".weight")] + ".weight_scale", sc.clone())
        expected[g.key] = fingerprint(fused_q, sc)
        report["groups"][g.key] = {
            "amax": float(amax),
            "scale": float(sc),
            "shape": list(fused_q.shape),
            "loader_changed": changed,
            "sha256_online": expected[g.key]["sha256"],
            "sha256_loaded": digest(loaded),
        }
        report["loader_changed_elements"] += changed
        a = float(amax)
        report["smallest_amax"] = a if report["smallest_amax"] is None else min(report["smallest_amax"], a)
        if progress and (i + 1) % 20 == 0:
            progress(f"{i + 1}/{len(groups)} groups")
    if applied != len(deltas):  # a pair in no group's names would have been caught above; a repeated tensor, here
        raise ValueError(f"{len(deltas)} adapter pairs, {applied} applied: the merge would drop some of the adapter")
    for name in sorted(set(src.map) - quantized):
        writer.add(name, src.get(name))
    weight_map = writer.finish()
    (out / INDEX).write_text(json.dumps({"metadata": {}, "weight_map": weight_map}, indent=2, sort_keys=True))
    cfg = src.config()
    cfg["quantization_config"] = quantization_config("compressed-tensors", ignored_for(cfg) + ignored_modules(kept))
    (out / "config.json").write_text(json.dumps(cfg, indent=2))
    for f in src.dir.iterdir():  # tokenizer, chat template, generation and processor files: the base's own
        if f.is_file() and not WEIGHTS.match(f.name) and f.name not in (INDEX, "config.json"):
            shutil.copy2(f, out / f.name)
    report["tensors"] = len(weight_map)
    if kept:
        report["kept_bf16"] = kept_report(keep, kept, seed)
    (out / "fp8_conversion.json").write_text(json.dumps(report, indent=1))
    (out / "expected_written.json").write_text(json.dumps(expected, indent=1, sort_keys=True))
    loaded_fp = expected_loaded(out)
    (out / "expected_loaded.json").write_text(json.dumps(loaded_fp, indent=1, sort_keys=True))
    differ = compare(expected, loaded_fp)
    meta = {
        "kind": "fp32 merge to FP8",
        "adapter_sha256": sha,
        "adapter_scale": scale,
        "adapter_layout": layout,
        "adapted_tensors": len(deltas),
        "dropped_k_eq_v_pairs": len(orphans),
        "null": null,
        "groups": len(groups),
        "quantized_groups": len(groups) - len(kept),
        **({"kept_bf16": kept_report(keep, kept, seed)} if kept else {}),
        **({"gptq": gptq_report} if gptq_report else {}),
        "written_vs_loaded_differ": differ,
        "loader_changed_elements": report["loader_changed_elements"],
        "smallest_amax": report["smallest_amax"],
    }
    (out / "MERGE_FP8.json").write_text(json.dumps(meta, indent=1) + "\n")
    return meta


def expected_online(source: Path, keep: Keep | None = None) -> dict:
    """What the engine holds after quantizing `source` on load: per module, the fused weight's bytes and scale (the
    groups `keep` leaves in bf16 are not quantized and not listed)."""
    src = Source(source)
    out = {}
    groups = find_groups(src.map, src.k_eq_v())
    skip = {g.key for g in kept_groups(groups, keep)} if keep else set()
    for g in (g for g in groups if g.key not in skip):
        tensors = {n: src.get(n) for n in dict.fromkeys(g.fused)}
        scale = scale_of(amax_of(tensors.values()))
        out[g.key] = fingerprint(torch.cat([quantize(tensors[n], scale) for n in g.fused]), scale)
    return out


def expected_loaded(checkpoint: Path) -> dict:
    """What the engine holds after loading a converted checkpoint: its shards, requantized as the loader does."""
    src = Source(checkpoint)
    out = {}
    for g in find_groups(src.map, src.k_eq_v()):
        scaled = [n[: -len(".weight")] + ".weight_scale" in src.map for n in g.names]
        if not any(scaled):  # a group kept in bf16 (--keep-bf16): not quantized, nothing to load as FP8
            continue
        if not all(scaled):
            raise ValueError(f"{g.key}: some of its shards have a weight_scale and some do not")
        shards = {n: src.get(n) for n in dict.fromkeys(g.fused)}
        scales = {n: src.get(n[: -len(".weight")] + ".weight_scale").reshape(1).to(torch.float32) for n in shards}
        top = torch.stack([s.reshape(()) for s in scales.values()]).max().reshape(1)
        fused = torch.cat([loader_requantize(shards[n], scales[n], top) for n in g.fused])
        out[g.key] = fingerprint(fused, top)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m decisio.hub_fp8", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    cv = sub.add_parser("convert", help="write the FP8 checkpoint of a bf16 Gemma 4 directory")
    cv.add_argument("--source", required=True)
    cv.add_argument("--out", required=True)
    cv.add_argument("--scheme", default="compressed-tensors", choices=["compressed-tensors", "fp8"])
    keep_help = (
        "leave these projections in bf16 (repeatable): layers:45-47, proj:down_proj,o_proj or "
        "group:layers.3.mlp.down_proj; a fused group (qkv_proj, gate_up_proj) is kept whole"
    )
    cv.add_argument("--keep-bf16", action="append", default=[], metavar="SPEC", help=keep_help)
    overwrite_help = (
        "write into a folder that already holds files (refused otherwise: vLLM's compiled-graph cache is keyed by the "
        "path, so clear it before serving a rewritten checkpoint)"
    )
    cv.add_argument("--overwrite", action="store_true", help=overwrite_help)
    ex = sub.add_parser("expected", help="the fingerprints the engine should hold")
    ex.add_argument("--source", help="a bf16 directory: after quantizing on load")
    ex.add_argument("--checkpoint", help="a converted directory: after loading it")
    ex.add_argument("--out", required=True)
    mg = sub.add_parser("merge", help="write the FP8 checkpoint of a base with a LoRA adapter merged in fp32")
    mg.add_argument("--source", required=True, help="the bf16 base directory")
    mg.add_argument("--adapter", required=True, help="the PEFT adapter directory")
    mg.add_argument("--out", required=True)
    mg.add_argument(
        "--device", default="cpu", help="where the fp32 merge runs (cuda for the 31B); the FP8 cast is on the CPU"
    )
    mg.add_argument(
        "--null", action="store_true", help="merge an adapter of scale 0: must equal `convert` byte for byte"
    )
    mg.add_argument("--keep-bf16", action="append", default=[], metavar="SPEC", help=keep_help)
    mg.add_argument("--overwrite", action="store_true", help=overwrite_help)
    mg.add_argument(
        "--gptq",
        metavar="CALIBRATION",
        help="round with error compensation (GPTQ) instead of to nearest: a gzip JSONL of requests (state, questions; "
        "the kit's format, no gold) rendered under the header layout; their activations give each layer's Hessian. "
        "The grid and each group's scale are unchanged. Needs transformers and a card for the 12B and the 31B",
    )
    mg.add_argument("--gptq-damp", type=float, default=0.01, help="the share of the mean Hessian diagonal added (0.01)")
    mg.add_argument("--gptq-blocksize", type=int, default=128, help="columns per block of the compensation (128)")
    mg.add_argument("--gptq-max-tokens", type=int, default=6000, help="calibration requests longer are left out")
    mg.add_argument("--gptq-limit", type=int, help="use only the first N calibration requests")
    mg.add_argument(
        "--seed", type=int, default=0, help="the stochastic rounding of the --keep-bf16 tensors (default 0)"
    )
    cp = sub.add_parser("compare", help="the modules at which two fingerprint files differ")
    cp.add_argument("a")
    cp.add_argument("b")
    a = ap.parse_args(argv)
    if a.cmd == "convert":
        r = convert(
            Path(a.source),
            Path(a.out),
            a.scheme,
            progress=lambda m: print(m, flush=True),
            keep=parse_keep(a.keep_bf16),
            overwrite=a.overwrite,
        )
        changed, smallest = r["loader_changed_elements"], r["smallest_amax"]
        print(f"{r['tensors']} tensors; the loader would change {changed} weights; smallest amax {smallest:.4g}")
        return 1 if changed else 0
    if a.cmd == "merge":
        meta = merge(
            Path(a.source),
            Path(a.adapter),
            Path(a.out),
            a.device,
            a.null,
            progress=lambda m: print(m, flush=True),
            keep=parse_keep(a.keep_bf16),
            seed=a.seed,
            overwrite=a.overwrite,
            gptq=(
                Gptq(Path(a.gptq), a.gptq_damp, a.gptq_blocksize, a.gptq_max_tokens, a.gptq_limit) if a.gptq else None
            ),
        )
        differ = meta["written_vs_loaded_differ"]
        print(
            f"{meta['groups']} groups, {meta['adapted_tensors']} adapter pairs applied "
            f"({meta['adapter_layout']} layout, {meta['dropped_k_eq_v_pairs']} v_proj copies of k_proj pairs left "
            "out); "
            f"the loader would change {meta['loader_changed_elements']} weights; written against loaded fingerprints: "
            + ("identical" if not differ else f"{len(differ)} differ: " + ", ".join(differ[:20]))
        )
        return 1 if (meta["loader_changed_elements"] or differ) else 0
    if a.cmd == "expected":
        if bool(a.source) == bool(a.checkpoint):
            ap.error("give --source or --checkpoint")
        fp = expected_online(Path(a.source)) if a.source else expected_loaded(Path(a.checkpoint))
        Path(a.out).write_text(json.dumps(fp, indent=1, sort_keys=True))
        print(f"{len(fp)} modules")
        return 0
    diff = compare(json.loads(Path(a.a).read_text()), json.loads(Path(a.b).read_text()))
    print("identical" if not diff else f"{len(diff)} differ: " + ", ".join(diff[:20]))
    return 1 if diff else 0


if __name__ == "__main__":
    sys.exit(main())
