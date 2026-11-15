# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""The vision weights of an MLX conversion (`--backend mlx --images on`): in the conversion, or in the base's add-on.

An MLX conversion made for text drops the vision tensors (the 12B's vision embedder and its projection, the 31B's
vision tower), so it cannot read pictures by itself. Each base has one small versioned repository, its vision add-on
(`Family.vision_addon_repo`): one `vision.safetensors` with exactly those tensors, copied bit for bit from the base's
own checkpoint in bf16. It is shared by the base and every model tuned from it, since tuning leaves the vision tensors
as they are. A conversion that has the tensors (the base's bf16 checkpoint, an MLX conversion that kept them) needs no
add-on.

The add-on is found, in this order, at `--vision-weights` (a file or a folder with `vision.safetensors`: offline use
and tests), or in the Hugging Face cache, downloaded once from the repository. It is pinned by content: the family
carries the file's sha256, which is checked on every start, and a file that differs is refused. Nothing is added to a
model repository's `decision_config.json` (a published repository's config must equal the profile).

Merging. mlx-vlm's loader reads every `*.safetensors` in a directory and its `sanitize` strips the leading `model.`
from a name, so the add-on's names (`model.vision_embedder.*`, `model.embed_vision.*`, and `model.vision_tower.*` for
the 31B) are the names it expects: no renaming is needed (a test checks the names and shapes against the architecture).
The merged model is therefore built without copying a weight: a small directory under the Hugging Face home holds links
to the conversion's files and to the add-on, and a `config.json` that is the conversion's with the base's
`vision_config` added (and audio off). mlx-vlm loads that directory. Quantised language weights stay as they are and
the vision tensors stay bf16, since mlx-vlm quantises only the modules that have scales in the weights.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from decisio.serve.mlx_memory import MemoryRefused

ADDON_FILE = "vision.safetensors"
# the model types of the architectures that read images on a Mac, by base key
ARCHITECTURES = {"gemma-4-12b": "gemma4_unified", "gemma-4-31b": "gemma4"}
VISION_KEYS = ("vision_embedder", "vision_tower", "embed_vision")


class ImagesRefused(MemoryRefused):
    """The conversion cannot read images (the server refuses to start, with the reason, as it does for memory)."""


def weight_names(model_dir) -> set[str]:
    """The tensor names of a conversion: its index file's, or the headers of its safetensors files."""
    d = Path(model_dir)
    index = d / "model.safetensors.index.json"
    if index.exists():
        return set(json.loads(index.read_text())["weight_map"])
    return safetensors_names(d.glob("*.safetensors"))


def safetensors_names(files) -> set[str]:
    names = set()
    for f in sorted(files):
        with open(f, "rb") as fh:
            n = int.from_bytes(fh.read(8), "little")
            names |= {k for k in json.loads(fh.read(n)) if k != "__metadata__"}
    return names


def has_vision(model_dir) -> bool:
    """Whether the conversion carries vision tensors and says so in its config."""
    d = Path(model_dir)
    config = json.loads((d / "config.json").read_text()) if (d / "config.json").exists() else {}
    names = weight_names(d)
    return bool(config.get("vision_config")) and any(k in n for n in names for k in VISION_KEYS)


def architecture_refusal(model_dir, family) -> str | None:
    """Why the conversion is not the architecture that reads this base's pictures, or None."""
    d = Path(model_dir)
    config = json.loads((d / "config.json").read_text()) if (d / "config.json").exists() else {}
    kind, want = config.get("model_type"), ARCHITECTURES.get(family.key)
    if want is None:
        return f"--images on on a Mac is for the Gemma 4 bases, not {family.key}"
    if kind != want:
        return (
            f"--images on: this conversion ({d.name}) has model_type {kind!r}, and the {family.key} base reads "
            f"pictures as {want!r}"
        )
    return None


def sha256_of(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def find_addon(family, vision_weights=None) -> Path:
    """The add-on file of `family`: `--vision-weights` (a file, or a folder holding `vision.safetensors`), else the
    family's repository from the Hugging Face cache (downloaded once). Raises ImagesRefused when there is none."""
    if not family.vision_addon_sha256:
        raise ImagesRefused(
            f"--images on: this conversion has no vision weights, and the {family.key} base has no vision add-on to "
            "supply them"
        )
    if vision_weights:
        p = Path(vision_weights)
        f = p / ADDON_FILE if p.is_dir() else p
        if not f.is_file():
            raise ImagesRefused(f"--vision-weights {vision_weights}: no {ADDON_FILE} there")
        return f
    from huggingface_hub import hf_hub_download

    try:
        return Path(hf_hub_download(family.vision_addon_repo, ADDON_FILE))
    except Exception as e:  # noqa: BLE001  (offline, not cached, or a repository that is not there)
        raise ImagesRefused(
            f"--images on: this conversion has no vision weights, and the vision add-on {family.vision_addon_repo} "
            f"could not be fetched ({type(e).__name__}). Download it once with network access, or give "
            f"--vision-weights PATH, the {ADDON_FILE} of that repository or a folder holding it"
        ) from e


def verify_addon(path, family) -> str:
    """The add-on's sha256, which must be the family's; ImagesRefused with both when it is not."""
    got = sha256_of(path)
    if got != family.vision_addon_sha256:
        raise ImagesRefused(
            f"--images on: the vision add-on {path} is not the one this decisio release is pinned to for "
            f"{family.key} (sha256 {got}, expected {family.vision_addon_sha256}); use the file of "
            f"{family.vision_addon_repo}"
        )
    return got


def merged_dir(model_dir, family, addon, config_of_base=None, root=None) -> Path:
    """A directory mlx-vlm loads as one conversion with the vision weights: links to every file of `model_dir` but its
    config, a link to the add-on, and the conversion's config with the base's `vision_config` (audio off). Nothing
    large is copied. `config_of_base`: the base's config.json as a dict (None: read from the base's checkpoint)."""
    d = Path(model_dir)
    config = json.loads((d / "config.json").read_text())
    if config_of_base is None:
        from decisio.families import read_config

        config_of_base = read_config(family.model, family.revision)
    vision = config_of_base.get("vision_config")
    if not vision:
        raise ImagesRefused(
            f"--images on: the vision_config of {family.model} could not be read (offline, and not in the cache); "
            "fetch its config.json once with network access"
        )
    config = {**config, "vision_config": vision, "audio_config": None}
    if "vision_soft_tokens_per_image" in config_of_base:
        config.setdefault("vision_soft_tokens_per_image", config_of_base["vision_soft_tokens_per_image"])
    if root is None:
        home = os.environ.get("HF_HOME") or str(Path.home() / ".cache" / "huggingface")
        root = Path(os.environ.get("DECISIO_MLX_VISION_DIR") or Path(home) / "decisio-mlx-vision")
    key = hashlib.sha256(
        json.dumps([str(d.resolve()), str(Path(addon).resolve()), config], sort_keys=True).encode()
    ).hexdigest()[:20]
    out = Path(root) / key
    out.mkdir(parents=True, exist_ok=True)
    for f in d.iterdir():
        if f.name in ("config.json", ADDON_FILE) or f.name.startswith("."):
            continue
        link = out / f.name
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(f.resolve())
    link = out / ADDON_FILE
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(Path(addon).resolve())
    (out / "config.json").write_text(json.dumps(config, indent=2))
    return out


def prepare(model_dir, family, vision_weights=None) -> tuple[Path, dict]:
    """(the directory to load, what /health says of the vision weights). A conversion that has the tensors is loaded as
    it is; one that has not gets the base's add-on, found, checked against the pin and merged."""
    why = architecture_refusal(model_dir, family)
    if why:
        raise ImagesRefused(why)
    if has_vision(model_dir):
        if vision_weights:
            raise ImagesRefused(
                f"--vision-weights {vision_weights}: this conversion ({Path(model_dir).name}) already has the vision "
                "weights; leave the flag out"
            )
        return Path(model_dir), {"source": "conversion"}
    addon = find_addon(family, vision_weights)
    sha = verify_addon(addon, family)
    return merged_dir(model_dir, family, addon), {
        "source": "add-on",
        "repository": family.vision_addon_repo,
        "file": str(addon),
        "sha256": sha,
    }
