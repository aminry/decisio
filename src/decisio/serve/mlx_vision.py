# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""The vision weights of an MLX conversion (`--backend mlx --images on`): in the conversion, or in the base's add-on.

An MLX conversion made for text drops the vision tensors (the 12B's vision embedder and its projection, the 31B's
vision tower), so it cannot read pictures by itself. Each base has one small versioned repository, its vision add-on
(`Family.vision_addon_repo`): one `vision.safetensors` with exactly those tensors, copied bit for bit from the base's
own checkpoint in bf16. It is shared by the base and every model tuned from it, since tuning leaves the vision tensors
as they are. A conversion that has the tensors (the base's bf16 checkpoint, an MLX conversion that kept them) needs no
add-on.

The add-on is two files: `vision.safetensors` and `vision_config.json` (the base's vision settings, so a start needs no
copy of the base's checkpoint). It is found at `--vision-weights` (a folder holding both, or the weights file with the
config beside it: offline use and tests), or else in the Hugging Face cache, downloaded once from the repository at the
revision the family pins. While a family's revision is empty (the repository is not published yet) a download is
refused and nothing is read from the hub: an unpinned head is never taken. Both files are pinned by content too: the
family carries their sha256, checked on every start, and a file that differs is refused. Nothing is added to a model
repository's `decision_config.json` (a published repository's config must equal the profile).

Merging. mlx-vlm's loader reads the files of the directory's index (every `*.safetensors` when there is none), and its
`sanitize` strips the leading `model.` from a name, so the add-on's names (`model.vision_embedder.*`,
`model.embed_vision.*`, and `model.vision_tower.*` for the 31B) are the names it expects: no renaming is needed (tests
check the names and shapes against the architecture). The merged model is therefore built without copying a weight: a
small directory under the Hugging Face home holds links to the conversion's files and to the add-on, an index that lists
the add-on's tensors too, and a `config.json` that is the conversion's with the add-on's `vision_config` added (and
audio off). mlx-vlm loads that directory. Quantised language weights stay as they are and the vision tensors stay bf16,
since mlx-vlm quantises only the modules that have scales in the weights.
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


CONFIG_FILE = "vision_config.json"


def find_addon(family, vision_weights=None) -> tuple[Path, Path]:
    """The add-on's files of `family`, (`vision.safetensors`, `vision_config.json`): at `--vision-weights` (a folder
    holding both, or the weights file with the config beside it), else the family's repository at its pinned revision,
    from the Hugging Face cache (downloaded once). Raises ImagesRefused when there is none; a repository whose revision
    is not pinned yet is never read."""
    if not family.vision_addon_sha256:
        raise ImagesRefused(
            f"--images on: this conversion has no vision weights, and the {family.key} base has no vision add-on to "
            "supply them"
        )
    if vision_weights:
        p = Path(vision_weights)
        f = p / ADDON_FILE if p.is_dir() else p
        c = f.parent / CONFIG_FILE
        if not f.is_file():
            raise ImagesRefused(f"--vision-weights {vision_weights}: no {ADDON_FILE} there")
        if not c.is_file():
            raise ImagesRefused(
                f"--vision-weights {vision_weights}: no {CONFIG_FILE} beside {f.name}; the add-on is the two files "
                f"of {family.vision_addon_repo}"
            )
        return f, c
    if not family.vision_addon_revision:
        raise ImagesRefused(
            f"--images on: the vision add-on for {family.key} is not published yet: pass --vision-weights PATH "
            f"(the folder of {family.vision_addon_repo}, or its {ADDON_FILE} with {CONFIG_FILE} beside it)"
        )
    from huggingface_hub import hf_hub_download

    try:
        return tuple(
            Path(hf_hub_download(family.vision_addon_repo, name, revision=family.vision_addon_revision))
            for name in (ADDON_FILE, CONFIG_FILE)
        )
    except Exception as e:  # noqa: BLE001  (offline, not cached, or a repository that is not there)
        raise ImagesRefused(
            f"--images on: this conversion has no vision weights, and the vision add-on {family.vision_addon_repo} "
            f"could not be fetched ({type(e).__name__}). Download it once with network access, or give "
            f"--vision-weights PATH, a folder holding its {ADDON_FILE} and {CONFIG_FILE}"
        ) from e


def verify_addon(path, family, config=None) -> str:
    """The add-on's sha256, which must be the family's, and its config's when given; ImagesRefused with both when one
    is not."""
    got = sha256_of(path)
    if got != family.vision_addon_sha256:
        raise ImagesRefused(
            f"--images on: the vision add-on {path} is not the one this decisio release is pinned to for "
            f"{family.key} (sha256 {got}, expected {family.vision_addon_sha256}); use the file of "
            f"{family.vision_addon_repo}"
        )
    if config is not None:
        got_c = sha256_of(config)
        if got_c != family.vision_addon_config_sha256:
            raise ImagesRefused(
                f"--images on: {config} is not the {CONFIG_FILE} this decisio release is pinned to for {family.key} "
                f"(sha256 {got_c}, expected {family.vision_addon_config_sha256}); use the file of "
                f"{family.vision_addon_repo}"
            )
    return got


def merged_dir(model_dir, family, addon, addon_config, root=None) -> Path:
    """A directory mlx-vlm loads as one conversion with the vision weights: links to every file of `model_dir` but its
    config, a link to the add-on, and the conversion's config with the add-on's `vision_config` (audio off). Nothing
    large is copied. `addon_config`: the add-on's `vision_config.json` as a dict."""
    d = Path(model_dir)
    config = json.loads((d / "config.json").read_text())
    vision = addon_config.get("vision_config")
    if not vision:
        raise ImagesRefused(f"--images on: the add-on's {CONFIG_FILE} has no vision_config")
    config = {**config, "vision_config": vision, "audio_config": None}
    if "vision_soft_tokens_per_image" in addon_config:
        config.setdefault("vision_soft_tokens_per_image", addon_config["vision_soft_tokens_per_image"])
    if root is None:
        home = os.environ.get("HF_HOME") or str(Path.home() / ".cache" / "huggingface")
        root = Path(os.environ.get("DECISIO_MLX_VISION_DIR") or Path(home) / "decisio-mlx-vision")
    key = hashlib.sha256(
        json.dumps([str(d.resolve()), str(Path(addon).resolve()), config], sort_keys=True).encode()
    ).hexdigest()[:20]
    out = Path(root) / key
    out.mkdir(parents=True, exist_ok=True)
    for f in d.iterdir():
        if f.name in ("config.json", "model.safetensors.index.json", ADDON_FILE) or f.name.startswith("."):
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
    # mlx-vlm's loader reads the files an index names and ignores any other: with an index, the add-on must be in it
    index = d / "model.safetensors.index.json"
    if index.exists():
        merged = json.loads(index.read_text())
        merged["weight_map"] = {
            **merged["weight_map"],
            **{k: ADDON_FILE for k in safetensors_names([Path(addon)])},
        }
        (out / "model.safetensors.index.json").write_text(json.dumps(merged, indent=2))
    else:
        link = out / "model.safetensors.index.json"
        if link.is_symlink() or link.exists():
            link.unlink()
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
    addon, addon_config = find_addon(family, vision_weights)
    sha = verify_addon(addon, family, addon_config)
    out = merged_dir(model_dir, family, addon, json.loads(Path(addon_config).read_text()))
    return out, {
        "source": "add-on",
        "repository": family.vision_addon_repo,
        **({"revision": family.vision_addon_revision} if family.vision_addon_revision else {}),
        "file": str(addon),
        "sha256": sha,
    }
