# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""LoRA adapters whose tensors are named for the other layout than the served class's.

vLLM maps an adapter's tensor names onto the served class's modules; names it cannot map load as a silent no-op. The two
layouts of a Gemma or Qwen adapter are the same weights under different names:

  text         `base_model.model.model.layers.N...`
               the text model's own (PEFT's original names); decisio's text classes (`--model-class hidden-readout` and
               `text-only`) need these
  multimodal   `base_model.model.model.language_model.layers.N...`
               the multimodal checkpoint's, which most published adapters carry; vLLM's own `*ForConditionalGeneration`
               class (`--model-class view`) needs these

An adapter in the other layout is **mapped, not refused**: a copy of the adapter folder is written to a temporary
directory for the life of the process, with the names in the safetensors header changed and the tensor bytes streamed
unchanged (`adapter_config.json` and the other small files are copied as they are), and the server serves that copy.
The mapping is reported (`/health`, `adapter_names`). An adapter that is in neither layout, or mixes them, is still
refused, with the reason, and before the engine is built (the engine load takes minutes).

Standard library only.
"""

from __future__ import annotations

import atexit
import hashlib
import json
import shutil
import tempfile
from pathlib import Path

WEIGHTS = "adapter_model.safetensors"
TEXT_PREFIX = "base_model.model.model.layers."
MULTIMODAL_INFIX = ".language_model.layers."
TEXT_INFIX = ".layers."
MULTIMODAL_PREFIX = "base_model.model.model.language_model.layers."
MAPPING = {
    "text": f"`{MULTIMODAL_INFIX}` -> `{TEXT_INFIX}` (the multimodal checkpoint's names mapped to the text class)",
    "multimodal": f"`{TEXT_PREFIX}` -> `{MULTIMODAL_PREFIX}` (the text names mapped to the multimodal class)",
}


def read_header(path) -> tuple[dict, int]:
    """(the safetensors header as a dict, the byte offset at which the tensor data starts)."""
    with open(path, "rb") as f:
        n = int.from_bytes(f.read(8), "little")
        header = json.loads(f.read(n))
    return header, 8 + n


def form_of(keys) -> str | None:
    """ "text", "multimodal", or None (neither, or a mixture) for an adapter's tensor names."""
    keys = list(keys)
    if keys and all(k.startswith(TEXT_PREFIX) for k in keys):
        return "text"
    if keys and all(MULTIMODAL_INFIX in k for k in keys):
        return "multimodal"
    return None


def wanted_form(arch: str | None) -> str | None:
    """The form the served class needs, from its architecture name; None where the class is not known to need one."""
    arch = arch or ""
    if arch.endswith("ForCausalLM"):
        return "text"
    if arch.endswith("ForConditionalGeneration"):
        return "multimodal"
    return None


def rename(key: str, to: str) -> str:
    if to == "text":
        return key.replace(MULTIMODAL_INFIX, TEXT_INFIX, 1)
    return key.replace(TEXT_PREFIX, MULTIMODAL_PREFIX, 1)


def rewrite(src: Path, dst: Path, to: str) -> int:
    """Write a copy of the adapter folder `src` into `dst` with the tensor names in the other layout; the tensor bytes
    are copied unchanged. Returns the number of tensors renamed."""
    dst.mkdir(parents=True, exist_ok=True)
    header, data_start = read_header(src / WEIGHTS)
    meta = header.pop("__metadata__", None)
    new = {rename(k, to): v for k, v in header.items()}
    if len(new) != len(header):
        raise ValueError(f"{src}: two tensor names become the same name in the {to} layout")
    if meta is not None:
        new = {"__metadata__": meta, **new}
    blob = json.dumps(new, separators=(",", ":")).encode()
    blob += b" " * (-len(blob) % 8)  # the tensor data starts on an 8-byte boundary, as safetensors writes it
    with open(src / WEIGHTS, "rb") as s, open(dst / WEIGHTS, "wb") as d:
        d.write(len(blob).to_bytes(8, "little"))
        d.write(blob)
        s.seek(data_start)
        shutil.copyfileobj(s, d, 16 << 20)
    for f in src.iterdir():
        if f.is_file() and f.name != WEIGHTS:
            shutil.copy2(f, dst / f.name)
    return len(header)


def identity(folder) -> dict:
    """Who an adapter is: the sha256 and size of its weights file as given (before any name mapping) and the sha256 of
    its `adapter_config.json`. Two folders with the same weights and rank are the same adapter; it goes into the
    serving-stack fingerprint (decisio.serve.stack)."""
    src = Path(folder)
    out = {"weights_sha256": _sha256(src / WEIGHTS), "weights_bytes": (src / WEIGHTS).stat().st_size}
    if (src / "adapter_config.json").is_file():
        out["config_sha256"] = _sha256(src / "adapter_config.json")
    return out


def _sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(16 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def prepare(adapters: dict, arch: str | None, workdir=None) -> tuple[dict, dict]:
    """(the adapter folders to serve, what was done to each), before the engine is built.

    `adapters` is {name: folder}; `arch` the architecture the engine will serve (a decisio class, or the checkpoint's
    own). Each folder is checked, and mapped to the served class's layout when it is in the other one. Raises
    ValueError, naming the adapter and the reason, for an adapter with no weights file or in neither layout."""
    want = wanted_form(arch)
    paths, report = {}, {}
    for name, folder in adapters.items():
        src = Path(folder)
        if not (src / WEIGHTS).is_file():
            raise ValueError(f"adapter {name}: no {WEIGHTS} in {src}")
        header, _ = read_header(src / WEIGHTS)
        keys = [k for k in header if k != "__metadata__"]
        have = form_of(keys)
        if want is None:  # a class not known to need one: served as given
            paths[name], report[name] = (
                str(src),
                {"names": have or "unrecognised", "mapped": False, "tensors": len(keys)},
            )
            continue
        if have is None:
            raise ValueError(
                f"adapter {name}: its tensor names are neither PEFT's original `{TEXT_PREFIX}...` nor the multimodal "
                f"checkpoint's `...{MULTIMODAL_INFIX}...`, or they mix the two; vLLM would load it as a silent no-op "
                f"on {arch or 'this class'}"
            )
        if have == want:
            paths[name], report[name] = str(src), {"names": have, "mapped": False, "tensors": len(keys)}
            continue
        base = Path(workdir) if workdir else Path(tempfile.mkdtemp(prefix="decisio-adapter-"))
        if not workdir:
            atexit.register(shutil.rmtree, base, True)
        dst = base / name
        rewrite(src, dst, want)
        paths[name] = str(dst)
        report[name] = {
            "names": have,
            "mapped": True,
            "mapped_to": want,
            "mapping": MAPPING[want],
            "tensors": len(keys),
        }
    return paths, report
