# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""Weight-name rules shared by decisio's model classes (no vLLM or torch import: usable anywhere)."""
from __future__ import annotations

from collections.abc import Iterable, Iterator


def is_vision_weight(name: str) -> bool:
    """A tensor of the checkpoint's vision tower (`model.visual.*` in the official checkpoint; the rule
    `decisio.serve.make_text_only` uses to build the text-only view)."""
    return name.startswith("visual.") or ".visual." in name


def without_vision(weights: Iterable[tuple[str, object]]) -> Iterator[tuple[str, object]]:
    return ((name, tensor) for name, tensor in weights if not is_vision_weight(name))
