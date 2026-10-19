# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""The text and image engines of `--image-model` share one card, and their default shares fit it.

The text engine alone takes 0.90 of the card and the image engine asked for 0.50, so `--image-model` with no memory flag
loaded the text engine, left 0.10 of the card and failed in the image engine with vLLM's "free memory is less than
desired" after the long load. The defaults are now the measured pair (0.47 and 0.49), and a pair that cannot fit is
refused before either engine is built.

M1  alone, the text engine takes 0.90; with --image-model the defaults are 0.47 and 0.49; --one-engine keeps the text
    share for the one engine; the other backends are not checked
M2  a pair over 0.96 is refused with both shares and where each came from; the measured pair and up to 0.96 pass
M3  the server refuses a pair that cannot fit before it builds an engine

  uv run pytest -q tests/unit/test_image_memory_shares.py
"""

import pytest

from decisio.serve import vllm_engine as ve


def test_m1_the_defaults_fit_one_card():
    assert ve.resolve_memory_shares(None, None, None) == (0.90, 0.49)  # no image model: the text engine alone
    assert ve.resolve_memory_shares(0.8, None, None) == (0.8, 0.49)
    assert ve.resolve_memory_shares(None, None, "Qwen/image") == (0.47, 0.49)  # the measured pair
    assert ve.resolve_memory_shares(None, None, "Qwen/image", one_engine=True) == (0.90, 0.49)
    assert ve.resolve_memory_shares(0.85, None, "Qwen/image", one_engine=True) == (0.85, 0.49)
    for backend in ("hf", "mlx"):  # no memory shares to fit
        assert ve.resolve_memory_shares(0.9, 0.9, "Qwen/image", backend=backend) == (0.9, 0.9)
    assert ve.IMAGE_PAIR == (0.47, 0.49) and ve.TEXT_SHARE == 0.90


def test_m2_a_pair_that_cannot_fit_is_refused_with_its_origins():
    ok = [(0.47, 0.49), (0.5, 0.46), (0.40, 0.40), (0.47, None), (None, 0.49), (None, None), (0.3, None)]
    for text, image in ok:
        t, i = ve.resolve_memory_shares(text, image, "Qwen/image")
        assert t + i <= 0.96 + 1e-9
    with pytest.raises(ValueError) as e:
        ve.resolve_memory_shares(0.90, None, "Qwen/image")  # the text engine at its own default leaves nothing
    msg = str(e.value)
    assert "0.90, given" in msg and "0.49, default" in msg and "1.39" in msg and "--one-engine" in msg
    for text, image in ((0.5, 0.47), (0.9, 0.9), (None, 0.6), (0.7, 0.3)):
        with pytest.raises(ValueError, match="share the card"):
            ve.resolve_memory_shares(text, image, "Qwen/image")
    with pytest.raises(ValueError, match="0.47, default"):
        ve.resolve_memory_shares(None, 0.6, "Qwen/image")


def test_m3_the_server_refuses_a_pair_that_cannot_fit_before_building_an_engine(capsys, monkeypatch):
    built = []
    monkeypatch.setattr(ve, "LettersEngine", lambda *a, **k: built.append(a) or None)
    monkeypatch.setattr(
        "sys.argv",
        ["decisio", "--base", "qwen3.6-35b-a3b", "--image-model", "Qwen/image", "--gpu-memory-utilization", "0.9"],
    )
    with pytest.raises(SystemExit) as e:
        ve.main()
    assert e.value.code == 2 and built == []
    err = capsys.readouterr().err
    assert "the text engine (0.90, given) and the image engine (0.49, default)" in err
