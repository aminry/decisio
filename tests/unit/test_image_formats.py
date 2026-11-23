# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""The image route opens only PNG, JPEG and WebP (decisio.serve.image_engine.load_image).

I1  PNG, JPEG and WebP load, as RGB
I2  another format (GIF, BMP) is refused as an unsupported format
I3  an EPS file is refused before Pillow parses it, so Ghostscript is never run on it

  uv run pytest -q tests/unit/test_image_formats.py
"""

import io

import pytest

PIL = pytest.importorskip("PIL")
from PIL import EpsImagePlugin, Image  # noqa: E402

from decisio.serve.image_engine import load_image  # noqa: E402


def _bytes(fmt: str) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), (200, 10, 10)).save(buf, format=fmt)
    return buf.getvalue()


@pytest.mark.parametrize("fmt", ["PNG", "JPEG", "WEBP"])
def test_i1_the_three_formats_load(fmt):
    img = load_image(_bytes(fmt))
    assert img.mode == "RGB" and img.size == (8, 8)


@pytest.mark.parametrize("fmt", ["GIF", "BMP"])
def test_i2_other_formats_are_refused(fmt):
    with pytest.raises(ValueError, match="unsupported image format"):
        load_image(_bytes(fmt))


def test_i3_an_eps_file_never_reaches_ghostscript(monkeypatch):
    def no_ghostscript(*a, **k):
        raise AssertionError("Ghostscript was called")

    monkeypatch.setattr(EpsImagePlugin, "Ghostscript", no_ghostscript)
    eps = b"%!PS-Adobe-3.0 EPSF-3.0\n%%BoundingBox: 0 0 8 8\n0 0 moveto 8 8 lineto stroke\nshowpage\n%%EOF\n"
    with pytest.raises(ValueError, match="unsupported image format"):
        load_image(eps)
