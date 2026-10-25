# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""A conversion or merge writes into a folder of its own (hub_fp8 convert|merge, hub_bf16 merge).

vLLM keys its compiled-graph cache by the checkpoint's path, so a checkpoint rewritten in place with another layout
reuses the old compiled graph: it crashed in the first forward pass ("not enough values to unpack") on a card, and with
equal argument counts it would run the wrong graph. So a folder that already holds files is refused unless
`--overwrite`.

O1  a new folder and an empty one are fine; a folder with files is refused, by every command, with the reason
O2  --overwrite writes into it (and the files of the new checkpoint are the ones there)
O3  the command lines refuse and accept the same way

  uv run pytest -q tests/unit/test_merge_output_folder.py
"""

import pytest
from test_hub_fp8 import checkpoint
from test_hub_fp8_merge import adapter, targets

from decisio import hub_bf16, hub_fp8


def test_o1_a_folder_with_files_is_refused_by_every_command(tmp_path):
    src, t = checkpoint(tmp_path)
    ad = adapter(tmp_path, targets(t))
    new, empty, used = tmp_path / "new", tmp_path / "empty", tmp_path / "used"
    empty.mkdir()
    used.mkdir()
    (used / "model-00001-of-00001.safetensors").write_bytes(b"old")
    assert hub_fp8.convert(src, new)["tensors"] > 0
    assert hub_fp8.convert(src, empty)["tensors"] > 0
    for call in (
        lambda: hub_fp8.convert(src, used),
        lambda: hub_fp8.merge(src, ad, used),
        lambda: hub_bf16.merge(src, ad, used),
        lambda: hub_fp8.convert(src, new),  # the second conversion into the first's folder
    ):
        with pytest.raises(ValueError, match="is not empty") as e:
            call()
        assert "compiled-graph cache" in str(e.value) and "--overwrite" in str(e.value)
    assert (used / "model-00001-of-00001.safetensors").read_bytes() == b"old"  # nothing was touched


def test_o2_overwrite_writes_into_it(tmp_path):
    src, t = checkpoint(tmp_path)
    ad = adapter(tmp_path, targets(t))
    out = tmp_path / "out"
    hub_fp8.convert(src, out)
    hub_fp8.convert(src, out, overwrite=True, keep=hub_fp8.parse_keep(["layers:1"]))
    assert "kept_bf16" in (out / "fp8_conversion.json").read_text()
    hub_fp8.merge(src, ad, tmp_path / "m")
    hub_fp8.merge(src, ad, tmp_path / "m", overwrite=True)
    hub_bf16.merge(src, ad, tmp_path / "b")
    hub_bf16.merge(src, ad, tmp_path / "b", overwrite=True)


def test_o3_the_command_lines(tmp_path, capsys):
    src, t = checkpoint(tmp_path)
    ad = adapter(tmp_path, targets(t))
    out = str(tmp_path / "out")
    assert hub_fp8.main(["convert", "--source", str(src), "--out", out]) == 0
    with pytest.raises(ValueError, match="is not empty"):
        hub_fp8.main(["convert", "--source", str(src), "--out", out])
    assert hub_fp8.main(["convert", "--source", str(src), "--out", out, "--overwrite"]) == 0
    with pytest.raises(ValueError, match="is not empty"):
        hub_fp8.main(["merge", "--source", str(src), "--adapter", str(ad), "--out", out])
    assert hub_bf16.main(["merge", "--source", str(src), "--adapter", str(ad), "--out", str(tmp_path / "b")]) == 0
    with pytest.raises(ValueError, match="is not empty"):
        hub_bf16.main(["merge", "--source", str(src), "--adapter", str(ad), "--out", str(tmp_path / "b")])
    assert (
        hub_bf16.main(
            ["merge", "--source", str(src), "--adapter", str(ad), "--out", str(tmp_path / "b"), "--overwrite"]
        )
        == 0
    )
