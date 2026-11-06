# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""The students' files from a directory (`DECISIO_STUDENTS_DIR`, the image with them built in: docker/Dockerfile.table;
docs/running.md): a pinned file under <dir>/<repo>/<revision>/<file> is read from there, before the Hugging Face cache
or the Hub, and checked against its pinned sha256 as any other.

D1  with the files in the directory and the Hub out of reach, the static student loads from the directory
D2  a file in the directory that is not the pinned one is refused, never served
D3  a directory without the file: the Hub (or its cache) as before; no variable: as before
D4  pinned_files lists every pinned student file (the encoder's only when asked), as the image fetches them
D5  docker/fetch_students.py fetches, checks and lays out the files where student_file reads them, and refuses a
    file whose sha256 is not the pin; it reads a model card's licence from its front matter

    uv run pytest -q tests/unit/test_fast_table_students_dir.py
"""

import importlib.util
import shutil
from pathlib import Path

import numpy as np
import pytest
from test_fast_table_kinds import tiny_static

from decisio.serve import fast_table as FT


def offline(*a, **k):
    raise OSError("no route to host")


def laid_out(src: Path, root: Path) -> Path:
    """The tiny static model's files under root, laid out as the image lays them out."""
    spec = FT.STATIC_MODELS["base"]
    d = root / spec["repo"] / spec["revision"]
    d.mkdir(parents=True)
    for f in spec["files_sha256"]:
        shutil.copy(src / f, d / f)
    return d


def test_d1_the_static_student_loads_from_the_directory_with_the_hub_out_of_reach(tmp_path, monkeypatch):
    import huggingface_hub

    (tmp_path / "src").mkdir()
    E = tiny_static(tmp_path / "src", monkeypatch)
    laid_out(tmp_path / "src", tmp_path / "students")
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", offline)
    monkeypatch.setenv(FT.STUDENTS_DIR_ENV, str(tmp_path / "students"))
    m = FT.StaticEncoder.load("base")
    got = m.encode(["a b"])[0]
    mean = E[[1, 2]].mean(axis=0)
    assert np.allclose(got, mean / np.linalg.norm(mean))


def test_d2_a_file_in_the_directory_that_is_not_the_pinned_one_is_refused(tmp_path, monkeypatch):
    (tmp_path / "src").mkdir()
    tiny_static(tmp_path / "src", monkeypatch)
    d = laid_out(tmp_path / "src", tmp_path / "students")
    (d / "tokenizer.json").write_text((d / "tokenizer.json").read_text().replace("[UNK]", "[UNX]"))
    monkeypatch.setenv(FT.STUDENTS_DIR_ENV, str(tmp_path / "students"))
    with pytest.raises(FT.TableUnavailable, match="not the pinned"):
        FT.StaticEncoder.load("base")


def test_d3_without_the_file_or_the_variable_the_hub_as_before(tmp_path, monkeypatch):
    (tmp_path / "src").mkdir()
    tiny_static(tmp_path / "src", monkeypatch)  # the Hub stand-in serves the files from src
    monkeypatch.setenv(FT.STUDENTS_DIR_ENV, str(tmp_path / "empty"))
    assert FT.StaticEncoder.load("base").encode(["a"]).shape == (1, 8)
    monkeypatch.delenv(FT.STUDENTS_DIR_ENV)
    spec = FT.STATIC_MODELS["base"]
    assert FT.student_file(spec["repo"], "tokenizer.json", spec["revision"]) == str(tmp_path / "src" / "tokenizer.json")


def test_d4_pinned_files_lists_every_pinned_student_file():
    files = FT.pinned_files()
    want = {(s["repo"], s["revision"], f, h) for s in FT.STATIC_MODELS.values() for f, h in s["files_sha256"].items()}
    assert {(x["repo"], x["revision"], x["file"], x["sha256"]) for x in files} == want
    assert {x["repo"] for x in files} == {"minishlab/potion-base-8M", "minishlab/potion-multilingual-128M"}
    enc = FT.pinned_files(encoder=True)
    assert len(enc) == len(files) + 2
    assert {(x["file"], x["sha256"]) for x in enc if x["repo"] == FT.ENCODER} == {
        (FT.ONNX_FILE, FT.ONNX_SHA256),
        (FT.TOKENIZER_FILE, FT.TOKENIZER_SHA256),
    }


def fetcher():
    path = Path(__file__).resolve().parents[2] / "docker" / "fetch_students.py"
    spec = importlib.util.spec_from_file_location("fetch_students", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_d5_the_fetch_script_lays_out_the_files_where_student_file_reads_them(tmp_path, monkeypatch):
    import huggingface_hub

    (tmp_path / "src").mkdir()
    tiny_static(tmp_path / "src", monkeypatch)
    pins = [x for x in FT.pinned_files() if x["repo"] == FT.STATIC_MODELS["base"]["repo"]]
    fs = fetcher()
    out = tmp_path / "students"
    fs.fetch(pins, out)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", offline)
    monkeypatch.setenv(FT.STUDENTS_DIR_ENV, str(out))
    assert FT.StaticEncoder.load("base").encode(["a"]).shape == (1, 8)
    bad = [{**pins[0], "sha256": "0" * 64}]
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda repo, f, **k: str(tmp_path / "src" / f))
    with pytest.raises(SystemExit, match="not the pinned"):
        fs.fetch(bad, tmp_path / "other")
    assert fs.card_licence("---\nlibrary_name: model2vec\nlicense: mit\ntags:\n- x\n---\n# card") == "mit"
    assert fs.card_licence("# no front matter\nlicense: mit") is None
