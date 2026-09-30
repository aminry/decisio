# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""The head's output-layer rows load from what `--model` names: a local directory or a Hugging Face repo id (the
quickstart's `--model Qwen/Qwen3.6-35B-A3B-FP8`), sharded with an index or as one file.

    uv run pytest -q tests/unit/test_lm_head.py
"""

import json

import pytest
import torch
from safetensors.torch import save_file

from decisio.serve.hidden_engine import HiddenEngine

W = torch.arange(12, dtype=torch.float32).reshape(4, 3)


def checkpoint(root, sharded):
    root.mkdir(parents=True, exist_ok=True)
    if sharded:
        save_file({"model.embed.weight": torch.zeros(2, 3)}, str(root / "model-00001-of-00002.safetensors"))
        save_file({"lm_head.weight": W}, str(root / "model-00002-of-00002.safetensors"))
        index = {
            "weight_map": {
                "model.embed.weight": "model-00001-of-00002.safetensors",
                "lm_head.weight": "model-00002-of-00002.safetensors",
            }
        }
        (root / "model.safetensors.index.json").write_text(json.dumps(index))
    else:
        save_file({"lm_head.weight": W}, str(root / "model.safetensors"))
    return root


@pytest.mark.parametrize("sharded", [True, False])
def test_local_directory(tmp_path, sharded):
    assert torch.equal(HiddenEngine._load_lm_head(str(checkpoint(tmp_path / "ckpt", sharded))), W)


@pytest.mark.parametrize("sharded", [True, False])
def test_repo_id_fetches_only_what_it_needs(tmp_path, monkeypatch, sharded):
    import huggingface_hub
    from huggingface_hub.errors import EntryNotFoundError

    cache = checkpoint(tmp_path / "cache", sharded)
    fetched = []

    def fake_download(repo_id, filename, **kw):
        assert repo_id == "Qwen/Qwen3.6-35B-A3B-FP8"
        fetched.append(filename)
        if not (cache / filename).exists():
            raise EntryNotFoundError(f"{filename} not in {repo_id}")
        return str(cache / filename)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)
    assert torch.equal(HiddenEngine._load_lm_head("Qwen/Qwen3.6-35B-A3B-FP8"), W)
    want = (
        ["model.safetensors.index.json", "model-00002-of-00002.safetensors"]
        if sharded
        else ["model.safetensors.index.json", "model.safetensors"]
    )
    assert fetched == want  # never the other shards


def test_clear_errors(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError, match="neither model.safetensors.index.json nor model.safetensors"):
        HiddenEngine._load_lm_head(str(empty))
    tied = tmp_path / "tied"
    tied.mkdir()
    save_file({"model.embed.weight": W}, str(tied / "model.safetensors"))
    with pytest.raises(ValueError, match="no lm_head.weight"):
        HiddenEngine._load_lm_head(str(tied))
    (tied / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"model.embed.weight": "model.safetensors"}})
    )
    with pytest.raises(ValueError, match="index.json has no lm_head.weight"):
        HiddenEngine._load_lm_head(str(tied))
    missing = checkpoint(tmp_path / "missing", sharded=True)
    (missing / "model-00002-of-00002.safetensors").unlink()
    with pytest.raises(FileNotFoundError, match="the shard the index names"):
        HiddenEngine._load_lm_head(str(missing))
