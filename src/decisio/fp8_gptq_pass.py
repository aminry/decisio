# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""The calibration pass of error-compensating FP8 rounding: layer by layer through the Gemma 4 text model.

`decisio.fp8_gptq` rounds one linear layer given the Hessian of its inputs. A model's layers read what the layers before
them write, so this module walks the text model once, a decoder layer at a time, the way GPTQ does:

    for each layer:
        build it from the (adapter-merged, fp32) weights;
        run the calibration sequences through it, collecting the Hessian of the inputs of its linear layers;
        round each fused group (q/k/v, o, gate/up, down) to FP8 with the group's fixed scale and the compensation;
        write the dequantized weights back and run the sequences through it again: those outputs are the next layer's
        inputs, so the next layer sees the error the rounding made.

Only one layer (and the cached hidden states of the calibration sequences) is on the device at a time, so the 31B's
62 GB of bf16 weights are never all there: they are read from the checkpoint layer by layer.

The decoder layer is the transformers implementation of the model (Gemma4 for the 31B, Gemma4Unified for the 12B),
called as the text model calls it (rotary embeddings and masks per layer type); `attention_k_eq_v` layers have no
v_proj, as in the checkpoint. Groups named in `keep` stay bf16 (--keep-bf16): they are run as merged, not rounded.

    results = gptq_pass(source, merged, sequences, device="cuda")      # {group key: {"codes", "scale", "info"}}

`merged(name)` returns the fp32 weight of a projection with the adapter merged in (hub_fp8.merged32); `sequences` are
lists of token ids (the rendered calibration requests). Nothing here reads a model's data: the sequences are given.
"""

from __future__ import annotations

import importlib
import inspect
import re
from collections import UserDict

import torch

from decisio import fp8_gptq, hub_fp8

LAYER_PREFIX = "model.language_model.layers"
EMBED = "model.language_model.embed_tokens.weight"
FAMILIES = {"gemma4": "Gemma4", "gemma4_unified": "Gemma4Unified"}


class TextStack:
    """The pieces of a Gemma 4 text model the pass needs, from a checkpoint's config.json: the text config, the layer
    class, the embedding, the rotary embedding and the mask functions of the family's transformers module."""

    def __init__(self, source: hub_fp8.Source, device: str, dtype=torch.bfloat16):
        cfg = source.config()
        model_type = cfg.get("model_type")
        if model_type not in FAMILIES:
            raise ValueError(f"the calibration pass reads {sorted(FAMILIES)} checkpoints, not {model_type!r}")
        name, pre = model_type, FAMILIES[model_type]
        self.modeling = importlib.import_module(f"transformers.models.{name}.modeling_{name}")
        config_module = importlib.import_module(f"transformers.models.{name}.configuration_{name}")
        text = dict(cfg.get("text_config") or cfg)
        self.cfg = getattr(config_module, f"{pre}TextConfig")(**{k: v for k, v in text.items() if k != "model_type"})
        self.cfg._attn_implementation = "sdpa"
        self.src, self.device, self.dtype = source, device, dtype
        self.layer_class = getattr(self.modeling, f"{pre}TextDecoderLayer")
        self.takes_per_layer_input = "per_layer_input" in inspect.signature(self.layer_class.forward).parameters
        if self.cfg.hidden_size_per_layer_input:
            raise ValueError("per-layer input embeddings are not part of the Gemma 4 12B and 31B; not supported here")
        embed = getattr(self.modeling, f"{pre}TextScaledWordEmbedding")(
            self.cfg.vocab_size, self.cfg.hidden_size, self.cfg.pad_token_id, embed_scale=self.cfg.hidden_size**0.5
        )
        embed.weight.data = source.get(EMBED).to(dtype)
        self.embed = embed.to(device)
        self.rotary = getattr(self.modeling, f"{pre}TextRotaryEmbedding")(config=self.cfg).to(device)
        self.layer_types = list(self.cfg.layer_types)

    def prepare(self, ids: list[int]):
        """(hidden states [1, T, H], {layer type: the kwargs a layer of that type is called with})."""
        x = torch.tensor([ids], dtype=torch.long, device=self.device)
        hidden = self.embed(x).to(self.dtype)
        position_ids = torch.arange(x.shape[1], device=self.device).unsqueeze(0)
        mask_kwargs = {
            "config": self.cfg,
            "inputs_embeds": hidden,
            "attention_mask": None,
            "past_key_values": None,
            "position_ids": position_ids,
        }
        masks = {
            "full_attention": self.modeling.create_causal_mask(**mask_kwargs),
            "sliding_attention": self.modeling.create_sliding_window_causal_mask(**mask_kwargs),
        }
        kwargs = {}
        for layer_type in set(self.layer_types):
            kwargs[layer_type] = {
                "position_embeddings": self.rotary(hidden, position_ids, layer_type),
                "attention_mask": masks[layer_type],
                "position_ids": position_ids,
                "shared_kv_states": UserDict(),
                "past_key_values": None,
                **({"per_layer_input": None} if self.takes_per_layer_input else {}),
            }
        return hidden, kwargs

    def layer(self, index: int, weight_of) -> torch.nn.Module:
        """Decoder layer `index` with its checkpoint tensors; `weight_of(name)` gives a projection's weight (fp32)."""
        layer = self.layer_class(self.cfg, index).to(device=self.device, dtype=self.dtype)
        prefix = f"{LAYER_PREFIX}.{index}."
        state = {}
        for name in self.src.map:
            if name.startswith(prefix):
                t = weight_of(name) if hub_fp8.PROJ.match(name) else self.src.get(name)
                state[name[len(prefix) :]] = t.to(device=self.device, dtype=self.dtype)
        layer.load_state_dict(state, strict=True)
        return layer.eval()


def tap(hessian):
    """A forward pre-hook that adds a linear layer's input to a Hessian."""

    def hook(_module, args):
        hessian.add(args[0])

    return hook


def _run(layer, hidden, kwargs):
    with torch.no_grad():
        return layer(hidden, **kwargs)


def gptq_pass(
    source: hub_fp8.Source,
    merged,
    sequences: list[list[int]],
    device: str = "cpu",
    keep: hub_fp8.Keep | None = None,
    blocksize: int = 128,
    damp: float = 0.01,
    dtype=torch.bfloat16,
    progress=None,
) -> dict:
    """The compensated FP8 codes of every quantized group of the checkpoint, from `sequences`.

    `merged(name)`: the fp32 weight of projection `name` with the adapter merged in. Returns
    {group key: {"codes": {tensor name: fp8 codes (cpu)}, "scale": fp32 [1], "info": {error_rtn, error_gptq, ...}}}; the
    scale of a group is the one `hub_fp8` uses (fp32(amax over its merged tensors) / 448). Groups in `keep` are not in
    it."""
    if not sequences:
        raise ValueError("no calibration sequences")
    stack = TextStack(source, device, dtype)
    groups = hub_fp8.find_groups(source.map, source.k_eq_v())
    kept = {g.key for g in hub_fp8.kept_groups(groups, keep)} if keep else set()
    by_layer: dict[int, list] = {}
    for g in groups:
        by_layer.setdefault(int(hub_fp8.GROUP_KEY.match(g.key)["idx"]), []).append(g)
    cache = {}  # merged fp32 weights of the layer being worked on, by name

    def weight_of(name):
        if name not in cache:
            cache[name] = merged(name).to(device=device, dtype=torch.float32)
        return cache[name]

    states, kw = [], []
    for ids in sequences:
        hidden, kwargs = stack.prepare(ids)
        states.append(hidden)
        kw.append(kwargs)
    results: dict = {}
    for index in range(stack.cfg.num_hidden_layers):
        cache.clear()
        layer = stack.layer(index, weight_of)
        taps = {
            "qkv": layer.self_attn.q_proj,
            "o": layer.self_attn.o_proj,
            "gate_up": layer.mlp.gate_proj,
            "down": layer.mlp.down_proj,
        }
        hessians = {k: fp8_gptq.Hessian(m.weight.shape[1]) for k, m in taps.items()}
        handles = [m.register_forward_pre_hook(tap(hessians[k])) for k, m in taps.items()]
        layer_type = stack.layer_types[index]
        for hidden, kwargs in zip(states, kw):
            _run(layer, hidden, kwargs[layer_type])
        for h in handles:
            h.remove()
        wanted = {"qkv_proj": "qkv", "o_proj": "o", "gate_up_proj": "gate_up", "down_proj": "down"}
        for g in by_layer.get(index, []):
            if g.key in kept:
                continue
            tensors = {n: weight_of(n) for n in dict.fromkeys(g.fused)}
            amax = torch.stack([t.abs().max() for t in tensors.values()]).max().cpu()
            scale = hub_fp8.scale_of(amax)
            h = hessians[wanted[hub_fp8.GROUP_KEY.match(g.key)["name"]]].matrix()
            codes, info = {}, {"error_rtn": 0.0, "error_gptq": 0.0, "dead_inputs": 0}
            for n in g.names:
                c, i = fp8_gptq.gptq_codes(tensors[n], h, scale, blocksize, damp)
                codes[n] = c.cpu()
                for k in ("error_rtn", "error_gptq", "dead_inputs"):
                    info[k] += i[k]
            results[g.key] = {"codes": codes, "scale": scale, "info": info}
            # the layer now holds the rounded weights: the next layer reads what they make
            for n, c in codes.items():
                module = layer.get_submodule(re.sub(r"\.weight$", "", n[len(f"{LAYER_PREFIX}.{index}.") :]))
                module.weight.data = (c.to(device).to(torch.float32) * scale.to(device)).to(dtype)
        states = [_run(layer, hidden, kwargs[layer_type]) for hidden, kwargs in zip(states, kw)]
        del layer
        if progress:
            progress(f"layer {index + 1}/{stack.cfg.num_hidden_layers}")
    return results
