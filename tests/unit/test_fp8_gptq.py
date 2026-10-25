# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""Error-compensating rounding to FP8 with a fixed scale (decisio.fp8_gptq), on a tiny layer:

G1  the Hessian: the running mean of 2 x x^T over batches equals the one-shot value, and rows are counted
G2  the codes are on the grid with the fixed scale: float8 values, dequantized by the same scale, in the same range
G3  on correlated inputs the compensated codes give a smaller layer-output error than round to nearest, measured on the
    calibration inputs and on fresh inputs of the same distribution
G4  with uncorrelated inputs (H diagonal) it is round to nearest: nothing to compensate
G5  a dead input (zero column) does not stop the factorization and its weights are rounded as they are
G6  determinism and shapes, refusals for a Hessian that does not fit
G7  the stored format is the one hub_fp8 writes: one scale, the loader's requantization is the identity on the codes

  uv run pytest -q tests/unit/test_fp8_gptq.py
"""

import pytest
import torch

from decisio import fp8_gptq as G
from decisio import hub_fp8

OUT, IN = 24, 64


def layer(seed=0, rank=8, noise=0.05, rows=4096):
    g = torch.Generator().manual_seed(seed)
    basis = torch.randn(rank, IN, generator=g)
    x = torch.randn(rows, rank, generator=g) @ basis + noise * torch.randn(rows, IN, generator=g)  # correlated inputs
    w = torch.randn(OUT, IN, generator=g) * 0.05
    return w, x


def scale_of(w):
    return hub_fp8.scale_of(w.abs().max().to(torch.bfloat16).reshape(()))


def err(w, wq, x):
    return float(((x @ (w - wq).T) ** 2).mean())


def test_g1_the_hessian_is_the_running_mean_of_2_x_xt():
    w, x = layer()
    h = G.Hessian(IN)
    for part in x.split(1000):
        h.add(part)
    assert h.rows == x.shape[0]
    one = G.Hessian(IN)
    one.add(x)
    assert torch.allclose(h.matrix(), one.matrix(), rtol=1e-4, atol=1e-5)
    assert torch.allclose(one.matrix(), 2.0 / x.shape[0] * x.T @ x, rtol=1e-4, atol=1e-5)
    with pytest.raises(ValueError, match="no inputs"):
        G.Hessian(IN).matrix()
    with pytest.raises(ValueError, match="features"):
        G.Hessian(IN).add(torch.zeros(3, IN + 1))
    batched = G.Hessian(IN)
    batched.add(x.reshape(4, 1024, IN))  # [batch, tokens, features] is flattened to rows
    assert batched.rows == x.shape[0]


def test_g2_the_codes_are_on_the_grid_with_the_fixed_scale():
    w, x = layer()
    h = G.Hessian(IN)
    h.add(x)
    s = scale_of(w)
    codes, info = G.gptq_codes(w, h.matrix(), s)
    assert codes.dtype == G.FP8 and codes.shape == w.shape
    deq = codes.to(torch.float32) * s
    assert float(deq.abs().max()) <= float(w.abs().max()) * 1.0001 + 1e-6  # the range of the group
    # re-rounding the dequantized values with the same scale gives the same codes: they are grid points of that scale
    assert torch.equal(G.rtn_codes(deq, s).view(torch.uint8), codes.view(torch.uint8))
    assert set(info) == {"error_rtn", "error_gptq", "dead_inputs", "damp_added"}


def test_g3_it_beats_round_to_nearest_on_correlated_inputs():
    w, x = layer(seed=1)
    h = G.Hessian(IN)
    h.add(x)
    s = scale_of(w)
    codes, info = G.gptq_codes(w, h.matrix(), s)
    rtn = G.rtn_codes(w, s)
    q_g, q_r = codes.to(torch.float32) * s, rtn.to(torch.float32) * s
    assert not torch.equal(codes.view(torch.uint8), rtn.view(torch.uint8))  # some codes moved a step
    assert err(w, q_g, x) < 0.5 * err(w, q_r, x)  # on the calibration inputs, by a clear margin (here 10 times or more)
    # fresh rows of the same distribution: the same basis (seed 1), another draw of the coefficients and the noise
    g = torch.Generator().manual_seed(99)
    b = torch.randn(8, IN, generator=torch.Generator().manual_seed(1))
    fresh = torch.randn(2048, 8, generator=g) @ b + 0.05 * torch.randn(2048, IN, generator=g)
    assert err(w, q_g, fresh) < err(w, q_r, fresh)  # and on inputs it has not seen
    assert info["error_gptq"] < info["error_rtn"] and info["error_rtn"] > 0
    # the Hessian-weighted error it reports is the layer-output error: 2/N sum over rows, summed over the outputs
    assert info["error_rtn"] == pytest.approx(2 * err(w, q_r, x) * OUT, rel=1e-3)


def test_g4_uncorrelated_inputs_leave_round_to_nearest():
    g = torch.Generator().manual_seed(3)
    w = torch.randn(OUT, IN, generator=g) * 0.05
    h = torch.eye(IN) * 2.0
    s = scale_of(w)
    codes, info = G.gptq_codes(w, h, s)
    assert torch.equal(codes.view(torch.uint8), G.rtn_codes(w, s).view(torch.uint8))
    assert info["error_gptq"] == pytest.approx(info["error_rtn"], rel=1e-6)


def test_g5_a_dead_input_does_not_stop_it():
    w, x = layer(seed=2)
    x[:, 5] = 0.0
    x[:, 17] = 0.0
    h = G.Hessian(IN)
    h.add(x)
    s = scale_of(w)
    codes, info = G.gptq_codes(w, h.matrix(), s)
    assert info["dead_inputs"] == 2
    rtn = G.rtn_codes(w, s)
    for j in (5, 17):  # the dead inputs' weights are rounded as they are
        assert torch.equal(codes[:, j].view(torch.uint8), rtn[:, j].view(torch.uint8))
    assert err(w, codes.to(torch.float32) * s, x) <= err(w, rtn.to(torch.float32) * s, x) * 1.0001


def test_g6_determinism_blocks_and_refusals():
    w, x = layer(seed=4)
    h = G.Hessian(IN)
    h.add(x)
    s = scale_of(w)
    a, _ = G.gptq_codes(w, h.matrix(), s)
    b, _ = G.gptq_codes(w, h.matrix(), s)
    assert torch.equal(a.view(torch.uint8), b.view(torch.uint8))
    c, _ = G.gptq_codes(w, h.matrix(), s, blocksize=16)  # the block size changes the order of updates, not the result
    assert err(w, c.to(torch.float32) * s, x) < err(w, G.rtn_codes(w, s).to(torch.float32) * s, x)
    with pytest.raises(ValueError, match="do not fit"):
        G.gptq_codes(w, torch.eye(IN + 1), s)
    with pytest.raises(ValueError, match="do not fit"):
        G.gptq_codes(w[0], h.matrix(), s)


def test_g7_the_stored_format_is_hub_fp8s(tmp_path):
    w, x = layer(seed=5)
    h = G.Hessian(IN)
    h.add(x)
    s = scale_of(w)
    codes, _ = G.gptq_codes(w, h.matrix(), s)
    again = hub_fp8.loader_requantize(codes, s, s.max())  # one scale: the loader's requantization is the identity
    assert torch.equal(again.view(torch.uint8), codes.view(torch.uint8))
    assert codes.is_contiguous() and codes.dtype == hub_fp8.FP8
