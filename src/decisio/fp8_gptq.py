# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""Error-compensating rounding to FP8 (e4m3) for a linear layer, with the scale fixed (GPTQ's column pass).

`hub_fp8` quantizes a fused group with one scale, `float32(amax) / 448`, and rounds every weight to the nearest e4m3
code. Round to nearest minimizes each weight's own error; the layer's output error is what matters, and the inputs a
layer sees are correlated, so an error made on one weight can be taken back by the weights that are still to be rounded.
This module does that with the Hessian of the layer's output error, H = 2/N * sum x x^T over calibration inputs x (the
rows the layer reads), column by column as in GPTQ (Frantar et al.): round column j to the grid, spread its error over
the columns after it with the rows of the Cholesky factor of H^-1, go on. The grid and the scale do not change: the
result is FP8 codes in the stored format of `hub_fp8` (one scale per fused group, the loader's requantization is the
identity on them), only some codes are one step from where round to nearest puts them.

    h = Hessian(in_features); h.add(x)                       # x: [rows, in_features], any number of batches
    codes, info = gptq_codes(w32, h.matrix(), scale)         # w32: [out, in] fp32, scale: the group's, shape [1]
    rtn = rtn_codes(w32, scale)                              # what round to nearest gives

`info` has the layer-output error (the Hessian-weighted squared error, sum over rows of (w - q) H (w - q)^T) of both,
so a caller can report what the compensation bought. Standard library and torch only.
"""

from __future__ import annotations

import torch

FP8 = torch.float8_e4m3fn
FP8_MAX = 448.0


def _codes(w: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """The static FP8 kernel as hub_fp8.quantize writes it: multiply by the fp32 reciprocal, clamp, round to nearest."""
    inv = torch.reciprocal(scale.to(torch.float32).reshape(()))
    return (w.to(torch.float32) * inv).clamp(-FP8_MAX, FP8_MAX).to(FP8)


def _dequant(codes: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return codes.to(torch.float32) * scale.to(torch.float32).reshape(())


def rtn_codes(w: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Round to nearest, the codes `hub_fp8` writes."""
    return _codes(w, scale)


class Hessian:
    """The running H = 2/N * sum x x^T of a layer's inputs, in fp32 on the device of the first batch."""

    def __init__(self, features: int):
        self.features, self.rows, self.h = features, 0, None

    def add(self, x: torch.Tensor) -> None:
        x = x.reshape(-1, x.shape[-1]).to(torch.float32)
        if x.shape[-1] != self.features:
            raise ValueError(f"inputs have {x.shape[-1]} features, the Hessian {self.features}")
        if self.h is None:
            self.h = torch.zeros(self.features, self.features, dtype=torch.float32, device=x.device)
        n, m = self.rows, x.shape[0]
        self.h *= n / (n + m)
        self.h += (2.0 / (n + m)) * (x.T @ x)
        self.rows = n + m

    def matrix(self) -> torch.Tensor:
        if self.h is None:
            raise ValueError("no inputs were added")
        return self.h


def output_error(w: torch.Tensor, wq: torch.Tensor, h: torch.Tensor) -> float:
    """sum over output rows of d H d^T with d = w - wq: the layer's output error under the calibration inputs
    (up to the factor 1/2 of H's definition)."""
    d = (w - wq).to(torch.float32)
    return float(((d @ h) * d).sum())


def gptq_codes(
    w: torch.Tensor, h: torch.Tensor, scale: torch.Tensor, blocksize: int = 128, damp: float = 0.01
) -> tuple[torch.Tensor, dict]:
    """FP8 codes for `w` ([out, in], fp32) at the fixed `scale`, with the error of each column spread over the columns
    after it. `h`: the Hessian ([in, in]); `damp`: the share of its mean diagonal added to every diagonal entry (GPTQ's
    dampening), which also keeps a dead input (a zero row and column) from stopping the factorization. Returns the codes
    and {"error_rtn", "error_gptq", "dead_inputs", "damp_added"}."""
    if w.dim() != 2 or h.shape != (w.shape[1], w.shape[1]):
        raise ValueError(f"weight {tuple(w.shape)} and Hessian {tuple(h.shape)} do not fit")
    w = w.to(torch.float32).clone()
    h = h.to(device=w.device, dtype=torch.float32).clone()
    scale = scale.to(w.device)
    original = w.clone()
    h0 = h.clone()
    n = w.shape[1]
    dead = torch.diag(h) == 0
    h[dead, dead] = 1.0  # a dead input has no say; its column is rounded as it is
    w[:, dead] = 0.0
    added = damp * float(torch.mean(torch.diag(h)))
    h += torch.eye(n, device=h.device, dtype=h.dtype) * added
    eye = torch.eye(n, device=h.device, dtype=h.dtype)
    for attempt in range(4):  # a Hessian that is not positive definite even so: more dampening, said in `info`
        try:
            chol = torch.linalg.cholesky(h)
            break
        except torch.linalg.LinAlgError:
            if attempt == 3:
                raise
            extra = added * 10.0 ** (attempt + 1)
            h += eye * extra
            added += extra
    hinv = torch.linalg.cholesky(torch.cholesky_inverse(chol), upper=True)  # the upper factor of H^-1
    codes = torch.empty_like(w, dtype=FP8)
    for i1 in range(0, n, blocksize):
        i2 = min(i1 + blocksize, n)
        block = w[:, i1:i2].clone()
        err = torch.zeros_like(block)
        hb = hinv[i1:i2, i1:i2]
        for j in range(i2 - i1):
            col = block[:, j]
            q = _codes(col, scale)
            codes[:, i1 + j] = q
            e = (col - _dequant(q, scale)) / hb[j, j]
            block[:, j:] -= e.unsqueeze(1) @ hb[j, j:].unsqueeze(0)
            err[:, j] = e
        w[:, i2:] -= err @ hinv[i1:i2, i2:]
    # a dead input is not compensated for: its weight is rounded from the original
    if bool(dead.any()):
        codes[:, dead] = _codes(original[:, dead], scale)
    wq = _dequant(codes, scale)
    return codes, {
        "error_rtn": output_error(original, _dequant(_codes(original, scale), scale), h0),
        "error_gptq": output_error(original, wq, h0),
        "dead_inputs": int(dead.sum()),
        "damp_added": added,
    }
