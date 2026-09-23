# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Fused Triton forward/backward for the HyperConnection grouped RMSNorm.

The HC residual stream is ``[..., hc_count * hidden]``; every ``hidden``-wide
branch is normalized independently in fp32 and scaled by ``1 + weight``
(Gemma style). The inductor-compiled eager chain spends ~130 ms of GPU time
and ~790 launches per microbatch on GB200 (forward + backward, 48 layers x 2
HyperConnections); this module does the same math in one kernel per direction
with a deterministic (non-atomic) weight-gradient reduction.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import torch

from nemo_automodel.shared.import_utils import null_decorator

try:
    import triton
    import triton.language as tl

    HAVE_TRITON = True
except ImportError:  # pragma: no cover - exercised only without triton
    HAVE_TRITON = False

if not HAVE_TRITON:
    triton = MagicMock()
    triton.jit = null_decorator
    tl = MagicMock()

_ROWS_PER_PROGRAM = 16


@triton.jit
def _grouped_rms_norm_fwd_kernel(
    X,
    W,
    Y,
    RSTD,
    group_size,
    eps,
    stride_row,
    GROUPS: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """One program per (row, branch): y = (x * rsqrt(mean(x^2) + eps)) * (1 + w) in fp32."""
    row = tl.program_id(0)
    group = tl.program_id(1)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < group_size
    offsets = row * stride_row + group * group_size + cols
    x = tl.load(X + offsets, mask=mask, other=0.0).to(tl.float32)
    variance = tl.sum(x * x, axis=0) / group_size
    rstd = 1.0 / tl.sqrt(variance + eps)
    w = tl.load(W + group * group_size + cols, mask=mask, other=0.0).to(tl.float32)
    y = (x * rstd) * (1.0 + w)
    tl.store(Y + offsets, y.to(Y.dtype.element_ty), mask=mask)
    tl.store(RSTD + row * GROUPS + group, rstd)


@triton.jit
def _grouped_rms_norm_bwd_kernel(
    X,
    W,
    DY,
    RSTD,
    DX,
    DW_PARTIAL,
    num_rows,
    group_size,
    stride_row,
    GROUPS: tl.constexpr,
    ROWS_PER_PROGRAM: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """One program per (row block, branch): dx for its rows and a per-block partial dw.

    With xhat = x * rstd and g = dy * (1 + w):
    dx = rstd * (g - xhat * mean(g * xhat)),  dw = sum_rows(dy * xhat).
    The dw partial of each row block is written to its own slot so the final
    reduction (``DW_PARTIAL.sum(0)``) is deterministic.
    """
    block = tl.program_id(0)
    group = tl.program_id(1)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < group_size
    w = tl.load(W + group * group_size + cols, mask=mask, other=0.0).to(tl.float32)
    dw = tl.zeros([BLOCK_N], dtype=tl.float32)
    row_start = block * ROWS_PER_PROGRAM
    for i in range(ROWS_PER_PROGRAM):
        row = row_start + i
        valid = row < num_rows
        offsets = row * stride_row + group * group_size + cols
        row_mask = mask & valid
        x = tl.load(X + offsets, mask=row_mask, other=0.0).to(tl.float32)
        dy = tl.load(DY + offsets, mask=row_mask, other=0.0).to(tl.float32)
        rstd = tl.load(RSTD + row * GROUPS + group, mask=valid, other=0.0)
        xhat = x * rstd
        g = dy * (1.0 + w)
        mean_g_xhat = tl.sum(g * xhat, axis=0) / group_size
        dx = rstd * (g - xhat * mean_g_xhat)
        tl.store(DX + offsets, dx.to(DX.dtype.element_ty), mask=row_mask)
        dw += dy * xhat
    tl.store(DW_PARTIAL + block * (GROUPS * group_size) + group * group_size + cols, dw, mask=mask)


class _GroupedRMSNormTriton(torch.autograd.Function):
    """Autograd wrapper around the fused grouped RMSNorm kernels."""

    @staticmethod
    def forward(
        ctx: Any, hidden_states: torch.Tensor, weight: torch.Tensor, group_size: int, eps: float
    ) -> torch.Tensor:
        """Normalize every ``group_size``-wide branch of the last axis.

        Args:
            ctx: Autograd context.
            hidden_states: Tensor of shape ``[..., groups * group_size]``; arbitrary leading dims.
            weight: Additive scale of shape ``[groups * group_size]`` applied as ``1 + weight``.
            group_size: Width of one branch.
            eps: Variance epsilon.

        Returns:
            Tensor of the same shape and dtype as ``hidden_states``.
        """
        hidden_size = hidden_states.shape[-1]
        groups = hidden_size // group_size
        x = hidden_states.contiguous()
        rows = x.numel() // hidden_size
        x2 = x.view(rows, hidden_size)
        y = torch.empty_like(x2)
        rstd = torch.empty(rows, groups, dtype=torch.float32, device=x.device)
        block_n = triton.next_power_of_2(group_size)
        if rows:
            _grouped_rms_norm_fwd_kernel[(rows, groups)](
                x2,
                weight,
                y,
                rstd,
                group_size,
                eps,
                x2.stride(0),
                GROUPS=groups,
                BLOCK_N=block_n,
                num_warps=8,
            )
        ctx.save_for_backward(x2, weight, rstd)
        ctx.group_size = group_size
        return y.view(hidden_states.shape)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, None, None]:
        """Return gradients for ``hidden_states`` (input dtype) and ``weight`` (weight dtype).

        Args:
            ctx: Context populated by :meth:`forward`.
            grad_output: Tensor of shape ``[..., groups * group_size]`` in the output dtype.

        Returns:
            ``(grad_hidden_states, grad_weight, None, None)``.
        """
        x2, weight, rstd = ctx.saved_tensors
        group_size = ctx.group_size
        rows, hidden_size = x2.shape
        groups = hidden_size // group_size
        dy = grad_output.contiguous().view(rows, hidden_size)
        dx = torch.empty_like(x2)
        row_blocks = max(triton.cdiv(rows, _ROWS_PER_PROGRAM), 1)
        dw_partial = torch.zeros(row_blocks, hidden_size, dtype=torch.float32, device=x2.device)
        if rows:
            _grouped_rms_norm_bwd_kernel[(row_blocks, groups)](
                x2,
                weight,
                dy,
                rstd,
                dx,
                dw_partial,
                rows,
                group_size,
                x2.stride(0),
                GROUPS=groups,
                ROWS_PER_PROGRAM=_ROWS_PER_PROGRAM,
                BLOCK_N=triton.next_power_of_2(group_size),
                num_warps=8,
            )
        dw = dw_partial.sum(dim=0).to(weight.dtype)
        return dx.view(grad_output.shape), dw, None, None


def grouped_rms_norm_triton(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    group_size: int,
    eps: float,
) -> torch.Tensor:
    """Fused grouped RMSNorm, numerically the fp32 chain of ``_grouped_rms_norm_fp32``.

    Args:
        hidden_states: CUDA tensor of shape ``[..., groups * group_size]`` (bf16 or fp32).
        weight: Tensor of shape ``[groups * group_size]``; applied as ``1 + weight``.
        group_size: Width of one branch; must divide the last axis.
        eps: Variance epsilon.

    Returns:
        Tensor of the same shape and dtype as ``hidden_states``.
    """
    return _GroupedRMSNormTriton.apply(hidden_states, weight, group_size, eps)


__all__ = ["HAVE_TRITON", "grouped_rms_norm_triton"]
