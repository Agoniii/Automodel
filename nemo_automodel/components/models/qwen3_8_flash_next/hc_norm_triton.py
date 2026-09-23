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

"""Fused Triton forward/backward for the Qwen3.8-Flash-Next RMSNorm variants.

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


def _launch_geometry(block_n: int) -> tuple[int, int]:
    """Rows per program and warps for a given padded row width.

    Wide rows (the 2560-wide HC branches) get one row per program with 8 warps;
    narrow rows (128/256-wide attention heads) batch many rows per program so
    the grid stays small and each program still moves a few KiB.

    Args:
        block_n: Power-of-two padded width of one normalized row.

    Returns:
        ``(rows_per_program, num_warps)``.
    """
    rows = max(1, min(64, 4096 // block_n))
    return rows, (8 if block_n >= 2048 else 4)


@triton.jit
def _grouped_rms_norm_fwd_kernel(
    X,
    W,
    Y,
    RSTD,
    num_rows,
    group_size,
    eps,
    stride_row,
    GROUPS: tl.constexpr,
    ROWS_PER_PROGRAM: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """One program per (row block, branch): y = (x * rsqrt(mean(x^2) + eps)) * (1 + w) in fp32."""
    block = tl.program_id(0)
    group = tl.program_id(1)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < group_size
    w = tl.load(W + group * group_size + cols, mask=mask, other=0.0).to(tl.float32)
    for i in range(ROWS_PER_PROGRAM):
        row = block * ROWS_PER_PROGRAM + i
        valid = row < num_rows
        row_mask = mask & valid
        offsets = row * stride_row + group * group_size + cols
        x = tl.load(X + offsets, mask=row_mask, other=0.0).to(tl.float32)
        variance = tl.sum(x * x, axis=0) / group_size
        rstd = 1.0 / tl.sqrt(variance + eps)
        y = (x * rstd) * (1.0 + w)
        tl.store(Y + offsets, y.to(Y.dtype.element_ty), mask=row_mask)
        tl.store(RSTD + row * GROUPS + group, rstd, mask=valid)


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


@triton.jit
def _rms_norm_gated_fwd_kernel(
    X,
    G,
    W,
    Y,
    RSTD,
    num_rows,
    width,
    eps,
    stride_row,
    USE_SIGMOID: tl.constexpr,
    ROWS_PER_PROGRAM: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """y = ((x * rsqrt(mean(x^2) + eps)) * w) * act(gate), all in fp32, stored in the input dtype."""
    block = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < width
    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
    for i in range(ROWS_PER_PROGRAM):
        row = block * ROWS_PER_PROGRAM + i
        valid = row < num_rows
        row_mask = mask & valid
        offsets = row * stride_row + cols
        x = tl.load(X + offsets, mask=row_mask, other=0.0).to(tl.float32)
        gate = tl.load(G + offsets, mask=row_mask, other=0.0).to(tl.float32)
        variance = tl.sum(x * x, axis=0) / width
        rstd = 1.0 / tl.sqrt(variance + eps)
        normalized = w * (x * rstd)
        sig = 1.0 / (1.0 + tl.exp(-gate))
        act = sig if USE_SIGMOID else gate * sig
        tl.store(Y + offsets, (normalized * act).to(Y.dtype.element_ty), mask=row_mask)
        tl.store(RSTD + row, rstd, mask=valid)


@triton.jit
def _rms_norm_gated_bwd_kernel(
    X,
    G,
    W,
    DY,
    RSTD,
    DX,
    DG,
    DW_PARTIAL,
    num_rows,
    width,
    stride_row,
    USE_SIGMOID: tl.constexpr,
    ROWS_PER_PROGRAM: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Backward of the gated RMSNorm with a per-block deterministic dw partial.

    n = w * xhat, y = n * act(gate):
    dn = dy * act, dgate = dy * n * act'(gate), dw = sum_rows(dn * xhat),
    dxhat = dn * w, dx = rstd * (dxhat - xhat * mean(dxhat * xhat)).
    """
    block = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < width
    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
    dw = tl.zeros([BLOCK_N], dtype=tl.float32)
    for i in range(ROWS_PER_PROGRAM):
        row = block * ROWS_PER_PROGRAM + i
        valid = row < num_rows
        row_mask = mask & valid
        offsets = row * stride_row + cols
        x = tl.load(X + offsets, mask=row_mask, other=0.0).to(tl.float32)
        gate = tl.load(G + offsets, mask=row_mask, other=0.0).to(tl.float32)
        dy = tl.load(DY + offsets, mask=row_mask, other=0.0).to(tl.float32)
        rstd = tl.load(RSTD + row, mask=valid, other=0.0)
        xhat = x * rstd
        n = w * xhat
        sig = 1.0 / (1.0 + tl.exp(-gate))
        if USE_SIGMOID:
            act = sig
            dact = sig * (1.0 - sig)
        else:
            act = gate * sig
            dact = sig + gate * sig * (1.0 - sig)
        dn = dy * act
        dgate = dy * n * dact
        dw += dn * xhat
        dxhat = dn * w
        mean_dxhat_xhat = tl.sum(dxhat * xhat, axis=0) / width
        dx = rstd * (dxhat - xhat * mean_dxhat_xhat)
        tl.store(DX + offsets, dx.to(DX.dtype.element_ty), mask=row_mask)
        tl.store(DG + offsets, dgate.to(DG.dtype.element_ty), mask=row_mask)
    tl.store(DW_PARTIAL + block * width + cols, dw, mask=mask)


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
        rows_per_program, num_warps = _launch_geometry(block_n)
        if rows:
            _grouped_rms_norm_fwd_kernel[(triton.cdiv(rows, rows_per_program), groups)](
                x2,
                weight,
                y,
                rstd,
                rows,
                group_size,
                eps,
                x2.stride(0),
                GROUPS=groups,
                ROWS_PER_PROGRAM=rows_per_program,
                BLOCK_N=block_n,
                num_warps=num_warps,
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
        block_n = triton.next_power_of_2(group_size)
        rows_per_program, num_warps = _launch_geometry(block_n)
        row_blocks = max(triton.cdiv(rows, rows_per_program), 1)
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
                ROWS_PER_PROGRAM=rows_per_program,
                BLOCK_N=block_n,
                num_warps=num_warps,
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


class _RMSNormGatedTriton(torch.autograd.Function):
    """Autograd wrapper around the fused gated RMSNorm kernels."""

    @staticmethod
    def forward(
        ctx: Any,
        hidden_states: torch.Tensor,
        gate: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
        use_sigmoid: bool,
    ) -> torch.Tensor:
        """Normalize the last axis, scale by ``weight`` and multiply by the activated gate.

        Args:
            ctx: Autograd context.
            hidden_states: Tensor of shape ``[..., width]``; arbitrary leading dims.
            gate: Gate logits of shape ``[..., width]`` (same shape as ``hidden_states``).
            weight: Multiplicative scale of shape ``[width]``.
            eps: Variance epsilon.
            use_sigmoid: ``True`` for a sigmoid gate, ``False`` for SiLU.

        Returns:
            Tensor of the same shape and dtype as ``hidden_states``.
        """
        width = hidden_states.shape[-1]
        x = hidden_states.contiguous()
        g = gate.contiguous()
        rows = x.numel() // width
        x2 = x.view(rows, width)
        g2 = g.view(rows, width)
        y = torch.empty_like(x2)
        rstd = torch.empty(rows, dtype=torch.float32, device=x.device)
        block_n = triton.next_power_of_2(width)
        rows_per_program, num_warps = _launch_geometry(block_n)
        if rows:
            _rms_norm_gated_fwd_kernel[(triton.cdiv(rows, rows_per_program),)](
                x2,
                g2,
                weight,
                y,
                rstd,
                rows,
                width,
                eps,
                x2.stride(0),
                USE_SIGMOID=use_sigmoid,
                ROWS_PER_PROGRAM=rows_per_program,
                BLOCK_N=block_n,
                num_warps=num_warps,
            )
        ctx.save_for_backward(x2, g2, weight, rstd)
        ctx.use_sigmoid = use_sigmoid
        return y.view(hidden_states.shape)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, None, None]:
        """Return gradients for ``hidden_states``, ``gate`` (their dtypes) and ``weight`` (weight dtype).

        Args:
            ctx: Context populated by :meth:`forward`.
            grad_output: Tensor of shape ``[..., width]`` in the output dtype.

        Returns:
            ``(grad_hidden_states, grad_gate, grad_weight, None, None)``.
        """
        x2, g2, weight, rstd = ctx.saved_tensors
        rows, width = x2.shape
        dy = grad_output.contiguous().view(rows, width)
        dx = torch.empty_like(x2)
        dg = torch.empty_like(g2)
        block_n = triton.next_power_of_2(width)
        rows_per_program, num_warps = _launch_geometry(block_n)
        row_blocks = max(triton.cdiv(rows, rows_per_program), 1)
        dw_partial = torch.zeros(row_blocks, width, dtype=torch.float32, device=x2.device)
        if rows:
            _rms_norm_gated_bwd_kernel[(row_blocks,)](
                x2,
                g2,
                weight,
                dy,
                rstd,
                dx,
                dg,
                dw_partial,
                rows,
                width,
                x2.stride(0),
                USE_SIGMOID=ctx.use_sigmoid,
                ROWS_PER_PROGRAM=rows_per_program,
                BLOCK_N=block_n,
                num_warps=num_warps,
            )
        dw = dw_partial.sum(dim=0).to(weight.dtype)
        return dx.view(grad_output.shape), dg.view(grad_output.shape), dw, None, None


def rms_norm_gated_triton(
    hidden_states: torch.Tensor,
    gate: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    use_sigmoid: bool,
) -> torch.Tensor:
    """Fused gated RMSNorm, numerically the fp32 chain of ``_rms_norm_gated_fp32``.

    Args:
        hidden_states: CUDA tensor of shape ``[..., width]`` (bf16 or fp32).
        gate: Gate logits of shape ``[..., width]``.
        weight: Multiplicative scale of shape ``[width]``.
        eps: Variance epsilon.
        use_sigmoid: ``True`` for a sigmoid output gate, ``False`` for SiLU.

    Returns:
        Tensor of the same shape and dtype as ``hidden_states``.
    """
    return _RMSNormGatedTriton.apply(hidden_states, gate, weight, eps, use_sigmoid)


__all__ = ["HAVE_TRITON", "grouped_rms_norm_triton", "rms_norm_gated_triton"]
