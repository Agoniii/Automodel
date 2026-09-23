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

"""Fused HyperConnection grouped RMSNorm matches the fp32 reference chain in forward and backward."""

import pytest
import torch

from nemo_automodel.components.models.qwen3_8_flash_next.hc_norm_triton import HAVE_TRITON, grouped_rms_norm_triton
from nemo_automodel.components.models.qwen3_8_flash_next.layers import (
    Qwen3_8_FlashNextGroupedRMSNorm,
    _grouped_rms_norm_fp32,
)

cuda_triton = pytest.mark.skipif(
    not (torch.cuda.is_available() and HAVE_TRITON), reason="fused grouped RMSNorm needs CUDA and triton"
)


def _case(rows: int, groups: int, group_size: int, x_dtype: torch.dtype, w_dtype: torch.dtype, seed: int):
    """Random residual stream ``[2, rows, groups * group_size]`` and additive weight ``[groups * group_size]``."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(2, rows, groups * group_size, device="cuda", generator=g).to(x_dtype)
    w = (0.1 * torch.randn(groups * group_size, device="cuda", generator=g)).to(w_dtype)
    return x, w


@cuda_triton
@pytest.mark.parametrize(
    "x_dtype,w_dtype",
    [(torch.bfloat16, torch.bfloat16), (torch.float32, torch.bfloat16), (torch.float32, torch.float32)],
)
@pytest.mark.parametrize("rows,group_size", [(37, 2560), (129, 96), (5, 8)])
def test_fused_grouped_rms_norm_matches_reference(
    rows: int, group_size: int, x_dtype: torch.dtype, w_dtype: torch.dtype
) -> None:
    x, w = _case(rows, 4, group_size, x_dtype, w_dtype, seed=rows + group_size)
    x_ref = x.clone().requires_grad_(True)
    w_ref = w.clone().requires_grad_(True)
    x_fused = x.clone().requires_grad_(True)
    w_fused = w.clone().requires_grad_(True)
    upstream = torch.randn_like(x, dtype=torch.float32).to(x_dtype)

    y_ref = _grouped_rms_norm_fp32(x_ref, w_ref, group_size, 1e-6)
    y_fused = grouped_rms_norm_triton(x_fused, w_fused, group_size, 1e-6)
    assert y_fused.dtype == y_ref.dtype == x_dtype and y_fused.shape == y_ref.shape
    y_ref.backward(upstream)
    y_fused.backward(upstream)

    tol = dict(rtol=2e-2, atol=2e-2) if x_dtype == torch.bfloat16 else dict(rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(y_fused, y_ref, **tol)
    torch.testing.assert_close(x_fused.grad, x_ref.grad, **tol)
    assert w_fused.grad.dtype == w_ref.grad.dtype == w_dtype
    # dw sums over all rows in fp32 in both paths; compare in fp32 with a tolerance for the final cast.
    torch.testing.assert_close(
        w_fused.grad.float(), w_ref.grad.float(), rtol=2e-2, atol=2e-2 * (1 + w_ref.grad.float().abs().max())
    )


@cuda_triton
def test_fused_grouped_rms_norm_weight_gradient_is_deterministic() -> None:
    """Two identical backward passes give bit-identical dw (no atomics)."""
    x, w = _case(200, 4, 256, torch.bfloat16, torch.bfloat16, seed=7)
    upstream = torch.randn_like(x)
    grads = []
    for _ in range(2):
        w_run = w.clone().requires_grad_(True)
        grouped_rms_norm_triton(x, w_run, 256, 1e-6).backward(upstream)
        grads.append(w_run.grad.clone())
    assert torch.equal(grads[0], grads[1])


@cuda_triton
def test_module_uses_fused_kernel_on_cuda() -> None:
    norm = Qwen3_8_FlashNextGroupedRMSNorm(4 * 64, group_size=64).cuda()
    with torch.no_grad():
        norm.weight.uniform_(-0.2, 0.2)
    x = torch.randn(3, 5, 256, device="cuda", dtype=torch.bfloat16)
    expected = _grouped_rms_norm_fp32(x, norm.weight, 64, norm.eps)
    torch.testing.assert_close(norm(x), expected, rtol=2e-2, atol=2e-2)


def test_module_falls_back_to_eager_chain_on_cpu() -> None:
    norm = Qwen3_8_FlashNextGroupedRMSNorm(4 * 8, group_size=8)
    x = torch.randn(2, 3, 32)
    torch.testing.assert_close(norm(x), _grouped_rms_norm_fp32(x, norm.weight, 8, norm.eps), rtol=0, atol=0)
