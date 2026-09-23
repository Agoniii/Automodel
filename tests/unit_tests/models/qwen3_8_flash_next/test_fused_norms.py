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

"""Fused gated RMSNorm and q/k RMSNorm match their fp32 reference chains in forward and backward."""

import pytest
import torch

from nemo_automodel.components.models.qwen3_8_flash_next.hc_norm_triton import HAVE_TRITON, rms_norm_gated_triton
from nemo_automodel.components.models.qwen3_8_flash_next.layers import (
    Qwen3_8_FlashNextRMSNormGated,
    _rms_norm_gated_fp32,
)
from nemo_automodel.components.models.qwen3_8_flash_next.qsa import Qwen3_8_FlashNextRMSNorm
from nemo_automodel.components.models.qwen3_next.layers import Qwen3NextRMSNorm

cuda_triton = pytest.mark.skipif(
    not (torch.cuda.is_available() and HAVE_TRITON), reason="fused norms need CUDA and triton"
)


def _tol(dtype: torch.dtype) -> dict:
    return dict(rtol=2e-2, atol=2e-2) if dtype == torch.bfloat16 else dict(rtol=1e-5, atol=1e-5)


@cuda_triton
@pytest.mark.parametrize("use_sigmoid", [True, False])
@pytest.mark.parametrize(
    "x_dtype,w_dtype",
    [(torch.bfloat16, torch.bfloat16), (torch.float32, torch.bfloat16), (torch.float32, torch.float32)],
)
@pytest.mark.parametrize("rows,width", [(3000, 128), (77, 96)])
def test_fused_gated_norm_matches_reference(
    rows: int, width: int, x_dtype: torch.dtype, w_dtype: torch.dtype, use_sigmoid: bool
) -> None:
    g = torch.Generator(device="cuda").manual_seed(rows + width)
    x = torch.randn(rows, width, device="cuda", generator=g).to(x_dtype)
    gate = (2.0 * torch.randn(rows, width, device="cuda", generator=g)).to(x_dtype)
    w = (1.0 + 0.1 * torch.randn(width, device="cuda", generator=g)).to(w_dtype)
    upstream = torch.randn(rows, width, device="cuda", generator=g).to(x_dtype)
    ref = [t.clone().requires_grad_(True) for t in (x, gate, w)]
    fused = [t.clone().requires_grad_(True) for t in (x, gate, w)]

    y_ref = _rms_norm_gated_fp32(ref[0], ref[1], ref[2], 1e-6, use_sigmoid)
    y_fused = rms_norm_gated_triton(fused[0], fused[1], fused[2], 1e-6, use_sigmoid)
    assert y_fused.dtype == y_ref.dtype == x_dtype
    y_ref.backward(upstream)
    y_fused.backward(upstream)

    tol = _tol(x_dtype)
    torch.testing.assert_close(y_fused, y_ref, **tol)
    torch.testing.assert_close(fused[0].grad, ref[0].grad, **tol)
    torch.testing.assert_close(fused[1].grad, ref[1].grad, **tol)
    assert fused[2].grad.dtype == ref[2].grad.dtype == w_dtype
    scale = 1 + ref[2].grad.float().abs().max()
    torch.testing.assert_close(fused[2].grad.float(), ref[2].grad.float(), rtol=2e-2, atol=2e-2 * scale)


@cuda_triton
@pytest.mark.parametrize("shape,dim", [((1, 64, 24, 256), 256), ((2, 33, 4, 128), 128)])
def test_fused_qk_norm_matches_parent(shape: tuple, dim: int) -> None:
    torch.manual_seed(dim)
    parent = Qwen3NextRMSNorm(dim).cuda()
    fused = Qwen3_8_FlashNextRMSNorm(dim).cuda()
    with torch.no_grad():
        parent.weight.uniform_(-0.3, 0.3)
        fused.weight.copy_(parent.weight)
    x = torch.randn(*shape, device="cuda", dtype=torch.bfloat16)
    upstream = torch.randn_like(x)
    x_ref = x.clone().requires_grad_(True)
    x_fused = x.clone().requires_grad_(True)
    parent(x_ref).backward(upstream)
    fused(x_fused).backward(upstream)
    torch.testing.assert_close(fused(x), parent(x), rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(x_fused.grad, x_ref.grad, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(
        fused.weight.grad, parent.weight.grad, rtol=2e-2, atol=2e-2 * (1 + parent.weight.grad.abs().max())
    )
    assert list(fused.state_dict()) == list(parent.state_dict()) == ["weight"]


@cuda_triton
def test_fused_norm_weight_gradients_are_deterministic() -> None:
    x = torch.randn(5000, 128, device="cuda", dtype=torch.bfloat16)
    gate = torch.randn_like(x)
    upstream = torch.randn_like(x)
    grads = []
    for _ in range(2):
        w = torch.ones(128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        rms_norm_gated_triton(x, gate, w, 1e-6, True).backward(upstream)
        grads.append(w.grad.clone())
    assert torch.equal(grads[0], grads[1])


def test_modules_fall_back_to_reference_chains_on_cpu() -> None:
    gated = Qwen3_8_FlashNextRMSNormGated(16, activation="sigmoid")
    x = torch.randn(3, 16)
    gate = torch.randn(3, 16)
    torch.testing.assert_close(
        gated(x, gate), _rms_norm_gated_fp32(x, gate, gated.weight, gated.variance_epsilon, True), rtol=0, atol=0
    )
    norm = Qwen3_8_FlashNextRMSNorm(8)
    y = torch.randn(2, 3, 8)
    torch.testing.assert_close(norm(y), Qwen3NextRMSNorm.forward(norm, y), rtol=0, atol=0)
