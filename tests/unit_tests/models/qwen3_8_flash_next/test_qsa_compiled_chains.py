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

"""The QSA pre/post-attention chains and the indexer front half equal their module paths."""

import pytest
import torch
from torch import nn

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.gpt_oss.rope_utils import apply_rotary_emb_qk
from nemo_automodel.components.models.qwen3_8_flash_next.layers import (
    Qwen3_8_FlashNextQSAAttention,
    _qsa_gate_and_project_out,
    _qsa_gate_and_project_out_compiled,
    _qsa_project_qkv,
    _qsa_project_qkv_compiled,
)
from nemo_automodel.components.models.qwen3_8_flash_next.qsa import (
    Qwen3_8_FlashNextQSAIndexer,
    _indexer_query_key,
    _indexer_query_key_compiled,
    apply_qsa_rope,
)
from tests.unit_tests.models.qwen3_8_flash_next.test_qwen3_8_flash_next_model import _tiny_config


def _backend() -> BackendConfig:
    return BackendConfig(linear="torch", attn="sdpa", rms_norm="torch", experts="torch", dispatcher="torch")


def _attention() -> Qwen3_8_FlashNextQSAAttention:
    torch.manual_seed(0)
    config = _tiny_config().text_config
    attn = Qwen3_8_FlashNextQSAAttention(config, layer_idx=0, backend=_backend())
    attn.init_weights(torch.device("cpu"), init_std=0.1)
    with torch.no_grad():
        attn.q_norm.weight.uniform_(-0.2, 0.2)
        attn.k_norm.weight.uniform_(-0.2, 0.2)
    return attn


def _freqs(batch: int, seq: int, rotary_dim: int, seed: int) -> torch.Tensor:
    ang = torch.randn(batch, seq, rotary_dim // 2, generator=torch.Generator().manual_seed(seed))
    return torch.cat([ang.cos(), ang.sin()], dim=-1)


def _module_path(attn: Qwen3_8_FlashNextQSAAttention, x: torch.Tensor, freqs: torch.Tensor):
    """The original module-based sequence of ops (projections, chunk, norms, RoPE)."""
    b, s, _ = x.shape
    q = attn.q_proj(x).view(b, s, -1, attn.head_dim * 2)
    k = attn.k_proj(x).view(b, s, -1, attn.head_dim)
    v = attn.v_proj(x).view(b, s, -1, attn.head_dim)
    q, gate = torch.chunk(q, 2, dim=-1)
    gate = gate.reshape(b, s, -1)
    q, k = apply_rotary_emb_qk(attn.q_norm(q), attn.k_norm(k), freqs, format="bshd", rope_fusion=False)
    return q, k, v, gate


def test_project_qkv_chain_is_bitwise_the_module_path() -> None:
    attn = _attention()
    x = torch.randn(2, 7, attn.q_proj.in_features, generator=torch.Generator().manual_seed(1))
    freqs = _freqs(2, 7, attn.head_dim // 2, seed=2)
    expected = _module_path(attn, x, freqs)
    actual = _qsa_project_qkv(
        x,
        attn.q_proj.weight,
        attn.k_proj.weight,
        attn.v_proj.weight,
        attn.q_norm.weight,
        attn.k_norm.weight,
        attn.q_norm.eps,
        freqs,
        attn.head_dim,
    )
    for got, want in zip(actual, expected):
        assert torch.equal(got, want)
    attn_out = torch.randn_like(actual[3])
    assert torch.equal(
        _qsa_gate_and_project_out(attn_out, actual[3], attn.o_proj.weight),
        attn.o_proj(attn_out * torch.sigmoid(actual[3])),
    )


def test_indexer_front_half_is_bitwise_the_module_path() -> None:
    torch.manual_seed(3)
    config = _tiny_config().text_config
    indexer = Qwen3_8_FlashNextQSAIndexer(config, _backend())
    indexer.init_weights(init_std=0.1)
    with torch.no_grad():
        indexer.q_layernorm.weight.uniform_(-0.2, 0.2)
        indexer.k_layernorm.weight.uniform_(-0.2, 0.2)
    x = torch.randn(2, 8, indexer.hidden_size)
    freqs = _freqs(2, 8, indexer.head_dim // 2, seed=4)
    # Legacy inline sequence from the indexer forward.
    projected = indexer.index_qk_proj(x)
    qw = indexer.num_query_heads * indexer.head_dim
    raw_q = projected[..., :qw].unflatten(-1, (indexer.num_query_heads, indexer.head_dim))
    raw_k = projected[..., qw:].unflatten(-1, (indexer.num_key_heads, indexer.head_dim))
    want_q = apply_qsa_rope(indexer.q_layernorm(raw_q), freqs)
    nb = 8 // indexer.compress_ratio
    grouped = raw_k[:, : nb * indexer.compress_ratio].unflatten(1, (nb, indexer.compress_ratio))
    ck = grouped.float().mean(dim=2).to(raw_k.dtype)
    want_k = apply_qsa_rope(indexer.k_layernorm(ck), freqs[:, : nb * indexer.compress_ratio : indexer.compress_ratio])
    got_q, got_k = _indexer_query_key(
        x,
        indexer.index_qk_proj.weight,
        indexer.q_layernorm.weight,
        indexer.k_layernorm.weight,
        indexer.q_layernorm.eps,
        freqs,
        indexer.num_query_heads,
        indexer.num_key_heads,
        indexer.head_dim,
        indexer.compress_ratio,
    )
    assert torch.equal(got_q, want_q) and torch.equal(got_k, want_k)


def test_adapter_or_te_projections_keep_the_module_path() -> None:
    attn = _attention()
    assert attn._plain_projections()

    class _Adapter(nn.Linear):
        pass

    attn.q_proj = _Adapter(attn.q_proj.in_features, attn.q_proj.out_features, bias=False)
    assert not attn._plain_projections()


@pytest.mark.runtime_budget(
    90, hard_timeout=300, reason="three inductor CPU compilations (qkv chain, output chain, indexer front half)"
)
def test_compiled_chains_match_eager_on_cpu() -> None:
    attn = _attention()
    x = torch.randn(1, 6, attn.q_proj.in_features, generator=torch.Generator().manual_seed(5))
    freqs = _freqs(1, 6, attn.head_dim // 2, seed=6)
    args = (
        x,
        attn.q_proj.weight,
        attn.k_proj.weight,
        attn.v_proj.weight,
        attn.q_norm.weight,
        attn.k_norm.weight,
        attn.q_norm.eps,
        freqs,
        attn.head_dim,
    )
    for got, want in zip(_qsa_project_qkv_compiled(*args), _qsa_project_qkv(*args)):
        torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-6)
    out = torch.randn(1, 6, attn.o_proj.in_features)
    gate = torch.randn_like(out)
    torch.testing.assert_close(
        _qsa_gate_and_project_out_compiled(out, gate, attn.o_proj.weight),
        _qsa_gate_and_project_out(out, gate, attn.o_proj.weight),
        rtol=1e-5,
        atol=1e-6,
    )
    config = _tiny_config().text_config
    indexer = Qwen3_8_FlashNextQSAIndexer(config, _backend())
    indexer.init_weights(init_std=0.1)
    xi = torch.randn(1, 8, indexer.hidden_size)
    fi = _freqs(1, 8, indexer.head_dim // 2, seed=7)
    iargs = (
        xi,
        indexer.index_qk_proj.weight,
        indexer.q_layernorm.weight,
        indexer.k_layernorm.weight,
        indexer.q_layernorm.eps,
        fi,
        indexer.num_query_heads,
        indexer.num_key_heads,
        indexer.head_dim,
        indexer.compress_ratio,
    )
    for got, want in zip(_indexer_query_key_compiled(*iargs), _indexer_query_key(*iargs)):
        torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-6)
