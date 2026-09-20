# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""Numerical media packing and CP parity, including GDN and nonzero PLE conv."""

import contextlib
import copy
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
from test_multimodal import _config, _model

from nemo_automodel.components.models.qwen3_8_flash_next.model import Qwen3_8_FlashNextForConditionalGeneration


def _hybrid_model() -> Qwen3_8_FlashNextForConditionalGeneration:
    config = _config()
    config.text_config.num_hidden_layers = 2
    config.text_config.layer_types = ["full_attention", "linear_attention"]
    config.text_config.linear_num_key_heads = 2
    config.text_config.linear_num_value_heads = 2
    config.text_config.linear_key_head_dim = 8
    config.text_config.linear_value_head_dim = 8
    config.text_config.router_aux_loss_coef = 0.0  # Explicit objective parity, no external aux scaler.
    model = _model(config)
    with torch.no_grad():
        model.model.language_model.layers["0"].ple.conv1d.weight.normal_(0, 0.03)
    return model


def _documents() -> list[dict[str, torch.Tensor]]:
    torch.manual_seed(832)
    return [
        {
            "input_ids": torch.tensor([[2, 62, *([60] * 8), 63, 3, 4, 5]]),
            "pixel_values": torch.randn(32, 24),
            "image_grid_thw": torch.tensor([[1, 4, 8]]),
        },
        {
            "input_ids": torch.tensor([[2, 62, *([61] * 4), 63, 3, 62, *([61] * 4), 63, 4]]),
            "pixel_values_videos": torch.randn(32, 24),
            "video_grid_thw": torch.tensor([[2, 4, 4]]),
        },
        {"input_ids": torch.tensor([[6, 7, 8, 9, 10]])},
    ]


def _pack(documents: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Combine document tensors into one physical packed row.

    Args:
        documents: Mappings with IDs [1, sequence], patches [patches, 24], grids [items, 3].
    Returns:
        Concatenated IDs [1, total_sequence], grids/patches in media order and
        cumulative token boundaries [documents + 1].
    """
    result = {"input_ids": torch.cat([doc["input_ids"] for doc in documents], dim=1)}
    lengths = torch.tensor([doc["input_ids"].shape[1] for doc in documents])
    result["cu_seqlens"] = torch.cat((lengths.new_zeros(1), lengths.cumsum(0)))
    for key in ("pixel_values", "pixel_values_videos", "image_grid_thw", "video_grid_thw"):
        values = [doc[key] for doc in documents if key in doc]
        if values:
            result[key] = torch.cat(values)
    return result


@pytest.mark.parametrize("metadata", ["cu_seqlens", "neat", "neat_single", "thd"])
def test_packed_media_matches_independent_documents_and_gradients(metadata: str) -> None:
    from nemo_automodel.components.datasets.vlm.collate_fns import (
        neat_packed_vlm_collater,
        packed_sequence_thd_vlm_collater,
    )

    model, reference = _hybrid_model(), _hybrid_model()
    docs = _documents()
    if metadata == "neat_single":
        docs = docs[:1]
    batch = _pack(docs)
    if metadata != "cu_seqlens":
        positions = [
            model.model.get_rope_index(
                doc["input_ids"], image_grid_thw=doc.get("image_grid_thw"), video_grid_thw=doc.get("video_grid_thw")
            )[0][:, 0]
            for doc in docs
        ]
        item = {key: value for key, value in batch.items() if key != "cu_seqlens"}
        item["input_ids"] = item["input_ids"][0]
        item["labels"] = item["input_ids"].clone()
        item["position_ids"] = torch.cat(positions, dim=1)
        item["attention_mask"] = torch.cat([torch.full_like(doc["input_ids"][0], i + 1) for i, doc in enumerate(docs)])
        item["mm_token_type_ids"] = torch.where(item["input_ids"] == 60, 1, torch.where(item["input_ids"] == 61, 2, 0))
        if metadata.startswith("neat"):
            batch = neat_packed_vlm_collater([item], max_length=40)
        else:
            batch = packed_sequence_thd_vlm_collater([item], max_length=40)
        # Both collators use BF16 media; align reference inputs with those values.
        for doc in docs:
            for key in ("pixel_values", "pixel_values_videos"):
                if key in doc:
                    doc[key] = doc[key].to(torch.bfloat16)
    length = sum(doc["input_ids"].shape[1] for doc in docs)
    actual = model(**batch).logits[:, :length]
    expected = torch.cat([reference(**doc).logits for doc in docs], dim=1)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
    upstream = torch.randn_like(actual)
    actual.backward(upstream)
    expected.backward(upstream)
    for (name, parameter), ref in zip(model.named_parameters(), reference.parameters(), strict=True):
        torch.testing.assert_close(parameter.grad, ref.grad, atol=2e-5, rtol=2e-4, msg=lambda error: f"{name}: {error}")
    # Later samples must not depend on earlier samples' media, hash history or recurrence.
    modified = {**batch, "pixel_values": batch["pixel_values"] + 0.25}
    torch.testing.assert_close(model(**modified).logits[:, 14:length], actual[:, 14:length], atol=2e-6, rtol=2e-5)


def _cp_worker(rank: int, world_size: int, rendezvous: str) -> None:
    from torch.distributed.device_mesh import init_device_mesh
    from torch.nn.parallel import DistributedDataParallel

    torch.set_num_threads(1)
    # Tiny replicated table; initialize before creating process groups.
    model, reference = _hybrid_model(), _hybrid_model()
    dist.init_process_group(
        "gloo", init_method=rendezvous, rank=rank, world_size=world_size, timeout=timedelta(seconds=60)
    )
    try:
        mesh = init_device_mesh("cpu", (world_size // 2, 2), mesh_dim_names=("dp", "cp"))
        cp_mesh = mesh["cp"]
        layers = model.model.language_model.layers
        layers["0"].self_attn.setup_cp_attention(cp_mesh)
        layers["1"].linear_attn._cp_mesh = cp_mesh
        if world_size == 4:
            from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper

            from nemo_automodel.components.moe.parallelizer import _apply_multimodal_tower_ac

            _apply_multimodal_tower_ac(model, ("all",))
            layers["1"] = checkpoint_wrapper(layers["1"])
        ddp = DistributedDataParallel(model)
        reference.load_state_dict(model.state_dict())
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.01)
        for packed, accumulation in ((False, 1), (True, 2), (True, 1)):
            for microstep in range(accumulation):
                all_batches = []
                for dp_rank in range(world_size // 2):
                    docs = _documents()
                    docs[0]["pixel_values"] = docs[0]["pixel_values"] + dp_rank * 0.2 + microstep * 0.1
                    all_batches.append(_pack(docs) if packed else docs[0])
                batch = all_batches[rank // 2]
                labels = batch["input_ids"].clone()
                full_logits = reference(**batch).logits.detach()
                local = copy.deepcopy(batch)
                local["labels"] = labels
                hook = model.prepare_model_inputs_for_cp(local)["cp_sharder"]
                _, local, _ = hook.shard_batch(cp_mesh, None, local)
                context = local["_qwen3_8_flash_next_cp_context"]
                sync = ddp.no_sync() if microstep + 1 < accumulation else contextlib.nullcontext()
                with sync:
                    actual = ddp(**local).logits
                    start = context.local_sequence_start
                    count = min(context.local_sequence_length, labels.shape[1] - start)
                    torch.testing.assert_close(
                        actual[:, :count], full_logits[:, start : start + count], atol=2e-6, rtol=2e-5
                    )
                    # DDP averages DP*CP; the explicit objective sums CP token slices.
                    loss = 2 * actual[:, :count].square().sum() / (full_logits.numel() * accumulation)
                    loss.backward()
                for full_batch in all_batches:
                    (reference(**full_batch).logits.square().mean() / (len(all_batches) * accumulation)).backward()
            for (name, parameter), ref in zip(model.named_parameters(), reference.parameters(), strict=True):
                torch.testing.assert_close(
                    parameter.grad, ref.grad, atol=2e-6, rtol=3e-4, msg=lambda error: f"{name}: {error}"
                )
            clip = 0.1 if packed else float("inf")
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            ref_norm = torch.nn.utils.clip_grad_norm_(reference.parameters(), clip)
            torch.testing.assert_close(norm, ref_norm, atol=2e-6, rtol=3e-4)
            optimizer.step()
            reference_optimizer.step()
            for parameter, ref in zip(model.parameters(), reference.parameters(), strict=True):
                torch.testing.assert_close(parameter, ref, atol=2e-6, rtol=3e-4)
            optimizer.zero_grad(set_to_none=True)
            reference_optimizer.zero_grad(set_to_none=True)
    finally:
        dist.destroy_process_group()


def test_dataset_packer_shift_and_mrope_match_individual_losses() -> None:
    import torch.nn.functional as F

    from nemo_automodel.components.datasets.vlm.collate_fns import neat_packed_vlm_collater
    from nemo_automodel.components.datasets.vlm.neat_packing_vlm import PackedDatasetWrapperConfig

    model = _hybrid_model()
    docs = _documents()
    samples = []
    for doc in docs:
        sample = {**doc, "input_ids": doc["input_ids"][0].clone()}
        sample["labels"] = sample["input_ids"].clone()
        sample["attention_mask"] = torch.ones_like(sample["input_ids"])
        samples.append(sample)
    dataset = PackedDatasetWrapperConfig(pack_size=40).build(
        inner_dataset=samples,
        bins=[[0, 1, 2]],
        padding_idx=0,
        get_rope_index=model.model.get_rope_index,
    )
    batch = neat_packed_vlm_collater([dataset[0]], max_length=40, materialize_4d_mask=False)
    logits = model(**batch).logits
    actual = F.cross_entropy(logits.reshape(-1, 64), batch["labels"].reshape(-1), reduction="sum")
    expected = 0
    for doc in docs:
        shifted = {**doc, "input_ids": doc["input_ids"][:, :-1]}
        for key in ("pixel_values", "pixel_values_videos"):
            if key in shifted:
                shifted[key] = shifted[key].to(torch.bfloat16)
        expected = expected + F.cross_entropy(
            model(**shifted).logits.reshape(-1, 64), doc["input_ids"][:, 1:].reshape(-1), reduction="sum"
        )
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
    actual.backward()
    assert torch.count_nonzero(model.model.visual.patch_embed.proj.weight.grad) > 0


@pytest.mark.parametrize("world_size", [2, 4])
@pytest.mark.runtime_budget(
    60, hard_timeout=90, reason="Real CP2 and DP2xCP2 workers compare hybrid-model gradients and updates"
)
def test_cp_and_dp_cp_media_gradients_norm_and_step(tmp_path: Path, world_size: int) -> None:
    torch.multiprocessing.spawn(
        _cp_worker, args=(world_size, (tmp_path / "rdzv").as_uri()), nprocs=world_size, join=True
    )


@pytest.mark.parametrize("kind", ["interior", "conflicting_mask", "custom_4d", "aligned_thd"])
def test_packed_media_rejects_ambiguous_boundaries(kind: str) -> None:
    from nemo_automodel.components.models.qwen3_8_flash_next.multimodal import normalize_multimodal_packing

    batch = {"input_ids": torch.ones(1, 6, dtype=torch.long)}
    if kind == "interior":
        batch.update(cu_seqlens=torch.tensor([0, 3, 6]), padding_mask=torch.tensor([[0, 0, 1, 0, 0, 0]]))
    elif kind == "conflicting_mask":
        batch.update(_packed_seq_ids=torch.tensor([[1, 1, 1, 2, 2, 2]]), attention_mask=torch.zeros(1, 6))
    elif kind == "custom_4d":
        batch.update(
            _packed_seq_ids=torch.tensor([[1, 1, 1, 2, 2, 2]]),
            attention_mask=torch.ones(1, 1, 6, 6, dtype=torch.bool).tril(),
        )
    else:
        batch.update(seq_lens=torch.tensor([[2, 2]]), seq_lens_padded=torch.tensor([[4, 2]]))
    with pytest.raises(ValueError):
        normalize_multimodal_packing(batch)


def test_cp_packing_recipe_resolves_typed_pipeline_and_capabilities() -> None:
    from types import SimpleNamespace

    import yaml

    from nemo_automodel._transformers.capabilities import ModelSupports
    from nemo_automodel.components.config.loader import ConfigNode
    from nemo_automodel.recipes._typed_config import RecipeConfig

    path = Path(__file__).resolve().parents[4] / (
        "examples/vlm_finetune/qwen3_8_flash_next/qwen3_8_flash_next_180b_medpix_packed4k_cp2_ep64.yaml"
    )
    raw = yaml.safe_load(path.read_text())
    # Resolve the real input-pipeline factories without constructing the 180B
    # model, downloading media, or importing the optional CUDA optimizer.
    cfg = RecipeConfig(
        ConfigNode(
            {
                key: raw[key]
                for key in (
                    "dataset",
                    "dataloader",
                    "processor",
                    "packed_sequence",
                    "validation_dataset",
                    "validation_dataloader",
                )
            }
        )
    )
    loader = cfg.vlm_dataloader
    assert loader.packing.pack_size == 4096
    assert loader.packing.packing_format == "neat"
    assert loader.pretokenization is not None
    assert loader.resolve_packing_attn_implementation(model_attn_implementation="flex", cp_size=2) == "sdpa"
    # The existing recipe evaluates unpacked examples under the same CP mesh.
    assert cfg.vlm_validation_dataloader.packing is None
    model = _hybrid_model()
    model.backend.attn = raw["model"]["backend"]["attn"]
    capabilities = ModelSupports(model, SimpleNamespace(cp_size=2, tp_size=1, pp_size=1, ep_size=64))
    assert capabilities.supports_cp
    assert capabilities.supports_cp_with_sequence_packing


def test_packed_gdn_cuda_dispatch_without_conv_kernel_preserves_documents(monkeypatch: pytest.MonkeyPatch) -> None:
    """Emulate CUDA dispatch on CPU while checking the actual convolution fallback."""
    from nemo_automodel.components.models.qwen3_5_moe.cp_linear_attn import torch_chunk_gated_delta_rule
    from nemo_automodel.components.models.qwen3_8_flash_next import layers as flash_layers

    layer = _hybrid_model().model.language_model.layers["1"].linear_attn
    layer.causal_conv1d_fn = None

    # Use a real torch recurrence with a varlen adapter in place of FLA so the
    # pre-fix path runs too: it resets recurrence but leaks through convolution.
    def varlen_reference(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        g: torch.Tensor,
        beta: torch.Tensor,
        cu_seqlens: torch.Tensor | None = None,
        cu_seqlens_cpu: torch.Tensor | None = None,
        **kwargs: object,
    ) -> tuple[torch.Tensor, None]:
        """Run independent recurrent segments.

        Args:
            q: Queries [1, sequence, heads, key_dim].
            k: Keys with the same layout as q.
            v: Values [1, sequence, heads, value_dim].
            g: Log-decay gates [1, sequence, heads].
            beta: Update gates [1, sequence, heads].
            cu_seqlens: Optional document boundaries [documents + 1].
            cu_seqlens_cpu: Unused CPU mirror of the boundaries.
            **kwargs: Non-tensor recurrence options; initial_state is None.

        Returns:
            Outputs [1, sequence, heads, value_dim] and no final state.
        """
        del cu_seqlens_cpu
        bounds = [0, q.shape[1]] if cu_seqlens is None else cu_seqlens.tolist()
        outputs = []
        for start, end in zip(bounds[:-1], bounds[1:]):
            out, _ = torch_chunk_gated_delta_rule(
                q[:, start:end],
                k[:, start:end],
                v[:, start:end],
                g=g[:, start:end],
                beta=beta[:, start:end],
                **kwargs,
            )
            outputs.append(out)
        return torch.cat(outputs, dim=1), None

    layer.chunk_gated_delta_rule = varlen_reference
    reference = copy.deepcopy(layer)
    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda self: True))
    # Only dispatch is emulated; keep normalization eager on these CPU tensors.
    monkeypatch.setattr(flash_layers, "_rms_norm_gated_fp32_compiled", flash_layers._rms_norm_gated_fp32)
    torch.manual_seed(913)
    x = torch.randn(1, 13, 16, requires_grad=True)
    reference_x = x.detach().clone().requires_grad_(True)
    bounds = [0, 1, 6, 13]  # Includes a document shorter than the convolution kernel.
    packed = layer(x, cu_seqlens=torch.tensor(bounds))
    independent = torch.cat(
        [reference(reference_x[:, start:end]) for start, end in zip(bounds[:-1], bounds[1:])], dim=1
    )
    torch.testing.assert_close(packed, independent, atol=2e-6, rtol=2e-5)
    upstream = torch.randn_like(packed)
    packed.backward(upstream)
    independent.backward(upstream)
    torch.testing.assert_close(x.grad, reference_x.grad, atol=2e-6, rtol=2e-5)
    for (name, parameter), ref in zip(layer.named_parameters(), reference.parameters(), strict=True):
        torch.testing.assert_close(parameter.grad, ref.grad, atol=2e-6, rtol=2e-5, msg=lambda error: f"{name}: {error}")
    changed = x.detach().clone()
    changed[:, :6] += 3
    with torch.no_grad():
        perturbed = layer(changed, cu_seqlens=torch.tensor(bounds))
    torch.testing.assert_close(perturbed[:, 6:], packed[:, 6:].detach(), atol=0, rtol=0)
