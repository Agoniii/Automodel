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

"""Real CPU vision/merger/Engram/QSA training and checkpoint regressions."""

import copy
from datetime import timedelta

import pytest
import torch
import torch.nn.functional as F
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.qwen3_8_flash_next.config import (
    Qwen3_8_FlashNextConfig,
    Qwen3_8_FlashNextTextConfig,
    Qwen3_8_FlashNextVisionConfig,
)
from nemo_automodel.components.models.qwen3_8_flash_next.engram import Qwen3_8_FlashNextEngramTableConfig
from nemo_automodel.components.models.qwen3_8_flash_next.model import Qwen3_8_FlashNextForConditionalGeneration


def _config() -> Qwen3_8_FlashNextConfig:
    return Qwen3_8_FlashNextConfig(
        text_config=Qwen3_8_FlashNextTextConfig(
            vocab_size=64,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
            layer_types=["full_attention"],
            full_attention_interval=1,
            moe_intermediate_size=8,
            shared_expert_intermediate_size=8,
            num_experts=2,
            num_experts_per_tok=1,
            hc_count=2,
            hc_lowrank=4,
            ple_layer_ids=[1],
            ple_embed_dim=32,
            heads_per_ngram=8,
            split_ngram_parts=4,
            indexer_budget=4,
            indexer_n_heads=2,
            indexer_kv_heads=1,
            indexer_head_dim=8,
            max_position_embeddings=64,
            partial_rotary_factor=1.0,
            rope_parameters={
                "rope_theta": 10000.0,
                "rope_type": "default",
                "partial_rotary_factor": 1.0,
                "mrope_section": [1, 1, 2],
                "mrope_interleaved": True,
            },
            dtype="float32",
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=1,
        ),
        vision_config=Qwen3_8_FlashNextVisionConfig(
            depth=1,
            hidden_size=16,
            intermediate_size=32,
            num_heads=2,
            num_position_embeddings=16,
            out_hidden_size=16,
            patch_size=2,
            spatial_merge_size=2,
            temporal_patch_size=2,
        ),
        image_token_id=60,
        video_token_id=61,
        vision_start_token_id=62,
        vision_end_token_id=63,
        language_model_only=False,
    )


def _model(
    config: Qwen3_8_FlashNextConfig | None = None, *, dtype: torch.dtype = torch.float32
) -> Qwen3_8_FlashNextForConditionalGeneration:
    torch.manual_seed(123)
    model = Qwen3_8_FlashNextForConditionalGeneration.from_config(
        config or _config(),
        backend=BackendConfig(
            linear="torch",
            attn="sdpa",
            rms_norm="torch",
            experts="torch",
            dispatcher="torch",
            enable_hf_state_dict_adapter=True,
        ),
        engram_table_config=Qwen3_8_FlashNextEngramTableConfig(
            num_embeddings=256,
            embedding_dim=2,
            initializer_range=0.05,
        ),
    )
    model.initialize_weights(buffer_device=torch.device("cpu"), dtype=dtype)
    # Scale the checkpoint's 16 hash-head ranges to the tiny fixture's table.
    embedding = model.model.language_model.layers["0"].ple.ple_embedding
    embedding.ngram_heads_vocab_sizes.fill_(13)
    embedding.ngram_heads_offsets.copy_(torch.arange(16) * 16)
    return model


def _image_batch() -> dict[str, torch.Tensor]:
    return {
        "input_ids": torch.tensor([[2, 62, 60, 60, 60, 60, 63, 3, 4, 5]]),
        "pixel_values": torch.randn(16, 24),
        "image_grid_thw": torch.tensor([[1, 4, 4]]),
    }


@pytest.mark.parametrize("modality", ["image", "video", "mixed"])
def test_media_training_updates_vision_merger_text_and_engram(modality: str) -> None:
    model = _model().train()
    batch = _image_batch()
    if modality == "video":
        batch = {
            "input_ids": torch.tensor([[2, 62, 61, 61, 61, 61, 63, 3, 62, 61, 61, 61, 61, 63, 4]]),
            "pixel_values_videos": torch.randn(32, 24),
            "video_grid_thw": torch.tensor([[2, 4, 4]]),
        }
    elif modality == "mixed":
        batch = {
            "input_ids": torch.tensor(
                [
                    [2, 62, 60, 60, 60, 60, 63, 3, 62, 61, 61, 61, 61, 63, 4],
                    [2, 3, 4, 5, 6, 7, 8, 9, 0, 0, 0, 0, 0, 0, 0],
                ]
            ),
            "pixel_values": torch.randn(16, 24),
            "image_grid_thw": torch.tensor([[1, 4, 4]]),
            "pixel_values_videos": torch.randn(16, 24),
            "video_grid_thw": torch.tensor([[1, 4, 4]]),
        }
        batch["attention_mask"] = batch["input_ids"] != 0
    tracked = {
        name: parameter
        for name, parameter in model.named_parameters()
        if name
        in {
            "model.visual.patch_embed.proj.weight",
            "model.visual.merger.linear_fc2.weight",
            "model.language_model.layers.0.self_attn.q_proj.weight",
            "model.language_model.layers.0.ple.ple_embedding.ngram_embedding.weight",
        }
    }
    assert len(tracked) == 4
    before = {name: value.detach().clone() for name, value in tracked.items()}
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    logits = model(**batch).logits
    loss = F.cross_entropy(logits[:, :-1].reshape(-1, 64), batch["input_ids"][:, 1:].reshape(-1), ignore_index=0)
    loss.backward()
    assert torch.isfinite(loss)
    for parameter in tracked.values():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert torch.count_nonzero(parameter.grad) > 0
    optimizer.step()
    for name, parameter in tracked.items():
        assert not torch.equal(parameter, before[name])
    indexer = model.model.language_model.layers["0"].self_attn.indexer
    assert all(not parameter.requires_grad and parameter.grad is None for parameter in indexer.parameters())


def test_image_positions_follow_spatial_grid_and_text_continuation() -> None:
    model = _model()
    batch = _image_batch()
    positions, _ = model.model.get_rope_index(batch["input_ids"], image_grid_thw=batch["image_grid_thw"])
    expected = torch.tensor(
        [
            [0, 1, 2, 2, 2, 2, 4, 5, 6, 7],
            [0, 1, 2, 2, 3, 3, 4, 5, 6, 7],
            [0, 1, 2, 3, 2, 3, 4, 5, 6, 7],
        ]
    ).unsqueeze(1)
    torch.testing.assert_close(positions, expected)
    torch.testing.assert_close(model(**batch).logits, model(**batch, position_ids=expected).logits)
    altered = {**batch, "pixel_values": batch["pixel_values"] + 0.5}
    assert not torch.equal(model(**batch).logits, model(**altered).logits)


def test_vision_matches_hf_outputs_and_random_upstream_gradients() -> None:
    model = _model()
    vision = model.model.visual
    reference = Qwen3VLVisionModel(copy.deepcopy(vision.config))
    reference.load_state_dict(vision.state_dict(), strict=True)
    pixels = torch.randn(16, 24)
    grid = torch.tensor([[1, 4, 4]])
    actual = vision(pixels, grid_thw=grid, return_dict=True).pooler_output
    expected = reference(pixels, grid_thw=grid, return_dict=True).pooler_output
    torch.testing.assert_close(actual, expected)
    gradient = torch.randn_like(actual)
    actual.backward(gradient)
    expected.backward(gradient)
    reference_parameters = dict(reference.named_parameters())
    for name, parameter in vision.named_parameters():
        torch.testing.assert_close(parameter.grad, reference_parameters[name].grad)


def test_multimodal_native_save_reload_and_vision_adapter_storage(tmp_path) -> None:
    model = _model()
    batch = _image_batch()
    expected = model(**batch).logits.detach()
    path = tmp_path / "model.pt"
    torch.save(model.state_dict(), path)
    reloaded = _model()
    reloaded.load_state_dict(torch.load(path, weights_only=True), strict=True)
    torch.testing.assert_close(reloaded(**batch).logits, expected)
    vision_state = {key: value for key, value in model.state_dict().items() if key.startswith("model.visual.")}
    exported = model.state_dict_adapter.to_hf(vision_state)
    restored = model.state_dict_adapter.from_hf(exported)
    assert restored.keys() == vision_state.keys()
    for key, value in vision_state.items():
        torch.testing.assert_close(restored[key], value)
        assert exported[key].data_ptr() == value.data_ptr()
    key = "model.visual.patch_embed.proj.weight"
    with torch.no_grad():
        exported[key].fill_(0.125)
    assert torch.all(model.model.visual.patch_embed.proj.weight == 0.125)


@pytest.mark.parametrize("failure", ["pixels", "grid", "count", "positions", "types", "packing"])
def test_invalid_multimodal_inputs_fail_explicitly(failure: str) -> None:
    model = _model()
    batch = _image_batch()
    if failure == "pixels":
        batch.pop("pixel_values")
        message = "requires pixel"
    elif failure == "grid":
        batch.pop("image_grid_thw")
        message = "grid_thw"
    elif failure == "count":
        batch["input_ids"][0, 2] = 2
        message = "count mismatch"
    elif failure == "positions":
        batch["position_ids"] = torch.arange(10).unsqueeze(0)
        message = "position_ids"
    elif failure == "types":
        batch["mm_token_type_ids"] = torch.zeros_like(batch["input_ids"])
        message = "mm_token_type_ids"
    elif failure == "packing":
        batch["cu_seqlens"] = torch.tensor([1, 10])
        message = "boundaries"
    with pytest.raises((ValueError, NotImplementedError), match=message):
        model(**batch)


def test_text_only_batch_keeps_zero_vision_gradients_and_identical_logits() -> None:
    model = _model()
    batch = {"input_ids": torch.tensor([[2, 3, 4, 5]])}
    logits = model(**batch).logits
    logits.square().mean().backward()
    for parameter in model.model.visual.parameters():
        assert parameter.grad is not None
        assert torch.count_nonzero(parameter.grad) == 0
    # Removing the unused tower must not change the language computation.
    model.model.visual = None
    torch.testing.assert_close(model(**batch).logits, logits)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_hybrid_gdn_qsa_multimodal_training(dtype: torch.dtype) -> None:
    config = _config()
    config.text_config.num_hidden_layers = 2
    config.text_config.layer_types = ["full_attention", "linear_attention"]
    config.text_config.linear_num_key_heads = 2
    config.text_config.linear_num_value_heads = 2
    config.text_config.linear_key_head_dim = 8
    config.text_config.linear_value_head_dim = 8
    model = _model(config, dtype=dtype)
    logits = model(**_image_batch()).logits
    logits.square().mean().backward()
    for parameter in (
        model.model.visual.patch_embed.proj.weight,
        model.model.language_model.layers["1"].linear_attn.in_proj_qkv.weight,
    ):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert torch.count_nonzero(parameter.grad) > 0


def test_vision_activation_checkpointing_preserves_gradients() -> None:
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointWrapper

    from nemo_automodel.components.distributed.parallelizer import get_model_layer_groups
    from nemo_automodel.components.moe.parallelizer import _apply_multimodal_tower_ac

    reference, checkpointed = _model(), _model()
    groups = get_model_layer_groups(checkpointed)
    assert groups["vision"] == list(checkpointed.model.visual.blocks)
    assert groups["language"] == list(checkpointed.model.language_model.layers.values())
    _apply_multimodal_tower_ac(checkpointed, ("all",))
    assert any(isinstance(module, CheckpointWrapper) for module in checkpointed.model.visual.modules())
    batch = _image_batch()
    expected, actual = reference(**batch).logits, checkpointed(**batch).logits
    torch.testing.assert_close(actual, expected)
    gradient = torch.randn_like(actual)
    expected.backward(gradient)
    actual.backward(gradient)
    for expected_parameter, actual_parameter in zip(reference.parameters(), checkpointed.parameters(), strict=True):
        torch.testing.assert_close(actual_parameter.grad, expected_parameter.grad)


def test_multimodal_capabilities_and_meta_vision_initialization() -> None:
    config = _config()
    caps = Qwen3_8_FlashNextForConditionalGeneration.get_capabilities(config)
    assert caps.supports_ep and caps.supports_cp and not caps.supports_tp and not caps.supports_pp
    assert caps.supports_thd
    reference = _model()
    # to_empty reproduces discarded storage at the meta materialization boundary.
    model = _model()
    model.model.visual.to(device="meta").to_empty(device="cpu")
    model.initialize_weights(buffer_device=torch.device("cpu"), dtype=torch.bfloat16)
    visual = model.model.visual
    torch.testing.assert_close(visual.rotary_pos_emb.inv_freq, reference.model.visual.rotary_pos_emb.inv_freq)
    assert visual.rotary_pos_emb.inv_freq.dtype == torch.float32
    assert visual.patch_embed.proj.weight.dtype == torch.bfloat16
    features = visual(torch.randn(16, 24, dtype=torch.bfloat16), grid_thw=torch.tensor([[1, 4, 4]])).pooler_output
    assert torch.isfinite(features).all()


@pytest.mark.parametrize("modality", ["image", "video", "mixed"])
def test_real_processor_collator_to_optimizer(modality: str) -> None:
    import numpy as np
    from PIL import Image
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import WhitespaceSplit
    from transformers import PreTrainedTokenizerFast, Qwen2VLImageProcessor, Qwen3VLProcessor, Qwen3VLVideoProcessor

    from nemo_automodel.components.datasets.vlm.collate_fns import default_collate_fn

    vocab = {
        "[PAD]": 0,
        "[UNK]": 1,
        "user": 2,
        "assistant": 3,
        "describe": 4,
        "bright": 5,
        "square": 6,
        "<|im_start|>": 7,
        "<|im_end|>": 8,
        "<|image_pad|>": 60,
        "<|video_pad|>": 61,
        "<|vision_start|>": 62,
        "<|vision_end|>": 63,
    }
    vocab.update({f"unused_{i}": i for i in range(9, 60)})
    tokenizer_backend = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    tokenizer_backend.pre_tokenizer = WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer_backend,
        pad_token="[PAD]",
        unk_token="[UNK]",
        eos_token="<|im_end|>",
        additional_special_tokens=[key for key in vocab if key.startswith("<|")],
        model_max_length=64,
        padding_side="right",
    )
    template = (
        "{% for message in messages %}{{ '<|im_start|>' + message['role'] + '\n' }}"
        "{% for item in message['content'] %}"
        "{% if item['type'] == 'image' %}{{ '<|vision_start|><|image_pad|><|vision_end|> ' }}"
        "{% elif item['type'] == 'video' %}{{ '<|vision_start|><|video_pad|><|vision_end|> ' }}"
        "{% elif item['type'] == 'text' %}{{ item['text'] + ' ' }}{% endif %}"
        "{% endfor %}{{ '<|im_end|>\n' }}{% endfor %}"
    )
    processor = Qwen3VLProcessor(
        tokenizer=tokenizer,
        chat_template=template,
        image_processor=Qwen2VLImageProcessor(
            patch_size=2, temporal_patch_size=2, merge_size=2, size={"shortest_edge": 64, "longest_edge": 64}
        ),
        video_processor=Qwen3VLVideoProcessor(
            patch_size=2,
            temporal_patch_size=2,
            merge_size=2,
            size={"shortest_edge": 64, "longest_edge": 64},
            do_sample_frames=False,
        ),
    )
    examples = [
        {
            "conversation": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": Image.new("RGB", (8, 8), (200, 40, 20))},
                        {"type": "text", "text": "describe"},
                    ],
                },
                {"role": "assistant", "content": [{"type": "text", "text": "bright square"}]},
            ]
        }
    ]
    if modality != "image":
        video = {"type": "video", "video": np.full((4, 8, 8, 3), 128, dtype=np.uint8)}
        content = examples[0]["conversation"][0]["content"]
        if modality == "video":
            content[0] = video
        else:
            content.insert(1, video)
    batch = default_collate_fn(examples, processor)
    labels = batch.pop("labels")
    assert torch.count_nonzero(labels != -100) >= 2
    if modality != "video":
        assert batch["pixel_values"].shape == (16, 24)
    if modality != "image":
        assert batch["pixel_values_videos"].shape == (8, 24)
    model = _model()
    parameter = model.model.visual.patch_embed.proj.weight
    before = parameter.detach().clone()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss = F.cross_entropy(model(**batch).logits.reshape(-1, 64), labels.reshape(-1))
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.count_nonzero(parameter.grad) > 0
    optimizer.step()
    assert not torch.equal(before, parameter)


def _mixed_rank_training_worker(rank: int, rendezvous: str) -> None:
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel

    torch.set_num_threads(1)
    # Construct before initializing distributed: this test compares replicated
    # DDP parameters, including a replicated tiny Engram table, against serial.
    config = _config()
    # Compare the explicit mean objective. MoE's separately scaled auxiliary
    # loss needs the recipe's microbatch scaler and is outside this DDP test.
    config.text_config.router_aux_loss_coef = 0.0
    model, reference = _model(config), _model(config)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2, timeout=timedelta(seconds=30))
    try:
        ddp = DistributedDataParallel(model)
        reference.load_state_dict(model.state_dict())
        optimizer = torch.optim.AdamW(ddp.parameters(), lr=1e-3)
        reference_optimizer = torch.optim.AdamW(reference.parameters(), lr=1e-3)
        for step in range(2):
            torch.manual_seed(900 + step)
            batches = [_image_batch(), {"input_ids": torch.tensor([[2, 3, 4, 5]])}]
            if step:
                batches[1] = {
                    "input_ids": torch.tensor([[2, 62, 61, 61, 61, 61, 63, 3]]),
                    "pixel_values_videos": torch.randn(16, 24),
                    "video_grid_thw": torch.tensor([[1, 4, 4]]),
                }
            local_logits = ddp(**batches[rank]).logits
            torch.testing.assert_close(local_logits, reference(**batches[rank]).logits)
            local_logits.square().mean().backward()
            for batch in batches:
                (reference(**batch).logits.square().mean() / 2).backward()
            for (name, actual), expected in zip(model.named_parameters(), reference.parameters(), strict=True):
                torch.testing.assert_close(
                    actual.grad,
                    expected.grad,
                    rtol=2e-4,
                    atol=1e-6,
                    msg=lambda message: f"step={step} {name}: {message}",
                )
            optimizer.step()
            reference_optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            reference_optimizer.zero_grad(set_to_none=True)
    finally:
        dist.destroy_process_group()


@pytest.mark.runtime_budget(30, hard_timeout=60, reason="Two real Gloo workers import models and run two steps")
def test_two_rank_image_text_video_gradients_match_serial_reference(tmp_path) -> None:
    # Real collectives, two optimization steps; the larger budget covers worker startup.
    torch.multiprocessing.start_processes(
        _mixed_rank_training_worker,
        args=((tmp_path / "rendezvous").as_uri(),),
        nprocs=2,
        join=True,
        start_method="spawn",
    )


def test_local_checkpoint_config_retains_vision_tower(tmp_path) -> None:
    config = _config()
    config.save_pretrained(tmp_path)
    model = Qwen3_8_FlashNextForConditionalGeneration.from_pretrained(
        str(tmp_path),
        backend=BackendConfig(linear="torch", attn="sdpa", rms_norm="torch", experts="torch", dispatcher="torch"),
        engram_table_config=Qwen3_8_FlashNextEngramTableConfig(num_embeddings=256, embedding_dim=2),
    )
    assert model.config.language_model_only is False
    assert model.model.visual is not None
