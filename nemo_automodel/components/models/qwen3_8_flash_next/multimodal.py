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

"""Image/video embedding and position contracts for Flash-Next training."""

from __future__ import annotations

import contextlib
from collections.abc import Callable
from typing import Any

import torch
from torch import nn
from torch.distributed.device_mesh import DeviceMesh

from nemo_automodel.components.distributed.context_parallel.sharder import ShardLayout
from nemo_automodel.components.distributed.cp_vision_frame_shard import maybe_distribute_visual
from nemo_automodel.components.models.qwen3_8_flash_next.config import Qwen3_8_FlashNextConfig
from nemo_automodel.shared.import_utils import safe_import

from .cp import packed_boundaries_from_seq_lens, shard_batch_for_qwen3_8_flash_next_cp

_, hf_vl = safe_import("transformers.models.qwen3_vl.modeling_qwen3_vl")


def normalize_multimodal_packing(batch: dict[str, Any]) -> None:
    """Normalize packed VLM metadata without changing the physical token layout.

    Args:
        batch: Mutable tensors: IDs/masks [1, sequence], optional indexed document
            map [1, sequence], cu_seqlens [documents + 1], or seq_lens [1, documents].
            Produces cu_seqlens and a boolean padding_mask [1, sequence]. Trailing
            pack padding is isolated as a separate segment. Media are untouched.
    """
    ids = batch["input_ids"]
    indexed = batch.pop("_packed_seq_ids", None)
    mask = batch.get("attention_mask")
    if indexed is None and mask is not None and mask.ndim == 4:
        # The neat collator omits document IDs for a single-document pack.
        # Accept precisely its ordinary padded causal mask, not arbitrary 4D attention.
        valid = mask[:, 0].diagonal(dim1=-2, dim2=-1).bool()
        expected = torch.ones_like(mask, dtype=torch.bool).tril() & valid[:, None, :, None] & valid[:, None, None, :]
        if (
            mask.dtype != torch.bool
            or mask.shape != (ids.shape[0], 1, ids.shape[1], ids.shape[1])
            or not torch.equal(mask, expected)
        ):
            raise ValueError("Nonstandard packed masks require explicit _packed_seq_ids")
        indexed = valid.long()
    lengths = batch.pop("seq_lens", None)
    padded = batch.pop("seq_lens_padded", None)
    boundaries = batch.get("cu_seqlens")
    if indexed is None and lengths is None and boundaries is None:
        if batch.get("qkv_format") == "thd":
            raise ValueError("Packed VLM inputs require document boundaries")
        return
    if ids.shape[0] != 1:
        raise ValueError("Flash-Next packing requires one physical row per microbatch")
    if indexed is not None:
        if indexed.shape != ids.shape or bool((indexed < 0).any()):
            raise ValueError("Packed document IDs must be nonnegative [1, sequence]")
        if mask is not None:
            if mask.ndim == 4:
                expected = (
                    (indexed[:, None, :, None] == indexed[:, None, None, :])
                    & (indexed[:, None, :, None] > 0)
                    & (indexed[:, None, None, :] > 0)
                ).tril()
                matches = mask.dtype == torch.bool and torch.equal(mask, expected)
            else:
                matches = torch.equal(mask, indexed) or torch.equal(mask, indexed > 0)
            if not matches:
                raise ValueError("Packed attention_mask disagrees with document IDs")
        runs, counts = torch.unique_consecutive(indexed[0], return_counts=True)
        valid_runs = runs[runs != 0]
        if not torch.equal(valid_runs, torch.arange(1, valid_runs.numel() + 1, device=runs.device)):
            raise ValueError("Packed document IDs must be consecutive increasing segments")
        if bool((runs[:-1] == 0).any()):
            raise ValueError("Packed padding is supported only at the row tail")
        derived = torch.cat((counts.new_zeros(1), counts.cumsum(0)))
        if boundaries is not None and not torch.equal(boundaries.to(derived), derived):
            raise ValueError("Conflicting packed boundaries and document IDs")
        boundaries = derived
        batch["padding_mask"] = indexed == 0
    elif lengths is not None:
        real = lengths.reshape(-1)
        real = real[real != -1000]
        if padded is not None:
            slots = padded.reshape(-1)
            slots = slots[slots != -1000]
            if slots.shape != real.shape or not torch.equal(slots[:-1], real[:-1]):
                raise ValueError("Per-document internal padding is unsupported; use sequence_alignment=1")
        derived = packed_boundaries_from_seq_lens(lengths, total_tokens=ids.shape[1])
        if boundaries is not None and not torch.equal(boundaries.to(derived), derived):
            raise ValueError("Conflicting cu_seqlens and seq_lens")
        boundaries = derived
        batch["padding_mask"] = torch.arange(ids.shape[1], device=ids.device)[None] >= real.sum()
    boundaries = boundaries.to(device=ids.device, dtype=torch.long).reshape(-1)
    if (
        boundaries.numel() < 2
        or int(boundaries[0]) != 0
        or int(boundaries[-1]) != ids.shape[1]
        or bool((boundaries.diff() <= 0).any())
    ):
        raise ValueError("Packed boundaries must increase from zero to the physical sequence length")
    batch["cu_seqlens"] = boundaries
    mask = batch.pop("attention_mask", None)
    if indexed is None and mask is not None:
        if mask.shape != ids.shape or not bool(((mask == 0) | (mask == 1)).all()):
            raise ValueError("Packed masks require explicit document IDs or binary validity")
        if batch.get("padding_mask") is not None and not torch.equal(batch["padding_mask"], ~mask.bool()):
            raise ValueError("Packed attention_mask and padding metadata disagree")
        batch["padding_mask"] = ~mask.bool()
    padding = batch.get("padding_mask")
    if padding is not None:
        if padding.shape != ids.shape or not bool(((padding == 0) | (padding == 1)).all()):
            raise ValueError("Packed padding_mask must contain binary values with shape [1, sequence]")
        valid = ~padding.bool()
        if not torch.equal(valid, torch.arange(ids.shape[1], device=ids.device)[None] < valid.sum(-1, keepdim=True)):
            raise ValueError("Packed padding is supported only at the row tail")
        batch["padding_mask"] = ~valid
    batch.pop("qkv_format", None)


class Qwen3_8_FlashNextMultimodalMixin:
    """Reuse the checkpoint's Qwen3-VL position rules without constructing its LM.

    The owning model registers ``visual`` at ``model.visual`` so vision weights
    retain their checkpoint names. Only embeddings are replaced: original IDs
    continue into the Flash-Next decoder for Engram hashing.
    """

    config: Qwen3_8_FlashNextConfig
    visual: nn.Module | None

    def prepare_multimodal_positions(self, batch: dict[str, Any]) -> None:
        """Normalize packing and create global mRoPE before sequence sharding.

        Args:
            batch: Full token IDs [batch, sequence], optional positions
                [3 or 4, batch, sequence], media patches [patches, patch_width],
                grids [media_items, 3] and packed metadata as documented by
                normalize_multimodal_packing. Updates positions and metadata in place.
        """
        normalize_multimodal_packing(batch)
        ids = batch["input_ids"]
        has_media = batch.get("pixel_values") is not None or batch.get("pixel_values_videos") is not None
        positions = batch.get("position_ids")
        if positions is None:
            mask = batch.get("attention_mask")
            if mask is None and batch.get("padding_mask") is not None:
                mask = ~batch["padding_mask"]
            positions, _ = self.get_rope_index(
                ids, batch.get("mm_token_type_ids"), batch.get("image_grid_thw"), batch.get("video_grid_thw"), mask
            )
            boundaries = batch.get("cu_seqlens")
            if boundaries is not None:
                positions = positions.clone()
                for start, end in zip(boundaries[:-1].tolist(), boundaries[1:].tolist()):
                    positions[:, :, start:end] -= positions[0:1, :, start : start + 1].clone()
            batch["position_ids"] = positions
        elif has_media and (
            positions.ndim != 3 or positions.shape[0] not in (3, 4) or positions.shape[1:] != ids.shape
        ):
            raise ValueError("Multimodal position_ids must have shape [3 or 4, batch, sequence]")

    def shard_multimodal_batch(
        self,
        cp_mesh: DeviceMesh,
        tp_mesh: DeviceMesh | None,
        batch: dict[str, Any],
        *,
        loss_mask: torch.Tensor | None = None,
        padding_token_id: int = 0,
        pad_multiple: int = 4,
    ) -> tuple[Callable[[], contextlib.AbstractContextManager[Any]], dict[str, Any], ShardLayout]:
        """Prepare positions and shard text, retaining replicated media for forward.

        Args:
            cp_mesh: Contiguous context-parallel mesh.
            tp_mesh: Optional size-one TP mesh.
            batch: Global IDs/labels [batch, sequence], media [patches, patch_width],
                grids [items, 3], and optional positions [3 or 4, batch, sequence].
            loss_mask: Optional global token weights [batch, sequence].
            padding_token_id: Token used for CP divisibility padding.
            pad_multiple: Required per-rank sequence multiple.

        Returns:
            Context factory, mutated local-token batch with replicated media/global
            IDs in CP context, and pre/post-padding global sequence lengths.
        """
        self.prepare_multimodal_positions(batch)
        batch.pop("mm_token_type_ids", None)  # Derived and validated before sharding.
        return shard_batch_for_qwen3_8_flash_next_cp(
            cp_mesh,
            tp_mesh,
            batch,
            loss_mask=loss_mask,
            padding_token_id=padding_token_id,
            pad_multiple=pad_multiple,
        )

    def get_vision_position_ids(
        self,
        start_position: int,
        grid_thw: torch.Tensor,
        temp_merge_size: int = 1,
        spatial_merge_size: int = 1,
        time_interval: int = 1,
        device: str | torch.device | None = None,
    ) -> torch.Tensor:
        """Apply the upstream Qwen3-VL per-media position contract.

        Args:
            start_position: Position offset after the preceding text.
            grid_thw: Integer tensor of shape [3], storing temporal/height/width.
            temp_merge_size: Temporal grid downsampling factor.
            spatial_merge_size: Spatial grid downsampling factor.
            time_interval: Scalar temporal position spacing.
            device: Output device.

        Returns:
            Integer positions of shape [3, merged_media_tokens], in T/H/W order.
        """
        return hf_vl.Qwen3VLModel.get_vision_position_ids(
            self, start_position, grid_thw, temp_merge_size, spatial_merge_size, time_interval, device
        )

    def get_rope_index(
        self,
        input_ids: torch.Tensor,
        mm_token_type_ids: torch.Tensor | None = None,
        image_grid_thw: torch.Tensor | None = None,
        video_grid_thw: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute mRoPE positions using Qwen3-VL's timestamped-video rules.

        Args:
            input_ids: Original tokenizer IDs of shape [batch, sequence].
            mm_token_type_ids: Optional modality IDs [batch, sequence], with
                text/image/video encoded as 0/1/2. If absent, derive from IDs.
            image_grid_thw: Image patch grids of shape [images, 3].
            video_grid_thw: Video patch grids of shape [videos, 3].
            attention_mask: Optional binary valid-token mask [batch, sequence].

        Returns:
            T/H/W positions [3, batch, sequence] and position deltas [batch, 1].
        """
        expected_types = torch.zeros_like(input_ids)
        expected_types.masked_fill_(input_ids == self.config.image_token_id, 1)
        expected_types.masked_fill_(input_ids == self.config.video_token_id, 2)
        for modality, grid in ((1, image_grid_thw), (2, video_grid_thw)):
            token_count = int((expected_types == modality).sum())
            if token_count and grid is None:
                raise ValueError("Media placeholders require grid_thw for mRoPE")
            if grid is not None:
                if grid.ndim != 2 or grid.shape[1] != 3:
                    raise ValueError("Media grid_thw must have shape [media_items, 3]")
                expected = int(grid.prod(-1).sum()) // self.config.vision_config.spatial_merge_size**2
                if token_count != expected:
                    raise ValueError(f"Media token/feature count mismatch: tokens={token_count}, features={expected}")
        if mm_token_type_ids is not None and not torch.equal(mm_token_type_ids, expected_types):
            raise ValueError("mm_token_type_ids must match the image/video placeholders in input_ids")
        return hf_vl.Qwen3VLModel.get_rope_index(
            self,
            input_ids=input_ids,
            mm_token_type_ids=expected_types,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            attention_mask=attention_mask,
        )

    def _splice_vision_embeddings(
        self,
        input_ids: torch.Tensor,
        inputs_embeds: torch.Tensor,
        *,
        pixel_values: torch.Tensor | None,
        pixel_values_videos: torch.Tensor | None,
        image_grid_thw: torch.Tensor | None,
        video_grid_thw: torch.Tensor | None,
        sequence_start: int = 0,
    ) -> torch.Tensor:
        """Encode both modalities in one tower call and splice their features.

        Args:
            input_ids: Global original tokenizer IDs [batch, global_sequence].
            inputs_embeds: Local token embeddings [batch, local_sequence, hidden].
                Without CP, local_sequence equals global_sequence.
            pixel_values: Flattened image patches [patches, channels * temporal_patch
                * patch_height * patch_width], in processor order.
            pixel_values_videos: Video patches with the same flattened layout.
            image_grid_thw: Integer image patch grids [images, 3], in token order.
            video_grid_thw: Integer video patch grids [videos, 3], in token order.
            sequence_start: Inclusive global token offset of the local embedding
                slice. The same interval is selected from every batch row.

        Returns:
            Embeddings [batch, local_sequence, hidden]; with media, a new tensor
            whose gradient flows into the vision tower and merger. No global
            [batch, global_sequence, hidden] embedding buffer is constructed.
        """
        if self.visual is None:
            raise ValueError("Image/video inputs require language_model_only=False")
        sequence_end = sequence_start + inputs_embeds.shape[1]
        if (
            inputs_embeds.shape[0] != input_ids.shape[0]
            or not 0 <= sequence_start <= sequence_end <= input_ids.shape[1]
        ):
            raise ValueError("Local embeddings must describe a valid sequence slice of the global input_ids")
        vision = self.config.vision_config
        merge = vision.spatial_merge_size
        patch_width = vision.in_channels * vision.temporal_patch_size * vision.patch_size**2
        patches, grids, masks, counts = [], [], [], []
        for pixels, grid, token_id in (
            (pixel_values, image_grid_thw, self.config.image_token_id),
            (pixel_values_videos, video_grid_thw, self.config.video_token_id),
        ):
            mask = input_ids == token_id
            token_count = int(mask.sum())
            if pixels is None:
                if token_count or grid is not None:
                    raise ValueError(f"Media token {token_id} requires pixel values and grid_thw")
                continue
            if grid is None or grid.ndim != 2 or grid.shape[1] != 3:
                raise ValueError("Media grid_thw must have shape [media_items, 3]")
            if grid.dtype not in (torch.int32, torch.int64) or bool((grid <= 0).any()):
                raise ValueError("Media grid_thw must contain positive integers")
            if bool((grid[:, 1:] % merge != 0).any()):
                raise ValueError("Media grid height and width must be divisible by spatial_merge_size")
            patch_count = int(grid.prod(-1).sum())
            if pixels.shape != (patch_count, patch_width):
                raise ValueError(f"Expected pixel_values shape {(patch_count, patch_width)}, got {tuple(pixels.shape)}")
            expected_tokens = patch_count // merge**2
            if not token_count or token_count != expected_tokens:
                raise ValueError(
                    f"Media token/feature count mismatch: tokens={token_count}, features={expected_tokens}"
                )
            patches.append(pixels.to(device=inputs_embeds.device, dtype=self.visual.dtype))
            grids.append(grid.to(device=inputs_embeds.device))
            masks.append(mask)
            counts.append(token_count)
        if not patches:
            # Every rank executes the same number of vision FSDP collectives,
            # including text-only ranks. The zero edge also runs its backward.
            patches.append(inputs_embeds.new_zeros((merge**2, patch_width), dtype=self.visual.dtype))
            grids.append(input_ids.new_tensor([[1, merge, merge]]))
        # The recipe publishes a CP-only group for this generic, opt-in policy.
        # Its differentiable gather routes gradients to each frame's compute rank.
        features = maybe_distribute_visual(self.visual, torch.cat(patches), torch.cat(grids)).pooler_output
        features = features.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype)
        if not counts:
            return inputs_embeds + features.sum() * 0
        for mask, media_features in zip(masks, features.split(counts), strict=True):
            local_mask = mask[:, sequence_start:sequence_end]
            if sequence_start != 0 or sequence_end != input_ids.shape[1]:
                # Feature rows follow row-major placeholder order across the
                # entire batch, not just within an individual sample or rank.
                feature_rows = mask.flatten().long().cumsum(0).view_as(mask) - 1
                local_rows = feature_rows[:, sequence_start:sequence_end][local_mask]
                media_features = media_features[local_rows]
            # Keep the empty source in the graph too: ranks without local media
            # must still participate in vision gather/FSDP backward collectives.
            inputs_embeds = inputs_embeds.masked_scatter(
                local_mask.unsqueeze(-1).expand_as(inputs_embeds), media_features
            )
        return inputs_embeds
