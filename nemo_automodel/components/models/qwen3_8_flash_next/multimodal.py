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

import torch
from torch import nn

from nemo_automodel.components.models.qwen3_8_flash_next.config import Qwen3_8_FlashNextConfig
from nemo_automodel.shared.import_utils import safe_import

_, hf_vl = safe_import("transformers.models.qwen3_vl.modeling_qwen3_vl")


class Qwen3_8_FlashNextMultimodalMixin:
    """Reuse the checkpoint's Qwen3-VL position rules without constructing its LM.

    The owning model registers ``visual`` at ``model.visual`` so vision weights
    retain their checkpoint names. Only embeddings are replaced: original IDs
    continue into the Flash-Next decoder for Engram hashing.
    """

    config: Qwen3_8_FlashNextConfig
    visual: nn.Module | None

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
    ) -> torch.Tensor:
        """Encode both modalities in one tower call and splice their features.

        Args:
            input_ids: Original tokenizer IDs [batch, sequence].
            inputs_embeds: Token embeddings [batch, sequence, hidden].
            pixel_values: Flattened image patches [patches, channels * temporal_patch
                * patch_height * patch_width], in processor order.
            pixel_values_videos: Video patches with the same flattened layout.
            image_grid_thw: Integer image patch grids [images, 3], in token order.
            video_grid_thw: Integer video patch grids [videos, 3], in token order.

        Returns:
            Embeddings [batch, sequence, hidden]; with media, a new tensor whose
            gradient flows into the vision tower and merger.
        """
        if self.visual is None:
            raise ValueError("Image/video inputs require language_model_only=False")
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
        features = self.visual(torch.cat(patches), grid_thw=torch.cat(grids), return_dict=True).pooler_output
        features = features.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype)
        if not counts:
            return inputs_embeds + features.sum() * 0
        for mask, media_features in zip(masks, features.split(counts), strict=True):
            inputs_embeds = inputs_embeds.masked_scatter(mask.unsqueeze(-1).expand_as(inputs_embeds), media_features)
        return inputs_embeds
