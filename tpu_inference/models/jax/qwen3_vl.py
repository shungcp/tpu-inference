# Copyright 2025 Google LLC
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
"""Qwen3-VL model implementation for TPU inference.

Key differences from Qwen2.5-VL:
- Vision encoder uses LayerNorm (not RMSNorm), patch_size=16, no windowed
  attention, and a learnable pos_embed added to patch embeddings.
- LLM backbone is Qwen3 (with QK-Norm), configured via nested text_config.
- Deepstack: intermediate vision features at layers [5, 11, 17] are extracted
  and injected into LLM hidden states at layers [0, 1, 2] at image token
  positions.
"""

import math
from functools import partial
from itertools import islice
from typing import Callable, List, Literal, NamedTuple, Optional, Tuple, TypedDict, Union

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax.sharding import Mesh
from transformers.models.qwen3_vl.configuration_qwen3_vl import (
    Qwen3VLConfig, Qwen3VLVisionConfig)
from vllm.config import VllmConfig

from tpu_inference import utils as utils
from tpu_inference.distributed.jax_parallel_state import get_pp_group
from tpu_inference.layers.common.attention_interface import \
    sharded_flash_attention
from tpu_inference.layers.common.attention_metadata import AttentionMetadata
from tpu_inference.layers.jax.linear import JaxEinsum
from tpu_inference.layers.jax.pp_utils import PPMissingLayer
from tpu_inference.logger import init_logger
from tpu_inference.models.jax.jax_intermediate_tensor import \
    JaxIntermediateTensors
from tpu_inference.models.jax.qwen2_5_vl import (
    SegmentIds, apply_rotary_pos_emb_vision,
    Qwen2_5_VisionRotaryEmbedding as Qwen3VLVisionRotaryEmbedding,
    generate_window_segment_ids as _generate_segment_ids)
from tpu_inference.models.jax.qwen3 import Qwen3Model
from tpu_inference.models.jax.utils.multi_modal_utils import (
    MultiModalEmbeddings, merge_multimodal_embeddings)
from tpu_inference.models.jax.utils.weight_utils import StandardWeightLoader

logger = init_logger(__name__)

init_fn = nnx.initializers.uniform()

DEFAULT_BLOCK_K_MAJOR = 128

# gelu with tanh approximation (matches PyTorch's gelu_pytorch_tanh)
_gelu_tanh = partial(jax.nn.gelu, approximate=True)


# ---------------------------------------------------------------------------
# Input type definitions
# ---------------------------------------------------------------------------

class Qwen3VLImagePixelInputs(TypedDict):
    type: Literal["pixel_values"]
    pixel_values: jax.Array
    image_grid_thw: tuple[tuple[int, int, int], ...]


# ---------------------------------------------------------------------------
# Vision encoder components
# ---------------------------------------------------------------------------

class Qwen3VLVisionMLP(nnx.Module):
    """Two-layer MLP used inside vision transformer blocks.

    Unlike the LLM MLP there is no gating projection: just fc1 → gelu → fc2.
    Both linear layers include a bias term.
    """

    def __init__(self, config: Qwen3VLVisionConfig, dtype: jnp.dtype,
                 rngs: nnx.Rngs):
        self.fc1 = nnx.Linear(
            config.hidden_size,
            config.intermediate_size,
            use_bias=True,
            param_dtype=dtype,
            kernel_init=nnx.with_partitioning(init_fn, (None, "model")),
            bias_init=nnx.with_partitioning(init_fn, ("model",)),
            rngs=rngs,
        )
        self.fc2 = nnx.Linear(
            config.intermediate_size,
            config.hidden_size,
            use_bias=True,
            param_dtype=dtype,
            kernel_init=nnx.with_partitioning(init_fn, ("model", None)),
            bias_init=nnx.with_partitioning(init_fn, (None,)),
            rngs=rngs,
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.fc2(_gelu_tanh(self.fc1(x)))


class Qwen3VLVisionAttention(nnx.Module):
    """Full attention for the Qwen3-VL vision transformer.

    Every block uses full (non-windowed) attention.  Different images in the
    batch are kept separate via segment IDs derived from cu_seqlens.
    """

    def __init__(self, config: Qwen3VLVisionConfig, dtype: jnp.dtype,
                 rngs: nnx.Rngs, mesh: Mesh):
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_size // config.num_heads

        sharding_size = mesh.shape["model"]
        self.num_heads = utils.get_padded_num_heads(self.num_heads,
                                                    sharding_size)
        self.mesh = mesh

        self.qkv_proj = nnx.Linear(
            self.hidden_size,
            3 * self.hidden_size,
            use_bias=True,
            param_dtype=dtype,
            kernel_init=nnx.with_partitioning(init_fn, (None, "model")),
            bias_init=nnx.with_partitioning(init_fn, ("model",)),
            rngs=rngs,
        )
        self.proj = nnx.Linear(
            self.hidden_size,
            self.hidden_size,
            use_bias=True,
            param_dtype=dtype,
            kernel_init=nnx.with_partitioning(init_fn, ("model", None)),
            bias_init=nnx.with_partitioning(init_fn, (None,)),
            rngs=rngs,
        )
        self.flash_attention = sharded_flash_attention(
            mesh=mesh,
            causal=False,
            sm_scale=1.0 / math.sqrt(self.head_dim),
            vmem_limit_bytes=128 * 1024 * 1024,
        )

    def __call__(
        self,
        x: jax.Array,
        rotary_pos_emb: jax.Array,
        cu_seqlens: jax.Array,
    ) -> jax.Array:
        T, B, D = x.shape
        assert B == 1, "Vision attention only supports batch size 1"

        qkv = self.qkv_proj(x)  # [T, B, 3*D]
        q, k, v = jnp.split(qkv, 3, axis=-1)  # 3 x [T, B, D]

        q = q.reshape(T, B, self.num_heads, self.head_dim)
        k = k.reshape(T, B, self.num_heads, self.head_dim)
        v = v.reshape(T, B, self.num_heads, self.head_dim)

        # [T, B, N, H] -> [B, T, N, H]
        q = jnp.transpose(q, (1, 0, 2, 3))
        k = jnp.transpose(k, (1, 0, 2, 3))
        v = jnp.transpose(v, (1, 0, 2, 3))

        q = apply_rotary_pos_emb_vision(q, rotary_pos_emb)
        k = apply_rotary_pos_emb_vision(k, rotary_pos_emb)

        # [B, T, N, H] -> [B, N, T, H]
        q = jnp.transpose(q, (0, 2, 1, 3))
        k = jnp.transpose(k, (0, 2, 1, 3))
        v = jnp.transpose(v, (0, 2, 1, 3))

        # Pad T to a multiple of DEFAULT_BLOCK_K_MAJOR for flash attention.
        T_attn = q.shape[2]
        padded_T = (T_attn + DEFAULT_BLOCK_K_MAJOR - 1) // DEFAULT_BLOCK_K_MAJOR * DEFAULT_BLOCK_K_MAJOR
        pad_width = ((0, 0), (0, 0), (0, padded_T - T_attn), (0, 0))
        q = jnp.pad(q, pad_width)
        k = jnp.pad(k, pad_width)
        v = jnp.pad(v, pad_width)

        segment_ids = _generate_segment_ids(cu_seqlens, T_attn, padded_T)
        output = self.flash_attention(q, k, v, segment_ids)
        output = output[:, :, :T_attn, :]  # unpad

        # [B, N, T, H] -> [T, B, D]
        output = jnp.transpose(output, (2, 0, 1, 3))
        output = output.reshape(T, B, D)
        return self.proj(output)


class Qwen3VLVisionBlock(nnx.Module):
    """Single transformer block in the Qwen3-VL vision encoder.

    Uses LayerNorm (not RMSNorm) for both pre-attention and pre-MLP norms.
    """

    def __init__(self, config: Qwen3VLVisionConfig, dtype: jnp.dtype,
                 rngs: nnx.Rngs, mesh: Mesh):
        dim = config.hidden_size
        self.norm1 = nnx.LayerNorm(
            dim,
            epsilon=1e-6,
            param_dtype=dtype,
            scale_init=nnx.with_partitioning(init_fn, (None,)),
            bias_init=nnx.with_partitioning(init_fn, (None,)),
            rngs=rngs,
        )
        self.norm2 = nnx.LayerNorm(
            dim,
            epsilon=1e-6,
            param_dtype=dtype,
            scale_init=nnx.with_partitioning(init_fn, (None,)),
            bias_init=nnx.with_partitioning(init_fn, (None,)),
            rngs=rngs,
        )
        self.attn = Qwen3VLVisionAttention(config=config, dtype=dtype,
                                           rngs=rngs, mesh=mesh)
        self.mlp = Qwen3VLVisionMLP(config=config, dtype=dtype, rngs=rngs)

    def __call__(self, x: jax.Array, rotary_pos_emb: jax.Array,
                 cu_seqlens: jax.Array) -> jax.Array:
        x = x + self.attn(self.norm1(x), rotary_pos_emb, cu_seqlens)
        x = x + self.mlp(self.norm2(x))
        return x


class Qwen3VLVisionPatchEmbed(nnx.Module):
    """3-D convolutional patch embedding (with bias, patch_size=16)."""

    def __init__(self, config: Qwen3VLVisionConfig, dtype: jnp.dtype,
                 rngs: nnx.Rngs):
        self.patch_size = config.patch_size
        self.temporal_patch_size = config.temporal_patch_size
        self.hidden_size = config.hidden_size
        kernel_size = (config.temporal_patch_size, config.patch_size,
                       config.patch_size)
        self.proj = nnx.Conv(
            in_features=config.in_channels,
            out_features=config.hidden_size,
            kernel_size=kernel_size,
            strides=kernel_size,
            use_bias=True,
            param_dtype=dtype,
            kernel_init=nnx.with_partitioning(
                init_fn, (None, None, None, None, "model")),
            bias_init=nnx.with_partitioning(init_fn, ("model",)),
            rngs=rngs,
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        # x: [L, C * T * H * W]
        L, dim = x.shape
        C = dim // (self.temporal_patch_size * self.patch_size * self.patch_size)
        x = x.reshape(L, C, self.temporal_patch_size, self.patch_size,
                      self.patch_size)
        x = jnp.transpose(x, (0, 2, 3, 4, 1))  # [L, T, H, W, C]
        x = self.proj(x)  # [L, 1, 1, 1, hidden]
        return x.reshape(L, self.hidden_size)


class Qwen3VLVisionPatchMerger(nnx.Module):
    """Spatial patch merger with LayerNorm.

    Used both as the final merger (use_postshuffle_norm=False) and as the
    deepstack intermediate mergers (use_postshuffle_norm=True).

    When use_postshuffle_norm=False the LayerNorm acts on the per-token
    hidden size *before* spatial merging; when True it acts on the merged
    (shuffled) hidden size *after* spatial merging.
    """

    def __init__(self, config: Qwen3VLVisionConfig, dtype: jnp.dtype,
                 rngs: nnx.Rngs, use_postshuffle_norm: bool = False):
        self.use_postshuffle_norm = use_postshuffle_norm
        self.spatial_merge_size = config.spatial_merge_size
        self.spatial_merge_unit = config.spatial_merge_size ** 2
        merged_size = config.hidden_size * self.spatial_merge_unit
        norm_size = merged_size if use_postshuffle_norm else config.hidden_size
        self.norm = nnx.LayerNorm(
            norm_size,
            epsilon=1e-6,
            param_dtype=dtype,
            scale_init=nnx.with_partitioning(init_fn, (None,)),
            bias_init=nnx.with_partitioning(init_fn, (None,)),
            rngs=rngs,
        )
        self.fc1 = nnx.Linear(
            merged_size,
            merged_size,
            use_bias=True,
            param_dtype=dtype,
            kernel_init=nnx.with_partitioning(init_fn, (None, "model")),
            bias_init=nnx.with_partitioning(init_fn, ("model",)),
            rngs=rngs,
        )
        self.fc2 = nnx.Linear(
            merged_size,
            config.out_hidden_size,
            use_bias=True,
            param_dtype=dtype,
            kernel_init=nnx.with_partitioning(init_fn, ("model", None)),
            bias_init=nnx.with_partitioning(init_fn, (None,)),
            rngs=rngs,
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        # x: [num_patches, hidden] (T*H*W before spatial merge)
        merged_size = self.fc1.in_features
        if self.use_postshuffle_norm:
            # merge first, then normalise
            x = self.norm(x.reshape(-1, merged_size))
        else:
            # normalise first, then merge
            x = self.norm(x).reshape(-1, merged_size)
        x = self.fc2(_gelu_tanh(self.fc1(x)))
        return x


# ---------------------------------------------------------------------------
# Vision transformer (with deepstack)
# ---------------------------------------------------------------------------

class Qwen3VLVisionTransformer(nnx.Module):
    """Qwen3-VL vision encoder with deepstack feature extraction.

    Differences from Qwen2_5_VisionTransformer:
    - No windowed attention; all blocks use full attention separated by
      cu_seqlens.
    - Learnable pos_embed (bilinear-interpolated) added to patch embeddings.
    - Intermediate hidden states at deepstack_visual_indexes are extracted
      through separate PatchMergers and returned alongside the final output.
    """

    def __init__(self, vllm_config: VllmConfig, rngs: nnx.Rngs, mesh: Mesh):
        model_config = vllm_config.model_config
        hf_config: Qwen3VLConfig = model_config.hf_config
        vision_config: Qwen3VLVisionConfig = hf_config.vision_config
        dtype = model_config.dtype

        self.config = vision_config
        self.dtype = dtype
        self.spatial_merge_size = vision_config.spatial_merge_size
        self.spatial_merge_unit = vision_config.spatial_merge_size ** 2
        self.patch_size = vision_config.patch_size
        self.deepstack_visual_indexes = vision_config.deepstack_visual_indexes

        # Learned position embedding (48×48 grid by default).
        self.num_grid_per_side = int(vision_config.num_position_embeddings ** 0.5)
        self.pos_embed = nnx.Embed(
            num_embeddings=vision_config.num_position_embeddings,
            features=vision_config.hidden_size,
            param_dtype=dtype,
            embedding_init=nnx.with_partitioning(init_fn, ("model", None)),
            rngs=rngs,
        )

        self.patch_embed = Qwen3VLVisionPatchEmbed(vision_config, dtype, rngs)

        head_dim = vision_config.hidden_size // vision_config.num_heads
        self.rotary_pos_emb = Qwen3VLVisionRotaryEmbedding(head_dim // 2)

        self.blocks = nnx.List([
            Qwen3VLVisionBlock(vision_config, dtype, rngs, mesh)
            for _ in range(vision_config.depth)
        ])

        # Main merger (applied to the final output).
        self.merger = Qwen3VLVisionPatchMerger(
            vision_config, dtype, rngs, use_postshuffle_norm=False)

        # One merger per deepstack layer (applied to intermediate outputs).
        self.deepstack_merger_list = nnx.List([
            Qwen3VLVisionPatchMerger(
                vision_config, dtype, rngs, use_postshuffle_norm=True)
            for _ in range(len(vision_config.deepstack_visual_indexes))
        ])

        additional_config = getattr(vllm_config, "additional_config", None) or {}
        self.enable_dynamic_image_sizes = additional_config.get(
            "enable_dynamic_image_sizes", False)

    # ------------------------------------------------------------------
    # Position embedding helpers
    # ------------------------------------------------------------------

    def _fast_pos_embed_interpolate(
            self, grid_thw: tuple[tuple[int, int, int], ...]) -> jax.Array:
        """Bilinear-interpolate pos_embed to match each image's grid size."""
        n = self.num_grid_per_side
        merge_size = self.spatial_merge_size
        all_embeds = []

        for (t, h, w) in grid_thw:
            # Sample continuous indices in [0, n-1] for height and width.
            h_idx = jnp.linspace(0.0, n - 1, h)  # [h]
            w_idx = jnp.linspace(0.0, n - 1, w)  # [w]

            h_lo = jnp.floor(h_idx).astype(jnp.int32)
            w_lo = jnp.floor(w_idx).astype(jnp.int32)
            h_hi = jnp.minimum(h_lo + 1, n - 1)
            w_hi = jnp.minimum(w_lo + 1, n - 1)
            dh = (h_idx - h_lo).astype(jnp.float32)  # [h]
            dw = (w_idx - w_lo).astype(jnp.float32)  # [w]

            # 2-D grid indices for the four bilinear corners.
            idx00 = h_lo[:, None] * n + w_lo[None, :]  # [h, w]
            idx01 = h_lo[:, None] * n + w_hi[None, :]
            idx10 = h_hi[:, None] * n + w_lo[None, :]
            idx11 = h_hi[:, None] * n + w_hi[None, :]

            w00 = (1 - dh[:, None]) * (1 - dw[None, :])  # [h, w]
            w01 = (1 - dh[:, None]) * dw[None, :]
            w10 = dh[:, None] * (1 - dw[None, :])
            w11 = dh[:, None] * dw[None, :]

            # Fetch all 4 bilinear corners in one lookup then interpolate.
            # self.pos_embed.embedding is an nnx.Param; use .value to get array.
            all_idx = jnp.concatenate([
                idx00.reshape(-1), idx01.reshape(-1),
                idx10.reshape(-1), idx11.reshape(-1),
            ])  # [4*h*w]
            all_emb = self.pos_embed.embedding.value[all_idx]  # [4*h*w, D]
            e00, e01, e10, e11 = jnp.split(all_emb, 4)
            e00 = e00.reshape(h, w, -1)
            e01 = e01.reshape(h, w, -1)
            e10 = e10.reshape(h, w, -1)
            e11 = e11.reshape(h, w, -1)
            emb = (w00[:, :, None] * e00 + w01[:, :, None] * e01
                   + w10[:, :, None] * e10 + w11[:, :, None] * e11)  # [h, w, D]

            # Repeat for t frames.
            emb = jnp.tile(emb[None], (t, 1, 1, 1)).reshape(t * h * w, -1)

            # Apply spatial merge permutation so that each group of
            # spatial_merge_unit adjacent tokens belongs to the same merged
            # patch — matching the order used by the patch merger.
            emb = emb.reshape(
                t, h // merge_size, merge_size, w // merge_size, merge_size, -1
            ).transpose(0, 1, 3, 2, 4, 5).reshape(-1, emb.shape[-1])

            all_embeds.append(emb)

        return jnp.concatenate(all_embeds, axis=0)  # [total_patches, D]

    def _rotary_pos_emb_for_grid(
            self, t: int, h: int, w: int) -> jax.Array:
        """Compute RoPE frequencies for a single image grid (t, h, w)."""
        merge_size = self.spatial_merge_size

        hpos_ids, wpos_ids = jnp.indices((h, w))
        hpos_ids = hpos_ids.reshape(
            h // merge_size, merge_size,
            w // merge_size, merge_size,
        ).transpose(0, 2, 1, 3).flatten()
        wpos_ids = wpos_ids.reshape(
            h // merge_size, merge_size,
            w // merge_size, merge_size,
        ).transpose(0, 2, 1, 3).flatten()

        pos_ids = jnp.stack([hpos_ids, wpos_ids], axis=-1)  # [h*w, 2]
        pos_ids = jnp.tile(pos_ids, (t, 1))  # [t*h*w, 2]

        max_size = max(h, w)
        rope_full = self.rotary_pos_emb(max_size)  # [max_size, head_dim//4]
        rope = rope_full[pos_ids].reshape(t * h * w, -1)  # [t*h*w, head_dim//2]
        return rope  # [total_patches, head_dim//2]

    def _compute_cu_seqlens(
            self, grid_thw: tuple[tuple[int, int, int], ...]) -> jax.Array:
        """Cumulative sequence lengths separating each image's patch sequence."""
        lengths = [t * h * w for (t, h, w) in grid_thw]
        cu = jnp.array([0] + list(np.cumsum(lengths)), dtype=jnp.int32)
        return cu

    # ------------------------------------------------------------------
    # Core forward
    # ------------------------------------------------------------------

    def compute_hidden_states(
        self,
        x: jax.Array,
        rotary_pos_emb: jax.Array,
        cu_seqlens: jax.Array,
    ) -> tuple[jax.Array, list[jax.Array]]:
        """Run all vision blocks and collect deepstack features."""
        hidden_states = self.patch_embed(x)  # [total_patches, D]
        seq_len = hidden_states.shape[0]
        hidden_states = jnp.expand_dims(hidden_states, axis=1)  # [T, 1, D]

        deepstack_features: list[jax.Array] = []
        for layer_num, blk in enumerate(self.blocks):
            hidden_states = blk(hidden_states, rotary_pos_emb, cu_seqlens)
            if layer_num in self.deepstack_visual_indexes:
                idx = self.deepstack_visual_indexes.index(layer_num)
                # Squeeze batch dim before merger.
                feat = self.deepstack_merger_list[idx](
                    hidden_states[:, 0, :])  # [merged_tokens, out_hidden]
                deepstack_features.append(feat)

        image_embeds = self.merger(hidden_states[:, 0, :])
        return image_embeds, deepstack_features

    @jax.jit
    def encode_jit(
        self,
        x: jax.Array,
        grid_thw: tuple[tuple[int, int, int], ...],
    ) -> tuple[jax.Array, list[jax.Array]]:
        rotary_pos_emb = self._rotary_pos_emb_for_all(grid_thw)
        cu_seqlens = self._compute_cu_seqlens(grid_thw)
        # Add learned position embeddings.
        pos_embeds = self._fast_pos_embed_interpolate(grid_thw)
        patches = self.patch_embed(x) + pos_embeds
        # Re-run patch_embed inside compute_hidden_states is wasteful;
        # pass pre-computed embeddings instead.
        return self._forward_from_patch_embeds(patches, rotary_pos_emb,
                                               cu_seqlens)

    def _rotary_pos_emb_for_all(
            self, grid_thw: tuple[tuple[int, int, int], ...]) -> jax.Array:
        parts = [self._rotary_pos_emb_for_grid(t, h, w)
                 for (t, h, w) in grid_thw]
        return jnp.concatenate(parts, axis=0)

    def _forward_from_patch_embeds(
        self,
        hidden_states: jax.Array,
        rotary_pos_emb: jax.Array,
        cu_seqlens: jax.Array,
    ) -> tuple[jax.Array, list[jax.Array]]:
        """Run transformer blocks on already-embedded patches."""
        hidden_states = jnp.expand_dims(hidden_states, axis=1)  # [T, 1, D]

        deepstack_features: list[jax.Array] = []
        for layer_num, blk in enumerate(self.blocks):
            hidden_states = blk(hidden_states, rotary_pos_emb, cu_seqlens)
            if layer_num in self.deepstack_visual_indexes:
                idx = self.deepstack_visual_indexes.index(layer_num)
                feat = self.deepstack_merger_list[idx](hidden_states[:, 0, :])
                deepstack_features.append(feat)

        image_embeds = self.merger(hidden_states[:, 0, :])
        return image_embeds, deepstack_features

    def __call__(
        self,
        x: jax.Array,
        grid_thw: tuple[tuple[int, int, int], ...],
    ) -> tuple[jax.Array, list[jax.Array]]:
        rotary_pos_emb = self._rotary_pos_emb_for_all(grid_thw)
        cu_seqlens = self._compute_cu_seqlens(grid_thw)
        pos_embeds = self._fast_pos_embed_interpolate(grid_thw)
        patches = self.patch_embed(x) + pos_embeds
        return self._forward_from_patch_embeds(patches, rotary_pos_emb,
                                               cu_seqlens)


# ---------------------------------------------------------------------------
# Language model with deepstack injection
# ---------------------------------------------------------------------------

class Qwen3VLTextModel(Qwen3Model):
    """Qwen3 language model extended with deepstack visual feature injection.

    At LLM layers 0 … len(deepstack_features)-1, the visual features from the
    corresponding deepstack level are *added* to the hidden states at image
    token positions.
    """

    def __init__(self, vllm_config: VllmConfig, rng: nnx.Rngs, mesh: Mesh,
                 prefix: str = "model") -> None:
        # Qwen3VLConfig does not proxy text_config attributes at the top level.
        # Temporarily replace hf_config with text_config so that the parent
        # Qwen3Model.__init__ can read hidden_size, num_hidden_layers, etc.
        model_config = vllm_config.model_config
        original_hf_config = model_config.hf_config
        model_config.hf_config = original_hf_config.text_config
        try:
            super().__init__(vllm_config, rng, mesh, prefix=prefix)
        finally:
            model_config.hf_config = original_hf_config

    def __call__(
        self,
        kv_caches: list[jax.Array],
        input_ids: Optional[jax.Array],
        attention_metadata: AttentionMetadata,
        inputs_embeds: Optional[jax.Array] = None,
        deepstack_features: Optional[list[jax.Array]] = None,
        visual_token_mask: Optional[jax.Array] = None,
    ) -> tuple[list[jax.Array], jax.Array]:
        if inputs_embeds is not None:
            x = inputs_embeds
        else:
            x = self.embed_tokens(input_ids)

        for i, layer in enumerate(
                islice(self.layers, self.start_layer, self.end_layer)):
            global_layer_idx = self.start_layer + i
            kv_cache = kv_caches[i]
            kv_cache, x = layer(kv_cache, x, attention_metadata)
            kv_caches[i] = kv_cache

            # Deepstack injection: add visual features at image token positions.
            if (deepstack_features is not None
                    and visual_token_mask is not None
                    and global_layer_idx < len(deepstack_features)):
                x = self._deepstack_inject(
                    x, visual_token_mask,
                    deepstack_features[global_layer_idx])

        x = self.norm(x)
        return kv_caches, x

    @staticmethod
    def _deepstack_inject(
        hidden_states: jax.Array,
        visual_token_mask: jax.Array,
        visual_embeds: jax.Array,
    ) -> jax.Array:
        """Add *visual_embeds* to *hidden_states* at visual token positions.

        Args:
            hidden_states: [T, D] language model hidden states.
            visual_token_mask: [T] boolean mask (True where image tokens are).
            visual_embeds: [num_visual_tokens, D] deepstack features.

        Returns:
            Updated hidden_states of shape [T, D].
        """
        addition = jnp.zeros_like(hidden_states)
        addition = addition.at[visual_token_mask].set(visual_embeds)
        return hidden_states + addition


# ---------------------------------------------------------------------------
# Top-level model
# ---------------------------------------------------------------------------

class Qwen3VLForConditionalGeneration(nnx.Module):
    """Qwen3-VL multimodal model for conditional generation on TPU."""

    WeightLoader = StandardWeightLoader

    def __init__(self, vllm_config: VllmConfig, rng_key: jax.Array,
                 mesh: Mesh) -> None:
        hf_config: Qwen3VLConfig = vllm_config.model_config.hf_config
        text_config = hf_config.text_config

        self.vllm_config = vllm_config
        self.rng = nnx.Rngs(rng_key)
        self.mesh = mesh
        self.config = hf_config
        self.text_config = text_config

        self.is_first_rank = get_pp_group().is_first_rank
        self.is_last_rank = get_pp_group().is_last_rank

        if self.is_first_rank:
            self.visual = Qwen3VLVisionTransformer(
                vllm_config=vllm_config, rngs=self.rng, mesh=mesh)
        else:
            self.visual = PPMissingLayer()

        self.model = Qwen3VLTextModel(
            vllm_config=vllm_config, rng=self.rng, mesh=mesh)

        if not text_config.tie_word_embeddings:
            if self.is_last_rank:
                vocab_size = vllm_config.model_config.get_vocab_size()
                self.lm_head = JaxEinsum(
                    einsum_str="TD,DV->TV",
                    kernel_shape=(text_config.hidden_size, vocab_size),
                    dtype=vllm_config.model_config.dtype,
                    param_dtype=vllm_config.model_config.dtype,
                    rngs=self.rng,
                    quant_config=vllm_config.quant_config,
                    prefix="lm_head",
                )
            else:
                self.lm_head = PPMissingLayer()

        # Transient state set by embed_multimodal/embed_input_ids and consumed
        # in __call__ during the prefill phase.
        # Must use nnx.data() so NNX allows storing JAX arrays here.
        self._pending_deepstack_features = nnx.data(None)
        self._pending_visual_token_mask = nnx.data(None)

    # ------------------------------------------------------------------
    # MRoPE helpers (adapted from Qwen2_5_VLForConditionalGeneration)
    # ------------------------------------------------------------------

    def get_mrope_input_positions(
        self,
        input_tokens: list[int],
        hf_config,
        image_grid_thw,
        video_grid_thw,
        second_per_grid_ts: list[float],
        context_len: int = 0,
        seq_len: Optional[int] = None,
        audio_feature_lengths=None,
        use_audio_in_video: bool = False,
    ) -> tuple[jax.Array, int]:
        """Compute MRoPE position IDs and delta for a multimodal sequence."""
        text_config = hf_config.text_config
        image_token_id = hf_config.image_token_id
        video_token_id = hf_config.video_token_id
        vision_start_token_id = hf_config.vision_start_token_id
        spatial_merge_size = hf_config.vision_config.spatial_merge_size
        tokens_per_second = 1.0  # Qwen3-VL does not expose tokens_per_second

        input_tokens_tensor = np.array(input_tokens)
        vision_start_indices = np.argwhere(
            input_tokens_tensor == vision_start_token_id).squeeze(1)
        vision_tokens = input_tokens_tensor[vision_start_indices + 1]
        image_nums = int(np.sum(vision_tokens == image_token_id))
        video_nums = int(np.sum(vision_tokens == video_token_id))
        llm_pos_ids_list: list = []

        st = 0
        remain_images, remain_videos = image_nums, video_nums
        image_index, video_index = 0, 0

        for _ in range(image_nums + video_nums):
            video_second_per_grid_t = 0.0
            if remain_images > 0:
                try:
                    ed_image = input_tokens.index(image_token_id, st)
                except ValueError:
                    ed_image = len(input_tokens) + 1
            else:
                ed_image = len(input_tokens) + 1

            if remain_videos > 0:
                try:
                    ed_video = input_tokens.index(video_token_id, st)
                except ValueError:
                    ed_video = len(input_tokens) + 1
            else:
                ed_video = len(input_tokens) + 1

            if ed_image < ed_video:
                t, h, w = (image_grid_thw[image_index][0],
                           image_grid_thw[image_index][1],
                           image_grid_thw[image_index][2])
                image_index += 1
                remain_images -= 1
                ed = ed_image
            else:
                t, h, w = (video_grid_thw[video_index][0],
                           video_grid_thw[video_index][1],
                           video_grid_thw[video_index][2])
                video_second_per_grid_t = (second_per_grid_ts[video_index]
                                           if second_per_grid_ts else 1.0)
                video_index += 1
                remain_videos -= 1
                ed = ed_video

            llm_grid_t = t
            llm_grid_h = h // spatial_merge_size
            llm_grid_w = w // spatial_merge_size
            text_len = ed - st

            st_idx = (llm_pos_ids_list[-1].max().item() + 1
                      if llm_pos_ids_list else 0)
            llm_pos_ids_list.append(
                jnp.broadcast_to(
                    jnp.arange(text_len, dtype=jnp.int32).reshape(1, -1),
                    (3, text_len)) + st_idx)

            t_index = (jnp.broadcast_to(
                jnp.arange(llm_grid_t, dtype=jnp.int32).reshape(-1, 1),
                (llm_grid_t, llm_grid_h * llm_grid_w)) *
                       video_second_per_grid_t * tokens_per_second
                       ).astype(jnp.int32).flatten()
            h_index = jnp.broadcast_to(
                jnp.arange(llm_grid_h, dtype=jnp.int32).reshape(1, -1, 1),
                (llm_grid_t, llm_grid_h, llm_grid_w)).flatten()
            w_index = jnp.broadcast_to(
                jnp.arange(llm_grid_w, dtype=jnp.int32).reshape(1, 1, -1),
                (llm_grid_t, llm_grid_h, llm_grid_w)).flatten()

            llm_pos_ids_list.append(
                jnp.stack([t_index, h_index, w_index]) + text_len + st_idx)
            st = ed + llm_grid_t * llm_grid_h * llm_grid_w

        if st < len(input_tokens):
            st_idx = (llm_pos_ids_list[-1].max().item() + 1
                      if llm_pos_ids_list else 0)
            text_len = len(input_tokens) - st
            llm_pos_ids_list.append(
                jnp.broadcast_to(
                    jnp.arange(text_len, dtype=jnp.int32).reshape(1, -1),
                    (3, text_len)) + st_idx)

        llm_positions = jnp.concatenate(llm_pos_ids_list, axis=1).reshape(3, -1)
        mrope_position_delta = (llm_positions.max() + 1 -
                                len(input_tokens)).item()
        llm_positions = llm_positions[:, context_len:seq_len]
        return llm_positions, mrope_position_delta

    # ------------------------------------------------------------------
    # Multimodal embedding pipeline
    # ------------------------------------------------------------------

    def _validate_and_reshape_mm_tensor(self, mm_input: object,
                                        name: str) -> jax.Array:
        if isinstance(mm_input, list):
            return jnp.concatenate([jnp.asarray(item) for item in mm_input],
                                   axis=0)
        if hasattr(mm_input, "ndim"):
            arr = jnp.asarray(mm_input)
            if arr.ndim == 2:
                return arr
            if arr.ndim == 3:
                return arr.reshape(-1, arr.shape[-1])
        raise ValueError(f"Incorrect type of {name}: {type(mm_input)}")

    def _parse_and_validate_image_input(
        self,
        image_grid_thw: tuple[tuple[int, int, int], ...],
        **kwargs: object,
    ) -> Optional[Qwen3VLImagePixelInputs]:
        pixel_values = kwargs.pop("pixel_values", None)
        if pixel_values is None:
            return None
        pixel_values = self._validate_and_reshape_mm_tensor(
            pixel_values, "image pixel values")
        return Qwen3VLImagePixelInputs(
            type="pixel_values",
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
        )

    def get_single_image_embedding(
        self,
        image_pixel_values: jax.Array,
        image_grid_thw: tuple[int, int, int],
    ) -> tuple[jax.Array, list[jax.Array]]:
        """Encode a single image and return (main_embeds, deepstack_features)."""
        return self.visual(image_pixel_values, (image_grid_thw,))

    def _process_image_input(
        self,
        image_input: Qwen3VLImagePixelInputs,
    ) -> tuple[tuple[jax.Array, ...], list[jax.Array]]:
        """Return per-image main embeddings and accumulated deepstack features."""
        grid_thw = image_input["image_grid_thw"]
        pixel_values = image_input["pixel_values"]

        all_image_embeds: list[jax.Array] = []
        # Accumulate deepstack features from all images (summed per level).
        accumulated_deepstack: list[Optional[jax.Array]] = [
            None] * len(self.visual.deepstack_visual_indexes)

        current_idx = 0
        for image_thw in grid_thw:
            t, h, w = image_thw
            image_size = t * h * w
            end_idx = current_idx + image_size
            image_pv = pixel_values[current_idx:end_idx, :]
            embeds, deepstack_feats = self.get_single_image_embedding(
                image_pv, image_thw)
            all_image_embeds.append(embeds)
            for j, feat in enumerate(deepstack_feats):
                accumulated_deepstack[j] = (
                    feat if accumulated_deepstack[j] is None
                    else jnp.concatenate([accumulated_deepstack[j], feat], axis=0))
            current_idx = end_idx

        image_embeds = jnp.concatenate(all_image_embeds, axis=0)
        merge_size = self.visual.spatial_merge_size
        sizes = np.prod(np.array(grid_thw, dtype=np.int64),
                        axis=-1) // merge_size // merge_size
        if sizes.size == 0:
            main_embeds_tuple: tuple[jax.Array, ...] = ()
        elif sizes.size == 1:
            main_embeds_tuple = (image_embeds,)
        else:
            split_indices = np.cumsum(sizes)[:-1]
            main_embeds_tuple = tuple(jnp.split(image_embeds, split_indices))

        deepstack_list = [f for f in accumulated_deepstack if f is not None]
        return main_embeds_tuple, deepstack_list

    def embed_multimodal(
        self,
        image_grid_thw: tuple[tuple[int, int, int], ...],
        **kwargs: object,
    ) -> MultiModalEmbeddings:
        if not self.is_first_rank:
            self._pending_deepstack_features = None
            return ()

        image_input = self._parse_and_validate_image_input(
            image_grid_thw, **kwargs)
        if image_input is None:
            self._pending_deepstack_features = None
            return []

        main_embeds_tuple, deepstack_list = self._process_image_input(
            image_input)
        self._pending_deepstack_features = deepstack_list
        return main_embeds_tuple

    def embed_input_ids(
        self,
        input_ids: jax.Array,
        multimodal_embeddings: jax.Array | None,
        *,
        is_multimodal: jax.Array | None = None,
    ) -> jax.Array:
        del is_multimodal
        if not self.is_first_rank:
            return None

        inputs_embeds = self.model.embed_tokens(input_ids)

        if (multimodal_embeddings is not None
                and multimodal_embeddings.shape[0] != 0):
            inputs_embeds = merge_multimodal_embeddings(
                input_ids, inputs_embeds, multimodal_embeddings,
                [self.config.image_token_id, self.config.video_token_id])
            # Record which positions received visual features for deepstack.
            self._pending_visual_token_mask = (
                (input_ids == self.config.image_token_id) |
                (input_ids == self.config.video_token_id))
        else:
            self._pending_visual_token_mask = None

        return inputs_embeds

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def __call__(
        self,
        kv_caches: list[jax.Array],
        input_ids: Optional[jax.Array],
        attention_metadata: AttentionMetadata,
        inputs_embeds: Optional[jax.Array] = None,
        _input_positions=None,
        _layer_name_to_kv_cache=None,
        _lora_metadata=None,
        intermediate_tensors: Optional[JaxIntermediateTensors] = None,
        is_first_rank: bool = True,
        is_last_rank: bool = True,
        *args,
    ) -> tuple[list[jax.Array], jax.Array | JaxIntermediateTensors,
               list[jax.Array]]:
        if not is_first_rank:
            assert intermediate_tensors is not None
            inputs_embeds = intermediate_tensors["hidden_states"]

        # Deepstack features are only available during prefill (inputs_embeds
        # is set) and only on the first pipeline-parallel rank.
        deepstack_features = None
        visual_token_mask = None
        if inputs_embeds is not None and is_first_rank:
            deepstack_features = self._pending_deepstack_features
            visual_token_mask = self._pending_visual_token_mask
            # Clear after consumption.
            self._pending_deepstack_features = None
            self._pending_visual_token_mask = None

        kv_caches, x = self.model(
            kv_caches=kv_caches,
            input_ids=input_ids,
            attention_metadata=attention_metadata,
            inputs_embeds=inputs_embeds,
            deepstack_features=deepstack_features,
            visual_token_mask=visual_token_mask,
        )

        if not is_last_rank:
            x = JaxIntermediateTensors(tensors={"hidden_states": x})

        return kv_caches, x, []

    def compute_logits(self, hidden_states: jax.Array) -> jax.Array:
        if hasattr(self, "lm_head"):
            return self.lm_head(hidden_states)
        return self.model.embed_tokens.decode(hidden_states)

    # ------------------------------------------------------------------
    # Weight loading
    # ------------------------------------------------------------------

    def load_weights(self, rng_key: jax.Array) -> None:
        self.rng = nnx.Rngs(rng_key)

        self.pp_missing_layers = []
        for path, module in nnx.iter_graph(self):
            if isinstance(module, PPMissingLayer):
                self.pp_missing_layers.append(
                    ".".join([str(s) for s in path]))

        # HF weight naming for Qwen3-VL:
        #   model.visual.*          → visual.*          (vision encoder)
        #   model.language_model.*  → model.*           (LLM backbone)
        #
        # The weight loader strips the ".weight" suffix unless the HF key
        # contains a string from keep_hf_weight_suffix_when_match.
        # We use "model.language_model" so that LLM weights keep their suffix
        # (the JaxEinsum/JaxEmbed modules store parameters as ".weight").
        # Vision encoder ".weight" suffixes are stripped; we map them to
        # ".kernel" or ".scale" as appropriate for nnx.Linear / nnx.LayerNorm.

        num_deepstack = len(self.config.vision_config.deepstack_visual_indexes)

        # Mappings: HF weight name (after suffix stripping where applicable)
        #           → JAX nnx.State path.
        mappings = {
            # ----------------------------------------------------------------
            # Vision blocks (matched by "blocks.*" pattern in weight loader)
            # ----------------------------------------------------------------
            "model.visual.blocks.*.norm1":
                "visual.blocks.*.norm1.scale",
            "model.visual.blocks.*.norm1.bias":
                "visual.blocks.*.norm1.bias",
            "model.visual.blocks.*.norm2":
                "visual.blocks.*.norm2.scale",
            "model.visual.blocks.*.norm2.bias":
                "visual.blocks.*.norm2.bias",
            "model.visual.blocks.*.attn.qkv":
                "visual.blocks.*.attn.qkv_proj.kernel",
            "model.visual.blocks.*.attn.qkv.bias":
                "visual.blocks.*.attn.qkv_proj.bias",
            "model.visual.blocks.*.attn.proj":
                "visual.blocks.*.attn.proj.kernel",
            "model.visual.blocks.*.attn.proj.bias":
                "visual.blocks.*.attn.proj.bias",
            "model.visual.blocks.*.mlp.linear_fc1":
                "visual.blocks.*.mlp.fc1.kernel",
            "model.visual.blocks.*.mlp.linear_fc1.bias":
                "visual.blocks.*.mlp.fc1.bias",
            "model.visual.blocks.*.mlp.linear_fc2":
                "visual.blocks.*.mlp.fc2.kernel",
            "model.visual.blocks.*.mlp.linear_fc2.bias":
                "visual.blocks.*.mlp.fc2.bias",
            # ----------------------------------------------------------------
            # Vision non-block weights (matched via direct key lookup)
            # ----------------------------------------------------------------
            # patch_embed: ".weight" stripped → "model.visual.patch_embed.proj"
            "model.visual.patch_embed.proj":
                "visual.patch_embed.proj.kernel",
            "model.visual.patch_embed.proj.bias":
                "visual.patch_embed.proj.bias",
            # pos_embed: HF key "model.visual.pos_embed.weight" → stripped
            "model.visual.pos_embed":
                "visual.pos_embed.embedding",
            # merger
            "model.visual.merger.norm":
                "visual.merger.norm.scale",
            "model.visual.merger.norm.bias":
                "visual.merger.norm.bias",
            "model.visual.merger.linear_fc1":
                "visual.merger.fc1.kernel",
            "model.visual.merger.linear_fc1.bias":
                "visual.merger.fc1.bias",
            "model.visual.merger.linear_fc2":
                "visual.merger.fc2.kernel",
            "model.visual.merger.linear_fc2.bias":
                "visual.merger.fc2.bias",
            # ----------------------------------------------------------------
            # LLM top-level weights (suffix NOT stripped, direct lookup)
            # ----------------------------------------------------------------
            "model.language_model.embed_tokens.weight":
                "model.embed_tokens.weight",
            "model.language_model.norm.weight":
                "model.norm.weight",
            # ----------------------------------------------------------------
            # LLM layer weights (matched by "layers.*" pattern)
            # ----------------------------------------------------------------
            "model.language_model.layers.*.input_layernorm.weight":
                "model.layers.*.input_layernorm.weight",
            "model.language_model.layers.*.post_attention_layernorm.weight":
                "model.layers.*.post_attention_layernorm.weight",
            "model.language_model.layers.*.self_attn.q_proj.weight":
                "model.layers.*.self_attn.q_proj.weight",
            "model.language_model.layers.*.self_attn.q_norm.weight":
                "model.layers.*.self_attn.q_norm.weight",
            "model.language_model.layers.*.self_attn.k_proj.weight":
                "model.layers.*.self_attn.k_proj.weight",
            "model.language_model.layers.*.self_attn.k_norm.weight":
                "model.layers.*.self_attn.k_norm.weight",
            "model.language_model.layers.*.self_attn.v_proj.weight":
                "model.layers.*.self_attn.v_proj.weight",
            "model.language_model.layers.*.self_attn.o_proj.weight":
                "model.layers.*.self_attn.o_proj.weight",
            "model.language_model.layers.*.mlp.gate_proj.weight":
                "model.layers.*.mlp.gate_proj.weight",
            "model.language_model.layers.*.mlp.up_proj.weight":
                "model.layers.*.mlp.up_proj.weight",
            "model.language_model.layers.*.mlp.down_proj.weight":
                "model.layers.*.mlp.down_proj.weight",
        }

        # Expand deepstack_merger_list entries individually (no generic handler
        # for this path pattern in the weight loader).
        for i in range(num_deepstack):
            pfx_hf = f"model.visual.deepstack_merger_list.{i}"
            pfx_jax = f"visual.deepstack_merger_list.{i}"
            mappings.update({
                f"{pfx_hf}.norm": f"{pfx_jax}.norm.scale",
                f"{pfx_hf}.norm.bias": f"{pfx_jax}.norm.bias",
                f"{pfx_hf}.linear_fc1": f"{pfx_jax}.fc1.kernel",
                f"{pfx_hf}.linear_fc1.bias": f"{pfx_jax}.fc1.bias",
                f"{pfx_hf}.linear_fc2": f"{pfx_jax}.fc2.kernel",
                f"{pfx_hf}.linear_fc2.bias": f"{pfx_jax}.fc2.bias",
            })

        # Temporarily patch hf_config → text_config so that get_default_maps
        # can read num_attention_heads / num_key_value_heads / hidden_size from
        # the LLM text config (Qwen3VLConfig does not expose these top-level).
        model_config = self.vllm_config.model_config
        original_hf_config = model_config.hf_config
        model_config.hf_config = original_hf_config.text_config
        try:
            loader = self.WeightLoader(self.vllm_config, self.mesh)
            loader.load_weights(
                self,
                mappings,
                # Keep ".weight" suffix for LLM weights so JaxEinsum /
                # JaxEmbed parameter paths stay as "...weight".
                keep_hf_weight_suffix_when_match=["model.language_model"],
            )
        finally:
            model_config.hf_config = original_hf_config
