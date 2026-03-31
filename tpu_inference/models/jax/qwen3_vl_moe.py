# Copyright 2026 Google LLC
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
"""Qwen3-VL-MoE model implementation for TPU inference.

Covers Qwen3-VL-30B-A3B-Instruct and Qwen3-VL-235B-A22B-Instruct.

Architecture: same vision encoder as Qwen3-VL (with deepstack), but the LLM
backbone uses sparse Mixture-of-Experts layers (Qwen3MoeDecoderLayer).

Key differences from Qwen3VLForConditionalGeneration (dense):
- LLM backbone: Qwen3VLMoeTextModel (extends Qwen3MoeModel) instead of
  Qwen3VLTextModel.
- HF weight format for MoE experts: bulk tensors
  "mlp.experts.gate_up_proj" (E, D, 2*F) and "mlp.experts.down_proj" (E, F, D)
  rather than per-expert "mlp.experts.{i}.gate_proj/up_proj/down_proj.weight".
  These are split and transposed during loading to match JaxMoE's internal
  layout.
- Router gate: simple linear "mlp.gate.weight" (E, D) — needs (1,0) transpose
  to fit JaxLinear's (D, E) kernel.
"""

import re
from typing import List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax.sharding import Mesh, NamedSharding
from transformers.models.qwen3_vl_moe.configuration_qwen3_vl_moe import (
    Qwen3VLMoeConfig)
from vllm.config import VllmConfig

from tpu_inference.distributed.jax_parallel_state import get_pp_group
from tpu_inference.layers.common.attention_metadata import AttentionMetadata
from tpu_inference.layers.jax.linear import JaxEinsum
from tpu_inference.layers.jax.moe.moe import JaxMoE
from tpu_inference.layers.jax.pp_utils import PPMissingLayer
from tpu_inference.logger import init_logger
from tpu_inference.models.jax.jax_intermediate_tensor import \
    JaxIntermediateTensors
from tpu_inference.models.jax.qwen3_moe import Qwen3MoeModel
from tpu_inference.models.jax.qwen3_vl import (
    Qwen3VLImagePixelInputs,
    Qwen3VLVisionTransformer,
    Qwen3VLTextModel,
)
from tpu_inference.models.jax.utils.multi_modal_utils import (
    MultiModalEmbeddings, merge_multimodal_embeddings)
from tpu_inference.models.jax.utils.weight_utils import (
    _load_and_shard_weight,
    check_all_loaded,
    get_default_maps,
    get_model_weights_files,
    get_param_and_sharding,
    model_weights_single_file_generator,
    shard_put,
)

logger = init_logger(__name__)

init_fn = nnx.initializers.uniform()


# ---------------------------------------------------------------------------
# Language model with deepstack injection (MoE backbone)
# ---------------------------------------------------------------------------

class Qwen3VLMoeTextModel(Qwen3MoeModel):
    """Qwen3MoeModel extended with deepstack visual feature injection.

    At LLM layers 0 … len(deepstack_features)-1, the visual features from the
    corresponding deepstack level are *added* to the hidden states at image
    token positions.
    """

    def __init__(self, vllm_config: VllmConfig, rng: nnx.Rngs, mesh: Mesh,
                 prefix: str = "model") -> None:
        # Qwen3VLMoeConfig does not proxy text_config attributes at the top
        # level. Temporarily replace hf_config with text_config so that the
        # parent Qwen3MoeModel.__init__ can read hidden_size, num_hidden_layers,
        # etc. directly from it.
        model_config = vllm_config.model_config
        original_hf_config = model_config.hf_config
        model_config.hf_config = original_hf_config.text_config
        try:
            super().__init__(vllm_config, rng, mesh, prefix=prefix)
        finally:
            model_config.hf_config = original_hf_config

    def __call__(
        self,
        kv_caches: List[jax.Array],
        input_ids: Optional[jax.Array],
        attention_metadata: AttentionMetadata,
        inputs_embeds: Optional[jax.Array] = None,
        deepstack_features: Optional[List[jax.Array]] = None,
        visual_token_mask: Optional[jax.Array] = None,
    ) -> Tuple[List[jax.Array], jax.Array]:
        if self.is_first_rank:
            if inputs_embeds is None:
                x = self.embed_tokens(input_ids)
            else:
                x = inputs_embeds
        else:
            assert inputs_embeds is not None
            x = inputs_embeds

        new_kv_caches = []
        for i, layer in enumerate(self.layers):
            if isinstance(layer, PPMissingLayer):
                new_kv_caches.append(kv_caches[i])
                continue

            kv_cache = kv_caches[i]
            kv_cache, x = layer(kv_cache, x, attention_metadata)
            new_kv_caches.append(kv_cache)

            # Deepstack injection: add visual features at image token positions.
            if (deepstack_features is not None
                    and visual_token_mask is not None
                    and i < len(deepstack_features)):
                x = Qwen3VLTextModel._deepstack_inject(
                    x, visual_token_mask, deepstack_features[i])

        if self.is_last_rank:
            x = self.norm(x)

        return new_kv_caches, x


# ---------------------------------------------------------------------------
# Top-level multimodal model
# ---------------------------------------------------------------------------

class Qwen3VLMoeForConditionalGeneration(nnx.Module):
    """Qwen3-VL-MoE multimodal model for conditional generation on TPU.

    Combines:
    - Qwen3VLVisionTransformer (unchanged from dense Qwen3-VL)
    - Qwen3VLMoeTextModel (MoE-based LLM backbone)
    - lm_head (if not using tied embeddings)
    """

    def __init__(self, vllm_config: VllmConfig, rng_key: jax.Array,
                 mesh: Mesh) -> None:
        hf_config: Qwen3VLMoeConfig = vllm_config.model_config.hf_config
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

        self.model = Qwen3VLMoeTextModel(
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

        # Transient state consumed in __call__ during the prefill phase.
        self._pending_deepstack_features = nnx.data(None)
        self._pending_visual_token_mask = nnx.data(None)

    # ------------------------------------------------------------------
    # MRoPE helpers (identical to Qwen3VLForConditionalGeneration)
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
        tokens_per_second = 1.0

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
    # Multimodal embedding pipeline (identical to Qwen3VLForConditionalGeneration)
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
                    else jnp.concatenate([accumulated_deepstack[j], feat],
                                         axis=0))
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
        kv_caches: List[jax.Array],
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
    ) -> tuple[List[jax.Array], jax.Array | JaxIntermediateTensors,
               List[jax.Array]]:
        if not is_first_rank:
            assert intermediate_tensors is not None
            inputs_embeds = intermediate_tensors["hidden_states"]

        deepstack_features = None
        visual_token_mask = None
        if inputs_embeds is not None and is_first_rank:
            deepstack_features = self._pending_deepstack_features
            visual_token_mask = self._pending_visual_token_mask
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

        num_deepstack = len(self.config.vision_config.deepstack_visual_indexes)

        # ------------------------------------------------------------------
        # Name mappings: HF weight key → JAX nnx.State path
        #
        # Keys containing "model.language_model" keep their ".weight" suffix
        # (set via keep_hf_weight_suffix_when_match below).
        # Vision ".weight" suffixes are stripped and mapped to ".kernel" or
        # ".scale" as appropriate.
        # Expert weights (gate_up_proj, down_proj) are handled separately.
        # ------------------------------------------------------------------
        mappings = {
            # ---- Vision blocks ----
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
            # ---- Vision non-block ----
            "model.visual.patch_embed.proj":
                "visual.patch_embed.proj.kernel",
            "model.visual.patch_embed.proj.bias":
                "visual.patch_embed.proj.bias",
            "model.visual.pos_embed":
                "visual.pos_embed.embedding",
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
            # ---- LLM top-level (suffix kept) ----
            "model.language_model.embed_tokens.weight":
                "model.embed_tokens.weight",
            "model.language_model.norm.weight":
                "model.norm.weight",
            "lm_head": "lm_head.weight",
            # ---- LLM layer: attention + layernorms (suffix kept) ----
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
            # NOTE: mlp.gate.weight is handled by _load_gate_weight (custom),
            # not here, because the NNX state path for the shared JaxLinear
            # router object is not predictable across NNX versions.
        }

        # Expand deepstack_merger_list entries.
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

        # Patch hf_config → text_config so that get_default_maps reads the
        # LLM attention dimensions (num_attention_heads, etc.) correctly.
        model_config = self.vllm_config.model_config
        original_hf_config = model_config.hf_config
        text_config = original_hf_config.text_config
        model_config.hf_config = text_config
        try:
            metadata_map = get_default_maps(model_config, self.mesh, mappings)
        finally:
            model_config.hf_config = original_hf_config

        keep_suffix = ["model.language_model"]

        # Get model state for direct parameter manipulation (expert loading).
        params = nnx.state(self)
        try:
            shardings = nnx.get_named_sharding(params, self.mesh)
        except TypeError:
            shardings = params

        # Single-pass over all weight files.
        weights_files = get_model_weights_files(
            model_config.model, self.vllm_config.load_config.download_dir)

        for weights_file in weights_files:
            for hf_key, hf_weight in model_weights_single_file_generator(
                    weights_file, framework="flax"):
                # MoE expert bulk weights are handled with custom split logic.
                if ("mlp.experts.gate_up_proj" in hf_key
                        and "language_model" in hf_key):
                    self._load_bulk_gate_up_proj(
                        params, shardings, hf_key, hf_weight)
                    continue
                elif ("mlp.experts.down_proj" in hf_key
                      and "language_model" in hf_key):
                    self._load_bulk_down_proj(
                        params, shardings, hf_key, hf_weight)
                    continue
                elif ("mlp.gate.weight" in hf_key
                      and "language_model" in hf_key):
                    self._load_gate_weight(
                        params, shardings, hf_key, hf_weight)
                    continue

                # Standard loading for all other weights.
                _load_and_shard_weight(
                    self.vllm_config, params, shardings, metadata_map,
                    self.mesh, hf_key, hf_weight, keep_suffix,
                    pp_missing_layers=self.pp_missing_layers)

        check_all_loaded(params)
        nnx.update(self, params)
        self._fuse_moe_expert_weights()

    def _fuse_moe_expert_weights(self) -> None:
        """Fuse kernel_gating_EDF + kernel_up_proj_EDF → kernel_gating_upproj_EDF.

        The GMM_TP forward pass (apply_jax in UnquantizedFusedMoEMethod) requires a
        pre-fused weight:
            kernel_gating_upproj_EDF = concat([kernel_gating_EDF, kernel_up_proj_EDF], axis=1)

        process_weights_after_loading normally does this fusion, but its gate is
            any(w is None for w in param._weights_to_load)
        which returns True when bulk loading is used (bypasses _weights_to_load mechanism).
        We perform the fusion directly on the live model after nnx.update().
        Using isinstance(module, JaxMoE) is necessary — hasattr-based filtering
        can match unrelated objects where the attribute is None, causing
        AttributeError: 'NoneType' object has no attribute 'value'.
        """
        fused_count = 0
        for _path, module in nnx.iter_graph(self):
            if not isinstance(module, JaxMoE):
                continue
            w_gate = module.kernel_gating_EDF.value  # (E, F, D) as stored by bulk loader
            w_up = module.kernel_up_proj_EDF.value   # (E, F, D) as stored by bulk loader
            # Concatenate along axis=1 (F dim) → (E, 2F, D), matching
            # process_weights_after_loading behaviour for GMM_TP backend.
            w13_val = jnp.concatenate([w_gate, w_up], axis=1)
            module.kernel_gating_upproj_EDF = nnx.Param(
                shard_put(w13_val, module.edf_sharding, mesh=self.mesh))
            del module.kernel_gating_EDF
            del module.kernel_up_proj_EDF
            fused_count += 1
        logger.info(f"Fused MoE expert weights in {fused_count} JaxMoE modules.")

    def _load_bulk_gate_up_proj(
        self,
        params,
        shardings,
        hf_key: str,
        hf_weight,
    ) -> None:
        """Load fused (E, D, 2*F) gate_up_proj into JaxMoE kernels.

        JaxMoE._load_weights expects per-expert weights in (E, F, D) layout
        (gate_proj.weight (F, D) per expert → concat → (E, F, D)).
        The bulk gate_up_proj is (E, D, F) per gate/up half, so we transpose
        each half to (E, F, D) before loading.
        """
        m = re.search(r"layers\.(\d+)", hf_key)
        if m is None:
            logger.warning(f"Cannot extract layer number from {hf_key}; skipping.")
            return
        layer_num = int(m.group(1))

        dtype = self.vllm_config.model_config.dtype
        weight = jnp.asarray(hf_weight)
        if weight.dtype != dtype:
            weight = weight.astype(dtype)

        # weight shape: (E, D, 2*F)
        E, D, two_F = weight.shape
        F = two_F // 2
        gate_half = weight[:, :, :F]  # (E, D, F)
        up_half = weight[:, :, F:]    # (E, D, F)

        # Transpose to (E, F, D) to match JaxMoE's per-expert loading layout.
        gate_efd = jnp.swapaxes(gate_half, 1, 2)  # (E, F, D)
        up_efd = jnp.swapaxes(up_half, 1, 2)      # (E, F, D)

        for jax_suffix, kernel_weight in [
            ("mlp.experts.kernel_gating_EDF", gate_efd),
            ("mlp.experts.kernel_up_proj_EDF", up_efd),
        ]:
            jax_key = f"model.layers.{layer_num}.{jax_suffix}"
            self._store_expert_kernel(params, shardings, jax_key, kernel_weight)

    def _load_bulk_down_proj(
        self,
        params,
        shardings,
        hf_key: str,
        hf_weight,
    ) -> None:
        """Load (E, F, D) down_proj into JaxMoE kernel_down_proj_EFD.

        JaxMoE._load_weights expects per-expert down_proj.weight (D, F) →
        concat → (E, D, F) layout. The bulk down_proj is (E, F, D), so we
        transpose to (E, D, F) before loading.
        """
        m = re.search(r"layers\.(\d+)", hf_key)
        if m is None:
            logger.warning(f"Cannot extract layer number from {hf_key}; skipping.")
            return
        layer_num = int(m.group(1))

        dtype = self.vllm_config.model_config.dtype
        weight = jnp.asarray(hf_weight)
        if weight.dtype != dtype:
            weight = weight.astype(dtype)

        # weight shape: (E, F, D) → transpose to (E, D, F)
        down_edf = jnp.swapaxes(weight, 1, 2)  # (E, D, F)

        jax_key = f"model.layers.{layer_num}.mlp.experts.kernel_down_proj_EFD"
        self._store_expert_kernel(params, shardings, jax_key, down_edf)

    def _load_gate_weight(
        self,
        params,
        shardings,
        hf_key: str,
        hf_weight,
    ) -> None:
        """Load MoE router gate weight: HF (E, D) → JAX JaxLinear (D, E).

        The NNX graph traversal may register the shared JaxLinear (self.gate /
        JaxMoE.router) under either:
          - model.layers.N.mlp.gate.weight       (Qwen3MoeSparseMoeBlock.gate)
          - model.layers.N.mlp.experts.router.weight  (JaxMoE.router)
        We try both paths so loading succeeds regardless of which one NNX chose.
        """
        m = re.search(r"layers\.(\d+)", hf_key)
        if m is None:
            logger.warning(f"Cannot extract layer number from {hf_key}; skipping.")
            return
        layer_num = int(m.group(1))

        dtype = self.vllm_config.model_config.dtype
        weight = jnp.asarray(hf_weight)
        if weight.dtype != dtype:
            weight = weight.astype(dtype)

        # HF shape: (E, D) → transpose to (D, E) for JaxLinear kernel.
        weight_de = weight.T

        for jax_key in [
            f"model.layers.{layer_num}.mlp.gate.weight",
            f"model.layers.{layer_num}.mlp.experts.router.weight",
        ]:
            try:
                param, sharding = get_param_and_sharding(params, shardings,
                                                         jax_key)
                spec = (sharding.spec
                        if isinstance(sharding, NamedSharding) else sharding)
                param.value = shard_put(weight_de, spec, mesh=self.mesh)
                logger.debug(f"Loaded gate weight {hf_key} → {jax_key}")
                return
            except ValueError:
                continue
        logger.warning(
            f"Could not find gate weight param for {hf_key}; "
            "tried mlp.gate.weight and mlp.experts.router.weight")

    def _store_expert_kernel(
        self,
        params,
        shardings,
        jax_key: str,
        kernel_weight: jax.Array,
    ) -> None:
        """Shard and assign *kernel_weight* to the param at *jax_key*."""
        try:
            param, sharding = get_param_and_sharding(params, shardings, jax_key)
        except ValueError:
            logger.warning(
                f"Skip {jax_key}: param not found (PP missing layer or typo).")
            return
        spec = sharding.spec if isinstance(sharding, NamedSharding) else sharding
        param.value = shard_put(kernel_weight, spec, mesh=self.mesh)
