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

# JAX (flax_nnx) implementation of GLM-4.7-Flash (Glm4MoeLiteForCausalLM).
#
# Architecture:
#   - MLA attention (same structure as DeepSeek-V3 / GLM-5), standard RoPE
#     (theta=1e6). Unlike GLM-5 there is no DSA indexer -- attention is plain
#     (dense) MLA.
#   - MoE layers (64 routed + 1 shared expert) with sigmoid routing
#     (noaux_tc / bias-corrected top-k, same scheme as DeepSeek-V3 / GLM-5).
#   - Expert checkpoint weights are stored unfused (separate gate_proj /
#     up_proj / down_proj per expert), unlike GLM-5's fused gate_up_proj
#     layout, so weight loading needs no split/transpose transform.
#
# Reference: huggingface/transformers models/glm4_moe_lite/modeling_glm4_moe_lite.py
# Derived from models/jax/glm5.py (GLM-5) with the DSA indexer removed.

import os
from itertools import islice
from typing import Iterable, List, Optional, Tuple

import jax
import jax.numpy as jnp
from flax import nnx
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P
from vllm.config import VllmConfig

from tpu_inference import utils
from tpu_inference.distributed.jax_parallel_state import get_pp_group
from tpu_inference.layers.common.attention_interface import mla_attention
from tpu_inference.layers.common.attention_metadata import AttentionMetadata
from tpu_inference.layers.common.moe import MoEBackend
from tpu_inference.layers.common.quantization import quantize_kv
from tpu_inference.layers.common.sharding import \
    ShardingAxisNameBase as ShardingAxisName
from tpu_inference.layers.common.sharding import ShardingAxisName2D
from tpu_inference.layers.jax import JaxModule
from tpu_inference.layers.jax.embed import JaxEmbed
from tpu_inference.layers.jax.layers import FlaxUtils
from tpu_inference.layers.jax.linear import JaxEinsum, JaxLinear
from tpu_inference.layers.jax.moe.utils import (get_expert_parallelism,
                                                 select_moe_backend)
from tpu_inference.layers.jax.norm import JaxRmsNorm
from tpu_inference.layers.jax.pp_utils import PPMissingLayer, make_layers
from tpu_inference.layers.jax.rope_interface import apply_rope
from tpu_inference.logger import init_logger
from tpu_inference.models.jax.deepseek_v3 import (DeepSeekV3Router,
                                                    DeepseekV3MLP,
                                                    MLAEinsum,
                                                    SharedFusedMoe)
from tpu_inference.models.jax.jax_intermediate_tensor import \
    JaxIntermediateTensors
from tpu_inference.models.jax.utils.weight_utils import (JaxAutoWeightsLoader,
                                                         LoadableWithIterator)

logger = init_logger(__name__)

init_fn = nnx.initializers.uniform()
modeling_flax_utils = FlaxUtils()

expert_axis_name = ShardingAxisName.ATTN_DATA_EXPERT


def _get_rope_theta(config, default: float = 1_000_000) -> float:
    """GLM-5 nests rope_theta under `rope_parameters`; GLM-4.7-Flash has it
    as a flat `rope_theta` attribute. Support both layouts."""
    rope_theta = getattr(config, "rope_theta", None)
    if rope_theta is not None:
        return rope_theta
    rope_params = getattr(config, "rope_parameters", {}) or {}
    return rope_params.get("rope_theta", default)


class Glm4MoeLiteAttention(JaxModule):
    """MLA attention for GLM-4.7-Flash (no DSA indexer, unlike GLM-5).

    Architecture (matches HF modeling_glm4_moe_lite.py Glm4MoeLiteAttention):
      Q path: x -> q_a_proj -> q_a_layernorm -> q_b_proj
              -> split(q_nope, q_pe) -> RoPE(q_pe)
              -> k_up_proj(q_nope) -> q_NTA (latent space, head-major)
      KV path: x -> kv_a_proj_with_mqa -> split(k_compressed, k_pe)
               -> kv_a_layernorm on k_compressed -> RoPE(k_pe)
               -> pass compressed k_SA and k_rope_SH to MLA kernel
      kv_b_proj is decomposed into k_up_proj and v_up_proj during weight loading
      (via MLAEinsum). The MLA kernel operates in latent space; v_up_proj maps
      output back to head dim space after attention.
    """

    def __init__(self,
                 config,
                 dtype: jnp.dtype,
                 rng: nnx.Rngs,
                 mesh: Mesh,
                 kv_cache_dtype: str,
                 quant_config,
                 prefix: str = ""):
        self.dtype = dtype
        self.mesh = mesh
        self.prefix = prefix

        self.hidden_size: int = config.hidden_size
        self.num_heads: int = config.num_attention_heads
        self.num_kv_heads: int = config.num_key_value_heads
        self.q_lora_rank: int = config.q_lora_rank
        self.kv_lora_rank: int = config.kv_lora_rank
        self.qk_nope_head_dim: int = config.qk_nope_head_dim
        self.qk_rope_head_dim: int = config.qk_rope_head_dim
        self.qk_head_dim: int = getattr(
            config, "qk_head_dim",
            self.qk_nope_head_dim + self.qk_rope_head_dim)
        self.v_head_dim: int = config.v_head_dim
        self.rms_norm_eps: float = config.rms_norm_eps

        self.rope_theta: float = _get_rope_theta(config)
        self.rope_scaling = None  # standard RoPE, no scaling
        self.rope_input_ordering: str = ("interleaved" if getattr(
            config, "rope_interleave", True) else "split")

        sharding_size = mesh.shape["model"]
        self.num_heads = utils.get_padded_num_heads(self.num_heads,
                                                    sharding_size)
        self.num_kv_heads = utils.get_padded_num_heads(self.num_kv_heads,
                                                       sharding_size)
        # Alias for MLAEinsum compatibility
        self.N = self.num_heads

        # softmax scale: (qk_head_dim)^(-0.5)
        self.scale = self.qk_head_dim**-0.5

        # anh_sharding describes the (kv_lora_rank, num_heads, head_dim)
        # layout of the k_up_proj/v_up_proj weights created by MLAEinsum
        # during weight loading (see deepseek_v3.py) -- heads (axis 1)
        # sharded by 'model'.
        self.anh_sharding = (None, ShardingAxisName2D.MLP_TENSOR, None)

        # Q path
        self.q_a_proj = JaxEinsum(
            "TD,DA->TA",
            (self.hidden_size, self.q_lora_rank),
            dtype=dtype,
            kernel_init=nnx.with_partitioning(init_fn, (None, None)),
            rngs=rng,
            quant_config=quant_config,
            prefix=prefix + ".q_a_proj",
        )
        self.q_a_layernorm = JaxRmsNorm(
            self.q_lora_rank,
            epsilon=self.rms_norm_eps,
            dtype=dtype,
            scale_init=nnx.with_partitioning(init_fn, (None, )),
            rngs=rng,
            quant_config=quant_config,
            prefix=prefix + ".q_a_layernorm",
        )
        self.q_b_proj = JaxEinsum(
            "TA,AP->TP",
            (self.q_lora_rank, self.num_heads * self.qk_head_dim),
            dtype=dtype,
            kernel_init=nnx.with_partitioning(init_fn, (None, "model")),
            rngs=rng,
            quant_config=quant_config,
            prefix=prefix + ".q_b_proj",
        )

        # KV path
        self.kv_a_proj_with_mqa = JaxEinsum(
            "SD,DA->SA",
            (self.hidden_size, self.kv_lora_rank + self.qk_rope_head_dim),
            dtype=dtype,
            kernel_init=nnx.with_partitioning(init_fn, (None, None)),
            rngs=rng,
            quant_config=quant_config,
            prefix=prefix + ".kv_a_proj_with_mqa",
        )
        self.kv_a_layernorm = JaxRmsNorm(
            self.kv_lora_rank,
            epsilon=self.rms_norm_eps,
            dtype=dtype,
            scale_init=nnx.with_partitioning(init_fn, (None, )),
            rngs=rng,
            quant_config=quant_config,
            prefix=prefix + ".kv_a_layernorm",
        )
        # kv_b_proj: decomposed into k_up_proj/v_up_proj via MLAEinsum.
        # After weight loading, self.k_up_proj and self.v_up_proj are created,
        # and kv_b_proj weight is deleted.
        self.kv_b_proj = MLAEinsum(
            mla_layer=self,
            einsum_str="SA,AL->SL",
            kernel_shape=(self.kv_lora_rank,
                          self.num_heads *
                          (self.qk_nope_head_dim + self.v_head_dim)),
            rngs=rng,
            quant_config=quant_config,
            dtype=dtype,
            kernel_init=nnx.with_partitioning(init_fn, (None, "model")),
            prefix=prefix + ".kv_b_proj",
        )

        # Output projection: (N, v_head_dim) -> hidden_size.
        # Use 3D kernel (N, H, D) so JaxAutoWeightsLoader's o_proj.weight
        # handler (which expects 3D) can reshape the HF (D, N*H) weight correctly.
        self.o_proj = JaxEinsum(
            "TNH,NHD->TD",
            (self.num_heads, self.v_head_dim, self.hidden_size),
            dtype=dtype,
            kernel_init=nnx.with_partitioning(init_fn, ("model", None, None)),
            rngs=rng,
            quant_config=quant_config,
            prefix=prefix + ".o_proj",
        )

        self.kv_cache_quantized_dtype = None
        self._k_scale = 1
        if kv_cache_dtype != "auto":
            self.kv_cache_quantized_dtype = utils.get_jax_dtype_from_str_dtype(
                kv_cache_dtype)

    def __call__(
        self,
        kv_cache: Optional[jax.Array],
        x: jax.Array,
        attention_metadata: AttentionMetadata,
    ) -> Tuple[jax.Array, jax.Array]:
        md = attention_metadata
        x = jnp.asarray(x, self.dtype)

        # ===== Query path =====
        q_TA = self.q_a_proj(x)
        q_TA = self.q_a_layernorm(q_TA)
        q_TP = self.q_b_proj(q_TA)
        q_TNH = q_TP.reshape(x.shape[0], self.num_heads, self.qk_head_dim)
        q_nope = q_TNH[..., :self.qk_nope_head_dim]
        q_rope_TNH = q_TNH[..., self.qk_nope_head_dim:]
        q_rope_TNH = apply_rope(q_rope_TNH,
                                md.input_positions,
                                self.qk_rope_head_dim,
                                self.rope_theta,
                                self.rope_scaling,
                                rope_input_ordering=self.rope_input_ordering)
        # Project q_nope into latent space via absorbed k_up_proj.
        # k_up_proj's einsum is "TNH,ANH->NTA" -- output is head-major (N,T,A).
        q_NTA = self.k_up_proj(q_nope)

        # ===== KV path (compressed, no kv_b_proj expansion) =====
        kv_SA = self.kv_a_proj_with_mqa(x)
        k_rope_SH = kv_SA[..., self.kv_lora_rank:]
        kv_SA = kv_SA[..., :self.kv_lora_rank]
        kv_SA = self.kv_a_layernorm(kv_SA)

        # RoPE on k_pe (single head)
        k_rope_SNH = k_rope_SH[:, None, :]  # [S, 1, rope_D]
        k_rope_SNH = apply_rope(k_rope_SNH,
                                md.input_positions,
                                self.qk_rope_head_dim,
                                self.rope_theta,
                                self.rope_scaling,
                                rope_input_ordering=self.rope_input_ordering)
        k_rope_SH = k_rope_SNH[:, 0, :]

        # KV cache quantization (if configured)
        q_scale = k_scale = None
        if self.kv_cache_quantized_dtype:
            k_scale = self._k_scale
            kv_SA, _ = quantize_kv(self.kv_cache_quantized_dtype,
                                   kv_SA, value=None, k_scale=k_scale)
            k_rope_SH, _ = quantize_kv(self.kv_cache_quantized_dtype,
                                       k_rope_SH, value=None, k_scale=k_scale)

        # ===== MLA attention via the shared mla_attention() interface =====
        # Uses the generic tpu_inference.layers.common.attention_interface
        # implementation (same one the vLLM/torchax path uses), which
        # correctly shards seq_lens/query_start_loc/etc. by the attention-DP
        # axis under DP-attention configs (required for MLA models on this
        # platform). A hand-rolled shard_map here would need to replicate
        # that DP-aware sharding to avoid shape mismatches in the kernel's
        # static validation (e.g. cu_q_lens length depends on DP replica
        # count, not just max_num_seqs + 1).
        new_kv_cache, outputs_NTA = mla_attention(
            q_NTA,
            q_rope_TNH,
            kv_SA,
            k_rope_SH,
            kv_cache,
            md,
            self.mesh,
            self.num_heads,
            self.qk_nope_head_dim,
            q_scale=q_scale,
            k_scale=k_scale,
            v_scale=k_scale,
            sm_scale=self.scale,
        )

        # v_up_proj's einsum is "NTA,ANH->TNH" -- consumes head-major input,
        # produces token-major output ready for o_proj.
        outputs_TNH = self.v_up_proj(outputs_NTA)

        # Output projection
        o = self.o_proj(outputs_TNH)
        return new_kv_cache, o


class Glm4MoeLiteMLP(JaxModule):
    """Dense SwiGLU MLP for the first_k_dense_replace layers."""

    def __init__(self,
                 hidden_size: int,
                 intermediate_size: int,
                 hidden_act: str,
                 dtype: jnp.dtype,
                 rng: nnx.Rngs,
                 quant_config,
                 prefix: str = ""):
        self.act_fn = modeling_flax_utils.ACT2FN[hidden_act]
        self.gate_proj = JaxLinear(
            hidden_size,
            intermediate_size,
            use_bias=False,
            dtype=dtype,
            kernel_init=nnx.with_partitioning(init_fn, (None, "model")),
            rngs=rng,
            quant_config=quant_config,
            prefix=prefix + ".gate_proj",
        )
        self.up_proj = JaxLinear(
            hidden_size,
            intermediate_size,
            use_bias=False,
            dtype=dtype,
            kernel_init=nnx.with_partitioning(init_fn, (None, "model")),
            rngs=rng,
            quant_config=quant_config,
            prefix=prefix + ".up_proj",
        )
        self.down_proj = JaxLinear(
            intermediate_size,
            hidden_size,
            use_bias=False,
            dtype=dtype,
            kernel_init=nnx.with_partitioning(init_fn, ("model", None)),
            rngs=rng,
            quant_config=quant_config,
            prefix=prefix + ".down_proj",
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class Glm4MoeLiteMoELayer(JaxModule):
    """MoE layer for GLM-4.7-Flash.

    64 routed experts + 1 shared expert. Routing uses sigmoid scoring_func
    with noaux_tc (bias-corrected) top-k, same scheme as DeepSeek-V3 / GLM-5.

    Expert checkpoint weights are stored unfused per-expert
    (gate_proj/up_proj/down_proj), matching JaxMoE's expected layout --
    unlike GLM-5's fused gate_up_proj checkpoint, no split/transpose is
    needed during weight loading.
    """

    def __init__(self,
                 hf_config,
                 dtype: jnp.dtype,
                 rng: nnx.Rngs,
                 mesh: Mesh,
                 moe_backend: MoEBackend,
                 num_expert_parallelism: int,
                 quant_config,
                 prefix: str = ""):
        hidden_size = hf_config.hidden_size
        n_routed_experts = getattr(hf_config, "n_routed_experts",
                                   getattr(hf_config, "num_local_experts", 64))
        moe_intermediate_size = hf_config.moe_intermediate_size
        num_experts_per_tok = hf_config.num_experts_per_tok
        n_group = getattr(hf_config, "n_group", 1)
        topk_group = getattr(hf_config, "topk_group", 1)
        norm_topk_prob = getattr(hf_config, "norm_topk_prob", True)
        routed_scaling_factor = getattr(hf_config, "routed_scaling_factor", 1.0)
        n_shared_experts = getattr(hf_config, "n_shared_experts", 1)
        hidden_act = hf_config.hidden_act
        scoring_func = getattr(hf_config, "scoring_func", "sigmoid")

        # When running via vllm.LLM() the mesh may be 2-D ('data', 'model')
        # and lack the 5-D axes (attn_dp, attn_dp_expert, expert).
        _mesh_axes = set(mesh.axis_names)

        def fa(ax):
            """Return ax with only the mesh-available axis names kept."""
            if ax is None:
                return None
            if isinstance(ax, str):
                return ax if ax in _mesh_axes else None
            kept = tuple(a for a in ax if a in _mesh_axes)
            return kept if kept else None

        if moe_backend == MoEBackend.GMM_TP:
            moe_activation_ffw_td = P(fa(ShardingAxisName.MLP_DATA), None)
            moe_activation_ffw_ted = P(fa(ShardingAxisName.MLP_DATA), None,
                                       fa(ShardingAxisName.MOE_TENSOR))
            moe_edf_sharding = P(None, fa(ShardingAxisName.ATTN_DATA_EXPERT),
                                 fa(ShardingAxisName.MOE_TENSOR))
            moe_efd_sharding = P(None, fa(ShardingAxisName.MOE_TENSOR),
                                 fa(ShardingAxisName.ATTN_DATA_EXPERT))
        else:
            moe_activation_ffw_td = P(fa(ShardingAxisName.MLP_DATA),
                                      fa(ShardingAxisName.MOE_TENSOR))
            moe_activation_ffw_ted = P(fa(ShardingAxisName.MLP_DATA), None,
                                       fa(ShardingAxisName.MOE_TENSOR))
            moe_edf_sharding = P(fa(ShardingAxisName.ATTN_DATA_EXPERT),
                                 None, None)
            moe_efd_sharding = P(fa(ShardingAxisName.ATTN_DATA_EXPERT),
                                 None, None)

        self.gate = DeepSeekV3Router(
            hidden_size=hidden_size,
            num_experts=n_routed_experts,
            num_experts_per_tok=num_experts_per_tok,
            n_groups=n_group,
            topk_groups=topk_group,
            norm_topk_prob=norm_topk_prob,
            rngs=rng,
            routed_scaling_factor=routed_scaling_factor,
            dtype=dtype,
            moe_backend=moe_backend,
            activation_ffw_td=P(fa(ShardingAxisName.MLP_DATA), None),
            ed_sharding=P(None, None),
            e_sharding=P(None, ),
            scoring_func=scoring_func,
            quant_config=quant_config,
        )

        self.shared_experts = DeepseekV3MLP(
            dtype=dtype,
            hidden_act=hidden_act,
            hidden_size=hidden_size,
            intermediate_size=n_shared_experts * moe_intermediate_size,
            rngs=rng,
            activation_ffw_td=P(fa(ShardingAxisName.MLP_DATA), None),
            df_sharding=P(None, fa(ShardingAxisName.ATTN_HEAD)),
            fd_sharding=P(fa(ShardingAxisName.ATTN_HEAD), None),
            quant_config=quant_config,
        )

        self.experts = SharedFusedMoe(
            dtype=dtype,
            num_local_experts=n_routed_experts,
            apply_expert_weight_before_computation=False,
            expert_axis_name=expert_axis_name,
            num_expert_parallelism=num_expert_parallelism,
            hidden_size=hidden_size,
            intermediate_size_moe=moe_intermediate_size,
            num_experts_per_tok=num_experts_per_tok,
            mesh=mesh,
            hidden_act=hidden_act,
            rngs=rng,
            quant_config=quant_config,
            activation_ffw_td=moe_activation_ffw_td,
            activation_ffw_ted=moe_activation_ffw_ted,
            edf_sharding=moe_edf_sharding,
            efd_sharding=moe_efd_sharding,
            moe_backend=moe_backend,
            qwix_quantized_weight_dtype=None,
            prefix=f"{prefix}.experts",
            router=self.gate,
            shared_experts=self.shared_experts,
            scoring_func=scoring_func,
            routed_scaling_factor=routed_scaling_factor,
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        # SharedFusedMoe.__call__ returns (hidden_states, expert_indices);
        # dense DeepseekV3MLP returns a single tensor. Normalize to a single
        # tensor here (matches DeepseekV3DecoderLayer's handling upstream).
        output = self.experts(x)
        if isinstance(output, tuple):
            output = output[0]
        return output


class Glm4MoeLiteDecoderLayer(JaxModule):

    def __init__(self,
                 config,
                 layer_index: int,
                 dtype: jnp.dtype,
                 rng: nnx.Rngs,
                 mesh: Mesh,
                 kv_cache_dtype: str,
                 moe_backend: MoEBackend,
                 num_expert_parallelism: int,
                 quant_config,
                 prefix: str = ""):
        hidden_size = config.hidden_size
        rms_norm_eps = config.rms_norm_eps

        self.input_layernorm = JaxRmsNorm(
            hidden_size,
            epsilon=rms_norm_eps,
            dtype=dtype,
            scale_init=nnx.with_partitioning(init_fn, (None, )),
            rngs=rng,
            quant_config=quant_config,
            prefix=prefix + ".input_layernorm",
        )
        self.self_attn = Glm4MoeLiteAttention(
            config=config,
            dtype=dtype,
            rng=rng,
            mesh=mesh,
            kv_cache_dtype=kv_cache_dtype,
            quant_config=quant_config,
            prefix=prefix + ".self_attn",
        )
        self.post_attention_layernorm = JaxRmsNorm(
            hidden_size,
            epsilon=rms_norm_eps,
            dtype=dtype,
            scale_init=nnx.with_partitioning(init_fn, (None, )),
            rngs=rng,
            quant_config=quant_config,
            prefix=prefix + ".post_attention_layernorm",
        )

        first_k_dense = getattr(config, "first_k_dense_replace", 1)
        if layer_index < first_k_dense:
            self.mlp = Glm4MoeLiteMLP(
                hidden_size=hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                dtype=dtype,
                rng=rng,
                quant_config=quant_config,
                prefix=prefix + ".mlp",
            )
        else:
            self.mlp = Glm4MoeLiteMoELayer(
                hf_config=config,
                dtype=dtype,
                rng=rng,
                mesh=mesh,
                moe_backend=moe_backend,
                num_expert_parallelism=num_expert_parallelism,
                quant_config=quant_config,
                prefix=prefix + ".mlp",
            )

    def __call__(
        self,
        kv_cache: jax.Array,
        x: jax.Array,
        attention_metadata: AttentionMetadata,
    ) -> Tuple[jax.Array, jax.Array]:
        hidden = self.input_layernorm(x)
        kv_cache, attn_out = self.self_attn(kv_cache, hidden, attention_metadata)
        x = x + attn_out

        residual = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        x = residual + x
        return kv_cache, x


class Glm4MoeLiteModel(JaxModule):

    def __init__(self,
                 vllm_config: VllmConfig,
                 rng: nnx.Rngs,
                 mesh: Mesh,
                 prefix: str = "model") -> None:
        model_config = vllm_config.model_config
        hf_config = model_config.hf_config
        vocab_size = model_config.get_vocab_size()
        dtype = model_config.dtype
        hidden_size = hf_config.hidden_size
        quant_config = vllm_config.quant_config

        self.is_first_rank = get_pp_group().is_first_rank
        self.is_last_rank = get_pp_group().is_last_rank

        num_expert_parallelism = get_expert_parallelism(expert_axis_name, mesh)
        total_tp = (vllm_config.sharding_config.tp_size *
                    vllm_config.sharding_config.attn_dp_size)
        use_ep = num_expert_parallelism > 1 and total_tp == 1
        moe_backend = select_moe_backend(use_ep)

        if vllm_config.load_config.load_format == "dummy" and moe_backend in MoEBackend.fused_moe_backends():
            raise ValueError(
                f"Dummy weights not supported for {MoEBackend.fused_moe_backends()} backends."
            )

        if self.is_first_rank:
            self.embed_tokens = JaxEmbed(
                num_embeddings=vocab_size,
                features=hidden_size,
                dtype=dtype,
                embedding_init=nnx.with_partitioning(init_fn, ("model", None)),
                rngs=rng,
                quant_config=quant_config,
                prefix=prefix + ".embed_tokens",
            )
        else:
            self.embed_tokens = PPMissingLayer()

        self.start_layer, self.end_layer, self.layers = make_layers(
            hf_config.num_hidden_layers,
            lambda layer_index: Glm4MoeLiteDecoderLayer(
                config=hf_config,
                layer_index=layer_index,
                dtype=dtype,
                rng=rng,
                mesh=mesh,
                kv_cache_dtype=vllm_config.cache_config.cache_dtype,
                moe_backend=moe_backend,
                num_expert_parallelism=num_expert_parallelism,
                quant_config=quant_config,
                prefix=f"{prefix}.layers.{layer_index}",
            ),
        )

        if self.is_last_rank:
            self.norm = JaxRmsNorm(
                hidden_size,
                epsilon=hf_config.rms_norm_eps,
                dtype=dtype,
                scale_init=nnx.with_partitioning(init_fn, (None, )),
                rngs=rng,
                quant_config=quant_config,
                prefix=prefix + ".norm",
            )
        else:
            self.norm = PPMissingLayer()

    def __call__(
        self,
        kv_caches: List[jax.Array],
        input_ids: Optional[jax.Array],
        attention_metadata: AttentionMetadata,
        inputs_embeds: Optional[jax.Array] = None,
    ) -> Tuple[List[jax.Array], jax.Array]:
        if inputs_embeds is not None:
            x = inputs_embeds
        else:
            x = self.embed_tokens(input_ids)

        for i, layer in enumerate(
                islice(self.layers, self.start_layer, self.end_layer)):
            kv_cache = kv_caches[i]
            kv_cache, x = layer(kv_cache, x, attention_metadata)
            kv_caches[i] = kv_cache

        x = self.norm(x)
        return kv_caches, x


class Glm4MoeLiteForCausalLM(JaxModule, LoadableWithIterator):
    """Top-level GLM-4.7-Flash model (Glm4MoeLiteForCausalLM).

    Weight loading is simpler than GLM-5: the checkpoint stores expert
    weights unfused per-expert (gate_proj/up_proj/down_proj), matching
    JaxMoE's expected layout directly, so no split/transpose transform is
    needed -- JaxAutoWeightsLoader handles it like DeepseekV3ForCausalLM.
    """

    def __init__(self, vllm_config: VllmConfig, rng_key: jax.Array,
                 mesh: Mesh) -> None:
        if getattr(vllm_config.model_config, "quantization", None) == "fp8":
            from tpu_inference.layers.jax.quantization.fp8 import Fp8Config
            hg_quant_config = getattr(vllm_config.model_config.hf_config,
                                      "quantization_config", {})
            vllm_config.quant_config = Fp8Config(hg_quant_config)

        self.vllm_config = vllm_config
        rng = nnx.Rngs(rng_key)
        self.mesh = mesh

        self.model = Glm4MoeLiteModel(
            vllm_config=vllm_config,
            rng=rng,
            mesh=mesh,
            prefix="model",
        )

        model_config = vllm_config.model_config
        if self.model.is_last_rank:
            vocab_size = model_config.get_vocab_size()
            hidden_size = model_config.hf_config.hidden_size
            self.lm_head = JaxEinsum(
                einsum_str="TD,DV->TV",
                kernel_shape=(hidden_size, vocab_size),
                dtype=model_config.dtype,
                rngs=rng,
                quant_config=None,  # lm_head typically not quantized
                prefix="lm_head",
            )
        else:
            self.lm_head = PPMissingLayer()

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
    ) -> Tuple[List[jax.Array], jax.Array | JaxIntermediateTensors,
               List[jax.Array], Optional[jax.Array]]:
        if not is_first_rank:
            assert intermediate_tensors is not None
            inputs_embeds = intermediate_tensors["hidden_states"]

        kv_caches, x = self.model(
            kv_caches,
            input_ids,
            attention_metadata,
            inputs_embeds,
        )

        if not is_last_rank:
            x = JaxIntermediateTensors(tensors={"hidden_states": x})

        # 4th element is expert routing indices (for enable_return_routed_experts);
        # unused here, so None -- matches run_model's out_shardings length of 4.
        return kv_caches, x, [], None

    def compute_logits(self, hidden_states: jax.Array) -> jax.Array:
        return self.lm_head(hidden_states)

    def load_weights(self, weights: Iterable) -> set[str]:
        """Load weights.

        GLM-4.7-Flash checkpoint expert weight layout is already unfused
        (mlp.experts.{i}.gate_proj / up_proj / down_proj), matching JaxMoE's
        (SharedFusedMoe) expected layout directly -- no transform needed.
        """
        if not isinstance(weights, Iterable):
            return super().load_weights(weights)

        num_layers = len(self.model.layers)
        total_layers = self.vllm_config.model_config.hf_config.num_hidden_layers
        # Skip layers beyond what the model uses. The checkpoint also
        # contains num_nextn_predict_layers extra MTP layer(s) appended after
        # num_hidden_layers real layers -- must skip those too.
        nextn_layers = getattr(
            self.vllm_config.model_config.hf_config,
            "num_nextn_predict_layers", 0)
        skip_layer_ids = list(range(num_layers, total_layers + nextn_layers))
        loader = JaxAutoWeightsLoader(
            self,
            skip_prefixes=(["lm_head"]
                           if not hasattr(self, "lm_head") else []),
            skip_substrs=[f"layers.{i}" for i in skip_layer_ids],
        )
        loaded = loader.load_weights(weights)

        if os.environ.get("VLLM_LOGGING_LEVEL", "").upper() == "DEBUG":
            logger.debug("Glm4MoeLiteForCausalLM parameter dtypes:")
            num_to_display = 3
            should_skip = False
            for name, param in self.named_parameters():
                if f"layers.{num_to_display}." in name:
                    should_skip = True
                if should_skip and "layers." in name:
                    continue
                v: jax.Array = param.value
                logger.debug(f"{name} : {v.dtype}{v.shape} on {v.device}")

        return loaded
