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

# JAX (flax_nnx) implementation of GLM-5 (GlmMoeDsaForCausalLM).
#
# Architecture:
#   - MLA attention (same structure as DeepSeek-V3) with standard RoPE (theta=1e6)
#   - DSA (Dynamic Sparse Attention) Indexer sub-module — weights loaded, attention
#     currently runs as full (dense) attention (TODO: implement JAX DSA masking)
#   - MoE layers (256 routed + 1 shared expert) with sigmoid routing
#   - Expert checkpoint weights use fused layout (gate_up_proj, transposed down_proj)
#     which are split/transposed during weight loading
#
# Reference: huggingface/transformers models/glm_moe_dsa/modeling_glm_moe_dsa.py

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
from tpu_inference.kernels.mla.v2.kernel import mla_ragged_paged_attention
from tpu_inference.layers.common.attention_interface import attention
from tpu_inference.layers.common.quantization import quantize_kv
from tpu_inference.layers.common.attention_metadata import AttentionMetadata
from tpu_inference.layers.common.moe import MoEBackend
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


class Glm5LayerNorm(JaxModule):
    """LayerNorm (with mean subtraction) for the DSA indexer's k_norm.

    PyTorch uses nn.LayerNorm for the indexer k_norm, which differs from
    RMSNorm by subtracting the mean. This implementation matches the reference.
    """

    def __init__(self, size: int, eps: float = 1e-6, dtype: jnp.dtype = jnp.bfloat16):
        self.eps = eps
        self.dtype = dtype
        # Params named .weight / .bias to match PyTorch nn.LayerNorm checkpoint keys.
        # Use nnx.with_partitioning so weight loading places them on TPU.
        _ones = nnx.with_partitioning(nnx.initializers.ones_init(), (None,))
        _zeros = nnx.with_partitioning(nnx.initializers.zeros_init(), (None,))
        self.weight = nnx.Param(_ones(jax.random.key(0), (size,), jnp.float32))
        self.bias = nnx.Param(_zeros(jax.random.key(0), (size,), jnp.float32))

    def __call__(self, x: jax.Array) -> jax.Array:
        x_f = x.astype(jnp.float32)
        mean = jnp.mean(x_f, axis=-1, keepdims=True)
        var = jnp.var(x_f, axis=-1, keepdims=True)
        x_norm = (x_f - mean) / jnp.sqrt(var + self.eps)
        return (self.weight * x_norm + self.bias).astype(self.dtype)


class Glm5Indexer(JaxModule):
    """DSA (Dynamic Sparse Attention) indexer for GLM-5.

    Computes per-query top-k token scores using lightweight projections.
    In the reference model this is used to mask attention to the top-index_topk
    tokens. Weight names match the PyTorch checkpoint exactly:
      indexer.wq_b, indexer.wk, indexer.k_norm, indexer.weights_proj

    Note: The DSA masking is not applied in the current JAX implementation
    (the attention kernel does not support dynamic sparse masks). The indexer
    weights are loaded correctly so that future DSA support can be added.
    TODO: integrate DSA scoring mask into the attention forward pass.
    """

    def __init__(self,
                 hidden_size: int,
                 q_lora_rank: int,
                 index_n_heads: int,
                 index_head_dim: int,
                 qk_rope_head_dim: int,
                 index_topk: int,
                 dtype: jnp.dtype,
                 rng: nnx.Rngs,
                 quant_config=None,
                 prefix: str = ""):
        self.index_n_heads = index_n_heads
        self.index_head_dim = index_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.index_topk = index_topk
        self.dtype = dtype

        # wq_b: (q_lora_rank → n_heads * head_dim)
        self.wq_b = JaxLinear(
            q_lora_rank,
            index_n_heads * index_head_dim,
            use_bias=False,
            dtype=dtype,
            kernel_init=nnx.with_partitioning(init_fn, (None, "model")),
            rngs=rng,
            quant_config=quant_config,
            prefix=prefix + ".wq_b",
        )
        # wk: (hidden_size → head_dim) — single-head key projection
        self.wk = JaxLinear(
            hidden_size,
            index_head_dim,
            use_bias=False,
            dtype=dtype,
            kernel_init=nnx.with_partitioning(init_fn, (None, None)),
            rngs=rng,
            quant_config=quant_config,
            prefix=prefix + ".wk",
        )
        # k_norm: LayerNorm (not RMSNorm) on index_head_dim
        self.k_norm = Glm5LayerNorm(index_head_dim, eps=1e-6, dtype=dtype)
        # weights_proj: (hidden_size → n_heads) — per-head gate weights for scoring.
        # quant_config=None: this is a BF16 parameter (listed in modules_to_not_convert
        # as "indexers_proj"), must NOT apply FP8 quantization.
        self.weights_proj = JaxLinear(
            hidden_size,
            index_n_heads,
            use_bias=False,
            dtype=dtype,
            kernel_init=nnx.with_partitioning(init_fn, ("model", None)),
            rngs=rng,
            quant_config=None,
            prefix=prefix + ".weights_proj",
        )

    def __call__(self, hidden_states: jax.Array,
                 q_resid: jax.Array) -> Optional[jax.Array]:
        # TODO: implement DSA top-k scoring and return indices for attention masking.
        # Current implementation is a no-op; attention falls back to full (dense) mode.
        return None


class Glm5Attention(JaxModule):
    """MLA attention for GLM-5.

    Architecture (matches HF modeling_glm_moe_dsa.py GlmMoeDsaAttention):
      Q path: x → q_a_proj → q_a_layernorm → q_b_proj
              → split(q_nope, q_pe) → RoPE(q_pe)
              → k_up_proj(q_nope) → q_TNA (latent space)
      KV path: x → kv_a_proj_with_mqa → split(k_compressed, k_pe)
               → kv_a_layernorm on k_compressed → RoPE(k_pe)
               → pass compressed k_SA and k_rope_SH to MLA kernel
      kv_b_proj is decomposed into k_up_proj and v_up_proj during weight loading
      (via MLAEinsum). The MLA kernel operates in latent space; v_up_proj maps
      output back to head dim space after attention.
      DSA: Indexer runs alongside attention (weights loaded, masking TODO).
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
        self.qk_head_dim: int = config.qk_head_dim  # = nope + rope = 256
        self.v_head_dim: int = config.v_head_dim  # = 256
        self.rms_norm_eps: float = config.rms_norm_eps

        rope_params = getattr(config, "rope_parameters", {})
        self.rope_theta: float = rope_params.get("rope_theta", 1_000_000)
        self.rope_scaling = None  # standard RoPE, no scaling

        sharding_size = mesh.shape["model"]
        self.num_heads = utils.get_padded_num_heads(self.num_heads,
                                                    sharding_size)
        self.num_kv_heads = utils.get_padded_num_heads(self.num_kv_heads,
                                                       sharding_size)
        # Alias for MLAEinsum compatibility
        self.N = self.num_heads

        # softmax scale: (qk_head_dim)^(-0.5)
        self.scale = self.qk_head_dim**-0.5

        # MLA sharding specs for 2D mesh pure-TP:
        # Q/output sharded by head (axis 1); KV/cache replicated.
        self.query_tnh = P(None, ShardingAxisName2D.MLP_TENSOR, None)
        self.keyvalue_skh = P()  # replicated — MLA KV is shared across heads
        self.attn_o_tnh = P(None, ShardingAxisName2D.MLP_TENSOR, None)
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

        # Output projection: (N, v_head_dim) → hidden_size.
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

        # DSA indexer (weights loaded; masking is a no-op for now)
        self.indexer = Glm5Indexer(
            hidden_size=self.hidden_size,
            q_lora_rank=self.q_lora_rank,
            index_n_heads=config.index_n_heads,
            index_head_dim=config.index_head_dim,
            qk_rope_head_dim=self.qk_rope_head_dim,
            index_topk=config.index_topk,
            dtype=dtype,
            rng=rng,
            quant_config=quant_config,
            prefix=prefix + ".indexer",
        )

        self.kv_cache_quantized_dtype = None
        self._k_scale = 1
        if kv_cache_dtype != "auto":
            self.kv_cache_quantized_dtype = utils.get_jax_dtype_from_str_dtype(
                kv_cache_dtype)

    def _create_mla_submodules(self, quant_config):
        """Create k_up_proj/v_up_proj shells for weight cache restore.

        Normally these are created dynamically by MLAEinsum.load_weights()
        during weight loading, which also deletes kv_b_proj's weight and
        weight_scale_inv. This method replicates that structural change
        so the model tree matches the cached state exactly.
        """
        if hasattr(self, 'k_up_proj'):
            return
        A = self.kv_lora_rank
        N = self.num_heads
        self.k_up_proj = JaxEinsum(
            einsum_str="TNH,ANH->TNA",
            kernel_shape=(A, N, self.qk_nope_head_dim),
            rngs=nnx.Rngs(0),
            prefix=self.prefix + ".k_up_proj",
            quant_config=quant_config,
        )
        self.v_up_proj = JaxEinsum(
            einsum_str="TNA,ANH->TNH",
            kernel_shape=(A, N, self.v_head_dim),
            rngs=nnx.Rngs(0),
            prefix=self.prefix + ".v_up_proj",
            quant_config=quant_config,
        )
        # MLAEinsum.load_weights deletes these after decomposition.
        # Must mirror that here so the tree structure matches the saved cache.
        kv_b = self.kv_b_proj
        if hasattr(kv_b, 'weight'):
            delattr(kv_b, 'weight')
        if hasattr(kv_b, 'weight_scale_inv'):
            delattr(kv_b, 'weight_scale_inv')
        if hasattr(kv_b, 'quant_method'):
            delattr(kv_b, 'quant_method')

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
        q_resid = q_TA  # saved for indexer
        q_TP = self.q_b_proj(q_TA)
        q_TNH = q_TP.reshape(x.shape[0], self.num_heads, self.qk_head_dim)
        q_nope = q_TNH[..., :self.qk_nope_head_dim]
        q_rope_TNH = q_TNH[..., self.qk_nope_head_dim:]
        q_rope_TNH = apply_rope(q_rope_TNH,
                                md.input_positions,
                                self.qk_rope_head_dim,
                                self.rope_theta,
                                self.rope_scaling)
        # Project q_nope into latent space via absorbed k_up_proj
        q_TNA = self.k_up_proj(q_nope)
        q_TNA = jax.lax.with_sharding_constraint(q_TNA, self.query_tnh)

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
                                self.rope_scaling)
        k_rope_SH = k_rope_SNH[:, 0, :]

        # DSA indexer (no-op for now; runs for side-effect of weight validation)
        self.indexer(x, q_resid)

        # KV cache quantization (if configured)
        q_scale = k_scale = None
        if self.kv_cache_quantized_dtype:
            k_scale = self._k_scale
            kv_SA, _ = quantize_kv(self.kv_cache_quantized_dtype,
                                   kv_SA, value=None, k_scale=k_scale)
            k_rope_SH, _ = quantize_kv(self.kv_cache_quantized_dtype,
                                       k_rope_SH, value=None, k_scale=k_scale)

        # ===== MLA attention via custom shard_map for 2D mesh =====
        # On 2D mesh (pure TP), the shared mla_attention() incorrectly shards
        # both tokens and cache pages by MLP_TENSOR.  Instead we shard Q by
        # head and replicate KV cache so every device sees all pages.
        _sm_scale = self.scale
        _q_scale = q_scale
        _k_scale = k_scale

        in_specs = (
            P(None, 'model', None),  # q_TNA:  shard heads
            P(None, 'model', None),  # q_rope: shard heads
            P(),                     # kv_SA:  replicated
            P(),                     # k_rope: replicated
            P(),                     # cache:  replicated
            P(),                     # seq_lens
            P(),                     # page_indices
            P(),                     # query_start_loc
            P(),                     # distribution
        )
        out_specs = (
            P(),                     # new_cache: replicated
            P(None, 'model', None),  # attn output: shard heads
        )

        def _mla_fn(q, q_rope, k, k_rope, cache, *args):
            out, new_cache = mla_ragged_paged_attention(
                q, q_rope, k, k_rope, cache, *args,
                sm_scale=_sm_scale,
                num_kv_pages_per_block=3,
                num_queries_per_block=1,
                q_scale=_q_scale,
                k_scale=_k_scale,
                v_scale=_k_scale,
            )
            return new_cache, out

        new_kv_cache, outputs_TNA = jax.jit(
            jax.shard_map(
                _mla_fn,
                mesh=self.mesh,
                in_specs=in_specs,
                out_specs=out_specs,
                check_vma=False,
            )
        )(q_TNA, q_rope_TNH, kv_SA, k_rope_SH, kv_cache,
          md.seq_lens, md.block_tables, md.query_start_loc,
          md.request_distribution)

        # Map from latent space to head dim via v_up_proj
        outputs_TNH = self.v_up_proj(outputs_TNA)

        # Output projection
        o = self.o_proj(outputs_TNH)
        return new_kv_cache, o


class Glm5MLP(JaxModule):
    """Dense SwiGLU MLP for the first_k_dense_replace=3 layers."""

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


class Glm5MoELayer(JaxModule):
    """MoE layer for GLM-5.

    256 routed experts + 1 shared expert.
    Routing uses sigmoid scoring_func (unlike DeepSeek-V3 which uses sigmoid too,
    but GLM-5 sets scoring_func="sigmoid" explicitly).

    Expert weight format in checkpoint differs from JAX layout:
      gate_up_proj [E, 2*F, D] → split+transpose → gate [E, D, F], up [E, D, F]
      down_proj    [E, D, F]   → transpose        → [E, F, D]
    The custom load_weights in Glm5ForCausalLM handles this transformation.
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
                                   getattr(hf_config, "num_local_experts", 256))
        moe_intermediate_size = hf_config.moe_intermediate_size
        num_experts_per_tok = hf_config.num_experts_per_tok
        n_group = getattr(hf_config, "n_group", 1)
        topk_group = getattr(hf_config, "topk_group", 1)
        norm_topk_prob = getattr(hf_config, "norm_topk_prob", True)
        routed_scaling_factor = getattr(hf_config, "routed_scaling_factor", 2.5)
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
        return self.experts(x)


class Glm5DecoderLayer(JaxModule):

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
        self.self_attn = Glm5Attention(
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

        first_k_dense = getattr(config, "first_k_dense_replace", 3)
        if layer_index < first_k_dense:
            self.mlp = Glm5MLP(
                hidden_size=hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                dtype=dtype,
                rng=rng,
                quant_config=quant_config,
                prefix=prefix + ".mlp",
            )
        else:
            self.mlp = Glm5MoELayer(
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


class Glm5Model(JaxModule):

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
            lambda layer_index: Glm5DecoderLayer(
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


class Glm5ForCausalLM(JaxModule, LoadableWithIterator):
    """Top-level GLM-5 model (GlmMoeDsaForCausalLM).

    Weight loading handles:
      1. Standard weights via JaxAutoWeightsLoader (attention, layernorm, router bias,
         shared experts, dense MLP, indexer projections).
      2. Expert fused weights: mlp.experts.gate_up_proj [E, 2F, D] is split into
         gate_proj [E, D, F] and up_proj [E, D, F]; mlp.experts.down_proj [E, D, F]
         is transposed to [E, F, D].  These are yielded as separate tensors before
         passing to JaxAutoWeightsLoader.
    """

    def __init__(self, vllm_config: VllmConfig, rng_key: jax.Array,
                 mesh: Mesh) -> None:
        if getattr(vllm_config.model_config, "quantization", None) == "fp8":
            # `get_tpu_quantization_config` returns None for "fp8" because
            # the work in #1623 is not fully merged. So this block overrides
            # the logic to return Fp8Config when model_config indicates fp8.
            # TODO(#1623): Remove this block when `get_tpu_quantization_config`
            # is updated.
            from tpu_inference.layers.jax.quantization.fp8 import Fp8Config
            hg_quant_config = getattr(vllm_config.model_config.hf_config,
                                      "quantization_config", {})
            vllm_config.quant_config = Fp8Config(hg_quant_config)

        self.vllm_config = vllm_config
        rng = nnx.Rngs(rng_key)
        self.mesh = mesh

        self.model = Glm5Model(
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
               List[jax.Array]]:
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

        return kv_caches, x, []

    def _prepare_for_weight_cache(self):
        """Create dynamic MLA submodules for weight cache restore.

        MLAEinsum.load_weights() normally creates k_up_proj/v_up_proj via
        setattr during weight loading. When restoring from cache, load_weights
        is skipped, so we must create these submodule shells beforehand.
        """
        quant_config = self.vllm_config.quant_config
        for layer in self.model.layers:
            if hasattr(layer, 'self_attn'):
                layer.self_attn._create_mla_submodules(quant_config)

    def compute_logits(self, hidden_states: jax.Array) -> jax.Array:
        return self.lm_head(hidden_states)

    def load_weights(self, weights: Iterable) -> set[str]:
        """Load weights, transforming fused expert weights to split layout.

        GLM-5 checkpoint expert weight layout:
          mlp.experts.gate_up_proj  [E, 2*F, D]  (fused gate+up, PyTorch conv)
          mlp.experts.down_proj     [E, D, F]     (transposed PyTorch conv)

        JaxMoE (SharedFusedMoe) expert weight layout:
          mlp.experts.gate_proj     [E, D, F]     (in_dim, out_dim per expert)
          mlp.experts.up_proj       [E, D, F]
          mlp.experts.down_proj     [E, F, D]
        """
        if not isinstance(weights, Iterable):
            return super().load_weights(weights)

        def _perm(w):
            # Swap last two dims. Works for torch.Tensor (.permute) and numpy (.transpose).
            if hasattr(w, "permute"):
                return w.permute(0, 2, 1)
            return w.transpose(0, 2, 1)

        def _transform(raw_weights):
            for name, weight in raw_weights:
                if "mlp.experts.gate_up_proj" in name:
                    # [E, 2*F, D] → split → each [E, F, D] (PyTorch: out,in)
                    # → swap last two dims → [E, D, F] (JAX: in, out)
                    F = weight.shape[1] // 2
                    gate_name = name.replace("experts.gate_up_proj",
                                             "experts.gate_proj")
                    up_name = name.replace("experts.gate_up_proj",
                                           "experts.up_proj")
                    yield gate_name, _perm(weight[:, :F, :])
                    yield up_name, _perm(weight[:, F:, :])
                elif "mlp.experts.down_proj" in name:
                    # [E, D, F] (PyTorch: out=D, in=F) → [E, F, D] (JAX: in, out)
                    yield name, _perm(weight)
                else:
                    yield name, weight

        num_layers = len(self.model.layers)
        total_layers = self.vllm_config.model_config.hf_config.num_hidden_layers
        # Skip layers beyond what the model uses.  num_hidden_layers == 78 but
        # the checkpoint also contains layer 78 (nextn prediction layer) and
        # potentially more.  We must skip all of them.
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
        loaded = loader.load_weights(_transform(weights))

        if os.environ.get("VLLM_LOGGING_LEVEL", "").upper() == "DEBUG":
            logger.debug("Glm5ForCausalLM parameter dtypes:")
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
