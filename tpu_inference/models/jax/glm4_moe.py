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

# JAX (flax_nnx) implementation of GLM-4.5-Air (Glm4MoeForCausalLM).
# Architecture: standard GQA with partial RoPE (partial_rotary_factor=0.5),
# attention biases, first_k_dense_replace=1 dense MLP, rest MoE (128 routed +
# 1 shared experts per layer).

import os
from itertools import islice
from typing import Iterable, List, Optional, Tuple

import jax
import jax.numpy as jnp
from flax import nnx
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P
from vllm.config import VllmConfig
from vllm.model_executor.models.utils import WeightsMapper

from tpu_inference import utils
from tpu_inference.distributed.jax_parallel_state import get_pp_group
from tpu_inference.layers.common.attention_interface import attention
from tpu_inference.layers.common.attention_metadata import AttentionMetadata
from tpu_inference.layers.common.moe import MoEBackend
from tpu_inference.layers.common.quantization import quantize_kv
from tpu_inference.layers.common.sharding import \
    ShardingAxisNameBase as ShardingAxisName
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
                                                    SharedFusedMoe)
from tpu_inference.models.jax.jax_intermediate_tensor import \
    JaxIntermediateTensors
from tpu_inference.models.jax.utils.weight_utils import (JaxAutoWeightsLoader,
                                                         LoadableWithIterator)

logger = init_logger(__name__)

init_fn = nnx.initializers.uniform()
modeling_flax_utils = FlaxUtils()

expert_axis_name = ShardingAxisName.ATTN_DATA_EXPERT


class Glm4MoeMLP(JaxModule):
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
        gate = self.act_fn(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)


class Glm4MoeAttention(JaxModule):
    """GQA attention with attention biases and partial RoPE.

    GLM-4.5-Air specifics:
    - attention_bias=True  → Q/K/V projections carry bias terms
    - partial_rotary_factor=0.5 → RoPE applied to first 50% of head_dim
    """

    def __init__(self,
                 config,
                 dtype: jnp.dtype,
                 rng: nnx.Rngs,
                 mesh: Mesh,
                 kv_cache_dtype: str,
                 quant_config,
                 prefix: str = ""):
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        rope_params = getattr(config, "rope_parameters", {}) or {}
        self.rope_theta = rope_params.get("rope_theta",
                                          getattr(config, "rope_theta", 10000))
        self.rope_scaling = getattr(config, "rope_scaling", None)
        self.partial_rotary_factor = rope_params.get(
            "partial_rotary_factor",
            getattr(config, "partial_rotary_factor", 0.5))

        self.head_dim_original = getattr(config, "head_dim",
                                         self.hidden_size // self.num_heads)
        self.head_dim = utils.get_padded_head_dim(self.head_dim_original)

        sharding_size = mesh.shape["model"]
        self.num_heads = utils.get_padded_num_heads(self.num_heads,
                                                    sharding_size)
        self.num_kv_heads = utils.get_padded_num_heads(self.num_kv_heads,
                                                       sharding_size)
        self.mesh = mesh

        attention_bias = getattr(config, "attention_bias", False)
        q_bias_shape = (self.num_heads,
                        self.head_dim) if attention_bias else None
        kv_bias_shape = (self.num_kv_heads,
                         self.head_dim) if attention_bias else None

        self.q_proj = JaxEinsum(
            "TD,DNH->TNH",
            (self.hidden_size, self.num_heads, self.head_dim),
            bias_shape=q_bias_shape,
            dtype=dtype,
            kernel_init=nnx.with_partitioning(init_fn, (None, "model", None)),
            bias_init=nnx.with_partitioning(init_fn,
                                            ("model", None)) if attention_bias
            else None,
            rngs=rng,
            quant_config=quant_config,
            prefix=prefix + ".q_proj",
        )
        self.k_proj = JaxEinsum(
            "TD,DKH->TKH",
            (self.hidden_size, self.num_kv_heads, self.head_dim),
            bias_shape=kv_bias_shape,
            dtype=dtype,
            kernel_init=nnx.with_partitioning(init_fn, (None, "model", None)),
            bias_init=nnx.with_partitioning(init_fn,
                                            ("model", None)) if attention_bias
            else None,
            rngs=rng,
            quant_config=quant_config,
            prefix=prefix + ".k_proj",
        )
        self.v_proj = JaxEinsum(
            "TD,DKH->TKH",
            (self.hidden_size, self.num_kv_heads, self.head_dim),
            bias_shape=kv_bias_shape,
            dtype=dtype,
            kernel_init=nnx.with_partitioning(init_fn, (None, "model", None)),
            bias_init=nnx.with_partitioning(init_fn,
                                            ("model", None)) if attention_bias
            else None,
            rngs=rng,
            quant_config=quant_config,
            prefix=prefix + ".v_proj",
        )
        self.o_proj = JaxEinsum(
            "TNH,NHD->TD",
            (self.num_heads, self.head_dim, self.hidden_size),
            dtype=dtype,
            kernel_init=nnx.with_partitioning(init_fn,
                                              ("model", None, None)),
            rngs=rng,
            quant_config=quant_config,
            prefix=prefix + ".o_proj",
        )

        self._k_scale = 1.0
        self._v_scale = 1.0
        self.kv_cache_quantized_dtype = None
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
        q = self.q_proj(x)
        q = apply_rope(q,
                       md.input_positions,
                       self.head_dim_original,
                       self.rope_theta,
                       self.rope_scaling,
                       rope_proportion=self.partial_rotary_factor)

        k = self.k_proj(x)
        k = apply_rope(k,
                       md.input_positions,
                       self.head_dim_original,
                       self.rope_theta,
                       self.rope_scaling,
                       rope_proportion=self.partial_rotary_factor)

        v = self.v_proj(x)

        k_scale = v_scale = None
        if self.kv_cache_quantized_dtype:
            k_scale = self._k_scale
            v_scale = self._v_scale
            k, v = quantize_kv(self.kv_cache_quantized_dtype, k, v, k_scale,
                               v_scale)

        new_kv_cache, outputs = attention(
            kv_cache,
            q,
            k,
            v,
            attention_metadata,
            self.mesh,
            self.head_dim_original,
            k_scale=k_scale,
            v_scale=v_scale,
        )
        o = self.o_proj(outputs)
        return new_kv_cache, o


class Glm4MoeMoELayer(JaxModule):
    """MoE layer for GLM-4.5-Air.

    Wraps DeepSeekV3Router + SharedFusedMoe (routed experts + 1 shared expert)
    with GLM-4.5-Air specific config.
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
                                   getattr(hf_config, "num_local_experts",
                                           128))
        moe_intermediate_size = hf_config.moe_intermediate_size
        num_experts_per_tok = hf_config.num_experts_per_tok
        n_group = getattr(hf_config, "n_group", 1)
        topk_group = getattr(hf_config, "topk_group", 1)
        norm_topk_prob = getattr(hf_config, "norm_topk_prob", True)
        routed_scaling_factor = getattr(hf_config, "routed_scaling_factor",
                                        1.0)
        n_shared_experts = getattr(hf_config, "n_shared_experts", 1)
        hidden_act = hf_config.hidden_act
        scoring_func = getattr(hf_config, "scoring_func", "softmax")

        # Filter sharding axes to those actually present in the mesh.
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
            moe_edf_sharding = P(fa(ShardingAxisName.ATTN_DATA_EXPERT), None,
                                 None)
            moe_efd_sharding = P(fa(ShardingAxisName.ATTN_DATA_EXPERT), None,
                                 None)

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

        # Use a local variable (not self.shared_experts) to avoid double-registration
        # in named_parameters(). The checkpoint stores shared_experts weights under
        # "experts.shared_experts.*", not "shared_experts.*". Storing it as both
        # self.shared_experts AND self.experts.shared_experts would make named_parameters()
        # yield two paths for the same params, causing "not initialized" errors since
        # the "shared_experts.*" (direct) path is never loaded from checkpoint.
        _shared_experts = DeepseekV3MLP(
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
            shared_experts=_shared_experts,
            scoring_func=scoring_func,
            routed_scaling_factor=routed_scaling_factor,
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.experts(x)


class Glm4MoeDecoderLayer(JaxModule):

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
        rms_norm_eps = config.rms_norm_eps
        hidden_size = config.hidden_size

        self.input_layernorm = JaxRmsNorm(
            hidden_size,
            epsilon=rms_norm_eps,
            dtype=dtype,
            scale_init=nnx.with_partitioning(init_fn, (None, )),
            rngs=rng,
            quant_config=quant_config,
            prefix=prefix + ".input_layernorm",
        )
        self.self_attn = Glm4MoeAttention(
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

        first_k_dense_replace = getattr(config, "first_k_dense_replace", 1)
        moe_layer_freq = getattr(config, "moe_layer_freq", 1)
        if layer_index < first_k_dense_replace:
            is_moe_layer = False
        else:
            is_moe_layer = (layer_index % moe_layer_freq == 0)

        if not is_moe_layer:
            self.mlp = Glm4MoeMLP(
                hidden_size=hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                dtype=dtype,
                rng=rng,
                quant_config=quant_config,
                prefix=prefix + ".mlp",
            )
        else:
            self.mlp = Glm4MoeMoELayer(
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
        hidden_states = self.input_layernorm(x)
        kv_cache, attn_output = self.self_attn(
            kv_cache,
            hidden_states,
            attention_metadata,
        )
        attn_output = attn_output + x

        residual = attn_output
        attn_output = self.post_attention_layernorm(attn_output)
        outputs = self.mlp(attn_output)
        outputs = residual + outputs
        return kv_cache, outputs


class Glm4MoeModel(JaxModule):

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
            lambda layer_index: Glm4MoeDecoderLayer(
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


class Glm4MoeForCausalLM(JaxModule, LoadableWithIterator):

    def __init__(self, vllm_config: VllmConfig, rng_key: jax.Array,
                 mesh: Mesh) -> None:
        self.vllm_config = vllm_config
        rng = nnx.Rngs(rng_key)
        self.mesh = mesh

        self.model = Glm4MoeModel(
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
                quant_config=vllm_config.quant_config,
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

    def compute_logits(self, hidden_states: jax.Array) -> jax.Array:
        return self.lm_head(hidden_states)

    def _parallel_read_weights(self) -> list[tuple[str, 'torch.Tensor']]:
        """Pre-read all safetensors checkpoint shards in parallel.

        Returns a list of (name, tensor) pairs in file order so that
        AutoWeightsLoader's groupby-based module routing works correctly.
        """
        import time
        from concurrent.futures import ThreadPoolExecutor, as_completed

        from safetensors import safe_open

        from tpu_inference.models.jax.utils.weight_utils import (
            get_model_weights_files)

        model_path = self.vllm_config.model_config.model
        download_dir = self.vllm_config.load_config.download_dir
        weights_files = get_model_weights_files(model_path, download_dir)

        logger.info(
            f"Parallel pre-reading {len(weights_files)} checkpoint shards...")
        t0 = time.perf_counter()

        def _read_single_file(filepath):
            result = []
            with safe_open(filepath, framework="pt") as f:
                for name in f.keys():
                    result.append((name, f.get_tensor(name)))
            return result

        # Each thread reads one file; results are stored per-index to
        # preserve the original file ordering after all threads finish.
        file_results = [None] * len(weights_files)
        max_workers = min(16, len(weights_files))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {
                executor.submit(_read_single_file, f): i
                for i, f in enumerate(weights_files)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                file_results[idx] = future.result()

        all_weights = []
        for file_ws in file_results:
            all_weights.extend(file_ws)

        elapsed = time.perf_counter() - t0
        logger.info(
            f"Pre-read {len(all_weights)} weight tensors from "
            f"{len(weights_files)} files in {elapsed:.1f}s")
        return all_weights

    def load_weights(self, weights: Iterable) -> set[str]:
        if not isinstance(weights, Iterable):
            return super().load_weights(weights)

        # Bypass vLLM's sequential iterator: pre-read all safetensors
        # shards in parallel for much faster checkpoint loading
        # (e.g. 47 shards for GLM-4.5-Air).
        try:
            weights = iter(self._parallel_read_weights())
        except Exception as e:
            logger.warning(
                f"Parallel pre-read failed ({e}), falling back to "
                "sequential loading.")

        num_layers = len(self.model.layers)
        hf_config = self.vllm_config.model_config.hf_config
        total_layers = hf_config.num_hidden_layers
        # num_nextn_predict_layers: extra speculative layers stored in checkpoint
        # as model.layers.{num_hidden_layers} ... must be skipped.
        num_nextn = getattr(hf_config, "num_nextn_predict_layers", 0)
        loader = JaxAutoWeightsLoader(
            self,
            skip_prefixes=(["lm_head"]
                           if not hasattr(self, "lm_head") else []),
            skip_substrs=[
                f"layers.{i}"
                for i in range(num_layers, total_layers + num_nextn)
            ],
        )
        # GLM-4.5-Air checkpoint stores shared expert weights directly under
        # "mlp.shared_experts.*", but the JAX model places them inside
        # SharedFusedMoe as "mlp.experts.shared_experts.*".  Remap here so
        # AutoWeightsLoader routes them to the correct module.
        mapper = WeightsMapper(
            orig_to_new_substr={".mlp.shared_experts.": ".mlp.experts.shared_experts."})
        loaded = loader.load_weights(weights, mapper=mapper)

        if os.environ.get("VLLM_LOGGING_LEVEL", "").upper() == "DEBUG":
            logger.debug("Glm4MoeForCausalLM parameter dtypes:")
            num_layers_to_display = 3
            should_skip = False
            for name, param in self.named_parameters():
                if f"layers.{num_layers_to_display}." in name:
                    should_skip = True
                if should_skip and "layers." in name:
                    continue
                v: jax.Array = param.value
                logger.debug(f"{name} : {v.dtype}{v.shape} on {v.device}")

        return loaded
