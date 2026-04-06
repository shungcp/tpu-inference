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
"""JAX (flax_nnx) implementation of ChatGLM4 / GLM-Z1 (ChatGLMForCausalLM).

Covers both GLM-4 (e.g. THUDM/glm-4-9b) and GLM-5/GLM-Z1
(e.g. THUDM/GLM-Z1-32B-0414), which all share the same HuggingFace
architecture string "ChatGLMModel" and ChatGLMConfig.

Key architectural differences from Llama/Qwen:
  - kv_channels:            head dimension (instead of hidden_size // num_heads)
  - multi_query_attention:  GQA flag; num KV heads = multi_query_group_num
  - add_qkv_bias:           Q/K/V projections have biases
  - add_bias_linear:        dense (o_proj) projection may have a bias
  - rope_ratio:             scales rope_theta (effective = rope_theta * rope_ratio)
  - ffn_hidden_size:        MLP intermediate size
  - layernorm_epsilon:      RMSNorm epsilon
  - num_layers:             number of transformer layers (not num_hidden_layers)

Checkpoint weight naming differs from standard HF models:
  - transformer.embedding.word_embeddings.weight  (embedding)
  - transformer.encoder.layers.{i}.self_attention.query_key_value.{weight,bias}
    (fused QKV, split at load time)
  - transformer.encoder.layers.{i}.self_attention.dense.weight  (o_proj)
  - transformer.encoder.layers.{i}.mlp.dense_h_to_4h.weight
    (fused gate+up, split at load time)
  - transformer.encoder.layers.{i}.mlp.dense_4h_to_h.weight    (down_proj)
  - transformer.encoder.final_layernorm.weight
  - transformer.output_layer.weight                             (lm_head)
"""

import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import islice
from typing import Iterable, List, Optional, Tuple

import jax
import jax.numpy as jnp
import torch
from flax import nnx
from jax.sharding import Mesh
from vllm.config import VllmConfig

from tpu_inference import utils
from tpu_inference.distributed.jax_parallel_state import get_pp_group
from tpu_inference.layers.common.attention_interface import attention
from tpu_inference.layers.common.attention_metadata import AttentionMetadata
from tpu_inference.layers.common.quantization import quantize_kv
from tpu_inference.layers.common.utils import cpu_mesh_context
from tpu_inference.layers.jax import JaxModule
from tpu_inference.layers.jax.embed import JaxEmbed
from tpu_inference.layers.jax.linear import JaxEinsum, JaxLinear
from tpu_inference.layers.jax.norm import JaxRmsNorm
from tpu_inference.layers.jax.pp_utils import PPMissingLayer, make_layers
from tpu_inference.layers.jax.rope_interface import apply_rope
from tpu_inference.layers.vllm.quantization.configs import VllmQuantConfig
from tpu_inference.logger import init_logger
from tpu_inference.models.jax.jax_intermediate_tensor import \
    JaxIntermediateTensors
from tpu_inference.models.jax.utils.weight_utils import (
    LoadableWithIterator, assign_and_shard_param, get_model_weights_files,
    model_weights_single_file_generator)
from tpu_inference.utils import t2j

logger = init_logger(__name__)

init_fn = nnx.initializers.uniform()


class ChatGLMMLP(JaxModule):

    def __init__(self,
                 config,
                 dtype: jnp.dtype,
                 rng: nnx.Rngs,
                 quant_config: VllmQuantConfig,
                 prefix: str = ""):
        hidden_size = config.hidden_size
        ffn_hidden_size = config.ffn_hidden_size

        self.gate_proj = JaxLinear(
            hidden_size,
            ffn_hidden_size,
            use_bias=False,
            dtype=dtype,
            kernel_init=nnx.with_partitioning(init_fn, (None, "model")),
            rngs=rng,
            quant_config=quant_config,
            prefix=prefix + ".gate_proj",
        )
        self.up_proj = JaxLinear(
            hidden_size,
            ffn_hidden_size,
            use_bias=False,
            dtype=dtype,
            kernel_init=nnx.with_partitioning(init_fn, (None, "model")),
            rngs=rng,
            quant_config=quant_config,
            prefix=prefix + ".up_proj",
        )
        self.down_proj = JaxLinear(
            ffn_hidden_size,
            hidden_size,
            use_bias=False,
            dtype=dtype,
            kernel_init=nnx.with_partitioning(init_fn, ("model", None)),
            rngs=rng,
            quant_config=quant_config,
            prefix=prefix + ".down_proj",
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        gate = jax.nn.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)


class ChatGLMAttention(JaxModule):

    def __init__(self,
                 config,
                 dtype: jnp.dtype,
                 rng: nnx.Rngs,
                 mesh: Mesh,
                 kv_cache_dtype: str,
                 quant_config: VllmQuantConfig,
                 prefix: str = ""):
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = (config.multi_query_group_num if getattr(
            config, "multi_query_attention", False) else
                             config.num_attention_heads)
        self.head_dim_original = config.kv_channels
        self.head_dim = utils.get_padded_head_dim(self.head_dim_original)

        # GLM-4 applies RoPE only to the first half of each head's features.
        # The saved inv_freq uses theta=base (no rope_ratio) and dim=kv_channels//2.
        # rope_ratio is used for NTK-aware dynamic scaling, not the base theta.
        self.rope_theta = getattr(config, "rope_theta", 10000.0)
        self.rotary_dim = self.head_dim_original // 2  # = 64 for kv_channels=128
        self.rope_scaling = getattr(config, "rope_scaling", None)

        sharding_size = mesh.shape["model"]
        self.num_heads = utils.get_padded_num_heads(self.num_heads,
                                                    sharding_size)
        self.num_kv_heads = utils.get_padded_num_heads(self.num_kv_heads,
                                                       sharding_size)
        self.mesh = mesh

        add_qkv_bias = getattr(config, "add_qkv_bias", False)
        add_bias_linear = getattr(config, "add_bias_linear", False)

        self.q_proj = JaxEinsum(
            "TD,DNH->TNH",
            (self.hidden_size, self.num_heads, self.head_dim),
            bias_shape=(self.num_heads,
                        self.head_dim) if add_qkv_bias else None,
            dtype=dtype,
            kernel_init=nnx.with_partitioning(init_fn, (None, "model", None)),
            bias_init=nnx.with_partitioning(init_fn,
                                            ("model",
                                             None)) if add_qkv_bias else None,
            rngs=rng,
            quant_config=quant_config,
            prefix=prefix + ".q_proj",
        )
        self.k_proj = JaxEinsum(
            "TD,DKH->TKH",
            (self.hidden_size, self.num_kv_heads, self.head_dim),
            bias_shape=(self.num_kv_heads,
                        self.head_dim) if add_qkv_bias else None,
            dtype=dtype,
            kernel_init=nnx.with_partitioning(init_fn, (None, "model", None)),
            bias_init=nnx.with_partitioning(init_fn,
                                            ("model",
                                             None)) if add_qkv_bias else None,
            rngs=rng,
            quant_config=quant_config,
            prefix=prefix + ".k_proj",
        )
        self.v_proj = JaxEinsum(
            "TD,DKH->TKH",
            (self.hidden_size, self.num_kv_heads, self.head_dim),
            bias_shape=(self.num_kv_heads,
                        self.head_dim) if add_qkv_bias else None,
            dtype=dtype,
            kernel_init=nnx.with_partitioning(init_fn, (None, "model", None)),
            bias_init=nnx.with_partitioning(init_fn,
                                            ("model",
                                             None)) if add_qkv_bias else None,
            rngs=rng,
            quant_config=quant_config,
            prefix=prefix + ".v_proj",
        )
        self.o_proj = JaxEinsum(
            "TNH,NHD->TD",
            (self.num_heads, self.head_dim, self.hidden_size),
            bias_shape=(self.hidden_size, ) if add_bias_linear else None,
            dtype=dtype,
            kernel_init=nnx.with_partitioning(init_fn,
                                              ("model", None, None)),
            bias_init=nnx.with_partitioning(
                init_fn, (None, )) if add_bias_linear else None,
            rngs=rng,
            quant_config=quant_config,
            prefix=prefix + ".o_proj",
        )

        self._q_scale = 1.0
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
        k = self.k_proj(x)
        v = self.v_proj(x)

        # GLM-4: RoPE applies to first rotary_dim features; remaining are NoPE.
        rdim = self.rotary_dim
        q = jnp.concatenate([
            apply_rope(q[..., :rdim], md.input_positions, rdim,
                       self.rope_theta, self.rope_scaling),
            q[..., rdim:],
        ], axis=-1)
        k = jnp.concatenate([
            apply_rope(k[..., :rdim], md.input_positions, rdim,
                       self.rope_theta, self.rope_scaling),
            k[..., rdim:],
        ], axis=-1)

        q_scale = k_scale = v_scale = None
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
            q_scale=q_scale,
            k_scale=k_scale,
            v_scale=v_scale,
        )
        o = self.o_proj(outputs)
        return new_kv_cache, o


class ChatGLMDecoderLayer(JaxModule):

    def __init__(self,
                 config,
                 dtype: jnp.dtype,
                 rng: nnx.Rngs,
                 mesh: Mesh,
                 kv_cache_dtype: str,
                 quant_config: VllmQuantConfig,
                 prefix: str = ""):
        hidden_size = config.hidden_size
        eps = config.layernorm_epsilon

        self.input_layernorm = JaxRmsNorm(
            hidden_size,
            epsilon=eps,
            dtype=dtype,
            scale_init=nnx.with_partitioning(init_fn, (None, )),
            rngs=rng,
            quant_config=quant_config,
            prefix=prefix + ".input_layernorm",
        )
        self.self_attn = ChatGLMAttention(
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
            epsilon=eps,
            dtype=dtype,
            scale_init=nnx.with_partitioning(init_fn, (None, )),
            rngs=rng,
            quant_config=quant_config,
            prefix=prefix + ".post_attention_layernorm",
        )
        self.mlp = ChatGLMMLP(
            config=config,
            dtype=dtype,
            rng=rng,
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
        kv_cache, attn_output = self.self_attn(kv_cache, hidden_states,
                                               attention_metadata)
        attn_output = attn_output + x

        residual = attn_output
        attn_output = self.post_attention_layernorm(attn_output)
        outputs = self.mlp(attn_output)
        return kv_cache, residual + outputs


class ChatGLMModel(JaxModule):

    def __init__(self,
                 vllm_config: VllmConfig,
                 rng: nnx.Rngs,
                 mesh: Mesh,
                 prefix: str = "transformer") -> None:
        model_config = vllm_config.model_config
        hf_config = model_config.hf_config
        vocab_size = model_config.get_vocab_size()
        dtype = model_config.dtype

        self.is_first_rank = get_pp_group().is_first_rank
        self.is_last_rank = get_pp_group().is_last_rank

        if self.is_first_rank:
            self.embed_tokens = JaxEmbed(
                num_embeddings=vocab_size,
                features=hf_config.hidden_size,
                dtype=dtype,
                embedding_init=nnx.with_partitioning(init_fn, ("model", None)),
                rngs=rng,
                quant_config=vllm_config.quant_config,
                prefix=prefix + ".embedding.word_embeddings",
            )
        else:
            self.embed_tokens = PPMissingLayer()

        self.start_layer, self.end_layer, self.layers = make_layers(
            hf_config.num_layers,
            lambda layer_index: ChatGLMDecoderLayer(
                config=hf_config,
                dtype=dtype,
                rng=rng,
                mesh=mesh,
                kv_cache_dtype=vllm_config.cache_config.cache_dtype,
                quant_config=vllm_config.quant_config,
                prefix=f"{prefix}.encoder.layers.{layer_index}",
            ),
        )

        if self.is_last_rank:
            self.norm = JaxRmsNorm(
                hf_config.hidden_size,
                epsilon=hf_config.layernorm_epsilon,
                dtype=dtype,
                scale_init=nnx.with_partitioning(init_fn, (None, )),
                rngs=rng,
                quant_config=vllm_config.quant_config,
                prefix=prefix + ".encoder.final_layernorm",
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


class ChatGLMForCausalLM(JaxModule, LoadableWithIterator):
    """ChatGLM4 / GLM-Z1 causal LM.

    Registered under the HF architecture string "ChatGLMModel".
    """

    def __init__(self, vllm_config: VllmConfig, rng_key: jax.Array,
                 mesh: Mesh) -> None:
        self.vllm_config = vllm_config
        rng = nnx.Rngs(rng_key)
        self.mesh = mesh

        self.model = ChatGLMModel(
            vllm_config=vllm_config,
            rng=rng,
            mesh=mesh,
            prefix="transformer",
        )

        model_config = vllm_config.model_config
        vocab_size = model_config.get_vocab_size()
        hidden_size = model_config.hf_config.hidden_size

        self.lm_head = JaxEinsum(
            einsum_str="TD,DV->TV",
            kernel_shape=(hidden_size, vocab_size),
            dtype=model_config.dtype,
            rngs=rng,
            quant_config=vllm_config.quant_config,
            prefix="transformer.output_layer",
        )

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
        kv_caches, x = self.model(kv_caches, input_ids, attention_metadata,
                                  inputs_embeds)
        if not is_last_rank:
            x = JaxIntermediateTensors(tensors={"hidden_states": x})
        return kv_caches, x, []

    def compute_logits(self, hidden_states: jax.Array) -> jax.Array:
        return self.lm_head(hidden_states)

    def load_weights(
            self,
            weights: Iterable[Tuple[str, torch.Tensor]]) -> set[str]:
        """Custom weight loader handling GLM4/Z1 fused QKV and gate+up weights.

        The checkpoint stores:
          - query_key_value: fused [q, k, v] → split into q_proj, k_proj, v_proj
          - dense_h_to_4h:   fused [gate, up] → split into gate_proj, up_proj

        Safetensors shards are loaded in parallel (one thread per file),
        matching the pattern used by load_hf_weights / qwen3_vl_moe.
        set.add() is GIL-protected so `loaded` tracking is thread-safe.
        """
        hf_config = self.vllm_config.model_config.hf_config
        num_heads = hf_config.num_attention_heads
        num_kv_heads = (hf_config.multi_query_group_num if getattr(
            hf_config, "multi_query_attention", False) else num_heads)
        head_dim = hf_config.kv_channels
        head_dim_padded = utils.get_padded_head_dim(head_dim)
        head_dim_pad = head_dim_padded - head_dim
        hidden_size = hf_config.hidden_size
        ffn_hidden_size = hf_config.ffn_hidden_size

        sharding_size = self.mesh.shape["model"]
        num_heads_padded = utils.get_padded_num_heads(num_heads, sharding_size)
        num_kv_heads_padded = utils.get_padded_num_heads(
            num_kv_heads, sharding_size)

        model_dtype = self.vllm_config.model_config.dtype

        # Build param lookup: path → nnx.Param
        params_by_name = {
            name: param
            for name, param in self.named_parameters()
        }

        loaded: set[str] = set()  # JAX param names, not HF keys

        weights_files = get_model_weights_files(
            self.vllm_config.model_config.model,
            self.vllm_config.load_config.download_dir)

        max_workers = min(64, len(weights_files))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(
                    self._load_single_weights_file,
                    f, params_by_name, loaded,
                    num_heads, num_kv_heads,
                    head_dim, head_dim_pad,
                    hidden_size, ffn_hidden_size,
                    num_heads_padded, num_kv_heads_padded,
                    model_dtype, self.mesh,
                )
                for f in weights_files
            ]
            for future in as_completed(futures):
                future.result()

        return loaded

    def _load_single_weights_file(
            self,
            weights_file: str,
            params_by_name: dict,
            loaded: set,
            num_heads: int,
            num_kv_heads: int,
            head_dim: int,
            head_dim_pad: int,
            hidden_size: int,
            ffn_hidden_size: int,
            num_heads_padded: int,
            num_kv_heads_padded: int,
            model_dtype,
            mesh) -> None:
        """Process one safetensors shard (called from ThreadPoolExecutor).

        Each shard covers a disjoint set of layer parameters, so concurrent
        calls writing to different VariableState objects are data-race-free.
        mesh must be passed explicitly because JAX mesh context is thread-local
        and worker threads do not inherit the main thread's active mesh.
        """
        def to_jax(t: torch.Tensor) -> jax.Array:
            with cpu_mesh_context():
                return t2j(t.contiguous(), use_dlpack=False).astype(model_dtype)

        def load_param(param_name: str, jax_w: jax.Array) -> None:
            if param_name not in params_by_name:
                logger.warning(f"MISSING param: {param_name}")
                return
            assign_and_shard_param(params_by_name[param_name], jax_w,
                                   param_name, mesh=mesh)
            loaded.add(param_name)

        for name, weight in model_weights_single_file_generator(
                weights_file, framework="pt"):

            # ── Embedding ────────────────────────────────────────────────────
            if name == "transformer.embedding.word_embeddings.weight":
                # shape [V, D] — kept as-is for JaxEmbed
                load_param("model.embed_tokens.weight", to_jax(weight))

            # ── Pre-computed RoPE frequencies — skip, we compute from config ─
            elif name == "transformer.rotary_pos_emb.inv_freq":
                pass

            # ── Final layernorm ───────────────────────────────────────────────
            elif name == "transformer.encoder.final_layernorm.weight":
                load_param("model.norm.weight", to_jax(weight))

            # ── LM head ──────────────────────────────────────────────────────
            elif name == "transformer.output_layer.weight":
                # [V, D] → transpose → [D, V]
                load_param("lm_head.weight", to_jax(weight.T))

            # ── Per-layer weights ─────────────────────────────────────────────
            elif m := re.match(
                    r"transformer\.encoder\.layers\.(\d+)\.(.*)", name):
                i = int(m.group(1))
                suffix = m.group(2)

                if suffix == "input_layernorm.weight":
                    load_param(
                        f"model.layers.{i}.input_layernorm.weight",
                        to_jax(weight),
                    )

                elif suffix == "post_attention_layernorm.weight":
                    load_param(
                        f"model.layers.{i}.post_attention_layernorm.weight",
                        to_jax(weight),
                    )

                elif suffix == "self_attention.query_key_value.weight":
                    # Fused [q, k, v]: [(N + 2K) * H, D]
                    q_size = num_heads * head_dim
                    kv_size = num_kv_heads * head_dim
                    q_w = weight[:q_size]
                    k_w = weight[q_size:q_size + kv_size]
                    v_w = weight[q_size + kv_size:]

                    for proj_w, proj_name, n_orig, n_pad in [
                        (q_w, "q_proj", num_heads, num_heads_padded),
                        (k_w, "k_proj", num_kv_heads, num_kv_heads_padded),
                        (v_w, "v_proj", num_kv_heads, num_kv_heads_padded),
                    ]:
                        # [n*H, D] → reshape [n, H, D] → permute [D, n, H]
                        jw = to_jax(
                            proj_w.reshape(n_orig, head_dim,
                                           hidden_size).permute(2, 0, 1))
                        with cpu_mesh_context():
                            if head_dim_pad > 0:
                                jw = jnp.pad(jw, ((0, 0), (0, 0),
                                                  (0, head_dim_pad)))
                            if n_pad > n_orig:
                                jw = jnp.repeat(jw, n_pad // n_orig, axis=1)
                        load_param(
                            f"model.layers.{i}.self_attn.{proj_name}.weight",
                            jw)

                elif suffix == "self_attention.query_key_value.bias":
                    # Fused bias: [(N + 2K) * H]
                    q_size = num_heads * head_dim
                    kv_size = num_kv_heads * head_dim
                    q_b = weight[:q_size]
                    k_b = weight[q_size:q_size + kv_size]
                    v_b = weight[q_size + kv_size:]

                    for proj_b, proj_name, n_orig, n_pad in [
                        (q_b, "q_proj", num_heads, num_heads_padded),
                        (k_b, "k_proj", num_kv_heads, num_kv_heads_padded),
                        (v_b, "v_proj", num_kv_heads, num_kv_heads_padded),
                    ]:
                        # [n*H] → reshape [n, H]
                        jw = to_jax(proj_b.reshape(n_orig, head_dim))
                        with cpu_mesh_context():
                            if head_dim_pad > 0:
                                jw = jnp.pad(jw, ((0, 0), (0, head_dim_pad)))
                            if n_pad > n_orig:
                                jw = jnp.repeat(jw, n_pad // n_orig, axis=0)
                        load_param(
                            f"model.layers.{i}.self_attn.{proj_name}.bias",
                            jw)

                elif suffix == "self_attention.dense.weight":
                    # o_proj: [D, N*H] → reshape [D, N, H] → permute [N, H, D]
                    jw = to_jax(
                        weight.reshape(hidden_size, num_heads,
                                       head_dim).permute(1, 2, 0))
                    with cpu_mesh_context():
                        if head_dim_pad > 0:
                            jw = jnp.pad(jw, ((0, 0), (0, head_dim_pad), (0, 0)))
                        if num_heads_padded > num_heads:
                            jw = jnp.repeat(jw, num_heads_padded // num_heads,
                                            axis=0)
                    load_param(f"model.layers.{i}.self_attn.o_proj.weight",
                               jw)

                elif suffix == "self_attention.dense.bias":
                    load_param(f"model.layers.{i}.self_attn.o_proj.bias",
                               to_jax(weight))

                elif suffix == "mlp.dense_h_to_4h.weight":
                    # Fused [gate, up]: [2*F, D] → split → [F, D] → transpose → [D, F]
                    gate_w = weight[:ffn_hidden_size]
                    up_w = weight[ffn_hidden_size:]
                    load_param(f"model.layers.{i}.mlp.gate_proj.weight",
                               to_jax(gate_w.T))
                    load_param(f"model.layers.{i}.mlp.up_proj.weight",
                               to_jax(up_w.T))

                elif suffix == "mlp.dense_4h_to_h.weight":
                    # down_proj: [D, F] → transpose → [F, D]
                    load_param(f"model.layers.{i}.mlp.down_proj.weight",
                               to_jax(weight.T))

                else:
                    logger.warning(f"Unhandled layer weight: {name}")

            else:
                logger.warning(f"Unhandled weight: {name}")
