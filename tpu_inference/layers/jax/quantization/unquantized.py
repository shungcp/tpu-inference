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

from typing import Optional

import jax
import jax.numpy as jnp
from flax import nnx
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from tpu_inference.layers.common.moe import MoEBackend, moe_apply
from tpu_inference.layers.common.process_weights.moe_weights import (
    FusedMoEWeights, UnfusedMoEWeights)
from tpu_inference.layers.common.quantization import unquantized as jax_common
from tpu_inference.layers.common.quantization.configs import QuantLinearConfig
from tpu_inference.layers.jax import JaxModule
from tpu_inference.layers.jax.linear import JaxEinsum
from tpu_inference.layers.jax.moe.moe import JaxMoE
from tpu_inference.layers.jax.quantization import QuantizeMethodBase
from tpu_inference.layers.jax.quantization.configs import QuantizationConfig
from tpu_inference.layers.common.utils import general_device_put
from tpu_inference.models.jax.utils.weight_utils import shard_put

from jax.experimental.layout import Layout
from jax.sharding import NamedSharding as NS


class UnquantizedLinearMethod(QuantizeMethodBase,
                              jax_common.UnquantizedLinearMethod):
    """Unquantized method for JAX Linear layer.
    """

    def apply_jax(self, layer: JaxModule, x: jax.Array) -> jax.Array:
        assert isinstance(layer, JaxEinsum)

        with jax.named_scope(layer._get_name()):
            if self.linear_config.fuse_matmuls:
                out = self._apply_fused(
                    x,
                    layer.weight.value,
                    layer.bias.value if layer.bias else None,
                    einsum_str=layer.einsum_str)
            else:
                raise NotImplementedError(
                    "Non-fused matmuls not implemented yet.")

        return out


class UnquantizedFusedMoEMethod(QuantizeMethodBase):
    """
    Unquantized method for JAXMoE layer.

    TODO (jacobplatin): support weight loading -- currently, model-dependent.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.extra_backend_kwargs = {}

    def process_weights_after_loading(self, layer: JaxMoE, *args,
                                      **kwargs) -> bool:
        """
        Process weights after loading.

        Please see https://github.com/vllm-project/tpu-inference/blob/bb1a88/tpu_inference/layers/common/moe.py#L39
        for more information on the expected weights per MoE backend.

        Args:
            layer: The layer to process.
        """
        if layer.moe_backend == MoEBackend.FUSED_MOE:
            if layer.edf_sharding:
                e2df_sharding = (layer.edf_sharding[0], None,
                                 layer.edf_sharding[1], layer.edf_sharding[2])
            # fuse the weights into w13: [Gate, Up]
            w_gate = layer.kernel_gating_EDF.value
            w_up = layer.kernel_up_proj_EDF.value

            # stack to create a 4d array
            w13_val = jnp.stack([w_gate, w_up], axis=1)

            layer.kernel_gating_upproj_E2DF = nnx.Param(
                shard_put(w13_val, shardings=e2df_sharding))

            del layer.kernel_gating_EDF
            del layer.kernel_up_proj_EDF

            ep_axis_name = layer.efd_sharding[0]

            self.extra_backend_kwargs = {
                "ep_axis_name": ep_axis_name,
                "bt": 32,
                "bf": 512,
                "bd1": 512,
                "bd2": 512,
                "btc": 64,
                "bfc": 256,
                "bd1c": 256,
                "bd2c": 256,
            }

        elif layer.moe_backend in [MoEBackend.GMM_EP, MoEBackend.GMM_TP]:
            if any(
                    any(w is None for w in param._weights_to_load) for param in
                [layer.kernel_gating_EDF, layer.kernel_up_proj_EDF]):
                return False

            # Each entry in _weights_to_load is (1, F, D); concat gives (E, F, D).
            # Concatenation and transpose on CPU to avoid OOM; shard_put must be
            # OUTSIDE cpu_mesh_context so it uses the TPU mesh.
            from tpu_inference.layers.common.utils import cpu_mesh_context
            with cpu_mesh_context():
                w_gate = jnp.concatenate(
                    layer.kernel_gating_EDF._weights_to_load, axis=0)
                w_up = jnp.concatenate(
                    layer.kernel_up_proj_EDF._weights_to_load, axis=0)
                if layer.moe_backend == MoEBackend.GMM_TP:
                    # Store gate and up as SEPARATE pre-transposed (E, D, F) arrays.
                    # Each is independently TP-sharded on the F axis (last dim) via
                    # edf_sharding = P(None, None, 'tp'), so each chip gets (E, D, F/TP)
                    # of the correct gate or up data.  This avoids:
                    #  - reorder_concatenated_tensor_for_sharding (no fused tensor)
                    #  - HLO copy buffers from slicing inside shard_map (no slicing)
                    #  - fuse_act num_lanes constraint (separate GMMs, no fuse_act)
                    w_gate_t = w_gate.swapaxes(1, 2)  # (E, F, D) → (E, D, F)
                    w_up_t = w_up.swapaxes(1, 2)      # (E, F, D) → (E, D, F)
                else:
                    w13_val = jnp.concatenate([w_gate, w_up], axis=1)  # (E, 2F, D)

            # shard_put outside cpu_mesh_context → uses the TPU mesh
            # Use Layout((0, 1, 2)) to match the GMM kernel's expected layout
            # and avoid XLA layout-conversion copies that cause HLO temp OOM.
            if layer.moe_backend == MoEBackend.GMM_TP:
                layout_3d = Layout((0, 1, 2))
                edf_ns = NS(layer.mesh, P(*layer.edf_sharding))
                # Two-step: shard_put moves CPU→TPU, then general_device_put
                # applies Layout((0,1,2)) to match GMM kernel's expected layout
                # and eliminate XLA layout-conversion copy buffers (HLO temp).
                w_gate_tpu = shard_put(w_gate_t, shardings=layer.edf_sharding)
                w_up_tpu = shard_put(w_up_t, shardings=layer.edf_sharding)
                layer.kernel_gating_EDF = nnx.Param(
                    general_device_put(w_gate_tpu, edf_ns, layout=layout_3d))
                layer.kernel_up_proj_EDF = nnx.Param(
                    general_device_put(w_up_tpu, edf_ns, layout=layout_3d))
                # Also re-put down_proj with correct layout.
                layer.kernel_down_proj_EFD = nnx.Param(
                    general_device_put(layer.kernel_down_proj_EFD.value,
                                       edf_ns, layout=layout_3d))
            else:
                # For EP: store as (E, 2F, D); edf_sharding = P('expert', None, None)
                # shards the E axis.  apply_jax swapaxes(1,2) → (E, D, 2F) = w1, and
                # E-sharding matches ep_p_spec = P(EXPERT) inside the shard_map.
                layer.kernel_gating_upproj_EDF = nnx.Param(
                    shard_put(w13_val, shardings=layer.edf_sharding))
                del layer.kernel_gating_EDF
                del layer.kernel_up_proj_EDF

        return True

    def apply_jax(self, layer: JaxModule, x: jax.Array) -> jax.Array:
        assert isinstance(layer, JaxMoE)

        x_TD = jnp.asarray(x, layer.dtype)
        x_TD = jax.lax.with_sharding_constraint(
            x_TD, NamedSharding(layer.mesh, P(*layer.activation_ffw_td)))

        router_logits = None
        w1_up = None
        # Fused weight backends
        if layer.moe_backend in MoEBackend.fused_moe_backends():
            # of shape TE, only 1D in this case
            router_logits = layer.router(x_TD)

            if layer.moe_backend == MoEBackend.FUSED_MOE:
                w13_weight = jnp.swapaxes(layer.kernel_gating_upproj_E2DF.value, 1, 2)
            elif layer.moe_backend == MoEBackend.GMM_TP:
                # Separate gate/up path: both pre-transposed to (E, D, F) with
                # P(None, None, 'tp') sharding on F.  Passed as separate inputs
                # through shard_map to avoid HLO copy buffers from slicing.
                w13_weight = layer.kernel_gating_EDF.value   # gate (E, D, F)
                w1_up = layer.kernel_up_proj_EDF.value       # up   (E, D, F)
            else:
                # GMM_EP: stored as (E, 2F, D) with E-axis sharding.
                w13_weight = jnp.swapaxes(layer.kernel_gating_upproj_EDF.value, 1, 2)
            w2_weight = layer.kernel_down_proj_EFD.value
            w2_weight = jnp.swapaxes(w2_weight, 1, 2)
            # TODO (jacobplatin/bzgoogle): we should support bias
            weights = FusedMoEWeights(
                w13_weight=w13_weight,
                w13_weight_scale=None,
                w13_bias=None,
                w2_weight=w2_weight,
                w2_weight_scale=None,
                w2_bias=None,
            )
        elif layer.moe_backend in [
                MoEBackend.DENSE_MAT, MoEBackend.MEGABLX_GMM
        ]:
            # Composed of weights_TX and indices_TX, so 2D in this case
            router_logits = layer.router(x_TD)
            # TODO (jacobplatin/bzgoogle): we should support bias
            weights = UnfusedMoEWeights(
                w1_weight=layer.kernel_gating_EDF.value,
                w1_weight_scale=None,
                w1_bias=None,
                w2_weight=layer.kernel_up_proj_EDF.value,
                w2_weight_scale=None,
                w2_bias=None,
                w3_weight=layer.kernel_down_proj_EFD.value,
                w3_weight_scale=None,
                w3_bias=None,
            )

        else:
            raise ValueError(f"Unsupported moe backend {layer.moe_backend}")
        return moe_apply(layer, x_TD, router_logits, weights,
                         layer.moe_backend, layer.mesh,
                         self.extra_backend_kwargs, w1_up=w1_up)


class UnquantizedConfig(QuantizationConfig):

    def get_quant_method(self, layer: JaxModule,
                         prefix: str) -> Optional[QuantizeMethodBase]:
        if isinstance(layer, JaxEinsum):
            # Derive output's last dim from the einsum string.
            einsum_str = layer.einsum_str.replace(" ", "")
            _, w_axis = einsum_str.split("->")[0].split(",")
            last_out_char = einsum_str.split("->")[1][-1]
            out_size = layer.kernel_shape[w_axis.index(last_out_char)]

            linear_config = QuantLinearConfig(enable_sp=False,
                                              output_sizes=[out_size])
            return UnquantizedLinearMethod(linear_config)
        if isinstance(layer, JaxMoE):
            return UnquantizedFusedMoEMethod()
        return None
