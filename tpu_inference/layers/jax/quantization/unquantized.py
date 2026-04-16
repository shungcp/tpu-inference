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
from tpu_inference.logger import init_logger
from tpu_inference.models.jax.utils.weight_utils import shard_put

_logger = init_logger(__name__)

from jax.experimental.layout import Layout
from jax.sharding import NamedSharding as NS

# Deferred layout operations. Layout application triggers
# jax.device_put(distributed_array, Format(layout, sharding)) which compiles
# XLA programs on the distributed mesh.  During async multi-host weight
# loading, hosts reach process_weights_after_loading at different times,
# so these programs execute in different order → GSPMD "launch group
# mismatch" → SLICE_FAILURE_SW_INJECT_ERROR.
# Instead, we collect layout functions here and apply them after all hosts
# sync via sync_global_devices in model_loader.py.
_deferred_layout_fns: list = []


def apply_deferred_layouts():
    """Apply all deferred layout operations. Must be called after
    sync_global_devices to ensure all hosts execute in the same order."""
    if _deferred_layout_fns:
        _logger.info("Applying %d deferred layout operations", len(_deferred_layout_fns))
    for fn in _deferred_layout_fns:
        fn()
    _deferred_layout_fns.clear()


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
        _logger.info("process_weights_after_loading: %s backend=%s", layer.prefix, layer.moe_backend)
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
            # Process gate/up eagerly when ready, independent of down_proj.
            # Gate and up are processed SEQUENTIALLY to limit peak CPU memory:
            #   each has 256 × (1,2048,6144) × 2B ≈ 6.4GB in _weights_to_load.
            #   Sequential + eager cleanup: peak ~19GB vs ~38GB simultaneous.
            import gc
            from tpu_inference.layers.common.utils import cpu_mesh_context

            gate_has_wtl = (hasattr(layer, 'kernel_gating_EDF') and
                           '_weights_to_load' in layer.kernel_gating_EDF.get_metadata())
            up_has_wtl = (hasattr(layer, 'kernel_up_proj_EDF') and
                         '_weights_to_load' in layer.kernel_up_proj_EDF.get_metadata())

            if gate_has_wtl and up_has_wtl:
                if any(
                        any(w is None for w in param._weights_to_load) for param in
                    [layer.kernel_gating_EDF, layer.kernel_up_proj_EDF]):
                    return False

                if layer.moe_backend == MoEBackend.GMM_TP:
                    # --- process gate ---
                    _logger.info("%s: concatenating gate weights on CPU", layer.prefix)
                    with cpu_mesh_context():
                        w_gate = jnp.concatenate(
                            layer.kernel_gating_EDF._weights_to_load, axis=0)
                    # Free _weights_to_load entries immediately
                    for i in range(len(layer.kernel_gating_EDF._weights_to_load)):
                        layer.kernel_gating_EDF._weights_to_load[i] = None
                    with cpu_mesh_context():
                        w_gate_t = w_gate.swapaxes(1, 2)  # (E, F, D) → (E, D, F)
                    del w_gate
                    layer.kernel_gating_EDF = nnx.Param(
                        shard_put(w_gate_t, shardings=layer.edf_sharding))
                    del w_gate_t
                    gc.collect()
                    jax.clear_caches()

                    _logger.info("%s: gate shard_put done, concatenating up weights on CPU", layer.prefix)
                    # --- process up ---
                    with cpu_mesh_context():
                        w_up = jnp.concatenate(
                            layer.kernel_up_proj_EDF._weights_to_load, axis=0)
                    for i in range(len(layer.kernel_up_proj_EDF._weights_to_load)):
                        layer.kernel_up_proj_EDF._weights_to_load[i] = None
                    with cpu_mesh_context():
                        w_up_t = w_up.swapaxes(1, 2)  # (E, F, D) → (E, D, F)
                    del w_up
                    layer.kernel_up_proj_EDF = nnx.Param(
                        shard_put(w_up_t, shardings=layer.edf_sharding))
                    del w_up_t
                    gc.collect()
                    jax.clear_caches()

                else:
                    # GMM_EP: fuse gate+up into (E, 2F, D)
                    with cpu_mesh_context():
                        w_gate = jnp.concatenate(
                            layer.kernel_gating_EDF._weights_to_load, axis=0)
                    for i in range(len(layer.kernel_gating_EDF._weights_to_load)):
                        layer.kernel_gating_EDF._weights_to_load[i] = None
                    gc.collect()
                    with cpu_mesh_context():
                        w_up = jnp.concatenate(
                            layer.kernel_up_proj_EDF._weights_to_load, axis=0)
                    for i in range(len(layer.kernel_up_proj_EDF._weights_to_load)):
                        layer.kernel_up_proj_EDF._weights_to_load[i] = None
                    with cpu_mesh_context():
                        w13_val = jnp.concatenate([w_gate, w_up], axis=1)
                    del w_gate, w_up
                    layer.kernel_gating_upproj_EDF = nnx.Param(
                        shard_put(w13_val, shardings=layer.edf_sharding))
                    del w13_val
                    del layer.kernel_gating_EDF
                    del layer.kernel_up_proj_EDF
                    gc.collect()
                    jax.clear_caches()

            # Wait for down_proj to be assigned by _load_weights via
            # assign_and_shard_param.  When expert weights are spread across
            # multiple safetensor files, gate/up may be fully populated (and
            # processed above) before down_proj is concatenated and assigned.
            if isinstance(layer.kernel_down_proj_EFD.value,
                          jax.ShapeDtypeStruct):
                return False

            # Also free down_proj _weights_to_load (6.4GB) if still held.
            if '_weights_to_load' in layer.kernel_down_proj_EFD.get_metadata():
                for i in range(len(layer.kernel_down_proj_EFD._weights_to_load)):
                    layer.kernel_down_proj_EFD._weights_to_load[i] = None
                gc.collect()

            # Use Layout((0, 1, 2)) to match the GMM kernel's expected layout
            # and avoid XLA layout-conversion copies that cause HLO temp OOM.
            # DEFERRED: layout application triggers jax.device_put on distributed
            # arrays, compiling XLA programs.  Collect it here; model_loader.py
            # calls apply_deferred_layouts() after sync_global_devices.
            if layer.moe_backend == MoEBackend.GMM_TP:
                def _apply_layout(layer=layer):
                    layout_3d = Layout((0, 1, 2))
                    edf_ns = NS(layer.mesh, P(*layer.edf_sharding))
                    _logger.info("%s: applying deferred layout to gate/up/down on TPU", layer.prefix)
                    layer.kernel_gating_EDF = nnx.Param(
                        general_device_put(layer.kernel_gating_EDF.value, edf_ns, layout=layout_3d))
                    layer.kernel_up_proj_EDF = nnx.Param(
                        general_device_put(layer.kernel_up_proj_EDF.value, edf_ns, layout=layout_3d))
                    layer.kernel_down_proj_EFD = nnx.Param(
                        general_device_put(layer.kernel_down_proj_EFD.value,
                                           edf_ns, layout=layout_3d))
                    _logger.info("%s: deferred layout done", layer.prefix)
                _deferred_layout_fns.append(_apply_layout)

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
