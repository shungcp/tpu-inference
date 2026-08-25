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

from dataclasses import dataclass

from tpu_inference.logger import init_logger

logger = init_logger(__name__)


@dataclass(frozen=True)
class TuningKey:
    case: str  # A string identifier for the case, support only: "batched_decode", "decode_only", "mixed"
    max_num_tokens: int  # Maximum number of tokens in the batch
    actual_num_q_heads: int  # Actual number of Q heads, <= num_q_heads in the model config, fixed at 128 for now
    actual_lkv_dim: int  # Actual NOPE head dimension, <= lkv_dim in the model config, fixed at 512 for now
    actual_r_dim: int  # Actual ROPE head dimension, <= r_dim in the model config, fixed at 64 for now
    kv_dtype: str = "float8_e4m3fn"  # KV cache and KV input data type, fixed at fp8 for now
    q_dtype: str = "float8_e4m3fn"  # Q activation dtype, fixed at fp8 for now
    page_size_per_kv_packing: int = 256  # Page size per KV packing, should be aligned with the kernel configuration
    kv_packing: int = 4  # Packing factor for KV, determined by the data type (e.g., 4 for fp8)
    max_num_seqs: int = 160  # Maximum number of sequences in the batch, should be large enough to cover all sequences in the batch
    pages_per_seq: int = 9  # Number of pages per sequence, determined by the maximum KV length and page size. Should be large enough to cover the longest sequence in the batch.

    s_dtype: str = "bfloat16"  # Post QK einsum data type feeding into softmax, fixed at bf16 for now
    soft_cap: float | None = None  # Optional softmax cap, if None, no capping is applied. If set, should be a positive value.
    # sm_scale: float = 0.1352337788608801 # Scaling factor applied to the softmax input
    # mask_value: float | None = -3.38953e+38 # Optional mask value for masked positions

    chunk_prefill_size: int | None = None  # Chunk size for prefill in the decode case, range from 1 to max_num_tokens with steps of powers of two
    sliding_window: int | None = None  # Sliding window size, [None, 5, 128]
    p_same_dtype_as_v: bool = True  # Whether the softmax input should have the same data type as V, fixed at True for now


@dataclass(frozen=True)
class TunableParams:
    num_kv_pages_per_block: int  # Number of KV pages to process per block. Range from 1 to as high as possible before OOM,
    # with steps of powers of two.
    num_queries_per_block: int  # for batched_decode, this is always 1
    vmem_limit_bytes: int  # 16MiB(?) to 64MiB, increments of 8MiB.

    # Select lowest value that gives the highest performance
    decode_batch_size: int = 1  # range from 1 to as high as possible before OOM with steps powers of two
    # Constraint: batch size % decode_batch_size = 0
    q_split: int = 1  # number of query split for running parallel.

    def __ge__(self, other) -> bool:
        if not isinstance(other, TunableParams):
            return NotImplemented

        # Higher decode_batch_size, num_kv_pages_per_block, and num_queries_per_block
        # represent more memory demanding configurations. For vmem_limit_bytes, a larger limit
        # is less OOM-prone, so the comparison is inverted.
        return (self.decode_batch_size >= other.decode_batch_size
                and self.num_kv_pages_per_block >= other.num_kv_pages_per_block
                and self.num_queries_per_block >= other.num_queries_per_block
                and self.vmem_limit_bytes <= other.vmem_limit_bytes)

    def __le__(self, other) -> bool:
        if not isinstance(other, TunableParams):
            return NotImplemented

        return (self.decode_batch_size <= other.decode_batch_size
                and self.num_kv_pages_per_block <= other.num_kv_pages_per_block
                and self.num_queries_per_block <= other.num_queries_per_block
                and self.vmem_limit_bytes >= other.vmem_limit_bytes)


tuned_params_mapping: dict[TuningKey, TunableParams] = {
    # Mistral-Small batched decode. (TPU v7-8)
    TuningKey(
        case="batched_decode",
        max_num_tokens=4,
        actual_num_q_heads=32,
        actual_lkv_dim=256,
        actual_r_dim=64,
        page_size_per_kv_packing=32,
        kv_packing=32,
        max_num_seqs=16,
        pages_per_seq=3,
    ):
    TunableParams(
        decode_batch_size=8,
        num_kv_pages_per_block=3,
        num_queries_per_block=1,
        vmem_limit_bytes=62914560,
    ),
    TuningKey(
        case="batched_decode",
        max_num_tokens=8,
        actual_num_q_heads=32,
        actual_lkv_dim=256,
        actual_r_dim=64,
        page_size_per_kv_packing=32,
        kv_packing=32,
        max_num_seqs=16,
        pages_per_seq=3,
    ):
    TunableParams(
        decode_batch_size=8,
        num_kv_pages_per_block=3,
        num_queries_per_block=1,
        vmem_limit_bytes=62914560,
    ),
    TuningKey(
        case="batched_decode",
        max_num_tokens=16,
        actual_num_q_heads=32,
        actual_lkv_dim=256,
        actual_r_dim=64,
        page_size_per_kv_packing=32,
        kv_packing=32,
        max_num_seqs=16,
        pages_per_seq=3,
    ):
    TunableParams(
        decode_batch_size=8,
        num_kv_pages_per_block=3,
        num_queries_per_block=1,
        vmem_limit_bytes=62914560,
    ),
    TuningKey(
        case="batched_decode",
        max_num_tokens=256,
        actual_num_q_heads=32,
        actual_lkv_dim=256,
        actual_r_dim=64,
        page_size_per_kv_packing=32,
        kv_packing=32,
        max_num_seqs=16,
        pages_per_seq=3,
    ):
    TunableParams(
        decode_batch_size=8,
        num_kv_pages_per_block=3,
        num_queries_per_block=1,
        vmem_limit_bytes=62914560,
    ),
    # DeepSeekV3 batched decode. (TPU v7-8)
    TuningKey(
        case="batched_decode",
        max_num_tokens=4,
        actual_num_q_heads=128,
        actual_lkv_dim=512,
        actual_r_dim=64,
    ):
    TunableParams(
        decode_batch_size=8,
        num_kv_pages_per_block=3,
        num_queries_per_block=1,
        vmem_limit_bytes=62914560,
    ),
    TuningKey(
        case="batched_decode",
        max_num_tokens=8,
        actual_num_q_heads=128,
        actual_lkv_dim=512,
        actual_r_dim=64,
    ):
    TunableParams(
        decode_batch_size=8,
        num_kv_pages_per_block=3,
        num_queries_per_block=1,
        vmem_limit_bytes=62914560,
    ),
    TuningKey(
        case="batched_decode",
        max_num_tokens=16,
        actual_num_q_heads=128,
        actual_lkv_dim=512,
        actual_r_dim=64,
    ):
    TunableParams(
        decode_batch_size=8,
        num_kv_pages_per_block=3,
        num_queries_per_block=1,
        vmem_limit_bytes=62914560,
    ),
    TuningKey(
        case="batched_decode",
        max_num_tokens=32,
        actual_num_q_heads=128,
        actual_lkv_dim=512,
        actual_r_dim=64,
    ):
    TunableParams(
        decode_batch_size=8,
        num_kv_pages_per_block=3,
        num_queries_per_block=1,
        vmem_limit_bytes=62914560,
    ),
    TuningKey(
        case="batched_decode",
        max_num_tokens=64,
        actual_num_q_heads=128,
        actual_lkv_dim=512,
        actual_r_dim=64,
    ):
    TunableParams(
        decode_batch_size=4,
        num_kv_pages_per_block=3,
        num_queries_per_block=1,
        vmem_limit_bytes=62914560,
    ),
    TuningKey(
        case="batched_decode",
        max_num_tokens=128,
        actual_num_q_heads=128,
        actual_lkv_dim=512,
        actual_r_dim=64,
    ):
    TunableParams(
        decode_batch_size=8,
        num_kv_pages_per_block=3,
        num_queries_per_block=1,
        vmem_limit_bytes=62914560,
    ),
    TuningKey(
        case="batched_decode",
        max_num_tokens=160,
        actual_num_q_heads=128,
        actual_lkv_dim=512,
        actual_r_dim=64,
    ):
    TunableParams(
        decode_batch_size=8,
        num_kv_pages_per_block=3,
        num_queries_per_block=1,
        vmem_limit_bytes=62914560,
    ),
    TuningKey(
        case="batched_decode",
        max_num_tokens=256,
        actual_num_q_heads=128,
        actual_lkv_dim=512,
        actual_r_dim=64,
    ):
    TunableParams(
        decode_batch_size=8,
        num_kv_pages_per_block=3,
        num_queries_per_block=1,
        vmem_limit_bytes=62914560,
    ),
    TuningKey(
        case="batched_decode",
        max_num_tokens=512,
        actual_num_q_heads=128,
        actual_lkv_dim=512,
        actual_r_dim=64,
    ):
    TunableParams(
        decode_batch_size=8,
        num_kv_pages_per_block=3,
        num_queries_per_block=1,
        vmem_limit_bytes=62914560,
    ),
    # Tuned parameters for Mistral-Large-3 on TPU v7x-16 (page_size=1024, kv_packing=32, max_num_seqs=8):
    TuningKey(
        case="batched_decode",
        max_num_tokens=4,
        actual_num_q_heads=128,
        actual_lkv_dim=512,
        actual_r_dim=64,
        page_size_per_kv_packing=32,
        kv_packing=32,
        max_num_seqs=8,
        pages_per_seq=3,
    ):
    TunableParams(
        decode_batch_size=8,
        num_kv_pages_per_block=1,
        num_queries_per_block=1,
        vmem_limit_bytes=62914560,
    ),
    TuningKey(
        case="batched_decode",
        max_num_tokens=8,
        actual_num_q_heads=128,
        actual_lkv_dim=512,
        actual_r_dim=64,
        page_size_per_kv_packing=32,
        kv_packing=32,
        max_num_seqs=8,
        pages_per_seq=3,
    ):
    TunableParams(
        decode_batch_size=4,
        num_kv_pages_per_block=3,
        num_queries_per_block=1,
        vmem_limit_bytes=62914560,
    ),
    TuningKey(
        case="batched_decode",
        max_num_tokens=16,
        actual_num_q_heads=128,
        actual_lkv_dim=512,
        actual_r_dim=64,
        page_size_per_kv_packing=32,
        kv_packing=32,
        max_num_seqs=8,
        pages_per_seq=3,
    ):
    TunableParams(
        decode_batch_size=2,
        num_kv_pages_per_block=1,
        num_queries_per_block=1,
        vmem_limit_bytes=62914560,
    ),
    TuningKey(
        case="batched_decode",
        max_num_tokens=32,
        actual_num_q_heads=128,
        actual_lkv_dim=512,
        actual_r_dim=64,
        page_size_per_kv_packing=32,
        kv_packing=32,
        max_num_seqs=8,
        pages_per_seq=3,
    ):
    TunableParams(
        decode_batch_size=4,
        num_kv_pages_per_block=1,
        num_queries_per_block=1,
        vmem_limit_bytes=62914560,
    ),
    TuningKey(
        case="batched_decode",
        max_num_tokens=64,
        actual_num_q_heads=128,
        actual_lkv_dim=512,
        actual_r_dim=64,
        page_size_per_kv_packing=32,
        kv_packing=32,
        max_num_seqs=8,
        pages_per_seq=3,
    ):
    TunableParams(
        decode_batch_size=4,
        num_kv_pages_per_block=1,
        num_queries_per_block=1,
        vmem_limit_bytes=62914560,
    ),
    TuningKey(
        case="batched_decode",
        max_num_tokens=128,
        actual_num_q_heads=128,
        actual_lkv_dim=512,
        actual_r_dim=64,
        page_size_per_kv_packing=32,
        kv_packing=32,
        max_num_seqs=8,
        pages_per_seq=3,
    ):
    TunableParams(
        decode_batch_size=4,
        num_kv_pages_per_block=3,
        num_queries_per_block=1,
        vmem_limit_bytes=62914560,
    ),
    TuningKey(
        case="batched_decode",
        max_num_tokens=160,
        actual_num_q_heads=128,
        actual_lkv_dim=512,
        actual_r_dim=64,
        page_size_per_kv_packing=32,
        kv_packing=32,
        max_num_seqs=8,
        pages_per_seq=3,
    ):
    TunableParams(
        decode_batch_size=8,
        num_kv_pages_per_block=1,
        num_queries_per_block=1,
        vmem_limit_bytes=62914560,
    ),
    TuningKey(
        case="batched_decode",
        max_num_tokens=256,
        actual_num_q_heads=128,
        actual_lkv_dim=512,
        actual_r_dim=64,
        page_size_per_kv_packing=32,
        kv_packing=32,
        max_num_seqs=8,
        pages_per_seq=3,
    ):
    TunableParams(
        decode_batch_size=1,
        num_kv_pages_per_block=3,
        num_queries_per_block=1,
        vmem_limit_bytes=62914560,
    ),
    TuningKey(
        case="batched_decode",
        max_num_tokens=512,
        actual_num_q_heads=128,
        actual_lkv_dim=512,
        actual_r_dim=64,
        page_size_per_kv_packing=32,
        kv_packing=32,
        max_num_seqs=8,
        pages_per_seq=3,
    ):
    TunableParams(
        decode_batch_size=4,
        num_kv_pages_per_block=3,
        num_queries_per_block=1,
        vmem_limit_bytes=62914560,
    ),
    TuningKey(
        case="batched_decode",
        max_num_tokens=1024,
        actual_num_q_heads=128,
        actual_lkv_dim=512,
        actual_r_dim=64,
        page_size_per_kv_packing=32,
        kv_packing=32,
        max_num_seqs=8,
        pages_per_seq=3,
    ):
    TunableParams(
        decode_batch_size=2,
        num_kv_pages_per_block=1,
        num_queries_per_block=1,
        vmem_limit_bytes=62914560,
    ),
    TuningKey(
        case="batched_decode",
        max_num_tokens=2048,
        actual_num_q_heads=128,
        actual_lkv_dim=512,
        actual_r_dim=64,
        page_size_per_kv_packing=32,
        kv_packing=32,
        max_num_seqs=8,
        pages_per_seq=3,
    ):
    TunableParams(
        decode_batch_size=4,
        num_kv_pages_per_block=3,
        num_queries_per_block=1,
        vmem_limit_bytes=62914560,
    ),
    # Kimi 2.6 batched_decode. (TPU v7-8)
    TuningKey(
        case="batched_decode",
        max_num_tokens=4,
        actual_num_q_heads=64,
        actual_lkv_dim=512,
        actual_r_dim=64,
        page_size_per_kv_packing=32,
        kv_packing=32,
        max_num_seqs=64,
        pages_per_seq=9,
    ):
    TunableParams(
        decode_batch_size=4,
        num_kv_pages_per_block=3,
        num_queries_per_block=1,
        vmem_limit_bytes=62914560,
    ),
    TuningKey(
        case="batched_decode",
        max_num_tokens=8,
        actual_num_q_heads=64,
        actual_lkv_dim=512,
        actual_r_dim=64,
        page_size_per_kv_packing=32,
        kv_packing=32,
        max_num_seqs=64,
        pages_per_seq=9,
    ):
    TunableParams(
        decode_batch_size=8,
        num_kv_pages_per_block=3,
        num_queries_per_block=1,
        vmem_limit_bytes=62914560,
    ),
    TuningKey(
        case="batched_decode",
        max_num_tokens=16,
        actual_num_q_heads=64,
        actual_lkv_dim=512,
        actual_r_dim=64,
        page_size_per_kv_packing=32,
        kv_packing=32,
        max_num_seqs=64,
        pages_per_seq=9,
    ):
    TunableParams(
        decode_batch_size=16,
        num_kv_pages_per_block=2,
        num_queries_per_block=1,
        vmem_limit_bytes=62914560,
    ),
    TuningKey(
        case="batched_decode",
        max_num_tokens=32,
        actual_num_q_heads=64,
        actual_lkv_dim=512,
        actual_r_dim=64,
        page_size_per_kv_packing=32,
        kv_packing=32,
        max_num_seqs=64,
        pages_per_seq=9,
    ):
    TunableParams(
        decode_batch_size=8,
        num_kv_pages_per_block=3,
        num_queries_per_block=1,
        vmem_limit_bytes=62914560,
    ),
    TuningKey(
        case="batched_decode",
        max_num_tokens=64,
        actual_num_q_heads=64,
        actual_lkv_dim=512,
        actual_r_dim=64,
        page_size_per_kv_packing=32,
        kv_packing=32,
        max_num_seqs=64,
        pages_per_seq=9,
    ):
    TunableParams(
        decode_batch_size=8,
        num_kv_pages_per_block=3,
        num_queries_per_block=1,
        vmem_limit_bytes=62914560,
    ),
    TuningKey(
        case="batched_decode",
        max_num_tokens=77,
        actual_num_q_heads=64,
        actual_lkv_dim=512,
        actual_r_dim=64,
        page_size_per_kv_packing=32,
        kv_packing=32,
        max_num_seqs=64,
        pages_per_seq=9,
    ):
    TunableParams(
        decode_batch_size=8,
        num_kv_pages_per_block=3,
        num_queries_per_block=1,
        vmem_limit_bytes=62914560,
    ),
    TuningKey(
        case="batched_decode",
        max_num_tokens=128,
        actual_num_q_heads=64,
        actual_lkv_dim=512,
        actual_r_dim=64,
        page_size_per_kv_packing=32,
        kv_packing=32,
        max_num_seqs=64,
        pages_per_seq=9,
    ):
    TunableParams(
        decode_batch_size=8,
        num_kv_pages_per_block=3,
        num_queries_per_block=1,
        vmem_limit_bytes=62914560,
    ),
    TuningKey(
        case="batched_decode",
        max_num_tokens=160,
        actual_num_q_heads=64,
        actual_lkv_dim=512,
        actual_r_dim=64,
        page_size_per_kv_packing=32,
        kv_packing=32,
        max_num_seqs=64,
        pages_per_seq=9,
    ):
    TunableParams(
        decode_batch_size=8,
        num_kv_pages_per_block=3,
        num_queries_per_block=1,
        vmem_limit_bytes=62914560,
    ),
    TuningKey(
        case="batched_decode",
        max_num_tokens=256,
        actual_num_q_heads=64,
        actual_lkv_dim=512,
        actual_r_dim=64,
        page_size_per_kv_packing=32,
        kv_packing=32,
        max_num_seqs=64,
        pages_per_seq=9,
    ):
    TunableParams(
        decode_batch_size=8,
        num_kv_pages_per_block=3,
        num_queries_per_block=1,
        vmem_limit_bytes=62914560,
    ),
    TuningKey(
        case="batched_decode",
        max_num_tokens=512,
        actual_num_q_heads=64,
        actual_lkv_dim=512,
        actual_r_dim=64,
        page_size_per_kv_packing=32,
        kv_packing=32,
        max_num_seqs=64,
        pages_per_seq=9,
    ):
    TunableParams(
        decode_batch_size=8,
        num_kv_pages_per_block=3,
        num_queries_per_block=1,
        vmem_limit_bytes=62914560,
    ),
    TuningKey(
        case="batched_decode",
        max_num_tokens=616,
        actual_num_q_heads=64,
        actual_lkv_dim=512,
        actual_r_dim=64,
        page_size_per_kv_packing=32,
        kv_packing=32,
        max_num_seqs=64,
        pages_per_seq=9,
    ):
    TunableParams(
        decode_batch_size=4,
        num_kv_pages_per_block=3,
        num_queries_per_block=1,
        vmem_limit_bytes=62914560,
    ),
    TuningKey(
        case="batched_decode",
        max_num_tokens=1024,
        actual_num_q_heads=64,
        actual_lkv_dim=512,
        actual_r_dim=64,
        page_size_per_kv_packing=32,
        kv_packing=32,
        max_num_seqs=64,
        pages_per_seq=9,
    ):
    TunableParams(
        decode_batch_size=8,
        num_kv_pages_per_block=3,
        num_queries_per_block=1,
        vmem_limit_bytes=62914560,
    ),
    TuningKey(
        case="batched_decode",
        max_num_tokens=2048,
        actual_num_q_heads=64,
        actual_lkv_dim=512,
        actual_r_dim=64,
        page_size_per_kv_packing=32,
        kv_packing=32,
        max_num_seqs=64,
        pages_per_seq=9,
    ):
    TunableParams(
        decode_batch_size=4,
        num_kv_pages_per_block=3,
        num_queries_per_block=1,
        vmem_limit_bytes=62914560,
    ),
    # Kimi 2.6 prefill/mixed. (TPU v7-8)
    TuningKey(
        case="mixed",
        max_num_tokens=4,
        actual_num_q_heads=64,
        actual_lkv_dim=512,
        actual_r_dim=64,
        page_size_per_kv_packing=32,
        kv_packing=32,
        max_num_seqs=64,
        pages_per_seq=9,
    ):
    TunableParams(
        q_split=16,
        num_kv_pages_per_block=1,
        num_queries_per_block=64,
        vmem_limit_bytes=62914560,
    ),
    TuningKey(
        case="mixed",
        max_num_tokens=8,
        actual_num_q_heads=64,
        actual_lkv_dim=512,
        actual_r_dim=64,
        page_size_per_kv_packing=32,
        kv_packing=32,
        max_num_seqs=64,
        pages_per_seq=9,
    ):
    TunableParams(
        q_split=16,
        num_kv_pages_per_block=1,
        num_queries_per_block=64,
        vmem_limit_bytes=62914560,
    ),
    TuningKey(
        case="mixed",
        max_num_tokens=16,
        actual_num_q_heads=64,
        actual_lkv_dim=512,
        actual_r_dim=64,
        page_size_per_kv_packing=32,
        kv_packing=32,
        max_num_seqs=64,
        pages_per_seq=9,
    ):
    TunableParams(
        q_split=16,
        num_kv_pages_per_block=1,
        num_queries_per_block=64,
        vmem_limit_bytes=62914560,
    ),
    TuningKey(
        case="mixed",
        max_num_tokens=32,
        actual_num_q_heads=64,
        actual_lkv_dim=512,
        actual_r_dim=64,
        page_size_per_kv_packing=32,
        kv_packing=32,
        max_num_seqs=64,
        pages_per_seq=9,
    ):
    TunableParams(
        q_split=16,
        num_kv_pages_per_block=1,
        num_queries_per_block=64,
        vmem_limit_bytes=62914560,
    ),
    TuningKey(
        case="mixed",
        max_num_tokens=64,
        actual_num_q_heads=64,
        actual_lkv_dim=512,
        actual_r_dim=64,
        page_size_per_kv_packing=32,
        kv_packing=32,
        max_num_seqs=64,
        pages_per_seq=9,
    ):
    TunableParams(
        q_split=16,
        num_kv_pages_per_block=1,
        num_queries_per_block=64,
        vmem_limit_bytes=62914560,
    ),
    TuningKey(
        case="mixed",
        max_num_tokens=77,
        actual_num_q_heads=64,
        actual_lkv_dim=512,
        actual_r_dim=64,
        page_size_per_kv_packing=32,
        kv_packing=32,
        max_num_seqs=64,
        pages_per_seq=9,
    ):
    TunableParams(
        q_split=16,
        num_kv_pages_per_block=1,
        num_queries_per_block=64,
        vmem_limit_bytes=62914560,
    ),
    TuningKey(
        case="mixed",
        max_num_tokens=128,
        actual_num_q_heads=64,
        actual_lkv_dim=512,
        actual_r_dim=64,
        page_size_per_kv_packing=32,
        kv_packing=32,
        max_num_seqs=64,
        pages_per_seq=9,
    ):
    TunableParams(
        q_split=16,
        num_kv_pages_per_block=1,
        num_queries_per_block=64,
        vmem_limit_bytes=62914560,
    ),
    TuningKey(
        case="mixed",
        max_num_tokens=160,
        actual_num_q_heads=64,
        actual_lkv_dim=512,
        actual_r_dim=64,
        page_size_per_kv_packing=32,
        kv_packing=32,
        max_num_seqs=64,
        pages_per_seq=9,
    ):
    TunableParams(
        q_split=16,
        num_kv_pages_per_block=1,
        num_queries_per_block=64,
        vmem_limit_bytes=62914560,
    ),
    TuningKey(
        case="mixed",
        max_num_tokens=256,
        actual_num_q_heads=64,
        actual_lkv_dim=512,
        actual_r_dim=64,
        page_size_per_kv_packing=32,
        kv_packing=32,
        max_num_seqs=64,
        pages_per_seq=9,
    ):
    TunableParams(
        q_split=16,
        num_kv_pages_per_block=1,
        num_queries_per_block=64,
        vmem_limit_bytes=62914560,
    ),
    TuningKey(
        case="mixed",
        max_num_tokens=512,
        actual_num_q_heads=64,
        actual_lkv_dim=512,
        actual_r_dim=64,
        page_size_per_kv_packing=32,
        kv_packing=32,
        max_num_seqs=64,
        pages_per_seq=9,
    ):
    TunableParams(
        q_split=16,
        num_kv_pages_per_block=1,
        num_queries_per_block=64,
        vmem_limit_bytes=62914560,
    ),
    TuningKey(
        case="mixed",
        max_num_tokens=1024,
        actual_num_q_heads=64,
        actual_lkv_dim=512,
        actual_r_dim=64,
        page_size_per_kv_packing=32,
        kv_packing=32,
        max_num_seqs=64,
        pages_per_seq=9,
    ):
    TunableParams(
        q_split=16,
        num_kv_pages_per_block=1,
        num_queries_per_block=64,
        vmem_limit_bytes=62914560,
    ),
    # Kimi 2.6 prefill/mixed. (TPU v7-8)
    TuningKey(
        case="mixed",
        max_num_tokens=4,
        actual_num_q_heads=64,
        actual_lkv_dim=512,
        actual_r_dim=64,
        kv_dtype="float8_e4m3fn",
        q_dtype="float8_e4m3fn",
        page_size_per_kv_packing=576,
        kv_packing=1024,
        max_num_seqs=77,
        pages_per_seq=9,
        s_dtype="bfloat16",
        soft_cap=None,
        chunk_prefill_size=None,
        sliding_window=None,
        p_same_dtype_as_v=True,
    ):
    TunableParams(
        q_split=16,
        num_kv_pages_per_block=1,
        num_queries_per_block=64,
        vmem_limit_bytes=62914560,
    ),
    TuningKey(
        case="mixed",
        max_num_tokens=8,
        actual_num_q_heads=64,
        actual_lkv_dim=512,
        actual_r_dim=64,
        kv_dtype="float8_e4m3fn",
        q_dtype="float8_e4m3fn",
        page_size_per_kv_packing=576,
        kv_packing=1024,
        max_num_seqs=77,
        pages_per_seq=9,
        s_dtype="bfloat16",
        soft_cap=None,
        chunk_prefill_size=None,
        sliding_window=None,
        p_same_dtype_as_v=True,
    ):
    TunableParams(
        q_split=16,
        num_kv_pages_per_block=1,
        num_queries_per_block=64,
        vmem_limit_bytes=62914560,
    ),
    TuningKey(
        case="mixed",
        max_num_tokens=16,
        actual_num_q_heads=64,
        actual_lkv_dim=512,
        actual_r_dim=64,
        kv_dtype="float8_e4m3fn",
        q_dtype="float8_e4m3fn",
        page_size_per_kv_packing=576,
        kv_packing=1024,
        max_num_seqs=77,
        pages_per_seq=9,
        s_dtype="bfloat16",
        soft_cap=None,
        chunk_prefill_size=None,
        sliding_window=None,
        p_same_dtype_as_v=True,
    ):
    TunableParams(
        q_split=16,
        num_kv_pages_per_block=1,
        num_queries_per_block=64,
        vmem_limit_bytes=62914560,
    ),
    TuningKey(
        case="mixed",
        max_num_tokens=32,
        actual_num_q_heads=64,
        actual_lkv_dim=512,
        actual_r_dim=64,
        kv_dtype="float8_e4m3fn",
        q_dtype="float8_e4m3fn",
        page_size_per_kv_packing=576,
        kv_packing=1024,
        max_num_seqs=77,
        pages_per_seq=9,
        s_dtype="bfloat16",
        soft_cap=None,
        chunk_prefill_size=None,
        sliding_window=None,
        p_same_dtype_as_v=True,
    ):
    TunableParams(
        q_split=16,
        num_kv_pages_per_block=1,
        num_queries_per_block=64,
        vmem_limit_bytes=62914560,
    ),
    TuningKey(
        case="mixed",
        max_num_tokens=64,
        actual_num_q_heads=64,
        actual_lkv_dim=512,
        actual_r_dim=64,
        kv_dtype="float8_e4m3fn",
        q_dtype="float8_e4m3fn",
        page_size_per_kv_packing=576,
        kv_packing=1024,
        max_num_seqs=77,
        pages_per_seq=9,
        s_dtype="bfloat16",
        soft_cap=None,
        chunk_prefill_size=None,
        sliding_window=None,
        p_same_dtype_as_v=True,
    ):
    TunableParams(
        q_split=16,
        num_kv_pages_per_block=1,
        num_queries_per_block=64,
        vmem_limit_bytes=62914560,
    ),
    TuningKey(
        case="mixed",
        max_num_tokens=77,
        actual_num_q_heads=64,
        actual_lkv_dim=512,
        actual_r_dim=64,
        kv_dtype="float8_e4m3fn",
        q_dtype="float8_e4m3fn",
        page_size_per_kv_packing=576,
        kv_packing=1024,
        max_num_seqs=77,
        pages_per_seq=9,
        s_dtype="bfloat16",
        soft_cap=None,
        chunk_prefill_size=None,
        sliding_window=None,
        p_same_dtype_as_v=True,
    ):
    TunableParams(
        q_split=16,
        num_kv_pages_per_block=1,
        num_queries_per_block=64,
        vmem_limit_bytes=62914560,
    ),
    TuningKey(
        case="mixed",
        max_num_tokens=128,
        actual_num_q_heads=64,
        actual_lkv_dim=512,
        actual_r_dim=64,
        kv_dtype="float8_e4m3fn",
        q_dtype="float8_e4m3fn",
        page_size_per_kv_packing=576,
        kv_packing=1024,
        max_num_seqs=77,
        pages_per_seq=9,
        s_dtype="bfloat16",
        soft_cap=None,
        chunk_prefill_size=None,
        sliding_window=None,
        p_same_dtype_as_v=True,
    ):
    TunableParams(
        q_split=16,
        num_kv_pages_per_block=1,
        num_queries_per_block=64,
        vmem_limit_bytes=62914560,
    ),
    TuningKey(
        case="mixed",
        max_num_tokens=160,
        actual_num_q_heads=64,
        actual_lkv_dim=512,
        actual_r_dim=64,
        kv_dtype="float8_e4m3fn",
        q_dtype="float8_e4m3fn",
        page_size_per_kv_packing=576,
        kv_packing=1024,
        max_num_seqs=77,
        pages_per_seq=9,
        s_dtype="bfloat16",
        soft_cap=None,
        chunk_prefill_size=None,
        sliding_window=None,
        p_same_dtype_as_v=True,
    ):
    TunableParams(
        q_split=16,
        num_kv_pages_per_block=1,
        num_queries_per_block=64,
        vmem_limit_bytes=62914560,
    ),
    TuningKey(
        case="mixed",
        max_num_tokens=256,
        actual_num_q_heads=64,
        actual_lkv_dim=512,
        actual_r_dim=64,
        kv_dtype="float8_e4m3fn",
        q_dtype="float8_e4m3fn",
        page_size_per_kv_packing=576,
        kv_packing=1024,
        max_num_seqs=77,
        pages_per_seq=9,
        s_dtype="bfloat16",
        soft_cap=None,
        chunk_prefill_size=None,
        sliding_window=None,
        p_same_dtype_as_v=True,
    ):
    TunableParams(
        q_split=16,
        num_kv_pages_per_block=1,
        num_queries_per_block=64,
        vmem_limit_bytes=62914560,
    ),
    TuningKey(
        case="mixed",
        max_num_tokens=512,
        actual_num_q_heads=64,
        actual_lkv_dim=512,
        actual_r_dim=64,
        kv_dtype="float8_e4m3fn",
        q_dtype="float8_e4m3fn",
        page_size_per_kv_packing=576,
        kv_packing=1024,
        max_num_seqs=77,
        pages_per_seq=9,
        s_dtype="bfloat16",
        soft_cap=None,
        chunk_prefill_size=None,
        sliding_window=None,
        p_same_dtype_as_v=True,
    ):
    TunableParams(
        q_split=16,
        num_kv_pages_per_block=1,
        num_queries_per_block=64,
        vmem_limit_bytes=62914560,
    ),
    TuningKey(
        case="mixed",
        max_num_tokens=1024,
        actual_num_q_heads=64,
        actual_lkv_dim=512,
        actual_r_dim=64,
        kv_dtype="float8_e4m3fn",
        q_dtype="float8_e4m3fn",
        page_size_per_kv_packing=576,
        kv_packing=1024,
        max_num_seqs=77,
        pages_per_seq=9,
        s_dtype="bfloat16",
        soft_cap=None,
        chunk_prefill_size=None,
        sliding_window=None,
        p_same_dtype_as_v=True,
    ):
    TunableParams(
        q_split=16,
        num_kv_pages_per_block=1,
        num_queries_per_block=64,
        vmem_limit_bytes=62914560,
    ),

    # GLM-4.7-Flash batched decode. (TPU v6e, w8a16/bf16 MLA)
    # Tuned via mla_kernel_tuner.py (Bayesian search, local, run_id=004,
    # notes/perf_loop/round6_mla_tuning/) across all 11 max_num_tokens
    # buckets the kernel is invoked with. num_kv_pages_per_block=2 was the
    # best or a statistical tie with the best (within ~1% of the fastest
    # config found) at every bucket, consistently beating both
    # num_kv_pages_per_block=1 (worse) and the untuned fallback default of 3
    # (never sampled by the search space, which only covers powers of two).
    **{
        TuningKey(
            case="batched_decode",
            max_num_tokens=max_num_tokens,
            actual_num_q_heads=20,
            actual_lkv_dim=512,
            actual_r_dim=64,
            kv_dtype="bfloat16",
            q_dtype="bfloat16",
            page_size_per_kv_packing=32,
            kv_packing=32,
            max_num_seqs=256,
            pages_per_seq=2,
        ): TunableParams(
            num_kv_pages_per_block=2,
            num_queries_per_block=1,
            vmem_limit_bytes=62914560,
            decode_batch_size=4,
        )
        for max_num_tokens in
        (4, 8, 16, 32, 64, 128, 160, 256, 512, 1024, 2048)
    },

    # GLM-4.7-Flash mixed (prefill). (TPU v6e, w8a16/bf16 MLA)
    # Tuned via a direct micro-benchmark sweep of mla_ragged_paged_attention
    # with prefill-shaped inputs (single sequence, q_len=max_num_tokens) --
    # mla_kernel_tuner.py's generate_cases() hardcodes case="batched_decode"
    # and always builds decode-shaped (q_len=1) inputs, so it has no path for
    # this case at all (notes/perf_loop/round7_mla_mixed_tuning/tune_mixed.py).
    # pages_per_seq=2 corresponds to a `--max-model-len 2048` server (e.g. the
    # correctness-check scripts used in earlier rounds). Untuned fallback
    # (get_tuned_params below) is num_kv_pages_per_block=1,
    # num_queries_per_block=16 -- at the realistic prefill sizes this matters
    # most for (max_num_tokens 128..2048), that fallback is 1.16x-2.13x
    # slower than the tuned config below, which was within 0.4% of the
    # fastest config found at every one of those 5 buckets (and within ~5%
    # at the small, decode-adjacent buckets where latency is noise-dominated
    # regardless of params).
    **{
        TuningKey(
            case="mixed",
            max_num_tokens=max_num_tokens,
            actual_num_q_heads=20,
            actual_lkv_dim=512,
            actual_r_dim=64,
            kv_dtype="bfloat16",
            q_dtype="bfloat16",
            page_size_per_kv_packing=32,
            kv_packing=32,
            max_num_seqs=256,
            pages_per_seq=2,
        ): TunableParams(
            num_kv_pages_per_block=2,
            num_queries_per_block=128,
            vmem_limit_bytes=62914560,
            q_split=8,
        )
        for max_num_tokens in
        (4, 8, 16, 32, 64, 128, 160, 256, 512, 1024, 2048)
    },

    # GLM-4.7-Flash mixed (prefill), pages_per_seq=8. (TPU v6e, w8a16/bf16 MLA)
    # IMPORTANT: pages_per_seq=8 (not 2) is what the actual
    # `vllm serve --max-model-len 8192` production/e2e-bench config computes
    # at runtime (confirmed from e2e_bench.log) -- num_kv_pages_per_block=1,
    # num_queries_per_block=128, q_split=8 was never worse than the untuned
    # fallback at any of the 11 buckets (within 1% at the tiny/noise-dominated
    # ones) and was 1.16x/1.44x faster at the realistic 1024/2048-token
    # buckets -- and within 2.5% of the true per-bucket best everywhere.
    **{
        TuningKey(
            case="mixed",
            max_num_tokens=max_num_tokens,
            actual_num_q_heads=20,
            actual_lkv_dim=512,
            actual_r_dim=64,
            kv_dtype="bfloat16",
            q_dtype="bfloat16",
            page_size_per_kv_packing=32,
            kv_packing=32,
            max_num_seqs=256,
            pages_per_seq=8,
        ): TunableParams(
            num_kv_pages_per_block=1,
            num_queries_per_block=128,
            vmem_limit_bytes=62914560,
            q_split=8,
        )
        for max_num_tokens in
        (4, 8, 16, 32, 64, 128, 160, 256, 512, 1024, 2048)
    },
}


def get_tuned_params(tuning_key: TuningKey) -> TunableParams:
    if tuning_key in tuned_params_mapping:
        return tuned_params_mapping[tuning_key]
    else:
        logger.warning(
            f"No tuned parameters found for the given tuning key: {tuning_key}, using default parameters"
        )
        if tuning_key.case == "mixed":
            return TunableParams(
                num_kv_pages_per_block=1,
                num_queries_per_block=16,
                vmem_limit_bytes=62914560,
            )

        # decode, batched_decode
        return TunableParams(
            decode_batch_size=4,
            num_kv_pages_per_block=3,
            num_queries_per_block=1,
            vmem_limit_bytes=62914560,
        )
