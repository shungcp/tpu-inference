# SPDX-License-Identifier: Apache-2.0
#
# End-to-end tests for Qwen3-VL multi-modal inference on TPU.
# Runs a single VQA prompt against each dense Qwen3-VL variant and
# compares the output to a known-good reference (greedy, temperature=0).
#
# Baselines: 2B/4B generated on TPU v6e-8; 32B baseline updated for v7x-8
# (minor formatting diff vs v6e-8 due to fp precision in tp=4 all-reduce).

import difflib
import os
import time
from dataclasses import asdict

import pytest
from vllm import LLM, EngineArgs, SamplingParams
from vllm.assets.image import ImageAsset
from vllm.multimodal.image import convert_image_mode

# ---------------------------------------------------------------------------
# Per-model baselines: (model_id, expected_text, gpu_memory_utilization)
# ---------------------------------------------------------------------------
_DENSE_MODELS = [
    pytest.param(
        "Qwen/Qwen3-VL-2B-Instruct",
        (
            "This image captures a beautiful spring scene in Japan, featuring the "
            "iconic Skytree Tower in Tokyo, Japan, surrounded by a profusion of "
            "cherry blossoms in full bloom. The photograph is taken from a low "
            "angle, looking up through the branches of cherry trees, which are in "
            "full pink bloom, creating a soft, dream"
        ),
        0.5,
        1,
        id="dense-2B",
    ),
    pytest.param(
        "Qwen/Qwen3-VL-4B-Instruct",
        (
            "This image captures a beautiful spring scene in Japan, featuring the "
            "**Tokyo Skytree** framed by the delicate pink blossoms of cherry "
            "trees (sakura).\n\n"
            "- **Foreground**: The image is dominated by out-of-focus pink cherry "
            "blossoms and dark tree branches, creating a soft, dreamy frame."
        ),
        0.5,
        1,
        id="dense-4B",
    ),
    pytest.param(
        "Qwen/Qwen3-VL-8B-Instruct",
        (
            "This image captures a beautiful and iconic scene: **the Tokyo "
            "Skytree, framed by blooming cherry blossoms (sakura) against a "
            "clear blue sky.**\n\n"
            "Here's a breakdown of the content:\n\n"
            "*   **Foreground:** The most prominent feature is the vibrant "
            "pink cherry blossoms. They are in full bloom"
        ),
        0.6,
        1,
        id="dense-8B",
    ),
    pytest.param(
        "Qwen/Qwen3-VL-32B-Instruct",
        (
            "The image features a beautiful springtime scene with **pink cherry "
            "blossoms** in full bloom in the foreground, framing a tall, iconic "
            "tower in the background against a **bright blue sky**.\n\n"
            "The tower is the **Tokyo Skytree**, a prominent landmark in Japan, "
            "recognizable by its distinctive lattice structure and observation deck"
        ),
        0.9,
        4,
        id="dense-32B",
    ),
]


@pytest.mark.parametrize("enable_dynamic_image_sizes", [False, True])
@pytest.mark.parametrize("model,expected_text,gpu_memory_utilization,tensor_parallel_size",
                         _DENSE_MODELS)
def test_qwen3_vl_inference(monkeypatch, model, expected_text,
                            gpu_memory_utilization, tensor_parallel_size,
                            enable_dynamic_image_sizes):
    """
    Runs Qwen3-VL multi-modal inference and verifies output against a
    known-good baseline generated with greedy decoding on TPU v6e-8.
    """
    os.environ["SKIP_JAX_PRECOMPILE"] = "1"  # Skip warmup to save time.
    os.environ["VLLM_XLA_CHECK_RECOMPILATION"] = "0"
    # os.environ["HF_HUB_OFFLINE"] = "1"  # Uncomment to use local cache only.

    monkeypatch.setenv("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

    # --- Configuration ---
    temperature = 0.0
    max_tokens = 64
    max_model_len = 4096
    modality = "image"

    print(f"Preparing for Qwen3-VL inference ({model})...")

    # --- Prepare Inputs ---
    image = convert_image_mode(ImageAsset("cherry_blossom").pil_image, "RGB")
    question = "What is the content of this image?"

    # Qwen3-VL uses the same chat template as Qwen2.5-VL
    prompt = ("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
              f"<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
              f"{question}<|im_end|>\n"
              "<|im_start|>assistant\n")

    # --- Setup vLLM Engine ---
    engine_args = EngineArgs(
        model=model,
        max_model_len=max_model_len,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        max_num_seqs=1,
        mm_processor_kwargs={
            "size": {
                "longest_edge": 1003520,
                "shortest_edge": 3136,
            },
            "fps": 1,
        },
        limit_mm_per_prompt={modality: 1},
    )
    engine_args = asdict(engine_args)
    if engine_args.get("additional_config") is None:
        engine_args["additional_config"] = {}

    engine_args["additional_config"][
        "enable_dynamic_image_sizes"] = enable_dynamic_image_sizes
    engine_args["compilation_config"]["cudagraph_capture_sizes"] = []

    llm = LLM(**engine_args)

    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
    )

    inputs = {
        "prompt": prompt,
        "multi_modal_data": {
            "image": image,
        },
    }

    # --- Run Inference twice (Run 2 shows XLA compilation speedup) ---
    try:
        for i in range(1, 3):
            print(f"Running inference (run {i}/2)...")
            t0 = time.perf_counter()
            outputs = llm.generate(inputs, sampling_params)
            elapsed = time.perf_counter() - t0

            generated_text = outputs[0].outputs[0].text.strip()

            print("-" * 50)
            print(f"Run {i} elapsed: {elapsed:.2f}s")
            print(f"Generated Text: {generated_text}")
            print("-" * 50)

            # Verify output quality only on the first run.
            if i == 1:
                similarity_score = difflib.SequenceMatcher(
                    None, generated_text, expected_text).ratio()
                print(f"Similarity Score: {similarity_score:.4f}")
                assert similarity_score >= 0.75, (
                    f"Text similarity too low ({similarity_score:.2f}).\n"
                    f"Expected: {expected_text}\n"
                    f"Actual:   {generated_text}")
    finally:
        # Always destroy the LLM to release the TPU before the next test,
        # even if an assertion fails.
        del llm
