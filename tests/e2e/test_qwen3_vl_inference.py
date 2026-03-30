# SPDX-License-Identifier: Apache-2.0
#
# End-to-end test for Qwen3-VL multi-modal inference on TPU.
# Runs a single VQA prompt against Qwen/Qwen3-VL-2B-Instruct and
# compares the output to a known-good reference.

import difflib
import os
import time
from dataclasses import asdict

import pytest
from vllm import LLM, EngineArgs, SamplingParams
from vllm.assets.image import ImageAsset
from vllm.multimodal.image import convert_image_mode

# Expected partial text from Qwen3-VL-2B-Instruct on the cherry-blossom image.
# Baseline generated with greedy decoding (temperature=0) on TPU v6e-8.
EXPECTED_TEXT = (
    "This image captures a beautiful spring scene in Japan, featuring the "
    "iconic Skytree Tower in Tokyo, Japan, surrounded by a profusion of "
    "cherry blossoms in full bloom. The photograph is taken from a low "
    "angle, looking up through the branches of cherry trees, which are in "
    "full pink bloom, creating a soft, dream"
)


@pytest.mark.parametrize("enable_dynamic_image_sizes", [False, True])
def test_qwen3_vl_inference(monkeypatch, enable_dynamic_image_sizes):
    """
    Runs Qwen3-VL-2B-Instruct multi-modal inference and verifies output.
    """
    os.environ["SKIP_JAX_PRECOMPILE"] = "1"  # Skip warmup to save time.
    os.environ["VLLM_XLA_CHECK_RECOMPILATION"] = "0"
    # os.environ["HF_HUB_OFFLINE"] = "1"  # Uncomment to use local cache only.

    monkeypatch.setenv("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

    # --- Configuration ---
    model = "Qwen/Qwen3-VL-2B-Instruct"
    tensor_parallel_size = 1
    temperature = 0.0
    max_tokens = 64
    max_model_len = 4096
    gpu_memory_utilization = 0.5
    modality = "image"

    print("Preparing for Qwen3-VL multi-modal inference...")

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
                None, generated_text, EXPECTED_TEXT).ratio()
            print(f"Similarity Score: {similarity_score:.4f}")
            assert similarity_score >= 0.75, (
                f"Text similarity too low ({similarity_score:.2f}).\n"
                f"Expected: {EXPECTED_TEXT}\n"
                f"Actual:   {generated_text}")
