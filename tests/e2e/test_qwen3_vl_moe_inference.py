# SPDX-License-Identifier: Apache-2.0
#
# End-to-end tests for Qwen3-VL-MoE multi-modal inference on TPU.
# Runs a single VQA prompt against each MoE Qwen3-VL variant and
# compares the output to a known-good reference (greedy, temperature=0).
#
# Baselines generated on TPU v6e-8.
# Update EXPECTED_TEXT after the first successful run for each model.

import difflib
import os
import time
from dataclasses import asdict

import pytest
from vllm import LLM, EngineArgs, SamplingParams
from vllm.assets.image import ImageAsset
from vllm.multimodal.image import convert_image_mode

# ---------------------------------------------------------------------------
# Per-model baselines: (model_id, expected_text, gpu_memory_utilization,
#                       tensor_parallel_size)
# ---------------------------------------------------------------------------
_MOE_MODELS = [
    pytest.param(
        "Qwen/Qwen3-VL-30B-A3B-Instruct",
        # TODO: fill in after first successful run
        "",
        0.9,
        8,
        id="30B-A3B",
    ),
    pytest.param(
        "Qwen/Qwen3-VL-235B-A22B-Instruct",
        # TODO: fill in after first successful run
        "",
        0.95,
        8,
        id="235B-A22B",
    ),
]


@pytest.mark.parametrize("enable_dynamic_image_sizes", [False, True])
@pytest.mark.parametrize(
    "model,expected_text,gpu_memory_utilization,tensor_parallel_size",
    _MOE_MODELS)
def test_qwen3_vl_moe_inference(monkeypatch, model, expected_text,
                                gpu_memory_utilization, tensor_parallel_size,
                                enable_dynamic_image_sizes):
    """
    Runs Qwen3-VL-MoE multi-modal inference and verifies output against a
    known-good baseline generated with greedy decoding on TPU v6e-8.
    """
    os.environ["SKIP_JAX_PRECOMPILE"] = "1"  # Skip warmup to save time.
    os.environ["VLLM_XLA_CHECK_RECOMPILATION"] = "0"
    # os.environ["HF_HUB_OFFLINE"] = "1"  # Uncomment to use local cache only.

    monkeypatch.setenv("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

    temperature = 0.0
    max_tokens = 64
    max_model_len = 4096
    modality = "image"

    print(f"Preparing for Qwen3-VL-MoE inference ({model})...")

    image = convert_image_mode(ImageAsset("cherry_blossom").pil_image, "RGB")
    question = "What is the content of this image?"

    prompt = ("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
              f"<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
              f"{question}<|im_end|>\n"
              "<|im_start|>assistant\n")

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

            if i == 1:
                if not expected_text:
                    # First run: no baseline yet — just print the output.
                    print(
                        "No expected_text set. Copy the generated text above "
                        "into the _MOE_MODELS list and re-run to enable "
                        "regression checking.")
                else:
                    similarity_score = difflib.SequenceMatcher(
                        None, generated_text, expected_text).ratio()
                    print(f"Similarity Score: {similarity_score:.4f}")
                    assert similarity_score >= 0.75, (
                        f"Text similarity too low ({similarity_score:.2f}).\n"
                        f"Expected: {expected_text}\n"
                        f"Actual:   {generated_text}")
    finally:
        del llm
