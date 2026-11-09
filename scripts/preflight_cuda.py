#!/usr/bin/env python3
"""Check that pinned MedGemma runs on this CUDA GPU with SDPA."""

from __future__ import annotations

import sys

MODEL_ID = "google/medgemma-1.5-4b-it"
MODEL_REVISION = "91850547d9f0b2fdd21aa7c5f4f3d1a8a52c243b"
DEVICE = "cuda:0"
ATTENTION_BACKEND = "sdpa"
PROMPT = "Reply with the word READY."


def fail(message: str) -> None:
    print(f"FAIL: {message}", file=sys.stderr)
    raise SystemExit(1)


def check_sdpa_masks(torch, device) -> None:
    import torch.nn.functional as F

    query = torch.zeros((1, 1, 1, 1), device=device)
    key = torch.zeros((1, 1, 2, 1), device=device)
    value = torch.tensor([[[[3.0], [99.0]]]], device=device)
    boolean_mask = torch.tensor([[[[True, False]]]], device=device)
    additive_mask = torch.tensor([[[[0.0, float("-inf")]]]], device=device)

    try:
        boolean_result = F.scaled_dot_product_attention(
            query, key, value, attn_mask=boolean_mask, is_causal=False
        )
        additive_result = F.scaled_dot_product_attention(
            query, key, value, attn_mask=additive_mask, is_causal=False
        )
    except Exception as exc:
        fail(f"SDPA mask probe failed ({type(exc).__name__}).")

    if not torch.allclose(boolean_result, torch.full_like(boolean_result, 3.0)):
        fail("SDPA boolean mask did not select the allowed value.")
    if not torch.allclose(additive_result, torch.full_like(additive_result, 3.0)):
        fail("SDPA additive mask did not select the allowed value.")


def main() -> None:
    try:
        import torch
        import transformers
        from transformers import AutoModelForMultimodalLM, AutoProcessor
    except ImportError as exc:
        fail(f"Training dependencies are missing ({type(exc).__name__}); install requirements-train.txt.")

    print("R01 CUDA preflight")
    print(
        f"PyTorch {torch.__version__} | CUDA build {torch.version.cuda} | "
        f"Transformers {transformers.__version__}"
    )

    if not torch.cuda.is_available():
        fail("CUDA is unavailable. Start a Pod with an NVIDIA GPU and CUDA-enabled PyTorch.")

    device = torch.device(DEVICE)
    gpu = torch.cuda.get_device_properties(device)
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    bf16_supported = torch.cuda.is_bf16_supported()
    print(
        f"GPU: {gpu.name} | VRAM free/total: {free_bytes / 1024**3:.1f}/"
        f"{total_bytes / 1024**3:.1f} GiB | BF16: "
        f"{'supported' if bf16_supported else 'unsupported'}"
    )
    if not bf16_supported:
        fail("This GPU does not support BF16; choose a BF16-capable Pod.")

    check_sdpa_masks(torch, device)
    print("SDPA boolean/additive mask probe: PASS")

    try:
        processor = AutoProcessor.from_pretrained(
            MODEL_ID, revision=MODEL_REVISION
        )
        model = AutoModelForMultimodalLM.from_pretrained(
            MODEL_ID,
            revision=MODEL_REVISION,
            dtype=torch.bfloat16,
            device_map={"": 0},
            attn_implementation=ATTENTION_BACKEND,
            low_cpu_mem_usage=True,
        )
    except Exception as exc:
        message = str(exc).lower()
        if "gated" in message or "401" in message or "403" in message:
            fail("MedGemma access denied. Accept its terms and log in to Hugging Face on this Pod.")
        if "out of memory" in message or "cuda out of memory" in message:
            fail("MedGemma did not fit in GPU memory. Choose a GPU with more VRAM.")
        fail(f"MedGemma load failed ({type(exc).__name__}); check Hub access and the pinned environment.")

    parameter = next(model.parameters())
    if parameter.device != device or parameter.dtype != torch.bfloat16:
        fail("MedGemma did not load in BF16 on cuda:0.")

    text_config = getattr(model.config, "text_config", model.config)
    selected_backend = getattr(text_config, "_attn_implementation", None)
    if selected_backend is None:
        selected_backend = getattr(model.config, "_attn_implementation", None)
    if selected_backend != ATTENTION_BACKEND:
        fail(f"Expected SDPA; Transformers selected {selected_backend!r}.")

    model.eval()
    messages = [
        {
            "role": "user",
            "content": [{"type": "text", "text": PROMPT}],
        }
    ]
    try:
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(device)
        torch.cuda.reset_peak_memory_stats(device)
        with torch.inference_mode():
            outputs = model(**inputs, use_cache=False, return_dict=True)
        logits = outputs.logits
        if not torch.isfinite(logits).all().item():
            fail("Text forward returned non-finite logits.")
    except SystemExit:
        raise
    except Exception as exc:
        fail(f"MedGemma text forward failed ({type(exc).__name__}).")

    torch.cuda.synchronize(device)
    peak_gib = torch.cuda.max_memory_allocated(device) / 1024**3
    print(f"Model: PASS | backend: {selected_backend} | revision: {MODEL_REVISION[:8]}")
    print(f"Text forward: PASS | tokens: {inputs['input_ids'].shape[-1]} | logits: finite")
    print(f"Peak allocated VRAM: {peak_gib:.1f} GiB")
    print("Candidate-isolation mask integration: not checked here (R04).")
    print("RESULT: PASS")


if __name__ == "__main__":
    main()
