# R01 — CUDA and MedGemma preflight

R01 checks that one RunPod NVIDIA GPU can run the pinned MedGemma 1.5 4B checkpoint with BF16 and SDPA, and complete a short text-only forward pass. It also probes PyTorch SDPA's boolean and additive mask semantics.

The probe does not validate Kev-style candidate isolation inside MedGemma. That real model integration belongs to R04.

## Fixed target

The script holds the model ID, immutable revision, device, backend, and short prompt as constants in `scripts/preflight_cuda.py`. `configs/training/medgemma_4b.yaml` is a human-readable reference only; the script does not read it. If the checkpoint changes, update both files.

`requirements-train.txt` holds the package pins. The script prints the installed PyTorch, Transformers, and PyTorch CUDA build versions for troubleshooting, but does not enforce every package or Python version at runtime.

## Run on RunPod

1. Start a Pod with one BF16-capable NVIDIA GPU. Around 24 GB VRAM is a practical starting point for this preflight; LoRA training memory must be measured separately.
2. Clone or pull the MedJev commit containing R01 in the Pod terminal.
3. Create an environment and install the CUDA-enabled PyTorch wheel matching the Pod, then install `requirements-train.txt`. No separate CUDA toolkit or `nvcc` is needed for this preflight.
4. Accept the MedGemma access terms and authenticate to Hugging Face on the Pod. Keep the token out of Git and source files.
5. From the repository root, run:

   ```bash
   python scripts/preflight_cuda.py
   ```

The script prints a short terminal summary and exits nonzero on the first required check that fails. It does not write a JSON report. The source code stays in the local Git repository; only the preflight runs on the Pod.

## Checks performed

- CUDA is visible; the script prints GPU name and free/total VRAM.
- The GPU supports BF16.
- PyTorch SDPA returns the expected result for boolean and additive masks.
- The pinned checkpoint loads in BF16 on `cuda:0` and Transformers resolves the text attention backend to SDPA.
- A short text-only forward pass completes with finite logits.
- The script prints peak allocated VRAM and reminds you that the real candidate-isolation mask is not checked here.

## Expected output

```text
R01 CUDA preflight
PyTorch ... | CUDA build ... | Transformers ...
GPU: ... | VRAM free/total: ... | BF16: supported
SDPA boolean/additive mask probe: PASS
Model: PASS | backend: sdpa | revision: ...
Text forward: PASS | tokens: ... | logits: finite
Peak allocated VRAM: ... GiB
Candidate-isolation mask integration: not checked here (R04).
RESULT: PASS
```

Common failures are reported briefly: missing CUDA means check the Pod GPU and CUDA-enabled PyTorch install; unsupported BF16 means choose compatible GPU hardware; gated model access means accept the MedGemma terms and authenticate on that Pod; out-of-memory means select a GPU with more VRAM.

A passing preflight establishes only the runtime, basic SDPA mask semantics, model load, and text forward path. It does not establish candidate isolation, head correctness, training, calibration, performance, or clinical validity.

## References

- [MedGemma 1.5 4B model card](https://huggingface.co/google/medgemma-1.5-4b-it)
- [Transformers attention backend and mask guidance](https://huggingface.co/docs/transformers/attention_interface)
- [PyTorch scaled dot product attention](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)
