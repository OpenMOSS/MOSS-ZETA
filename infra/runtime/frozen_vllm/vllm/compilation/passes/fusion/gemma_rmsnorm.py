# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch


def should_enable_patterns(
    quantization: str | None,
    use_fp32_epilogue: bool,
) -> bool:
    """Gate Gemma fusion on quantized checkpoints until explicitly validated."""
    return quantization is None or use_fp32_epilogue


def kernel_gamma_and_epilogue(
    weight: torch.Tensor,
    input_dtype: torch.dtype,
    use_fp32_epilogue: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Prepare a safe gamma for FlashInfer's Gemma RMSNorm fusion.

    The TRTLLM all-reduce fusion kernel reads ``rms_gamma`` as the same type
    as the activation input. Passing the FP32 ``weight + 1`` tensor directly
    would therefore reinterpret its memory as BF16/FP16. In the opt-in
    precision-preserving mode, give the kernel a correctly typed all-ones
    gamma and apply the exact FP32 effective gamma after the kernel returns.
    The default path keeps the historical activation-dtype cast and has no
    extra epilogue.
    """
    effective_gamma = weight.float() + 1.0
    if not use_fp32_epilogue:
        return effective_gamma.to(input_dtype), None

    kernel_gamma = torch.ones_like(weight, dtype=input_dtype)
    return kernel_gamma, effective_gamma


def apply_gamma_epilogue(
    normalized: torch.Tensor,
    effective_gamma: torch.Tensor | None,
) -> torch.Tensor:
    """Apply Gemma's FP32 effective gamma to a fused normalized output."""
    if effective_gamma is None:
        return normalized
    return (normalized.float() * effective_gamma).to(normalized.dtype)
