"""Forward and backward hooks for computing per-sample gradients on nn.Linear layers."""

import torch
from torch import nn


def linear_forward_hook(module: nn.Linear, input: tuple, output: torch.Tensor) -> None:
    """Save input activations during forward pass.

    Registered on LoRA's lora_A and lora_B nn.Linear modules.
    The saved activations are used in the backward hook to compute
    per-sample gradients via outer product.
    """
    module._dp_activations = input[0].detach()


def linear_backward_hook(
    module: nn.Linear, grad_input: tuple, grad_output: tuple
) -> None:
    """Compute per-sample gradients from saved activations and grad_output.

    For a Linear layer y = xW^T + b:
        per-sample grad w.r.t. W = grad_output^T @ activation   (per sample)

    Handles both 2D inputs [B, in] and 3D inputs [B, seq, in].
    """
    activations = module._dp_activations  # [B, (seq,) in_features]
    grad_out = grad_output[0]  # [B, (seq,) out_features]

    if activations.dim() == 3:
        # [B, seq, out] x [B, seq, in] -> [B, out, in] (sum over seq)
        per_sample_grad = torch.einsum("bso,bsi->boi", grad_out, activations)
    elif activations.dim() == 2:
        # [B, out] x [B, in] -> [B, out, in]
        per_sample_grad = torch.einsum("bo,bi->boi", grad_out, activations)
    else:
        # Flatten intermediate dims: [B, d1, d2, ..., in] -> [B, N, in]
        shape = activations.shape
        B = shape[0]
        activations_2d = activations.reshape(B, -1, shape[-1])
        grad_out_2d = grad_out.reshape(B, -1, grad_out.shape[-1])
        per_sample_grad = torch.einsum("bso,bsi->boi", grad_out_2d, activations_2d)

    module._dp_per_sample_grad_weight = per_sample_grad

    # Bias gradient if bias exists and requires grad
    if module.bias is not None and module.bias.requires_grad:
        if grad_out.dim() == 3:
            module._dp_per_sample_grad_bias = grad_out.sum(dim=1)  # [B, out]
        else:
            module._dp_per_sample_grad_bias = grad_out  # [B, out]

    del module._dp_activations


def clear_per_sample_grads(module: nn.Module) -> None:
    """Remove stored per-sample gradient tensors from a module."""
    if hasattr(module, "_dp_per_sample_grad_weight"):
        del module._dp_per_sample_grad_weight
    if hasattr(module, "_dp_per_sample_grad_bias"):
        del module._dp_per_sample_grad_bias
    if hasattr(module, "_dp_activations"):
        del module._dp_activations
