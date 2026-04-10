"""Ghost clipping: memory-efficient per-sample gradient clipping.

Instead of materializing full per-sample gradient tensors [B, out, in],
ghost clipping computes per-sample gradient *norms* using only scalar
quantities per sample per layer, then performs a second backward pass
with per-sample-scaled losses to produce the clipped gradient sum directly.

For nn.Linear with y = xW^T + b:
    ||dL/dW_i||^2 = ||grad_out_i||^2 * ||act_i||^2    (exact for 2D)
    ||dL/dW_i||^2 <= (sum_s ||g_s|| * ||a_s||)^2       (upper bound for 3D)

The upper bound (Cauchy-Schwarz) is safe for DP: over-clipping never
under-estimates privacy, it only hurts utility slightly.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
from torch import nn

from dp_lora.grad_sample.hooks import clear_per_sample_grads

logger = logging.getLogger(__name__)


def ghost_forward_hook(module: nn.Linear, input: tuple, output: torch.Tensor) -> None:
    """Save input activations for norm computation (not the full tensor)."""
    # We need per-position norms for the 3D Cauchy-Schwarz bound
    act = input[0].detach()
    if act.dim() == 3:
        # [B, seq, in] -> per-position norms [B, seq]
        module._dp_act_pos_norms = act.norm(2, dim=-1)
    elif act.dim() == 2:
        # [B, in] -> per-sample norms [B]
        module._dp_act_norms_sq = act.square().sum(dim=1)
    else:
        B = act.shape[0]
        act_2d = act.reshape(B, -1, act.shape[-1])
        module._dp_act_pos_norms = act_2d.norm(2, dim=-1)


def ghost_backward_hook(
    module: nn.Linear, grad_input: tuple, grad_output: tuple
) -> None:
    """Compute per-sample gradient norm^2 from norms of activations and grad_output."""
    grad_out = grad_output[0]

    if hasattr(module, "_dp_act_norms_sq"):
        # 2D case: exact formula ||dL/dW_i||^2 = ||g_i||^2 * ||a_i||^2
        go_norms_sq = grad_out.square().sum(dim=1)  # [B]
        module._dp_grad_norm_sq = go_norms_sq * module._dp_act_norms_sq
        del module._dp_act_norms_sq
    elif hasattr(module, "_dp_act_pos_norms"):
        # 3D case: Cauchy-Schwarz upper bound
        # ||sum_s g_s (x) a_s||_F <= (sum_s ||g_s|| * ||a_s||)^2
        if grad_out.dim() == 3:
            go_pos_norms = grad_out.norm(2, dim=-1)  # [B, seq]
        else:
            go_pos_norms = grad_out.norm(2, dim=-1).unsqueeze(1)

        act_pos_norms = module._dp_act_pos_norms  # [B, seq]
        # Upper bound: (sum of products)^2
        module._dp_grad_norm_sq = (go_pos_norms * act_pos_norms).sum(dim=1).square()
        del module._dp_act_pos_norms

    # Bias norm^2 (if applicable)
    if module.bias is not None and module.bias.requires_grad:
        if grad_out.dim() == 3:
            bias_grad = grad_out.sum(dim=1)  # [B, out]
            module._dp_bias_norm_sq = bias_grad.square().sum(dim=1)
        else:
            module._dp_bias_norm_sq = grad_out.square().sum(dim=1)


def clear_ghost_state(module: nn.Module) -> None:
    """Remove stored ghost clipping state from a module."""
    for attr in (
        "_dp_act_norms_sq",
        "_dp_act_pos_norms",
        "_dp_grad_norm_sq",
        "_dp_bias_norm_sq",
    ):
        if hasattr(module, attr):
            delattr(module, attr)


class GhostClippingModule:
    """Memory-efficient per-sample gradient clipping for LoRA layers.

    Uses two backward passes:
      Pass 1: Compute per-sample gradient norms (cheap, no materialization)
      Pass 2: Re-run backward with per-sample scaled losses to produce
              the clipped gradient sum directly in param.grad

    The GhostClippingModule replaces GradSampleModule when ghost_clipping=True.
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        method: str = "ffa",
        adapter_name: str = "default",
    ):
        self.model = model
        self.method = method
        self.adapter_name = adapter_name
        self._hooks: list[torch.utils.hooks.RemovableHook] = []
        self._hooked_modules: list[tuple[str, nn.Linear]] = []

        self._attach_hooks()

    def _attach_hooks(self) -> None:
        """Find LoRA layers and register ghost clipping hooks."""
        lora_layer_count = 0

        for name, module in self.model.named_modules():
            if not hasattr(module, "lora_A") or not hasattr(module, "lora_B"):
                continue
            if (
                self.adapter_name not in module.lora_A
                or self.adapter_name not in module.lora_B
            ):
                continue

            lora_a = module.lora_A[self.adapter_name]
            lora_b = module.lora_B[self.adapter_name]

            if self.method == "ffa":
                for param in lora_a.parameters():
                    param.requires_grad = False
                self._register_hooks(lora_b, f"{name}.lora_B.{self.adapter_name}")
            else:
                self._register_hooks(lora_a, f"{name}.lora_A.{self.adapter_name}")
                self._register_hooks(lora_b, f"{name}.lora_B.{self.adapter_name}")

            lora_layer_count += 1

        logger.info(
            "Ghost clipping: attached hooks to %d LoRA layer(s) (method=%s)",
            lora_layer_count,
            self.method,
        )
        if lora_layer_count == 0:
            raise ValueError(f"No LoRA layers found for adapter '{self.adapter_name}'.")

    def _register_hooks(self, linear_module: nn.Linear, name: str) -> None:
        h_fwd = linear_module.register_forward_hook(ghost_forward_hook)
        h_bwd = linear_module.register_full_backward_hook(ghost_backward_hook)
        self._hooks.extend([h_fwd, h_bwd])
        self._hooked_modules.append((name, linear_module))

    def get_per_sample_norm_sq(self) -> torch.Tensor:
        """After forward+backward, return per-sample global gradient norm².

        Returns:
            Tensor of shape [B] with the (upper bound of) squared gradient
            norm for each sample in the batch.
        """
        norm_sq = None
        for _, module in self._hooked_modules:
            if hasattr(module, "_dp_grad_norm_sq"):
                if norm_sq is None:
                    norm_sq = module._dp_grad_norm_sq.clone()
                else:
                    norm_sq += module._dp_grad_norm_sq
            if hasattr(module, "_dp_bias_norm_sq"):
                if norm_sq is None:
                    norm_sq = module._dp_bias_norm_sq.clone()
                else:
                    norm_sq += module._dp_bias_norm_sq
        return norm_sq

    def compute_clip_factors(self, max_grad_norm: float) -> torch.Tensor:
        """Compute per-sample clipping factors from stored norms.

        Returns:
            Tensor of shape [B] with clip_factor_i = min(1, C / ||g_i||).
        """
        norm_sq = self.get_per_sample_norm_sq()
        per_sample_norms = norm_sq.sqrt()
        return (max_grad_norm / (per_sample_norms + 1e-8)).clamp(max=1.0)

    def clear_state(self) -> None:
        """Free stored norm tensors."""
        for _, module in self._hooked_modules:
            clear_ghost_state(module)

    def get_trainable_params(self) -> list[nn.Parameter]:
        """Return the list of LoRA parameters that are trainable."""
        params = []
        for _, module in self._hooked_modules:
            params.append(module.weight)
            if module.bias is not None and module.bias.requires_grad:
                params.append(module.bias)
        return params

    @property
    def num_trainable_params(self) -> int:
        return sum(p.numel() for p in self.get_trainable_params())

    def remove_hooks(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def __del__(self):
        self.remove_hooks()
