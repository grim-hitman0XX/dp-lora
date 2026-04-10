"""Wraps a PEFT model to compute per-sample gradients on LoRA layers."""

from __future__ import annotations

import logging
from typing import Optional

import torch
from torch import nn

from dp_lora.grad_sample.hooks import (
    clear_per_sample_grads,
    linear_backward_hook,
    linear_forward_hook,
)

logger = logging.getLogger(__name__)


class GradSampleModule:
    """Attaches per-sample gradient hooks to LoRA layers in a PEFT model.

    Supports two methods:
      - "vanilla": Both lora_A and lora_B are trainable and hooked.
      - "ffa": lora_A is frozen (FFA-LoRA), only lora_B is hooked.

    After a forward+backward pass, per-sample gradients are available via
    ``get_per_sample_grads()``.
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
        """Find LoRA layers and register forward/backward hooks."""
        lora_layer_count = 0

        for name, module in self.model.named_modules():
            if not hasattr(module, "lora_A") or not hasattr(module, "lora_B"):
                continue

            # nn.ModuleDict supports `in` and `[]` but not `.get()`
            if (
                self.adapter_name not in module.lora_A
                or self.adapter_name not in module.lora_B
            ):
                continue
            lora_a = module.lora_A[self.adapter_name]
            lora_b = module.lora_B[self.adapter_name]

            if self.method == "ffa":
                # FFA-LoRA: freeze A, only hook B
                for param in lora_a.parameters():
                    param.requires_grad = False
                self._register_hooks(lora_b, f"{name}.lora_B.{self.adapter_name}")
            else:
                # Vanilla: hook both A and B
                self._register_hooks(lora_a, f"{name}.lora_A.{self.adapter_name}")
                self._register_hooks(lora_b, f"{name}.lora_B.{self.adapter_name}")

            lora_layer_count += 1

        logger.info(
            "Attached per-sample gradient hooks to %d LoRA layer(s) "
            "(method=%s, adapter=%s)",
            lora_layer_count,
            self.method,
            self.adapter_name,
        )

        if lora_layer_count == 0:
            raise ValueError(
                f"No LoRA layers found for adapter '{self.adapter_name}'. "
                "Make sure the model has been wrapped with PEFT's get_peft_model()."
            )

    def _register_hooks(self, linear_module: nn.Linear, name: str) -> None:
        h_fwd = linear_module.register_forward_hook(linear_forward_hook)
        h_bwd = linear_module.register_full_backward_hook(linear_backward_hook)
        self._hooks.extend([h_fwd, h_bwd])
        self._hooked_modules.append((name, linear_module))

    def get_per_sample_grads(self) -> list[tuple[nn.Parameter, torch.Tensor]]:
        """Collect per-sample gradients from all hooked modules.

        Returns:
            List of (parameter, per_sample_grad) tuples.
            per_sample_grad has shape [batch_size, *param.shape].
        """
        grads = []
        for name, module in self._hooked_modules:
            if hasattr(module, "_dp_per_sample_grad_weight"):
                grads.append((module.weight, module._dp_per_sample_grad_weight))
            if hasattr(module, "_dp_per_sample_grad_bias"):
                grads.append((module.bias, module._dp_per_sample_grad_bias))
        return grads

    def clear_per_sample_grads(self) -> None:
        """Free stored per-sample gradient tensors."""
        for _, module in self._hooked_modules:
            clear_per_sample_grads(module)

    def get_trainable_params(self) -> list[nn.Parameter]:
        """Return the list of LoRA parameters that are trainable (have hooks)."""
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
        """Remove all registered hooks."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        logger.info("Removed all per-sample gradient hooks.")

    def __del__(self):
        self.remove_hooks()
