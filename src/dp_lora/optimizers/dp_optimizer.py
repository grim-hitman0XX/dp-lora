"""Differentially private optimizer: per-sample clipping + Gaussian noise."""

from __future__ import annotations

from typing import Callable, Optional

import torch
from torch import nn
from torch.optim import Optimizer


def _generate_noise(
    std: float,
    reference: torch.Tensor,
    generator: Optional[torch.Generator] = None,
    secure_mode: bool = False,
) -> torch.Tensor:
    """Generate Gaussian noise with mean 0 and given standard deviation.

    Args:
        std: Standard deviation of the noise.
        reference: Reference tensor for shape and device.
        generator: Optional PyTorch random number generator.
        secure_mode: If True, generate noise resistant to floating point attacks
                     (see https://arxiv.org/abs/2107.10138).
    """
    if std == 0:
        return torch.zeros(reference.shape, device=reference.device)

    if secure_mode:
        torch.normal(
            mean=0,
            std=std,
            size=(1, 1),
            device=reference.device,
            generator=generator,
        )
        total = torch.zeros(reference.shape, device=reference.device)
        for _ in range(4):
            total += torch.normal(
                mean=0,
                std=std,
                size=reference.shape,
                device=reference.device,
                generator=generator,
            )
        return total / 2
    else:
        return torch.normal(
            mean=0,
            std=std,
            size=reference.shape,
            device=reference.device,
            generator=generator,
        )


class DPOptimizer:
    """Wraps a PyTorch optimizer with DP-SGD: per-sample clipping and noise.

    Supports virtual batching: when ``signal_skip_step(do_skip=True)`` is called
    before ``step()``, the optimizer clips and accumulates gradients but defers
    noise addition and the actual parameter update until a non-skipped step.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        *,
        noise_multiplier: float,
        max_grad_norm: float,
        expected_batch_size: int,
        generator: Optional[torch.Generator] = None,
        secure_mode: bool = False,
    ):
        self.optimizer = optimizer
        self.noise_multiplier = noise_multiplier
        self.max_grad_norm = max_grad_norm
        self.expected_batch_size = expected_batch_size
        self.generator = generator
        self.secure_mode = secure_mode
        self._step_hooks: list[Callable[[], None]] = []

        # Virtual batching state
        self._step_skip_queue: list[bool] = []
        self._is_last_step_skipped: bool = False
        self._accumulated_iterations: int = 0

        # Summed clipped gradients, keyed by id(param).
        # Stored here (not on param objects) to avoid issues with
        # serialization, model copying, or FSDP.
        self._summed_grads: dict[int, torch.Tensor] = {}

    # ------------------------------------------------------------------
    # Virtual batching signaling
    # ------------------------------------------------------------------

    def signal_skip_step(self, do_skip: bool = True) -> None:
        """Signal whether the next step() should skip noise+update."""
        self._step_skip_queue.append(do_skip)

    def _check_skip_next_step(self) -> bool:
        if self._step_skip_queue:
            return self._step_skip_queue.pop(0)
        return False

    # ------------------------------------------------------------------
    # Core DP operations
    # ------------------------------------------------------------------

    def clip_and_accumulate(
        self, per_sample_grads: list[tuple[nn.Parameter, torch.Tensor]]
    ) -> dict[str, float]:
        """Clip per-sample gradients and accumulate into internal summed_grads.

        Args:
            per_sample_grads: List of (parameter, per_sample_grad_tensor) pairs.
                Each per_sample_grad_tensor has shape [B, *param.shape].

        Returns:
            Dict with 'mean_clip_factor' and 'num_clipped' for diagnostics.
        """
        if not per_sample_grads:
            return {"mean_clip_factor": 1.0, "num_clipped": 0}

        batch_size = per_sample_grads[0][1].shape[0]
        device = per_sample_grads[0][1].device

        # 1. Compute per-sample global gradient norm
        per_sample_norms_sq = torch.zeros(batch_size, device=device)
        for _, psg in per_sample_grads:
            flat = psg.reshape(batch_size, -1)
            per_sample_norms_sq += flat.norm(2, dim=1).square()
        per_sample_norms = per_sample_norms_sq.sqrt()

        # 2. Compute clipping factors: min(1, C / ||g_i||)
        clip_factors = (self.max_grad_norm / (per_sample_norms + 1e-8)).clamp(max=1.0)

        # 3. Clip and accumulate into _summed_grads
        for param, psg in per_sample_grads:
            cf = clip_factors.reshape(-1, *([1] * (psg.dim() - 1)))
            clipped_sum = (psg * cf).sum(dim=0)

            pid = id(param)
            if pid in self._summed_grads:
                self._summed_grads[pid] = self._summed_grads[pid] + clipped_sum
            else:
                self._summed_grads[pid] = clipped_sum

        num_clipped = int((clip_factors < 1.0).sum().item())
        return {
            "mean_clip_factor": clip_factors.mean().item(),
            "num_clipped": num_clipped,
        }

    def add_noise_and_finalize(self) -> None:
        """Add noise to accumulated clipped grads, scale, write to param.grad."""
        for group in self.optimizer.param_groups:
            for param in group["params"]:
                if not param.requires_grad:
                    continue
                pid = id(param)
                summed = self._summed_grads.get(pid)
                if summed is None:
                    continue
                noise = _generate_noise(
                    std=self.noise_multiplier * self.max_grad_norm,
                    reference=summed,
                    generator=self.generator,
                    secure_mode=self.secure_mode,
                )
                param.grad = (summed + noise).view_as(param) / self.expected_batch_size

    # ------------------------------------------------------------------
    # Step / zero_grad
    # ------------------------------------------------------------------

    def step(
        self,
        per_sample_grads: Optional[list[tuple[nn.Parameter, torch.Tensor]]] = None,
    ) -> dict[str, float]:
        """Full DP-SGD step with virtual batching support.

        1. Clip per-sample grads and accumulate.
        2. If skip signaled: return early (no noise, no update).
        3. Otherwise: add noise, finalize grad, step, fire hooks.
        """
        stats = {"mean_clip_factor": 1.0, "num_clipped": 0}
        if per_sample_grads is not None:
            stats = self.clip_and_accumulate(per_sample_grads)

        self._accumulated_iterations += 1

        if self._check_skip_next_step():
            self._is_last_step_skipped = True
            stats["skipped"] = True
            return stats

        self.add_noise_and_finalize()
        self.optimizer.step()

        for fn in self._step_hooks:
            fn()

        self._is_last_step_skipped = False
        self._accumulated_iterations = 0
        stats["skipped"] = False
        return stats

    def zero_grad(self) -> None:
        """Clear gradients, respecting virtual batching state."""
        if not self._is_last_step_skipped:
            self._summed_grads.clear()
        self.optimizer.zero_grad()

    # ------------------------------------------------------------------
    # Hooks and utilities
    # ------------------------------------------------------------------

    def attach_step_hook(self, fn: Callable[[], None]) -> None:
        """Register a function called after each real optimizer step."""
        self._step_hooks.append(fn)

    @property
    def param_groups(self):
        return self.optimizer.param_groups

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)
