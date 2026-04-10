"""Main orchestrator: wraps model, optimizer, and dataloader for DP-LoRA training."""

from __future__ import annotations

import logging
from typing import Optional, Union

import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from dp_lora.accounting.accountant import PrivacyAccountant, get_noise_multiplier
from dp_lora.config import DPLoRAConfig
from dp_lora.data.poisson_loader import create_poisson_dataloader
from dp_lora.grad_sample.grad_sample_module import GradSampleModule
from dp_lora.grad_sample.ghost_clipping import GhostClippingModule
from dp_lora.optimizers.dp_optimizer import DPOptimizer

logger = logging.getLogger(__name__)


class DPLoRAEngine:
    """Differentially Private LoRA training engine.

    Orchestrates model wrapping (per-sample gradient hooks), optimizer wrapping
    (clipping + noise), dataloader replacement (Poisson sampling), and privacy
    accounting.

    Usage::

        engine = DPLoRAEngine()
        model, optimizer, dataloader = engine.make_private_with_epsilon(
            model=peft_model,
            optimizer=adam_optimizer,
            data_loader=train_loader,
            target_epsilon=8.0,
            target_delta=1e-5,
            epochs=3,
            max_grad_norm=1.0,
            method="ffa",
        )

        for batch in dataloader:
            optimizer.zero_grad()
            loss = model(**batch).loss
            loss.backward()
            per_sample_grads = engine.grad_sample_module.get_per_sample_grads()
            optimizer.step(per_sample_grads)
            engine.grad_sample_module.clear_per_sample_grads()

        print(f"Final epsilon: {engine.get_epsilon():.2f}")
    """

    def __init__(self):
        self.accountant: Optional[PrivacyAccountant] = None
        self.grad_sample_module: Optional[GradSampleModule] = None
        self.ghost_clipping_module: Optional[GhostClippingModule] = None
        self._dp_optimizer: Optional[DPOptimizer] = None
        self._ghost_clipping: bool = False

    def make_private(
        self,
        *,
        model: nn.Module,
        optimizer: Optimizer,
        data_loader: DataLoader,
        noise_multiplier: float,
        max_grad_norm: float,
        method: str = "ffa",
        adapter_name: str = "default",
        poisson_sampling: bool = True,
        ghost_clipping: bool = False,
        secure_mode: bool = False,
        generator: Optional[torch.Generator] = None,
    ) -> tuple[nn.Module, DPOptimizer, DataLoader]:
        """Add DP guarantees to model, optimizer, and dataloader.

        Args:
            model: A PEFT model (from get_peft_model).
            optimizer: Any PyTorch optimizer.
            data_loader: Training DataLoader.
            noise_multiplier: Ratio of noise std to clipping norm (sigma).
            max_grad_norm: Per-sample gradient clipping threshold (C).
            method: "vanilla" (train A and B) or "ffa" (freeze A, train B).
            adapter_name: Name of the LoRA adapter in the PEFT model.
            poisson_sampling: Whether to replace dataloader with Poisson-sampled.
            ghost_clipping: Use memory-efficient ghost clipping (norm-only).
            secure_mode: Use secure noise generation (slower, attack-resistant).
            generator: Optional RNG for reproducibility.

        Returns:
            Tuple of (model, dp_optimizer, dp_dataloader).
        """
        self._ghost_clipping = ghost_clipping

        # 1. Wrap model with per-sample gradient hooks
        if ghost_clipping:
            self.ghost_clipping_module = GhostClippingModule(
                model, method=method, adapter_name=adapter_name
            )
            self.grad_sample_module = None
            num_params = self.ghost_clipping_module.num_trainable_params
        else:
            self.grad_sample_module = GradSampleModule(
                model, method=method, adapter_name=adapter_name
            )
            self.ghost_clipping_module = None
            num_params = self.grad_sample_module.num_trainable_params

        logger.info(
            "DP-LoRA: %d trainable parameters (method=%s, ghost_clipping=%s)",
            num_params,
            method,
            ghost_clipping,
        )

        # 2. Replace dataloader with Poisson-sampled
        if poisson_sampling:
            dp_data_loader = create_poisson_dataloader(data_loader)
            sample_rate = 1 / len(dp_data_loader)
        else:
            dp_data_loader = data_loader
            sample_rate = data_loader.batch_size / len(data_loader.dataset)

        expected_batch_size = int(len(data_loader.dataset) * sample_rate)

        # 3. Wrap optimizer with DP (clipping + noise)
        self._dp_optimizer = DPOptimizer(
            optimizer,
            noise_multiplier=noise_multiplier,
            max_grad_norm=max_grad_norm,
            expected_batch_size=expected_batch_size,
            generator=generator,
            secure_mode=secure_mode,
        )

        # 4. Set up privacy accountant
        self.accountant = PrivacyAccountant(
            noise_multiplier=noise_multiplier,
            sample_rate=sample_rate,
            delta=1e-5,  # Will be overridden by make_private_with_epsilon
        )
        self._dp_optimizer.attach_step_hook(self.accountant.step)

        logger.info(
            "DP-LoRA engine ready: sigma=%.4f, C=%.2f, sample_rate=%.4f, "
            "expected_batch_size=%d",
            noise_multiplier,
            max_grad_norm,
            sample_rate,
            expected_batch_size,
        )

        return model, self._dp_optimizer, dp_data_loader

    def make_private_with_epsilon(
        self,
        *,
        model: nn.Module,
        optimizer: Optimizer,
        data_loader: DataLoader,
        target_epsilon: float,
        target_delta: float,
        epochs: int,
        max_grad_norm: float,
        method: str = "ffa",
        adapter_name: str = "default",
        poisson_sampling: bool = True,
        ghost_clipping: bool = False,
        secure_mode: bool = False,
        generator: Optional[torch.Generator] = None,
    ) -> tuple[nn.Module, DPOptimizer, DataLoader]:
        """Add DP guarantees with automatic noise calibration.

        Computes the noise multiplier needed to achieve the target
        (epsilon, delta) budget after the specified number of epochs,
        then calls make_private().

        Args:
            target_epsilon: Desired privacy budget epsilon.
            target_delta: Desired privacy budget delta.
            epochs: Number of training epochs (for noise calibration).
            ghost_clipping: Use memory-efficient ghost clipping.
            (remaining args same as make_private)

        Returns:
            Tuple of (model, dp_optimizer, dp_dataloader).
        """
        sample_rate = data_loader.batch_size / len(data_loader.dataset)

        noise_multiplier = get_noise_multiplier(
            target_epsilon=target_epsilon,
            target_delta=target_delta,
            sample_rate=sample_rate,
            epochs=epochs,
        )

        logger.info(
            "Computed noise_multiplier=%.4f for target (eps=%.2f, delta=%.1e, "
            "epochs=%d, sample_rate=%.4f)",
            noise_multiplier,
            target_epsilon,
            target_delta,
            epochs,
            sample_rate,
        )

        model, dp_optimizer, dp_data_loader = self.make_private(
            model=model,
            optimizer=optimizer,
            data_loader=data_loader,
            noise_multiplier=noise_multiplier,
            max_grad_norm=max_grad_norm,
            method=method,
            adapter_name=adapter_name,
            poisson_sampling=poisson_sampling,
            ghost_clipping=ghost_clipping,
            secure_mode=secure_mode,
            generator=generator,
        )

        # Update accountant delta to match target
        self.accountant.delta = target_delta

        return model, dp_optimizer, dp_data_loader

    def get_epsilon(self, delta: Optional[float] = None) -> float:
        """Get current privacy expenditure."""
        if self.accountant is None:
            raise RuntimeError("Engine not initialized. Call make_private() first.")
        return self.accountant.get_epsilon(delta)
