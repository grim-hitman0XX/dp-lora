"""End-to-end test: create a PEFT model, call make_private, train one step."""

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from peft import LoraConfig, get_peft_model

from dp_lora import DPLoRAEngine


def _make_tiny_model():
    """Create a tiny transformer-like model with LoRA."""
    base = nn.Sequential(
        nn.Linear(16, 32),
        nn.ReLU(),
        nn.Linear(32, 16),
    )
    # PEFT needs a model with named modules matching target_modules
    # Use a simple wrapper
    return base


class SimpleModel(nn.Module):
    """Simple model compatible with PEFT LoRA."""
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(16, 32)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(32, 4)

    def forward(self, x):
        return self.linear2(self.relu(self.linear1(x)))


@pytest.fixture
def peft_model():
    torch.manual_seed(42)
    base = SimpleModel()
    config = LoraConfig(
        r=4,
        lora_alpha=8,
        target_modules=["linear1", "linear2"],
        lora_dropout=0.0,
        bias="none",
    )
    return get_peft_model(base, config)


@pytest.fixture
def train_loader():
    torch.manual_seed(42)
    X = torch.randn(100, 16)
    y = torch.randint(0, 4, (100,))
    ds = TensorDataset(X, y)
    return DataLoader(ds, batch_size=10)


class TestMakePrivate:
    def test_make_private_returns_correct_types(self, peft_model, train_loader):
        engine = DPLoRAEngine()
        model, optimizer_dp, dp_loader = engine.make_private(
            model=peft_model,
            optimizer=torch.optim.Adam(peft_model.parameters(), lr=1e-3),
            data_loader=train_loader,
            noise_multiplier=1.0,
            max_grad_norm=1.0,
        )
        assert engine.grad_sample_module is not None
        assert engine.accountant is not None

    def test_ffa_freezes_lora_a(self, peft_model, train_loader):
        engine = DPLoRAEngine()
        engine.make_private(
            model=peft_model,
            optimizer=torch.optim.Adam(peft_model.parameters(), lr=1e-3),
            data_loader=train_loader,
            noise_multiplier=1.0,
            max_grad_norm=1.0,
            method="ffa",
        )
        # Check that lora_A parameters are frozen
        for name, param in peft_model.named_parameters():
            if "lora_A" in name:
                assert not param.requires_grad, f"{name} should be frozen in FFA mode"

    def test_vanilla_keeps_lora_a_trainable(self, peft_model, train_loader):
        engine = DPLoRAEngine()
        engine.make_private(
            model=peft_model,
            optimizer=torch.optim.Adam(peft_model.parameters(), lr=1e-3),
            data_loader=train_loader,
            noise_multiplier=1.0,
            max_grad_norm=1.0,
            method="vanilla",
        )
        has_trainable_a = False
        for name, param in peft_model.named_parameters():
            if "lora_A" in name and param.requires_grad:
                has_trainable_a = True
        assert has_trainable_a


class TestTrainOneStep:
    def test_one_step_vanilla(self, peft_model, train_loader):
        engine = DPLoRAEngine()
        model, dp_opt, dp_loader = engine.make_private(
            model=peft_model,
            optimizer=torch.optim.Adam(peft_model.parameters(), lr=1e-3),
            data_loader=train_loader,
            noise_multiplier=1.0,
            max_grad_norm=1.0,
            method="vanilla",
            poisson_sampling=False,  # Use regular loader for deterministic test
        )

        batch = next(iter(dp_loader))
        X, y = batch

        dp_opt.zero_grad()
        out = model(X)
        loss = nn.functional.cross_entropy(out, y)
        loss.backward()

        grads = engine.grad_sample_module.get_per_sample_grads()
        assert len(grads) > 0, "Should have per-sample gradients"

        stats = dp_opt.step(grads)
        engine.grad_sample_module.clear_per_sample_grads()

        assert engine.get_epsilon() > 0, "Epsilon should be positive after one step"

    def test_one_step_ffa(self, peft_model, train_loader):
        engine = DPLoRAEngine()
        model, dp_opt, dp_loader = engine.make_private(
            model=peft_model,
            optimizer=torch.optim.Adam(
                [p for p in peft_model.parameters() if p.requires_grad], lr=1e-3
            ),
            data_loader=train_loader,
            noise_multiplier=1.0,
            max_grad_norm=1.0,
            method="ffa",
            poisson_sampling=False,
        )

        batch = next(iter(dp_loader))
        X, y = batch

        dp_opt.zero_grad()
        out = model(X)
        loss = nn.functional.cross_entropy(out, y)
        loss.backward()

        grads = engine.grad_sample_module.get_per_sample_grads()
        assert len(grads) > 0

        stats = dp_opt.step(grads)
        engine.grad_sample_module.clear_per_sample_grads()

        assert engine.get_epsilon() > 0


class TestMakePrivateWithEpsilon:
    def test_computes_noise_multiplier(self, peft_model, train_loader):
        engine = DPLoRAEngine()
        model, dp_opt, dp_loader = engine.make_private_with_epsilon(
            model=peft_model,
            optimizer=torch.optim.Adam(peft_model.parameters(), lr=1e-3),
            data_loader=train_loader,
            target_epsilon=8.0,
            target_delta=1e-5,
            epochs=1,
            max_grad_norm=1.0,
        )
        assert dp_opt.noise_multiplier > 0
