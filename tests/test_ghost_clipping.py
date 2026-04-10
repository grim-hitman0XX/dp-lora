"""Test ghost clipping norms match materialized per-sample gradient norms."""

import pytest
import torch
from torch import nn

from peft import LoraConfig, get_peft_model

from dp_lora.grad_sample.grad_sample_module import GradSampleModule
from dp_lora.grad_sample.ghost_clipping import GhostClippingModule


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(16, 32)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(32, 4)

    def forward(self, x):
        return self.linear2(self.relu(self.linear1(x)))


def _make_peft_model():
    torch.manual_seed(42)
    base = SimpleModel()
    config = LoraConfig(
        r=4, lora_alpha=8,
        target_modules=["linear1", "linear2"],
        lora_dropout=0.0, bias="none",
    )
    return get_peft_model(base, config)


class TestGhostNorms2D:
    """2D input: ghost norms should be EXACT match to materialized norms."""

    def test_ghost_norms_exact_2d(self):
        torch.manual_seed(42)
        model = _make_peft_model()
        X = torch.randn(5, 16)
        y = torch.randint(0, 4, (5,))

        # --- Materialized path ---
        gsm = GradSampleModule(model, method="vanilla")
        model.zero_grad()
        out = model(X)
        loss = nn.functional.cross_entropy(out, y)
        loss.backward()

        per_sample_grads = gsm.get_per_sample_grads()
        B = per_sample_grads[0][1].shape[0]
        materialized_norms_sq = torch.zeros(B)
        for _, psg in per_sample_grads:
            flat = psg.reshape(B, -1)
            materialized_norms_sq += flat.norm(2, dim=1).square()

        gsm.clear_per_sample_grads()
        gsm.remove_hooks()

        # --- Ghost path ---
        gcm = GhostClippingModule(model, method="vanilla")
        model.zero_grad()
        out = model(X)
        loss = nn.functional.cross_entropy(out, y)
        loss.backward()

        ghost_norms_sq = gcm.get_per_sample_norm_sq()
        gcm.clear_state()
        gcm.remove_hooks()

        # For 2D inputs, ghost norms should be exact
        torch.testing.assert_close(
            ghost_norms_sq, materialized_norms_sq, atol=1e-4, rtol=1e-3
        )

    def test_clip_factors_2d(self):
        torch.manual_seed(42)
        model = _make_peft_model()
        X = torch.randn(5, 16)
        y = torch.randint(0, 4, (5,))

        gcm = GhostClippingModule(model, method="ffa")
        model.zero_grad()
        out = model(X)
        loss = nn.functional.cross_entropy(out, y)
        loss.backward()

        clip_factors = gcm.compute_clip_factors(max_grad_norm=1.0)
        assert clip_factors.shape == (5,)
        assert (clip_factors <= 1.0).all()
        assert (clip_factors > 0).all()

        gcm.clear_state()
        gcm.remove_hooks()


class TestGhostNorms3D:
    """3D input: ghost norms should UPPER BOUND materialized norms."""

    def test_ghost_norms_upper_bound_3d(self):
        torch.manual_seed(42)
        # Use a simple model with 3D path
        layer = nn.Linear(8, 4, bias=False)
        B, seq = 5, 10
        X = torch.randn(B, seq, 8, requires_grad=True)

        # --- Materialized per-sample grad norms ---
        from dp_lora.grad_sample.hooks import linear_forward_hook, linear_backward_hook
        h1 = layer.register_forward_hook(linear_forward_hook)
        h2 = layer.register_full_backward_hook(linear_backward_hook)

        out = layer(X)
        out.sum().backward()
        mat_psg = layer._dp_per_sample_grad_weight  # [B, 4, 8]
        mat_norms_sq = mat_psg.reshape(B, -1).norm(2, dim=1).square()
        h1.remove()
        h2.remove()

        # --- Ghost norms ---
        from dp_lora.grad_sample.ghost_clipping import ghost_forward_hook, ghost_backward_hook
        h3 = layer.register_forward_hook(ghost_forward_hook)
        h4 = layer.register_full_backward_hook(ghost_backward_hook)

        layer.zero_grad()
        X2 = X.detach().requires_grad_(True)
        out2 = layer(X2)
        out2.sum().backward()
        ghost_norms_sq = layer._dp_grad_norm_sq
        h3.remove()
        h4.remove()

        # Ghost norms should be >= materialized norms (upper bound)
        assert (ghost_norms_sq >= mat_norms_sq - 1e-4).all(), (
            f"Ghost norms should upper-bound materialized norms.\n"
            f"Ghost:  {ghost_norms_sq}\n"
            f"Actual: {mat_norms_sq}"
        )


class TestGhostFFA:
    """Verify ghost clipping freezes lora_A in FFA mode."""

    def test_ffa_freezes_a(self):
        model = _make_peft_model()
        gcm = GhostClippingModule(model, method="ffa")

        for name, param in model.named_parameters():
            if "lora_A" in name:
                assert not param.requires_grad, f"{name} should be frozen"

        gcm.remove_hooks()


class TestGhostClippingModuleAPI:
    """Test the GhostClippingModule API matches GradSampleModule's."""

    def test_num_trainable_params(self):
        model = _make_peft_model()
        gsm = GradSampleModule(model, method="vanilla")
        gsm_params = gsm.num_trainable_params
        gsm.remove_hooks()

        gcm = GhostClippingModule(model, method="vanilla")
        gcm_params = gcm.num_trainable_params
        gcm.remove_hooks()

        assert gsm_params == gcm_params

    def test_no_lora_layers_raises(self):
        model = nn.Linear(4, 2)
        with pytest.raises(ValueError, match="No LoRA layers"):
            GhostClippingModule(model)
