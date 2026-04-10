"""Test per-sample gradient hooks against ground truth (one-sample-at-a-time)."""

import pytest
import torch
from torch import nn

from dp_lora.grad_sample.hooks import (
    clear_per_sample_grads,
    linear_backward_hook,
    linear_forward_hook,
)


def _ground_truth_per_sample_grads(layer, inputs, loss_fn):
    """Compute per-sample gradients by running each sample individually."""
    B = inputs.shape[0]
    grads = []
    for i in range(B):
        layer.zero_grad()
        out = layer(inputs[i : i + 1])
        loss = loss_fn(out)
        loss.backward()
        grads.append(layer.weight.grad.clone())
    return torch.stack(grads, dim=0)  # [B, out, in]


class TestLinearHooks2D:
    """Test hooks with 2D input [B, in_features]."""

    def test_per_sample_grad_matches_ground_truth(self):
        torch.manual_seed(42)
        layer = nn.Linear(8, 4, bias=False)
        inputs = torch.randn(5, 8)

        # Ground truth
        gt = _ground_truth_per_sample_grads(
            layer, inputs, lambda out: out.sum()
        )

        # Hook-based
        h1 = layer.register_forward_hook(linear_forward_hook)
        h2 = layer.register_full_backward_hook(linear_backward_hook)

        layer.zero_grad()
        out = layer(inputs)
        out.sum().backward()

        hook_grads = layer._dp_per_sample_grad_weight

        h1.remove()
        h2.remove()

        assert hook_grads.shape == gt.shape
        torch.testing.assert_close(hook_grads, gt, atol=1e-5, rtol=1e-4)

    def test_per_sample_grad_with_nonuniform_loss(self):
        torch.manual_seed(42)
        layer = nn.Linear(8, 4, bias=False)
        inputs = torch.randn(5, 8)
        targets = torch.randn(5, 4)

        # Ground truth
        gt = _ground_truth_per_sample_grads(
            layer, inputs,
            lambda out: nn.functional.mse_loss(out, targets[0:1], reduction="sum"),
        )

        # For hook-based, we need a different approach since loss involves targets
        # We need to compute one-at-a-time with proper target indexing
        grads = []
        for i in range(5):
            layer.zero_grad()
            out = layer(inputs[i : i + 1])
            loss = nn.functional.mse_loss(out, targets[i : i + 1], reduction="sum")
            loss.backward()
            grads.append(layer.weight.grad.clone())
        gt = torch.stack(grads, dim=0)

        # Hook-based
        h1 = layer.register_forward_hook(linear_forward_hook)
        h2 = layer.register_full_backward_hook(linear_backward_hook)

        layer.zero_grad()
        out = layer(inputs)
        loss = nn.functional.mse_loss(out, targets, reduction="sum")
        loss.backward()

        hook_grads = layer._dp_per_sample_grad_weight
        h1.remove()
        h2.remove()

        torch.testing.assert_close(hook_grads, gt, atol=1e-5, rtol=1e-4)


class TestLinearHooks3D:
    """Test hooks with 3D input [B, seq, in_features] (transformer-style)."""

    def test_per_sample_grad_3d(self):
        torch.manual_seed(42)
        layer = nn.Linear(8, 4, bias=False)
        B, seq = 3, 5
        inputs = torch.randn(B, seq, 8)

        # Ground truth: per-sample gradient sums over seq dim
        grads = []
        for i in range(B):
            layer.zero_grad()
            out = layer(inputs[i : i + 1])  # [1, seq, 4]
            out.sum().backward()
            grads.append(layer.weight.grad.clone())
        gt = torch.stack(grads, dim=0)  # [B, 4, 8]

        # Hook-based
        h1 = layer.register_forward_hook(linear_forward_hook)
        h2 = layer.register_full_backward_hook(linear_backward_hook)

        layer.zero_grad()
        out = layer(inputs)
        out.sum().backward()

        hook_grads = layer._dp_per_sample_grad_weight
        h1.remove()
        h2.remove()

        assert hook_grads.shape == (B, 4, 8)
        torch.testing.assert_close(hook_grads, gt, atol=1e-5, rtol=1e-4)


class TestBiasGrad:
    """Test that bias per-sample grads are computed when bias exists."""

    def test_bias_grad_2d(self):
        torch.manual_seed(42)
        layer = nn.Linear(8, 4, bias=True)
        inputs = torch.randn(5, 8)

        h1 = layer.register_forward_hook(linear_forward_hook)
        h2 = layer.register_full_backward_hook(linear_backward_hook)

        layer.zero_grad()
        out = layer(inputs)
        out.sum().backward()

        assert hasattr(layer, "_dp_per_sample_grad_bias")
        assert layer._dp_per_sample_grad_bias.shape == (5, 4)

        h1.remove()
        h2.remove()


class TestClearGrads:
    def test_clear_removes_attrs(self):
        layer = nn.Linear(4, 2)
        layer._dp_per_sample_grad_weight = torch.zeros(1)
        layer._dp_per_sample_grad_bias = torch.zeros(1)
        layer._dp_activations = torch.zeros(1)

        clear_per_sample_grads(layer)

        assert not hasattr(layer, "_dp_per_sample_grad_weight")
        assert not hasattr(layer, "_dp_per_sample_grad_bias")
        assert not hasattr(layer, "_dp_activations")
