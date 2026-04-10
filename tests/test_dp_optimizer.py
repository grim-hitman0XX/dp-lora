"""Test DPOptimizer: clipping, noise addition, and step."""

import pytest
import torch
from torch import nn

from dp_lora.optimizers.dp_optimizer import DPOptimizer, _generate_noise


class TestGenerateNoise:
    def test_noise_shape_and_device(self):
        ref = torch.zeros(3, 4)
        noise = _generate_noise(1.0, ref)
        assert noise.shape == (3, 4)

    def test_zero_std_returns_zeros(self):
        ref = torch.zeros(3, 4)
        noise = _generate_noise(0.0, ref)
        assert torch.all(noise == 0)

    def test_noise_std_approximately_correct(self):
        torch.manual_seed(42)
        ref = torch.zeros(10000)
        noise = _generate_noise(2.0, ref)
        # std should be approximately 2.0
        assert abs(noise.std().item() - 2.0) < 0.1

    def test_secure_mode(self):
        ref = torch.zeros(100)
        noise = _generate_noise(1.0, ref, secure_mode=True)
        assert noise.shape == (100,)


class TestClipAndAccumulate:
    def test_clipping_bounds_norm(self):
        param = nn.Parameter(torch.zeros(4, 3))
        optimizer = torch.optim.SGD([param], lr=1.0)
        dp_opt = DPOptimizer(
            optimizer, noise_multiplier=0.0, max_grad_norm=1.0,
            expected_batch_size=2,
        )

        # Create per-sample grads with large norms
        psg = torch.randn(2, 4, 3) * 10  # Large gradients
        per_sample_grads = [(param, psg)]

        stats = dp_opt.clip_and_accumulate(per_sample_grads)

        # After clipping, each per-sample grad should have norm <= C=1.0
        # The param.grad is the mean of clipped grads
        # We verify by checking that the clip happened
        assert stats["num_clipped"] == 2  # Both samples should be clipped

    def test_no_clipping_when_below_threshold(self):
        param = nn.Parameter(torch.zeros(4, 3))
        optimizer = torch.optim.SGD([param], lr=1.0)
        dp_opt = DPOptimizer(
            optimizer, noise_multiplier=0.0, max_grad_norm=100.0,
            expected_batch_size=2,
        )

        psg = torch.randn(2, 4, 3) * 0.01  # Small gradients
        per_sample_grads = [(param, psg)]

        stats = dp_opt.clip_and_accumulate(per_sample_grads)
        assert stats["num_clipped"] == 0

    def test_summed_grad_is_sum_of_clipped(self):
        param = nn.Parameter(torch.zeros(2, 2))
        optimizer = torch.optim.SGD([param], lr=1.0)
        dp_opt = DPOptimizer(
            optimizer, noise_multiplier=0.0, max_grad_norm=100.0,
            expected_batch_size=3,
        )

        # Small grads that won't be clipped
        psg = torch.tensor([[[1.0, 2.0], [3.0, 4.0]],
                            [[5.0, 6.0], [7.0, 8.0]],
                            [[9.0, 10.0], [11.0, 12.0]]])
        dp_opt.clip_and_accumulate([(param, psg)])

        # clip_and_accumulate writes to _summed_grads dict (raw sum, not scaled)
        expected_sum = psg.sum(dim=0)
        torch.testing.assert_close(dp_opt._summed_grads[id(param)], expected_sum)

        # After finalize, param.grad = (summed + noise) / expected_batch_size
        dp_opt.add_noise_and_finalize()
        expected_grad = expected_sum / 3  # noise_multiplier=0 so no noise
        torch.testing.assert_close(param.grad, expected_grad)


class TestAddNoiseAndFinalize:
    def test_noise_added_to_grad(self):
        torch.manual_seed(42)
        param = nn.Parameter(torch.zeros(100))
        optimizer = torch.optim.SGD([param], lr=1.0)
        dp_opt = DPOptimizer(
            optimizer, noise_multiplier=1.0, max_grad_norm=1.0,
            expected_batch_size=10,
        )

        # Manually set summed_grads (normally done by clip_and_accumulate)
        dp_opt._summed_grads[id(param)] = torch.zeros(100)
        dp_opt.add_noise_and_finalize()

        # Grad should no longer be zero (noise was added)
        assert param.grad.abs().sum() > 0


class TestStep:
    def test_step_calls_hooks(self):
        param = nn.Parameter(torch.zeros(4))
        param.grad = torch.zeros(4)
        optimizer = torch.optim.SGD([param], lr=1.0)
        dp_opt = DPOptimizer(
            optimizer, noise_multiplier=0.0, max_grad_norm=1.0,
            expected_batch_size=1,
        )

        hook_called = [False]
        dp_opt.attach_step_hook(lambda: hook_called.__setitem__(0, True))
        dp_opt.step()
        assert hook_called[0]
