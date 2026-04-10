"""Test virtual batching: DPOptimizer skip/accumulate and VirtualBatchManager."""

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from peft import LoraConfig, get_peft_model

from dp_lora.optimizers.dp_optimizer import DPOptimizer
from dp_lora import DPLoRAEngine
from dp_lora.data.virtual_batch import VirtualBatchManager


class TestSkipAccumulate:
    """Test DPOptimizer's signal_skip_step and accumulation behavior."""

    def test_skip_defers_update(self):
        """When skip=True, step() clips+accumulates but doesn't update params."""
        param = nn.Parameter(torch.ones(4))
        optimizer = torch.optim.SGD([param], lr=1.0)
        dp_opt = DPOptimizer(
            optimizer, noise_multiplier=0.0, max_grad_norm=100.0,
            expected_batch_size=6,
        )

        original_param = param.data.clone()

        # Signal skip and call step
        dp_opt.signal_skip_step(do_skip=True)
        psg = torch.randn(3, 4)
        dp_opt.step([(param, psg)])

        # Param should NOT have changed (step was skipped)
        torch.testing.assert_close(param.data, original_param)

        # But summed_grad should be populated in the optimizer's dict
        assert id(param) in dp_opt._summed_grads

    def test_accumulation_across_micro_batches(self):
        """Clipped grads accumulate across multiple skipped steps."""
        param = nn.Parameter(torch.zeros(4))
        optimizer = torch.optim.SGD([param], lr=1.0)
        dp_opt = DPOptimizer(
            optimizer, noise_multiplier=0.0, max_grad_norm=100.0,
            expected_batch_size=6,
        )

        # Micro-batch 1: skip
        psg1 = torch.ones(3, 4)  # Each sample has grad [1,1,1,1]
        dp_opt.signal_skip_step(do_skip=True)
        dp_opt.step([(param, psg1)])
        dp_opt.zero_grad()

        # Micro-batch 2: don't skip (finalize)
        psg2 = torch.ones(3, 4) * 2  # Each sample has grad [2,2,2,2]
        dp_opt.signal_skip_step(do_skip=False)
        dp_opt.step([(param, psg2)])

        # summed_grad should be sum of both micro-batches:
        # psg1.sum(0) = [3,3,3,3], psg2.sum(0) = [6,6,6,6], total = [9,9,9,9]
        # param.grad = total / expected_batch_size = [9,9,9,9] / 6 = [1.5,1.5,1.5,1.5]
        expected = torch.tensor([1.5, 1.5, 1.5, 1.5])
        torch.testing.assert_close(param.grad, expected)

    def test_zero_grad_preserves_summed_grad_on_skip(self):
        """zero_grad() keeps summed_grad when last step was skipped."""
        param = nn.Parameter(torch.zeros(4))
        optimizer = torch.optim.SGD([param], lr=1.0)
        dp_opt = DPOptimizer(
            optimizer, noise_multiplier=0.0, max_grad_norm=100.0,
            expected_batch_size=1,
        )

        psg = torch.ones(1, 4)
        dp_opt.signal_skip_step(do_skip=True)
        dp_opt.step([(param, psg)])

        # After skipped step, zero_grad should keep summed_grad
        dp_opt.zero_grad()
        assert id(param) in dp_opt._summed_grads

    def test_zero_grad_clears_summed_grad_on_real_step(self):
        """zero_grad() clears summed_grad after a real step."""
        param = nn.Parameter(torch.zeros(4))
        optimizer = torch.optim.SGD([param], lr=1.0)
        dp_opt = DPOptimizer(
            optimizer, noise_multiplier=0.0, max_grad_norm=100.0,
            expected_batch_size=1,
        )

        psg = torch.ones(1, 4)
        dp_opt.step([(param, psg)])  # No skip signal → real step

        dp_opt.zero_grad()
        assert id(param) not in dp_opt._summed_grads

    def test_hooks_fire_only_on_real_step(self):
        """Step hooks fire only on non-skipped steps."""
        param = nn.Parameter(torch.zeros(4))
        optimizer = torch.optim.SGD([param], lr=1.0)
        dp_opt = DPOptimizer(
            optimizer, noise_multiplier=0.0, max_grad_norm=100.0,
            expected_batch_size=1,
        )

        call_count = [0]
        dp_opt.attach_step_hook(lambda: call_count.__setitem__(0, call_count[0] + 1))

        psg = torch.ones(1, 4)

        # Skip → hook should NOT fire
        dp_opt.signal_skip_step(do_skip=True)
        dp_opt.step([(param, psg)])
        assert call_count[0] == 0

        # Don't skip → hook SHOULD fire
        dp_opt.signal_skip_step(do_skip=False)
        dp_opt.step([(param, psg)])
        assert call_count[0] == 1

    def test_privacy_accountant_counts_logical_batches(self):
        """Privacy accountant should count logical batches, not micro-batches."""
        param = nn.Parameter(torch.zeros(4))
        optimizer = torch.optim.SGD([param], lr=1.0)
        dp_opt = DPOptimizer(
            optimizer, noise_multiplier=1.0, max_grad_norm=1.0,
            expected_batch_size=1,
        )

        step_count = [0]
        dp_opt.attach_step_hook(lambda: step_count.__setitem__(0, step_count[0] + 1))

        psg = torch.ones(1, 4)

        # 4 micro-batches: 3 skipped + 1 real = 1 logical batch
        for i in range(3):
            dp_opt.signal_skip_step(do_skip=True)
            dp_opt.step([(param, psg)])
            dp_opt.zero_grad()
        dp_opt.signal_skip_step(do_skip=False)
        dp_opt.step([(param, psg)])
        dp_opt.zero_grad()

        assert step_count[0] == 1  # Only 1 logical batch


class TestVirtualBatchManagerEndToEnd:
    """End-to-end test with a PEFT model and VirtualBatchManager."""

    def _make_model_and_engine(self):
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = nn.Linear(8, 16)
                self.relu = nn.ReLU()
                self.linear2 = nn.Linear(16, 2)

            def forward(self, x):
                return self.linear2(self.relu(self.linear1(x)))

        torch.manual_seed(42)
        base = SimpleModel()
        config = LoraConfig(
            r=2, lora_alpha=4,
            target_modules=["linear1", "linear2"],
            lora_dropout=0.0, bias="none",
        )
        model = get_peft_model(base, config)

        X = torch.randn(60, 8)
        y = torch.randint(0, 2, (60,))
        ds = TensorDataset(X, y)
        loader = DataLoader(ds, batch_size=20)

        engine = DPLoRAEngine()
        model, dp_opt, dp_loader = engine.make_private(
            model=model,
            optimizer=torch.optim.Adam(
                [p for p in model.parameters() if p.requires_grad], lr=1e-3
            ),
            data_loader=loader,
            noise_multiplier=1.0,
            max_grad_norm=1.0,
            method="ffa",
            poisson_sampling=False,
        )
        return model, dp_opt, dp_loader, engine

    def test_virtual_batch_trains_without_error(self):
        model, dp_opt, dp_loader, engine = self._make_model_and_engine()

        with VirtualBatchManager(
            data_loader=dp_loader,
            max_physical_batch_size=5,
            optimizer=dp_opt,
        ) as vb_loader:
            for batch in vb_loader:
                X, y = batch
                dp_opt.zero_grad()
                out = model(X)
                loss = nn.functional.cross_entropy(out, y)
                loss.backward()
                grads = engine.grad_sample_module.get_per_sample_grads()
                dp_opt.step(grads)
                engine.grad_sample_module.clear_per_sample_grads()

        # Should have processed all data
        assert engine.accountant.steps > 0
