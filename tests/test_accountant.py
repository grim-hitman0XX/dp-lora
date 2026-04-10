"""Test privacy accountant and noise multiplier calibration."""

import pytest

from dp_lora.accounting.accountant import PrivacyAccountant, get_noise_multiplier


class TestPrivacyAccountant:
    def test_zero_steps_gives_zero_epsilon(self):
        acc = PrivacyAccountant(noise_multiplier=1.0, sample_rate=0.01, delta=1e-5)
        assert acc.get_epsilon() == 0.0

    def test_epsilon_increases_with_steps(self):
        acc = PrivacyAccountant(noise_multiplier=1.0, sample_rate=0.01, delta=1e-5)
        acc.step()
        eps1 = acc.get_epsilon()
        acc.step()
        eps2 = acc.get_epsilon()
        assert eps2 > eps1 > 0

    def test_higher_noise_gives_lower_epsilon(self):
        acc_low = PrivacyAccountant(noise_multiplier=0.5, sample_rate=0.01, delta=1e-5)
        acc_high = PrivacyAccountant(noise_multiplier=2.0, sample_rate=0.01, delta=1e-5)

        for _ in range(100):
            acc_low.step()
            acc_high.step()

        assert acc_high.get_epsilon() < acc_low.get_epsilon()

    def test_steps_counter(self):
        acc = PrivacyAccountant(noise_multiplier=1.0, sample_rate=0.01, delta=1e-5)
        for _ in range(10):
            acc.step()
        assert acc.steps == 10


class TestGetNoiseMultiplier:
    def test_returns_positive_sigma(self):
        sigma = get_noise_multiplier(
            target_epsilon=8.0, target_delta=1e-5,
            sample_rate=0.01, epochs=1,
        )
        assert sigma > 0

    def test_tighter_epsilon_requires_more_noise(self):
        sigma_loose = get_noise_multiplier(
            target_epsilon=8.0, target_delta=1e-5,
            sample_rate=0.01, epochs=1,
        )
        sigma_tight = get_noise_multiplier(
            target_epsilon=1.0, target_delta=1e-5,
            sample_rate=0.01, epochs=1,
        )
        assert sigma_tight > sigma_loose

    def test_more_epochs_requires_more_noise(self):
        sigma_1 = get_noise_multiplier(
            target_epsilon=8.0, target_delta=1e-5,
            sample_rate=0.01, epochs=1,
        )
        sigma_10 = get_noise_multiplier(
            target_epsilon=8.0, target_delta=1e-5,
            sample_rate=0.01, epochs=10,
        )
        assert sigma_10 > sigma_1

    def test_achieves_target_epsilon(self):
        """Verify that the computed sigma actually achieves the target."""
        target_eps = 4.0
        delta = 1e-5
        sample_rate = 0.01
        epochs = 3
        steps = int(1 / sample_rate) * epochs

        sigma = get_noise_multiplier(
            target_epsilon=target_eps, target_delta=delta,
            sample_rate=sample_rate, epochs=epochs,
        )

        # Now verify by running the accountant
        acc = PrivacyAccountant(noise_multiplier=sigma, sample_rate=sample_rate, delta=delta)
        for _ in range(steps):
            acc.step()

        actual_eps = acc.get_epsilon()
        assert abs(actual_eps - target_eps) < 0.1  # Within 0.1 tolerance
