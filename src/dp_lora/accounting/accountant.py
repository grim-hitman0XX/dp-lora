"""Privacy accounting using Google's dp_accounting library (RDP accountant)."""

import logging
from typing import Optional

from dp_accounting import dp_event as event
from dp_accounting.rdp import rdp_privacy_accountant as rdp
from scipy import optimize as opt

logger = logging.getLogger(__name__)

RDP_ORDERS = (
    [1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 3.0, 3.5, 4.0, 4.5]
    + list(range(5, 64))
    + [128, 256, 512]
)


class PrivacyAccountant:
    """Tracks cumulative privacy expenditure using Renyi Differential Privacy.

    Uses Google's dp_accounting library for tight composition bounds
    under Poisson subsampled Gaussian mechanism.
    """

    def __init__(self, noise_multiplier: float, sample_rate: float, delta: float):
        self.noise_multiplier = noise_multiplier
        self.sample_rate = sample_rate
        self.delta = delta
        self._steps = 0
        self._accountant = rdp.RdpAccountant(RDP_ORDERS)

    def step(self) -> None:
        """Record one DP-SGD step (one batch processed)."""
        self._accountant.compose(
            event.PoissonSampledDpEvent(
                self.sample_rate,
                event.GaussianDpEvent(self.noise_multiplier),
            )
        )
        self._steps += 1

    def get_epsilon(self, delta: Optional[float] = None) -> float:
        """Compute current (epsilon, delta)-DP guarantee.

        Args:
            delta: Override the default delta. If None, uses the delta
                   provided at initialization.

        Returns:
            The epsilon value for the given delta.
        """
        d = delta if delta is not None else self.delta
        if self._steps == 0:
            return 0.0
        return self._accountant.get_epsilon(d)

    @property
    def steps(self) -> int:
        return self._steps


def get_noise_multiplier(
    *,
    target_epsilon: float,
    target_delta: float,
    sample_rate: float,
    epochs: int,
    steps_per_epoch: Optional[int] = None,
) -> float:
    """Compute the noise multiplier (sigma) needed to achieve a target privacy budget.

    Uses binary search (Brent's method) over the RDP accountant to find the
    noise multiplier that yields exactly target_epsilon after the specified
    number of training steps.

    Args:
        target_epsilon: Desired epsilon for the privacy guarantee.
        target_delta: Desired delta for the privacy guarantee.
        sample_rate: Probability of including each record in a batch (B/N).
        epochs: Number of training epochs.
        steps_per_epoch: Steps per epoch. If None, computed as int(1/sample_rate).

    Returns:
        The noise multiplier (sigma) to use in DP-SGD.
    """
    if steps_per_epoch is None:
        steps_per_epoch = int(1 / sample_rate)

    total_steps = steps_per_epoch * epochs

    def objective(noise_multiplier: float) -> float:
        accountant = rdp.RdpAccountant(RDP_ORDERS)
        accountant.compose(
            event.SelfComposedDpEvent(
                event.PoissonSampledDpEvent(
                    sample_rate, event.GaussianDpEvent(noise_multiplier)
                ),
                total_steps,
            )
        )
        return accountant.get_epsilon(target_delta) - target_epsilon

    try:
        optimal_noise = opt.brentq(objective, 1e-6, 1000.0)
    except ValueError:
        raise ValueError(
            f"Could not find a noise multiplier for epsilon={target_epsilon}, "
            f"delta={target_delta}, sample_rate={sample_rate}, "
            f"epochs={epochs}. The target may be infeasible."
        )

    logger.info("Computed noise multiplier: %.4f", optimal_noise)
    return optimal_noise
