from dataclasses import dataclass
from typing import Optional


@dataclass
class DPLoRAConfig:
    """Configuration for differentially private LoRA fine-tuning."""

    # Privacy parameters
    target_epsilon: float = 8.0
    target_delta: float = 1e-5
    max_grad_norm: float = 1.0
    noise_multiplier: Optional[float] = None  # Auto-computed from epsilon if None

    # LoRA method
    method: str = "ffa"  # "vanilla" or "ffa"
    adapter_name: str = "default"

    # Data
    poisson_sampling: bool = True

    # Memory optimization
    ghost_clipping: bool = False
    max_physical_batch_size: Optional[int] = None

    # Training
    epochs: int = 1  # Needed for noise multiplier calibration

    def __post_init__(self):
        if self.method not in ("vanilla", "ffa"):
            raise ValueError(f"method must be 'vanilla' or 'ffa', got '{self.method}'")
        if self.target_epsilon <= 0:
            raise ValueError(
                f"target_epsilon must be positive, got {self.target_epsilon}"
            )
        if self.target_delta <= 0 or self.target_delta >= 1:
            raise ValueError(f"target_delta must be in (0, 1), got {self.target_delta}")
        if self.max_grad_norm <= 0:
            raise ValueError(
                f"max_grad_norm must be positive, got {self.max_grad_norm}"
            )
