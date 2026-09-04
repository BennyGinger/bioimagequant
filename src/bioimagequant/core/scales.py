import numpy as np


def validate_optional_scale(value: float | None,
                            *,
                            name: str,
                            allow_zero: bool = False,
                            ) -> None:
    """Validate optional temporal or spatial acquisition scales."""
    if value is None:
        return
    valid_range = value >= 0 if allow_zero else value > 0
    if not np.isfinite(value) or not valid_range:
        qualifier = "non-negative" if allow_zero else "greater than zero"
        raise ValueError(f"{name} must be finite and {qualifier}.")
