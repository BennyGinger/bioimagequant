from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from bioimagequant.core.validation import validate_positive_integer


@dataclass(frozen=True, slots=True)
class BinningSettings:
    """Validated distance-bin dimensions shared by profile calculations."""

    bin_width: float = 5.0
    maximum_distance: float | None = None

    def __post_init__(self) -> None:
        """Validate the bin width and optional distance limit."""
        if not np.isfinite(self.bin_width) or self.bin_width <= 0:
            raise ValueError(
                "bin_width must be finite and greater than zero.")
        if self.maximum_distance is not None and (
                not np.isfinite(self.maximum_distance)
                or self.maximum_distance < 0):
            raise ValueError(
                "maximum_distance must be finite and non-negative.")

    @classmethod
    def from_maximum_bins(cls,
                          bin_width: float,
                          maximum_bins: int | None,
                          ) -> "BinningSettings":
        """Build settings from a requested maximum number of bins."""
        if maximum_bins is None:
            return cls(bin_width=bin_width)
        validate_positive_integer(maximum_bins, name="maximum_bins")
        maximum_distance = float(np.nextafter(
            bin_width * maximum_bins, -np.inf))
        return cls(
            bin_width=bin_width,
            maximum_distance=maximum_distance,)


@dataclass(frozen=True)
class DistanceProfile:
    """Binned intensity measurements as a function of reference distance."""

    bin_start: NDArray[np.float64]
    bin_end: NDArray[np.float64]
    distance: NDArray[np.float64]
    mean_intensity: NDArray[np.float64]
    standard_deviation: NDArray[np.float64]
    pixel_count: NDArray[np.int64]


@dataclass(frozen=True, slots=True)
class ProfileResult:
    """One calculated profile and the identities needed for tabulation."""

    frame_index: int
    intensity_channel: str | None
    reference_name: str
    reference_channel: str | None
    roi_name: str | None
    roi_channel: str | None
    profile: DistanceProfile | None
