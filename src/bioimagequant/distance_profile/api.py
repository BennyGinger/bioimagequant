from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
import logging
from typing import Any

from numpy.typing import NDArray
import pandas as pd

from bioimagequant.core.arrays import ArrayData
from bioimagequant.core.scales import validate_optional_scale
from bioimagequant.core.validation import validate_binary_array
from bioimagequant.distance_profile.models import BinningSettings
from bioimagequant.distance_profile.stack import calculate_profiles
from bioimagequant.distance_profile.table import profiles_to_table


logger = logging.getLogger(__name__)


@dataclass
class DistanceProfiler:
    """Build binned distance profiles from named arrays."""

    interval: float | None = None
    pixel_size: float | None = None
    _intensity: ArrayData | None = field(init=False, default=None, repr=False)
    _references: dict[str, ArrayData] = field(
        init=False, default_factory=dict, repr=False)
    _rois: dict[str, ArrayData] = field(
        init=False, default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        """Validate optional temporal and spatial acquisition scales."""
        validate_optional_scale(
            self.interval, name="interval", allow_zero=True)
        validate_optional_scale(self.pixel_size, name="pixel_size")

    def add_intensity(self,
                      array: NDArray[Any],
                      axes: str,
                      *,
                      channel_labels: Sequence[str] | None = None,
                      ) -> None:
        """Register a 2D intensity image or time/channel stack."""
        intensity = ArrayData(array, axes, channel_labels)
        for name, mask in (*self._references.items(), *self._rois.items()):
            _validate_mask_compatibility(name, mask, intensity)
        self._intensity = intensity

    def add_ref(self,
                array: NDArray[Any],
                axes: str,
                *,
                name: str,
                channel_labels: Sequence[str] | None = None,
                ) -> None:
        """Register one named binary reference mask or stack."""
        self._add_mask(
            self._references, array, axes, name, channel_labels,
            role="reference")

    def add_roi(self,
                array: NDArray[Any],
                axes: str,
                *,
                name: str,
                channel_labels: Sequence[str] | None = None,
                ) -> None:
        """Register one named binary inclusion mask or stack."""
        self._add_mask(
            self._rois, array, axes, name, channel_labels, role="ROI")

    def _add_mask(self,
                  collection: dict[str, ArrayData],
                  array: NDArray[Any],
                  axes: str,
                  name: str,
                  channel_labels: Sequence[str] | None,
                  *,
                  role: str,
                  ) -> None:
        """Validate and register a named reference or ROI mask."""
        normalized_name, mask = _resolve_binary_mask(array,
                                                     axes,
                                                     name=name,
                                                     channel_labels=channel_labels,
                                                     role=role,)
        if self._intensity is not None:
            _validate_mask_compatibility(normalized_name, mask, self._intensity)

        if normalized_name in collection:
            logger.warning("Overwriting %s array %r.", role, normalized_name)
        collection[normalized_name] = mask

    def calculate(self,
                  *,
                  bin_width: float = 5.0,
                  maximum_bins: int | None = None,
                  workers: int = 1,
                  ) -> pd.DataFrame:
        """
        Calculate every profile, optionally across frame processes.
        """
        intensity = self._intensity
        if intensity is None:
            raise ValueError(
                "Add an intensity array before calculating profiles.")
        if not self._references:
            raise ValueError(
                "Add at least one reference array before calculating profiles.")

        settings = BinningSettings.from_maximum_bins(
            bin_width, maximum_bins)

        profiles = calculate_profiles(
            intensity=intensity,
            references=self._references,
            rois=self._rois,
            settings=settings,
            workers=workers,)
        return profiles_to_table(
            profiles,
            bin_width=settings.bin_width,
            interval=self.interval,
            pixel_size=self.pixel_size,)


def _resolve_binary_mask(
        array: NDArray[Any],
        axes: str,
        *,
        name: str,
        channel_labels: Sequence[str] | None,
        role: str,
        ) -> tuple[str, ArrayData]:
    """Normalize and validate one named binary mask."""
    normalized_name = name.strip()
    if not normalized_name:
        raise ValueError(f"{role.capitalize()} name cannot be empty.")

    mask = ArrayData(array, axes, channel_labels)
    if mask.channel_labels is None:
        raise ValueError(
            f"{role.capitalize()} arrays require a channel label.")
    validate_binary_array(
        mask.array, description=f"{role.capitalize()} array")
    return normalized_name, mask


def _validate_mask_compatibility(name: str,
                                 mask: ArrayData,
                                 intensity: ArrayData,
                                 ) -> None:
    """Require a mask to match the intensity space and frame count."""
    if mask.spatial_shape != intensity.spatial_shape:
        raise ValueError(
            f"Mask {name!r} shape {mask.spatial_shape} does not match intensity "
            f"shape {intensity.spatial_shape}.")
    if mask.frame_count not in {1, intensity.frame_count}:
        raise ValueError(
            f"Mask {name!r} has {mask.frame_count} frames; expected 1 or "
            f"{intensity.frame_count}.")
