"""Single-plane distance-based intensity profiles."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TypeAlias, cast

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import distance_transform_edt


Sampling: TypeAlias = float | tuple[float, float]


@dataclass(frozen=True)
class DistanceProfile:
    """Binned intensity measurements as a function of reference distance."""

    bin_start: NDArray[np.float64]
    bin_end: NDArray[np.float64]
    distance: NDArray[np.float64]
    mean_intensity: NDArray[np.float64]
    standard_deviation: NDArray[np.float64]
    pixel_count: NDArray[np.int64]


def _as_2d_array(array: NDArray[Any], *, name: str) -> NDArray[Any]:
    result = np.asarray(array)
    if result.ndim != 2:
        raise ValueError(f"{name} must be a 2D array; got shape {result.shape}.")
    return result


def _validated_sampling(sampling: Sampling) -> float | tuple[float, float]:
    values = np.asarray(sampling, dtype=float)
    if values.ndim > 1 or values.size not in (1, 2):
        raise ValueError(
            "sampling must be one positive value or two Y/X values.")
    if not np.all(np.isfinite(values)) or np.any(values <= 0):
        raise ValueError(
            "sampling values must be finite and greater than zero.")
    if values.size == 1:
        return float(values.item())
    return float(values[0]), float(values[1])


def _distance_map(reference_mask: NDArray[Any],
                  *,
                  sampling: Sampling = 1.0,
                  ) -> NDArray[np.float64]:
    """Return each pixel's Euclidean distance to the reference mask."""
    reference = _as_2d_array(
        reference_mask, name="reference_mask").astype(bool, copy=False)
    if not np.any(reference):
        raise ValueError(
            "reference_mask must contain at least one reference pixel.")
    return cast(
        NDArray[np.float64],
        distance_transform_edt(
            np.logical_not(reference),
            sampling=_validated_sampling(sampling)),)


def _validate_same_shape(array: NDArray[Any],
                         reference: NDArray[Any],
                         *,
                         name: str,
                         ) -> None:
    if array.shape != reference.shape:
        raise ValueError(
            f"{name} must have the same shape as image; got "
            f"{array.shape} and {reference.shape}.")


def _validate_profile_settings(bin_width: float,
                               maximum_distance: float | None,
                               ) -> None:
    if not np.isfinite(bin_width) or bin_width <= 0:
        raise ValueError("bin_width must be finite and greater than zero.")
    if maximum_distance is not None and (
            not np.isfinite(maximum_distance) or maximum_distance < 0):
        raise ValueError(
            "maximum_distance must be finite and non-negative.")


def _resolve_inclusion_mask(inclusion_mask: NDArray[Any] | None,
                            image_shape: tuple[int, int],
                            ) -> NDArray[np.bool_]:
    if inclusion_mask is None:
        return np.ones(image_shape, dtype=bool)
    included = _as_2d_array(inclusion_mask, name="inclusion_mask")
    if included.shape != image_shape:
        raise ValueError(
            "inclusion_mask must have the same shape as image; got "
            f"{included.shape} and {image_shape}.")
    return included.astype(bool, copy=False)


def _select_profile_pixels(
        values: NDArray[np.float64],
        distances: NDArray[np.float64],
        included: NDArray[np.bool_],
        ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    selected = included & np.isfinite(values)
    return distances[selected], values[selected]


def _create_distance_bins(
        selected_distances: NDArray[np.float64],
        bin_width: float,
        maximum_distance: float | None,
        ) -> tuple[NDArray[np.float64], float]:
    if maximum_distance is None:
        if selected_distances.size == 0:
            raise ValueError(
                "No finite image pixels are selected for profiling.")
        upper_limit = float(selected_distances.max())
    else:
        upper_limit = float(maximum_distance)
    bin_count = int(np.floor(upper_limit / bin_width)) + 1
    edges = np.arange(bin_count + 1, dtype=float) * bin_width
    return edges, upper_limit


def _aggregate_distance_bins(
        selected_distances: NDArray[np.float64],
        selected_values: NDArray[np.float64],
        edges: NDArray[np.float64],
        upper_limit: float,
        ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.int64]]:
    bin_width = float(edges[1] - edges[0])
    bin_count = edges.size - 1
    within_range = selected_distances <= upper_limit
    distances = selected_distances[within_range]
    values = selected_values[within_range]

    indices = np.floor(distances / bin_width).astype(np.int64)
    indices = np.minimum(indices, bin_count - 1)
    counts = np.bincount(indices, minlength=bin_count).astype(np.int64)
    sums = np.bincount(indices, weights=values, minlength=bin_count)
    squared_sums = np.bincount(
        indices, weights=np.square(values), minlength=bin_count)

    means = np.full(bin_count, np.nan, dtype=float)
    deviations = np.full(bin_count, np.nan, dtype=float)
    populated = counts > 0
    means[populated] = sums[populated] / counts[populated]
    variances = np.zeros(bin_count, dtype=float)
    variances[populated] = (
        squared_sums[populated] / counts[populated]
        - np.square(means[populated]))
    deviations[populated] = np.sqrt(np.maximum(variances[populated], 0.0))
    return means, deviations, counts


def distance_profile(image: NDArray[Any],
                     reference_mask: NDArray[Any],
                     *,
                     inclusion_mask: NDArray[Any] | None = None,
                     bin_width: float = 5.0,
                     sampling: Sampling = 1.0,
                     maximum_distance: float | None = None,
                     ) -> DistanceProfile:
    """Average a 2D image in bins of distance from a reference mask."""
    _validate_profile_settings(bin_width, maximum_distance)
    values = _as_2d_array(image, name="image").astype(float, copy=False)
    reference = _as_2d_array(reference_mask, name="reference_mask")
    _validate_same_shape(reference, values, name="reference_mask")
    included = _resolve_inclusion_mask(
        inclusion_mask, (values.shape[0], values.shape[1]))

    distances = _distance_map(reference, sampling=sampling)
    selected_distances, selected_values = _select_profile_pixels(
        values, distances, included)
    edges, upper_limit = _create_distance_bins(
        selected_distances, bin_width, maximum_distance)
    means, deviations, counts = _aggregate_distance_bins(
        selected_distances, selected_values, edges, upper_limit)

    return DistanceProfile(
        bin_start=edges[:-1],
        bin_end=edges[1:],
        distance=(edges[:-1] + edges[1:]) / 2,
        mean_intensity=means,
        standard_deviation=deviations,
        pixel_count=counts,)
