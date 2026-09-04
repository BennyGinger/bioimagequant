from typing import Any

import numpy as np
from numpy.typing import NDArray

from bioimagequant.core.validation import as_2d_array, validate_same_shape
from bioimagequant.distance_profile.models import BinningSettings, DistanceProfile


def profile_from_distance_map(
        image: NDArray[Any],
        distances: NDArray[Any],
        *,
        settings: BinningSettings,
        inclusion_mask: NDArray[Any] | None = None,
        ) -> DistanceProfile:
    """
    Bin image intensities using a distance map and validated settings.
    """
    values = as_2d_array(image, name="image").astype(float, copy=False)
    distance_values = as_2d_array(distances, name="distances").astype(float, copy=False)
    validate_same_shape(distance_values, values, name="distances", expected_name="image")
    included = _resolve_inclusion_mask(inclusion_mask, (values.shape[0], values.shape[1]))

    selected_distances, selected_values = _select_pixels(values, distance_values, included)
    edges, means, deviations, counts = _aggregate_bins(selected_distances, selected_values, settings)

    return DistanceProfile(
        bin_start=edges[:-1],
        bin_end=edges[1:],
        distance=(edges[:-1] + edges[1:]) / 2,
        mean_intensity=means,
        standard_deviation=deviations,
        pixel_count=counts,)


def _resolve_inclusion_mask(
        inclusion_mask: NDArray[Any] | None,
        image_shape: tuple[int, int],
        ) -> NDArray[np.bool_]:
    """Return a boolean inclusion mask covering the requested image shape."""
    if inclusion_mask is None:
        return np.ones(image_shape, dtype=bool)
    included = as_2d_array(inclusion_mask, name="inclusion_mask")
    if included.shape != image_shape:
        raise ValueError(f"inclusion_mask must have the same shape as image; got {included.shape} and {image_shape}.")
    return included.astype(bool, copy=False)


def _select_pixels(
        values: NDArray[np.float64],
        distances: NDArray[np.float64],
        included: NDArray[np.bool_],
        ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Select included pixels whose intensity values are finite."""
    selected = included & np.isfinite(values)
    return distances[selected], values[selected]


def _aggregate_bins(
        selected_distances: NDArray[np.float64],
        selected_values: NDArray[np.float64],
        settings: BinningSettings,
        ) -> tuple[
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.int64],
        ]:
    """Create bins, assign pixels, and calculate their statistics."""
    edges, upper_limit = _create_bins(selected_distances, settings)
    bin_count = edges.size - 1
    indices, values = _assign_bins(
        selected_distances,
        selected_values,
        edges,
        upper_limit,)
    means, deviations, counts = _calculate_bin_statistics(
        indices, values, bin_count)
    return edges, means, deviations, counts


def _create_bins(
        selected_distances: NDArray[np.float64],
        settings: BinningSettings,
        ) -> tuple[NDArray[np.float64], float]:
    """Create equally spaced bin edges covering the requested distance."""
    if settings.maximum_distance is None:
        if selected_distances.size == 0:
            raise ValueError(
                "No finite image pixels are selected for profiling.")
        upper_limit = float(selected_distances.max())
    else:
        upper_limit = float(settings.maximum_distance)

    bin_count = int(np.floor(
        upper_limit / settings.bin_width)) + 1
    edges = np.arange(bin_count + 1, dtype=float) * settings.bin_width
    return edges, upper_limit


def _assign_bins(
        selected_distances: NDArray[np.float64],
        selected_values: NDArray[np.float64],
        edges: NDArray[np.float64],
        upper_limit: float,
        ) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    """Keep values within range and assign each one to a bin index."""
    within_range = selected_distances <= upper_limit
    distances = selected_distances[within_range]
    values = selected_values[within_range]

    bin_width = float(edges[1] - edges[0])
    bin_count = edges.size - 1
    indices = np.floor(distances / bin_width).astype(np.int64)
    indices = np.minimum(indices, bin_count - 1)
    return indices, values


def _calculate_bin_statistics(
        indices: NDArray[np.int64],
        values: NDArray[np.float64],
        bin_count: int,
        ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.int64]]:
    """Calculate intensity statistics for each assigned bin."""
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
