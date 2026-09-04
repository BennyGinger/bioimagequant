from typing import Any

from numpy.typing import NDArray

from bioimagequant.core.validation import as_2d_array, validate_same_shape
from bioimagequant.distance_profile.binning import profile_from_distance_map
from bioimagequant.distance_profile.geometry import Sampling, distance_map
from bioimagequant.distance_profile.models import BinningSettings, DistanceProfile


def distance_profile(image: NDArray[Any],
                     reference_mask: NDArray[Any],
                     *,
                     inclusion_mask: NDArray[Any] | None = None,
                     bin_width: float = 5.0,
                     sampling: Sampling = 1.0,
                     maximum_distance: float | None = None,
                     ) -> DistanceProfile:
    """Calculate a binned intensity profile from a 2D reference mask."""
    settings = BinningSettings(bin_width, maximum_distance)
    values = as_2d_array(image, name="image")
    reference = as_2d_array(reference_mask, name="reference_mask")
    validate_same_shape(
        reference, values, name="reference_mask", expected_name="image")

    distances = distance_map(reference, sampling=sampling)
    return profile_from_distance_map(
        values,
        distances,
        settings=settings,
        inclusion_mask=inclusion_mask,)
