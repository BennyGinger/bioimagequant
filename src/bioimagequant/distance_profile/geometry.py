from typing import Any, TypeAlias, cast

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import distance_transform_edt

from bioimagequant.core.validation import as_2d_array


Sampling: TypeAlias = float | tuple[float, float]


def distance_map(reference_mask: NDArray[Any],
                 *,
                 sampling: Sampling = 1.0,
                 ) -> NDArray[np.float64]:
    """
    Return each pixel's Euclidean distance to a 2D reference mask.
    """
    reference = as_2d_array(reference_mask, name="reference_mask")
    if not np.any(reference):
        raise ValueError(
            "reference_mask must contain at least one reference pixel.")
    return cast(
        NDArray[np.float64],
        distance_transform_edt(
            np.logical_not(reference.astype(bool, copy=False)),
            sampling=_validate_sampling(sampling)),)


def _validate_sampling(sampling: Sampling) -> float | tuple[float, float]:
    """Normalize and validate scalar or Y/X pixel sampling."""
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
