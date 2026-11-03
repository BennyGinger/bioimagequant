"""Label-preserving, frame-local cell compartment geometry."""

from dataclasses import dataclass
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import distance_transform_edt, find_objects
from skimage.segmentation import expand_labels

from bioimagequant.core.scales import validate_optional_scale
from bioimagequant.core.validation import as_2d_array


@dataclass(frozen=True)
class CompartmentSettings:
    """
    Configure membrane thickness and optional full-cell expansion, in pixels.

    Thickness must be positive; dilation may be zero. Distances are Euclidean.
    """

    membrane_thickness: float = 6.0
    dilation_radius: float = 0.0

    def __post_init__(self) -> None:
        """
        Validate the compartment distances using the shared scale rules.
        """
        validate_optional_scale(self.membrane_thickness, name="membrane_thickness")
        validate_optional_scale(self.dilation_radius, name="dilation_radius", allow_zero=True)


@dataclass(frozen=True)
class CompartmentGeometry:
    """
    Prepared full-cell geometry reusable while membrane thickness changes.
    """

    full: NDArray[Any]
    cells: tuple[tuple[int, tuple[slice, ...], NDArray[np.float64], tuple[int, ...]], ...]

    def masks(self, membrane_thickness: float) -> dict[str, NDArray[Any]]:
        """
        Threshold cached distances, preserving one interior cytoplasm pixel.
        """
        validate_optional_scale(membrane_thickness, name="membrane_thickness")
        cyto = np.zeros_like(self.full)
        for label, bounds, distances, fallback in self.cells:
            retained = distances > membrane_thickness
            if not retained.any():
                retained[fallback] = True
            cyto[bounds][retained] = label
        return {"full": self.full, "mb": np.where(cyto != 0, 0, self.full), "cyto": cyto}


def prepare_compartments(labels: NDArray[Any], dilation_radius: float = 0.0) -> CompartmentGeometry:
    """
    Expand one labelled 2D frame and cache independent cropped cell distances.

    Neighbouring labels and pixels outside the image count as background for
    each cell. The source array is never modified or renumbered.
    """
    labels = as_2d_array(labels, name="labels")
    if not np.issubdtype(labels.dtype, np.integer) or np.any(labels < 0):
        raise ValueError("labels must contain non-negative integers.")
    validate_optional_scale(dilation_radius, name="dilation_radius", allow_zero=True)
    full = labels
    if dilation_radius:
        # expand_labels supports fractional distances; its bundled annotation says int.
        full = expand_labels(labels, cast(Any, dilation_radius))
    cells = []
    for label, bounds in enumerate(find_objects(full), start=1):
        if bounds is None:
            continue
        cell = full[bounds] == label
        distances = cast(NDArray[np.float64], distance_transform_edt(np.pad(cell, 1)))[1:-1, 1:-1]
        coordinates = np.argwhere(cell)
        centroid = coordinates.mean(axis=0)
        nearest = np.argmin(np.sum((coordinates - centroid) ** 2, axis=1))
        fallback = tuple(int(value) for value in coordinates[nearest])
        cells.append((label, bounds, distances, fallback))
    return CompartmentGeometry(full=full, cells=tuple(cells))


def compartment_masks(labels: NDArray[Any],
                      settings: CompartmentSettings,
                      ) -> dict[str, NDArray[Any]]:
    """
    Return full, membrane and cytoplasm masks using the shared prepared geometry.
    """
    return prepare_compartments(labels, settings.dilation_radius).masks(settings.membrane_thickness)
