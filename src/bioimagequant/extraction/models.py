from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
import logging
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from bioimagequant.core.arrays import ArrayData


logger = logging.getLogger(__name__)
ArrayRole = Literal["intensity", "object_labels", "reference"]


@dataclass
class ArrayInputs:
    intensity: ArrayData | None = None
    object_labels: dict[str, ArrayData] = field(default_factory=dict)
    references: dict[str, ArrayData] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.intensity is None:
            return
        for name, labels in self.object_labels.items():
            self._validate_object_labels(name, labels, self.intensity)

    def add_array(self,
                  role: ArrayRole,
                  array: NDArray[Any],
                  axes: str,
                  *,
                  name: str | None = None,
                  channel_labels: Sequence[str] | None = None,
                  ) -> None:
        """Register and validate one extraction input array."""
        data = ArrayData(array=array, axes=axes, channel_labels=channel_labels)

        if role == "intensity":
            for object_name, labels in self.object_labels.items():
                self._validate_object_labels(object_name, labels, data)
            self.intensity = data
            return

        if name is None:
            raise ValueError(
                "Name must be provided for object labels or reference arrays.")

        if role == "object_labels":
            if self.intensity is not None:
                self._validate_object_labels(name, data, self.intensity)
            for reference_name, reference in self.references.items():
                self._validate_reference(reference_name, reference, name, data)
            if name in self.object_labels:
                logger.warning("Overwriting object labels %r.", name)
            self.object_labels[name] = data
            return

        if role == "reference":
            if data.channel_labels is None:
                raise ValueError(
                    f"Reference array {name!r} requires a channel label for "
                    "the output table.")
            if not np.all((data.array == 0) | (data.array == 1)):
                raise ValueError(
                    f"Reference array {name!r} must contain only binary values 0 and 1.")
            for object_name, labels in self.object_labels.items():
                self._validate_reference(name, data, object_name, labels)
            if name in self.references:
                logger.warning("Overwriting reference array %r.", name)
            self.references[name] = data
            return

        raise ValueError(f"Unknown role: {role}")

    @staticmethod
    def _validate_object_labels(name: str,
                                labels: ArrayData,
                                intensity: ArrayData,
                                ) -> None:
        intensity_axes = intensity.axes.removesuffix("C")
        labels_axes = labels.axes.removesuffix("C")
        if intensity_axes != labels_axes:
            raise ValueError(
                f"Intensity axes {intensity.axes!r} are incompatible with "
                f"object labels {name!r} axes {labels.axes!r}. After removing "
                f"channel axes, expected {intensity_axes!r} but received "
                f"{labels_axes!r}.")
        if intensity.frame_count != labels.frame_count:
            raise ValueError(
                f"Intensity array has {intensity.frame_count} frames, but "
                f"object labels {name!r} have {labels.frame_count} frames.")

        intensity_shape = (intensity.array.shape[:-1]
                           if "C" in intensity.axes else intensity.array.shape)
        labels_shape = (labels.array.shape[:-1]
                        if "C" in labels.axes else labels.array.shape)
        if intensity_shape != labels_shape:
            raise ValueError(
                f"Intensity array shape without channels is {intensity_shape}, "
                f"but object labels {name!r} shape without channels is "
                f"{labels_shape}.")

    @staticmethod
    def _validate_reference(reference_name: str,
                            reference: ArrayData,
                            object_name: str,
                            labels: ArrayData,
                            ) -> None:
        if reference.spatial_axes != labels.spatial_axes:
            raise ValueError(
                f"Reference {reference_name!r} spatial axes "
                f"{reference.spatial_axes!r} are incompatible with object labels "
                f"{object_name!r} axes {labels.spatial_axes!r}.")
        if reference.spatial_shape != labels.spatial_shape:
            raise ValueError(
                f"Reference {reference_name!r} spatial shape "
                f"{reference.spatial_shape} does not match object labels "
                f"{object_name!r} shape {labels.spatial_shape}.")
        if reference.frame_count not in {1, labels.frame_count}:
            raise ValueError(
                f"Reference {reference_name!r} has {reference.frame_count} "
                f"frames, but object labels {object_name!r} have "
                f"{labels.frame_count} frames.")
