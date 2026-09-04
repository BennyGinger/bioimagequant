from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from bioimagequant.core.arrays import ArrayData
from bioimagequant.distance_profile.distance import distance_profile
from bioimagequant.distance_profile.models import ProfileResult


def calculate_profiles(*,
                       intensity: ArrayData,
                       references: dict[str, ArrayData],
                       rois: dict[str, ArrayData],
                       bin_width: float,
                       maximum_bins: int | None,
                       ) -> list[ProfileResult]:
    """Calculate every frame, channel, reference and ROI combination."""
    output: list[ProfileResult] = []
    cutoff = _maximum_distance(bin_width, maximum_bins)

    for reference_name, reference in references.items():
        for reference_channel, reference_stack in reference.iter_channels():
            for roi_item in (None, *_roi_channels(rois)):
                roi_name, roi_channel, roi_stack = (
                    roi_item if roi_item is not None else (None, None, None))

                for intensity_channel, intensity_stack in intensity.iter_channels():
                    for frame_index in range(intensity.frame_count):
                        image = intensity_stack.frame(frame_index).array
                        ref = _mask_frame(reference_stack, frame_index)
                        roi = (_mask_frame(roi_stack, frame_index)
                               if roi_stack is not None else None)
                        profile = (distance_profile(
                            image,
                            ref,
                            inclusion_mask=roi,
                            bin_width=bin_width,
                            maximum_distance=cutoff,)
                            if roi is None or np.any(roi) else None)
                        output.append(ProfileResult(
                            frame_index,
                            intensity_channel,
                            reference_name,
                            reference_channel,
                            roi_name,
                            roi_channel,
                            profile,))
    return output


def _roi_channels(
        rois: dict[str, ArrayData],
        ) -> tuple[tuple[str, str | None, ArrayData], ...]:
    return tuple(
        (name, channel, stack)
        for name, roi in rois.items()
        for channel, stack in roi.iter_channels())


def _mask_frame(mask: ArrayData, index: int) -> NDArray[np.bool_]:
    frame_index = 0 if mask.frame_count == 1 else index
    return mask.frame(frame_index).array.astype(bool, copy=False)


def _maximum_distance(bin_width: float,
                      maximum_bins: int | None,
                      ) -> float | None:
    if maximum_bins is None:
        return None
    return float(np.nextafter(bin_width * maximum_bins, -np.inf))
