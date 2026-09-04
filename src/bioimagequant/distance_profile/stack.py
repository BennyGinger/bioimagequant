from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from bioimagequant.core.arrays import ArrayData
from bioimagequant.core.execution import run_jobs
from bioimagequant.distance_profile.binning import profile_from_distance_map
from bioimagequant.distance_profile.geometry import distance_map
from bioimagequant.distance_profile.models import BinningSettings, ProfileResult


def calculate_profiles(*,
                       intensity: ArrayData,
                       references: dict[str, ArrayData],
                       rois: dict[str, ArrayData],
                       settings: BinningSettings,
                       workers: int,
                       ) -> list[ProfileResult]:
    """
    Calculate profiles using independent reference-frame jobs.
    """
    roi_channels = _roi_channels(rois)
    jobs = (
        (frame_index,
         intensity,
         reference_name,
         reference_channel,
         reference_stack,
         roi_channels,
         settings,)
        for reference_name, reference in references.items()
        for reference_channel, reference_stack in reference.iter_channels()
        for frame_index in range(intensity.frame_count)
    )
    grouped_results = run_jobs(
        _calculate_reference_frame, jobs, workers=workers)
    return [
        result
        for frame_results in grouped_results
        for result in frame_results]


def _calculate_reference_frame(
        frame_index: int,
        intensity: ArrayData,
        reference_name: str,
        reference_channel: str | None,
        reference: ArrayData,
        roi_channels: tuple[tuple[str, str | None, ArrayData], ...],
        settings: BinningSettings,
        ) -> list[ProfileResult]:
    """
    Profile all intensity and ROI variants against one distance map.
    """
    reference_frame = _mask_frame(reference, frame_index)
    distances = distance_map(reference_frame)
    intensity_frames = tuple(
        (channel, stack.frame(frame_index).array)
        for channel, stack in intensity.iter_channels())
    roi_frames = (
        (None, None, None),
        *((name, channel, _mask_frame(stack, frame_index))
          for name, channel, stack in roi_channels),)

    return [
        ProfileResult(
            frame_index,
            intensity_channel,
            reference_name,
            reference_channel,
            roi_name,
            roi_channel,
            (profile_from_distance_map(
                image,
                distances,
                settings=settings,
                inclusion_mask=roi,)
             if roi is None or np.any(roi) else None),)
        for roi_name, roi_channel, roi in roi_frames
        for intensity_channel, image in intensity_frames]


def _roi_channels(
        rois: dict[str, ArrayData],
        ) -> tuple[tuple[str, str | None, ArrayData], ...]:
    """Flatten named ROI arrays into their individual channel stacks."""
    return tuple(
        (name, channel, stack)
        for name, roi in rois.items()
        for channel, stack in roi.iter_channels())


def _mask_frame(mask: ArrayData, index: int) -> NDArray[np.bool_]:
    """Return a binary frame, reusing frame zero for a static mask."""
    frame_index = 0 if mask.frame_count == 1 else index
    return mask.frame(frame_index).array.astype(bool, copy=False)
