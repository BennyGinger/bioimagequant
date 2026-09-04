from typing import Any

import numpy as np
import pandas as pd

from bioimagequant.distance_profile.models import ProfileResult


def profiles_to_table(profiles: list[ProfileResult],
                      *,
                      bin_width: float,
                      interval: float | None,
                      pixel_size: float | None,
                      ) -> pd.DataFrame:
    """Convert frame profiles into an aligned long-form table."""
    completed = [item.profile for item in profiles if item.profile is not None]
    if not completed:
        raise ValueError("Distance profiling produced no populated results.")

    bin_count = max(profile.pixel_count.size for profile in completed)
    records: list[dict[str, Any]] = []

    for item in profiles:
        for bin_index in range(bin_count):
            start = bin_index * bin_width
            end = start + bin_width
            centre = (start + end) / 2
            profile = item.profile

            if profile is not None and bin_index < profile.pixel_count.size:
                mean = profile.mean_intensity[bin_index]
                deviation = profile.standard_deviation[bin_index]
                count = profile.pixel_count[bin_index]
            else:
                mean, deviation, count = np.nan, np.nan, 0

            records.append({
                "frame": item.frame_index + 1,
                "time_sec": (item.frame_index * interval
                             if interval is not None else np.nan),
                "channel": item.intensity_channel,
                "reference": item.reference_name,
                "reference_channel": item.reference_channel,
                "roi": item.roi_name,
                "roi_channel": item.roi_channel,
                "bin": bin_index,
                "dist_start_px": start,
                "dist_end_px": end,
                "dist_px": centre,
                "dist_start_um": (start * pixel_size
                                  if pixel_size is not None else np.nan),
                "dist_end_um": (end * pixel_size
                                if pixel_size is not None else np.nan),
                "dist_um": (centre * pixel_size
                            if pixel_size is not None else np.nan),
                "mean_intensity": mean,
                "std_intensity": deviation,
                "pixel_count": count,})

    return pd.DataFrame.from_records(records)
