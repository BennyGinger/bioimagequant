from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from joblib import Parallel, delayed
from numpy.typing import NDArray
import numpy as np
import pandas as pd

from bioimagequant.core.scales import validate_optional_scale
from bioimagequant.extraction.models import ArrayInputs
from bioimagequant.extraction.regions import quantify_region_frame


MEMMAP_THRESHOLD = "10M"


@dataclass
class ExtractData:
    """Arrays and acquisition metadata used for region quantification."""

    array_data: ArrayInputs = field(default_factory=ArrayInputs)
    interval: float | None = None
    pixel_size: float | None = None

    def __post_init__(self) -> None:
        validate_optional_scale(
            self.interval, name="interval", allow_zero=True)
        validate_optional_scale(self.pixel_size, name="pixel_size")

    def add_intensity(self,
                      array: NDArray[Any],
                      axes: str,
                      *,
                      channel_labels: Sequence[str] | None = None,
                      ) -> None:
        """Register the intensity array to quantify."""
        self.array_data.add_array(
            role="intensity",
            array=array,
            axes=axes,
            channel_labels=channel_labels,)

    def add_labels(self,
                   array: NDArray[Any],
                   axes: str,
                   *,
                   name: str,
                   channel_labels: Sequence[str] | None = None,
                   ) -> None:
        """Register one named object-label array."""
        self.array_data.add_array(
            role="object_labels",
            array=array,
            axes=axes,
            name=name,
            channel_labels=channel_labels,)

    def add_ref(self,
                array: NDArray[Any],
                axes: str,
                *,
                name: str,
                channel_labels: Sequence[str] | None = None,
                ) -> None:
        """Register one named binary reference-mask array."""
        self.array_data.add_array(
            role="reference",
            array=array,
            axes=axes,
            name=name,
            channel_labels=channel_labels,)

    def quantify(self,
                 additional_properties: str | Sequence[str] | None = None,
                 *,
                 workers: int = 1,
                 ) -> pd.DataFrame:
        """Quantify every object-label channel and frame."""
        if isinstance(workers, bool) or workers < 1:
            raise ValueError(
                "workers must be an integer greater than or equal to 1.")

        intensity = self.array_data.intensity
        if intensity is None:
            raise ValueError(
                "Add an intensity array before quantification.")
        if not self.array_data.object_labels:
            raise ValueError(
                "Add at least one object-label array before quantification.")

        jobs = (
            (frame_index,
             intensity,
             label_stack,
             name,
             object_channel,
             additional_properties,
             self.array_data.references,)
            for name, labels in self.array_data.object_labels.items()
            for object_channel, label_stack in labels.iter_channels()
            for frame_index in range(label_stack.frame_count)
        )

        if workers == 1:
            results = [quantify_region_frame(*job) for job in jobs]
        else:
            results = Parallel(
                n_jobs=workers,
                backend="loky",
                max_nbytes=MEMMAP_THRESHOLD,
                mmap_mode="r",
            )(
                delayed(quantify_region_frame)(*job)
                for job in jobs)

        if not results:
            return pd.DataFrame()

        dataframe = pd.concat(results, ignore_index=True)
        dataframe["time_sec"] = (
            (dataframe["frame"] - 1) * self.interval
            if self.interval is not None else pd.NA)

        if "dist_pixel" in dataframe:
            dataframe["dist_um"] = (
                dataframe["dist_pixel"] * self.pixel_size
                if self.pixel_size is not None else np.nan)

        return dataframe
