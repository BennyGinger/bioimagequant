from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class ArrayData:
    """A 2D spatial array with explicit axes and channel identities."""

    array: NDArray[Any]
    axes: str
    channel_labels: Sequence[str] | None = None

    def __post_init__(self) -> None:
        """Validate metadata and normalize the channel axis and labels."""
        axes = self.axes.upper()
        array = np.asarray(self.array)

        if len(axes) != array.ndim:
            raise ValueError(
                f"Axes {axes!r} do not match array shape {array.shape}.")
        if len(set(axes)) != len(axes):
            raise ValueError(f"Axes must be unique; got {axes!r}.")
        if set(axes) - {"T", "C", "Y", "X"}:
            raise ValueError(
                "Bioimagequant supports only T, C, Y and X axes. "
                "Project Z before adding arrays.")
        if "Y" not in axes or "X" not in axes:
            raise ValueError(
                f"Arrays require Y and X axes; got {axes!r}.")

        labels = (tuple(self.channel_labels)
                  if self.channel_labels is not None else None)
        if "C" in axes:
            channel_axis = axes.index("C")
            if labels is None:
                raise ValueError(
                    "Channel labels are required when axes contain C.")
            if len(labels) != array.shape[channel_axis]:
                raise ValueError(
                    f"Received {len(labels)} channel labels for "
                    f"{array.shape[channel_axis]} channels.")
            if channel_axis != array.ndim - 1:
                array = np.moveaxis(array, channel_axis, -1)
                axes = axes.replace("C", "") + "C"
        elif labels is not None and len(labels) != 1:
            raise ValueError(
                "An array without C accepts at most one channel label.")

        object.__setattr__(self, "array", array)
        object.__setattr__(self, "axes", axes)
        object.__setattr__(self, "channel_labels", labels)

    @property
    def frame_count(self) -> int:
        """Return the number of time frames represented by the array."""
        return self.array.shape[self.axes.index("T")] if "T" in self.axes else 1

    @property
    def channel_count(self) -> int:
        """Return the number of image channels represented by the array."""
        return self.array.shape[-1] if "C" in self.axes else 1

    @property
    def spatial_axes(self) -> str:
        """Return the axes remaining after time and channel are removed."""
        return "".join(axis for axis in self.axes if axis not in {"T", "C"})

    @property
    def spatial_shape(self) -> tuple[int, ...]:
        """Return the shape corresponding to the spatial axes."""
        return tuple(
            size for axis, size in zip(self.axes, self.array.shape, strict=True)
            if axis not in {"T", "C"})

    def frame(self, index: int) -> ArrayData:
        """Return one frame as an ``ArrayData`` view."""
        if "T" not in self.axes:
            if index != 0:
                raise IndexError("A static array only has frame index 0.")
            return self
        if not 0 <= index < self.frame_count:
            raise IndexError(
                f"Frame {index} is outside 0 to {self.frame_count - 1}.")
        return ArrayData(
            array=np.take(self.array, index, axis=self.axes.index("T")),
            axes=self.axes.replace("T", ""),
            channel_labels=self.channel_labels,)

    def iter_frames(self) -> Iterator[tuple[int, ArrayData]]:
        """Iterate over frame indices and their ``ArrayData`` views."""
        for index in range(self.frame_count):
            yield index, self.frame(index)

    def channel(self, index: int) -> tuple[str | None, ArrayData]:
        """Return one channel and its identity as an ``ArrayData`` view."""
        if "C" not in self.axes:
            if index != 0:
                raise IndexError("An array without C only has channel index 0.")
            label = (self.channel_labels[0]
                     if self.channel_labels is not None else None)
            return label, self
        if not 0 <= index < self.channel_count:
            raise IndexError(
                f"Channel {index} is outside 0 to {self.channel_count - 1}.")
        assert self.channel_labels is not None
        label = self.channel_labels[index]
        return label, ArrayData(
            array=self.array[..., index],
            axes=self.axes.removesuffix("C"),
            channel_labels=(label,),)

    def iter_channels(self) -> Iterator[tuple[str | None, ArrayData]]:
        """Iterate over channel identities and their ``ArrayData`` views."""
        for index in range(self.channel_count):
            yield self.channel(index)
