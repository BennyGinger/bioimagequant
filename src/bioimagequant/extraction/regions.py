from collections.abc import Sequence
from typing import Any

from numpy.typing import NDArray
import pandas as pd
from skimage.measure import regionprops_table

from bioimagequant.core.arrays import ArrayData
from bioimagequant.extraction.compartments import CompartmentSettings, compartment_masks
from bioimagequant.extraction.distances import reference_distance_table
from bioimagequant.extraction.formatting import format_region_table


DEFAULT_REGION_PROPERTIES = (
    "area",
    "centroid",
    "intensity_mean",
    "label",
    "perimeter",
    "solidity",)


def extract_region_frame(image: NDArray[Any],
                         labels: NDArray[Any],
                         *,
                         additional_properties: str | Sequence[str] | None = None,
                         ) -> pd.DataFrame:
    """Extract region properties from one labeled image frame."""
    properties = _resolve_region_properties(additional_properties)
    result = regionprops_table(
        label_image=labels,
        intensity_image=image,
        properties=properties,
        separator="_",)
    return pd.DataFrame(result)


def quantify_region_frame(frame_index: int,
                          intensity: ArrayData,
                          labels: ArrayData,
                          object_name: str,
                          object_channel: str | None,
                          additional_properties: str | Sequence[str] | None,
                          references: dict[str, ArrayData],
                          compartments: CompartmentSettings | None = None,
                          ) -> pd.DataFrame:
    """
    Quantify one frame, sharing full-cell positions across its compartments.
    """
    image_frame = intensity.frame(frame_index)
    label_frame = labels.frame(frame_index)
    masks = (compartment_masks(label_frame.array, compartments)
             if compartments is not None else {"full": label_frame.array})
    full_table = extract_region_frame(image_frame.array,
                                      masks["full"],
                                      additional_properties=additional_properties,
                                      )
    tables: list[pd.DataFrame] = []
    for name, mask in masks.items():
        table = (full_table.copy() if name == "full"
                 else _extract_compartment(image_frame.array, mask, full_table, additional_properties))
        table = format_region_table(table,
                                    channel_labels=image_frame.channel_labels,
                                    label_axes=label_frame.axes,
                                    )
        table["compartment"] = name
        tables.append(table)
    frame_df = pd.concat(tables, ignore_index=True)

    reference_df = reference_distance_table(
        masks["full"],
        references,
        frame_index=frame_index,)
    if not reference_df.empty:
        frame_df = frame_df.merge(reference_df, on="label", how="left")

    frame_df["frame"] = frame_index + 1
    frame_df["object_name"] = object_name
    frame_df["object_channel"] = object_channel
    return frame_df


def _extract_compartment(image: NDArray[Any],
                         mask: NDArray[Any],
                         full_table: pd.DataFrame,
                         additional_properties: str | Sequence[str] | None,
                         ) -> pd.DataFrame:
    """
    Measure a compartment while retaining parent centroids and empty regions.

    A singleton full cell has no membrane after the cytoplasm fallback. Its
    membrane area, perimeter and intensities are zero; undefined extra shape
    properties remain missing.
    """
    measured = extract_region_frame(image, mask, additional_properties=additional_properties)
    table = measured.set_index("label").reindex(full_table["label"]).reset_index()
    absent = ~table["label"].isin(measured["label"])
    zero_columns = [name for name in table.columns
                    if name in {"area", "perimeter"} or name.startswith("intensity_")]
    table.loc[absent, zero_columns] = 0
    for axis in range(mask.ndim):
        name = f"centroid_{axis}"
        table[name] = full_table[name].to_numpy()
    return table


def _resolve_region_properties(
        additional_properties: str | Sequence[str] | None = None,
        ) -> tuple[str, ...]:
    """Combine default and requested region properties without duplicates."""
    if additional_properties is None:
        return DEFAULT_REGION_PROPERTIES
    if isinstance(additional_properties, str):
        additional_properties = (additional_properties,)
    return tuple(dict.fromkeys(
        (*DEFAULT_REGION_PROPERTIES, *additional_properties)))
