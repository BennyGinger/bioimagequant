"""Compartment geometry and the public quantification contract."""

import numpy as np
import pandas as pd
import pytest
from numpy.typing import NDArray
from skimage.morphology import disk, erosion
from skimage.segmentation import expand_labels

from bioimagequant import CompartmentSettings, ExtractData, compartment_masks


@pytest.mark.parametrize("radius", [1, 3, 6])
def test_distance_erosion_matches_isolated_cells_at_contacts_and_borders(radius: int) -> None:
    """
    Preserve labels and reproduce disk erosion, treating image edges as background.
    """
    labels = np.zeros((60, 70), dtype=np.uint32)
    labels[:20, :20] = 4
    labels[:20, 20:40] = 90
    labels[25:59, 7:40] = 127
    labels[29:32, 11:14] = 0
    original = labels.copy()
    masks = compartment_masks(labels, CompartmentSettings(membrane_thickness=radius))
    expected = np.zeros_like(labels)
    for label in (4, 90, 127):
        retained = erosion(labels == label, disk(radius), mode="min")
        expected[retained] = label
    np.testing.assert_array_equal(masks["cyto"], expected)
    np.testing.assert_array_equal(masks["full"], masks["cyto"] + masks["mb"])
    assert not np.any((masks["cyto"] != 0) & (masks["mb"] != 0))
    np.testing.assert_array_equal(labels, original)


def test_dilation_defines_full_and_never_merges_neighbouring_labels() -> None:
    """
    Expand the full region first and retain identities at competing boundaries.
    """
    labels = np.zeros((15, 15), dtype=np.uint16)
    labels[0:4, 1:4] = 7
    labels[0:4, 7:10] = 19
    settings = CompartmentSettings(membrane_thickness=1, dilation_radius=3)
    masks = compartment_masks(labels, settings)
    np.testing.assert_array_equal(masks["full"], expand_labels(labels, 3))
    assert set(np.unique(masks["full"])) == {0, 7, 19}
    np.testing.assert_array_equal(masks["full"][labels > 0], labels[labels > 0])
    np.testing.assert_array_equal(masks["full"], masks["cyto"] + masks["mb"])
    for name, repeated in compartment_masks(labels, settings).items():
        np.testing.assert_array_equal(masks[name], repeated)


def test_fallback_uses_nearest_interior_pixel_when_centroid_is_outside_cell() -> None:
    """
    Reserve one valid cytoplasm pixel and exclude it from the membrane.
    """
    labels = np.zeros((9, 9), dtype=np.uint16)
    labels[1, 1:8] = labels[7, 1:8] = 24
    labels[1:8, 1] = 24
    coordinates = np.argwhere(labels == 24)
    centroid = coordinates.mean(axis=0)
    assert labels[tuple(np.round(centroid).astype(int))] == 0
    masks = compartment_masks(labels, CompartmentSettings(membrane_thickness=20))
    cyto = np.argwhere(masks["cyto"] == 24)
    assert len(cyto) == 1
    distances = np.sum((coordinates - centroid) ** 2, axis=1)
    assert np.sum((cyto[0] - centroid) ** 2) == pytest.approx(distances.min())
    assert masks["mb"][tuple(cyto[0])] == 0
    np.testing.assert_array_equal(masks["full"], masks["cyto"] + masks["mb"])


@pytest.mark.parametrize("name, value", [
    ("membrane_thickness", 0), ("membrane_thickness", -1),
    ("membrane_thickness", np.inf), ("membrane_thickness", np.nan),
    ("dilation_radius", -1), ("dilation_radius", np.inf), ("dilation_radius", np.nan),
])
def test_compartment_distances_are_validated(name: str, value: float) -> None:
    """
    Reject invalid radii at the standalone numerical API boundary.
    """
    with pytest.raises(ValueError, match=name):
        CompartmentSettings(**{name: value})


def _extractor() -> tuple[ExtractData, NDArray[np.uint16]]:
    """
    Prepare two frames, two object channels and two signed intensity channels.
    """
    image = np.arange(2 * 2 * 12 * 12, dtype=np.float32).reshape(2, 2, 12, 12) - 40
    labels = np.zeros((2, 2, 12, 12), dtype=np.uint16)
    labels[:, 0, 0:5, 0:5] = 7
    labels[:, 1, 6:11, 6:11] = 7
    extractor = ExtractData(interval=2, pixel_size=0.5)
    extractor.add_intensity(image, "TCYX", channel_labels=("GFP", "RFP"))
    extractor.add_labels(labels, "TCYX", name="tracking", channel_labels=("A", "B"))
    reference = np.zeros((12, 12), dtype=np.uint8)
    reference[0, 11] = 1
    extractor.add_ref(reference, "YX", name="needle", channel_labels=("reference",))
    return extractor, labels


def test_quantification_reuses_dilated_full_positions_and_preserves_weighted_intensity() -> None:
    """
    Emit six groups, sharing full centroids/distances and preserving signed sums.
    """
    extractor, labels = _extractor()
    original = labels.copy()
    settings = CompartmentSettings(membrane_thickness=2, dilation_radius=2)
    table = extractor.quantify(compartments=settings, additional_properties="eccentricity")
    assert len(table) == 2 * 2 * 2 * 3
    assert set(table["compartment"]) == {"full", "mb", "cyto"}
    keys = ["frame", "object_name", "object_channel", "label", "intensity_channel"]
    assert not table.duplicated([*keys, "compartment"]).any()
    for _, group in table.groupby(keys):
        rows = group.set_index("compartment")
        assert rows["centroid_x"].nunique() == rows["centroid_y"].nunique() == 1
        assert rows["dist_pixel"].nunique() == 1
        assert rows.loc["full", "area"] == rows.loc[["mb", "cyto"], "area"].sum()
        weighted = rows["area"] * rows["intensity_mean"]
        assert weighted["full"] == pytest.approx(weighted["mb"] + weighted["cyto"])
    assert "eccentricity" in table
    assert (table["intensity_mean"] < 0).any()
    # Expansion at the image border moves the centroid; it must be measured afterwards.
    expanded = expand_labels(labels[0, 0], 2)
    expected = np.argwhere(expanded == 7).mean(axis=0)
    first = table.loc[(table["frame"] == 1) & (table["object_channel"] == "A")]
    np.testing.assert_allclose(first["centroid_y"], expected[0])
    np.testing.assert_allclose(first["centroid_x"], expected[1])
    assert expected[0] != np.argwhere(labels[0, 0] == 7).mean(axis=0)[0]
    np.testing.assert_array_equal(labels, original)
    parallel = extractor.quantify(compartments=settings,
                                   additional_properties="eccentricity",
                                   workers=2,
                                   )
    pd.testing.assert_frame_equal(table, parallel)


def test_singleton_fallback_keeps_empty_membrane_row_with_zero_measurements() -> None:
    """
    Keep all three rows even when the entire full cell becomes cytoplasm.
    """
    extractor = ExtractData()
    labels = np.zeros((5, 5), dtype=np.uint16)
    labels[2, 3] = 24
    extractor.add_intensity(np.full((5, 5), -4.0), "YX")
    extractor.add_labels(labels, "YX", name="segmentation")
    table = extractor.quantify(compartments=CompartmentSettings()).set_index("compartment")
    assert table.loc["cyto", "area"] == 1
    assert table.loc["cyto", "intensity_mean"] == -4
    assert table.loc["mb", "area"] == table.loc["mb", "intensity_mean"] == 0
    np.testing.assert_allclose(table["centroid_y"], 2)
    np.testing.assert_allclose(table["centroid_x"], 3)


def test_disabled_and_empty_extraction_always_include_compartment_column() -> None:
    """
    Preserve the full-only contract and stable columns for empty input frames.
    """
    extractor, labels = _extractor()
    assert set(extractor.quantify()["compartment"]) == {"full"}
    labels[:] = 0
    table = extractor.quantify(compartments=CompartmentSettings())
    assert table.empty and "compartment" in table


def test_geometry_is_constructed_once_per_frame_and_object_channel(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Avoid repeating morphology for individual intensity channels or properties.
    """
    calls = 0

    def record(labels: NDArray[np.uint16], settings: CompartmentSettings) -> dict[str, NDArray[np.uint16]]:
        """
        Count geometry calls while executing the real implementation.
        """
        nonlocal calls
        calls += 1
        return compartment_masks(labels, settings)

    monkeypatch.setattr("bioimagequant.extraction.regions.compartment_masks", record)
    extractor, _ = _extractor()
    extractor.quantify(compartments=CompartmentSettings(), additional_properties=["eccentricity", "bbox"])
    assert calls == 4
