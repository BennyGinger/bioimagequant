import numpy as np
import pandas as pd
import pytest

from bioimagequant import ExtractData


def _extractor(*,
               interval: float | None = None,
               pixel_size: float | None = None,
               ) -> ExtractData:
    image = np.zeros((2, 2, 4, 4), dtype=np.uint16)
    image[:, 0] = 10
    image[:, 1] = 20
    labels = np.zeros((2, 4, 4), dtype=np.uint16)
    labels[:, 1:3, 1:3] = 1

    extractor = ExtractData(interval=interval, pixel_size=pixel_size)
    extractor.add_intensity(
        image, "TCYX", channel_labels=("GFP", "RFP"))
    extractor.add_labels(labels, "TYX", name="tracking")
    return extractor


def test_parallel_quantification_matches_serial() -> None:
    pd.testing.assert_frame_equal(
        _extractor().quantify(workers=2),
        _extractor().quantify(workers=1),)


def test_quantification_adds_time_in_seconds() -> None:
    dataframe = _extractor(interval=10).quantify()
    assert dataframe.loc[dataframe["frame"] == 1, "time_sec"].eq(0).all()
    assert dataframe.loc[dataframe["frame"] == 2, "time_sec"].eq(10).all()


def test_missing_interval_produces_missing_time() -> None:
    assert _extractor().quantify()["time_sec"].isna().all()


@pytest.mark.parametrize("interval", [-1, np.nan, np.inf])
def test_rejects_invalid_interval(interval: float) -> None:
    with pytest.raises(ValueError, match="interval"):
        ExtractData(interval=interval)


@pytest.mark.parametrize("pixel_size", [0, -1, np.nan, np.inf])
def test_rejects_invalid_pixel_size(pixel_size: float) -> None:
    with pytest.raises(ValueError, match="pixel_size"):
        ExtractData(pixel_size=pixel_size)


def test_quantification_adds_reference_distances() -> None:
    extractor = _extractor(pixel_size=0.5)
    reference = np.zeros((2, 2, 4, 4), dtype=np.uint8)
    reference[:, 0, 1, 1] = 1
    reference[:, 1, 3, 3] = 1
    extractor.add_ref(
        reference,
        "TCYX",
        name="needle",
        channel_labels=("GFP", "RFP"),)

    dataframe = extractor.quantify()

    assert len(dataframe) == 8
    assert set(dataframe["ref_label_name"]) == {"needle"}
    assert set(dataframe["ref_channel"]) == {"GFP", "RFP"}
    np.testing.assert_allclose(
        dataframe["dist_um"], dataframe["dist_pixel"] * 0.5)


def test_empty_reference_frame_produces_missing_distance() -> None:
    extractor = _extractor()
    reference = np.zeros((2, 4, 4), dtype=np.uint8)
    reference[0, 1, 1] = 1
    extractor.add_ref(
        reference, "TYX", name="needle", channel_labels=("GFP",))

    dataframe = extractor.quantify()

    assert dataframe.loc[
        dataframe["frame"] == 1, "dist_pixel"].eq(0).all()
    assert dataframe.loc[
        dataframe["frame"] == 2, "dist_pixel"].isna().all()


def test_static_reference_is_applied_to_every_frame() -> None:
    extractor = _extractor()
    reference = np.zeros((4, 4), dtype=np.uint8)
    reference[1, 1] = 1
    extractor.add_ref(
        reference, "YX", name="needle", channel_labels=("GFP",))

    assert extractor.quantify()["dist_pixel"].eq(0).all()


def test_reference_arrays_must_be_binary() -> None:
    extractor = _extractor()
    reference = np.zeros((2, 4, 4), dtype=np.uint8)
    reference[0, 1, 1] = 2
    with pytest.raises(ValueError, match="binary values"):
        extractor.add_ref(
            reference, "TYX", name="needle", channel_labels=("GFP",))


def test_reference_requires_channel_identity() -> None:
    with pytest.raises(ValueError, match="channel label"):
        _extractor().add_ref(
            np.zeros((2, 4, 4), dtype=np.uint8), "TYX", name="needle")


@pytest.mark.parametrize("workers", [0, -1, True])
def test_quantify_rejects_invalid_worker_count(workers: int) -> None:
    with pytest.raises(ValueError, match="workers"):
        _extractor().quantify(workers=workers)
