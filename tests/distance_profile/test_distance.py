import numpy as np
import pytest

from bioimagequant.distance_profile.binning import profile_from_distance_map
from bioimagequant.distance_profile.distance import distance_profile
from bioimagequant.distance_profile.geometry import distance_map
from bioimagequant.distance_profile.models import BinningSettings


def test_distance_map_uses_arbitrary_reference_shape() -> None:
    reference = np.zeros((5, 5), dtype=bool)
    reference[2, 1:4] = True
    result = distance_map(reference)
    np.testing.assert_allclose(result[2, 1:4], 0)
    assert result[0, 2] == pytest.approx(2)
    assert result[2, 0] == pytest.approx(1)


def test_distance_map_respects_anisotropic_sampling() -> None:
    reference = np.zeros((3, 3), dtype=bool)
    reference[1, 1] = True
    result = distance_map(reference, sampling=(2, 0.5))
    assert result[0, 1] == pytest.approx(2)
    assert result[1, 0] == pytest.approx(0.5)


def test_distance_profile_bins_included_pixels() -> None:
    result = distance_profile(
        np.array([[10, 20, 30, 40, 50]], dtype=float),
        np.array([[1, 0, 0, 0, 0]], dtype=bool),
        inclusion_mask=np.array([[0, 1, 1, 1, 0]], dtype=bool),
        bin_width=2,)
    np.testing.assert_array_equal(result.pixel_count, [1, 2])
    np.testing.assert_allclose(result.mean_intensity, [20, 35])
    np.testing.assert_allclose(result.standard_deviation, [0, 5])
    np.testing.assert_allclose(result.distance, [1, 3])


def test_distance_profile_uses_whole_image_without_roi() -> None:
    result = distance_profile(
        np.array([[1, 3, 5]], dtype=float),
        np.array([[0, 1, 0]], dtype=bool),
        bin_width=1,)
    np.testing.assert_array_equal(result.pixel_count, [1, 2])
    np.testing.assert_allclose(result.mean_intensity, [3, 3])


def test_precomputed_distance_map_matches_complete_profile() -> None:
    image = np.array([[1, 3, 5]], dtype=float)
    reference = np.array([[0, 1, 0]], dtype=bool)

    complete = distance_profile(image, reference, bin_width=1)
    precomputed = profile_from_distance_map(
        image,
        distance_map(reference),
        settings=BinningSettings(bin_width=1),)

    np.testing.assert_allclose(
        precomputed.mean_intensity, complete.mean_intensity)
    np.testing.assert_array_equal(
        precomputed.pixel_count, complete.pixel_count)


def test_distance_profile_ignores_non_finite_intensities() -> None:
    result = distance_profile(
        np.array([[1, np.nan, 5]], dtype=float),
        np.array([[1, 0, 0]], dtype=bool),
        bin_width=1,)
    np.testing.assert_array_equal(result.pixel_count, [1, 0, 1])
    np.testing.assert_allclose(result.mean_intensity[[0, 2]], [1, 5])
    assert np.isnan(result.mean_intensity[1])


@pytest.mark.parametrize(
    ("image", "reference", "message"),
    [
        (np.zeros((2, 2, 2)), np.zeros((2, 2)), "image must be a 2D array"),
        (np.zeros((2, 2)), np.zeros((3, 3)), "same shape as image"),
        (np.zeros((2, 2)), np.zeros((2, 2)), "at least one reference pixel"),
    ],)
def test_distance_profile_rejects_invalid_inputs(
        image: np.ndarray,
        reference: np.ndarray,
        message: str,
        ) -> None:
    with pytest.raises(ValueError, match=message):
        distance_profile(image, reference)


@pytest.mark.parametrize(
    ("bin_width", "maximum_distance", "message"),
    [
        (0, None, "bin_width"),
        (np.inf, None, "bin_width"),
        (5, -1, "maximum_distance"),
        (5, np.inf, "maximum_distance"),
    ],)
def test_distance_profile_rejects_invalid_binning_settings(
        bin_width: float,
        maximum_distance: float | None,
        message: str,
        ) -> None:
    image = np.ones((2, 2))
    reference = np.array([[1, 0], [0, 0]])

    with pytest.raises(ValueError, match=message):
        distance_profile(
            image,
            reference,
            bin_width=bin_width,
            maximum_distance=maximum_distance,)
