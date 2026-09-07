import numpy as np
import pandas as pd
import pytest

from bioimagequant import DistanceProfiler
from bioimagequant.distance_profile import stack as profile_stack


def _profiler() -> DistanceProfiler:
    image = np.array([
        [[[10, 20, 30, 40, 50]], [[100, 200, 300, 400, 500]]],
        [[[15, 25, 35, 45, 55]], [[150, 250, 350, 450, 550]]],
    ], dtype=float)
    reference = np.zeros((2, 1, 5), dtype=np.uint8)
    reference[0, 0, 0] = 1
    reference[1, 0, 2] = 1
    roi = np.zeros((2, 1, 5), dtype=np.uint8)
    roi[:, 0, 1:4] = 1

    profiler = DistanceProfiler(interval=60, pixel_size=0.5)
    profiler.add_intensity(
        image, "TCYX", channel_labels=("GFP", "RFP"))
    profiler.add_ref(
        reference, "TYX", name="edge", channel_labels=("RFP",))
    profiler.add_roi(
        roi, "TYX", name="tailfin", channel_labels=("RFP",))
    return profiler


def test_calculate_builds_all_conditions_and_aligned_bins() -> None:
    result = _profiler().calculate(bin_width=2)
    assert len(result) == 24
    assert set(result["channel"]) == {"GFP", "RFP"}
    assert set(result["ref_label_name"]) == {"edge"}
    assert set(result["ref_channel"]) == {"RFP"}
    assert set(result["roi_label_name"].dropna()) == {"tailfin"}
    assert result["roi_label_name"].isna().any()
    assert set(result["roi_channel"].dropna()) == {"RFP"}
    np.testing.assert_allclose(result["dist_um"].unique(), [0.5, 1.5, 2.5])

    selected = result[
        (result["frame"] == 1)
        & (result["channel"] == "GFP")
        & (result["roi_label_name"] == "tailfin")]
    np.testing.assert_array_equal(selected["pixel_count"], [1, 2, 0])
    np.testing.assert_allclose(selected["mean_intensity"].iloc[:2], [20, 35])
    assert np.isnan(selected["mean_intensity"].iloc[2])


def test_maximum_bins_limits_output() -> None:
    assert set(_profiler().calculate(
        bin_width=2, maximum_bins=2)["bin"]) == {0, 1}


def test_parallel_profiles_match_serial_profiles() -> None:
    profiler = _profiler()

    serial = profiler.calculate(bin_width=2, workers=1)
    parallel = profiler.calculate(bin_width=2, workers=2)

    pd.testing.assert_frame_equal(parallel, serial)


def test_distance_map_is_calculated_once_per_reference_frame(
        monkeypatch: pytest.MonkeyPatch,
        ) -> None:
    original = profile_stack.distance_map
    calls = 0

    def count_calls(reference: np.ndarray) -> np.ndarray:
        nonlocal calls
        calls += 1
        return original(reference)

    monkeypatch.setattr(profile_stack, "distance_map", count_calls)

    _profiler().calculate(bin_width=2, workers=1)

    assert calls == 2


def test_static_masks_are_reused_for_every_frame() -> None:
    profiler = DistanceProfiler()
    profiler.add_intensity(
        np.ones((2, 1, 3)), "TYX", channel_labels=("GFP",))
    profiler.add_ref(
        np.array([[1, 0, 0]], dtype=np.uint8),
        "YX",
        name="edge",
        channel_labels=("GFP",))
    assert set(profiler.calculate(bin_width=1)["frame"]) == {1, 2}


def test_rejects_z_axes() -> None:
    with pytest.raises(ValueError, match="Project Z"):
        DistanceProfiler().add_intensity(
            np.zeros((2, 3, 3)), "ZYX", channel_labels=("GFP",))


def test_rejects_fractional_maximum_bins() -> None:
    with pytest.raises(ValueError, match="maximum_bins"):
        _profiler().calculate(maximum_bins=1.5)  # type: ignore[arg-type]


@pytest.mark.parametrize("maximum_bins", [0, -1, True])
def test_rejects_invalid_maximum_bins(maximum_bins: int) -> None:
    with pytest.raises(ValueError, match="maximum_bins"):
        _profiler().calculate(maximum_bins=maximum_bins)


@pytest.mark.parametrize("workers", [0, -1, True, 1.5])
def test_rejects_invalid_worker_count(workers: int) -> None:
    with pytest.raises(ValueError, match="workers"):
        _profiler().calculate(workers=workers)  # type: ignore[arg-type]


def test_missing_optional_metadata_produces_missing_columns() -> None:
    profiler = DistanceProfiler()
    profiler.add_intensity(np.ones((1, 3)), "YX")
    profiler.add_ref(
        np.array([[1, 0, 0]]),
        "YX",
        name="edge",
        channel_labels=("GFP",))
    result = profiler.calculate(bin_width=1)
    assert result["channel"].isna().all()
    assert result["time_sec"].isna().all()
    assert result["dist_um"].isna().all()
