from dataclasses import dataclass

from bioimagequant.distance_profile.distance import DistanceProfile


@dataclass(frozen=True, slots=True)
class ProfileResult:
    """One calculated profile and the identities needed for tabulation."""

    frame_index: int
    intensity_channel: str | None
    reference_name: str
    reference_channel: str | None
    roi_name: str | None
    roi_channel: str | None
    profile: DistanceProfile | None
