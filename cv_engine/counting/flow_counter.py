import time
from dataclasses import dataclass

from cv_engine.counting.zone_manager import (
    ZONE_COUNT,
    ZONE_IGNORE,
    ZONE_OBSERVATION,
    ZoneManager,
)
from cv_engine.models.track import Track, TrackState


@dataclass
class _CountedEntry:
    track_id: int
    count_time: float


class FlowCounter:
    """Count only observation → count. Ignore ignore-zone and count-zone spawns."""

    def __init__(
        self,
        zone_manager: ZoneManager,
        camera_role: str = "IN",
        line_coords: dict | None = None,
        entry_side: str = "above",
    ):
        self._zones = zone_manager
        self._camera_role = self._normalize_role(camera_role)
        self._states: dict[int, TrackState] = {}
        self._counted_registry: list[_CountedEntry] = []
        self._in_delta = 0
        self._out_delta = 0

    @staticmethod
    def _normalize_role(role: str) -> str:
        r = (role or "IN").lower()
        if r in ("in", "entry", "footfall"):
            return "IN"
        if r in ("out", "exit"):
            return "OUT"
        return r.upper()

    def update(self, tracks: list[Track]) -> tuple[int, int]:
        self._in_delta = 0
        self._out_delta = 0
        now = time.monotonic()
        self._prune_registry(now)

        for track in tracks:
            tid = track.track_id
            if self._is_already_counted(tid):
                track.counted = True
                track.state = TrackState.COUNTED
                continue

            zone = self._zones.zone_at(track.centroid)
            track.current_zone = zone
            if zone is None:
                continue

            state = self._states.get(tid, TrackState.NEW)

            if state == TrackState.REJECTED:
                track.state = TrackState.REJECTED
                continue

            if state == TrackState.NEW:
                if zone == ZONE_OBSERVATION:
                    self._states[tid] = TrackState.OBSERVATION
                    track.state = TrackState.OBSERVATION
                else:
                    # Spawn in count/ignore, or enter from outside straight into count
                    self._states[tid] = TrackState.REJECTED
                    track.state = TrackState.REJECTED
                continue

            if state == TrackState.OBSERVATION:
                if zone == ZONE_COUNT:
                    self._mark_counted(track, tid, now)
                elif zone == ZONE_IGNORE:
                    self._states[tid] = TrackState.REJECTED
                    track.state = TrackState.REJECTED

        return self._in_delta, self._out_delta

    def _mark_counted(self, track: Track, tid: int, now: float) -> None:
        self._states[tid] = TrackState.COUNTED
        track.state = TrackState.COUNTED
        track.counted = True
        self._register_count(tid, now)
        if self._camera_role == "IN":
            self._in_delta += 1
        elif self._camera_role == "OUT":
            self._out_delta += 1

    def _is_already_counted(self, track_id: int) -> bool:
        return any(e.track_id == track_id for e in self._counted_registry)

    def _register_count(self, track_id: int, count_time: float) -> None:
        if not self._is_already_counted(track_id):
            self._counted_registry.append(_CountedEntry(track_id=track_id, count_time=count_time))

    def _prune_registry(self, now: float, max_age_seconds: float = 3600.0) -> None:
        self._counted_registry = [
            e for e in self._counted_registry if now - e.count_time < max_age_seconds
        ]

    @property
    def counted_track_ids(self) -> set[int]:
        return {e.track_id for e in self._counted_registry}
