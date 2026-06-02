import cv2
import numpy as np

from cv_engine.counting.zone_generator import ZonePolygons
from cv_engine.models.track import Track, TrackState


def _point_in_poly(point: tuple[float, float], poly: list[tuple[float, float]]) -> bool:
    if len(poly) < 3:
        return False
    arr = np.array(poly, dtype=np.float32)
    return cv2.pointPolygonTest(arr, point, False) >= 0


class ZoneCounter:
    def __init__(self, zones: ZonePolygons):
        self._zones = zones
        self._states: dict[int, TrackState] = {}
        self._paths: dict[int, str] = {}  # "in" | "out"
        self._in_delta = 0
        self._out_delta = 0

    def update(self, tracks: list[Track]) -> tuple[int, int]:
        self._in_delta = 0
        self._out_delta = 0

        for track in tracks:
            tid = track.track_id
            if self._states.get(tid) == TrackState.COUNTED:
                continue

            zone = self._which_zone(track.centroid)
            if zone is None:
                continue

            state = self._states.get(tid, TrackState.NEW)

            if state == TrackState.NEW:
                if zone == "entry":
                    self._states[tid] = TrackState.ENTRY
                    self._paths[tid] = "in"
                elif zone == "exit":
                    self._states[tid] = TrackState.EXIT
                    self._paths[tid] = "out"
                continue

            path = self._paths.get(tid)
            if path == "in":
                if state == TrackState.ENTRY and zone == "buffer":
                    self._states[tid] = TrackState.BUFFER
                elif state == TrackState.BUFFER and zone == "exit":
                    self._states[tid] = TrackState.COUNTED
                    self._in_delta += 1
            elif path == "out":
                if state == TrackState.EXIT and zone == "buffer":
                    self._states[tid] = TrackState.BUFFER
                elif state == TrackState.BUFFER and zone == "entry":
                    self._states[tid] = TrackState.COUNTED
                    self._out_delta += 1

        return self._in_delta, self._out_delta

    def _which_zone(self, centroid: tuple[float, float]) -> str | None:
        # Buffer first — entry/exit/buffer share edges at the counting line
        if _point_in_poly(centroid, self._zones.buffer):
            return "buffer"
        if _point_in_poly(centroid, self._zones.entry):
            return "entry"
        if _point_in_poly(centroid, self._zones.exit):
            return "exit"
        return None

    def get_zone_overlay(self) -> dict:
        return {
            "entry": self._zones.entry,
            "buffer": self._zones.buffer,
            "exit": self._zones.exit,
        }
