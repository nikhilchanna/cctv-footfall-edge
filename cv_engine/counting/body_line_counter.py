"""Body track line-cross — IN/OUT from user-defined direction vector in the image plane."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

from cv_engine.models.track import Track


def _point_side(pt: tuple[float, float], line_pt1: tuple[float, float], line_pt2: tuple[float, float]) -> float:
    lx = line_pt2[0] - line_pt1[0]
    ly = line_pt2[1] - line_pt1[1]
    px = pt[0] - line_pt1[0]
    py = pt[1] - line_pt1[1]
    return lx * py - ly * px


def _normalize(dx: float, dy: float) -> tuple[float, float]:
    length = math.hypot(dx, dy)
    if length < 1e-6:
        return (0.0, 1.0)
    return (dx / length, dy / length)


@dataclass
class BodyLineConfig:
    # Unit vector — motion aligned with this at line cross = IN
    in_vector: tuple[float, float] = (0.0, 1.0)
    min_motion_px: float = 1.5
    dedup_ms: int = 500


class BodyLineCounter:
    def __init__(self, line_coords: dict, config: BodyLineConfig | None = None):
        self._p1 = (float(line_coords.get("x1", 0)), float(line_coords.get("y1", 200)))
        self._p2 = (float(line_coords.get("x2", 640)), float(line_coords.get("y2", 200)))
        self._cfg = config or BodyLineConfig()
        self._in_vec = _normalize(self._cfg.in_vector[0], self._cfg.in_vector[1])
        self._track_side: dict[int, float] = {}
        self._track_centroid: dict[int, tuple[float, float]] = {}
        self._last_cross_at: dict[int, float] = {}

    @staticmethod
    def vector_from_drag(in_vector_coords: dict) -> tuple[float, float]:
        dx = float(in_vector_coords.get("x2", 0)) - float(in_vector_coords.get("x1", 0))
        dy = float(in_vector_coords.get("y2", 0)) - float(in_vector_coords.get("y1", 0))
        return _normalize(dx, dy)

    def update(self, tracks: list[Track]) -> tuple[int, int]:
        in_delta = 0
        out_delta = 0
        now = time.monotonic()
        active = set()
        ivx, ivy = self._in_vec

        for track in tracks:
            tid = track.track_id
            active.add(tid)
            curr = track.centroid
            side = _point_side(curr, self._p1, self._p2)
            if abs(side) < 1e-3:
                self._track_centroid[tid] = curr
                continue

            prev_side = self._track_side.get(tid)
            prev_pt = self._track_centroid.get(tid)

            if prev_side is not None and prev_pt is not None and prev_side * side < 0:
                mx = curr[0] - prev_pt[0]
                my = curr[1] - prev_pt[1]
                motion_len = math.hypot(mx, my)
                if motion_len >= self._cfg.min_motion_px:
                    dot = mx * ivx + my * ivy
                    last = self._last_cross_at.get(tid, 0.0)
                    if (now - last) * 1000 >= self._cfg.dedup_ms:
                        self._last_cross_at[tid] = now
                        if dot > 0:
                            in_delta += 1
                        elif dot < 0:
                            out_delta += 1

            self._track_side[tid] = side
            self._track_centroid[tid] = curr

        stale = [tid for tid in self._track_side if tid not in active]
        for tid in stale:
            self._track_side.pop(tid, None)
            self._track_centroid.pop(tid, None)

        return in_delta, out_delta

    def get_line_overlay(self) -> dict:
        return {
            "line": [self._p1, self._p2],
            "in_vector": self._in_vec,
        }

    def in_arrow(self) -> tuple[tuple[float, float], tuple[float, float]]:
        mx = (self._p1[0] + self._p2[0]) / 2
        my = (self._p1[1] + self._p2[1]) / 2
        ivx, ivy = self._in_vec
        tail = (mx - ivx * 50, my - ivy * 50)
        tip = (mx + ivx * 50, my + ivy * 50)
        return tail, tip
