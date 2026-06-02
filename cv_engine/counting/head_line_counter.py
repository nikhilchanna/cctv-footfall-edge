"""Head centroid line-cross — IN/OUT from user-defined direction vector."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

from cv_engine.counting.body_line_counter import BodyLineCounter
from cv_engine.models.detection import Detection


def _point_side(pt, p1, p2):
    lx = p2[0] - p1[0]
    ly = p2[1] - p1[1]
    px = pt[0] - p1[0]
    py = pt[1] - p1[1]
    return lx * py - ly * px


@dataclass
class HeadLineConfig:
    dedup_ms: int = 400
    grid_cells: int = 8
    match_distance_px: float = 60.0
    in_vector: tuple[float, float] = (0.0, 1.0)
    min_motion_px: float = 1.5


@dataclass
class _Tracklet:
    centroid: tuple[float, float]


class HeadLineCounter:
    def __init__(self, line_coords: dict, config: HeadLineConfig | None = None):
        self._p1 = (float(line_coords.get("x1", 0)), float(line_coords.get("y1", 200)))
        self._p2 = (float(line_coords.get("x2", 640)), float(line_coords.get("y2", 200)))
        self._cfg = config or HeadLineConfig()
        self._in_vec = BodyLineCounter.vector_from_drag(
            {"x1": 0, "y1": 0, "x2": self._cfg.in_vector[0], "y2": self._cfg.in_vector[1]}
        )
        self._tracklets: list[_Tracklet] = []
        self._track_side: list[float] = []
        self._bin_last_count: dict[int, float] = {}

    def _line_bin(self, cx: float, cy: float) -> int:
        dx = self._p2[0] - self._p1[0]
        dy = self._p2[1] - self._p1[1]
        length_sq = dx * dx + dy * dy
        if length_sq < 1e-6:
            return 0
        t = max(0.0, min(1.0, ((cx - self._p1[0]) * dx + (cy - self._p1[1]) * dy) / length_sq))
        return min(self._cfg.grid_cells - 1, int(t * self._cfg.grid_cells))

    def update(self, head_detections: list[Detection]) -> tuple[int, int]:
        now = time.monotonic()
        in_delta = 0
        out_delta = 0
        ivx, ivy = self._in_vec
        used_prev: set[int] = set()

        for det in head_detections:
            curr = det.centroid()
            side = _point_side(curr, self._p1, self._p2)
            if abs(side) < 1e-3:
                continue

            best_i = None
            best_dist = self._cfg.match_distance_px
            for i, tr in enumerate(self._tracklets):
                if i in used_prev:
                    continue
                dist = math.hypot(tr.centroid[0] - curr[0], tr.centroid[1] - curr[1])
                if dist <= best_dist:
                    best_dist = dist
                    best_i = i

            if best_i is not None:
                prev_pt = self._tracklets[best_i].centroid
                prev_side = self._track_side[best_i]
                if prev_side * side < 0:
                    mx = curr[0] - prev_pt[0]
                    my = curr[1] - prev_pt[1]
                    if math.hypot(mx, my) >= self._cfg.min_motion_px:
                        dot = mx * ivx + my * ivy
                        if dot != 0:
                            bin_idx = self._line_bin(curr[0], curr[1])
                            last = self._bin_last_count.get(bin_idx, 0.0)
                            if (now - last) * 1000 >= self._cfg.dedup_ms:
                                self._bin_last_count[bin_idx] = now
                                if dot > 0:
                                    in_delta += 1
                                else:
                                    out_delta += 1
                self._tracklets[best_i].centroid = curr
                self._track_side[best_i] = side
                used_prev.add(best_i)
            else:
                self._tracklets.append(_Tracklet(centroid=curr))
                self._track_side.append(side)

        if len(self._tracklets) > 200:
            self._tracklets = self._tracklets[-100:]
            self._track_side = self._track_side[-100:]

        return in_delta, out_delta

    def get_line_overlay(self) -> dict:
        return {"line": [self._p1, self._p2], "in_vector": self._in_vec}

    def in_arrow(self):
        mx = (self._p1[0] + self._p2[0]) / 2
        my = (self._p1[1] + self._p2[1]) / 2
        ivx, ivy = self._in_vec
        tail = (mx - ivx * 50, my - ivy * 50)
        tip = (mx + ivx * 50, my + ivy * 50)
        return tail, tip
