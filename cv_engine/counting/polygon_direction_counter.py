"""Count IN/OUT inside polygon — motion aligned with IN or OUT vectors."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import cv2
import numpy as np

from cv_engine.counting.body_line_counter import BodyLineCounter
from cv_engine.models.detection import Detection
from cv_engine.models.track import Track


def _normalize(dx: float, dy: float) -> tuple[float, float]:
    length = math.hypot(dx, dy)
    if length < 1e-6:
        return (0.0, 1.0)
    return (dx / length, dy / length)


@dataclass
class PolygonDirectionConfig:
    in_vector: tuple[float, float] = (0.0, 1.0)
    out_vector: tuple[float, float] | None = None
    min_motion_px: float = 2.0
    dedup_ms: int = 600
    head_match_px: float = 60.0


class PolygonDirectionCounter:
    """Inside zone only: motion · in_vector → IN, motion · out_vector → OUT."""

    def __init__(
        self,
        polygon: list[tuple[float, float]],
        in_vector: tuple[float, float],
        out_vector: tuple[float, float] | None = None,
        config: PolygonDirectionConfig | None = None,
    ):
        if len(polygon) < 3:
            raise ValueError("polygon needs at least 3 points")
        self._poly = [tuple(p) for p in polygon]
        self._cfg = config or PolygonDirectionConfig()
        self._in_vec = _normalize(in_vector[0], in_vector[1])
        if out_vector is not None:
            self._out_vec = _normalize(out_vector[0], out_vector[1])
        elif self._cfg.out_vector is not None:
            self._out_vec = _normalize(self._cfg.out_vector[0], self._cfg.out_vector[1])
        else:
            self._out_vec = (-self._in_vec[0], -self._in_vec[1])
        self._prev_centroid: dict[int, tuple[float, float]] = {}
        self._last_count_at: dict[int, float] = {}
        self._head_tracklets: list[tuple[float, float]] = []

    @staticmethod
    def vector_from_drag(coords: dict) -> tuple[float, float]:
        return BodyLineCounter.vector_from_drag(coords)

    def _inside(self, pt: tuple[float, float]) -> bool:
        arr = np.array(self._poly, dtype=np.float32)
        return cv2.pointPolygonTest(arr, pt, False) >= 0

    def _motion_count(self, entity_id: int, curr: tuple[float, float]) -> tuple[int, int]:
        if not self._inside(curr):
            self._prev_centroid.pop(entity_id, None)
            return 0, 0

        prev = self._prev_centroid.get(entity_id)
        self._prev_centroid[entity_id] = curr
        if prev is None or not self._inside(prev):
            return 0, 0

        mx = curr[0] - prev[0]
        my = curr[1] - prev[1]
        if math.hypot(mx, my) < self._cfg.min_motion_px:
            return 0, 0

        dot_in = mx * self._in_vec[0] + my * self._in_vec[1]
        dot_out = mx * self._out_vec[0] + my * self._out_vec[1]
        if dot_in <= 0 and dot_out <= 0:
            return 0, 0

        now = time.monotonic()
        last = self._last_count_at.get(entity_id, 0.0)
        if (now - last) * 1000 < self._cfg.dedup_ms:
            return 0, 0

        if dot_in > dot_out and dot_in > 0:
            self._last_count_at[entity_id] = now
            return 1, 0
        if dot_out > dot_in and dot_out > 0:
            self._last_count_at[entity_id] = now
            return 0, 1
        return 0, 0

    def update_tracks(self, tracks: list[Track]) -> tuple[int, int]:
        in_total = out_total = 0
        seen_inside: set[int] = set()
        for track in tracks:
            tid = track.track_id
            if not self._inside(track.centroid):
                self._prev_centroid.pop(tid, None)
                continue
            seen_inside.add(tid)
            in_d, out_d = self._motion_count(tid, track.centroid)
            in_total += in_d
            out_total += out_d

        stale = [tid for tid in self._prev_centroid if tid not in seen_inside]
        for tid in stale:
            self._prev_centroid.pop(tid, None)
        return in_total, out_total

    def update_heads(self, head_detections: list[Detection]) -> tuple[int, int]:
        in_total = out_total = 0
        inside = [d.centroid() for d in head_detections if self._inside(d.centroid())]
        used: set[int] = set()
        matched_slots: set[int] = set()

        for curr in inside:
            best_i = None
            best_dist = self._cfg.head_match_px
            for i, prev_pt in enumerate(self._head_tracklets):
                if i in used:
                    continue
                dist = math.hypot(prev_pt[0] - curr[0], prev_pt[1] - curr[1])
                if dist <= best_dist:
                    best_dist = dist
                    best_i = i

            if best_i is not None:
                entity_id = 10_000 + best_i
                used.add(best_i)
                matched_slots.add(best_i)
                self._head_tracklets[best_i] = curr
            else:
                entity_id = 10_000 + len(self._head_tracklets)
                self._head_tracklets.append(curr)
                matched_slots.add(len(self._head_tracklets) - 1)

            in_d, out_d = self._motion_count(entity_id, curr)
            in_total += in_d
            out_total += out_d

        for i in range(len(self._head_tracklets)):
            if i not in matched_slots:
                self._prev_centroid.pop(10_000 + i, None)

        if len(self._head_tracklets) > 150:
            self._head_tracklets = self._head_tracklets[-80:]

        return in_total, out_total

    def filter_inside_tracks(self, tracks: list[Track]) -> list[Track]:
        return [t for t in tracks if self._inside(t.centroid)]

    def filter_inside_heads(self, head_detections: list[Detection]) -> list[Detection]:
        return [d for d in head_detections if self._inside(d.centroid())]

    def count_inside(self, centroids: list[tuple[float, float]]) -> int:
        return sum(1 for c in centroids if self._inside(c))

    def get_overlay(self) -> dict:
        return {
            "polygon": self._poly,
            "in_vector": self._in_vec,
            "out_vector": self._out_vec,
        }

    def _arrow_from_center(self, vec: tuple[float, float], offset: float) -> tuple[tuple[float, float], tuple[float, float]]:
        xs = [p[0] for p in self._poly]
        ys = [p[1] for p in self._poly]
        mx = sum(xs) / len(xs)
        my = sum(ys) / len(ys)
        vx, vy = vec
        tail = (mx - vx * offset, my - vy * offset)
        tip = (mx + vx * offset, my + vy * offset)
        return tail, tip

    def in_arrow(self) -> tuple[tuple[float, float], tuple[float, float]]:
        return self._arrow_from_center(self._in_vec, 50)

    def out_arrow(self) -> tuple[tuple[float, float], tuple[float, float]]:
        return self._arrow_from_center(self._out_vec, 70)
