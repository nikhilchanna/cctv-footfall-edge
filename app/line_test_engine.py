"""Polygon + direction test engine — body or head, role/direction filters."""

from dataclasses import dataclass, field

import numpy as np

from cv_engine.config import load_engine_config
from cv_engine.counting.body_line_counter import BodyLineCounter
from cv_engine.counting.polygon_direction_counter import PolygonDirectionCounter, PolygonDirectionConfig
from cv_engine.detectors.detector_factory import build_detectors
from cv_engine.models.detection import Detection
from cv_engine.models.track import Track
from cv_engine.tracking.tracker_factory import build_tracker


def _apply_direction(in_d: int, out_d: int, count_direction: str) -> tuple[int, int]:
    if count_direction == "in_only":
        return in_d, 0
    if count_direction == "out_only":
        return 0, out_d
    return in_d, out_d


@dataclass
class LineTestResult:
    current_count: int = 0
    in_count_delta: int = 0
    out_count_delta: int = 0
    active_tracks: list[Track] = field(default_factory=list)
    detections: list[Detection] = field(default_factory=list)
    head_detections: list[Detection] = field(default_factory=list)
    density_level: str = "LOW"
    active_footfall_path: str = "polygon"


class LineOnlyTestEngine:
    def __init__(
        self,
        polygon: list[list[float]],
        in_vector_coords: dict | None = None,
        out_vector_coords: dict | None = None,
        count_mode: str = "head_line",
        camera_role: str = "footfall",
        count_direction: str = "both",
        cv_engine_config: dict | None = None,
        frame_width: int | None = None,
        frame_height: int | None = None,
        line_coords: dict | None = None,
    ):
        merged = dict(cv_engine_config or {})
        merged.setdefault("footfall", {})
        merged["footfall"]["mode"] = count_mode if count_mode in ("body_line", "head_line") else "head_line"
        merged["footfall"]["camera_role"] = camera_role
        merged["footfall"]["count_direction"] = count_direction

        # Config loader still wants line_coords — use polygon bbox center line as dummy
        pts = [(float(p[0]), float(p[1])) for p in polygon]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        dummy_line = line_coords or {
            "x1": int(min(xs)),
            "y1": int(sum(ys) / len(ys)),
            "x2": int(max(xs)),
            "y2": int(sum(ys) / len(ys)),
        }

        cfg = load_engine_config(
            merged,
            line_coords=dummy_line,
            frame_width=frame_width,
            frame_height=frame_height,
        )
        dets = build_detectors(cfg)
        self._person = dets["person"]
        self._head = dets.get("head")
        self._tracker = build_tracker(cfg.tracker)

        in_vec = (0.0, 1.0)
        out_vec = (0.0, -1.0)
        if in_vector_coords:
            in_vec = BodyLineCounter.vector_from_drag(in_vector_coords)
        if out_vector_coords:
            out_vec = BodyLineCounter.vector_from_drag(out_vector_coords)

        self._count_mode = count_mode if count_mode in ("body_line", "head_line") else "head_line"
        self._camera_role = camera_role
        self._count_direction = count_direction
        self._in_vector_coords = in_vector_coords
        self._out_vector_coords = out_vector_coords
        self._polygon = pts
        self._counter = PolygonDirectionCounter(
            pts,
            in_vec,
            out_vec,
            PolygonDirectionConfig(in_vector=in_vec, out_vector=out_vec),
        )
        self._frame_idx = 0

    def process(self, frame: np.ndarray) -> LineTestResult:
        self._frame_idx += 1
        person_dets = self._person.detect(frame)
        head_dets = self._head.detect(frame) if self._head else []

        if self._camera_role == "occupancy_only":
            if self._count_mode == "head_line" and head_dets:
                centroids = [d.centroid() for d in head_dets]
                count = self._counter.count_inside(centroids)
            else:
                tracks = self._tracker.update(person_dets, frame)
                count = self._counter.count_inside([t.centroid for t in tracks])
            return LineTestResult(
                current_count=count,
                in_count_delta=0,
                out_count_delta=0,
                active_tracks=[],
                detections=person_dets,
                head_detections=head_dets,
                active_footfall_path="occupancy_only",
            )

        if self._count_mode == "head_line":
            inside_heads = self._counter.filter_inside_heads(head_dets)
            in_d, out_d = self._counter.update_heads(head_dets)
            in_d, out_d = _apply_direction(in_d, out_d, self._count_direction)
            return LineTestResult(
                current_count=len(inside_heads),
                in_count_delta=in_d,
                out_count_delta=out_d,
                active_tracks=[],
                detections=person_dets,
                head_detections=inside_heads,
                active_footfall_path="head_polygon",
            )

        tracks = self._tracker.update(person_dets, frame)
        inside_tracks = self._counter.filter_inside_tracks(tracks)
        in_d, out_d = self._counter.update_tracks(tracks)
        in_d, out_d = _apply_direction(in_d, out_d, self._count_direction)
        return LineTestResult(
            current_count=len(inside_tracks),
            in_count_delta=in_d,
            out_count_delta=out_d,
            active_tracks=inside_tracks,
            detections=person_dets,
            head_detections=self._counter.filter_inside_heads(head_dets),
            active_footfall_path="body_polygon",
        )

    def get_zone_overlay(self) -> dict:
        overlay = self._counter.get_overlay()
        overlay["in_arrow"] = list(self._counter.in_arrow())
        overlay["out_arrow"] = list(self._counter.out_arrow())
        return overlay

    def get_status_fields(self) -> dict:
        ivx, ivy = self._counter._in_vec
        ovx, ovy = self._counter._out_vec
        path = self._count_mode
        if self._camera_role == "occupancy_only":
            path = "occupancy_only"
        elif self._count_mode == "head_line":
            path = "head_polygon"
        else:
            path = "body_polygon"
        return {
            "density_level": "LOW",
            "footfall_mode": self._count_mode,
            "camera_role": self._camera_role,
            "count_direction": self._count_direction,
            "active_footfall_path": path,
            "in_vector": {"dx": ivx, "dy": ivy},
            "out_vector": {"dx": ovx, "dy": ovy},
            "in_vector_coords": self._in_vector_coords,
            "out_vector_coords": self._out_vector_coords,
            "polygon_points": len(self._polygon),
        }
