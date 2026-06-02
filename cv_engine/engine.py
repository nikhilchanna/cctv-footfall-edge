import json
import logging
import time

import numpy as np

from cv_engine.config.schema import EngineSchema, TrackerSchema
from cv_engine.counting.flow_counter import FlowCounter
from cv_engine.counting.zone_manager import ZoneManager
from cv_engine.detectors.detector_factory import build_head_detector
from cv_engine.models.counting_result import CountingResult
from cv_engine.recovery.occlusion_manager import OcclusionManager
from cv_engine.tracking.bytetrack_tracker import ByteTrackTracker

logger = logging.getLogger(__name__)

DEBUG_LOG = "/Users/home/analytics_footfall/.cursor/debug-f3319a.log"


def _dbg(hypothesis_id, location, message, data=None):
    # region agent log
    try:
        with open(DEBUG_LOG, "a") as f:
            f.write(
                json.dumps(
                    {
                        "sessionId": "f3319a",
                        "hypothesisId": hypothesis_id,
                        "location": location,
                        "message": message,
                        "data": data or {},
                        "timestamp": int(time.time() * 1000),
                    }
                )
                + "\n"
            )
    except Exception:
        pass
    # endregion


class CrowdCountingEngine:
    """
    Head detect → ByteTrack → occlusion recovery → zone transition → count.
    Detections never become counts — only valid observation→count transitions do.
    """

    def __init__(self, config: EngineSchema):
        self._cfg = config
        zone_polygons = config.zone_polygons
        if zone_polygons is None:
            from cv_engine.config import resolve_zone_polygons

            zone_polygons = resolve_zone_polygons(
                config.zones, config.line_coords
            )

        self._detector = build_head_detector(config)
        self._tracker = ByteTrackTracker(config.tracker)
        self._occlusion = OcclusionManager(config.occlusion)
        self._zone_manager = ZoneManager(zone_polygons)
        self._flow_counter = FlowCounter(
            self._zone_manager,
            camera_role=config.footfall.camera_role,
            line_coords=config.line_coords,
            entry_side=config.zones.entry_side,
        )
        self._frame_idx = 0
        self._track_history: dict[int, list[tuple[float, float]]] = {}
        self._prev_tracks: dict[int, object] = {}
        self._active_path = "zone_transition"

    @property
    def active_footfall_path(self) -> str:
        return self._active_path

    def _apply_direction_filter(self, in_d: int, out_d: int) -> tuple[int, int]:
        direction = self._cfg.footfall.count_direction
        if direction == "in_only":
            return in_d, 0
        if direction == "out_only":
            return 0, out_d
        return in_d, out_d

    def process(self, frame: np.ndarray) -> CountingResult:
        self._frame_idx += 1
        role = self._cfg.footfall.camera_role

        if role == "occupancy_only":
            detections = self._detector.detect(frame)
            tracks = self._tracker.update(detections, frame)
            return CountingResult(
                in_count_delta=0,
                out_count_delta=0,
                confidence=self._mean_confidence(detections, tracks),
                active_tracks=tracks,
                current_count=len(tracks),
                density_level="LOW",
                detections=detections,
            )

        detections = self._detector.detect(frame)
        tracks = self._tracker.update(detections, frame)

        active_ids = {t.track_id for t in tracks}
        for tid, prev in self._prev_tracks.items():
            if tid not in active_ids:
                self._occlusion.register_lost(
                    prev, self._track_history.get(tid, [prev.centroid])
                )
        tracks = self._occlusion.recover(tracks, detections, self._frame_idx)

        in_d, out_d = self._flow_counter.update(tracks)
        in_d, out_d = self._apply_direction_filter(in_d, out_d)

        if self._frame_idx <= 3 or self._frame_idx % 30 == 0:
            # region agent log
            zone_hist: dict[str, int] = {}
            spawn_count = 0
            spawn_in_count = 0
            for t in tracks:
                z = t.current_zone or "none"
                zone_hist[z] = zone_hist.get(z, 0) + 1
                st = self._flow_counter._states.get(t.track_id)
                if st is None or st.value == "NEW":
                    spawn_count += 1
                    if z == "count":
                        spawn_in_count += 1
            _dbg(
                "H1",
                "engine.py:process",
                "frame summary",
                {
                    "frame": self._frame_idx,
                    "dets": len(detections),
                    "tracks": len(tracks),
                    "in_d": in_d,
                    "out_d": out_d,
                    "zone_hist": zone_hist,
                    "new_spawns": spawn_count,
                    "spawn_in_count_zone": spawn_in_count,
                    "total_counted": len(self._flow_counter.counted_track_ids),
                },
            )
            # endregion

        self._prev_tracks = {t.track_id: t for t in tracks}
        for t in tracks:
            hist = self._track_history.setdefault(t.track_id, [])
            hist.append(t.centroid)
            if len(hist) > 5:
                hist.pop(0)

        confidence = self._mean_confidence(detections, tracks)
        return CountingResult(
            in_count_delta=in_d,
            out_count_delta=out_d,
            confidence=confidence,
            active_tracks=tracks,
            current_count=len(tracks),
            density_level=self._density_from_tracks(tracks, frame),
            detections=detections,
        )

    def _mean_confidence(self, detections, tracks) -> float:
        if tracks:
            return sum(t.confidence for t in tracks) / len(tracks)
        if detections:
            return sum(d.confidence for d in detections) / len(detections)
        return 0.0

    def _density_from_tracks(self, tracks, frame) -> str:
        if not tracks:
            return "LOW"
        h, w = frame.shape[:2]
        area = max(1, w * h)
        ratio = len(tracks) / area
        if ratio > 0.008:
            return "HIGH"
        if ratio > 0.002:
            return "MEDIUM"
        return "LOW"

    def get_status_fields(self) -> dict:
        return {
            "density_level": "LOW",
            "footfall_mode": "zone_transition",
            "camera_role": self._cfg.footfall.camera_role,
            "count_direction": self._cfg.footfall.count_direction,
            "active_footfall_path": self._active_path,
        }

    def get_zone_overlay(self) -> dict:
        return self._zone_manager.get_overlay()
