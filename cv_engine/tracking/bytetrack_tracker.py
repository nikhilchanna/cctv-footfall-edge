import time

import numpy as np
import supervision as sv

from cv_engine.config.schema import TrackerSchema
from cv_engine.models.detection import Detection
from cv_engine.models.track import Track, TrackState
from cv_engine.tracking.base import Tracker


def _detections_to_sv(detections: list[Detection]) -> sv.Detections:
    if not detections:
        return sv.Detections.empty()
    xyxy = np.array([d.bbox for d in detections], dtype=np.float32)
    conf = np.array([d.confidence for d in detections], dtype=np.float32)
    cls = np.zeros(len(detections), dtype=np.int32)
    return sv.Detections(xyxy=xyxy, confidence=conf, class_id=cls)


class ByteTrackTracker(Tracker):
    def __init__(self, cfg: TrackerSchema):
        self._tracker = sv.ByteTrack(
            track_activation_threshold=cfg.track_thresh,
            lost_track_buffer=cfg.track_buffer,
        )
        self._prev_centroids: dict[int, tuple[float, float]] = {}
        self._first_seen: dict[int, float] = {}

    def update(self, detections: list[Detection], frame: np.ndarray) -> list[Track]:
        now = time.monotonic()
        sv_dets = _detections_to_sv(detections)
        tracked = self._tracker.update_with_detections(sv_dets)
        tracks: list[Track] = []
        if tracked.tracker_id is None:
            return tracks

        for i, tid in enumerate(tracked.tracker_id):
            if tid is None or tid < 0:
                continue
            tid = int(tid)
            x1, y1, x2, y2 = tracked.xyxy[i]
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            prev = self._prev_centroids.get(tid, (cx, cy))
            vx, vy = cx - prev[0], cy - prev[1]
            conf = float(tracked.confidence[i]) if tracked.confidence is not None else 0.5
            if tid not in self._first_seen:
                self._first_seen[tid] = now
            tracks.append(
                Track(
                    track_id=tid,
                    centroid=(float(cx), float(cy)),
                    first_seen=self._first_seen[tid],
                    last_seen=now,
                    counted=False,
                    current_zone=None,
                    confidence=conf,
                    bbox=(float(x1), float(y1), float(x2), float(y2)),
                    velocity=(float(vx), float(vy)),
                    state=TrackState.NEW,
                )
            )
        self._prev_centroids = {t.track_id: t.centroid for t in tracks}
        return tracks

    def reset(self) -> None:
        self._tracker.reset()
        self._prev_centroids = {}
        self._first_seen = {}
