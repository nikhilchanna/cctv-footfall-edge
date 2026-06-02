import numpy as np
import supervision as sv

from cv_engine.config import TrackerConfig
from cv_engine.models.detection import Detection
from cv_engine.models.track import Track, TrackState
from cv_engine.tracking.base import Tracker
from cv_engine.tracking.bytetrack_tracker import _detections_to_sv, _sv_to_tracks


class BotSortAdapter(Tracker):
    # BoT-SORT not in supervision 0.28 — same ByteTrack backend, different buffer
    def __init__(self, cfg: TrackerConfig):
        self._tracker = sv.ByteTrack(
            track_activation_threshold=cfg.track_thresh,
            lost_track_buffer=cfg.track_buffer * 2,
            minimum_matching_threshold=0.7,
        )
        self._frame_idx = 0
        self._prev_centroids: dict[int, tuple[float, float]] = {}

    def update(self, detections: list[Detection], frame: np.ndarray) -> list[Track]:
        self._frame_idx += 1
        sv_dets = _detections_to_sv(detections)
        tracked = self._tracker.update_with_detections(sv_dets)
        tracks = _sv_to_tracks(tracked, self._frame_idx, self._prev_centroids)
        self._prev_centroids = {t.track_id: t.centroid for t in tracks}
        return tracks

    def reset(self) -> None:
        self._tracker.reset()
        self._frame_idx = 0
        self._prev_centroids = {}
