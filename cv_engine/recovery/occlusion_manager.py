import math
import time
from dataclasses import dataclass, field

from cv_engine.config.schema import OcclusionSchema
from cv_engine.models.detection import Detection
from cv_engine.models.track import Track, TrackState


@dataclass
class _LostTrack:
    track: Track
    lost_at: float
    history: list[tuple[float, float]] = field(default_factory=list)


class OcclusionManager:
    def __init__(self, cfg: OcclusionSchema):
        self._timeout = cfg.lost_track_timeout_seconds
        self._threshold = cfg.reattach_threshold_px
        self._lost: dict[int, _LostTrack] = {}

    def register_lost(self, track: Track, history: list[tuple[float, float]]) -> None:
        if track.counted or track.state == TrackState.COUNTED:
            return
        self._lost[track.track_id] = _LostTrack(
            track=track,
            lost_at=time.monotonic(),
            history=list(history[-5:]),
        )

    def recover(
        self,
        active_tracks: list[Track],
        detections: list[Detection],
        frame_idx: int,
    ) -> list[Track]:
        active_ids = {t.track_id for t in active_tracks}
        now = time.monotonic()
        recovered: list[Track] = list(active_tracks)

        for tid in list(self._lost.keys()):
            if tid in active_ids:
                del self._lost[tid]

        used_dets: set[int] = set()
        for tid, lost in list(self._lost.items()):
            if now - lost.lost_at > self._timeout:
                del self._lost[tid]
                continue

            pred = self._predict_centroid(lost, now)
            best_idx = None
            best_dist = self._threshold
            for i, det in enumerate(detections):
                if i in used_dets:
                    continue
                cx, cy = det.centroid()
                dist = math.hypot(cx - pred[0], cy - pred[1])
                if dist < best_dist:
                    best_dist = dist
                    best_idx = i

            if best_idx is not None:
                det = detections[best_idx]
                used_dets.add(best_idx)
                cx, cy = det.centroid()
                restored = Track(
                    track_id=tid,
                    centroid=(cx, cy),
                    first_seen=lost.track.first_seen,
                    last_seen=now,
                    counted=lost.track.counted,
                    current_zone=lost.track.current_zone,
                    confidence=det.confidence,
                    bbox=det.bbox,
                    velocity=lost.track.velocity,
                    state=lost.track.state,
                )
                recovered.append(restored)
                del self._lost[tid]

        return recovered

    def _predict_centroid(self, lost: _LostTrack, now: float) -> tuple[float, float]:
        hist = lost.history
        elapsed = max(1.0, (now - lost.lost_at) * 7.0)  # ~7 fps guess for prediction steps
        if len(hist) >= 2:
            vx = hist[-1][0] - hist[-2][0]
            vy = hist[-1][1] - hist[-2][1]
        else:
            vx, vy = lost.track.velocity
        return (
            lost.track.centroid[0] + vx * elapsed,
            lost.track.centroid[1] + vy * elapsed,
        )
