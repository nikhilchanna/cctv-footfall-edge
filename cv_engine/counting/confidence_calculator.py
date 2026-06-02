from cv_engine.config import ConfidenceConfig
from cv_engine.models.detection import Detection
from cv_engine.models.track import Track


class ConfidenceCalculator:
    def __init__(self, cfg: ConfidenceConfig):
        self._w_det = cfg.det_weight
        self._w_trk = cfg.track_weight
        self._w_den = cfg.density_weight

    def calculate(
        self,
        detections: list[Detection],
        tracks: list[Track],
        density_level: str,
    ) -> float:
        mean_det = (
            sum(d.confidence for d in detections) / len(detections) if detections else 0.0
        )
        track_ratio = min(1.0, len(tracks) / max(1, len(detections))) if detections else 0.5
        density_conf = {"LOW": 0.9, "MEDIUM": 0.7, "HIGH": 0.5}.get(density_level, 0.5)
        raw = (
            self._w_det * mean_det
            + self._w_trk * track_ratio
            + self._w_den * density_conf
        )
        return max(0.0, min(1.0, raw))
