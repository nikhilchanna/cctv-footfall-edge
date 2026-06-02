import numpy as np

from cv_engine.config import DensityConfig
from cv_engine.models.detection import Detection


class DensityAnalyzer:
    def __init__(self, cfg: DensityConfig, roi_area: float | None = None):
        self._low = cfg.low_threshold
        self._high = cfg.high_threshold
        self._roi_area = roi_area

    def analyze(self, frame: np.ndarray, detections: list[Detection]) -> str:
        person_dets = [d for d in detections if d.detector_type != "head_scut"]
        visible_area = self._roi_area or (frame.shape[0] * frame.shape[1])
        if visible_area <= 0:
            return "LOW"
        ratio = len(person_dets) / visible_area
        if ratio >= self._high:
            return "HIGH"
        if ratio >= self._low:
            return "MEDIUM"
        return "LOW"
