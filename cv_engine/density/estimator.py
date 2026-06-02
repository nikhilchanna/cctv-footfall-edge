import numpy as np

from cv_engine.models.detection import Detection


class DensityEstimator:
    def estimate_count(self, frame: np.ndarray, detections: list[Detection]) -> float:
        person_dets = [d for d in detections if d.detector_type != "head_scut"]
        area = frame.shape[0] * frame.shape[1]
        if area <= 0 or not person_dets:
            return 0.0
        ratio = len(person_dets) / area
        return ratio * area
