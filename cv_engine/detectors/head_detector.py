import numpy as np

from cv_engine.detectors.base import Detector
from cv_engine.model_pool import ModelPool
from cv_engine.models.detection import Detection


class HeadDetector(Detector):
    """Ultralytics head detector — SCUT-HEAD or CrowdHuman weights."""

    def __init__(
        self,
        model_path: str,
        detector_type: str,
        class_ids: list[int],
        conf_threshold: float = 0.30,
        iou_threshold: float = 0.45,
        min_bbox_pixels: int = 6,
        max_aspect_ratio: float = 4.0,
    ):
        self._model_path = model_path
        self._detector_type = detector_type
        self._class_ids = class_ids
        self._conf = conf_threshold
        self._iou = iou_threshold
        self._min_bbox = min_bbox_pixels
        self._max_aspect = max_aspect_ratio

    @property
    def detector_type(self) -> str:
        return self._detector_type

    def detect(self, frame: np.ndarray) -> list[Detection]:
        results = ModelPool.infer(
            self._model_path,
            frame,
            classes=self._class_ids,
            conf=self._conf,
            iou=self._iou,
            verbose=False,
        )
        detections: list[Detection] = []
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0].item())
                det = Detection(bbox=(x1, y1, x2, y2), confidence=conf)
                if self._valid_geometry(det):
                    detections.append(det)
        return detections

    def _valid_geometry(self, det: Detection) -> bool:
        w, h = det.width(), det.height()
        if w < self._min_bbox or h < self._min_bbox:
            return False
        if w <= 0 or h <= 0:
            return False
        aspect = max(w / h, h / w)
        if aspect > self._max_aspect:
            return False
        return True
