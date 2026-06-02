import numpy as np
import json
import time

from cv_engine.detectors.base import Detector
from cv_engine.model_pool import ModelPool
from cv_engine.models.detection import Detection


class YOLODetector(Detector):
    def __init__(
        self,
        model_path: str,
        class_ids: list[int],
        detector_type: str,
        conf_threshold: float = 0.35,
        iou_threshold: float = 0.45,
    ):
        self._model_path = model_path
        self._class_ids = class_ids
        self._detector_type = detector_type
        self._conf = conf_threshold
        self._iou = iou_threshold
        self._call_count = 0

    @property
    def detector_type(self) -> str:
        return self._detector_type

    def detect(self, frame: np.ndarray) -> list[Detection]:
        self._call_count += 1
        try:
            results = ModelPool.infer(
                self._model_path,
                frame,
                classes=self._class_ids,
                conf=self._conf,
                iou=self._iou,
                verbose=False,
            )
        except Exception as exc:
            # region agent log
            if self._call_count <= 3 or self._call_count % 50 == 0:
                try:
                    with open("/Users/home/analytics_footfall/.cursor/debug-b17557.log", "a") as f:
                        f.write(
                            json.dumps(
                                {
                                    "sessionId": "b17557",
                                    "hypothesisId": "A",
                                    "location": "yolo_detector.py:detect",
                                    "message": "infer failed",
                                    "data": {
                                        "model": self._model_path,
                                        "error": str(exc)[:200],
                                        "call": self._call_count,
                                    },
                                    "timestamp": int(time.time() * 1000),
                                }
                            )
                            + "\n"
                        )
                except Exception:
                    pass
            # endregion
            raise
        detections: list[Detection] = []
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0].item())
                cls_id = int(box.cls[0].item())
                detections.append(
                    Detection(
                        bbox=(x1, y1, x2, y2),
                        confidence=conf,
                        class_id=cls_id,
                        detector_type=self._detector_type,
                    )
                )
        if self._call_count <= 2 or (self._call_count % 60 == 0 and len(detections) == 0):
            # region agent log
            try:
                with open("/Users/home/analytics_footfall/.cursor/debug-b17557.log", "a") as f:
                    f.write(
                        json.dumps(
                            {
                                "sessionId": "b17557",
                                "hypothesisId": "A",
                                "location": "yolo_detector.py:detect",
                                "message": "infer result",
                                "data": {
                                    "model": self._model_path,
                                    "det_count": len(detections),
                                    "frame_shape": list(frame.shape),
                                    "call": self._call_count,
                                },
                                "timestamp": int(time.time() * 1000),
                            }
                        )
                        + "\n"
                    )
            except Exception:
                pass
            # endregion
        return detections
