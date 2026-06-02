from cv_engine.detectors.base import Detector
from cv_engine.models.detection import Detection


class DetectorFusion:
    def __init__(self, head_inside_margin: float = 5.0, nms_iou: float = 0.5):
        self._margin = head_inside_margin
        self._nms_iou = nms_iou

    def fuse(
        self,
        density_level: str,
        person_detector: Detector,
        head_detector: Detector | None,
        frame,
    ) -> list[Detection]:
        if density_level == "LOW":
            return person_detector.detect(frame)
        if density_level == "MEDIUM":
            person = person_detector.detect(frame)
            head = head_detector.detect(frame) if head_detector else []
            return self._merge(person + head)
        # HIGH — head only, fallback person if no head
        if head_detector:
            head = head_detector.detect(frame)
            if head:
                return self._merge(head)
        return person_detector.detect(frame)

    def _merge(self, detections: list[Detection]) -> list[Detection]:
        filtered = self._suppress_heads_inside_person(detections)
        return self._nms(filtered)

    def _centroid_in_box(self, cx, cy, bbox, margin) -> bool:
        x1, y1, x2, y2 = bbox
        return (x1 - margin) <= cx <= (x2 + margin) and (y1 - margin) <= cy <= (y2 + margin)

    def _suppress_heads_inside_person(self, detections: list[Detection]) -> list[Detection]:
        persons = [d for d in detections if d.detector_type != "head_scut"]
        heads = [d for d in detections if d.detector_type == "head_scut"]
        kept_heads = []
        for h in heads:
            cx, cy = h.centroid()
            inside = any(
                self._centroid_in_box(cx, cy, p.bbox, self._margin) for p in persons
            )
            if not inside:
                kept_heads.append(h)
        return persons + kept_heads

    def _nms(self, detections: list[Detection]) -> list[Detection]:
        if not detections:
            return []
        by_type: dict[str, list[Detection]] = {}
        for d in detections:
            by_type.setdefault(d.detector_type, []).append(d)

        out: list[Detection] = []
        for group in by_type.values():
            sorted_d = sorted(group, key=lambda x: x.confidence, reverse=True)
            kept: list[Detection] = []
            for d in sorted_d:
                if any(d.iou(k) > self._nms_iou for k in kept):
                    continue
                kept.append(d)
            out.extend(kept)
        return out
