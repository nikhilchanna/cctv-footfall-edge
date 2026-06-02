import numpy as np
import pytest

from cv_engine.fusion.detector_fusion import DetectorFusion
from cv_engine.models.detection import Detection


class FakeDetector:
    def __init__(self, dets, dtype):
        self._dets = dets
        self._dtype = dtype

    @property
    def detector_type(self):
        return self._dtype

    def detect(self, frame):
        return self._dets


def test_low_uses_person_only():
    person = FakeDetector(
        [Detection((0, 0, 50, 100), 0.9, 0, "person_coco")],
        "person_coco",
    )
    head = FakeDetector(
        [Detection((10, 10, 20, 20), 0.8, 0, "head_scut")],
        "head_scut",
    )
    fusion = DetectorFusion()
    out = fusion.fuse("LOW", person, head, np.zeros((480, 640, 3), dtype=np.uint8))
    assert len(out) == 1
    assert out[0].detector_type == "person_coco"


def test_head_inside_person_suppressed():
    person = FakeDetector(
        [Detection((0, 0, 100, 100), 0.9, 0, "person_coco")],
        "person_coco",
    )
    head = FakeDetector(
        [Detection((40, 40, 60, 60), 0.8, 0, "head_scut")],
        "head_scut",
    )
    fusion = DetectorFusion()
    out = fusion.fuse("MEDIUM", person, head, np.zeros((480, 640, 3), dtype=np.uint8))
    types = [d.detector_type for d in out]
    assert "head_scut" not in types


def test_high_uses_head():
    person = FakeDetector([], "person_coco")
    head = FakeDetector(
        [Detection((0, 0, 20, 20), 0.8, 0, "head_scut")],
        "head_scut",
    )
    fusion = DetectorFusion()
    out = fusion.fuse("HIGH", person, head, np.zeros((480, 640, 3), dtype=np.uint8))
    assert len(out) == 1
    assert out[0].detector_type == "head_scut"
