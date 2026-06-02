import numpy as np
import pytest

from cv_engine.config import DensityConfig
from cv_engine.density.analyzer import DensityAnalyzer
from cv_engine.models.detection import Detection


def _det():
    return Detection(bbox=(0, 0, 10, 10), confidence=0.9, class_id=0, detector_type="person_coco")


@pytest.fixture
def analyzer():
    return DensityAnalyzer(DensityConfig(low_threshold=0.0001, high_threshold=0.001))


def test_low_density(analyzer):
    frame = np.zeros((1000, 1000, 3), dtype=np.uint8)
    assert analyzer.analyze(frame, [_det()]) == "LOW"


def test_medium_density(analyzer):
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    dets = [_det() for _ in range(5)]
    assert analyzer.analyze(frame, dets) == "MEDIUM"


def test_high_density(analyzer):
    frame = np.zeros((50, 50, 3), dtype=np.uint8)
    dets = [_det() for _ in range(10)]
    assert analyzer.analyze(frame, dets) == "HIGH"
