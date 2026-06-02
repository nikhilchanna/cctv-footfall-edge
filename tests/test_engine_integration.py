import numpy as np
import pytest

from cv_engine.config import load_engine_config
from cv_engine.engine import CrowdCountingEngine
from cv_engine.models.detection import Detection


class MockHeadDetector:
    def __init__(self, boxes):
        self._boxes = boxes

    @property
    def detector_type(self):
        return "head_scut"

    def detect(self, frame):
        return self._boxes


@pytest.fixture
def engine(monkeypatch):
    cfg = load_engine_config(
        {},
        line_coords={"x1": 0, "y1": 300, "x2": 1280, "y2": 300},
    )
    eng = CrowdCountingEngine(cfg)
    eng._detector = MockHeadDetector(
        [Detection((100, 100, 150, 200), 0.9)]
    )
    return eng


def test_process_returns_counting_result(engine):
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    result = engine.process(frame)
    assert result.current_count >= 0
    assert result.density_level in ("LOW", "MEDIUM", "HIGH")
    assert 0.0 <= result.confidence <= 1.0
    assert isinstance(result.in_count_delta, int)
    assert isinstance(result.out_count_delta, int)


def test_zone_overlay(engine):
    overlay = engine.get_zone_overlay()
    assert "entry" in overlay
    assert "buffer" in overlay
    assert "exit" in overlay
