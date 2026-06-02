"""Footfall config and occupancy-only behavior."""

import numpy as np

from cv_engine.config import load_engine_config
from cv_engine.engine import CrowdCountingEngine


def test_occupancy_only_zeros_footfall():
    cfg = load_engine_config(
        {"footfall": {"camera_role": "occupancy_only"}},
        line_coords={"x1": 0, "y1": 100, "x2": 200, "y2": 100},
    )
    eng = CrowdCountingEngine(cfg)

    class FakeDet:
        detector_type = "head_scut"

        def detect(self, frame):
            return []

    eng._detector = FakeDet()
    eng._tracker.update = lambda d, f: []

    result = eng.process(np.zeros((480, 640, 3), dtype=np.uint8))
    assert result.in_count_delta == 0
    assert result.out_count_delta == 0


def test_count_direction_in_only():
    cfg = load_engine_config(
        {"footfall": {"count_direction": "in_only"}},
        line_coords={"x1": 0, "y1": 100, "x2": 200, "y2": 100},
    )
    assert cfg.footfall.count_direction == "in_only"
