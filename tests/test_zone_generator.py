import pytest

from cv_engine.config import ZoneSchema, resolve_zone_polygons
from cv_engine.counting.zone_generator import ZoneConfigError, ZoneGenerator


def test_horizontal_line():
    zones = ZoneGenerator.generate_from_line(
        {"x1": 0, "y1": 300, "x2": 1280, "y2": 300},
        observation_offset=80,
        count_zone_width=80,
        ignore_offset=80,
        frame_width=1280,
        frame_height=720,
    )
    assert len(zones.observation) == 4
    assert len(zones.count) == 4
    assert len(zones.ignore) == 4
    assert len(zones.entry) == 4


def test_vertical_line():
    zones = ZoneGenerator.generate_from_line(
        {"x1": 400, "y1": 0, "x2": 400, "y2": 720},
        observation_offset=50,
        count_zone_width=60,
        ignore_offset=50,
    )
    assert len(zones.observation) == 4


def test_diagonal_line():
    zones = ZoneGenerator.generate_from_line(
        {"x1": 0, "y1": 0, "x2": 640, "y2": 480},
        observation_offset=40,
        count_zone_width=40,
        ignore_offset=40,
    )
    assert len(zones.count) == 4


def test_zero_length_raises():
    with pytest.raises(ZoneConfigError):
        ZoneGenerator.generate_from_line(
            {"x1": 10, "y1": 10, "x2": 10, "y2": 10},
            80, 80, 80,
        )


def test_manual_override():
    cfg = ZoneSchema(
        auto_generate=False,
        observation_points=[(0, 0), (100, 0), (100, 50), (0, 50)],
        count_points=[(0, 50), (100, 50), (100, 100), (0, 100)],
        ignore_points=[(0, 100), (100, 100), (100, 150), (0, 150)],
    )
    zones = resolve_zone_polygons(cfg, line_coords={"x1": 0, "y1": 0, "x2": 100, "y2": 0})
    assert zones.observation[0] == (0, 0)
