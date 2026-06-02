import pytest

from cv_engine.models.detection import Detection


@pytest.fixture
def box_a():
    return Detection(bbox=(0, 0, 10, 10), confidence=0.9)


@pytest.fixture
def box_b():
    return Detection(bbox=(5, 5, 15, 15), confidence=0.8)


def test_centroid(box_a):
    assert box_a.centroid() == (5.0, 5.0)


def test_area(box_a):
    assert box_a.area() == 100.0


def test_width_height(box_a):
    assert box_a.width() == 10.0
    assert box_a.height() == 10.0


def test_frozen_immutable(box_a):
    with pytest.raises(Exception):
        box_a.confidence = 0.1  # type: ignore
