from cv_engine.counting.head_line_counter import HeadLineCounter, HeadLineConfig
from cv_engine.models.detection import Detection


def _head(cx, cy):
    half = 10
    return Detection(
        (cx - half, cy - half, cx + half, cy + half),
        0.9,
        0,
        "head_scut",
    )


def test_head_cross_in_counts():
    counter = HeadLineCounter(
        {"x1": 0, "y1": 100, "x2": 200, "y2": 100},
        HeadLineConfig(in_vector=(0.0, 1.0)),
    )
    counter.update([_head(50, 80)])
    in_d, out_d = counter.update([_head(50, 120)])
    assert in_d == 1 and out_d == 0


def test_head_cross_out_counts():
    counter = HeadLineCounter(
        {"x1": 0, "y1": 100, "x2": 200, "y2": 100},
        HeadLineConfig(in_vector=(0.0, 1.0)),
    )
    counter.update([_head(50, 120)])
    in_d, out_d = counter.update([_head(50, 80)])
    assert in_d == 0 and out_d == 1


def test_head_horizontal_in_vector():
    counter = HeadLineCounter(
        {"x1": 100, "y1": 0, "x2": 100, "y2": 200},
        HeadLineConfig(in_vector=(1.0, 0.0), match_distance_px=80.0),
    )
    counter.update([_head(70, 100)])
    in_d, out_d = counter.update([_head(130, 100)])
    assert in_d == 1 and out_d == 0
