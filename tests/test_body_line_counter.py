import time

from cv_engine.counting.body_line_counter import BodyLineCounter, BodyLineConfig
from cv_engine.models.track import Track, TrackState


def _track(tid, cx, cy):
    now = time.monotonic()
    return Track(
        track_id=tid,
        centroid=(cx, cy),
        first_seen=now,
        last_seen=now,
        counted=False,
        current_zone=None,
        confidence=0.9,
        bbox=(cx - 10, cy - 20, cx + 10, cy),
        velocity=(0.0, 0.0),
        direction=0,
        age=1,
        state=TrackState.NEW,
    )


def test_vertical_line_horizontal_motion_in_right():
    # Line vertical, people walk right = IN
    counter = BodyLineCounter(
        {"x1": 100, "y1": 0, "x2": 100, "y2": 200},
        BodyLineConfig(in_vector=(1.0, 0.0)),
    )
    counter.update([_track(1, 70, 100)])
    in_d, out_d = counter.update([_track(1, 130, 100)])
    assert in_d == 1 and out_d == 0


def test_vertical_line_horizontal_motion_out_left():
    counter = BodyLineCounter(
        {"x1": 100, "y1": 0, "x2": 100, "y2": 200},
        BodyLineConfig(in_vector=(1.0, 0.0)),
    )
    counter.update([_track(1, 130, 100)])
    in_d, out_d = counter.update([_track(1, 70, 100)])
    assert in_d == 0 and out_d == 1


def test_horizontal_line_down_motion():
    counter = BodyLineCounter(
        {"x1": 0, "y1": 100, "x2": 200, "y2": 100},
        BodyLineConfig(in_vector=(0.0, 1.0)),
    )
    counter.update([_track(1, 50, 80)])
    in_d, out_d = counter.update([_track(1, 50, 120)])
    assert in_d == 1 and out_d == 0


def test_diagonal_in_vector():
    # Line horizontal, IN arrow points down-right
    inv = BodyLineCounter.vector_from_drag({"x1": 0, "y1": 0, "x2": 10, "y2": 10})
    counter = BodyLineCounter(
        {"x1": 0, "y1": 100, "x2": 200, "y2": 100},
        BodyLineConfig(in_vector=inv),
    )
    counter.update([_track(1, 40, 80)])
    in_d, out_d = counter.update([_track(1, 60, 120)])
    assert in_d == 1 and out_d == 0
