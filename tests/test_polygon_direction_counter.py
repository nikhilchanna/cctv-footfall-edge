import time

from cv_engine.counting.polygon_direction_counter import PolygonDirectionCounter, PolygonDirectionConfig
from cv_engine.models.detection import Detection
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


def _head(cx, cy):
    return Detection((cx - 8, cy - 8, cx + 8, cy + 8), 0.9, 0, "head_scut")


def test_inside_motion_in():
    poly = [(0, 0), (200, 0), (200, 200), (0, 200)]
    counter = PolygonDirectionCounter(poly, (1.0, 0.0))
    counter.update_tracks([_track(1, 50, 100)])
    in_d, out_d = counter.update_tracks([_track(1, 80, 100)])
    assert in_d == 1 and out_d == 0


def test_inside_motion_out():
    poly = [(0, 0), (200, 0), (200, 200), (0, 200)]
    counter = PolygonDirectionCounter(poly, (1.0, 0.0))
    counter.update_tracks([_track(1, 80, 100)])
    in_d, out_d = counter.update_tracks([_track(1, 50, 100)])
    assert in_d == 0 and out_d == 1


def test_outside_no_count():
    poly = [(0, 0), (100, 0), (100, 100), (0, 100)]
    counter = PolygonDirectionCounter(poly, (1.0, 0.0))
    counter.update_tracks([_track(1, 150, 50)])
    in_d, out_d = counter.update_tracks([_track(1, 180, 50)])
    assert in_d == 0 and out_d == 0


def test_enter_from_outside_no_immediate_count():
    """First frame inside must not count motion from outside."""
    poly = [(0, 0), (200, 0), (200, 200), (0, 200)]
    counter = PolygonDirectionCounter(poly, (1.0, 0.0))
    counter.update_tracks([_track(1, 250, 100)])
    in_d, out_d = counter.update_tracks([_track(1, 50, 100)])
    assert in_d == 0 and out_d == 0
    in_d, out_d = counter.update_tracks([_track(1, 80, 100)])
    assert in_d == 1 and out_d == 0


def test_motion_only_when_both_frames_inside():
    poly = [(0, 0), (200, 0), (200, 200), (0, 200)]
    counter = PolygonDirectionCounter(poly, (0.0, 1.0))
    counter.update_tracks([_track(1, 100, 50)])
    in_d, out_d = counter.update_tracks([_track(1, 100, 80)])
    assert in_d == 1 and out_d == 0
    counter.update_tracks([_track(1, 100, 250)])
    in_d, out_d = counter.update_tracks([_track(1, 100, 50)])
    assert in_d == 0 and out_d == 0


def test_explicit_out_vector():
    poly = [(0, 0), (200, 0), (200, 200), (0, 200)]
    counter = PolygonDirectionCounter(poly, (1.0, 0.0), (-1.0, 0.0))
    counter.update_tracks([_track(1, 100, 100)])
    in_d, out_d = counter.update_tracks([_track(1, 60, 100)])
    assert in_d == 0 and out_d == 1
