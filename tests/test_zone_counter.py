import time

from cv_engine.counting.flow_counter import FlowCounter
from cv_engine.counting.zone_generator import ZoneGenerator
from cv_engine.counting.zone_manager import ZoneManager
from cv_engine.models.track import Track, TrackState


def _zones():
    return ZoneGenerator.generate_from_line(
        {"x1": 0, "y1": 300, "x2": 1280, "y2": 300},
        observation_offset=150,
        count_zone_width=100,
        ignore_offset=100,
        frame_width=1280,
        frame_height=720,
        entry_side="above",
    )


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
        bbox=(cx - 10, cy - 10, cx + 10, cy + 10),
        state=TrackState.NEW,
    )


def test_in_count():
    counter = FlowCounter(ZoneManager(_zones()), camera_role="IN")
    counter.update([_track(1, 640, 200)])
    in_d, out_d = counter.update([_track(1, 640, 300)])
    assert in_d == 1
    assert out_d == 0


def test_out_count():
    counter = FlowCounter(ZoneManager(_zones()), camera_role="OUT")
    counter.update([_track(2, 640, 200)])
    in_d, out_d = counter.update([_track(2, 640, 300)])
    assert out_d == 1


def test_no_double_count():
    counter = FlowCounter(ZoneManager(_zones()), camera_role="IN")
    counter.update([_track(3, 640, 200)])
    counter.update([_track(3, 640, 300)])
    in_d, _ = counter.update([_track(3, 640, 300)])
    assert in_d == 0
