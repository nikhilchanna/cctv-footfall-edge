"""Flow counting engine tests — zone transitions only, never raw detections."""

import time

import numpy as np
import pytest

from cv_engine.config import load_engine_config, resolve_zone_polygons
from cv_engine.config.schema import OcclusionSchema, ZoneSchema
from cv_engine.counting.flow_counter import FlowCounter
from cv_engine.counting.zone_generator import ZoneGenerator
from cv_engine.counting.zone_manager import ZoneManager
from cv_engine.engine import CrowdCountingEngine
from cv_engine.models.detection import Detection
from cv_engine.models.track import Track, TrackState
from cv_engine.recovery.occlusion_manager import OcclusionManager


LINE = {"x1": 0, "y1": 300, "x2": 1280, "y2": 300}


def _zones():
    return ZoneGenerator.generate_from_line(
        LINE,
        observation_offset=150,
        count_zone_width=100,
        ignore_offset=100,
        frame_width=1280,
        frame_height=720,
        entry_side="above",
    )


def _manager():
    return ZoneManager(_zones())


def _track(tid, cx, cy, *, counted=False, state=TrackState.NEW):
    now = time.monotonic()
    return Track(
        track_id=tid,
        centroid=(cx, cy),
        first_seen=now,
        last_seen=now,
        counted=counted,
        current_zone=None,
        confidence=0.9,
        bbox=(cx - 10, cy - 10, cx + 10, cy + 10),
        state=state,
    )


# 1. Single person crossing
def test_single_person_crossing_in():
    counter = FlowCounter(_manager(), camera_role="IN")
    in_d, out_d = counter.update([_track(1, 640, 200)])
    assert in_d == 0
    in_d, out_d = counter.update([_track(1, 640, 300)])
    assert in_d == 1
    assert out_d == 0


# 2. Multiple people crossing
def test_multiple_people_crossing():
    counter = FlowCounter(_manager(), camera_role="IN")
    counter.update([_track(1, 400, 200), _track(2, 800, 200)])
    in_d, _ = counter.update([_track(1, 400, 300), _track(2, 800, 300)])
    assert in_d == 2


# 3. Person stops in Observation Zone
def test_person_stops_in_observation():
    counter = FlowCounter(_manager(), camera_role="IN")
    counter.update([_track(3, 640, 200)])
    for _ in range(60):
        in_d, _ = counter.update([_track(3, 640, 200)])
        assert in_d == 0


# 4. Person stops in Count Zone
def test_person_stops_in_count_zone():
    counter = FlowCounter(_manager(), camera_role="IN")
    counter.update([_track(4, 640, 200)])
    counter.update([_track(4, 640, 300)])
    for _ in range(60):
        in_d, _ = counter.update([_track(4, 640, 300)])
        assert in_d == 0


# 5. Dense crowd moving slowly
def test_dense_crowd_moving_slowly():
    counter = FlowCounter(_manager(), camera_role="IN")
    ids = list(range(10, 20))
    for tid in ids:
        counter.update([_track(tid, 200 + tid * 50, 200)])
    total = 0
    for tid in ids:
        in_d, _ = counter.update([_track(tid, 200 + tid * 50, 300)])
        total += in_d
    assert total == len(ids)


# 6. Dense crowd completely stopped
def test_dense_crowd_stopped():
    counter = FlowCounter(_manager(), camera_role="IN")
    crowd = [_track(i, 100 + i * 60, 200) for i in range(20, 30)]
    for _ in range(30):
        in_d, _ = counter.update(crowd)
        assert in_d == 0


# 7. Occlusion recovery
def test_occlusion_recovery():
    occ = OcclusionManager(OcclusionSchema(lost_track_timeout_seconds=5.0, reattach_threshold_px=80))
    track = _track(7, 640, 250, state=TrackState.OBSERVATION)
    occ.register_lost(track, [(640, 250), (640, 260)])
    det = Detection(bbox=(630, 270, 650, 290), confidence=0.85)
    recovered = occ.recover([], [det], frame_idx=1)
    assert any(t.track_id == 7 for t in recovered)


# 8. Lost track recovery preserves counted flag
def test_lost_track_no_duplicate_count():
    counter = FlowCounter(_manager(), camera_role="IN")
    counter.update([_track(8, 640, 200)])
    counter.update([_track(8, 640, 300)])
    in_d, _ = counter.update([_track(8, 640, 300)])
    assert in_d == 0
    assert 8 in counter.counted_track_ids


# 9. Duplicate count prevention
def test_duplicate_count_prevention():
    counter = FlowCounter(_manager(), camera_role="IN")
    counter.update([_track(9, 640, 200)])
    counter.update([_track(9, 640, 300)])
    in_d, _ = counter.update([_track(9, 640, 360)])
    assert in_d == 0


# 10. Multiple lanes (two independent tracks)
def test_multiple_lanes():
    counter = FlowCounter(_manager(), camera_role="IN")
    counter.update([_track(10, 300, 200), _track(11, 900, 200)])
    in_d, _ = counter.update([_track(10, 300, 300), _track(11, 900, 300)])
    assert in_d == 2


# 11. Auto-generated zones
def test_auto_generated_zones():
    zones = ZoneGenerator.generate_from_line(
        LINE, 150, 100, 100, frame_width=1280, frame_height=720
    )
    mgr = ZoneManager(zones)
    assert mgr.zone_at((640, 200)) == "observation"
    assert mgr.zone_at((640, 300)) == "count"
    assert mgr.zone_at((640, 360)) == "ignore"


# 12. Manual polygon zones
def test_manual_polygon_zones():
    auto = _zones()
    cfg = ZoneSchema(
        auto_generate=False,
        observation_points=auto.observation,
        count_points=auto.count,
        ignore_points=auto.ignore,
    )
    resolved = resolve_zone_polygons(cfg, LINE)
    mgr = ZoneManager(resolved)
    assert mgr.zone_at((640, 200)) == "observation"


# 13. Zone editing persistence (saved polygons beat auto_generate)
def test_zone_editing_persistence():
    custom_obs = [(0, 100), (1280, 100), (1280, 200), (0, 200)]
    custom_cnt = [(0, 280), (1280, 280), (1280, 320), (0, 320)]
    custom_ign = [(0, 400), (1280, 400), (1280, 500), (0, 500)]
    cfg = ZoneSchema(
        auto_generate=True,
        observation_points=custom_obs,
        count_points=custom_cnt,
        ignore_points=custom_ign,
    )
    resolved = resolve_zone_polygons(cfg, LINE)
    mgr = ZoneManager(resolved)
    assert mgr.zone_at((640, 150)) == "observation"
    assert mgr.zone_at((640, 300)) == "count"


# 14. Camera role IN
def test_camera_role_in():
    counter = FlowCounter(_manager(), camera_role="IN")
    counter.update([_track(14, 640, 200)])
    in_d, out_d = counter.update([_track(14, 640, 300)])
    assert in_d == 1
    assert out_d == 0


# 15. Camera role OUT
def test_camera_role_out():
    counter = FlowCounter(_manager(), camera_role="OUT")
    counter.update([_track(15, 640, 200)])
    in_d, out_d = counter.update([_track(15, 640, 300)])
    assert in_d == 0
    assert out_d == 1


def test_invalid_spawn_in_count_zone():
    counter = FlowCounter(_manager(), camera_role="IN")
    in_d, _ = counter.update([_track(16, 640, 300)])
    assert in_d == 0


def test_fast_crosser_in_count_zone_not_counted():
    """No spawn rescue — must be seen in observation first."""
    counter = FlowCounter(_manager(), camera_role="IN")
    t = _track(161, 640, 300)
    t.velocity = (5.0, 3.0)
    in_d, _ = counter.update([t])
    assert in_d == 0


def test_ignore_zone_never_counted():
    counter = FlowCounter(_manager(), camera_role="IN")
    counter.update([_track(17, 640, 360)])
    in_d, _ = counter.update([_track(17, 640, 300)])
    assert in_d == 0


def test_ignore_zone_never_counted_out_camera():
    counter = FlowCounter(_manager(), camera_role="OUT")
    counter.update([_track(18, 640, 360)])
    in_d, out_d = counter.update([_track(18, 640, 300)])
    assert in_d == 0
    assert out_d == 0


def test_observation_to_count_out_camera():
    counter = FlowCounter(_manager(), camera_role="OUT")
    counter.update([_track(19, 640, 200)])
    in_d, out_d = counter.update([_track(19, 640, 300)])
    assert in_d == 0
    assert out_d == 1


def test_legacy_zone_overlay_keys():
    overlay = _manager().get_overlay()
    assert "entry" in overlay
    assert "buffer" in overlay
    assert "exit" in overlay


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
    cfg = load_engine_config({}, line_coords=LINE)
    eng = CrowdCountingEngine(cfg)
    eng._detector = MockHeadDetector(
        [Detection((100, 100, 150, 200), 0.9)]
    )
    return eng


def test_engine_integration(engine):
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    result = engine.process(frame)
    assert result.current_count >= 0
    assert 0.0 <= result.confidence <= 1.0
    overlay = engine.get_zone_overlay()
    assert "entry" in overlay
