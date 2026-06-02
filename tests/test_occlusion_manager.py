import time

from cv_engine.config.schema import OcclusionSchema
from cv_engine.models.detection import Detection
from cv_engine.models.track import Track, TrackState
from cv_engine.recovery.occlusion_manager import OcclusionManager


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
        velocity=(2, 0),
        state=TrackState.OBSERVATION,
    )


def test_reattach_same_id():
    mgr = OcclusionManager(OcclusionSchema(lost_track_timeout_seconds=10.0, reattach_threshold_px=50))
    lost = _track(42, 100, 200)
    mgr.register_lost(lost, [(96, 200), (98, 200), (100, 200)])

    det = Detection(bbox=(112, 190, 132, 210), confidence=0.85)
    active = []
    for frame_idx in range(11, 16):
        active = mgr.recover(active, [det], frame_idx)

    assert any(t.track_id == 42 for t in active)


def test_expire_lost():
    mgr = OcclusionManager(OcclusionSchema(lost_track_timeout_seconds=0.01, reattach_threshold_px=50))
    mgr.register_lost(_track(99, 50, 50), [(50, 50)])
    time.sleep(0.02)
    active = mgr.recover([], [], 1)
    assert active == []
