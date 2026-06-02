import numpy as np

from cv_engine.config.schema import TrackerSchema
from cv_engine.models.detection import Detection
from cv_engine.tracking.bytetrack_tracker import ByteTrackTracker


def _det(x, y, w=40, h=80, conf=0.9):
    return Detection(bbox=(x, y, x + w, y + h), confidence=conf)


def test_bytetrack_stable_ids():
    tracker = ByteTrackTracker(TrackerSchema(type="bytetrack", track_thresh=0.3, track_buffer=30))
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    t1 = tracker.update([_det(100, 200)], frame)
    t2 = tracker.update([_det(102, 200)], frame)
    t3 = tracker.update([_det(104, 200)], frame)

    ids = [t.track_id for t in t1 + t2 + t3 if t.track_id >= 0]
    assert len(set(ids)) >= 1
    if t1 and t2:
        assert t1[0].track_id == t2[0].track_id
