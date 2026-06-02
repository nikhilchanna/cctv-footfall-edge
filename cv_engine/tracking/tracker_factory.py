from cv_engine.config.schema import TrackerSchema
from cv_engine.tracking.base import Tracker
from cv_engine.tracking.bytetrack_tracker import ByteTrackTracker


def build_tracker(cfg: TrackerSchema) -> Tracker:
    t = (cfg.type or "bytetrack").lower()
    if t in ("bytetrack", "botsort"):
        return ByteTrackTracker(cfg)
    raise ValueError(f"unsupported tracker type: {cfg.type}")
