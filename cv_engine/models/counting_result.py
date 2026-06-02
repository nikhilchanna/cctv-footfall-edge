from dataclasses import dataclass, field

from cv_engine.models.detection import Detection
from cv_engine.models.track import Track


@dataclass
class CountingResult:
    in_count_delta: int
    out_count_delta: int
    confidence: float
    active_tracks: list[Track] = field(default_factory=list)
    current_count: int = 0
    density_level: str = "LOW"
    detections: list[Detection] = field(default_factory=list)

    @property
    def active_track_count(self) -> int:
        return len(self.active_tracks)
