from dataclasses import dataclass, field
from enum import Enum


class TrackState(str, Enum):
    NEW = "NEW"
    OBSERVATION = "OBSERVATION"
    COUNT_ZONE = "COUNT_ZONE"
    COUNTED = "COUNTED"
    REJECTED = "REJECTED"


@dataclass
class Track:
    track_id: int
    centroid: tuple[float, float]
    first_seen: float
    last_seen: float
    counted: bool
    current_zone: str | None
    confidence: float
    bbox: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    velocity: tuple[float, float] = (0.0, 0.0)
    state: TrackState = TrackState.NEW
    # Legacy counters / video test still read these
    direction: int = 0
    age: int = 0
