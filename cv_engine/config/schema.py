from dataclasses import dataclass, field

from cv_engine.counting.zone_generator import ZonePolygons


@dataclass
class DetectorSchema:
    model_path: str
    detector_type: str  # head_scut | head_crowdhuman
    conf_threshold: float = 0.30
    iou_threshold: float = 0.45
    class_ids: list[int] = field(default_factory=lambda: [0])
    min_bbox_pixels: int = 6
    max_aspect_ratio: float = 4.0


@dataclass
class TrackerSchema:
    type: str = "bytetrack"
    track_thresh: float = 0.5
    track_buffer: int = 30


@dataclass
class ZoneSchema:
    auto_generate: bool = True
    observation_offset_pixels: int = 150
    count_zone_width_pixels: int = 100
    ignore_offset_pixels: int = 100
    entry_side: str = "above"
    observation_points: list[tuple[float, float]] | None = None
    count_points: list[tuple[float, float]] | None = None
    ignore_points: list[tuple[float, float]] | None = None


@dataclass
class OcclusionSchema:
    lost_track_timeout_seconds: float = 5.0
    reattach_threshold_px: float = 80.0


@dataclass
class FootfallSchema:
    camera_role: str = "IN"  # IN | OUT | occupancy_only
    count_direction: str = "both"  # in_only | out_only | both — legacy filter


@dataclass
class InferenceSchema:
    device: str = "cuda:0"
    max_concurrent: int = 2


@dataclass
class EngineSchema:
    detector: DetectorSchema
    tracker: TrackerSchema
    zones: ZoneSchema
    occlusion: OcclusionSchema
    footfall: FootfallSchema
    inference: InferenceSchema
    line_coords: dict | None = None
    zone_polygons: ZonePolygons | None = None
