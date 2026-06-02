from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime

class CctvConfigBase(BaseModel):
    config_data: Dict[str, Any] = Field(..., description="JSON configuration for CCTV cameras")

class CctvConfigCreate(CctvConfigBase):
    pass

class CctvConfigResponse(CctvConfigBase):
    id: int
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True

class ErrorReport(BaseModel):
    source: str = Field(..., description="Source of the error (e.g., 'DataProcessorThread', 'ApiCallingThread')")
    error_message: str
    traceback: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ProcessorCursorStatus(BaseModel):
    last_processed_at: Optional[str] = None
    last_minute_bucket: Optional[str] = None
    mode: str = "live"
    backfill_from: Optional[str] = None
    backfill_to: Optional[str] = None


class ProcessorStatusResponse(BaseModel):
    cctv_id: str
    cctv_name: str = ""
    source_type: str = ""
    processing_state: str = "running"  # running | paused | stopped
    frames_this_minute: int = 0
    current_peak_people: int = 0
    last_snapshot_latency_ms: float = 0.0
    consecutive_failures: int = 0
    window_in: int = 0
    window_out: int = 0
    stream_fps: float = 7.0
    cursor: Optional[ProcessorCursorStatus] = None
    active_footfall_path: str = "body_zones"
    footfall_mode: str = "hybrid"
    camera_role: str = "footfall"
    count_direction: str = "both"
    density_level: str = "LOW"


class ClearCameraDataResponse(BaseModel):
    cctv_id: str
    footfall_rows_deleted: int
    peak_rows_deleted: int
    cursor_reset: bool


class VideoTestZonesRequest(BaseModel):
    line_coords: Dict[str, float] = Field(..., description="Counting line {x1,y1,x2,y2}")
    width: int
    height: int
    observation_offset_pixels: int = Field(150, ge=20, le=500)
    count_zone_width_pixels: int = Field(100, ge=20, le=300)
    ignore_offset_pixels: int = Field(100, ge=20, le=500)
    entry_side: str = Field("above", description="above | below — far side is observation")


class VideoTestStartRequest(BaseModel):
    video_path: str = Field(..., description="Absolute path to MP4 on edge host")
    line_coords: Dict[str, float] = Field(..., description="Counting line {x1,y1,x2,y2}")
    camera_role: str = Field("IN", description="IN | OUT | occupancy_only")
    count_direction: str = Field("both", description="in_only | out_only | both")
    entry_side: str = Field("above", description="above | below")
    observation_offset_pixels: int = Field(150, ge=20, le=500)
    count_zone_width_pixels: int = Field(100, ge=20, le=300)
    ignore_offset_pixels: int = Field(100, ge=20, le=500)
    head_conf_threshold: float = Field(0.22, ge=0.1, le=0.9)
    zones: Optional[Dict[str, Any]] = Field(
        None,
        description="Optional saved polygons observation/count/ignore with points arrays",
    )
    cv_engine: Optional[Dict[str, Any]] = None
    # Legacy fields ignored by new pipeline
    polygon: Optional[List[List[float]]] = None
    in_vector: Optional[Dict[str, int]] = None
    out_vector: Optional[Dict[str, int]] = None
    count_mode: Optional[str] = None


class VideoTestPreviewRequest(BaseModel):
    video_path: str = Field(..., description="Absolute path to MP4 on edge host")
