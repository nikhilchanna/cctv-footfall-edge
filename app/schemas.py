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


class MinutePeakStatus(BaseModel):
    minute_bucket: str
    people_count: int
    image_path: Optional[str] = None
    source: str = "live"


class ProcessorStatusResponse(BaseModel):
    cctv_id: str
    cctv_name: str
    source_type: str
    frames_this_minute: int = 0
    current_peak_people: int = 0
    last_snapshot_latency_ms: float = 0.0
    consecutive_failures: int = 0
    window_in: int = 0
    window_out: int = 0
    stream_fps: float = 7.0
    cursor: ProcessorCursorStatus
    last_saved_peak: Optional[MinutePeakStatus] = None
