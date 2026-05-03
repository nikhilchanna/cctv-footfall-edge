from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
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
