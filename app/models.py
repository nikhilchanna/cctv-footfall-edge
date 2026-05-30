from sqlalchemy import Column, Integer, String, DateTime, JSON, UniqueConstraint
from sqlalchemy.sql import func
from app.database import Base

class DataTracker(Base):
    __tablename__ = "data_tracker_table"
    __table_args__ = (
        UniqueConstraint(
            "cctvid", "starttime", "endtime", name="uq_tracker_window"
        ),
    )

    id = Column(Integer, primary_key=True, index=True)
    ctr_in = Column(Integer, default=0)
    ctr_out = Column(Integer, default=0)
    timewindow = Column(Integer, default=10) # Time window in seconds
    starttime = Column(DateTime(timezone=True))
    endtime = Column(DateTime(timezone=True))
    cctvid = Column(String, index=True)
    cctvname = Column(String)
    
    # Statuses: "Pending", "In-progress", "Successful", "Failed"
    data_to_server_ack = Column(String, default="Pending") 
    
    createdAt = Column(DateTime(timezone=True), server_default=func.now())
    last_api_call = Column(DateTime(timezone=True), nullable=True)
    api_call_ctr = Column(Integer, default=0)
    lastupdated = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

class CctvConfig(Base):
    """
    To store the externalized JSON config for CCTV cameras.
    This allows updating via UI easily.
    """
    __tablename__ = "cctv_config"

    id = Column(Integer, primary_key=True, index=True)
    config_data = Column(JSON, nullable=False) # Store the JSON payload
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class ProcessingCursor(Base):
    __tablename__ = "processing_cursor"

    cctvid = Column(String, primary_key=True)
    last_processed_at = Column(DateTime(timezone=True), nullable=True)
    last_minute_bucket = Column(DateTime(timezone=True), nullable=True)
    last_live_seq = Column(Integer, default=0)
    mode = Column(String, default="live")
    backfill_from = Column(DateTime(timezone=True), nullable=True)
    backfill_to = Column(DateTime(timezone=True), nullable=True)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class MinutePeakSnapshot(Base):
    __tablename__ = "minute_peak_snapshot"
    __table_args__ = (
        UniqueConstraint("cctvid", "minute_bucket", name="uq_minute_peak"),
    )

    id = Column(Integer, primary_key=True, index=True)
    cctvid = Column(String, index=True, nullable=False)
    minute_bucket = Column(DateTime(timezone=True), nullable=False)
    people_count = Column(Integer, default=0)
    image_path = Column(String, nullable=True)
    captured_at = Column(DateTime(timezone=True), nullable=True)
    source = Column(String, default="live")
    uploaded_to_server = Column(String, default="Pending")  # Pending, Successful, Failed
    server_path = Column(String, nullable=True)
    uploaded_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class CameraStatus(Base):
    __tablename__ = "camera_status"

    cctvid = Column(String, primary_key=True)
    cctvname = Column(String, nullable=True)
    status = Column(String, default="configured")  # configured, connected, disconnected, no_data, auth_failed, error
    message = Column(String, nullable=True)
    detail = Column(String, nullable=True)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
