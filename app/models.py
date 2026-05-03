from sqlalchemy import Column, Integer, String, DateTime, JSON
from sqlalchemy.sql import func
from app.database import Base

class DataTracker(Base):
    __tablename__ = "data_tracker_table"

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
