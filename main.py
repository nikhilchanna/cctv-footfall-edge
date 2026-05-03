import os
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import func
from typing import List
from contextlib import asynccontextmanager
from apscheduler.schedulers.background import BackgroundScheduler
from fastapi.middleware.cors import CORSMiddleware
from apscheduler.triggers.interval import IntervalTrigger
from apscheduler.triggers.cron import CronTrigger

from app.database import engine, Base, get_db, SessionLocal
import app.models as models
import app.schemas as schemas
import app.tasks as tasks
from app.cv_processor import CCTVProcessor
import logging

# Set up a basic logger for error reporting
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Dictionary to hold running CCTV processors
active_processors = {}

# Initialize the database and create tables if they don't exist
Base.metadata.create_all(bind=engine)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Setup Background Scheduler
    scheduler = BackgroundScheduler()
    
    # Task 1: API Calling Thread (runs every 10 seconds, adjust as needed)
    scheduler.add_job(
        tasks.api_calling_thread_job, 
        trigger=IntervalTrigger(seconds=10),
        id="api_calling_thread",
        name="API Calling Thread",
        replace_existing=True
    )
    
    # Task 2: Retry Failed Thread (runs every 30 seconds, adjust as needed)
    scheduler.add_job(
        tasks.retry_failed_thread_job,
        trigger=IntervalTrigger(seconds=30),
        id="retry_failed_thread",
        name="Retry Failed Thread",
        replace_existing=True
    )
    
    # Task 3: Daily Cleanup Thread (runs daily at 1:00 AM IST)
    scheduler.add_job(
        tasks.daily_cleanup_job,
        trigger=CronTrigger(hour=1, minute=0, timezone='Asia/Kolkata'),
        id="daily_cleanup_thread",
        name="Daily Cleanup Thread",
        replace_existing=True
    )
    
    scheduler.start()
    logger.info("Background jobs scheduled.")
    
    # Startup: Initialize CCTV Processors
    db = SessionLocal()
    try:
        config_record = db.query(models.CctvConfig).first()
        if config_record and config_record.config_data:
            cameras = config_record.config_data.get("cameras", [])
            for cam in cameras:
                cam_id = cam.get("id")
                processor = CCTVProcessor(
                    cctv_id=cam_id,
                    cctv_name=cam.get("name"),
                    stream_url=cam.get("rtsp_url"),
                    line_coords=cam.get("line_coords", {'x1': 0, 'y1': 200, 'x2': 640, 'y2': 200}),
                    window_size=cam.get("window_size", 10)
                )
                processor.start()
                active_processors[cam_id] = processor
            logger.info(f"Started {len(active_processors)} CCTV Processors.")
    except Exception as e:
        logger.error(f"Failed to start CCTV Processors: {e}")
    finally:
        db.close()
    
    yield
    
    # Shutdown
    scheduler.shutdown()
    logger.info("Background jobs shut down.")
    
    # Stop all CCTV processors
    for cam_id, processor in active_processors.items():
        processor.stop()
        processor.join(timeout=2)
    logger.info("All CCTV Processors stopped.")

app = FastAPI(title="Footfall Counter Service", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health", tags=["Telemetry"])
def health_check():
    """Telemetry API to check service health."""
    # In the future, we can add thread health statuses here.
    return {
        "status": "healthy",
        "service": "Footfall Counter Backend",
        "database": "connected"
    }

@app.post("/report-error", tags=["Error Reporting"])
def report_error(error: schemas.ErrorReport):
    """API to report and log errors from any background thread."""
    logger.error(f"Error from {error.source}: {error.error_message} | Traceback: {error.traceback}")
    # We could also save this to a database table if needed later.
    return {"status": "Error logged successfully"}

@app.post("/config", response_model=schemas.CctvConfigResponse, tags=["CCTV Configuration"])
def update_cctv_config(config: schemas.CctvConfigCreate, db: Session = Depends(get_db)):
    """Update or Create the CCTV JSON configuration."""
    db_config = db.query(models.CctvConfig).first()
    if db_config:
        # Update existing
        db_config.config_data = config.config_data
    else:
        # Create new
        db_config = models.CctvConfig(config_data=config.config_data)
        db.add(db_config)
    
    db.commit()
    db.refresh(db_config)
    return db_config

@app.get("/config", response_model=schemas.CctvConfigResponse, tags=["CCTV Configuration"])
def get_cctv_config(db: Session = Depends(get_db)):
    """Fetch the current CCTV JSON configuration."""
    db_config = db.query(models.CctvConfig).first()
    if not db_config:
        raise HTTPException(status_code=404, detail="Configuration not found")
    return db_config

@app.get("/totals/{cctv_id}", tags=["Analytics"])
def get_total_counts(cctv_id: str, db: Session = Depends(get_db)):
    """Fetch the total cumulative In and Out counts for a specific CCTV camera."""
    totals = db.query(
        func.sum(models.DataTracker.ctr_in).label('total_in'),
        func.sum(models.DataTracker.ctr_out).label('total_out')
    ).filter(models.DataTracker.cctvid == cctv_id).first()
    
    return {
        "cctv_id": cctv_id,
        "total_in": totals.total_in or 0,
        "total_out": totals.total_out or 0
    }
