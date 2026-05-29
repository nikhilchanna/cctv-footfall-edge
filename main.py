import os
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import func
from typing import List
from contextlib import asynccontextmanager
from apscheduler.schedulers.background import BackgroundScheduler
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
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

app.mount("/media", StaticFiles(directory="."), name="media")

from fastapi.responses import FileResponse, StreamingResponse
import asyncio

async def frame_generator(cctv_id: str):
    while True:
        processor = active_processors.get(cctv_id)
        if processor and processor.last_frame:
            with processor.lock:
                frame_bytes = processor.last_frame
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        await asyncio.sleep(0.04) # Serve at ~25 FPS

@app.get("/stream_video", tags=["Media"])
def stream_video(path: str):
    """Serve any local video file by absolute path."""
    import os
    if os.path.exists(path):
        return FileResponse(path, media_type="video/mp4")
    raise HTTPException(status_code=404, detail="Video file not found")

@app.get("/stream/{cctv_id}", tags=["Media"])
def stream_cctv(cctv_id: str):
    """MJPEG stream endpoint for real-time video preview with bounding boxes."""
    if cctv_id not in active_processors:
        if cctv_id == "demo":
            logger.info("Auto-starting demo CCTV processor...")
            processor = CCTVProcessor(
                cctv_id="demo",
                cctv_name="Demo Camera",
                stream_url="demo",
                line_coords={'x1': 0, 'y1': 200, 'x2': 640, 'y2': 200},
                window_size=10
            )
            processor.start()
            active_processors["demo"] = processor
        else:
            raise HTTPException(status_code=404, detail="CCTV Processor not found or not active")
    return StreamingResponse(frame_generator(cctv_id), media_type="multipart/x-mixed-replace; boundary=frame")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
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
    """Update or Create the CCTV JSON configuration and dynamically reload processors."""
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
    
    # Dynamically apply new configuration parameters to processors
    try:
        cameras = config.config_data.get("cameras", [])
        for cam in cameras:
            cam_id = cam.get("id")
            # If processor is already running, stop it
            if cam_id in active_processors:
                logger.info(f"Stopping active CCTV processor {cam_id} for configuration update...")
                active_processors[cam_id].stop()
                active_processors[cam_id].join(timeout=2)
                del active_processors[cam_id]
            
            # Start new processor with updated line_coords and rtsp_url
            logger.info(f"Restarting CCTV processor {cam_id} with updated configurations...")
            processor = CCTVProcessor(
                cctv_id=cam_id,
                cctv_name=cam.get("name"),
                stream_url=cam.get("rtsp_url"),
                line_coords=cam.get("line_coords", {'x1': 0, 'y1': 200, 'x2': 640, 'y2': 200}),
                window_size=cam.get("window_size", 10)
            )
            processor.start()
            active_processors[cam_id] = processor
    except Exception as e:
        logger.error(f"Failed to dynamically apply updated configuration: {e}")
        
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
