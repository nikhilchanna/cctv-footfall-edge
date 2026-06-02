import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import func
from typing import List, Optional
from contextlib import asynccontextmanager
from apscheduler.schedulers.background import BackgroundScheduler
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from apscheduler.triggers.interval import IntervalTrigger
from apscheduler.triggers.cron import CronTrigger

from app.database import engine, get_db, SessionLocal, init_db
import app.models as models
import app.schemas as schemas
import app.tasks as tasks
from app.cv_processor import CCTVProcessor
from app import hikvision_preview
from app.hikvision_snapshot import fetch_snapshot_bytes, substream_channel_id
from app import hikvision_playback
from app.camera_status import sync_configured_cameras, upsert_camera_status
from app.camera_data import clear_camera_data, reset_live_processor_counters
from app.peak_upload import peak_upload_job
from app.video_test_runner import VideoTestSession, grab_first_frame
import logging
import time

# Set up a basic logger for error reporting
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import torch

_cv_device = os.getenv("CV_ENGINE_DEVICE", "cuda:0")
if torch.cuda.is_available():
    logger.info("CUDA enabled: %s (%s)", _cv_device, torch.cuda.get_device_name(0))
else:
    logger.warning(
        "CUDA requested (%s) but no GPU driver — using cpu. Run scripts/install-nvidia-driver.sh then reboot.",
        _cv_device,
    )

# Stagger ISAPI restarts so DVR is not hit by all cameras at once
PROCESSOR_RESTART_STAGGER_SEC = 0.25

# Dictionary to hold running CCTV processors
active_processors = {}
_suspended_for_preview: list = []
# User halted cameras — watchdog and config reload must not restart these
_halted_processors: dict[str, str] = {}  # cam_id -> "paused" | "stopped"
_video_test_session: VideoTestSession | None = None
_processors_before_video_test: list[dict] = []


def _get_config_record(db: Session):
    return db.query(models.CctvConfig).first()


def _get_dvr_config(db: Session) -> dict:
    record = _get_config_record(db)
    if record and record.config_data:
        return record.config_data.get("dvr", {})
    return {}


def _persist_dvr_credentials(db: Session, ip: str, username: str, password: str):
    record = _get_config_record(db)
    config_data = dict(record.config_data) if record and record.config_data else {}
    config_data["dvr"] = {"ip": ip, "username": username, "password": password}
    if record:
        record.config_data = config_data
    else:
        record = models.CctvConfig(config_data=config_data)
        db.add(record)
    db.commit()


def _build_source_config(cam: dict) -> dict:
    source_type = cam.get("source_type")
    if not source_type:
        rtsp = cam.get("rtsp_url") or cam.get("rtsp")
        if rtsp == "demo":
            source_type = "demo"
        elif rtsp and (str(rtsp).startswith("rtsp://") or str(rtsp).startswith("http://")):
            source_type = "rtsp"
        else:
            source_type = "isapi"

    channel_id = cam.get("channel_id") or cam.get("id")
    if source_type == "isapi" and channel_id and str(channel_id).endswith("01"):
        channel_id = substream_channel_id(str(channel_id))

    channel_num = cam.get("channel")
    if channel_num is None and channel_id and str(channel_id).isdigit():
        channel_num = int(str(channel_id)) // 100

    return {
        "type": source_type,
        "channel_id": channel_id,
        "poll_fps": cam.get("poll_fps", 7),
        "channel": channel_num or 1,
        "stream_type": cam.get("stream_type", 2),
    }


def _start_processor(cam: dict, dvr_config: dict) -> CCTVProcessor:
    source_config = _build_source_config(cam)
    processor = CCTVProcessor(
        cctv_id=cam.get("id"),
        cctv_name=cam.get("name"),
        line_coords=cam.get("line_coords", {"x1": 0, "y1": 200, "x2": 640, "y2": 200}),
        window_size=cam.get("window_size", 10),
        stream_url=cam.get("rtsp_url") or cam.get("rtsp") or "demo",
        source_config=source_config,
        dvr_config=dvr_config,
        cv_engine_config=cam.get("cv_engine"),
    )
    processor.start()
    return processor


def _suspend_processors_for_video_test(db: Session) -> int:
    """Stop live processors so YOLO is not called from two threads (Mac segfault)."""
    global _processors_before_video_test
    record = _get_config_record(db)
    _processors_before_video_test = []
    if record and record.config_data:
        for cam in record.config_data.get("cameras", []):
            cam_id = cam.get("id")
            if cam_id and cam_id in active_processors:
                _processors_before_video_test.append(cam)
    stopped = 0
    for cam_id in list(active_processors.keys()):
        logger.info("Pausing CCTV processor %s for video test", cam_id)
        _stop_processor(cam_id)
        stopped += 1
    return stopped


def _resume_processors_after_video_test() -> int:
    global _processors_before_video_test
    if not _processors_before_video_test:
        return 0
    db = SessionLocal()
    resumed = 0
    try:
        dvr_config = _get_dvr_config(db)
        for cam in _processors_before_video_test:
            cam_id = cam.get("id")
            if not cam_id or cam_id in _halted_processors:
                continue
            if _restart_processor_safe(cam, dvr_config):
                resumed += 1
    finally:
        db.close()
        _processors_before_video_test = []
    return resumed


def _stop_processor(cam_id: str) -> None:
    processor = active_processors.pop(cam_id, None)
    if not processor:
        return
    processor.stop()
    processor.join(timeout=2)


def _find_camera_config(cam_id: str, db: Session) -> dict | None:
    record = _get_config_record(db)
    if not record or not record.config_data:
        return None
    for cam in record.config_data.get("cameras", []):
        if cam.get("id") == cam_id:
            return cam
    return None


def _processing_state(cam_id: str) -> str:
    if cam_id in _halted_processors:
        return _halted_processors[cam_id]
    proc = active_processors.get(cam_id)
    if proc is not None and proc.is_alive():
        return "running"
    return "stopped"


def _halt_processor(cam_id: str, mode: str, *, cctvname: str | None = None) -> None:
    if mode not in ("paused", "stopped"):
        raise ValueError(f"invalid halt mode: {mode}")
    _halted_processors[cam_id] = mode
    _stop_processor(cam_id)
    upsert_camera_status(
        cam_id,
        mode,
        cctvname=cctvname,
        message=f"Processing {mode} by user",
        detail=None,
    )
    logger.info("Processor %s for camera %s", mode, cam_id)


def _resume_processor(cam_id: str, db: Session) -> bool:
    _halted_processors.pop(cam_id, None)
    cam = _find_camera_config(cam_id, db)
    if not cam:
        return False
    dvr_config = _get_dvr_config(db)
    return _restart_processor_safe(cam, dvr_config, force=True)


def _restart_processor_safe(cam: dict, dvr_config: dict, *, force: bool = False) -> bool:
    cam_id = cam.get("id")
    if not cam_id:
        return False
    if not force and cam_id in _halted_processors:
        logger.info("Skipping restart for halted camera %s", cam_id)
        return False
    try:
        _stop_processor(cam_id)
        active_processors[cam_id] = _start_processor(cam, dvr_config)
        _maybe_schedule_backfill(cam, dvr_config)
        logger.info("Processor running for camera %s", cam_id)
        return True
    except Exception as exc:
        logger.error("Failed to start processor %s: %s", cam_id, exc)
        upsert_camera_status(
            cam_id,
            "error",
            cctvname=cam.get("name"),
            message="Processor failed to start",
            detail=str(exc),
        )
        return False


def _sync_active_processors(db: Optional[Session] = None) -> int:
    """Restart configured cameras whose processor thread is missing or dead."""
    if _suspended_for_preview or _video_test_session is not None:
        return 0

    own_db = db is None
    if own_db:
        db = SessionLocal()
    restarted = 0
    try:
        record = _get_config_record(db)
        if not record or not record.config_data:
            return 0
        cameras = record.config_data.get("cameras", [])
        dvr_config = record.config_data.get("dvr", {})

        for cam in cameras:
            cam_id = cam.get("id")
            if not cam_id:
                continue
            if cam_id in _halted_processors:
                continue
            proc = active_processors.get(cam_id)
            if proc is not None and proc.is_alive():
                continue
            if _restart_processor_safe(cam, dvr_config):
                restarted += 1
    finally:
        if own_db:
            db.close()
    return restarted


def _maybe_schedule_backfill(cam: dict, dvr_config: dict):
    source_config = _build_source_config(cam)
    if source_config.get("type") != "isapi" or not dvr_config.get("ip"):
        return
    channel_id = source_config.get("channel_id") or cam.get("id")
    hikvision_playback.schedule_backfill_if_needed(
        cam.get("id"), channel_id, dvr_config
    )


def _suspend_processors_for_dvr_preview(db: Session):
    """Stop RTSP processors so DVR ISAPI snapshots are not starved."""
    global _suspended_for_preview

    if _suspended_for_preview:
        return

    db_config = db.query(models.CctvConfig).first()
    if db_config and db_config.config_data:
        _suspended_for_preview = list(
            db_config.config_data.get("cameras", [])
        )

    for cam_id, processor in list(active_processors.items()):
        logger.info("Pausing CCTV processor %s for DVR line-drawing preview", cam_id)
        processor.stop()
        processor.join(timeout=2)
        del active_processors[cam_id]


def _resume_processors_after_dvr_preview():
    """Restart RTSP processors after DVR preview ends."""
    global _suspended_for_preview

    if not _suspended_for_preview:
        return

    for cam in _suspended_for_preview:
        cam_id = cam.get("id")
        if not cam_id or cam_id in active_processors:
            continue
        if cam_id in _halted_processors:
            continue
        logger.info("Resuming CCTV processor %s after DVR preview", cam_id)
        dvr_config = _get_dvr_config(SessionLocal())
        _restart_processor_safe(cam, dvr_config)

    _suspended_for_preview = []

# Create tables and apply pending SQL migrations on startup.
init_db()

@asynccontextmanager
def _restore_dvr_preview_session(db: Session):
    """Re-connect HMI preview session from saved config after edge restart."""
    dvr = _get_dvr_config(db)
    ip = dvr.get("ip")
    if not ip or ip == "demo":
        return
    username = dvr.get("username")
    password = dvr.get("password")
    if not username or not password:
        return
    try:
        info = hikvision_preview.validate_and_connect(ip, username, password)
        if info:
            cameras = hikvision_preview.discover_cameras()
            hikvision_preview.register_cameras(cameras)
            logger.info(
                "Restored DVR preview session for %s (%s channels)",
                ip,
                len(cameras),
            )
    except Exception as exc:
        logger.warning("DVR preview restore on startup failed: %s", exc)


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

    scheduler.add_job(
        peak_upload_job,
        trigger=IntervalTrigger(seconds=60),
        id="peak_upload_thread",
        name="Peak Image Upload Thread",
        replace_existing=True
    )

    def processor_watchdog():
        try:
            n = _sync_active_processors()
            if n:
                logger.info("Processor watchdog restarted %s camera(s)", n)
        except Exception as exc:
            logger.warning("Processor watchdog failed: %s", exc)

    scheduler.add_job(
        processor_watchdog,
        trigger=IntervalTrigger(seconds=30),
        id="processor_watchdog",
        name="Processor Watchdog",
        replace_existing=True,
    )
    
    scheduler.start()
    logger.info("Background jobs scheduled.")
    
    # Startup: Initialize CCTV Processors
    db = SessionLocal()
    try:
        config_record = db.query(models.CctvConfig).first()
        if config_record and config_record.config_data:
            cameras = config_record.config_data.get("cameras", [])
            dvr_config = config_record.config_data.get("dvr", {})
            for cam in cameras:
                cam_id = cam.get("id")
                processor = _start_processor(cam, dvr_config)
                active_processors[cam_id] = processor
                _maybe_schedule_backfill(cam, dvr_config)
            logger.info(f"Started {len(active_processors)} CCTV Processors.")
            if cameras:
                sync_configured_cameras(cameras, db=db)
            _restore_dvr_preview_session(db)
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

    hikvision_preview.stop_preview()
    hikvision_preview.close_client()

app = FastAPI(title="Footfall Counter Service", version="1.0.0", lifespan=lifespan)

app.mount("/media", StaticFiles(directory="."), name="media")

_static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.isdir(_static_dir):
    app.mount("/edge-ui", StaticFiles(directory=_static_dir, html=True), name="edge-ui")

from fastapi.responses import FileResponse, StreamingResponse, Response
import asyncio

async def frame_generator(cctv_id: str):
    while True:
        processor = active_processors.get(cctv_id)
        interval = 1.0 / 7
        if processor:
            interval = processor.get_stream_interval()
            if processor.last_frame:
                with processor.lock:
                    frame_bytes = processor.last_frame
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
                )
        await asyncio.sleep(interval)

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
                line_coords={"x1": 0, "y1": 200, "x2": 640, "y2": 200},
                window_size=10,
                stream_url="demo",
                source_config={"type": "demo"},
            )
            processor.start()
            active_processors["demo"] = processor
        else:
            raise HTTPException(status_code=404, detail="CCTV Processor not found or not active")
    return StreamingResponse(frame_generator(cctv_id), media_type="multipart/x-mixed-replace; boundary=frame")


async def video_test_frame_generator():
    while True:
        session = _video_test_session
        interval = 0.05
        if session and session.running:
            session.tick()
            if session.last_frame:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + session.last_frame + b"\r\n"
                )
            interval = session.get_stream_interval()
        await asyncio.sleep(interval)


@app.get("/test", tags=["Test"])
def test_page_redirect():
    """Video test UI — analytics on MP4, no DB."""
    page = os.path.join(os.path.dirname(__file__), "static", "video_test.html")
    if os.path.isfile(page):
        return FileResponse(page)
    raise HTTPException(status_code=404, detail="Test page missing")


@app.post("/test/video/preview", tags=["Test"])
def video_test_preview(req: schemas.VideoTestPreviewRequest):
    """First frame for line drawing — no analytics, no DB."""
    result = grab_first_frame(req.video_path)
    if not result.get("ok"):
        raise HTTPException(status_code=400, detail=result.get("error", "Preview failed"))
    return result


@app.post("/test/video/zones", tags=["Test"])
def video_test_zones(req: schemas.VideoTestZonesRequest):
    """Preview auto-generated observation / count / ignore zones from a line."""
    from cv_engine.counting.zone_generator import ZoneGenerator

    lc = req.line_coords
    line_len = (
        (lc.get("x2", 0) - lc.get("x1", 0)) ** 2
        + (lc.get("y2", 0) - lc.get("y1", 0)) ** 2
    ) ** 0.5
    if line_len < 10:
        raise HTTPException(status_code=400, detail="Line too short")

    try:
        zones = ZoneGenerator.generate_from_line(
            lc,
            req.observation_offset_pixels,
            req.count_zone_width_pixels,
            req.ignore_offset_pixels,
            frame_width=req.width,
            frame_height=req.height,
            entry_side=req.entry_side,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "observation": zones.observation,
        "count": zones.count,
        "ignore": zones.ignore,
        "entry": zones.entry,
        "buffer": zones.buffer,
        "exit": zones.exit,
    }


@app.post("/test/video/start", tags=["Test"])
def start_video_test(req: schemas.VideoTestStartRequest, db: Session = Depends(get_db)):
    global _video_test_session
    if not req.line_coords:
        raise HTTPException(status_code=400, detail="Draw counting line first")

    lc = req.line_coords
    line_len = (
        (lc.get("x2", 0) - lc.get("x1", 0)) ** 2
        + (lc.get("y2", 0) - lc.get("y1", 0)) ** 2
    ) ** 0.5
    if line_len < 10:
        raise HTTPException(status_code=400, detail="Counting line too short — drag a longer line")

    if _video_test_session:
        _video_test_session.stop()
        _video_test_session = None

    paused = _suspend_processors_for_video_test(db)
    manual_zones = req.zones if req.zones else None
    session = VideoTestSession(
        video_path=req.video_path,
        line_coords=req.line_coords,
        camera_role=req.camera_role,
        count_direction=req.count_direction,
        entry_side=req.entry_side,
        observation_offset_pixels=req.observation_offset_pixels,
        count_zone_width_pixels=req.count_zone_width_pixels,
        ignore_offset_pixels=req.ignore_offset_pixels,
        head_conf_threshold=req.head_conf_threshold,
        manual_zones=manual_zones,
        cv_engine_config=req.cv_engine,
    )
    session.start()
    if not session.running:
        _resume_processors_after_video_test()
        raise HTTPException(status_code=400, detail=session.stats.get("error", "Failed to start test"))

    _video_test_session = session
    return {
        "success": True,
        "video_path": session.video_path,
        "no_db": True,
        "live_processors_paused": paused,
    }


@app.post("/test/video/stop", tags=["Test"])
def stop_video_test():
    global _video_test_session
    if _video_test_session:
        _video_test_session.stop()
        _video_test_session = None
    resumed = _resume_processors_after_video_test()
    return {"success": True, "live_processors_resumed": resumed}


@app.get("/test/video/status", tags=["Test"])
def video_test_status():
    session = _video_test_session
    if not session:
        return {"running": False, "no_db": True}
    return session.stats


@app.get("/test/video/stream", tags=["Test"])
def video_test_stream():
    if not _video_test_session or not _video_test_session.running:
        raise HTTPException(status_code=404, detail="No video test running")
    return StreamingResponse(
        video_test_frame_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )

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

    cameras = config.config_data.get("cameras", [])
    try:
        sync_configured_cameras(cameras, db=db)
    except Exception as e:
        logger.warning("Failed to sync camera status rows: %s", e)
    
    # Dynamically apply new configuration parameters to processors
    try:
        global _suspended_for_preview

        hikvision_preview.stop_preview()
        dvr_config = config.config_data.get("dvr", {})
        for idx, cam in enumerate(cameras):
            cam_id = cam.get("id")
            if not cam_id:
                continue
            if idx > 0:
                time.sleep(PROCESSOR_RESTART_STAGGER_SEC)
            logger.info("Restarting CCTV processor %s with updated configurations", cam_id)
            _restart_processor_safe(cam, dvr_config)
        _suspended_for_preview = []
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


@app.get("/processor/{cctv_id}/status", response_model=schemas.ProcessorStatusResponse, tags=["Analytics"])
def get_processor_status(cctv_id: str, db: Session = Depends(get_db)):
    halted = _halted_processors.get(cctv_id)
    if halted:
        cam = _find_camera_config(cctv_id, db)
        return {
            "cctv_id": cctv_id,
            "cctv_name": (cam or {}).get("name", cctv_id),
            "source_type": (cam or {}).get("source_type", ""),
            "processing_state": halted,
        }

    processor = active_processors.get(cctv_id)
    if not processor or not processor.is_alive():
        _sync_active_processors(db)
        processor = active_processors.get(cctv_id)

    if processor and processor.is_alive():
        data = processor.get_status()
        data["processing_state"] = "running"
        return data

    cam = _find_camera_config(cctv_id, db)
    if cam:
        return {
            "cctv_id": cctv_id,
            "cctv_name": cam.get("name", cctv_id),
            "source_type": cam.get("source_type", ""),
            "processing_state": "stopped",
        }
    raise HTTPException(status_code=404, detail="CCTV Processor not found or not active")


@app.post("/processor/{cctv_id}/pause", tags=["Analytics"])
def pause_processor(cctv_id: str, db: Session = Depends(get_db)):
    cam = _find_camera_config(cctv_id, db)
    if not cam:
        raise HTTPException(status_code=404, detail="Camera not in configuration")
    _halt_processor(cctv_id, "paused", cctvname=cam.get("name"))
    return {"success": True, "cctv_id": cctv_id, "processing_state": "paused"}


@app.post("/processor/{cctv_id}/stop", tags=["Analytics"])
def stop_processor_route(cctv_id: str, db: Session = Depends(get_db)):
    cam = _find_camera_config(cctv_id, db)
    if not cam:
        raise HTTPException(status_code=404, detail="Camera not in configuration")
    _halt_processor(cctv_id, "stopped", cctvname=cam.get("name"))
    return {"success": True, "cctv_id": cctv_id, "processing_state": "stopped"}


@app.post("/processor/{cctv_id}/resume", tags=["Analytics"])
def resume_processor(cctv_id: str, db: Session = Depends(get_db)):
    cam = _find_camera_config(cctv_id, db)
    if not cam:
        raise HTTPException(status_code=404, detail="Camera not in configuration")
    if not _resume_processor(cctv_id, db):
        raise HTTPException(status_code=500, detail="Failed to resume processor")
    return {"success": True, "cctv_id": cctv_id, "processing_state": "running"}


@app.post("/processors/pause-all", tags=["Analytics"])
def pause_all_processors(db: Session = Depends(get_db)):
    record = _get_config_record(db)
    if not record or not record.config_data:
        return {"success": True, "paused": []}
    paused = []
    for cam in record.config_data.get("cameras", []):
        cam_id = cam.get("id")
        if not cam_id:
            continue
        _halt_processor(cam_id, "paused", cctvname=cam.get("name"))
        paused.append(cam_id)
    return {"success": True, "paused": paused}


@app.post("/processors/resume-all", tags=["Analytics"])
def resume_all_processors(db: Session = Depends(get_db)):
    record = _get_config_record(db)
    if not record or not record.config_data:
        return {"success": True, "resumed": []}
    dvr_config = record.config_data.get("dvr", {})
    resumed = []
    for cam in record.config_data.get("cameras", []):
        cam_id = cam.get("id")
        if not cam_id:
            continue
        _halted_processors.pop(cam_id, None)
        if _restart_processor_safe(cam, dvr_config, force=True):
            resumed.append(cam_id)
    return {"success": True, "resumed": resumed}


@app.delete(
    "/processor/{cctv_id}/data",
    response_model=schemas.ClearCameraDataResponse,
    tags=["Analytics"],
)
def clear_processor_data(cctv_id: str, db: Session = Depends(get_db)):
    """Delete footfall windows, peak snapshots, and processing cursor for one camera."""
    cam = _find_camera_config(cctv_id, db)
    if not cam:
        raise HTTPException(status_code=404, detail="Camera not in configuration")

    result = clear_camera_data(cctv_id, db)
    proc = active_processors.get(cctv_id)
    if proc and proc.is_alive():
        reset_live_processor_counters(proc)

    return result


@app.get("/processor/{cctv_id}/minute-peaks", tags=["Analytics"])
def get_minute_peaks(cctv_id: str, limit: int = 15, db: Session = Depends(get_db)):
    """Recent peak images for UI (newest first, up to 15)."""
    rows = (
        db.query(models.MinutePeakSnapshot)
        .filter(models.MinutePeakSnapshot.cctvid == cctv_id)
        .order_by(models.MinutePeakSnapshot.captured_at.desc().nullslast())
        .limit(min(limit, 15))
        .all()
    )
    return [
        {
            "id": row.id,
            "minute_bucket": row.minute_bucket.isoformat() if row.minute_bucket else None,
            "people_count": row.people_count,
            "image_path": row.image_path,
            "source": row.source,
            "uploaded_to_server": row.uploaded_to_server,
            "uploaded_at": row.uploaded_at.isoformat() if row.uploaded_at else None,
            "server_path": row.server_path,
        }
        for row in rows
    ]


@app.get("/processor/{cctv_id}/thumbnail", tags=["Analytics"])
def get_processor_thumbnail(cctv_id: str):
    """Latest annotated frame as JPEG (for Analytics tiles, not MJPEG)."""
    processor = active_processors.get(cctv_id)
    if processor and processor.last_frame:
        with processor.lock:
            return Response(
                processor.last_frame,
                media_type="image/jpeg",
                headers={"Cache-Control": "no-store"},
            )
    snapshot_path = os.path.join("snapshots", f"snapshot_{cctv_id}.jpg")
    if os.path.isfile(snapshot_path):
        return FileResponse(snapshot_path, media_type="image/jpeg")
    raise HTTPException(status_code=404, detail="No thumbnail available")


@app.get("/cameras/status", tags=["Analytics"])
def get_cameras_status(db: Session = Depends(get_db)):
    _sync_active_processors(db)
    rows = db.query(models.CameraStatus).all()
    return [
        {
            "cctvid": r.cctvid,
            "cctvname": r.cctvname,
            "status": r.status,
            "message": r.message,
            "detail": r.detail,
            "processing_state": _processing_state(r.cctvid)
            if r.cctvid != "_dvr"
            else None,
            "updated_at": r.updated_at.isoformat() if r.updated_at else None,
        }
        for r in rows
    ]


@app.get("/analytics/summary", tags=["Analytics"])
def get_analytics_summary(db: Session = Depends(get_db)):
    """Per-camera counts and status for edge HTML dashboard."""
    config_record = db.query(models.CctvConfig).first()
    cameras = []
    if config_record and config_record.config_data:
        cameras = config_record.config_data.get("cameras", [])

    status_map = {r.cctvid: r for r in db.query(models.CameraStatus).all()}
    result = []
    for cam in cameras:
        cid = cam.get("id")
        if not cid:
            continue
        totals = db.query(
            func.sum(models.DataTracker.ctr_in).label("total_in"),
            func.sum(models.DataTracker.ctr_out).label("total_out"),
        ).filter(models.DataTracker.cctvid == cid).first()
        st = status_map.get(cid)
        proc = active_processors.get(cid)
        result.append({
            "cctvid": cid,
            "cctvname": cam.get("name"),
            "total_in": totals.total_in or 0 if totals else 0,
            "total_out": totals.total_out or 0 if totals else 0,
            "processor_active": proc is not None and proc.is_alive(),
            "processing_state": _processing_state(cid),
            "status": st.status if st else "unknown",
            "message": st.message if st else None,
            "window_in": proc.ctr_in if proc else 0,
            "window_out": proc.ctr_out if proc else 0,
        })
    return result

from pydantic import BaseModel
import requests
from requests.auth import HTTPDigestAuth
from urllib.parse import quote

class DvrConnectRequest(BaseModel):
    ip: str
    username: str
    password: str


class PreviewStartRequest(BaseModel):
    channel_id: str


def _connect_dvr_legacy(req: DvrConnectRequest):
    """Legacy ISAPI XML discovery fallback when pyHik is unavailable."""

    logger.info(f"Authenticating with Hikvision DVR via ISAPI (legacy): {req.ip}")
    response = requests.get(
        f"http://{req.ip}/ISAPI/System/deviceInfo",
        auth=HTTPDigestAuth(req.username, req.password),
        timeout=5,
        verify=False,
    )

    if response.status_code != 200:
        upsert_camera_status(
            "_dvr",
            "auth_failed",
            message="DVR authentication failed",
            detail=f"HTTP {response.status_code}",
        )
        return {
            "success": False,
            "detail": f"DVR authentication failed with status code {response.status_code}",
        }

    xml_response = requests.get(
        f"http://{req.ip}/ISAPI/Streaming/channels",
        auth=HTTPDigestAuth(req.username, req.password),
        timeout=5,
        verify=False,
    )

    cameras = []
    encoded_password = quote(req.password)
    seen_channels = set()

    if xml_response.status_code == 200:
        import xml.etree.ElementTree as ET
        root = ET.fromstring(xml_response.content)
        namespace = ""
        if "xmlns" in root.tag:
            namespace = root.tag.split("}")[0] + "}"

        for channel in root.findall(f".//{namespace}StreamingChannel"):
            id_elem = channel.find(f"{namespace}id")
            name_elem = channel.find(f"{namespace}channelName")
            if id_elem is None:
                continue

            channel_id = id_elem.text
            if not channel_id.endswith("01"):
                continue

            cam_no = int(channel_id) // 100
            if cam_no in seen_channels:
                continue
            seen_channels.add(cam_no)

            name = name_elem.text if name_elem is not None else f"Camera {cam_no}"
            rtsp_url = (
                f"rtsp://{req.username}:{encoded_password}@{req.ip}:554"
                f"/Streaming/Channels/{channel_id}"
            )

            cameras.append({
                "camera_name": name,
                "camera_number": cam_no,
                "channel_id": channel_id,
                "channel": cam_no,
                "stream_type": 1,
                "rtsp": rtsp_url,
            })

    if not cameras:
        for cam_no in range(1, 9):
            channel_id = f"{cam_no}01"
            rtsp_url = (
                f"rtsp://{req.username}:{encoded_password}@{req.ip}:554"
                f"/Streaming/Channels/{channel_id}"
            )
            cameras.append({
                "camera_name": f"DVR Camera {cam_no}",
                "camera_number": cam_no,
                "channel_id": channel_id,
                "channel": cam_no,
                "stream_type": 1,
                "rtsp": rtsp_url,
            })

    hikvision_preview.dvr_credentials["ip"] = req.ip
    hikvision_preview.dvr_credentials["username"] = req.username
    hikvision_preview.dvr_credentials["password"] = req.password
    hikvision_preview.register_cameras(cameras)

    try:
        db = SessionLocal()
        _persist_dvr_credentials(db, req.ip, req.username, req.password)
    finally:
        db.close()

    return {
        "success": True,
        "cameras": cameras,
        "backend": "ISAPI legacy",
    }


@app.post("/dvr/connect", tags=["DVR HMI"])
def connect_dvr(req: DvrConnectRequest):
    """Validate Hikvision DVR connection via ISAPI and auto-populate available camera channels."""
    if req.ip == "demo":
        logger.info("Initializing simulated demo DVR channels...")
        demo_cameras = [
            {
                "camera_name": "Demo Entrance Camera",
                "camera_number": 1,
                "channel_id": "demo",
                "channel": 0,
                "stream_type": 1,
                "rtsp": "demo",
            }
        ]
        hikvision_preview.register_cameras(demo_cameras)
        return {
            "success": True,
            "cameras": demo_cameras,
            "backend": "demo",
        }

    try:
        info = hikvision_preview.validate_and_connect(
            req.ip, req.username, req.password
        )
        if info:
            cameras = hikvision_preview.discover_cameras()
            hikvision_preview.register_cameras(cameras)
            try:
                db = SessionLocal()
                _persist_dvr_credentials(db, req.ip, req.username, req.password)
            finally:
                db.close()
            logger.info(
                "Discovered %s channels via pyHik for %s",
                len(cameras),
                req.ip,
            )
            return {
                "success": True,
                "cameras": cameras,
                "device": info.get("deviceName", req.ip),
                "backend": "pyHik ISAPIClient",
            }
    except Exception as e:
        logger.warning("pyHik connect failed, using legacy ISAPI: %s", e)
        if "401" in str(e) or "403" in str(e) or "auth" in str(e).lower():
            upsert_camera_status(
                "_dvr",
                "auth_failed",
                message="DVR authentication failed",
                detail=str(e),
            )

    try:
        return _connect_dvr_legacy(req)
    except Exception as e:
        logger.error(f"Failed to auto-discover DVR channels: {e}")
        upsert_camera_status(
            "_dvr",
            "error",
            message="Failed to connect to DVR",
            detail=str(e),
        )
        return {
            "success": False,
            "detail": f"Failed to connect to DVR: {str(e)}",
        }


@app.post("/dvr/preview/start", tags=["DVR HMI"])
def dvr_preview_start(req: PreviewStartRequest, db: Session = Depends(get_db)):
    """Start lightweight ISAPI snapshot polling for line-drawing (one camera)."""
    _suspend_processors_for_dvr_preview(db)

    if not hikvision_preview.start_preview(req.channel_id):
        _resume_processors_after_dvr_preview()
        raise HTTPException(
            status_code=404,
            detail=f"Could not start preview for channel {req.channel_id}",
        )
    return {"success": True, "channel_id": req.channel_id}


@app.post("/dvr/preview/stop", tags=["DVR HMI"])
def dvr_preview_stop():
    """Stop snapshot polling and release DVR preview connection."""
    hikvision_preview.stop_preview()
    _resume_processors_after_dvr_preview()
    return {"success": True}


@app.get("/dvr/session", tags=["DVR HMI"])
def dvr_session_status():
    """Return whether a DVR login session is still active (credentials + discovered channels)."""
    return hikvision_preview.get_session_status()


@app.get("/dvr/snapshot/{channel_id}", tags=["DVR HMI"])
def dvr_channel_snapshot(channel_id: str):
    """One-off ISAPI JPEG for channel picker thumbnails (no preview session)."""
    if channel_id == "demo":
        demo_path = Path("sample.mp4")
        if demo_path.is_file():
            import cv2

            cap = cv2.VideoCapture(str(demo_path))
            ok, frame = cap.read()
            cap.release()
            if ok and frame is not None:
                import cv2 as cv

                ok_enc, buf = cv.imencode(".jpg", frame, [cv.IMWRITE_JPEG_QUALITY, 70])
                if ok_enc:
                    return Response(buf.tobytes(), media_type="image/jpeg")
        raise HTTPException(status_code=404, detail="Demo snapshot unavailable")

    if channel_id not in hikvision_preview.camera_store:
        raise HTTPException(status_code=404, detail="Camera not found")

    snap_id = substream_channel_id(channel_id)
    data = fetch_snapshot_bytes(snap_id, hikvision_preview.dvr_credentials)
    if not data:
        raise HTTPException(status_code=503, detail="Snapshot unavailable")
    return Response(data, media_type="image/jpeg", headers={"Cache-Control": "no-store"})


@app.get("/dvr/frame/{channel_id}", tags=["DVR HMI"])
def dvr_preview_frame(channel_id: str):
    """Return latest cached JPEG for line-drawing UI (instant, no blocking)."""
    if channel_id not in hikvision_preview.camera_store and channel_id != "demo":
        raise HTTPException(status_code=404, detail="Camera not found")

    # Only serve frames from an active preview session started via /dvr/preview/start.
    # Auto-starting here used to poll ISAPI alongside analytics and starve the DVR.
    frame_data = hikvision_preview.get_cached_frame(channel_id)

    if frame_data:
        return Response(
            frame_data,
            media_type="image/jpeg",
            headers={"Cache-Control": "no-store"},
        )

    return Response(status_code=204)
