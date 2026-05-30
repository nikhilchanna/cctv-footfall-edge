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

from app.database import engine, get_db, SessionLocal, init_db
import app.models as models
import app.schemas as schemas
import app.tasks as tasks
from app.cv_processor import CCTVProcessor
from app import hikvision_preview
from app.hikvision_snapshot import substream_channel_id
from app import hikvision_playback
from app.camera_status import sync_configured_cameras, upsert_camera_status
from app.peak_upload import peak_upload_job
import logging

# Set up a basic logger for error reporting
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Dictionary to hold running CCTV processors
active_processors = {}
_suspended_for_preview: list = []


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
    )
    processor.start()
    return processor


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
        if cam_id in active_processors:
            continue

        logger.info("Resuming CCTV processor %s after DVR preview", cam_id)
        dvr_config = _get_dvr_config(SessionLocal())
        processor = _start_processor(cam, dvr_config)
        active_processors[cam_id] = processor

    _suspended_for_preview = []

# Create tables and apply pending SQL migrations on startup.
init_db()

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

    scheduler.add_job(
        peak_upload_job,
        trigger=IntervalTrigger(seconds=20),
        id="peak_upload_thread",
        name="Peak Image Upload Thread",
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
            dvr_config = config_record.config_data.get("dvr", {})
            for cam in cameras:
                cam_id = cam.get("id")
                processor = _start_processor(cam, dvr_config)
                active_processors[cam_id] = processor
                _maybe_schedule_backfill(cam, dvr_config)
            logger.info(f"Started {len(active_processors)} CCTV Processors.")
            if cameras:
                sync_configured_cameras(cameras, db=db)
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
        _suspended_for_preview = []
        dvr_config = config.config_data.get("dvr", {})
        for cam in cameras:
            cam_id = cam.get("id")
            if cam_id in active_processors:
                logger.info(f"Stopping active CCTV processor {cam_id} for configuration update...")
                active_processors[cam_id].stop()
                active_processors[cam_id].join(timeout=2)
                del active_processors[cam_id]

            logger.info(f"Restarting CCTV processor {cam_id} with updated configurations...")
            processor = _start_processor(cam, dvr_config)
            active_processors[cam_id] = processor
            _maybe_schedule_backfill(cam, dvr_config)
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
def get_processor_status(cctv_id: str):
    processor = active_processors.get(cctv_id)
    if not processor:
        raise HTTPException(status_code=404, detail="CCTV Processor not found or not active")
    return processor.get_status()


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
    rows = db.query(models.CameraStatus).all()
    return [
        {
            "cctvid": r.cctvid,
            "cctvname": r.cctvname,
            "status": r.status,
            "message": r.message,
            "detail": r.detail,
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
            "processor_active": proc is not None,
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


@app.get("/dvr/frame/{channel_id}", tags=["DVR HMI"])
def dvr_preview_frame(channel_id: str):
    """Return latest cached JPEG for line-drawing UI (instant, no blocking)."""
    if channel_id not in hikvision_preview.camera_store and channel_id != "demo":
        raise HTTPException(status_code=404, detail="Camera not found")

    hikvision_preview.ensure_preview(channel_id)

    frame_data = hikvision_preview.get_cached_frame(channel_id)

    if frame_data:
        return Response(
            frame_data,
            media_type="image/jpeg",
            headers={"Cache-Control": "no-store"},
        )

    return Response(status_code=204)
