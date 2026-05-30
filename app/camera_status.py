"""Persist and query per-camera health/status for UI alerts."""

from datetime import datetime, timezone
from typing import Optional

from sqlalchemy.orm import Session

from app.database import SessionLocal
from app.models import CameraStatus


def upsert_camera_status(
    cctvid: str,
    status: str,
    *,
    cctvname: Optional[str] = None,
    message: Optional[str] = None,
    detail: Optional[str] = None,
    db: Optional[Session] = None,
) -> None:
    own_db = db is None
    if own_db:
        db = SessionLocal()
    try:
        row = db.query(CameraStatus).filter(CameraStatus.cctvid == cctvid).first()
        if not row:
            row = CameraStatus(cctvid=cctvid)
            db.add(row)
        row.status = status
        if cctvname is not None:
            row.cctvname = cctvname
        if message is not None:
            row.message = message
        if detail is not None:
            row.detail = detail
        row.updated_at = datetime.now(timezone.utc)
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        if own_db:
            db.close()


def sync_configured_cameras(cameras: list, db: Optional[Session] = None) -> None:
    """Mark all configured cameras; remove status rows for removed cams."""
    own_db = db is None
    if own_db:
        db = SessionLocal()
    try:
        configured_ids = {c.get("id") for c in cameras if c.get("id")}
        for cam in cameras:
            cid = cam.get("id")
            if not cid:
                continue
            upsert_camera_status(
                cid,
                "configured",
                cctvname=cam.get("name"),
                message="Camera configured",
                db=db,
            )
        for row in db.query(CameraStatus).all():
            if row.cctvid not in configured_ids:
                db.delete(row)
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        if own_db:
            db.close()
