"""Upload minute peak JPEGs to the central analytics server."""

import logging
import os
from datetime import datetime, timezone
from typing import Optional

import requests
from sqlalchemy.orm import Session

from app.database import SessionLocal
from app.models import MinutePeakSnapshot
from app.peak_retention import prune_uploaded_peaks
from app.error_reporting import report_internal_error

logger = logging.getLogger(__name__)

PEAK_UPLOAD_URL = os.getenv(
    "PEAK_UPLOAD_URL",
    "http://localhost:8081/api/v1/peak-images",
)


def _minute_folder(minute_bucket: datetime) -> str:
    if minute_bucket.tzinfo is None:
        minute_bucket = minute_bucket.replace(tzinfo=timezone.utc)
    return minute_bucket.astimezone(timezone.utc).strftime("%Y%m%d%H%M")


def upload_peak_entry(entry: MinutePeakSnapshot, db: Session) -> bool:
    if not entry.image_path or not os.path.isfile(entry.image_path):
        entry.uploaded_to_server = "Failed"
        db.commit()
        return False

    minute_bucket = entry.minute_bucket
    captured_at = entry.captured_at or minute_bucket
    folder = _minute_folder(minute_bucket)

    try:
        with open(entry.image_path, "rb") as handle:
            files = {"file": (f"{entry.cctvid}.jpg", handle, "image/jpeg")}
            data = {
                "cctvId": entry.cctvid,
                "peopleCount": str(entry.people_count or 0),
                "minuteBucket": minute_bucket.isoformat(),
                "capturedAt": captured_at.isoformat() if captured_at else "",
                "source": entry.source or "live",
                "folder": folder,
            }
            response = requests.post(PEAK_UPLOAD_URL, files=files, data=data, timeout=30)

        if response.status_code in (200, 201):
            body = response.json() if response.content else {}
            entry.uploaded_to_server = "Successful"
            entry.server_path = body.get("relativePath") or body.get("path")
            entry.uploaded_at = datetime.now(timezone.utc)
            db.commit()
            prune_uploaded_peaks(entry.cctvid, db)
            db.commit()
            logger.info(
                "Uploaded peak for %s bucket=%s path=%s",
                entry.cctvid,
                minute_bucket,
                entry.server_path,
            )
            return True

        entry.uploaded_to_server = "Failed"
        db.commit()
        logger.warning(
            "Peak upload failed for %s: HTTP %s %s",
            entry.cctvid,
            response.status_code,
            response.text[:200],
        )
        return False
    except Exception as exc:
        entry.uploaded_to_server = "Failed"
        db.commit()
        logger.warning("Peak upload error for %s: %s", entry.cctvid, exc)
        return False


def peak_upload_job():
    db: Session = SessionLocal()
    try:
        pending = (
            db.query(MinutePeakSnapshot)
            .filter(MinutePeakSnapshot.uploaded_to_server == "Pending")
            .order_by(MinutePeakSnapshot.captured_at.asc())
            .limit(10)
            .all()
        )
        for entry in pending:
            entry.uploaded_to_server = "In-progress"
        db.commit()

        for entry in pending:
            try:
                upload_peak_entry(entry, db)
            except Exception as exc:
                report_internal_error("PeakUploadJob", str(exc))
                entry.uploaded_to_server = "Failed"
                db.commit()
    except Exception as exc:
        db.rollback()
        report_internal_error("PeakUploadJob", str(exc))
    finally:
        db.close()
