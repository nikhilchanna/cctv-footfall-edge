"""Upload peak JPEGs (5-min buckets) to the central analytics server."""

import logging
import os
from datetime import datetime, timezone

import requests
from sqlalchemy.orm import Session

from app.database import SessionLocal
from app.models import MinutePeakSnapshot
from app.peak_retention import prune_uploaded_peaks
from app.error_reporting import report_internal_error
from app.peak_bucket import (
    floor_peak_bucket,
    peak_bucket_folder,
    peak_bucket_is_closed,
)

logger = logging.getLogger(__name__)

PEAK_UPLOAD_URL = os.getenv(
    "PEAK_UPLOAD_URL",
    "http://localhost:8081/api/v1/peak-images",
)

MAX_UPLOAD_RETRIES = 5
BASE_RETRY_SECONDS = 60
STUCK_IN_PROGRESS_SECONDS = 300


def _retry_backoff_seconds(retry_ctr: int) -> int:
    """Exponential backoff capped at 8 minutes."""
    return min(BASE_RETRY_SECONDS * (2 ** min(retry_ctr, 3)), 480)


def _ready_for_retry(entry: MinutePeakSnapshot, now: datetime) -> bool:
    status = entry.uploaded_to_server or "Pending"
    if status == "Pending":
        return True
    if status == "Successful":
        return False

    retry_ctr = entry.upload_retry_ctr or 0
    if retry_ctr >= MAX_UPLOAD_RETRIES:
        return False

    last = entry.last_upload_attempt
    if status == "In-progress":
        if last is None:
            return True
        return (now - last).total_seconds() >= STUCK_IN_PROGRESS_SECONDS

    if status == "Failed":
        if last is None:
            return True
        return (now - last).total_seconds() >= _retry_backoff_seconds(retry_ctr)

    return False


def upload_peak_entry(entry: MinutePeakSnapshot, db: Session) -> bool:
    now = datetime.now(timezone.utc)
    entry.last_upload_attempt = now

    if not entry.image_path or not os.path.isfile(entry.image_path):
        entry.uploaded_to_server = "Failed"
        entry.upload_retry_ctr = (entry.upload_retry_ctr or 0) + 1
        db.commit()
        return False

    minute_bucket = floor_peak_bucket(entry.minute_bucket)
    entry.minute_bucket = minute_bucket

    if not peak_bucket_is_closed(minute_bucket, now):
        entry.uploaded_to_server = "Pending"
        db.commit()
        return False

    captured_at = entry.captured_at or minute_bucket
    folder = peak_bucket_folder(minute_bucket)

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
            entry.uploaded_at = now
            entry.upload_retry_ctr = 0
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
        entry.upload_retry_ctr = (entry.upload_retry_ctr or 0) + 1
        db.commit()
        logger.warning(
            "Peak upload failed for %s: HTTP %s %s (retry %s/%s)",
            entry.cctvid,
            response.status_code,
            response.text[:200],
            entry.upload_retry_ctr,
            MAX_UPLOAD_RETRIES,
        )
        return False
    except Exception as exc:
        entry.uploaded_to_server = "Failed"
        entry.upload_retry_ctr = (entry.upload_retry_ctr or 0) + 1
        db.commit()
        logger.warning(
            "Peak upload error for %s: %s (retry %s/%s)",
            entry.cctvid,
            exc,
            entry.upload_retry_ctr,
            MAX_UPLOAD_RETRIES,
        )
        return False


def peak_upload_job():
    db: Session = SessionLocal()
    try:
        now = datetime.now(timezone.utc)
        candidates = (
            db.query(MinutePeakSnapshot)
            .filter(
                MinutePeakSnapshot.uploaded_to_server.in_(
                    ["Pending", "Failed", "In-progress"]
                )
            )
            .order_by(MinutePeakSnapshot.captured_at.asc())
            .limit(50)
            .all()
        )
        pending = [
            e
            for e in candidates
            if _ready_for_retry(e, now)
            and peak_bucket_is_closed(floor_peak_bucket(e.minute_bucket), now)
        ][:10]
        if not pending:
            return

        for entry in pending:
            entry.uploaded_to_server = "In-progress"
        db.commit()

        for entry in pending:
            try:
                upload_peak_entry(entry, db)
            except Exception as exc:
                report_internal_error("PeakUploadJob", str(exc))
                entry.uploaded_to_server = "Failed"
                entry.upload_retry_ctr = (entry.upload_retry_ctr or 0) + 1
                entry.last_upload_attempt = datetime.now(timezone.utc)
                db.commit()
    except Exception as exc:
        db.rollback()
        report_internal_error("PeakUploadJob", str(exc))
    finally:
        db.close()
