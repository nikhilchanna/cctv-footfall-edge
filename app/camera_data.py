"""Delete per-camera analytics rows and peak JPEGs on edge."""

import logging
import os

from sqlalchemy.orm import Session

from app.models import DataTracker, MinutePeakSnapshot, ProcessingCursor

logger = logging.getLogger(__name__)


def clear_camera_data(cctv_id: str, db: Session) -> dict:
    peaks = (
        db.query(MinutePeakSnapshot)
        .filter(MinutePeakSnapshot.cctvid == cctv_id)
        .all()
    )
    for row in peaks:
        if row.image_path and os.path.isfile(row.image_path):
            try:
                os.remove(row.image_path)
            except OSError as exc:
                logger.warning("Could not delete peak file %s: %s", row.image_path, exc)

    peak_deleted = (
        db.query(MinutePeakSnapshot)
        .filter(MinutePeakSnapshot.cctvid == cctv_id)
        .delete(synchronize_session=False)
    )
    tracker_deleted = (
        db.query(DataTracker)
        .filter(DataTracker.cctvid == cctv_id)
        .delete(synchronize_session=False)
    )
    cursor = (
        db.query(ProcessingCursor)
        .filter(ProcessingCursor.cctvid == cctv_id)
        .first()
    )
    cursor_reset = cursor is not None
    if cursor:
        db.delete(cursor)

    db.commit()
    logger.info(
        "Cleared data for %s: footfall=%s peaks=%s cursor=%s",
        cctv_id,
        tracker_deleted,
        peak_deleted,
        cursor_reset,
    )
    return {
        "cctv_id": cctv_id,
        "footfall_rows_deleted": tracker_deleted,
        "peak_rows_deleted": peak_deleted,
        "cursor_reset": cursor_reset,
    }


def reset_live_processor_counters(processor) -> None:
    """Zero in-memory window counters after DB clear."""
    processor.ctr_in = 0
    processor.ctr_out = 0
    processor.frames_this_minute = 0
    processor.stats["frames_this_minute"] = 0
    processor.stats["current_peak_people"] = 0
    processor.minute_state = {
        "bucket": None,
        "peak_count": 0,
        "best_jpeg": None,
        "best_at": None,
    }
