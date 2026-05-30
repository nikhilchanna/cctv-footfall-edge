"""Keep at most 15 successfully uploaded peak records per camera on edge."""

import os
from typing import Optional

from sqlalchemy.orm import Session

from app.models import MinutePeakSnapshot

MAX_UPLOADED_PEAKS_PER_CAM = 15


def prune_uploaded_peaks(cctvid: str, db: Session) -> int:
    """Delete oldest uploaded peaks (and files) beyond the retention limit."""
    uploaded = (
        db.query(MinutePeakSnapshot)
        .filter(
            MinutePeakSnapshot.cctvid == cctvid,
            MinutePeakSnapshot.uploaded_to_server == "Successful",
        )
        .order_by(MinutePeakSnapshot.uploaded_at.desc().nullslast())
        .all()
    )
    if len(uploaded) <= MAX_UPLOADED_PEAKS_PER_CAM:
        return 0

    to_remove = uploaded[MAX_UPLOADED_PEAKS_PER_CAM:]
    removed = 0
    for row in to_remove:
        if row.image_path and os.path.isfile(row.image_path):
            try:
                os.remove(row.image_path)
            except OSError:
                pass
        db.delete(row)
        removed += 1
    return removed
