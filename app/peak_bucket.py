"""Peak snapshot window — one JPEG per camera per bucket (default 5 min)."""

import os
from datetime import datetime, timedelta, timezone
from typing import Optional

PEAK_BUCKET_MINUTES = int(os.getenv("PEAK_BUCKET_MINUTES", "5"))


def normalize_peak_time(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def floor_peak_bucket(dt: datetime) -> datetime:
    """Snap to start of current peak window in UTC."""
    dt = normalize_peak_time(dt).replace(second=0, microsecond=0)
    floored_minute = (dt.minute // PEAK_BUCKET_MINUTES) * PEAK_BUCKET_MINUTES
    return dt.replace(minute=floored_minute)


def peak_bucket_end(bucket_start: datetime) -> datetime:
    return floor_peak_bucket(bucket_start) + timedelta(minutes=PEAK_BUCKET_MINUTES)


def peak_bucket_is_closed(bucket_start: datetime, now: Optional[datetime] = None) -> bool:
    """True once the bucket window has finished (safe to upload)."""
    now = normalize_peak_time(now or datetime.now(timezone.utc))
    return now >= peak_bucket_end(bucket_start)


def peak_bucket_folder(bucket_start: datetime) -> str:
    return floor_peak_bucket(bucket_start).strftime("%Y%m%d%H%M")


def peak_image_filename(cctv_id: str, bucket_start: datetime) -> str:
    return f"{cctv_id}_{peak_bucket_folder(bucket_start)}.jpg"
