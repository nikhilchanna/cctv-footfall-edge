"""
Hikvision NVR recording backfill for minute peak snapshots (not footfall counts).
"""

import logging
import os
import threading
import time
import uuid
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional

import cv2
import numpy as np
import requests
from requests.auth import HTTPDigestAuth
from sqlalchemy.orm import Session
from ultralytics import YOLO

from app.database import SessionLocal
from app.hikvision_snapshot import fetch_snapshot_bytes
from app.models import MinutePeakSnapshot, ProcessingCursor

logger = logging.getLogger(__name__)

BACKFILL_THROTTLE_SECONDS = 0.4
_yolo_model = None
_yolo_lock = threading.Lock()


def _get_yolo():
    global _yolo_model
    with _yolo_lock:
        if _yolo_model is None:
            _yolo_model = YOLO("yolov8n.pt")
        return _yolo_model


def _format_hik_time(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_playback_uri(search_xml: bytes) -> Optional[str]:
    try:
        root = ET.fromstring(search_xml)
    except ET.ParseError:
        return None

    for elem in root.iter():
        if elem.tag.endswith("playbackURI") and elem.text:
            return elem.text.strip()
    return None


def search_recording_playback_uri(
    credentials: Dict[str, str],
    channel_id: str,
    start: datetime,
    end: datetime,
) -> Optional[str]:
    ip = credentials.get("ip")
    username = credentials.get("username")
    password = credentials.get("password")
    if not ip or not username:
        return None

    search_id = str(uuid.uuid4())
    body = f"""<?xml version="1.0" encoding="UTF-8"?>
<CMSearchDescription>
  <searchID>{search_id}</searchID>
  <trackList><trackID>{channel_id}</trackID></trackList>
  <timeSpanList>
    <timeSpan>
      <startTime>{_format_hik_time(start)}</startTime>
      <endTime>{_format_hik_time(end)}</endTime>
    </timeSpan>
  </timeSpanList>
  <maxResults>5</maxResults>
  <searchResultPostion>0</searchResultPostion>
  <metadataList>
    <metadataDescriptor>//recordType.meta.std-cgi.com</metadataDescriptor>
  </metadataList>
</CMSearchDescription>"""

    try:
        response = requests.post(
            f"http://{ip}/ISAPI/ContentMgmt/search",
            data=body.encode("utf-8"),
            auth=HTTPDigestAuth(username, password),
            headers={"Content-Type": "application/xml"},
            timeout=15,
            verify=False,
        )
        if response.status_code != 200:
            logger.warning(
                "ContentMgmt search failed channel=%s status=%s",
                channel_id,
                response.status_code,
            )
            return None
        return _parse_playback_uri(response.content)
    except Exception as exc:
        logger.warning("ContentMgmt search error channel=%s: %s", channel_id, exc)
        return None


def download_playback_jpeg(
    credentials: Dict[str, str],
    playback_uri: str,
) -> Optional[bytes]:
    ip = credentials.get("ip")
    username = credentials.get("username")
    password = credentials.get("password")
    if not ip or not username:
        return None

    url = (
        f"http://{ip}/ISAPI/ContentMgmt/download"
        f"?playbackURI={requests.utils.quote(playback_uri, safe='')}"
    )
    try:
        response = requests.get(
            url,
            auth=HTTPDigestAuth(username, password),
            timeout=20,
            verify=False,
        )
        if response.status_code == 200 and response.content.startswith(b"\xff\xd8"):
            return response.content
    except Exception as exc:
        logger.warning("Playback download error: %s", exc)
    return None


def count_people(jpeg_bytes: bytes) -> int:
    nparr = np.frombuffer(jpeg_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        return 0

    model = _get_yolo()
    results = model(frame, classes=[0], verbose=False)
    count = 0
    for result in results:
        count += len(result.boxes)
    return count


def upsert_minute_peak(
    db: Session,
    cctvid: str,
    minute_bucket: datetime,
    people_count: int,
    image_path: str,
    captured_at: datetime,
    source: str,
) -> None:
    existing = (
        db.query(MinutePeakSnapshot)
        .filter(
            MinutePeakSnapshot.cctvid == cctvid,
            MinutePeakSnapshot.minute_bucket == minute_bucket,
        )
        .first()
    )

    if existing:
        if people_count > existing.people_count or (
            people_count == existing.people_count
            and captured_at >= existing.captured_at
        ):
            existing.people_count = people_count
            existing.image_path = image_path
            existing.captured_at = captured_at
            existing.source = source
    else:
        db.add(
            MinutePeakSnapshot(
                cctvid=cctvid,
                minute_bucket=minute_bucket,
                people_count=people_count,
                image_path=image_path,
                captured_at=captured_at,
                source=source,
                uploaded_to_server="Pending",
            )
        )


def backfill_missing_minutes(
    cctvid: str,
    channel_id: str,
    credentials: Dict[str, str],
    from_time: datetime,
    to_time: datetime,
) -> int:
    """Backfill minute peak snapshots for [from_time, to_time). Returns minutes filled."""
    if from_time >= to_time:
        return 0

    os.makedirs("snapshots/peaks", exist_ok=True)
    filled = 0
    cursor = from_time.replace(second=0, microsecond=0)
    end = to_time.replace(second=0, microsecond=0)

    db: Session = SessionLocal()
    try:
        proc_cursor = (
            db.query(ProcessingCursor).filter(ProcessingCursor.cctvid == cctvid).first()
        )
        if proc_cursor:
            proc_cursor.mode = "backfill"
            proc_cursor.backfill_from = from_time
            proc_cursor.backfill_to = to_time
            db.commit()

        while cursor < end:
            minute_end = cursor + timedelta(minutes=1)
            playback_uri = search_recording_playback_uri(
                credentials, channel_id, cursor, minute_end
            )

            people_count = 0
            image_path = ""
            captured_at = cursor

            if playback_uri:
                jpeg = download_playback_jpeg(credentials, playback_uri)
                if jpeg:
                    people_count = count_people(jpeg)
                    stamp = cursor.strftime("%Y%m%d%H%M")
                    image_path = f"snapshots/peaks/{cctvid}_{stamp}.jpg"
                    with open(image_path, "wb") as handle:
                        handle.write(jpeg)
                    filled += 1

            if image_path:
                upsert_minute_peak(
                    db,
                    cctvid,
                    cursor,
                    people_count,
                    image_path,
                    captured_at,
                    "backfill",
                )
                db.commit()

            cursor = minute_end
            time.sleep(BACKFILL_THROTTLE_SECONDS)

        if proc_cursor:
            proc_cursor.mode = "live"
            proc_cursor.backfill_from = None
            proc_cursor.backfill_to = None
            db.commit()
    except Exception as exc:
        db.rollback()
        logger.error("Backfill failed for %s: %s", cctvid, exc)
    finally:
        db.close()

    return filled


def schedule_backfill_if_needed(
    cctvid: str,
    channel_id: str,
    credentials: Dict[str, str],
) -> None:
    """Start background backfill for minute peaks when cursor is behind current time."""

    def worker():
        db: Session = SessionLocal()
        try:
            cursor = (
                db.query(ProcessingCursor)
                .filter(ProcessingCursor.cctvid == cctvid)
                .first()
            )
            if not cursor or not cursor.last_processed_at:
                return

            now = datetime.now(timezone.utc)
            gap_start = cursor.last_processed_at.replace(second=0, microsecond=0)
            gap_end = now.replace(second=0, microsecond=0)
            if gap_end <= gap_start + timedelta(minutes=1):
                return

            logger.info(
                "Starting minute peak backfill for %s from %s to %s",
                cctvid,
                gap_start,
                gap_end,
            )
            backfill_missing_minutes(cctvid, channel_id, credentials, gap_start, gap_end)
        finally:
            db.close()

    threading.Thread(
        target=worker, daemon=True, name=f"backfill-{cctvid}"
    ).start()
