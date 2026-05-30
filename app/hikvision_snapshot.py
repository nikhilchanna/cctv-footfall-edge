"""
Shared Hikvision ISAPI snapshot client for preview and analytics ingestion.
"""

import logging
import threading
import time
from typing import Any, Dict, Optional

import cv2
import numpy as np
import requests
import urllib3
from requests.auth import HTTPDigestAuth

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

logger = logging.getLogger(__name__)

SNAPSHOT_WIDTH = 1280
SNAPSHOT_HEIGHT = 720


def substream_channel_id(channel_id: str) -> str:
    """Map main stream channel (101) to substream (102) for smaller JPEGs."""
    if channel_id.endswith("01"):
        return f"{channel_id[:-2]}02"
    return channel_id


def fetch_snapshot_bytes(
    channel_id: str,
    credentials: Dict[str, str],
    *,
    width: int = SNAPSHOT_WIDTH,
    height: int = SNAPSHOT_HEIGHT,
    timeout: float = 8,
) -> Optional[bytes]:
    ip = credentials.get("ip")
    username = credentials.get("username")
    password = credentials.get("password")

    if not ip or not username:
        return None

    url = (
        f"http://{ip}/ISAPI/Streaming/channels/{channel_id}/picture"
        f"?videoResolutionWidth={width}"
        f"&videoResolutionHeight={height}"
    )

    try:
        response = requests.get(
            url,
            auth=HTTPDigestAuth(username, password),
            timeout=timeout,
            verify=False,
        )
        if (
            response.status_code == 200
            and response.content
            and response.content.startswith(b"\xff\xd8")
        ):
            return response.content

        logger.warning(
            "Snapshot failed channel=%s status=%s bytes=%s",
            channel_id,
            response.status_code,
            len(response.content) if response.content else 0,
        )
    except Exception as exc:
        logger.warning("Snapshot error channel=%s: %s", channel_id, exc)

    return None


def fetch_snapshot_pyhik(
    channel: int,
    stream_type: int,
    isapi_client: Any,
) -> Optional[bytes]:
    if not isapi_client:
        return None

    try:
        data = isapi_client.get_snapshot(channel=channel, stream_type=stream_type)
        if data and data.startswith(b"\xff\xd8"):
            return data
    except Exception as exc:
        logger.warning("pyHik snapshot error channel=%s: %s", channel, exc)

    return None


def fetch_snapshot(
    channel_id: str,
    channel: int,
    stream_type: int,
    credentials: Dict[str, str],
    isapi_client: Any = None,
) -> Optional[bytes]:
    data = fetch_snapshot_bytes(channel_id, credentials)
    if data:
        return data
    return fetch_snapshot_pyhik(channel, stream_type, isapi_client)


class ISAPISnapshotCapture:
    """Poll ISAPI JPEG snapshots at a fixed FPS for analytics ingestion."""

    def __init__(
        self,
        channel_id: str,
        credentials: Dict[str, str],
        *,
        channel: int = 1,
        stream_type: int = 2,
        poll_fps: float = 7,
        isapi_client: Any = None,
    ):
        self.channel_id = channel_id
        self.credentials = credentials
        self.channel = channel
        self.stream_type = stream_type
        self.poll_fps = poll_fps
        self.isapi_client = isapi_client
        self.interval = 1.0 / poll_fps if poll_fps > 0 else 0.083

        self.running = True
        self.ret = False
        self.frame = None
        self.lock = threading.Lock()
        self.last_frame_time = time.time()
        self.is_file = False
        self.consecutive_failures = 0
        self.last_latency_ms = 0.0

        self.thread = threading.Thread(target=self._poll_reader, daemon=True)
        self.thread.start()

    def _poll_reader(self):
        while self.running:
            loop_start = time.time()
            jpg = fetch_snapshot(
                self.channel_id,
                self.channel,
                self.stream_type,
                self.credentials,
                self.isapi_client,
            )

            if jpg:
                nparr = np.frombuffer(jpg, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if frame is not None:
                    with self.lock:
                        self.ret = True
                        self.frame = frame
                        self.last_frame_time = time.time()
                        self.last_latency_ms = (time.time() - loop_start) * 1000
                    self.consecutive_failures = 0
                else:
                    self.consecutive_failures += 1
            else:
                self.consecutive_failures += 1

            elapsed = time.time() - loop_start
            sleep_for = max(0, self.interval - elapsed)
            if sleep_for:
                time.sleep(sleep_for)

    def read(self):
        with self.lock:
            if self.frame is not None:
                return self.ret, self.frame.copy()
            return self.ret, None

    def get_last_frame_time(self):
        with self.lock:
            return self.last_frame_time

    def get_poll_health(self):
        with self.lock:
            return {
                "last_latency_ms": round(self.last_latency_ms, 1),
                "consecutive_failures": self.consecutive_failures,
            }

    def release(self):
        self.running = False
        self.thread.join(timeout=2)

    def isOpened(self):
        return True
