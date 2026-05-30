"""
ISAPI snapshot preview for DVR line-drawing (pyHik).
Lightweight alternative to starting CCTVProcessor / RTSP during HMI setup.
"""

import logging
import os
import threading
import time
from typing import Any, Dict, List, Optional
from urllib.parse import quote

import cv2
import requests
import urllib3
from requests.auth import HTTPDigestAuth

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

logger = logging.getLogger(__name__)

try:
    from pyhik.isapi import ISAPIClient
except ImportError:
    ISAPIClient = None

FRAME_STALE_SECONDS = 15
POLL_INTERVAL_SECONDS = 0.5
SNAPSHOT_WIDTH = 1280
SNAPSHOT_HEIGHT = 720

isapi_client = None
dvr_credentials: Dict[str, str] = {}
camera_store: Dict[str, Dict[str, Any]] = {}
lock = threading.Lock()

stream_manager = {
    "camera_id": None,
    "frame": None,
    "thread": None,
    "stop_event": None,
    "last_frame_time": 0,
    "mode": None,
    "session_id": 0,
}


def close_client():

    global isapi_client

    if isapi_client:
        try:
            isapi_client.close()
        except Exception:
            pass
        isapi_client = None

    dvr_credentials.clear()


def validate_and_connect(ip: str, username: str, password: str) -> Optional[Dict[str, Any]]:

    global isapi_client

    if ISAPIClient is None:
        raise ImportError("pyHik is not installed")

    close_client()

    client = ISAPIClient(
        host=ip,
        username=username,
        password=password,
        verify_ssl=False,
    )

    info = client.get_device_info()

    if not info:
        client.close()
        return None

    isapi_client = client
    dvr_credentials["ip"] = ip
    dvr_credentials["username"] = username
    dvr_credentials["password"] = password

    return info


def _build_rtsp_url(channel_id: str) -> str:

    ip = dvr_credentials.get("ip", "")
    username = dvr_credentials.get("username", "")
    password = quote(dvr_credentials.get("password", ""))

    return (
        f"rtsp://{username}:{password}@{ip}:554"
        f"/Streaming/Channels/{channel_id}"
    )


def discover_cameras() -> List[Dict[str, Any]]:

    cameras: List[Dict[str, Any]] = []

    if not isapi_client:
        return cameras

    try:
        for cam in isapi_client.get_cameras():
            channel = cam.id
            stream_type = 1
            channel_id = f"{channel}01"

            for stream in cam.streams:
                if stream.id.endswith("01") or stream.type_id == 1:
                    channel_id = stream.id
                    stream_type = 1
                    break

            cameras.append({
                "camera_name": cam.name or f"Camera {channel}",
                "camera_number": channel,
                "channel_id": channel_id,
                "channel": channel,
                "stream_type": stream_type,
                "rtsp": _build_rtsp_url(channel_id),
            })

    except Exception as e:
        logger.warning("get_cameras failed: %s", e)

    if cameras:
        return cameras

    try:
        for stream in isapi_client.get_streaming_channels():
            if not stream.enabled:
                continue
            if stream.id.endswith("02") or stream.type_id == 2:
                continue

            channel = stream.channel_id
            channel_id = stream.id

            cameras.append({
                "camera_name": stream.name or f"Camera {channel}",
                "camera_number": channel,
                "channel_id": channel_id,
                "channel": channel,
                "stream_type": 1,
                "rtsp": _build_rtsp_url(channel_id),
            })

    except Exception as e:
        logger.warning("get_streaming_channels failed: %s", e)

    if cameras:
        return cameras

    for cam_no in range(1, 9):
        channel_id = f"{cam_no}01"
        cameras.append({
            "camera_name": f"Camera {cam_no}",
            "camera_number": cam_no,
            "channel_id": channel_id,
            "channel": cam_no,
            "stream_type": 1,
            "rtsp": _build_rtsp_url(channel_id),
        })

    return cameras


def register_cameras(cameras: List[Dict[str, Any]]):

    camera_store.clear()

    for cam in cameras:
        key = str(cam.get("channel_id", cam.get("camera_number")))
        camera_store[key] = cam


def _fetch_snapshot_raw(channel_id: str) -> Optional[bytes]:

    ip = dvr_credentials.get("ip")
    username = dvr_credentials.get("username")
    password = dvr_credentials.get("password")

    if not ip or not username:
        return None

    url = (
        f"http://{ip}/ISAPI/Streaming/channels/{channel_id}/picture"
        f"?videoResolutionWidth={SNAPSHOT_WIDTH}"
        f"&videoResolutionHeight={SNAPSHOT_HEIGHT}"
    )

    try:
        response = requests.get(
            url,
            auth=HTTPDigestAuth(username, password),
            timeout=8,
            verify=False,
        )

        if (
            response.status_code == 200
            and response.content.startswith(b"\xff\xd8")
        ):
            return response.content

        logger.warning(
            "Raw snapshot failed for channel %s: status=%s bytes=%s",
            channel_id,
            response.status_code,
            len(response.content) if response.content else 0,
        )

    except Exception as e:
        logger.warning("Raw snapshot error for channel %s: %s", channel_id, e)

    return None


def _fetch_snapshot_pyhik(channel: int, stream_type: int) -> Optional[bytes]:

    if not isapi_client:
        return None

    try:
        data = isapi_client.get_snapshot(
            channel=channel,
            stream_type=stream_type,
        )

        if data and data.startswith(b"\xff\xd8"):
            return data

    except Exception as e:
        logger.warning("pyHik snapshot error channel %s: %s", channel, e)

    return None


def fetch_snapshot(channel_id: str, channel: int, stream_type: int) -> Optional[bytes]:

    data = _fetch_snapshot_raw(channel_id)
    if data:
        return data

    return _fetch_snapshot_pyhik(channel, stream_type)


def _fetch_demo_frame() -> Optional[bytes]:

    source = "sample.mp4"

    if not os.path.exists(source):
        return None

    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        return None

    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        return None

    frame = cv2.resize(
        frame,
        (SNAPSHOT_WIDTH, int(frame.shape[0] * SNAPSHOT_WIDTH / frame.shape[1])),
    )
    ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])

    return jpg.tobytes() if ok else None


def stop_preview():

    global stream_manager

    with lock:
        stop_event = stream_manager["stop_event"]
        thread = stream_manager["thread"]

    if stop_event:
        stop_event.set()

    if thread and thread.is_alive():
        thread.join(timeout=3.0)

    with lock:
        stream_manager["camera_id"] = None
        stream_manager["frame"] = None
        stream_manager["thread"] = None
        stream_manager["stop_event"] = None
        stream_manager["last_frame_time"] = 0
        stream_manager["mode"] = None

    logger.info("DVR preview stopped")


def start_preview(channel_id: str) -> bool:

    global stream_manager

    stop_preview()

    if channel_id == "demo":
        camera = {
            "channel_id": "demo",
            "channel": 0,
            "stream_type": 1,
            "mode": "demo",
        }
    else:
        camera = camera_store.get(channel_id)

        if not camera:
            logger.warning("Unknown preview channel: %s", channel_id)
            return False

    stop_event = threading.Event()
    mode = camera.get("mode", "isapi")
    channel = camera.get("channel", 1)
    stream_type = camera.get("stream_type", 1)

    with lock:
        stream_manager["session_id"] = stream_manager.get("session_id", 0) + 1
        session_id = stream_manager["session_id"]

    def worker():

        logger.info(
            "Preview polling %s (mode=%s, channel=%s, stream=%s, session=%s)",
            channel_id,
            mode,
            channel,
            stream_type,
            session_id,
        )

        while not stop_event.is_set():

            if mode == "demo":
                jpg = _fetch_demo_frame()
            else:
                jpg = fetch_snapshot(channel_id, channel, stream_type)

            if jpg and not stop_event.is_set():
                with lock:
                    if (
                        stream_manager.get("session_id") == session_id
                        and stream_manager.get("camera_id") == channel_id
                    ):
                        stream_manager["frame"] = jpg
                        stream_manager["last_frame_time"] = time.time()

            stop_event.wait(POLL_INTERVAL_SECONDS)

    thread = threading.Thread(target=worker, daemon=True)

    with lock:
        stream_manager["camera_id"] = channel_id
        stream_manager["stop_event"] = stop_event
        stream_manager["thread"] = thread
        stream_manager["mode"] = mode

    thread.start()
    return True


def get_cached_frame(channel_id: str) -> Optional[bytes]:

    with lock:
        frame_data = stream_manager["frame"]
        last_frame = stream_manager["last_frame_time"]
        active_camera = stream_manager["camera_id"]

    if (
        active_camera == channel_id
        and frame_data is not None
        and (time.time() - last_frame) <= FRAME_STALE_SECONDS
    ):
        return frame_data

    return None


def ensure_preview(channel_id: str) -> bool:

    with lock:
        current = stream_manager["camera_id"]
        last_frame = stream_manager["last_frame_time"]
        has_frame = stream_manager["frame"] is not None

    if current != channel_id:
        return start_preview(channel_id)

    if has_frame and (time.time() - last_frame) > FRAME_STALE_SECONDS:
        return start_preview(channel_id)

    return True


def get_session_status() -> Dict[str, Any]:
    """Return current DVR connection state for session restore in the HMI portal."""

    with lock:
        active_channel = stream_manager["camera_id"]

    return {
        "connected": bool(dvr_credentials.get("ip")),
        "ip": dvr_credentials.get("ip"),
        "username": dvr_credentials.get("username"),
        "cameras": list(camera_store.values()),
        "active_preview_channel": active_channel,
    }
