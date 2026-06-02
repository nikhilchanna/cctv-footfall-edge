"""MP4 test harness — head detect + zone transition counting, no DB."""
import base64
import json
import logging
import os
import time

import cv2
import numpy as np

from cv_engine.config import load_engine_config
from cv_engine.engine import CrowdCountingEngine

logger = logging.getLogger(__name__)
DEBUG_LOG = "/Users/home/analytics_footfall/.cursor/debug-b17557.log"
JPEG_ENCODE_PARAMS = [cv2.IMWRITE_JPEG_QUALITY, 70]

ZONE_COLORS = {
    "entry": (0, 255, 0),
    "buffer": (0, 255, 255),
    "exit": (0, 0, 255),
}


def _agent_log(hypothesis_id, location, message, data=None):
    try:
        with open(DEBUG_LOG, "a") as f:
            f.write(
                json.dumps(
                    {
                        "sessionId": "b17557",
                        "hypothesisId": hypothesis_id,
                        "location": location,
                        "message": message,
                        "data": data or {},
                        "timestamp": int(time.time() * 1000),
                    }
                )
                + "\n"
            )
    except Exception:
        pass


def grab_first_frame(video_path: str) -> dict:
    path = os.path.abspath(video_path)
    if not os.path.isfile(path):
        return {"ok": False, "error": f"File not found: {path}"}

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return {"ok": False, "error": "Cannot open video"}

    ret, frame = cap.read()
    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()

    if not ret or frame is None:
        return {"ok": False, "error": "Cannot read first frame"}

    h, w = frame.shape[:2]
    ok, jpg = cv2.imencode(".jpg", frame, JPEG_ENCODE_PARAMS)
    if not ok:
        return {"ok": False, "error": "Failed to encode frame"}

    return {
        "ok": True,
        "video_path": path,
        "width": w,
        "height": h,
        "reported_size": [fw, fh],
        "image_base64": base64.b64encode(jpg.tobytes()).decode("ascii"),
    }


def _build_cv_config(session) -> dict:
    merged = dict(session.cv_engine_config)
    merged.setdefault("footfall", {})
    merged["footfall"]["camera_role"] = session.camera_role
    merged["footfall"]["count_direction"] = session.count_direction

    det = merged.setdefault("detector", {})
    if session.head_conf_threshold is not None:
        det["conf_threshold"] = session.head_conf_threshold

    zones = merged.setdefault("zones", {})
    zones["entry_side"] = session.entry_side
    zones["observation_offset_pixels"] = session.observation_offset_pixels
    zones["count_zone_width_pixels"] = session.count_zone_width_pixels
    zones["ignore_offset_pixels"] = session.ignore_offset_pixels

    if session.manual_zones:
        zones["auto_generate"] = False
        for key in ("observation", "count", "ignore"):
            pts = session.manual_zones.get(key)
            if pts:
                zones[key] = {"points": pts}
    else:
        zones["auto_generate"] = True

    return merged


class VideoTestSession:
    """Single-threaded video test — same engine as live CCTV processors."""

    def __init__(
        self,
        video_path: str,
        line_coords: dict,
        camera_role: str = "IN",
        count_direction: str = "both",
        entry_side: str = "above",
        observation_offset_pixels: int = 150,
        count_zone_width_pixels: int = 100,
        ignore_offset_pixels: int = 100,
        head_conf_threshold: float = 0.22,
        manual_zones: dict | None = None,
        cv_engine_config: dict | None = None,
    ):
        self.video_path = os.path.abspath(video_path)
        self.line_coords = line_coords
        self.camera_role = camera_role
        self.count_direction = count_direction
        self.entry_side = entry_side
        self.observation_offset_pixels = observation_offset_pixels
        self.count_zone_width_pixels = count_zone_width_pixels
        self.ignore_offset_pixels = ignore_offset_pixels
        self.head_conf_threshold = head_conf_threshold
        self.manual_zones = manual_zones
        self.cv_engine_config = cv_engine_config or {}
        self.running = False
        self.last_frame = None
        self.stats: dict = {"running": False}
        self.ctr_in = 0
        self.ctr_out = 0
        self._frame_idx = 0
        self.engine: CrowdCountingEngine | None = None
        self._cap = None

    def get_stream_interval(self) -> float:
        return 0.1  # 10 FPS target for video test

    def start(self) -> None:
        if not os.path.isfile(self.video_path):
            self.stats = {"error": f"File not found: {self.video_path}", "running": False}
            return

        lc = self.line_coords or {}
        line_len = (
            (lc.get("x2", 0) - lc.get("x1", 0)) ** 2
            + (lc.get("y2", 0) - lc.get("y1", 0)) ** 2
        ) ** 0.5
        if line_len < 10:
            self.stats = {"error": "Counting line too short — draw a longer line", "running": False}
            return

        self._cap = cv2.VideoCapture(self.video_path)
        if not self._cap.isOpened():
            self.stats = {"error": "Cannot open video", "running": False}
            return

        ret, frame = self._cap.read()
        fw = fh = None
        if ret and frame is not None:
            fh, fw = frame.shape[:2]
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        try:
            cfg = load_engine_config(
                _build_cv_config(self),
                line_coords=self.line_coords,
                frame_width=fw,
                frame_height=fh,
            )
            self.engine = CrowdCountingEngine(cfg)
        except Exception as exc:
            self.stats = {"error": str(exc), "running": False}
            _agent_log("A", "video_test_runner.py:start", "engine init failed", {"error": str(exc)})
            self._cap.release()
            self._cap = None
            return

        self.running = True
        self.stats = {"running": True, "no_db": True, "video_path": self.video_path}
        _agent_log(
            "F",
            "video_test_runner.py:start",
            "zone test started",
            {"path": self.video_path, "line": self.line_coords, "frame": [fw, fh]},
        )

    def stop(self) -> None:
        self.running = False
        if self._cap:
            self._cap.release()
            self._cap = None
        self.stats["running"] = False

    def tick(self) -> bool:
        if not self.running or not self._cap or not self.engine:
            return False

        ret, frame = self._cap.read()
        if not ret or frame is None:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            return True

        self._frame_idx += 1
        try:
            result = self.engine.process(frame)
        except Exception as exc:
            _agent_log("A", "video_test_runner.py:tick", "process error", {"error": str(exc)})
            return True

        self.ctr_in += result.in_count_delta
        self.ctr_out += result.out_count_delta

        for track in result.active_tracks:
            x1, y1, x2, y2 = track.bbox
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"ID:{track.track_id}",
                (int(x1), int(y1) - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 0),
                1,
            )

        overlay = self.engine.get_zone_overlay()
        for zone_name, color in ZONE_COLORS.items():
            pts = overlay.get(zone_name)
            if pts and len(pts) >= 3:
                arr = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(frame, [arr], True, color, 2)

        lc = self.line_coords
        cv2.line(
            frame,
            (int(lc.get("x1", 0)), int(lc.get("y1", 0))),
            (int(lc.get("x2", 0)), int(lc.get("y2", 0))),
            (255, 0, 0),
            2,
        )

        eng = self.engine.get_status_fields()
        cv2.putText(
            frame,
            f"TEST | IN:{self.ctr_in} OUT:{self.ctr_out} TRACKS:{len(result.active_tracks)} "
            f"{eng.get('active_footfall_path')} {eng.get('camera_role')}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
        )

        h, w = frame.shape[:2]
        self.stats = {
            "running": True,
            "no_db": True,
            **eng,
            "line_coords": self.line_coords,
            "zones_overlay": overlay,
            "video_path": self.video_path,
            "frame_idx": self._frame_idx,
            "frame_size": [w, h],
            "frame_mean": round(float(frame.mean()), 2),
            "head_detections": len(result.detections),
            "track_count": len(result.active_tracks),
            "current_count": result.current_count,
            "density_level": result.density_level,
            "window_in": self.ctr_in,
            "window_out": self.ctr_out,
            "confidence": round(result.confidence, 3),
        }

        ok, jpeg_bytes = cv2.imencode(".jpg", frame, JPEG_ENCODE_PARAMS)
        if ok:
            self.last_frame = jpeg_bytes.tobytes()
        return True


VideoTestRunner = VideoTestSession
