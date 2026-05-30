import cv2
import threading
import time
import logging
import subprocess
import os
import numpy as np
from datetime import datetime, timezone
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError
from deep_sort_realtime.deepsort_tracker import DeepSort
from ultralytics import YOLO

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

from app.database import SessionLocal
from app.models import DataTracker, ProcessingCursor, MinutePeakSnapshot
from app.error_reporting import report_internal_error
from app.camera_status import upsert_camera_status
from app.hikvision_snapshot import ISAPISnapshotCapture

logger = logging.getLogger(__name__)

DEFAULT_POLL_FPS = 7
JPEG_ENCODE_PARAMS = [cv2.IMWRITE_JPEG_QUALITY, 70]


class FFmpegVideoCapture:
    """
    High-performance bufferless capture wrapper.
    For local files: Uses standard OpenCV VideoCapture.
    For live RTSP feeds: Spawns an FFMPEG subprocess with low-latency flags,
    pipes MJPEG stream to stdout, and decodes JPEGs in a background thread.
    """

    def __init__(self, src, fps=5):
        self.src = src
        self.fps = fps
        self.ret = False
        self.frame = None
        self.running = True
        self.lock = threading.Lock()
        self.consecutive_failures = 0
        self.last_latency_ms = 0.0

        self.is_file = src == "sample.mp4" or not (
            src.startswith("rtsp://") or src.startswith("http://")
        )

        if self.is_file:
            self.cap = cv2.VideoCapture(src)
            self.thread = threading.Thread(target=self._file_reader)
        else:
            self.cap = None
            self.last_frame_time = time.time()
            self.thread = threading.Thread(target=self._rtsp_reader)

        self.thread.daemon = True
        self.thread.start()

    def _file_reader(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                if self.src == "sample.mp4":
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    time.sleep(0.01)
                    continue
                logger.warning(f"Stream connection lost: {self.src}. Reconnecting...")
                time.sleep(2)
                self.cap.release()
                self.cap = cv2.VideoCapture(self.src)
                continue
            with self.lock:
                self.ret = ret
                self.frame = frame
                self.last_frame_time = time.time()
            time.sleep(1.0 / self.fps)

    def _rtsp_reader(self):
        command = [
            "ffmpeg",
            "-rtsp_transport", "tcp",
            "-fflags", "nobuffer",
            "-flags", "low_delay",
            "-i", self.src,
            "-an",
            "-vf", f"fps={self.fps},scale=960:-1",
            "-f", "image2pipe",
            "-vcodec", "mjpeg",
            "-q:v", "5",
            "-",
        ]

        while self.running:
            logger.info(f"Spawning FFMPEG pipe decoder for RTSP stream: {self.src}")
            try:
                process = subprocess.Popen(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    bufsize=10**8,
                )
            except FileNotFoundError:
                logger.error(
                    "ffmpeg executable not found in PATH! Falling back to OpenCV native RTSP capture."
                )
                self.cap = cv2.VideoCapture(self.src)
                self._file_reader()
                return

            buffer = b""
            while self.running and process.poll() is None:
                try:
                    chunk = process.stdout.read(4096)
                    if not chunk:
                        time.sleep(0.01)
                        continue
                    buffer += chunk

                    while True:
                        start = buffer.find(b"\xff\xd8")
                        end = buffer.find(b"\xff\xd9")
                        if start != -1 and end != -1:
                            if start < end:
                                jpg_bytes = buffer[start : end + 2]
                                buffer = buffer[end + 2 :]

                                nparr = np.frombuffer(jpg_bytes, np.uint8)
                                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                                if frame is not None:
                                    with self.lock:
                                        self.ret = True
                                        self.frame = frame
                                        self.last_frame_time = time.time()
                            else:
                                buffer = buffer[start:]
                        else:
                            break
                except Exception as e:
                    logger.error(f"FFmpeg subprocess read error: {e}")
                    break

            try:
                process.kill()
                process.wait(timeout=1)
            except Exception:
                pass

            if self.running:
                logger.warning(
                    "FFmpeg subprocess stream disconnected. Reconnecting in 2 seconds..."
                )
                time.sleep(2)

    def read(self):
        with self.lock:
            if self.frame is not None:
                return self.ret, self.frame.copy()
            return self.ret, None

    def get_last_frame_time(self):
        with self.lock:
            return getattr(self, "last_frame_time", time.time())

    def get_poll_health(self):
        return {"last_latency_ms": 0.0, "consecutive_failures": self.consecutive_failures}

    def release(self):
        self.running = False
        if self.cap:
            self.cap.release()
        self.thread.join(timeout=1)

    def isOpened(self):
        if self.is_file:
            return self.cap.isOpened()
        return True


def _floor_minute(dt: datetime) -> datetime:
    return dt.replace(second=0, microsecond=0)


class CCTVProcessor(threading.Thread):
    def __init__(
        self,
        cctv_id: str,
        cctv_name: str,
        line_coords: dict,
        window_size: int = 10,
        stream_url: str = None,
        source_config: dict = None,
        dvr_config: dict = None,
    ):
        super().__init__()
        self.cctv_id = cctv_id
        self.cctv_name = cctv_name
        self.line_coords = line_coords
        self.window_size = window_size
        self.stream_url = stream_url or "demo"
        self.source_config = source_config or {}
        self.dvr_config = dvr_config or {}
        self.running = True

        self.model = YOLO("yolov8n.pt")
        self.tracker = DeepSort(max_age=30, n_init=3, nms_max_overlap=1.0)

        self.track_history = {}
        self.ctr_in = 0
        self.ctr_out = 0
        self.window_start_time = datetime.now(timezone.utc)

        self.last_frame = None
        self.lock = threading.Lock()

        self.minute_state = {
            "bucket": None,
            "peak_count": 0,
            "best_jpeg": None,
            "best_at": None,
        }
        self.frames_this_minute = 0
        self.stream_fps = float(self.source_config.get("poll_fps", DEFAULT_POLL_FPS))
        self.stats = {
            "source_type": self.source_config.get("type", "rtsp"),
            "frames_this_minute": 0,
            "current_peak_people": 0,
            "last_snapshot_latency_ms": 0.0,
            "consecutive_failures": 0,
            "mode": "live",
            "stream_fps": self.stream_fps,
        }

    def get_stream_interval(self) -> float:
        fps = self.stream_fps if self.stream_fps > 0 else DEFAULT_POLL_FPS
        return 1.0 / fps

    def _resolve_source_type(self) -> str:
        explicit = self.source_config.get("type")
        if explicit:
            return explicit
        if self.stream_url == "demo":
            return "demo"
        if self.stream_url and (
            self.stream_url.startswith("rtsp://")
            or self.stream_url.startswith("http://")
        ):
            return "rtsp"
        if self.dvr_config.get("ip") and self.source_config.get("channel_id"):
            return "isapi"
        return "demo"

    def _create_capture(self):
        source_type = self._resolve_source_type()
        self.stats["source_type"] = source_type

        if source_type == "isapi":
            channel_id = self.source_config.get("channel_id", self.cctv_id)
            poll_fps = float(self.source_config.get("poll_fps", DEFAULT_POLL_FPS))
            self.stream_fps = poll_fps
            self.stats["stream_fps"] = poll_fps
            channel = int(self.source_config.get("channel", 1))
            stream_type = int(self.source_config.get("stream_type", 2))
            return ISAPISnapshotCapture(
                channel_id=channel_id,
                credentials=self.dvr_config,
                channel=channel,
                stream_type=stream_type,
                poll_fps=poll_fps,
            )

        if source_type == "demo":
            return FFmpegVideoCapture("sample.mp4")

        return FFmpegVideoCapture(self.stream_url)

    def is_crossing_line(self, pt1, pt2, line_pt1, line_pt2):
        def ccw(A, B, C):
            return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

        A, B = line_pt1, line_pt2
        C, D = pt1, pt2

        intersect = ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)
        if intersect:
            if pt1[1] < pt2[1]:
                return 1
            return -1
        return 0

    def save_window_data(self, end_time, current_in, current_out):
        db: Session = SessionLocal()
        try:
            existing = (
                db.query(DataTracker)
                .filter(
                    DataTracker.cctvid == self.cctv_id,
                    DataTracker.starttime == self.window_start_time,
                    DataTracker.endtime == end_time,
                )
                .first()
            )
            if existing:
                logger.info(
                    f"[{self.cctv_id}] Skipping duplicate window {self.window_start_time} -> {end_time}"
                )
                return

            tracker_entry = DataTracker(
                ctr_in=current_in,
                ctr_out=current_out,
                timewindow=self.window_size,
                starttime=self.window_start_time,
                endtime=end_time,
                cctvid=self.cctv_id,
                cctvname=self.cctv_name,
                data_to_server_ack="Pending",
            )
            db.add(tracker_entry)
            db.commit()
            logger.info(
                f"[{self.cctv_id}] Saved window: In={current_in}, Out={current_out}"
            )
        except IntegrityError:
            db.rollback()
            logger.info(
                f"[{self.cctv_id}] Duplicate window ignored via constraint"
            )
        except Exception as e:
            db.rollback()
            report_internal_error("CCTVProcessor_DB", str(e))
        finally:
            db.close()
            self.window_start_time = end_time

    def _update_processing_cursor(self, current_time: datetime):
        db: Session = SessionLocal()
        try:
            minute_bucket = _floor_minute(current_time)
            cursor = (
                db.query(ProcessingCursor)
                .filter(ProcessingCursor.cctvid == self.cctv_id)
                .first()
            )
            if not cursor:
                cursor = ProcessingCursor(cctvid=self.cctv_id)
                db.add(cursor)

            cursor.last_processed_at = current_time
            cursor.last_minute_bucket = minute_bucket
            cursor.last_live_seq = (cursor.last_live_seq or 0) + 1
            if cursor.mode != "backfill":
                cursor.mode = "live"
            db.commit()
        except Exception as e:
            db.rollback()
            report_internal_error("CCTVProcessor_Cursor", str(e))
        finally:
            db.close()

    def _save_minute_peak(self, minute_bucket: datetime, people_count: int, jpeg_bytes: bytes):
        os.makedirs("snapshots/peaks", exist_ok=True)
        stamp = minute_bucket.strftime("%Y%m%d%H%M")
        image_path = os.path.join("snapshots", "peaks", f"{self.cctv_id}_{stamp}.jpg")

        with open(image_path, "wb") as handle:
            handle.write(jpeg_bytes)

        db: Session = SessionLocal()
        try:
            existing = (
                db.query(MinutePeakSnapshot)
                .filter(
                    MinutePeakSnapshot.cctvid == self.cctv_id,
                    MinutePeakSnapshot.minute_bucket == minute_bucket,
                )
                .first()
            )
            if existing:
                if people_count > existing.people_count or (
                    people_count == existing.people_count
                    and existing.source != "live"
                ):
                    existing.people_count = people_count
                    existing.image_path = image_path
                    existing.captured_at = datetime.now(timezone.utc)
                    existing.source = "live"
                    existing.uploaded_to_server = "Pending"
            else:
                db.add(
                    MinutePeakSnapshot(
                        cctvid=self.cctv_id,
                        minute_bucket=minute_bucket,
                        people_count=people_count,
                        image_path=image_path,
                        captured_at=datetime.now(timezone.utc),
                        source="live",
                        uploaded_to_server="Pending",
                    )
                )
            db.commit()
            logger.info(
                f"[{self.cctv_id}] Saved minute peak bucket={minute_bucket} people={people_count}"
            )
        except IntegrityError:
            db.rollback()
        except Exception as e:
            db.rollback()
            report_internal_error("CCTVProcessor_MinutePeak", str(e))
        finally:
            db.close()

    def _handle_minute_rollover(self, current_time: datetime, people_count: int, frame):
        minute_bucket = _floor_minute(current_time)
        state = self.minute_state

        if state["bucket"] is None:
            state["bucket"] = minute_bucket
            state["peak_count"] = people_count
            state["best_at"] = current_time
            ret_enc, jpeg_bytes = cv2.imencode(".jpg", frame, JPEG_ENCODE_PARAMS)
            if ret_enc:
                state["best_jpeg"] = jpeg_bytes.tobytes()
            self.frames_this_minute = 1
            self.stats["frames_this_minute"] = 1
            self.stats["current_peak_people"] = people_count
            return

        if minute_bucket != state["bucket"]:
            if state["best_jpeg"] is not None:
                self._save_minute_peak(
                    state["bucket"], state["peak_count"], state["best_jpeg"]
                )
            state["bucket"] = minute_bucket
            state["peak_count"] = people_count
            state["best_at"] = current_time
            ret_enc, jpeg_bytes = cv2.imencode(".jpg", frame, JPEG_ENCODE_PARAMS)
            state["best_jpeg"] = jpeg_bytes.tobytes() if ret_enc else None
            self.frames_this_minute = 1
            self.stats["frames_this_minute"] = 1
            self.stats["current_peak_people"] = people_count
            return

        self.frames_this_minute += 1
        self.stats["frames_this_minute"] = self.frames_this_minute
        if people_count >= state["peak_count"]:
            state["peak_count"] = people_count
            state["best_at"] = current_time
            ret_enc, jpeg_bytes = cv2.imencode(".jpg", frame, JPEG_ENCODE_PARAMS)
            if ret_enc:
                state["best_jpeg"] = jpeg_bytes.tobytes()
            self.stats["current_peak_people"] = people_count

    def get_status(self) -> dict:
        db: Session = SessionLocal()
        try:
            cursor = (
                db.query(ProcessingCursor)
                .filter(ProcessingCursor.cctvid == self.cctv_id)
                .first()
            )
            last_peak = (
                db.query(MinutePeakSnapshot)
                .filter(MinutePeakSnapshot.cctvid == self.cctv_id)
                .order_by(MinutePeakSnapshot.minute_bucket.desc())
                .first()
            )
            cursor_data = {
                "last_processed_at": cursor.last_processed_at.isoformat()
                if cursor and cursor.last_processed_at
                else None,
                "last_minute_bucket": cursor.last_minute_bucket.isoformat()
                if cursor and cursor.last_minute_bucket
                else None,
                "mode": cursor.mode if cursor else "live",
                "backfill_from": cursor.backfill_from.isoformat()
                if cursor and cursor.backfill_from
                else None,
                "backfill_to": cursor.backfill_to.isoformat()
                if cursor and cursor.backfill_to
                else None,
            }
            peak_data = None
            if last_peak:
                peak_data = {
                    "minute_bucket": last_peak.minute_bucket.isoformat(),
                    "people_count": last_peak.people_count,
                    "image_path": last_peak.image_path,
                    "source": last_peak.source,
                }
        finally:
            db.close()

        with self.lock:
            return {
                "cctv_id": self.cctv_id,
                "cctv_name": self.cctv_name,
                "source_type": self.stats.get("source_type"),
                "frames_this_minute": self.stats.get("frames_this_minute", 0),
                "current_peak_people": self.stats.get("current_peak_people", 0),
                "last_snapshot_latency_ms": self.stats.get("last_snapshot_latency_ms", 0),
                "consecutive_failures": self.stats.get("consecutive_failures", 0),
                "window_in": self.ctr_in,
                "window_out": self.ctr_out,
                "stream_fps": self.stream_fps,
                "cursor": cursor_data,
                "last_saved_peak": peak_data,
            }

    def _report_status(self, status: str, message: str, detail: str = None):
        try:
            upsert_camera_status(
                self.cctv_id,
                status,
                cctvname=self.cctv_name,
                message=message,
                detail=detail,
            )
        except Exception as exc:
            logger.warning("[%s] Failed to persist camera status: %s", self.cctv_id, exc)

    def run(self):
        logger.info(
            f"Starting CCTV Processor for {self.cctv_id} ({self.cctv_name}) "
            f"source={self._resolve_source_type()}"
        )

        self._report_status("connected", "Processor started")

        is_demo = self._resolve_source_type() == "demo"
        cap = self._create_capture()

        if not cap.isOpened():
            if not is_demo:
                logger.error(
                    f"Cannot open source for CCTV {self.cctv_id}. AUTOMATIC FALLBACK TO DEMO VIDEO."
                )
                is_demo = True
                cap.release()
                cap = FFmpegVideoCapture("sample.mp4")
                if not cap.isOpened():
                    self._report_status("error", "Cannot open demo fallback video")
                    report_internal_error(
                        "CCTVProcessor", "Cannot open fallback stream sample.mp4"
                    )
                    return
            else:
                self._report_status("disconnected", "Cannot open camera stream")
                report_internal_error(
                    "CCTVProcessor", f"Cannot open stream for CCTV {self.cctv_id}"
                )
                return

        os.makedirs("snapshots", exist_ok=True)
        time.sleep(0.5)

        ret, frame = cap.read()
        if ret and frame is not None:
            snapshot_path = os.path.join("snapshots", f"snapshot_{self.cctv_id}.jpg")
            cv2.imwrite(snapshot_path, frame)
            logger.info(f"Saved initial snapshot for {self.cctv_id} to {snapshot_path}")

        l_pt1 = (self.line_coords.get("x1", 0), self.line_coords.get("y1", 200))
        l_pt2 = (self.line_coords.get("x2", 640), self.line_coords.get("y2", 200))

        while self.running:
            ret, frame = cap.read()

            if hasattr(cap, "get_poll_health"):
                health = cap.get_poll_health()
                self.stats["last_snapshot_latency_ms"] = health.get("last_latency_ms", 0)
                self.stats["consecutive_failures"] = health.get("consecutive_failures", 0)
                failures = self.stats["consecutive_failures"]
                if failures >= 3:
                    self._report_status(
                        "no_data",
                        "Camera not returning snapshots",
                        detail=f"consecutive_failures={failures}",
                    )
                elif failures == 0:
                    self._report_status("connected", "Receiving frames")

            if not is_demo and not getattr(cap, "is_file", True):
                if time.time() - cap.get_last_frame_time() > 5.0:
                    self._report_status("no_data", "No frame data for 5+ seconds")
                    logger.warning(
                        f"[{self.cctv_id}] Stale stream detected (no data for 5s). Reconnecting..."
                    )
                    cap.release()
                    cap = self._create_capture()
                    continue

            if not ret or frame is None:
                if is_demo:
                    time.sleep(0.04)
                    continue
                self._report_status("disconnected", "Failed to read frame")
                logger.warning(f"Failed to read frame from {self.cctv_id}. Reconnecting...")
                time.sleep(2)
                cap.release()
                cap = self._create_capture()
                continue

            current_time = datetime.now(timezone.utc)

            if (current_time - self.window_start_time).total_seconds() >= self.window_size:
                in_count = self.ctr_in
                out_count = self.ctr_out
                self.ctr_in = 0
                self.ctr_out = 0
                threading.Thread(
                    target=self.save_window_data,
                    args=(current_time, in_count, out_count),
                ).start()

            results = self.model(frame, classes=[0], verbose=False)

            bbs = []
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    conf = box.conf[0].item()
                    bbs.append(([x1, y1, x2 - x1, y2 - y1], conf, "person"))

            people_count = len(bbs)
            self._handle_minute_rollover(current_time, people_count, frame)
            self._update_processing_cursor(current_time)

            tracks = self.tracker.update_tracks(bbs, frame=frame)

            for track in tracks:
                if not track.is_confirmed():
                    continue

                track_id = track.track_id
                ltrb = track.to_ltrb()

                center_x = int((ltrb[0] + ltrb[2]) / 2)
                center_y = int((ltrb[1] + ltrb[3]) / 2)
                current_pt = (center_x, center_y)

                cv2.rectangle(
                    frame,
                    (int(ltrb[0]), int(ltrb[1])),
                    (int(ltrb[2]), int(ltrb[3])),
                    (0, 255, 0),
                    2,
                )
                cv2.putText(
                    frame,
                    f"ID: {track_id}",
                    (int(ltrb[0]), int(ltrb[1]) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )

                if track_id in self.track_history:
                    prev_pt = self.track_history[track_id]
                    direction = self.is_crossing_line(prev_pt, current_pt, l_pt1, l_pt2)
                    if direction == 1:
                        self.ctr_in += 1
                        logger.info(f"[{self.cctv_id}] Person crossed IN")
                        del self.track_history[track_id]
                        continue
                    if direction == -1:
                        self.ctr_out += 1
                        logger.info(f"[{self.cctv_id}] Person crossed OUT")
                        del self.track_history[track_id]
                        continue

                self.track_history[track_id] = current_pt

            cv2.line(frame, l_pt1, l_pt2, (255, 0, 0), 3)
            cv2.putText(
                frame,
                f"IN: {self.ctr_in} | OUT: {self.ctr_out} | PEAK: {self.stats['current_peak_people']}",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                3,
            )

            ret_enc, jpeg_bytes = cv2.imencode(".jpg", frame, JPEG_ENCODE_PARAMS)
            if ret_enc:
                with self.lock:
                    self.last_frame = jpeg_bytes.tobytes()

        cap.release()
        logger.info(f"Stopped CCTV Processor for {self.cctv_id}")

    def stop(self):
        self.running = False
