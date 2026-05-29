import cv2
import threading
import time
import logging
from datetime import datetime, timezone
from sqlalchemy.orm import Session
from deep_sort_realtime.deepsort_tracker import DeepSort
from ultralytics import YOLO

import os
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

from app.database import SessionLocal
from app.models import DataTracker
from app.tasks import report_internal_error

logger = logging.getLogger(__name__)

class CCTVProcessor(threading.Thread):
    def __init__(self, cctv_id: str, cctv_name: str, stream_url: str, line_coords: dict, window_size: int = 10):
        super().__init__()
        self.cctv_id = cctv_id
        self.cctv_name = cctv_name
        self.stream_url = stream_url
        self.line_coords = line_coords  # e.g., {'x1': 100, 'y1': 200, 'x2': 500, 'y2': 200}
        self.window_size = window_size
        self.running = True
        
        # Load YOLOv8 model (using the nano version for speed, can be changed to yolov8m.pt or yolov8l.pt for better accuracy)
        # It will download the model weights automatically on first run
        self.model = YOLO("yolov8n.pt") 
        
        # Initialize DeepSORT Tracker
        self.tracker = DeepSort(max_age=30, n_init=3, nms_max_overlap=1.0)
        
        # Tracking state
        self.track_history = {}
        self.ctr_in = 0
        self.ctr_out = 0
        self.window_start_time = datetime.now(timezone.utc)
        
        # Live streaming frame cache
        self.last_frame = None
        self.lock = threading.Lock()

    def is_crossing_line(self, pt1, pt2, line_pt1, line_pt2):
        """
        Check if a line segment (pt1 to pt2) intersects the virtual line (line_pt1 to line_pt2).
        Returns direction (1 for 'In', -1 for 'Out', 0 for no crossing) based on cross product.
        """
        def ccw(A, B, C):
            return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])
            
        A, B = line_pt1, line_pt2
        C, D = pt1, pt2
        
        intersect = ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)
        if intersect:
            # Determine direction using cross product (simplified vertical crossing detection)
            # If line is horizontal, crossing top to bottom vs bottom to top
            if pt1[1] < pt2[1]:
                return 1 # In
            else:
                return -1 # Out
        return 0

    def save_window_data(self, end_time, current_in, current_out):
        """Save the aggregated counts to the database."""
        db: Session = SessionLocal()
        try:
            tracker_entry = DataTracker(
                ctr_in=current_in,
                ctr_out=current_out,
                timewindow=self.window_size,
                starttime=self.window_start_time,
                endtime=end_time,
                cctvid=self.cctv_id,
                cctvname=self.cctv_name,
                data_to_server_ack="Pending"
            )
            db.add(tracker_entry)
            db.commit()
            logger.info(f"[{self.cctv_id}] Saved window: In={current_in}, Out={current_out}")
        except Exception as e:
            db.rollback()
            report_internal_error("CCTVProcessor_DB", str(e))
        finally:
            db.close()
            self.window_start_time = end_time

    def run(self):
        logger.info(f"Starting CCTV Processor for {self.cctv_id} ({self.cctv_name})")
        
        # Check if running in demo mode
        is_demo = self.stream_url == "demo"
        source = "sample.mp4" if is_demo else self.stream_url
        cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
        
        if not cap.isOpened():
            if not is_demo:
                logger.error(f"Cannot open stream {source} for CCTV {self.cctv_id}. AUTOMATIC FALLBACK TO DEMO VIDEO.")
                is_demo = True
                source = "sample.mp4"
                cap = cv2.VideoCapture(source)
                if not cap.isOpened():
                    report_internal_error("CCTVProcessor", f"Cannot open fallback stream sample.mp4")
                    return
            else:
                report_internal_error("CCTVProcessor", f"Cannot open stream {source} for CCTV {self.cctv_id}")
                return

        # Take a sample picture at the start of the feed
        import os
        os.makedirs("snapshots", exist_ok=True)
        
        # Clear buffer to get a fresh frame
        for _ in range(5):
            cap.grab()
            
        ret, frame = cap.read()
        if ret:
            snapshot_path = os.path.join("snapshots", f"snapshot_{self.cctv_id}.jpg")
            cv2.imwrite(snapshot_path, frame)
            logger.info(f"Saved initial snapshot for {self.cctv_id} to {snapshot_path}")

        l_pt1 = (self.line_coords.get('x1', 0), self.line_coords.get('y1', 200))
        l_pt2 = (self.line_coords.get('x2', 640), self.line_coords.get('y2', 200))

        while self.running:
            # Buffer clearing: Grab frames without decoding to stay real-time (skip for demo file)
            if not is_demo and cap.get(cv2.CAP_PROP_BUFFERSIZE) > 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

            ret, frame = cap.read()
            if not ret or frame is None:
                if is_demo:
                    # Loop the demo video
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                logger.warning(f"Failed to read frame from {self.cctv_id}. Reconnecting...")
                time.sleep(2)
                cap.release()
                cap = cv2.VideoCapture(self.stream_url, cv2.CAP_FFMPEG)
                continue
                
            current_time = datetime.now(timezone.utc)
            
            # Check if window expired
            if (current_time - self.window_start_time).total_seconds() >= self.window_size:
                in_count = self.ctr_in
                out_count = self.ctr_out
                self.ctr_in = 0
                self.ctr_out = 0
                
                # Run DB save asynchronously to not block video processing
                threading.Thread(target=self.save_window_data, args=(current_time, in_count, out_count)).start()

            # YOLOv8 Detection
            results = self.model(frame, classes=[0], verbose=False) # class 0 is 'person'
            
            bbs = []
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    conf = box.conf[0].item()
                    # format: ( [left, top, w, h], confidence, detection_class )
                    bbs.append(([x1, y1, x2 - x1, y2 - y1], conf, 'person'))

            # Update tracker
            tracks = self.tracker.update_tracks(bbs, frame=frame)
            
            for track in tracks:
                if not track.is_confirmed():
                    continue
                
                track_id = track.track_id
                ltrb = track.to_ltrb() # left, top, right, bottom
                
                # Calculate center point of the bounding box
                center_x = int((ltrb[0] + ltrb[2]) / 2)
                center_y = int((ltrb[1] + ltrb[3]) / 2)
                current_pt = (center_x, center_y)
                
                # Draw bounding box and label
                cv2.rectangle(frame, (int(ltrb[0]), int(ltrb[1])), (int(ltrb[2]), int(ltrb[3])), (0, 255, 0), 2)
                cv2.putText(frame, f"ID: {track_id}", (int(ltrb[0]), int(ltrb[1]) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                if track_id in self.track_history:
                    prev_pt = self.track_history[track_id]
                    
                    # Check for line crossing
                    direction = self.is_crossing_line(prev_pt, current_pt, l_pt1, l_pt2)
                    if direction == 1:
                        self.ctr_in += 1
                        logger.info(f"[{self.cctv_id}] Person crossed IN")
                        # Clear history to avoid double counting
                        del self.track_history[track_id]
                        continue
                    elif direction == -1:
                        self.ctr_out += 1
                        logger.info(f"[{self.cctv_id}] Person crossed OUT")
                        del self.track_history[track_id]
                        continue
                        
                self.track_history[track_id] = current_pt
            
            # Draw customizable horizontal/vertical counting line
            cv2.line(frame, l_pt1, l_pt2, (255, 0, 0), 3) # Blue line
            
            # Draw real-time totals overlay on stream (highly visual for POC)
            cv2.putText(frame, f"IN: {self.ctr_in} | OUT: {self.ctr_out}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            
            # Save the processed frame to thread-safe last_frame cache for streaming
            ret_enc, jpeg_bytes = cv2.imencode('.jpg', frame)
            if ret_enc:
                with self.lock:
                    self.last_frame = jpeg_bytes.tobytes()
                
        cap.release()
        logger.info(f"Stopped CCTV Processor for {self.cctv_id}")

    def stop(self):
        self.running = False
