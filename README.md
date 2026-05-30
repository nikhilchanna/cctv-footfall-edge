# How to Test the Footfall Counter Project Locally.

Follow these step-by-step instructions to start the project and test the YOLOv8 + DeepSORT computer vision pipeline using a local sample video instead of an actual CCTV stream.

### Prerequisites
1. **Python 3.9+** installed on your system.
2. **PostgreSQL** running locally with a database named `footfall_db`.
3. A sample video showing people walking (save it as `sample.mp4` inside the project folder).

---

### Step 1: Set up the Python Environment
Open a terminal in `g:\Java Projects\Gigs\footfall-counter` and run:

```bash
# 1. Create a virtual environment
python -m venv venv

# 2. Activate the virtual environment (Windows)
.\venv\Scripts\activate

# 2. Activate the virtual environment (mac/Linux)
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```
*(Note: When you run this for the first time, YOLOv8 will automatically download its model weights file `yolov8n.pt` into your folder).*

### Step 2: Configure the Database URL
If your PostgreSQL credentials differ from the default (`postgres`/`postgres`), set the `DATABASE_URL` environment variable. Otherwise, the app defaults to `postgresql://postgres:postgres@localhost:5432/footfall_db`.

Make sure the `footfall_db` database exists in your PostgreSQL instance.

### Step 3: Start the Backend Server
Start the FastAPI server using Uvicorn:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```
Upon startup, the server will automatically:
- Connect to PostgreSQL and create the `data_tracker_table` and `cctv_config` tables.
- Start the background scheduled tasks (API polling, retries, and cleanup).

### Step 4: Configure the "CCTV" via API
Now we need to tell the server to process your sample video. Open your browser and go to the built-in Swagger UI:
👉 **http://localhost:8000/docs**

1. Expand the `POST /config` endpoint.
2. Click **"Try it out"**.
3. In the Request Body, paste the following JSON configuration:

```json
{
  "config_data": {
    "cameras": [
      {
        "id": "CAM_001",
        "name": "Front Door Video",
        "rtsp_url": "sample.mp4", 
        "window_size": 10,
        "line_coords": {
          "x1": 0,
          "y1": 300,
          "x2": 1000,
          "y2": 300
        }
      }
    ]
  }
}
```
*Notice how `rtsp_url` is set to `sample.mp4`! OpenCV automatically handles local files the same way it handles RTSP URLs.*

4. Click **Execute**. The config will be saved to the database.

*(Note: Because the server loads CCTV streams on startup, **restart your Uvicorn server** once after saving the config so the CV Thread picks up the new config).*

### Step 5: Observe the Processing
Watch your terminal where Uvicorn is running.
- You will see logs indicating `Started 1 CCTV Processors`.
- As the video plays silently in the background, YOLOv8 and DeepSORT will detect people.
- If someone crosses the virtual line (y=300), you will see debug logs: `[CAM_001] Person crossed IN` or `Person crossed OUT`.
- **Every 10 seconds**, the system will flush the window's count to PostgreSQL: `[CAM_001] Saved window: In=3, Out=1`.

### Step 6: Verify the Database & Background Threads
1. Open your PostgreSQL GUI (like pgAdmin or DBeaver).
2. Check the `data_tracker_table`. You will see rows populated with your `ctr_in` and `ctr_out` for `CAM_001`.
3. Check the `data_to_server_ack` column. The background API Calling thread will try to send this data to the external server. Because the external server is currently a dummy URL, it will mark the rows as `Failed`.
4. The `Retry Failed Thread` will periodically wake up, find those failed entries, and retry them, incrementing the `api_call_ctr` each time!

---

### Hikvision ISAPI snapshot mode (production DVR)

For Hikvision DVRs, analytics ingestion uses ISAPI JPEG snapshots instead of RTSP:

- Endpoint: `GET /ISAPI/Streaming/channels/{channel_id}/picture`
- Default poll rate: **7 FPS** on substream channel (`102` instead of `101`)
- Footfall counting (YOLO + DeepSORT + 10s windows → `data_tracker_table` → POST to footfall-server) is unchanged
- One peak JPEG per minute is stored in `minute_peak_snapshot` (highest people count in that minute)
- Resume watermark is stored in `processing_cursor` (application-level, not provided by Hikvision)
- On restart: footfall counts resume live (gap accepted); minute peaks may be backfilled via NVR `ContentMgmt` APIs

Example camera config:

```json
{
  "config_data": {
    "dvr": { "ip": "192.168.1.34", "username": "admin", "password": "secret" },
    "cameras": [{
      "id": "101",
      "name": "Camera 1",
      "source_type": "isapi",
      "channel_id": "102",
      "poll_fps": 7,
      "line_coords": { "x1": 0, "y1": 300, "x2": 1000, "y2": 300 },
      "window_size": 10
    }]
  }
}
```

Processor status API: `GET /processor/{cctv_id}/status`

### Analytics UI, peaks, and edge dashboard

- `GET /processor/{cctv_id}/thumbnail` — JPEG snapshot tile (refresh every few seconds in UI)
- `GET /processor/{cctv_id}/minute-peaks?limit=15` — recent peak images on edge
- `GET /cameras/status` — persisted camera alerts (`configured`, `connected`, `no_data`, `auth_failed`, …)
- `GET /analytics/summary` — per-camera counts for lightweight UI
- Edge HTML dashboard: `http://<host>:8000/edge-ui/`
- Peak images upload to server every 20s (`PEAK_UPLOAD_URL`, default `http://localhost:8081/api/v1/peak-images`)
- After upload, edge keeps at most **15 successful uploads per camera** (older files removed)

Schema changes for existing Postgres are applied automatically on startup via versioned SQL migrations in `app/migrations/versions/` (tracked in `edge_schema_migrations`). To apply manually without starting the API:

```bash
python migrate.py
```

Add new migrations as `NNN_description.sql` (three-digit prefix, e.g. `002_add_foo.sql`).
