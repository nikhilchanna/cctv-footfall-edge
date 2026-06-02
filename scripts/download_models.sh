#!/usr/bin/env bash
# Download person + head YOLO weights for edge devices.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p models

echo "Downloading yolov8n.pt (person)..."
python -c "from ultralytics import YOLO; m=YOLO('yolov8n.pt'); import shutil; shutil.copy(m.ckpt_path if hasattr(m,'ckpt_path') else 'yolov8n.pt', 'models/yolov8n.pt')" 2>/dev/null || {
  python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
  cp yolov8n.pt models/yolov8n.pt 2>/dev/null || true
}

echo "Downloading SCUT head nano.pt..."
TMP=/tmp/scut-yolo-$$
git clone --depth 1 https://github.com/Abcfsa/YOLOv8_head_detector.git "$TMP"
cp "$TMP/nano.pt" models/scut_head_yolov8n.pt
rm -rf "$TMP"

echo "Verify load..."
python -c "
from ultralytics import YOLO
for p in ['models/yolov8n.pt','models/scut_head_yolov8n.pt']:
    m=YOLO(p)
    print(p, m.names)
"
echo "Done."
